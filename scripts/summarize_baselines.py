"""Tables (results/baselines/summary.md) and figures (figures/baselines_*.png) for the method comparison."""

import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpca.decode import readout  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
         "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
C_CPCA, C_CPCA2, C_SEQ, C_HMM, C_BAYES, C_PRE = "#1f4e79", "#6aa7d8", "#d4652f", "#1f8a65", "#7a5195", "#b9c0ca"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
                     "legend.fontsize": 11.5, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def label(name):
    rat, day = name.split("_")
    return f"{rat}\n{day[:2]}/{day[2:4]}"


def wrap(text, width=18):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def hits_of(items, key="map"):
    """(hits, tests) over components/factors and directions."""
    h = t = 0
    for c in items:
        for s, m in c[key].items():
            if np.isfinite(m["rho"]):
                t += 1
                h += int(abs(m["rho"]) > m["null95"])
    return h, t


# ----------------------------------------------------------------------------- synthetic
def figure_synthetic(syn):
    fig, axes = plt.subplots(1, 2, figsize=(21, 7.2), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    x = np.arange(len(syn))
    methods = [("cPCA uncentred", C_CPCA), ("cPCA centred", C_CPCA2), ("seqNMF", C_SEQ)]
    width = 0.8 / len(methods)
    for i, (m, colour) in enumerate(methods):
        vals = [np.median([abs(r[0]) for r in s["methods"][m]["routes"]]) for s in syn]
        nulls = [np.nanmedian([r[1] for r in s["methods"][m]["routes"]]) for s in syn]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.9, color=colour, label=m, zorder=2)
        ax.hlines(nulls, pos - width * 0.45, pos + width * 0.45, colors="black", linewidth=2.2, zorder=3,
                  label="chance level (95th percentile of cell-permutation null)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([s["condition"].split(" ", 1)[0] + "\n" + wrap(s["condition"].split(" ", 1)[1], 15) for s in syn], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|Spearman| between a component's (or factor's) member\norder and the planted route (median over routes)")
    ax.set_title("Methods that output a cell order: route recovery on planted data")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ax = axes[1]
    auc = [s["methods"]["HMM"]["auc_replay"] for s in syn]
    rep = [s["methods"]["HMM"].get("congruent_replays", np.nan) for s in syn]
    vol = [s["methods"]["HMM"].get("congruent_volleys", np.nan) for s in syn]
    ax.bar(x - 0.27, auc, width=0.26, color=C_HMM, label="AUC: congruence separates planted replays from volleys", zorder=2)
    ax.bar(x, rep, width=0.26, color="#9fd3bd", label="fraction of planted replays called congruent (99%)", zorder=2)
    ax.bar(x + 0.27, vol, width=0.26, color="#d9d9d9", label="fraction of volleys called congruent (false alarms)", zorder=2)
    ax.axhline(0.5, color="black", lw=1.2, ls="--", label="AUC of a random detector")
    ax.set_xticks(x)
    ax.set_xticklabels([s["condition"].split(" ", 1)[0] for s in syn])
    ax.set_ylim(0, 1.05)
    ax.set_title("HMM (Maboudi et al.): per-event congruence as a replay detector")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False)
    fig.suptitle("Synthetic planted routes: each method scored on what it outputs", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_synthetic.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- hc-11
def load_cpca(name):
    f = RES / "hc11" / f"{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def figure_hc11(base):
    cp = {b["name"]: load_cpca(b["name"]) for b in base}
    fig, axes = plt.subplots(2, 3, figsize=(23, 13))
    x = np.arange(len(base))
    labels = [label(b["name"]) for b in base]

    def post_pre_bars(ax, post, pre, chance_post=None, chance_pre=None, ylabel="", title="", colour=C_CPCA):
        ax.bar(x - 0.2, post, width=0.38, color=colour, label="POST sleep (after the novel maze)", zorder=2)
        ax.bar(x + 0.2, pre, width=0.38, color=C_PRE, label="PRE sleep (before, control)", zorder=2)
        if chance_post is not None:
            ax.hlines(chance_post, x - 0.39, x - 0.01, colors="black", linewidth=2.2, zorder=3, label="expected by chance")
            ax.hlines(chance_pre, x + 0.01, x + 0.39, colors="black", linewidth=2.2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
        ax.set_axisbelow(True)

    # (a) cPCA centred: components matching the map
    v = "centred, T = event duration"
    post = [hits_of(cp[b["name"]]["epochs"]["post"]["variants"][v]["components"])[0] if cp[b["name"]] else np.nan for b in base]
    pre = [hits_of(cp[b["name"]]["epochs"]["pre"]["variants"][v]["components"])[0] if cp[b["name"]] else np.nan for b in base]
    cpost = [0.05 * hits_of(cp[b["name"]]["epochs"]["post"]["variants"][v]["components"])[1] if cp[b["name"]] else np.nan for b in base]
    cpre = [0.05 * hits_of(cp[b["name"]]["epochs"]["pre"]["variants"][v]["components"])[1] if cp[b["name"]] else np.nan for b in base]
    post_pre_bars(axes[0, 0], post, pre, cpost, cpre, "(component, direction) pairs whose\norder matches the place fields",
                  "cPCA (centred)\ncomponents whose internal order matches the map", C_CPCA)
    # (b) seqNMF factors matching the map
    post = [hits_of(b["epochs"]["post"]["seqNMF"]["factors"])[0] for b in base]
    pre = [hits_of(b["epochs"]["pre"]["seqNMF"]["factors"])[0] for b in base]
    cpost = [0.05 * hits_of(b["epochs"]["post"]["seqNMF"]["factors"])[1] for b in base]
    cpre = [0.05 * hits_of(b["epochs"]["pre"]["seqNMF"]["factors"])[1] for b in base]
    post_pre_bars(axes[0, 1], post, pre, cpost, cpre, "(factor, direction) pairs whose\nlag order matches the place fields",
                  "seqNMF\nfactors whose cell lags match the map", C_SEQ)
    for xi, b in zip(x, base):
        n_sig = sum(f["significant"] for f in b["epochs"]["post"]["seqNMF"]["factors"])
        axes[0, 1].annotate(f"{n_sig} sig.", (xi - 0.2, post[int(xi)] + 0.05), ha="center", va="bottom", fontsize=8.5, color=C_SEQ)
    # (c) HMM congruence
    post = [100 * b["epochs"]["post"]["HMM"]["congruent_99"] for b in base]
    pre = [100 * b["epochs"]["pre"]["HMM"]["congruent_99"] for b in base]
    post_pre_bars(axes[0, 2], post, pre, [1.0] * len(base), [1.0] * len(base), "% of events congruent with the model\n(likelihood above 99% of transition shuffles)",
                  "HMM (Maboudi et al.)\nevents with sequential structure, label-free", C_HMM)
    # (d) HMM latent map: decoding error of running position via lsPF
    post = [b["epochs"]["post"]["HMM"]["lspf"]["median_error_m"] for b in base]
    pre = [b["epochs"]["pre"]["HMM"]["lspf"]["median_error_m"] for b in base]
    sh_post = [b["epochs"]["post"]["HMM"]["lspf"]["shuffled_median_error_m"] for b in base]
    sh_pre = [b["epochs"]["pre"]["HMM"]["lspf"]["shuffled_median_error_m"] for b in base]
    ax = axes[1, 0]
    ax.bar(x - 0.2, post, width=0.38, color=C_HMM, label="POST model", zorder=2)
    ax.bar(x + 0.2, pre, width=0.38, color=C_PRE, label="PRE model", zorder=2)
    ax.hlines(sh_post, x - 0.39, x - 0.01, colors="black", linewidth=2.2, zorder=3, label="error with shuffled positions (chance)")
    ax.hlines(sh_pre, x + 0.01, x + 0.39, colors="black", linewidth=2.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("median error decoding running position\nthrough the hidden states (m); lower is better")
    ax.set_title("HMM latent map\nposition decoded via latent-state place fields (behaviour used post hoc)")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    # (e) Bayesian decoding reference
    post = [100 * b["epochs"]["post"]["Bayes"]["frac_sig"] for b in base]
    pre = [100 * b["epochs"]["pre"]["Bayes"]["frac_sig"] for b in base]
    post_pre_bars(axes[1, 1], post, pre, [5.0] * len(base), [5.0] * len(base), "% of events with a significant\ndecoded trajectory (p < 0.05)",
                  "Reference: Bayesian decoding\n(takes the place map as input)", C_BAYES)
    # (f) runtime
    ax = axes[1, 2]
    names = ["cPCA", "seqNMF", "HMM", "Bayes"]
    rt = np.array([[cp[b["name"]]["epochs"]["post"]["variants"][v]["runtime_s"] if cp[b["name"]] else np.nan,
                    b["epochs"]["post"]["seqNMF"]["runtime_s"], b["epochs"]["post"]["HMM"]["runtime_s"], b["epochs"]["post"]["Bayes"]["runtime_s"]] for b in base])
    med = np.nanmedian(rt, axis=0)
    ax.bar(np.arange(4), med, color=[C_CPCA, C_SEQ, C_HMM, C_BAYES], zorder=2)
    for i, m in enumerate(med):
        ax.text(i, m * 1.15, f"{m:.0f} s", ha="center", va="bottom", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("run time per session, POST\n(seconds, log scale)")
    ax.set_title("Cost\nmedian over sessions, including each method's shuffle tests")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("hc-11, eight sessions: every method on the same POST and PRE sleep events, each scored on its own output", y=0.995, fontsize=16)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.11, hspace=0.5, wspace=0.32)
    fig.savefig(FIG / "baselines_hc11.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_example(base, name="Achilles_11012013"):
    """One session: the best map-matching cPCA component, the best seqNMF factor, and the HMM latent map."""
    from cpca import event_spike_times, fit, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    b = next(x for x in base if x["name"] == name)
    cpr = load_cpca(name)
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    run = running(session)
    intervals = nrem_intervals(session, "post")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    centres = place_centres(session, cell_ids, run)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    # (a) cPCA
    v = "centred, T = event duration"
    res = fit(times, duration * 1000, rng=np.random.default_rng(0), center=True, scale=True, ica=False)
    comps = cpr["epochs"]["post"]["variants"][v]["components"]
    best = max(((abs(c["map"][s]["rho"]) - c["map"][s]["null95"], c, s) for c in comps for s in c["map"] if np.isfinite(c["map"][s]["rho"])), key=lambda t: t[0])
    _, c, s = best
    _, t, _, inc, _ = readout(res["patterns"][:, c["index"]], res["T_read_ms"], min_weight=0.2)
    use = inc & np.isfinite(t) & np.isfinite(centres[s])
    axes[0].scatter(centres[s][use], t[use], s=55, color=C_CPCA, zorder=3)
    axes[0].set_title(f"cPCA component {c['index'] + 1}: {use.sum()} member cells\nSpearman {c['map'][s]['rho']:+.2f} (chance {c['map'][s]['null95']:.2f})")
    axes[0].set_xlabel(f"place-field centre, direction {'+' if s > 0 else '−'} (m)")
    axes[0].set_ylabel("member's relative time in the component (ms)")
    # (b) seqNMF
    factors = b["epochs"]["post"]["seqNMF"]["factors"]
    cands = [(abs(f["map"][s]["rho"]) - f["map"][s]["null95"], f, s) for f in factors for s in f["map"] if np.isfinite(f["map"][s]["rho"])]
    if cands:
        _, f, s = max(cands, key=lambda t: t[0])
        lag = np.asarray(f["lag_ms"])
        use = np.isfinite(lag) & np.isfinite(centres[s])
        jitter = np.random.default_rng(0).uniform(-2, 2, size=use.sum())
        axes[1].scatter(centres[s][use], lag[use] + jitter, s=55, color=C_SEQ, zorder=3)
        axes[1].set_title(f"seqNMF factor {f['k'] + 1}: {use.sum()} member cells{' (significant)' if f['significant'] else ' (not significant)'}\nSpearman {f['map'][s]['rho']:+.2f} (chance {f['map'][s]['null95']:.2f})")
        axes[1].set_xlabel(f"place-field centre, direction {'+' if s > 0 else '−'} (m)")
        axes[1].set_ylabel("cell's peak lag in the factor (ms, 10 ms bins)")
    # (c) HMM lsPF
    lspf = np.asarray(b["epochs"]["post"]["HMM"]["lspf"]["lspf"])
    mass = np.asarray(b["epochs"]["post"]["HMM"]["lspf"]["state_mass"])
    keep = mass > 0.005 * mass.sum()
    order = np.argsort(np.argmax(lspf[keep], axis=1))
    im = axes[2].imshow(lspf[keep][order], aspect="auto", cmap="magma", extent=(0, run["length"], keep.sum(), 0))
    axes[2].set_xlabel("position on the track (m)")
    axes[2].set_ylabel("hidden state (sorted by preferred position)")
    e = b["epochs"]["post"]["HMM"]["lspf"]
    axes[2].set_title(f"HMM latent-state place fields (POST model)\ndecoding error {e['median_error_m']:.2f} m vs {e['shuffled_median_error_m']:.2f} m with shuffled positions")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02, label="P(state | position)")
    for ax in axes[:2]:
        ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(f"{name.replace('_', ' ')}, POST sleep: what each method's output looks like", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "baselines_example.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def write_summary(syn, base):
    lines = ["# Method comparison: results", "",
             "All methods see the same events and cells; the waking map is used only to score. "
             "Order-producing methods (cPCA, seqNMF) are scored per component/factor by the Spearman correlation between member order and the answer key "
             "against a cell-permutation null; the HMM is scored on per-event congruence and on latent-state place-field decoding; "
             "Bayesian decoding takes the map as input and is the reference ceiling. PP-Seq (Julia) and convSeq (research code) were not run; see docs/baselines.md.", ""]
    if syn:
        lines += ["## Synthetic planted routes (one draw per condition, 80 cells, 3000 events)", "",
                  "| condition | replay | cPCA uncentred | cPCA centred | seqNMF (significant factors / K) | HMM congruence: AUC replay vs volley; % replays / % volleys flagged | count PCA/ICA |",
                  "|---|---|---|---|---|---|---|"]
        for s in syn:
            m = s["methods"]
            def fmt(routes):
                return "; ".join(f"{abs(r[0]):.2f} [{r[1]:.2f}]{'*' if np.isfinite(r[1]) and abs(r[0]) > r[1] else ''}" for r in routes)
            lines.append(f"| {s['condition']} | {100*s['fraction_replay']:.0f}% | {fmt(m['cPCA uncentred']['routes'])} (k={m['cPCA uncentred']['k']}) | "
                         f"{fmt(m['cPCA centred']['routes'])} (k={m['cPCA centred']['k']}) | {fmt(m['seqNMF']['routes'])} ({m['seqNMF']['n_significant']}/{m['seqNMF']['n_factors']}, λ={m['seqNMF']['lambda']:.1e}) | "
                         f"AUC {m['HMM']['auc_replay']:.2f}; {100*m['HMM'].get('congruent_replays', np.nan):.0f}% / {100*m['HMM'].get('congruent_volleys', np.nan):.0f}% | {m['count PCA/ICA']['n_assemblies']} assemblies, no order |")
    if base:
        cp = {b["name"]: load_cpca(b["name"]) for b in base}
        v = "centred, T = event duration"
        lines += ["", "## hc-11: POST vs PRE sleep", "",
                  "| session | cells / place (+,−) | cPCA (centred) map hits POST / PRE [tests] | seqNMF map hits POST / PRE [tests] (significant factors) | HMM congruent 99% POST / PRE | HMM lsPF error POST (shuffled) / PRE (shuffled), m | Bayes significant POST / PRE (strict) |",
                  "|---|---|---|---|---|---|---|"]
        tot = {k: [0, 0, 0, 0] for k in ("cpca", "seq")}
        for b in base:
            e = b["epochs"]
            c = cp[b["name"]]
            cph, cpt = hits_of(c["epochs"]["post"]["variants"][v]["components"]) if c else (np.nan, np.nan)
            crh, crt = hits_of(c["epochs"]["pre"]["variants"][v]["components"]) if c else (np.nan, np.nan)
            sph, spt = hits_of(e["post"]["seqNMF"]["factors"])
            srh, srt = hits_of(e["pre"]["seqNMF"]["factors"])
            if c:
                tot["cpca"] = [a + x for a, x in zip(tot["cpca"], (cph, cpt, crh, crt))]
            tot["seq"] = [a + x for a, x in zip(tot["seq"], (sph, spt, srh, srt))]
            nsig = (sum(f["significant"] for f in e["post"]["seqNMF"]["factors"]), sum(f["significant"] for f in e["pre"]["seqNMF"]["factors"]))
            lp, lr = e["post"]["HMM"]["lspf"], e["pre"]["HMM"]["lspf"]
            lines.append(f"| {b['name']} | {e['post']['n_cells']} / {e['post']['n_place'][1.0]},{e['post']['n_place'][-1.0]} | {cph} / {crh} [{cpt}/{crt}] | "
                         f"{sph} / {srh} [{spt}/{srt}] ({nsig[0]}/{nsig[1]}) | {100*e['post']['HMM']['congruent_99']:.1f}% / {100*e['pre']['HMM']['congruent_99']:.1f}% | "
                         f"{lp['median_error_m']:.2f} ({lp['shuffled_median_error_m']:.2f}) / {lr['median_error_m']:.2f} ({lr['shuffled_median_error_m']:.2f}) | "
                         f"{100*e['post']['Bayes']['frac_sig']:.1f}% / {100*e['pre']['Bayes']['frac_sig']:.1f}% ({100*e['post']['Bayes']['frac_strict']:.1f}% / {100*e['pre']['Bayes']['frac_strict']:.1f}%) |")
        lines.append(f"| **total** | | **{tot['cpca'][0]} / {tot['cpca'][2]}** [{tot['cpca'][1]}/{tot['cpca'][3]}] (chance ≈ {0.05*tot['cpca'][1]:.1f} / {0.05*tot['cpca'][3]:.1f}) | "
                     f"**{tot['seq'][0]} / {tot['seq'][2]}** [{tot['seq'][1]}/{tot['seq'][3]}] (chance ≈ {0.05*tot['seq'][1]:.1f} / {0.05*tot['seq'][3]:.1f}) | | | |")
        lines += ["", "seqNMF λ chosen by the crossover rule (2 λ₀), per session POST/PRE: " +
                  "; ".join(f"{b['name'].split('_')[0]}: {b['epochs']['post']['seqNMF']['lambda']:.1e}/{b['epochs']['pre']['seqNMF']['lambda']:.1e}" for b in base),
                  "", "HMM cross-validated likelihood gain per bin over surrogates (POST), temporal shuffle / time swap: " +
                  "; ".join(f"{b['name'].split('_')[0]}: {b['epochs']['post']['HMM']['cv_gain']['temporal_shuffle']:.2f}/{b['epochs']['post']['HMM']['cv_gain']['time_swap']:.2f}" for b in base)]
    (RES / "baselines" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    syn_f = RES / "baselines" / "synthetic.pkl"
    syn = pickle.load(open(syn_f, "rb")) if syn_f.exists() else []
    base = [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / "baselines" / "*.pkl")) if not f.endswith("synthetic.pkl")]
    base.sort(key=lambda b: ORDER.index(b["name"]))
    if syn:
        figure_synthetic(syn)
    if base:
        figure_hc11(base)
        if any(b["name"] == "Achilles_11012013" for b in base):
            figure_example(base)
    write_summary(syn, base)


if __name__ == "__main__":
    main()
