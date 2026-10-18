"""Tables (results/baselines/summary.md) and figures for the method comparison.

Figures
- figures/comparison_synthetic.png : planted routes; order recovery for the order-producing methods
                                      (cPCA, seqNMF, PP-Seq) and per-event detection for HMM and PP-Seq.
- figures/comparison_hc11.png      : hc-11, every method relative to its own chance level, POST vs PRE,
                                      per-session points, and maze specificity.
- figures/comparison_example.png   : one session, what each method's output looks like.
Earlier figures (6-panel hc-11 grid, map-match bars) are superseded and no longer produced.
"""

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
COL = {"cpca": "#1f4e79", "cpca2": "#6aa7d8", "seq": "#d4652f", "pp": "#c9a227", "hmm": "#1f8a65", "hmm2": "#9fd3bd", "bayes": "#7a5195", "pre": "#c3c9d2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
CPCA_VARIANT = "centred, T = event duration"


# ----------------------------------------------------------------------------- helpers
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


def hits_of(items):
    """(hits, tests) over components/factors/types and directions."""
    h = t = 0
    for c in items:
        for s, m in c["map"].items():
            if np.isfinite(m["rho"]):
                t += 1
                h += int(abs(m["rho"]) > m["null95"])
    return h, t


def load_cpca(name):
    f = RES / "hc11" / f"{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def load_ppseq_synthetic():
    out = {}
    for f in glob.glob(str(RES / "baselines" / "ppseq_synthetic" / "*.pkl")):
        r = pickle.load(open(f, "rb"))
        out[r["condition"]] = r
    return out


def load_ppseq_hc11(name):
    f = RES / "baselines" / f"ppseq_{name}.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


def chance_bar(ax, pos, vals, nulls, width, colour, lab, first):
    ax.bar(pos, vals, width=width * 0.9, color=colour, label=lab, zorder=2)
    ax.hlines(nulls, pos - width * 0.45, pos + width * 0.45, colors="black", linewidth=2.2, zorder=3,
              label="chance level (95th percentile of the cell-permutation null)" if first else None)


# ----------------------------------------------------------------------------- synthetic
def figure_synthetic(syn, pps):
    fig, axes = plt.subplots(2, 1, figsize=(17, 11.5), gridspec_kw={"height_ratios": [1.35, 1]})
    x = np.arange(len(syn))
    # top: order recovery
    ax = axes[0]
    series = [("cPCA, uncentred", COL["cpca"], lambda s, p: s["methods"]["cPCA uncentred"]["routes"]),
              ("cPCA, centred", COL["cpca2"], lambda s, p: s["methods"]["cPCA centred"]["routes"]),
              ("seqNMF", COL["seq"], lambda s, p: s["methods"]["seqNMF"]["routes"]),
              ("PP-Seq", COL["pp"], lambda s, p: p["routes"] if p else [])]
    width = 0.8 / len(series)
    for i, (name, colour, get) in enumerate(series):
        vals, nulls = [], []
        for s in syn:
            routes = get(s, pps.get(s["condition"]))
            vals.append(np.median([abs(r[0]) for r in routes]) if routes else np.nan)
            nulls.append(np.nanmedian([r[1] for r in routes]) if routes else np.nan)
        chance_bar(ax, x - 0.4 + width * (i + 0.5), vals, nulls, width, colour, name, i == 0)
    ax.set_xticks(x)
    ax.set_xticklabels([s["condition"].split(" ", 1)[0] + "\n" + wrap(s["condition"].split(" ", 1)[1], 17) for s in syn], fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("|Spearman|, member order vs planted route\n(median over the planted routes)")
    ax.set_title("Methods that output a cell order: how well is each planted route recovered?")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    # bottom: per-event detection
    ax = axes[1]
    hmm_auc = [s["methods"]["HMM"]["auc_replay"] for s in syn]
    hmm_fa = [s["methods"]["HMM"].get("congruent_volleys", np.nan) for s in syn]
    hmm_hit = [s["methods"]["HMM"].get("congruent_replays", np.nan) for s in syn]
    pp_auc = [pps[s["condition"]]["auc_replay"] if s["condition"] in pps else np.nan for s in syn]
    pp_hit = [pps[s["condition"]]["flagged_replays"] if s["condition"] in pps else np.nan for s in syn]
    pp_fa = [pps[s["condition"]]["flagged_volleys"] if s["condition"] in pps else np.nan for s in syn]
    w = 0.13
    ax.bar(x - 2.5 * w, hmm_auc, width=w * 0.92, color=COL["hmm"], label="HMM congruence: AUC (replays vs volleys)", zorder=2)
    ax.bar(x - 1.5 * w, hmm_hit, width=w * 0.92, color=COL["hmm2"], label="HMM: share of planted replays flagged", zorder=2)
    ax.bar(x - 0.5 * w, hmm_fa, width=w * 0.92, color=COL["hmm2"], hatch="////", edgecolor="white", label="HMM: share of volleys flagged (false alarms)", zorder=2)
    ax.bar(x + 0.5 * w, pp_auc, width=w * 0.92, color=COL["pp"], label="PP-Seq sequence types: AUC (replays vs volleys)", zorder=2)
    ax.bar(x + 1.5 * w, pp_hit, width=w * 0.92, color="#e8d58a", label="PP-Seq: share of planted replays flagged", zorder=2)
    ax.bar(x + 2.5 * w, pp_fa, width=w * 0.92, color="#e8d58a", hatch="////", edgecolor="white", label="PP-Seq: share of volleys flagged (false alarms)", zorder=2)
    ax.axhline(0.5, color="black", lw=1.3, ls="--", label="AUC of a random detector", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s["condition"].split(" ", 1)[0] for s in syn])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("per-event detection")
    ax.set_title("Methods that mark single events as sequences: can they tell planted replays from synchronous volleys?\n"
                 "(conditions A–C have no volleys, so only the flagged share is defined)", fontsize=12.5)
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("Synthetic planted routes (80 cells, 3000 events): each method scored on what it outputs", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "comparison_synthetic.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- hc-11, integrated
def figure_hc11(base, pps):
    """Every method relative to its own chance level, POST vs PRE; per-session points; maze specificity."""
    cp = {b["name"]: load_cpca(b["name"]) for b in base}
    FLOOR = 0.15
    methods = [
        ("cPCA (centred)\ncomponents whose order\nmatches the map\n[label-free]", COL["cpca"]),
        ("PP-Seq\nsequence types whose\noffsets match the map\n[label-free]", COL["pp"]),
        ("seqNMF\nfactors whose lags\nmatch the map\n[label-free]", COL["seq"]),
        ("HMM congruence\nevents with sequential\nstructure\n[label-free]", COL["hmm"]),
        ("HMM latent map\nposition decoded via\nstate place fields\n[behaviour used post hoc]", COL["hmm2"]),
        ("Bayesian decoding\nevents with a decoded\ntrajectory\n[place map as input]", COL["bayes"]),
    ]

    def per_session(b, epoch):
        e = b["epochs"][epoch]
        c = cp[b["name"]]
        p = pps.get(b["name"])
        rows = []
        h, t = hits_of(c["epochs"][epoch]["variants"][CPCA_VARIANT]["components"]) if c else (np.nan, 0)
        rows.append((h, 0.05 * t))
        if p and epoch in p["epochs"]:
            h, t = hits_of(p["epochs"][epoch]["types"])
            rows.append((h, 0.05 * t))
        else:
            rows.append((np.nan, 0))
        h, t = hits_of(e["seqNMF"]["factors"])
        rows.append((h, 0.05 * t))
        rows.append((100 * e["HMM"]["congruent_99"], 1.0))
        l = e["HMM"]["lspf"]
        rows.append((l["shuffled_median_error_m"], l["median_error_m"]))
        rows.append((100 * e["Bayes"]["frac_sig"], 5.0))
        return rows

    table = {epoch: [per_session(b, epoch) for b in base] for epoch in ("post", "pre")}
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(methods))
    pooled_all = {}
    for i, (epoch, shade, lab) in enumerate((("post", 1.0, "POST sleep (after the novel maze), pooled over sessions"), ("pre", 0.35, "PRE sleep (before the maze; control), pooled"))):
        pos = x - 0.2 + 0.4 * i
        pooled = []
        for m in range(len(methods)):
            num = np.array([row[m][0] for row in table[epoch]], float)
            den = np.array([row[m][1] for row in table[epoch]], float)
            ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
            pooled.append(np.nanmedian(num[ok] / den[ok]) if m == 4 else (num[ok].sum() / den[ok].sum() if den[ok].sum() > 0 else np.nan))
            ratios = np.where(ok, num / np.where(den > 0, den, np.nan), np.nan)
            zero = ok & (num == 0)
            jitter = np.linspace(-0.12, 0.12, len(ratios))
            ax.scatter(pos[m] + jitter[~zero & ok], ratios[~zero & ok], s=26, color="black", alpha=0.75 if epoch == "post" else 0.4, zorder=4,
                       label="single sessions" if (i == 0 and m == 0) else None)
            ax.scatter(pos[m] + jitter[zero], np.full(zero.sum(), FLOOR), s=32, facecolors="none", edgecolors="black", alpha=0.75 if epoch == "post" else 0.4, zorder=4,
                       label="session with zero matches (drawn at the floor)" if (i == 0 and m == 0) else None)
        pooled_all[epoch] = pooled
        ax.bar(pos, np.maximum(np.nan_to_num(pooled, nan=FLOOR), FLOOR), width=0.38, color=[c for _, c in methods], alpha=shade,
               edgecolor="black" if epoch == "pre" else "none", linewidth=0.8, zorder=2, label=lab)
        for p_, val in zip(pos, pooled):
            if np.isfinite(val):
                ax.text(p_, max(val, FLOOR) * 1.12, f"{val:.1f}×", ha="center", va="bottom", fontsize=10.5, fontweight="bold" if epoch == "post" else "normal")
    ax.axhline(1.0, color="black", lw=1.6, ls="--", zorder=3, label="chance level of each method (1×)")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 60)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in methods], fontsize=10)
    ax.set_ylabel("score relative to the method's own chance level (×, log scale)")
    ax.set_title("hc-11, eight sessions: every method against its own chance level, POST vs PRE sleep", pad=12)
    ax.yaxis.grid(True, which="both", color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    handles, labels_ = ax.get_legend_handles_labels()
    want = ("POST sleep (after the novel maze), pooled over sessions", "PRE sleep (before the maze; control), pooled", "chance level of each method (1×)", "single sessions", "session with zero matches (drawn at the floor)")
    order = [labels_.index(l) for l in want if l in labels_]
    ax.legend([handles[o] for o in order], [labels_[o] for o in order], loc="upper left", frameon=False, fontsize=10)
    # right: specificity
    ax = axes[1]
    for m, (name, colour) in enumerate(methods):
        spec = []
        for rp, rr in zip(table["post"], table["pre"]):
            a = rp[m][0] / rp[m][1] if rp[m][1] > 0 else np.nan
            b_ = rr[m][0] / rr[m][1] if rr[m][1] > 0 else np.nan
            if np.isfinite(a) and np.isfinite(b_):
                spec.append(np.log2(max(a, FLOOR)) - np.log2(max(b_, FLOOR)))
        spec = np.array(spec)
        if spec.size:
            jitter = np.linspace(-0.15, 0.15, spec.size)
            ax.scatter(np.full(spec.size, m) + jitter, np.clip(spec, -4.3, 4.3), s=40, color=colour, edgecolors="black", linewidths=0.5, zorder=3)
            ax.hlines(np.median(spec), m - 0.3, m + 0.3, colors="black", linewidth=2.6, zorder=4)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_ylim(-4.5, 4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("\n")[0] for m, _ in methods], fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("specificity to the experienced maze:\nlog₂(POST ÷ chance) − log₂(PRE ÷ chance), per session")
    ax.set_title("Is the signal about the maze the animal just ran?\n(above 0: stronger in POST than PRE; thick line: median)", pad=12)
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, -0.33, "Sessions with 0 or 1 matches give floor artefacts; points are clipped to ±4.3.", transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "comparison_hc11.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return pooled_all


# ----------------------------------------------------------------------------- example session
def figure_example(base, pps, name="Achilles_11012013"):
    """One session: the best map-matching cPCA component, PP-Seq type and seqNMF factor, and the HMM latent map."""
    from cpca import event_spike_times, fit, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    b = next(x for x in base if x["name"] == name)
    cpr = load_cpca(name)
    p = pps.get(name)
    path = next(q for q in session_paths() if q.name.startswith(name))
    session = load_session(path)
    run = running(session)
    intervals = nrem_intervals(session, "post")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    centres = place_centres(session, cell_ids, run)
    fig, axes = plt.subplots(1, 4, figsize=(23, 5.8))

    def scatter_best(ax, items, lag_of, colour, title_fn, ylabel):
        cands = [(abs(c["map"][s]["rho"]) - c["map"][s]["null95"], c, s) for c in items for s in c["map"] if np.isfinite(c["map"][s]["rho"])]
        if not cands:
            ax.text(0.5, 0.5, "no component with enough\nmembers to test", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title_fn(None, None))
            return
        _, c, s = max(cands, key=lambda t: t[0])
        lag = np.asarray(lag_of(c), float)
        use = np.isfinite(lag) & np.isfinite(centres[s])
        ax.scatter(centres[s][use], lag[use], s=55, color=colour, zorder=3)
        ax.set_title(title_fn(c, s) + f"\nSpearman {c['map'][s]['rho']:+.2f} (chance {c['map'][s]['null95']:.2f}), {use.sum()} cells")
        ax.set_xlabel(f"place-field centre, running direction {'+' if s > 0 else '−'} (m)")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
        ax.set_axisbelow(True)

    res = fit(times, duration * 1000, rng=np.random.default_rng(0), center=True, scale=True, ica=False)
    comps = cpr["epochs"]["post"]["variants"][CPCA_VARIANT]["components"]

    def cpca_lag(c):
        _, t, _, inc, _ = readout(res["patterns"][:, c["index"]], res["T_read_ms"], min_weight=0.2)
        return np.where(inc, t, np.nan)

    scatter_best(axes[0], comps, cpca_lag, COL["cpca"], lambda c, s: f"cPCA component {c['index'] + 1 if c else '—'} (centred)", "member's relative time in the component (ms)")
    if p:
        scatter_best(axes[1], p["epochs"]["post"]["types"], lambda c: c["lag_ms"], COL["pp"], lambda c, s: f"PP-Seq sequence type {c['r'] + 1 if c else '—'} (R = {p['epochs']['post']['best_setting']['num_sequence_types']})", "neuron offset in the sequence type (ms)")
    scatter_best(axes[2], b["epochs"]["post"]["seqNMF"]["factors"], lambda c: c["lag_ms"], COL["seq"], lambda c, s: f"seqNMF factor {c['k'] + 1 if c else '—'}" + (" (significant)" if c and c["significant"] else ""), "cell's peak lag in the factor (ms, 10 ms bins)")
    lspf = np.asarray(b["epochs"]["post"]["HMM"]["lspf"]["lspf"])
    mass = np.asarray(b["epochs"]["post"]["HMM"]["lspf"]["state_mass"])
    keep = mass > 0.005 * mass.sum()
    order = np.argsort(np.argmax(lspf[keep], axis=1))
    im = axes[3].imshow(lspf[keep][order], aspect="auto", cmap="magma", extent=(0, run["length"], keep.sum(), 0))
    axes[3].set_xlabel("position on the track (m)")
    axes[3].set_ylabel("hidden state (sorted by preferred position)")
    e = b["epochs"]["post"]["HMM"]["lspf"]
    axes[3].set_title(f"HMM latent-state place fields (POST model)\ndecoding error {e['median_error_m']:.2f} m vs {e['shuffled_median_error_m']:.2f} m with shuffled positions")
    fig.colorbar(im, ax=axes[3], fraction=0.04, pad=0.02, label="P(state | position)")
    fig.suptitle(f"{name.replace('_', ' ')}, POST sleep: what each method's output looks like (behaviour used only on the x-axes)", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "comparison_example.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def write_summary(syn, pps, base, pph):
    lines = ["# Method comparison: results", "",
             "All methods see the same events and cells; the waking map is used only to score. "
             "Order-producing methods (cPCA, seqNMF, PP-Seq) are scored per component/factor/sequence type by the Spearman correlation between member order and the answer key "
             "against a cell-permutation null; the HMM is scored on per-event congruence and on latent-state place-field decoding; "
             "Bayesian decoding takes the map as input and is the reference ceiling. convSeq was not run; see docs/baselines.md.", ""]
    if syn:
        lines += ["## Synthetic planted routes (one draw per condition, 80 cells, 3000 events)", "",
                  "| condition | replay | cPCA uncentred | cPCA centred | seqNMF (significant / K) | PP-Seq (R chosen) | detection AUC, replays vs volleys: HMM congruence / PP-Seq sequence types | count PCA/ICA |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in syn:
            m = s["methods"]
            p = pps.get(s["condition"])

            def fmt(routes):
                return "; ".join(f"{abs(r[0]):.2f} [{r[1]:.2f}]{'*' if np.isfinite(r[1]) and abs(r[0]) > r[1] else ''}" for r in routes)
            pp_cell = f"{fmt(p['routes'])} (R={p['best_setting']['num_sequence_types']})" if p else "—"
            pp_auc = f"{p['auc_replay']:.2f}" if p and np.isfinite(p["auc_replay"]) else "—"
            lines.append(f"| {s['condition']} | {100*s['fraction_replay']:.0f}% | {fmt(m['cPCA uncentred']['routes'])} (k={m['cPCA uncentred']['k']}) | "
                         f"{fmt(m['cPCA centred']['routes'])} (k={m['cPCA centred']['k']}) | {fmt(m['seqNMF']['routes'])} ({m['seqNMF']['n_significant']}/{m['seqNMF']['n_factors']}) | {pp_cell} | "
                         f"{m['HMM']['auc_replay']:.2f} / {pp_auc} | {m['count PCA/ICA']['n_assemblies']} assemblies, no order |")
    if base:
        cp = {b["name"]: load_cpca(b["name"]) for b in base}
        lines += ["", "## hc-11: POST vs PRE sleep", "",
                  "| session | cells / place (+,−) | cPCA (centred) map hits POST / PRE [tests] | PP-Seq map hits POST / PRE [tests] (R) | seqNMF map hits POST / PRE [tests] (significant) | HMM congruent 99% POST / PRE | HMM lsPF error POST (shuffled) / PRE (shuffled), m | Bayes significant POST / PRE |",
                  "|---|---|---|---|---|---|---|---|"]
        tot = {k: [0, 0, 0, 0] for k in ("cpca", "pp", "seq")}
        for b in base:
            e = b["epochs"]
            c = cp[b["name"]]
            p = pph.get(b["name"])
            cph, cpt = hits_of(c["epochs"]["post"]["variants"][CPCA_VARIANT]["components"]) if c else (0, 0)
            crh, crt = hits_of(c["epochs"]["pre"]["variants"][CPCA_VARIANT]["components"]) if c else (0, 0)
            sph, spt = hits_of(e["post"]["seqNMF"]["factors"])
            srh, srt = hits_of(e["pre"]["seqNMF"]["factors"])
            if p:
                pph_, ppt = hits_of(p["epochs"]["post"]["types"])
                prh, prt = hits_of(p["epochs"]["pre"]["types"])
                tot["pp"] = [a + v for a, v in zip(tot["pp"], (pph_, ppt, prh, prt))]
                pp_cell = f"{pph_} / {prh} [{ppt}/{prt}] ({p['epochs']['post']['best_setting']['num_sequence_types']}/{p['epochs']['pre']['best_setting']['num_sequence_types']})"
            else:
                pp_cell = "—"
            tot["cpca"] = [a + v for a, v in zip(tot["cpca"], (cph, cpt, crh, crt))]
            tot["seq"] = [a + v for a, v in zip(tot["seq"], (sph, spt, srh, srt))]
            nsig = (sum(f["significant"] for f in e["post"]["seqNMF"]["factors"]), sum(f["significant"] for f in e["pre"]["seqNMF"]["factors"]))
            lp, lr = e["post"]["HMM"]["lspf"], e["pre"]["HMM"]["lspf"]
            lines.append(f"| {b['name']} | {e['post']['n_cells']} / {e['post']['n_place'][1.0]},{e['post']['n_place'][-1.0]} | {cph} / {crh} [{cpt}/{crt}] | {pp_cell} | "
                         f"{sph} / {srh} [{spt}/{srt}] ({nsig[0]}/{nsig[1]}) | {100*e['post']['HMM']['congruent_99']:.1f}% / {100*e['pre']['HMM']['congruent_99']:.1f}% | "
                         f"{lp['median_error_m']:.2f} ({lp['shuffled_median_error_m']:.2f}) / {lr['median_error_m']:.2f} ({lr['shuffled_median_error_m']:.2f}) | "
                         f"{100*e['post']['Bayes']['frac_sig']:.1f}% / {100*e['pre']['Bayes']['frac_sig']:.1f}% |")
        lines.append(f"| **total** | | **{tot['cpca'][0]} / {tot['cpca'][2]}** [{tot['cpca'][1]}/{tot['cpca'][3]}] (chance ≈ {0.05*tot['cpca'][1]:.1f} / {0.05*tot['cpca'][3]:.1f}) | "
                     f"**{tot['pp'][0]} / {tot['pp'][2]}** [{tot['pp'][1]}/{tot['pp'][3]}] (chance ≈ {0.05*tot['pp'][1]:.1f} / {0.05*tot['pp'][3]:.1f}) | "
                     f"**{tot['seq'][0]} / {tot['seq'][2]}** [{tot['seq'][1]}/{tot['seq'][3]}] (chance ≈ {0.05*tot['seq'][1]:.1f} / {0.05*tot['seq'][3]:.1f}) | | | |")
        if pph:
            lines += ["", "### PP-Seq sequence types on hc-11 (POST): members, offset span, latent events per posterior sample", ""]
            for b in base:
                p = pph.get(b["name"])
                if not p:
                    continue
                for epoch in ("post", "pre"):
                    ep = p["epochs"][epoch]
                    lines.append(f"- {b['name']} {epoch.upper()} (R={ep['best_setting']['num_sequence_types']}, {ep['n_events_fitted']} events fitted, {100*ep['flagged_fraction']:.0f}% of events carry a sequence type that spans ≥ 20 ms): " +
                                 "; ".join(f"type {t['r'] + 1}: {t['n_members']} cells, span {t['lag_span_ms']:.0f} ms, width {t['width_ms']:.0f} ms, {t['mean_latent_events']:.0f} events" for t in ep["types"]))
    (RES / "baselines" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    for stale in ("baselines_hc11.png", "baselines_synthetic.png", "baselines_example.png", "baselines_hc11_integrated.png", "hc11_map_matches.png"):
        (FIG / stale).unlink(missing_ok=True)
    syn_f = RES / "baselines" / "synthetic.pkl"
    syn = pickle.load(open(syn_f, "rb")) if syn_f.exists() else []
    pps = load_ppseq_synthetic()
    base = [pickle.load(open(f, "rb")) for f in glob.glob(str(RES / "baselines" / "*.pkl")) if not Path(f).name.startswith(("synthetic", "ppseq_"))]
    base.sort(key=lambda b: ORDER.index(b["name"]))
    pph = {b["name"]: load_ppseq_hc11(b["name"]) for b in base}
    pph = {k: v for k, v in pph.items() if v}
    if syn:
        figure_synthetic(syn, pps)
    if base:
        figure_hc11(base, pph)
        if any(b["name"] == "Achilles_11012013" for b in base):
            figure_example(base, pph)
    write_summary(syn, pps, base, pph)


if __name__ == "__main__":
    main()
