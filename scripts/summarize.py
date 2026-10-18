"""Tables (results/summary.md) and figures (figures/) from results/synthetic and results/hc11.

Figures: synthetic_routes.png (cPCA variants on planted routes), hc11_spectrum.png, hc11_example_components.png.
The method comparison figures are made by scripts/summarize_baselines.py."""

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
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402
from cpca import event_spike_times, fit, population_bursts  # noqa: E402
from cpca.decode import readout  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
         "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def label(name):
    rat, day = name.split("_")
    return f"{rat}\n{day[:2]}/{day[2:4]}"


# ----------------------------------------------------------------------------- synthetic
def synthetic_figure(results):
    from scripts.run_synthetic import CONDITIONS, VARIANTS, aggregate
    table = aggregate(results)
    conditions, variants = list(CONDITIONS), list(VARIANTS)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(len(conditions))
    width = 0.8 / len(variants)
    colours = ["#1f4e79", "#6aa7d8", "#1f8a65", "#d4652f"]
    for i, v in enumerate(variants):
        med = [np.median(table[(c, v)]["rho"]) for c in conditions]
        hit = [np.mean(table[(c, v)]["hit"]) for c in conditions]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, med, width=width * 0.9, color=colours[i], label=v, zorder=2)
        for p, h, m in zip(pos, hit, med):
            ax.text(p, m + 0.02, f"{100*h:.0f}%", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split(" ", 1)[0] + "\n" + "\n".join(_wrap(c.split(" ", 1)[1], 22)) for c in conditions], fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("median |Spearman|: pattern members' times vs planted route positions")
    ax.set_title("Synthetic routes: recovery per planted route (numbers above bars: share of route/draw pairs above chance)")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthetic_routes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


# ----------------------------------------------------------------------------- hc-11
def hits(block_variant):
    """Number of (component, direction) tests above chance, number of tests, and the hitting components."""
    n_hit, n_test, rows = 0, 0, []
    for c in block_variant["components"]:
        for sign, m in c["map"].items():
            if np.isfinite(m["rho"]):
                n_test += 1
                if abs(m["rho"]) > m["null95"]:
                    n_hit += 1
                    rows.append((c["index"], sign, m["rho"], m["null95"], m["n_members_with_field"], c["timing_z"], c["span_rad"], c["eigen_share"]))
    return n_hit, n_test, rows


def hc11_tables(sessions, lines):
    variants = list(next(iter(sessions[0]["epochs"].values()))["variants"])
    lines += ["", "## hc-11: components whose internal order matches the place map", "",
              "For every kept component and each running direction: Spearman between members' relative times and their place-field centres, "
              "against the 95th percentile of a cell-permutation null. 'hits / tests' counts the (component, direction) pairs above chance; "
              "about 5% of tests are expected by chance. POST is sleep after the novel maze; PRE is sleep before it (control).", ""]
    for v in variants:
        lines += [f"### {v}", "", "| session | maze | cells | place cells (+/−) | POST events | POST k | POST hits / tests | PRE events | PRE k | PRE hits / tests | split-half match POST / PRE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        tot = {"post": [0, 0], "pre": [0, 0]}
        for s in sessions:
            cells = []
            for epoch in ("post", "pre"):
                b = s["epochs"][epoch]
                h, t, _ = hits(b["variants"][v])
                tot[epoch][0] += h
                tot[epoch][1] += t
                cells.append((b, h, t))
            bp, hp, tp = cells[0]
            br, hr, tr = cells[1]
            lines.append(f"| {s['name']} | {s['maze']} | {bp['n_cells']} | {bp['n_place'][1.0]}/{bp['n_place'][-1.0]} | {bp['n_events']} | {bp['variants'][v]['n_kept']} | {hp} / {tp} | "
                         f"{br['n_events']} | {br['variants'][v]['n_kept']} | {hr} / {tr} | {bp['variants'][v]['split_half_match']:.2f} / {br['variants'][v]['split_half_match']:.2f} |")
        lines.append(f"| **total** | | | | | | **{tot['post'][0]} / {tot['post'][1]}** (chance ≈ {0.05*tot['post'][1]:.1f}) | | | **{tot['pre'][0]} / {tot['pre'][1]}** (chance ≈ {0.05*tot['pre'][1]:.1f}) | |")
        lines.append("")
    # detail of hitting components, default variant, POST
    v0 = variants[0]
    lines += [f"### Components matching the map in POST ({v0})", "", "| session | component | direction | rho | null 95% | members with a field | timing z | span (deg) | eigenvalue share |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sessions:
        for row in hits(s["epochs"]["post"]["variants"][v0])[2]:
            lines.append(f"| {s['name']} | {row[0] + 1} | {'+' if row[1] > 0 else '−'} | {row[2]:+.2f} | {row[3]:.2f} | {row[4]} | {row[5]:.1f} | {np.degrees(row[6]):.0f} | {100*row[7]:.1f}% |")
    lines += ["", "### Count PCA/ICA baseline (same events, POST)", "", "| session | assemblies | assemblies with compact fields (spread below null 5%) | best Jaccard of each complex component's members with a count assembly (median) |", "|---|---|---|---|"]
    for s in sessions:
        b = s["epochs"]["post"]["count_baseline"]
        compact = sum(1 for sp, n05 in zip(b["field_spread"], b["field_spread_null05"]) if np.isfinite(sp) and sp < n05)
        jac = np.median(b["jaccard_complex_to_count"]) if b["jaccard_complex_to_count"] else np.nan
        lines.append(f"| {s['name']} | {b['n_assemblies']} | {compact} | {jac:.2f} |")


def hc11_example_figure(name="Achilles_11012013", variant="centred, T = event duration", n_show=4):
    """Recompute one variant for one session and draw the components that match the map above chance."""
    settings = {"uncentred, T = event duration": dict(center=False, scale=False), "centred, T = event duration": dict(center=True, scale=True)}[variant]
    path = next(p for p in session_paths() if p.name.startswith(name))
    session = load_session(path)
    run = running(session)
    intervals = nrem_intervals(session, "post")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    res = fit(times, duration * 1000, rng=np.random.default_rng(0), ica=False, **settings)
    centres = place_centres(session, cell_ids, run)
    stored = pickle.load(open(RESULTS / "hc11" / f"{name}.pkl", "rb"))
    comps = stored["epochs"]["post"]["variants"][variant]["components"]
    passing = []
    for c in comps:
        for sign, m in c["map"].items():
            if np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"]:
                passing.append((abs(m["rho"]) - m["null95"], c, sign))
    passing.sort(key=lambda t: -t[0])
    passing = passing[:n_show]
    if not passing:
        return
    fig, axes = plt.subplots(1, len(passing), figsize=(4.4 * len(passing), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (_, c, sign) in zip(axes, passing):
        j = c["index"]
        _, t, _, inc, _ = readout(res["patterns"][:, j], res["T_read_ms"], min_weight=0.2)
        cen = centres[sign]
        use = inc & np.isfinite(t) & np.isfinite(cen)
        ax.scatter(cen[use], t[use], s=60, color="#1f4e79", zorder=3)
        ax.set_xlabel(f"place-field centre, running direction {'+' if sign > 0 else '−'} (m)")
        ax.set_ylabel("member's relative time in the component (ms)")
        m = c["map"][sign]
        ax.set_title(f"component {j + 1}: {m['n_members_with_field']} member cells with a field\nSpearman {m['rho']:+.2f} (chance level {m['null95']:.2f})", fontsize=11)
        ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(f"{name.replace('_', ' ')}, POST sleep, {variant}: components whose members' order follows the track", y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES / "hc11_example_components.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def hc11_spectrum_figure(sessions, name="Achilles_11012013"):
    s = next(x for x in sessions if x["name"] == name)
    fig, ax = plt.subplots(figsize=(9, 5))
    for v, colour in (("uncentred, T = event duration", "#1f4e79"), ("centred, T = event duration", "#d4652f")):
        b = s["epochs"]["post"]["variants"][v]
        share = 100 * np.array(b["eigen_share"])
        ax.plot(np.arange(1, len(share) + 1), share, marker="o", color=colour, label=f"{v} (k = {b['n_kept']})")
        ax.axhline(100 * b["noise_bound_share"], color=colour, ls="--", lw=1)
    ax.set_xlabel("component")
    ax.set_ylabel("share of total variance (%)")
    ax.set_title(f"{name.replace('_', ' ')}, POST: eigenvalue spectrum; dashed = Marchenko–Pastur noise bound")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#e6e9ee")
    fig.tight_layout()
    fig.savefig(FIGURES / "hc11_spectrum.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    lines = ["# Results", ""]
    syn_file = RESULTS / "synthetic" / "results.pkl"
    if syn_file.exists():
        from scripts.run_synthetic import write_summary
        results = pickle.load(open(syn_file, "rb"))
        synthetic_figure(results)
        lines += ["## Synthetic routes", "", f"See `results/synthetic/summary.md` and `figures/synthetic_routes.png`.", ""]
    sessions = [pickle.load(open(f, "rb")) for f in glob.glob(str(RESULTS / "hc11" / "*.pkl"))]
    if sessions:
        sessions.sort(key=lambda s: ORDER.index(s["name"]))
        hc11_tables(sessions, lines)
        hc11_spectrum_figure(sessions)
        hc11_example_figure()
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
