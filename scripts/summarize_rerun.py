"""Tables and figures for the matched rerun (results/rerun): cPCA, seqNMF (K x L grid), PP-Seq (corrected).

Every method saw the same fixed subset of at most 1500 peak-centred event windows per dataset.
HMM congruence and Bayesian decoding are carried over from results/baselines (original windows,
all events) and marked as such in the figure.

Outputs
- results/rerun/summary.md
- figures/comparison_synthetic.png   (replaces the earlier version)
- figures/comparison_hc11.png        (replaces the earlier version)
"""

import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.routes import CONDITIONS  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
         "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
COL = {"cpca": "#1f4e79", "cpca2": "#6aa7d8", "seq": "#d4652f", "seq2": "#f0b58a", "pp": "#c9a227", "pp2": "#e8d58a",
       "hmm": "#1f8a65", "hmm2": "#9fd3bd", "bayes": "#7a5195", "pre": "#c3c9d2"}
SEQ_RECOMMENDED = (15, 120.0)        # K larger than the expected number of sequences, L about one replay (paper's guidance)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12.5,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


# ----------------------------------------------------------------------------- loading
def load(path):
    return pickle.load(open(path, "rb")) if Path(path).exists() else None


def load_synthetic():
    out = {}
    for name in CONDITIONS:
        letter = name[0]
        d = {"condition": name}
        for m in ("cpca", "seqnmf", "ppseq"):
            r = load(RES / "rerun" / "synthetic" / f"{letter}.{m}.pkl")
            if r:
                d[m] = r[m]
                d.setdefault("fraction_replay", r.get("fraction_replay"))
                d.setdefault("n_events", r.get("n_events"))
        out[name] = d
    return out


def load_hc11():
    out = {}
    for name in ORDER:
        d = {"name": name, "epochs": {"post": {}, "pre": {}}}
        for m in ("cpca", "seqnmf", "ppseq"):
            r = load(RES / "rerun" / f"{name}.{m}.pkl")
            if r:
                d["maze"] = r["maze"]
                d["circular"] = r["circular"]
                for ep in ("post", "pre"):
                    blk = r["epochs"].get(ep, {})
                    d["epochs"][ep][m] = blk.get(m)
                    for k in ("n_events_all", "n_events", "n_cells", "n_place"):
                        if k in blk:
                            d["epochs"][ep][k] = blk[k]
        d["baseline"] = load(RES / "baselines" / f"{name}.pkl")      # HMM, Bayes, old seqNMF (original windows)
        d["cpca_all"] = load(RES / "hc11" / f"{name}.pkl")           # cPCA on all events (original pipeline)
        out[name] = d
    return out


# ----------------------------------------------------------------------------- statistics
def hits_of(tests):
    """(hits, tests) over components/factors/types x directions."""
    h = t = 0
    for c in tests or []:
        for m in c["map"].values():
            if np.isfinite(m["rho"]):
                t += 1
                h += int(abs(m["rho"]) > m["null95"])
    return h, t


def clopper_pearson(h, n, level=0.95):
    if n == 0:
        return np.nan, np.nan
    lo = beta.ppf((1 - level) / 2, h, n - h + 1) if h > 0 else 0.0
    hi = beta.ppf(1 - (1 - level) / 2, h + 1, n - h) if h < n else 1.0
    return lo, hi


def ratio_with_ci(h, n, p0=0.05):
    """Hits relative to chance (h / (p0 n)) with the Clopper-Pearson interval mapped the same way."""
    if n == 0:
        return np.nan, (np.nan, np.nan)
    lo, hi = clopper_pearson(h, n)
    return h / (p0 * n), (lo / p0, hi / p0)


def best_route(routes):
    """routes: list of (rho, null95, n, idx) per planted route -> (median |rho|, median null95, fraction above)."""
    if not routes:
        return np.nan, np.nan, np.nan
    r = np.array([abs(x[0]) for x in routes], float)
    n = np.array([x[1] for x in routes], float)
    return float(np.median(r)), float(np.nanmedian(n)), float(np.mean(r > n))


def seq_grid_pick(grid, key_routes="routes"):
    """(recommended cell, best cell by median |rho| or by hits) from a seqNMF grid."""
    rec = next((g for g in grid if (g["K"], g["L_ms"]) == SEQ_RECOMMENDED), None)
    if key_routes == "routes":
        best = max(grid, key=lambda g: np.median([abs(x[0]) for x in g["routes"]]) if g["routes"] else -1)
    else:
        best = max(grid, key=lambda g: (hits_of(g["tests"])[0], -hits_of(g["tests"])[1]))
    return rec, best


# ----------------------------------------------------------------------------- figures
def wrap(text, width=17):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def figure_synthetic(syn):
    names = list(CONDITIONS)
    fig, axes = plt.subplots(2, 1, figsize=(17, 11.5), gridspec_kw={"height_ratios": [1.35, 1]})
    x = np.arange(len(names))
    ax = axes[0]
    series = [("cPCA, uncentred", COL["cpca"], lambda d: d.get("cpca", {}).get("uncentred", {}).get("routes")),
              ("cPCA, centred", COL["cpca2"], lambda d: d.get("cpca", {}).get("centred", {}).get("routes")),
              ("seqNMF, K = 15, L = 120 ms (paper's guidance)", COL["seq"], lambda d: (seq_grid_pick(d["seqnmf"]["grid"])[0] or {}).get("routes") if "seqnmf" in d else None),
              ("seqNMF, best of the K x L grid (optimistic)", COL["seq2"], lambda d: seq_grid_pick(d["seqnmf"]["grid"])[1]["routes"] if "seqnmf" in d else None),
              ("PP-Seq, type membership", COL["pp"], lambda d: d.get("ppseq", {}).get("routes_type")),
              ("PP-Seq, package membership", COL["pp2"], lambda d: d.get("ppseq", {}).get("routes_package"))]
    width = 0.84 / len(series)
    for i, (label, colour, get) in enumerate(series):
        vals, nulls = [], []
        for n in names:
            med, null, _ = best_route(get(syn[n]) or [])
            vals.append(med)
            nulls.append(null)
        pos = x - 0.42 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.9, color=colour, label=label, zorder=2)
        ax.hlines(nulls, pos - width * 0.45, pos + width * 0.45, colors="black", linewidth=2.0, zorder=3,
                  label="chance (95th percentile of the cell-permutation null)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ", 1)[0] + "\n" + wrap(n.split(" ", 1)[1]) for n in names], fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("|Spearman|, member order vs planted route\n(median over the planted routes)")
    ax.set_title("Planted routes, identical 1500-event subsets: how well does each method recover each route?")
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax = axes[1]
    pp_auc = [syn[n].get("ppseq", {}).get("auc_replay", np.nan) for n in names]
    pp_hit = [syn[n].get("ppseq", {}).get("flagged_replays", np.nan) for n in names]
    pp_fa = [syn[n].get("ppseq", {}).get("flagged_volleys", np.nan) for n in names]
    old = {s["condition"]: s for s in (load(RES / "baselines" / "synthetic.pkl") or [])}
    hmm_auc = [old.get(n, {}).get("methods", {}).get("HMM", {}).get("auc_replay", np.nan) for n in names]
    hmm_hit = [old.get(n, {}).get("methods", {}).get("HMM", {}).get("congruent_replays", np.nan) for n in names]
    hmm_fa = [old.get(n, {}).get("methods", {}).get("HMM", {}).get("congruent_volleys", np.nan) for n in names]
    w = 0.13
    ax.bar(x - 2.5 * w, hmm_auc, width=w * 0.92, color=COL["hmm"], label="HMM congruence: AUC, replays vs volleys (all 3000 events)", zorder=2)
    ax.bar(x - 1.5 * w, hmm_hit, width=w * 0.92, color=COL["hmm2"], label="HMM: share of planted replays flagged", zorder=2)
    ax.bar(x - 0.5 * w, hmm_fa, width=w * 0.92, color=COL["hmm2"], hatch="////", edgecolor="white", label="HMM: share of volleys flagged (false alarms)", zorder=2)
    ax.bar(x + 0.5 * w, pp_auc, width=w * 0.92, color=COL["pp"], label="PP-Seq sequence types: AUC, replays vs volleys", zorder=2)
    ax.bar(x + 1.5 * w, pp_hit, width=w * 0.92, color=COL["pp2"], label="PP-Seq: share of planted replays flagged", zorder=2)
    ax.bar(x + 2.5 * w, pp_fa, width=w * 0.92, color=COL["pp2"], hatch="////", edgecolor="white", label="PP-Seq: share of volleys flagged (false alarms)", zorder=2)
    ax.axhline(0.5, color="black", lw=1.3, ls="--", label="AUC of a random detector", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" ", 1)[0] for n in names])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("per-event detection")
    ax.set_title("Methods that mark single events as sequences (conditions A–C contain no volleys, so only the flagged share is defined)", fontsize=12.5)
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.suptitle("Synthetic planted routes: each method scored on what it outputs, after correcting the seqNMF and PP-Seq procedures", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "comparison_synthetic.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_hc11(hc):
    """Every method relative to its own chance level with 95% intervals, POST vs PRE, pooled over sessions; per-session dots."""
    FLOOR = 0.12
    methods = [
        ("cPCA (centred)\nall events\n[label-free]", COL["cpca"], lambda d, ep: hits_of(d["cpca_all"]["epochs"][ep]["variants"]["centred, T = event duration"]["components"]) if d["cpca_all"] else (0, 0), 0.05),
        ("cPCA (centred)\nsame 1500 events\n[label-free]", COL["cpca2"], lambda d, ep: hits_of((d["epochs"][ep].get("cpca") or {}).get("centred", {}).get("tests")), 0.05),
        ("PP-Seq, corrected\nsame 1500 events\n[label-free]", COL["pp"], lambda d, ep: hits_of((d["epochs"][ep].get("ppseq") or {}).get("tests_type")), 0.05),
        ("seqNMF, K=15 L=120\nsame 1500 events\n[label-free]", COL["seq"], lambda d, ep: hits_of((seq_grid_pick((d["epochs"][ep].get("seqnmf") or {}).get("grid", []), "tests")[0] or {}).get("tests")) if d["epochs"][ep].get("seqnmf") else (0, 0), 0.05),
        ("HMM congruence\nall events, original windows\n[label-free]", COL["hmm"], lambda d, ep: (100 * d["baseline"]["epochs"][ep]["HMM"]["congruent_99"], 1.0) if d["baseline"] else (np.nan, 0), None),
        ("Bayesian decoding\nall events, original windows\n[place map as input]", COL["bayes"], lambda d, ep: (100 * d["baseline"]["epochs"][ep]["Bayes"]["frac_sig"], 5.0) if d["baseline"] else (np.nan, 0), None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    x = np.arange(len(methods))
    pooled_table = {}
    for i, (epoch, shade, lab) in enumerate((("post", 1.0, "POST sleep (after the novel maze), pooled over sessions"), ("pre", 0.35, "PRE sleep (before the maze; control), pooled"))):
        pos = x - 0.2 + 0.4 * i
        pooled, lo_all, hi_all = [], [], []
        for m, (name, colour, get, p0) in enumerate(methods):
            vals = [get(hc[n], epoch) for n in ORDER]
            if p0 is not None:                                  # counts: pool hits and tests, binomial interval
                H = sum(v[0] for v in vals)
                N = sum(v[1] for v in vals)
                r, (lo, hi) = ratio_with_ci(H, N, p0)
                per = [v[0] / (p0 * v[1]) if v[1] else np.nan for v in vals]
                zero = np.array([v[1] > 0 and v[0] == 0 for v in vals])
            else:                                               # percentages: pooled mean over sessions, interval = session spread
                per = np.array([v[0] / v[1] for v in vals], float)
                r = float(np.nanmean(per))
                lo, hi = np.nanpercentile(per, 25), np.nanpercentile(per, 75)
                zero = np.zeros(len(vals), bool)
            pooled.append(r)
            lo_all.append(lo)
            hi_all.append(hi)
            per = np.array(per, float)
            jitter = np.linspace(-0.12, 0.12, len(per))
            ok = np.isfinite(per) & ~zero
            ax.scatter(pos[m] + jitter[ok], per[ok], s=24, color="black", alpha=0.75 if epoch == "post" else 0.4, zorder=4, label="single sessions" if (i == 0 and m == 0) else None)
            ax.scatter(pos[m] + jitter[zero], np.full(zero.sum(), FLOOR), s=30, facecolors="none", edgecolors="black", alpha=0.75 if epoch == "post" else 0.4, zorder=4,
                       label="session with zero matches (drawn at the floor)" if (i == 0 and m == 0) else None)
        pooled_table[epoch] = (pooled, lo_all, hi_all)
        bars = np.maximum(np.nan_to_num(pooled, nan=FLOOR), FLOOR)
        ax.bar(pos, bars, width=0.38, color=[c for _, c, _, _ in methods], alpha=shade, edgecolor="black" if epoch == "pre" else "none", linewidth=0.8, zorder=2, label=lab)
        for p_, val, lo, hi in zip(pos, pooled, lo_all, hi_all):
            if np.isfinite(val):
                ax.errorbar(p_, max(val, FLOOR), yerr=[[max(val, FLOOR) - max(lo, FLOOR)], [max(hi, FLOOR) - max(val, FLOOR)]], fmt="none", ecolor="black", elinewidth=1.4, capsize=4, zorder=5,
                            label="95% interval (binomial for counts; interquartile across sessions for percentages)" if (i == 0 and p_ == pos[0]) else None)
                ax.text(p_, max(hi, FLOOR) * 1.15, f"{val:.1f}×", ha="center", va="bottom", fontsize=10.5, fontweight="bold" if epoch == "post" else "normal")
    ax.axhline(1.0, color="black", lw=1.6, ls="--", zorder=3, label="chance level of each method (1×)")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 80)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _, _, _ in methods], fontsize=9.5)
    ax.set_ylabel("score relative to the method's own chance level (×, log scale)")
    ax.set_title("hc-11, eight sessions: every method against its own chance level, POST vs PRE sleep", pad=12)
    ax.yaxis.grid(True, which="both", color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    handles, labels_ = ax.get_legend_handles_labels()
    want = ("POST sleep (after the novel maze), pooled over sessions", "PRE sleep (before the maze; control), pooled", "chance level of each method (1×)",
            "95% interval (binomial for counts; interquartile across sessions for percentages)", "single sessions", "session with zero matches (drawn at the floor)")
    order = [labels_.index(l) for l in want if l in labels_]
    ax.legend([handles[o] for o in order], [labels_[o] for o in order], loc="upper left", frameon=False, fontsize=9.5)
    # right: pooled counts with intervals, POST and PRE side by side, for the four count-based methods
    ax = axes[1]
    count_methods = [m for m in methods if m[3] is not None]
    xx = np.arange(len(count_methods))
    for i, (epoch, shade) in enumerate((("post", 1.0), ("pre", 0.35))):
        pooled, lo_all, hi_all = pooled_table[epoch]
        idx = [k for k, m in enumerate(methods) if m[3] is not None]
        vals = [pooled[k] for k in idx]
        lo = [lo_all[k] for k in idx]
        hi = [hi_all[k] for k in idx]
        pos = xx - 0.2 + 0.4 * i
        ax.errorbar(pos, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="o", color="black", alpha=shade, ecolor="black", elinewidth=1.4, capsize=5, markersize=7, zorder=4)
        for p_, k in zip(pos, idx):
            ax.scatter(p_, pooled[k], s=90, color=methods[k][1], alpha=shade, edgecolors="black", zorder=5)
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.set_xticks(xx)
    ax.set_xticklabels([m[0].split("\n")[0] + "\n" + m[0].split("\n")[1] for m in count_methods], fontsize=9.5)
    ax.set_ylabel("map-matching tests relative to 5% chance, pooled over sessions\n(dark = POST, light = PRE; bars = 95% binomial interval)")
    ax.set_title("Order-producing methods: does the interval exclude chance,\nand does POST exceed PRE?", pad=12)
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "comparison_hc11.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return pooled_table


# ----------------------------------------------------------------------------- tables
def fmt_routes(routes):
    if not routes:
        return "—"
    return "; ".join(f"{abs(r[0]):.2f} [{r[1]:.2f}]{'*' if np.isfinite(r[1]) and abs(r[0]) > r[1] else ''}" for r in routes)


def fmt_hits(h, t):
    if t == 0:
        return "—"
    lo, hi = clopper_pearson(h, t)
    return f"{h}/{t} ({100*h/t:.0f}%, CI {100*lo:.0f}–{100*hi:.0f}%)"


def write_summary(syn, hc):
    L = ["# Matched rerun: cPCA, seqNMF (K x L grid), PP-Seq (corrected)", "",
         "Every method saw the same fixed subset of at most 1500 peak-centred event windows per dataset (each window is the event's own "
         "duration centred on its population peak). Order-producing methods are scored identically (`cpca.evaluate.order_vs_place`): "
         "Spearman between a component/factor/type's member order and the answer key, against the 95th percentile of a cell-permutation null; "
         "* marks above chance. Counts carry 95% Clopper–Pearson intervals. Corrections relative to the first comparison are listed in docs/baselines.md.", "",
         "## Synthetic planted routes", "",
         "| condition | replay | cPCA uncentred | cPCA centred | seqNMF K=15 L=120 (paper's guidance) | seqNMF best of 9 (K, L) cells | PP-Seq type membership (R, warp) | PP-Seq package membership | PP-Seq detection AUC (sequence types) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in syn.items():
        c = d.get("cpca", {})
        sq = d.get("seqnmf")
        rec, best = seq_grid_pick(sq["grid"]) if sq else (None, None)
        pp = d.get("ppseq")
        L.append(f"| {name} | {100*d.get('fraction_replay', np.nan):.0f}% | {fmt_routes(c.get('uncentred', {}).get('routes'))} (k={c.get('uncentred', {}).get('n_kept', '—')}) | "
                 f"{fmt_routes(c.get('centred', {}).get('routes'))} (k={c.get('centred', {}).get('n_kept', '—')}) | {fmt_routes(rec['routes']) if rec else '—'} | "
                 f"{fmt_routes(best['routes']) if best else '—'} (K={best['K']}, L={best['L_ms']:.0f}) | "
                 f"{fmt_routes(pp['routes_type']) if pp else '—'}" + (f" (R={pp['best_setting']['num_sequence_types']}, {'warp' if pp['best_setting']['num_warp_values'] > 1 else 'no warp'})" if pp else "") +
                 f" | {fmt_routes(pp['routes_package']) if pp else '—'} | " + (f"{pp['auc_replay']:.2f}" if pp and np.isfinite(pp['auc_replay']) else "—") + " |")
    L += ["", "## hc-11, POST vs PRE sleep (same 1500 events per epoch for cPCA, seqNMF, PP-Seq)", "",
          "| session | cells / place (+,−) | events (all → used) | cPCA centred, all events POST / PRE | cPCA centred, subset POST / PRE | PP-Seq type rule POST / PRE (R) | PP-Seq package rule POST / PRE | seqNMF K=15 L=120 POST / PRE | seqNMF best cell POST / PRE |",
          "|---|---|---|---|---|---|---|---|---|"]
    tot = {k: [0, 0, 0, 0] for k in ("all", "sub", "ppt", "ppp", "seqrec", "seqbest")}

    def acc(key, post, pre):
        tot[key] = [a + b for a, b in zip(tot[key], (*post, *pre))]

    for name in ORDER:
        d = hc[name]
        e = d["epochs"]
        cells = e["post"].get("n_cells", "—")
        place = e["post"].get("n_place", {})
        allc = {ep: hits_of(d["cpca_all"]["epochs"][ep]["variants"]["centred, T = event duration"]["components"]) if d["cpca_all"] else (0, 0) for ep in ("post", "pre")}
        sub = {ep: hits_of((e[ep].get("cpca") or {}).get("centred", {}).get("tests")) for ep in ("post", "pre")}
        ppt = {ep: hits_of((e[ep].get("ppseq") or {}).get("tests_type")) for ep in ("post", "pre")}
        ppp = {ep: hits_of((e[ep].get("ppseq") or {}).get("tests_package")) for ep in ("post", "pre")}
        rec = {ep: hits_of((seq_grid_pick((e[ep].get("seqnmf") or {}).get("grid", []), "tests")[0] or {}).get("tests")) if e[ep].get("seqnmf") else (0, 0) for ep in ("post", "pre")}
        bst = {ep: hits_of(seq_grid_pick((e[ep].get("seqnmf") or {}).get("grid", []), "tests")[1]["tests"]) if e[ep].get("seqnmf") else (0, 0) for ep in ("post", "pre")}
        for key, dd in (("all", allc), ("sub", sub), ("ppt", ppt), ("ppp", ppp), ("seqrec", rec), ("seqbest", bst)):
            acc(key, dd["post"], dd["pre"])
        R = {ep: (e[ep].get("ppseq") or {}).get("best_setting", {}).get("num_sequence_types", "—") for ep in ("post", "pre")}
        L.append(f"| {name} | {cells} / {place.get(1.0, '—')},{place.get(-1.0, '—')} | {e['post'].get('n_events_all', '—')} → {e['post'].get('n_events', '—')}; {e['pre'].get('n_events_all', '—')} → {e['pre'].get('n_events', '—')} | "
                 f"{allc['post'][0]}/{allc['post'][1]} / {allc['pre'][0]}/{allc['pre'][1]} | {sub['post'][0]}/{sub['post'][1]} / {sub['pre'][0]}/{sub['pre'][1]} | "
                 f"{ppt['post'][0]}/{ppt['post'][1]} / {ppt['pre'][0]}/{ppt['pre'][1]} ({R['post']}/{R['pre']}) | {ppp['post'][0]}/{ppp['post'][1]} / {ppp['pre'][0]}/{ppp['pre'][1]} | "
                 f"{rec['post'][0]}/{rec['post'][1]} / {rec['pre'][0]}/{rec['pre'][1]} | {bst['post'][0]}/{bst['post'][1]} / {bst['pre'][0]}/{bst['pre'][1]} |")
    L.append("| **pooled (hits/tests, % and 95% CI; chance 5%)** | | | " + " | ".join(
        f"**POST** {fmt_hits(tot[k][0], tot[k][1])}<br>**PRE** {fmt_hits(tot[k][2], tot[k][3])}" for k in ("all", "sub", "ppt", "ppp", "seqrec", "seqbest")) + " |")
    L += ["", "### Reading the pooled row", "",
          "A method shows maze-specific order when its POST interval excludes 5% and its PRE interval does not. "
          "Note that components of one decomposition share cells, so the binomial intervals are somewhat optimistic for cPCA; the session-level pattern "
          "(which sessions carry the POST hits) is given in the per-session rows.", ""]
    L += ["## PP-Seq sequence types on hc-11 (members under the type rule / the package rule, offset span, latent events per posterior sample)", ""]
    for name in ORDER:
        for ep in ("post", "pre"):
            p = hc[name]["epochs"][ep].get("ppseq")
            if not p:
                continue
            s = p["best_setting"]
            L.append(f"- {name} {ep.upper()} (R={s['num_sequence_types']}, {'warp' if s['num_warp_values'] > 1 else 'no warp'}, {100*p['flagged_fraction']:.0f}% of events carry a type spanning ≥ 20 ms): " +
                     "; ".join(f"{t['n_members']}/{t['n_members_package']} cells, span {t['lag_span_ms']:.0f} ms, {t['mean_latent_events']:.0f} events" for t in p["types"]))
    (RES / "rerun" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    FIG.mkdir(exist_ok=True)
    syn = load_synthetic()
    hc = load_hc11()
    figure_synthetic(syn)
    figure_hc11(hc)
    write_summary(syn, hc)


if __name__ == "__main__":
    main()
