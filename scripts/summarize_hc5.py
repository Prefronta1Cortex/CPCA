"""Tables and figures for the hc-5 wheel benchmark (results of scripts/run_hc5.py).

Outputs:
    results/hc5/summary.md
    figures/hc5_sequences.png / .pdf     what each method recovers, on held-out runs, for one predetermined example
    figures/hc5_benchmark.png / .pdf     recovery, success rate, coverage and run time across all sessions and fits
    figures/hc5_sensitivity.png / .pdf   settings that were not the headline (seqNMF K x L, cPCA centring, PP-Seq membership)

Headline settings were fixed before scoring (scripts/run_hc5.py): cPCA uncentred, seqNMF K = 5 with
L = one window, PP-Seq with the type membership rule.
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import beta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import hc5  # noqa: E402

RES = ROOT / "results" / "hc5"
FIG = ROOT / "figures"
EXAMPLE = ("i01_maze06_MS.002", 0, 0)          # session, repeat, fold: the catalog's "good behavior" session, first split
CONDS = ("aligned", "unaligned", "shuffled")
COND_LABEL = {"aligned": "Aligned windows", "unaligned": "Unaligned windows", "shuffled": "Shuffled control"}
COND_NOTE = {"aligned": "every window starts at wheel onset",
             "unaligned": "each window starts at its own random time before onset",
             "shuffled": "each cell's spikes shifted at random within a run"}

# (key, label, result file, variant, colour, marker)
METHODS = [("cpca", "Complex PCA", "cpca", "uncentred", "#1f4e79", "o"),
           ("seqnmf", "seqNMF", "seqnmf", "K5_Lfull", "#d4652f", "s"),
           ("ppseq", "PP-Seq", "ppseq", "type_rule", "#b8901a", "D")]
REF = ("reference", "average")
REF_COL = "#8a919c"
INK = "#1d2430"
SOFT = "#6b7380"
GRID = "#e9ecf1"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
    "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
    "xtick.major.size": 3, "ytick.major.size": 3, "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# ----------------------------------------------------------------------------- loading
def load():
    """records[(session, condition, method_key)] -> list of per-fit dicts (one per repeat x fold)."""
    rec = defaultdict(list)
    raw = {}
    for path in sorted(RES.glob("*/*.pkl")):
        r = pickle.load(open(path, "rb"))
        raw[(r["name"], r["condition"], r["repeat"], r["fold"], r["method"])] = r
        for variant, v in r["variants"].items():
            s = v["score"]
            rec[(r["name"], r["condition"], r["method"], variant)].append(
                {"rho": s["rho"], "p": s["p"], "null95": s["null95"], "coverage": s["coverage"], "n": s["n"], "n_key": s["n_key"],
                 "precision": s["precision"], "n_outputs": s["n_outputs"], "runtime_s": v.get("runtime_s", np.nan),
                 "repeat": r["repeat"], "fold": r["fold"], "extra": {k: v[k] for k in ("best_setting", "n_kept", "K", "L_ms") if k in v}})
    return rec, raw


def sessions(rec):
    return sorted({k[0] for k in rec})


def fits(rec, name, cond, method, variant):
    return rec.get((name, cond, method, variant), [])


def short(name):
    return name.replace("i01_", "").replace("_MS", "")


def clopper(k, n, a=0.05):
    if n == 0:
        return np.nan, np.nan
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def rho_of(f):
    return f["rho"] if np.isfinite(f["rho"]) else 0.0          # no testable output = nothing recovered


def session_median(rec, name, cond, method, variant):
    fs = fits(rec, name, cond, method, variant)
    return float(np.median([rho_of(f) for f in fs])) if fs else np.nan


# ----------------------------------------------------------------------------- figure 1
def tinted(colour, dark=False):
    return LinearSegmentedColormap.from_list("t", ["#ffffff", colour] + (["#0b0f16"] if dark else []), N=256)


def heat(ax, maps, order, key, W, colour, title, subtitle):
    """Held-out rate maps of the cells with a finite order, sorted by it, each row scaled to its peak.

    A dot on each row marks that cell's held-out time field: a recovered sequence is a column of
    dots running from top left to bottom right.
    """
    use = np.flatnonzero(np.isfinite(order))
    use = use[np.argsort(order[use], kind="stable")]
    m = maps[use]
    m = m / np.maximum(m.max(1, keepdims=True), 1e-9)
    ax.imshow(m, aspect="auto", cmap=tinted(colour), vmin=0, vmax=1.35, extent=[0, W, len(use), 0], interpolation="bilinear")
    ax.scatter(key[use], np.arange(len(use)) + 0.5, s=13, color=INK, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(0, W)
    ax.set_ylim(len(use), 0)
    ax.set_title(title, fontsize=11.5, pad=16)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=3)
    return len(use)


def best_order_vector(r, variant):
    v = r["variants"][variant]
    i = v["score"]["index"]
    if i < 0:
        return np.full(r["n_cells"], np.nan)
    o = np.asarray(v["orders"][i], float)
    return np.where(np.isfinite(r["key"]), o, np.nan)


def figure_sequences(rec, raw):
    name, rep, fold = EXAMPLE
    if not all((name, c, rep, fold, m[2]) in raw for c in ("aligned", "unaligned") for m in METHODS):
        print("figure 1: example fits not complete yet")
        return
    session = hc5.load_session(name)
    W = session["W_s"]
    any_r = raw[(name, "aligned", rep, fold, "cpca")]
    test, train, key = any_r["test"], any_r["train"], any_r["key"]
    maps, _ = hc5.rate_maps(session, test, smooth_s=0.75)

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 5, left=0.035, right=0.985, top=0.755, bottom=0.11, wspace=0.12, hspace=0.62)
    fig.text(0.035, 0.962, "What each method recovers from the wheel runs it was given", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.928, f"Session {short(name)}: {len(session['onsets_s'])} wheel runs, {len(session['cell_ids'])} CA1 principal cells. "
             f"Each method is fit on {len(train)} runs; every panel shows the other {len(test)} runs, which no method saw.",
             fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.905, "Each row is one cell's average firing rate on those runs against time since wheel onset (scaled to its peak), rows in the order the method "
             "returned. The dot is the cell's", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.882, "true firing time on the held-out runs: dots falling from top left to bottom right mean the method recovered the sequence.",
             fontsize=10.5, color=SOFT)

    # answer key
    ax = fig.add_subplot(gs[0, 0])
    heat(ax, maps, key, key, W, INK, "Answer key", f"held-out runs sorted by their own peak · {int(np.isfinite(key).sum())} cells")
    ax.set_ylabel("cells", color=SOFT)
    ax.set_xticks([0, W / 2, W])
    ax.set_xticklabels(["0", f"{W / 2:g}", f"{W:g} s"])

    # windows schematic
    ax = fig.add_subplot(gs[1, 0])
    off_u = raw[(name, "unaligned", rep, fold, "cpca")]["offsets_s"]
    L_u = raw[(name, "unaligned", rep, fold, "cpca")]["T_ms"] / 1000
    for i in range(len(train)):
        ax.plot([0, W], [i, i], color=REF_COL, lw=5, solid_capstyle="butt", alpha=0.55)
        ax.plot([off_u[i], off_u[i] + L_u], [i + 0.38, i + 0.38], color=INK, lw=2.6, solid_capstyle="butt")
    ax.axvline(0, color=INK, lw=0.8, ls=(0, (2, 1.6)))
    ax.set_ylim(len(train) - 0.3, -0.9)
    ax.set_xlim(-session["pre_s"], W + session["pre_s"])
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_xticks([-session["pre_s"], 0, W])
    ax.set_xticklabels([f"−{session['pre_s']:.0f}", "onset", f"{W:g} s"])
    ax.set_xlabel("time since wheel onset", color=SOFT)
    ax.set_title("Windows the methods get", fontsize=11.5, pad=16)
    ax.text(0, 1.015, "per training run: grey aligned, black unaligned", transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")

    columns = [("Align-and-average", REF[0], REF[1], REF_COL)] + [(lab, f, var, col) for _, lab, f, var, col, _ in METHODS]
    for row, cond in enumerate(("aligned", "unaligned")):
        for c, (label, method, variant, colour) in enumerate(columns):
            r = raw[(name, cond, rep, fold, method)]
            s = r["variants"][variant]["score"]
            order = best_order_vector(r, variant)
            sub = (f"ρ = {s['rho']:.2f} · p = {s['p']:.3f} · {s['n']} of {s['n_key']} cells" if np.isfinite(s["rho"])
                   else "no output with 6 or more cells that have a time field")
            ax = fig.add_subplot(gs[row, c + 1])
            heat(ax, maps, order, key, W, colour, label, sub)
            ax.set_xticks([0, W / 2, W])
            ax.set_xticklabels(["0", f"{W / 2:g}", f"{W:g} s"])
            if row == 1:
                ax.set_xlabel("time since wheel onset (held-out runs)", color=SOFT)
            if c == 0:
                pos = ax.get_position()
                fig.text(pos.x0, pos.y1 + 0.068, f"{COND_LABEL[cond]}", fontsize=12.5, fontweight="bold")
                fig.text(pos.x0 + 0.13, pos.y1 + 0.068, COND_NOTE[cond] + (f" ({L_u:.0f} s windows)" if cond == "unaligned" else f" ({W:.0f} s windows)"),
                         fontsize=10.5, color=SOFT)
                fig.add_artist(Line2D([pos.x0, 0.985], [pos.y1 + 0.062] * 2, color="#d5d9e0", lw=0.8))
    fig.text(0.035, 0.035, "Align-and-average is not a method: each cell's peak in the average of the training windows, i.e. what you get by assuming the windows share a clock.",
             fontsize=9, color=SOFT)
    fig.text(0.035, 0.015, "Each method's best output is shown (example fixed in advance: first split, first half). p is against a null that takes the best of that method's outputs "
             "under each permutation of the time fields.", fontsize=9, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hc5_sequences.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 2
def strip(ax, rec, names, cond, x0=0.0, show_ref=True):
    """Per-session medians (dots, joined across methods), median across sessions (bar), each method's chance (dash)."""
    xs = {m[0]: x0 + i for i, m in enumerate(METHODS)}
    per = {m[0]: np.array([session_median(rec, n, cond, m[2], m[3]) for n in names]) for m in METHODS}
    for j in range(len(names)):
        ys = [per[m[0]][j] for m in METHODS]
        ax.plot([xs[m[0]] for m in METHODS], ys, color="#cdd2da", lw=0.9, zorder=1)
    for key, label, f, var, col, mk in METHODS:
        y = per[key]
        jitter = np.linspace(-0.09, 0.09, len(names))
        ax.scatter(xs[key] + jitter, y, s=30, color=col, marker=mk, edgecolor="white", linewidth=0.7, zorder=3, alpha=0.95)
        med = np.nanmedian(y)
        q1, q3 = np.nanpercentile(y, [25, 75])
        ax.plot([xs[key] - 0.28, xs[key] + 0.28], [med, med], color=col, lw=3.2, solid_capstyle="round", zorder=4)
        ax.plot([xs[key] + 0.33] * 2, [q1, q3], color=col, lw=1.4, zorder=4)
        nulls = [f_["null95"] for n in names for f_ in fits(rec, n, cond, f, var) if np.isfinite(f_["null95"])]
        if nulls:
            c = float(np.median(nulls))
            ax.fill_between([xs[key] - 0.42, xs[key] + 0.42], -1, c, facecolor="white", edgecolor="#dde1e8", hatch="////", lw=0, zorder=0.5)
            ax.plot([xs[key] - 0.42, xs[key] + 0.42], [c, c], color=INK, lw=1.0, ls=(0, (2, 1.6)), zorder=2)
        ax.text(xs[key], -0.62, f"{med:.2f}", ha="center", va="center", fontsize=9.5, color=col, fontweight="bold")
    if show_ref:
        ref = np.array([session_median(rec, n, cond, *REF) for n in names])
        if np.isfinite(ref).any():
            m_ = np.nanmedian(ref)
            ax.axhspan(*np.nanpercentile(ref, [25, 75]), color=REF_COL, alpha=0.13, lw=0, zorder=0)
            ax.axhline(m_, color=REF_COL, lw=1.2, zorder=0)
            ax.text(x0 - 0.5, np.nanpercentile(ref, 75) + 0.025, f"align-and-average {m_:.2f}", ha="left", va="bottom", fontsize=8.8, color=SOFT,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=5)
    ax.axhline(0, color="#b9bfc8", lw=0.8, zorder=0)
    ax.set_xticks([xs[m[0]] for m in METHODS])
    ax.set_xticklabels([m[1] for m in METHODS])
    ax.set_xlim(x0 - 0.55, x0 + len(METHODS) - 0.45)
    ax.set_ylim(-0.72, 1.0)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    return per


def figure_benchmark(rec):
    names = sessions(rec)
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.755, bottom=0.07, wspace=0.26, hspace=0.6)
    fig.text(0.035, 0.962, "Recovering the wheel sequence from runs the method did not see", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.927, f"{len(names)} sessions of one rat (hc-5). Each session is split in half three times; each method is fit on one half and scored on the other, "
             "then the halves swap (6 fits per session and condition).", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.904, "Score: Spearman correlation between the order of the best output's member cells and their time fields on the held-out runs "
             "(signed: a sequence read backwards scores negative).", fontsize=10.5, color=SOFT)

    letters = iter("ABCDEF")
    for i, cond in enumerate(CONDS):
        ax = fig.add_subplot(gs[0, i])
        strip(ax, rec, names, cond, show_ref=True)
        ax.set_title(f"{next(letters)}   {COND_LABEL[cond]}", pad=18)
        ax.text(0, 1.02, COND_NOTE[cond], transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")
        if i == 0:
            ax.set_ylabel("held-out order correlation ρ\n(dots: sessions, median of 6 fits)")
    handles = [Line2D([], [], marker="o", color=SOFT, ls="", markersize=5.5, markeredgecolor="white"), Line2D([], [], color=INK, lw=3.2),
               Line2D([], [], color=INK, lw=1.4), plt.Rectangle((0, 0), 1, 1, fc="white", ec="#b9bfc8", hatch="////", lw=0.8),
               plt.Rectangle((0, 0), 1, 1, color=REF_COL, alpha=0.25), Line2D([], [], color="#cdd2da", lw=0.9)]
    fig.legend(handles, ["one session (median of its fits)", "median over sessions (value at the bottom)", "interquartile range over sessions",
                         "chance zone: below that method's 95% null (best of its outputs)", "align-and-average reference (median, IQR)", "same session across methods"],
               loc="upper left", bbox_to_anchor=(0.03, 0.89), ncol=3, frameon=False, fontsize=9.2, handlelength=2.0, columnspacing=2.2)

    # D: fraction of fits with a recovered sequence
    ax = fig.add_subplot(gs[1, 0])
    width = 0.24
    for j, (key, label, f, var, col, mk) in enumerate(METHODS):
        for i, cond in enumerate(CONDS):
            fs = [x for n in names for x in fits(rec, n, cond, f, var)]
            k = sum(np.isfinite(x["p"]) and x["p"] < 0.05 for x in fs)
            lo, hi = clopper(k, len(fs))
            x = i + (j - 1) * width
            ax.bar(x, k / max(len(fs), 1), width * 0.86, color=col, alpha=0.9 if cond != "shuffled" else 0.45, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK, lw=1.0, zorder=3)
            ax.text(x, hi + 0.03, f"{k}/{len(fs)}", ha="center", va="bottom", fontsize=8, color=SOFT, rotation=90)
    ax.axhline(0.05, color=INK, lw=0.9, ls=(0, (2, 1.6)))
    ax.text(2.45, 0.065, "5% expected by chance", ha="right", fontsize=8.5, color=SOFT)
    ax.set_xticks(range(len(CONDS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("fits that recover the sequence\n(p < 0.05; bars: 95% binomial interval)")
    ax.set_title(f"{next(letters)}   How often a fit recovers the sequence", pad=10)

    # E: coverage
    ax = fig.add_subplot(gs[1, 1])
    for j, (key, label, f, var, col, mk) in enumerate(METHODS):
        for i, cond in enumerate(("aligned", "unaligned")):
            y = [np.median([x["coverage"] for x in fits(rec, n, cond, f, var)]) for n in names if fits(rec, n, cond, f, var)]
            x = i + (j - 1) * width
            ax.scatter(x + np.linspace(-0.05, 0.05, len(y)), y, s=26, color=col, marker=mk, edgecolor="white", linewidth=0.6, zorder=3)
            if y:
                ax.plot([x - 0.1, x + 0.1], [np.median(y)] * 2, color=col, lw=3, solid_capstyle="round", zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([COND_LABEL["aligned"], COND_LABEL["unaligned"]])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("cells with a time field that are\nmembers of the best output (sessions)")
    ax.set_title(f"{next(letters)}   How much of the sequence the best output covers", pad=10)
    ax.legend([Line2D([], [], marker=m[5], color=m[4], ls="", markersize=6.5, markeredgecolor="white") for m in METHODS], [m[1] for m in METHODS],
              loc="lower left", frameon=False, ncol=3, fontsize=9, handletextpad=0.3, columnspacing=1.0)

    # F: run time
    ax = fig.add_subplot(gs[1, 2])
    for j, (key, label, f, var, col, mk) in enumerate(METHODS):
        t = np.array([x["runtime_s"] for n in names for c in CONDS for x in fits(rec, n, c, f, var)])
        t = t[np.isfinite(t) & (t > 0)]
        if not t.size:
            continue
        ax.scatter(np.full(t.size, j) + np.random.default_rng(j).uniform(-0.16, 0.16, t.size), t, s=9, color=col, alpha=0.45, lw=0, zorder=2)
        ax.plot([j - 0.26, j + 0.26], [np.median(t)] * 2, color=col, lw=3, solid_capstyle="round", zorder=3)
        ax.text(j, np.median(t) * 3.2, fmt_time(np.median(t)), ha="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 2e4)
    ax.set_yticks([1e-3, 1, 60, 3600])
    ax.set_yticklabels(["1 ms", "1 s", "1 min", "1 h"])
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[1] for m in METHODS])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("time per fit, one CPU core")
    ax.set_title(f"{next(letters)}   Cost of one fit, settings selection included", pad=10)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hc5_benchmark.{ext}", dpi=200)
    plt.close(fig)


def fmt_time(s):
    return f"{s * 1000:.0f} ms" if s < 1 else (f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min")


# ----------------------------------------------------------------------------- figure 3
def figure_sensitivity(rec):
    names = sessions(rec)
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 4, left=0.05, right=0.985, top=0.74, bottom=0.14, wspace=0.42, width_ratios=[1.25, 1.25, 1, 1])
    fig.text(0.035, 0.94, "Settings that were not the headline", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.885, "Same fits and scoring as the benchmark. Headline settings, fixed in advance, are outlined.", fontsize=10.5, color=SOFT)
    letters = iter("ABCD")
    Ks, Ls = (3, 5, 10), ("half", "full")
    for i, cond in enumerate(("aligned", "unaligned")):
        ax = fig.add_subplot(gs[0, i])
        frac = np.full((len(Ks), len(Ls)), np.nan)
        med = np.full((len(Ks), len(Ls)), np.nan)
        for a, K in enumerate(Ks):
            for b, L in enumerate(Ls):
                fs = [x for n in names for x in fits(rec, n, cond, "seqnmf", f"K{K}_L{L}")]
                if fs:
                    frac[a, b] = np.mean([np.isfinite(x["p"]) and x["p"] < 0.05 for x in fs])
                    med[a, b] = np.median([rho_of(x) for x in fs])
        ax.imshow(frac, cmap=tinted("#d4652f"), vmin=0, vmax=1.15, aspect="auto")
        for a in range(len(Ks)):
            for b in range(len(Ls)):
                if np.isfinite(frac[a, b]):
                    ax.text(b, a, f"{frac[a, b]:.0%}\nρ {med[a, b]:.2f}", ha="center", va="center", fontsize=10, color=INK,
                            fontweight="bold" if (Ks[a], Ls[b]) == SEQ_HEADLINE else "normal")
        hk, hl = Ks.index(SEQ_HEADLINE[0]), Ls.index(SEQ_HEADLINE[1])
        ax.add_patch(plt.Rectangle((hl - 0.5, hk - 0.5), 1, 1, fill=False, ec=INK, lw=2))
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels(["L = half a window", "L = one window"])
        ax.set_yticks(range(len(Ks)))
        ax.set_yticklabels([f"K = {k}" for k in Ks])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.set_title(f"{next(letters)}   seqNMF, {COND_LABEL[cond].lower()}", pad=18)
        ax.text(0, 1.02, "fits that recover the sequence; median ρ", transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")

    for (f, a_var, b_var, a_lab, b_lab, col, title) in (
            ("cpca", "uncentred", "centred", "uncentred", "centred", "#1f4e79", "Complex PCA: centring"),
            ("ppseq", "type_rule", "package_rule", "type rule", "package rule", "#b8901a", "PP-Seq: membership rule")):
        ax = fig.add_subplot(gs[0, 2 if f == "cpca" else 3])
        for i, cond in enumerate(("aligned", "unaligned")):
            a = np.array([session_median(rec, n, cond, f, a_var) for n in names])
            b = np.array([session_median(rec, n, cond, f, b_var) for n in names])
            x0 = i * 2.4
            for y1, y2 in zip(a, b):
                ax.plot([x0, x0 + 1], [y1, y2], color="#cdd2da", lw=0.9, zorder=1)
            ax.scatter(np.full(len(a), x0), a, s=28, color=col, edgecolor=INK, linewidth=1.2, zorder=3)
            ax.scatter(np.full(len(b), x0 + 1), b, s=28, color="white", edgecolor=col, linewidth=1.4, zorder=3)
            ax.text(x0 + 0.5, 1.07, COND_LABEL[cond].split()[0], ha="center", fontsize=9.5, color=SOFT)
        ax.set_xticks([0, 1, 2.4, 3.4])
        ax.set_xticklabels([a_lab, b_lab, a_lab, b_lab], fontsize=8.8, rotation=30, ha="right")
        ax.set_xlim(-0.5, 3.9)
        ax.set_ylim(-0.6, 1.12)
        ax.axhline(0, color="#b9bfc8", lw=0.8, zorder=0)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("held-out ρ (sessions, median of 6 fits)" if f == "cpca" else "")
        ax.set_title(f"{next(letters)}   {title}", pad=18)
        ax.text(0, 1.02, "headline = filled", transform=ax.transAxes, fontsize=9, color=SOFT, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hc5_sensitivity.{ext}", dpi=200)
    plt.close(fig)


SEQ_HEADLINE = (5, "full")


# ----------------------------------------------------------------------------- summary
def summary(rec, raw):
    names = sessions(rec)
    lines = ["# hc-5 wheel benchmark: complex PCA, seqNMF, PP-Seq", "",
             "Generated by `scripts/summarize_hc5.py` from `results/hc5/` (`scripts/run_hc5.py`). Design and rationale: `docs/benchmark_hc5.md`.", "",
             "Each session's wheel runs are split in half three times; each method is fit on one half and scored on the other, then the halves swap "
             "(6 fits per session and condition; the shuffled control uses the first split only, 2 fits). Score: signed Spearman ρ between the order of "
             "the best output's members and their held-out time fields; p against a null that takes the best of the method's outputs under each "
             "permutation. Cells: entries are the median ρ over fits, then the number of fits with p < 0.05.", ""]
    sess_rows = []
    for n in names:
        s = hc5.load_session(n)
        sess_rows.append(f"| {short(n)} | {s['W_s']:.0f} s | {len(s['onsets_s'])} | {len(s['cell_ids'])} |")
    lines += ["## Sessions", "", "| session | window W | wheel runs | CA1 principal cells |", "|---|---|---|---|"] + sess_rows + [""]
    cols = [("align-and-average (reference)",) + REF] + [(m[1], m[2], m[3]) for m in METHODS] + [
        ("cPCA centred", "cpca", "centred"), ("PP-Seq package rule", "ppseq", "package_rule")]
    for cond in CONDS:
        lines += [f"## {COND_LABEL[cond]} ({COND_NOTE[cond]})", "", "| session | " + " | ".join(c[0] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for n in names:
            cells = []
            for _, f, var in cols:
                fs = fits(rec, n, cond, f, var)
                if not fs:
                    cells.append("—")
                    continue
                k = sum(np.isfinite(x["p"]) and x["p"] < 0.05 for x in fs)
                cells.append(f"{np.median([rho_of(x) for x in fs]):.2f} ({k}/{len(fs)})")
            lines.append(f"| {short(n)} | " + " | ".join(cells) + " |")
        pooled = []
        for _, f, var in cols:
            fs = [x for n in names for x in fits(rec, n, cond, f, var)]
            k = sum(np.isfinite(x["p"]) and x["p"] < 0.05 for x in fs)
            lo, hi = clopper(k, len(fs))
            pooled.append(f"**{np.median([rho_of(x) for x in fs]) if fs else np.nan:.2f}** ({k}/{len(fs)}, {k / max(len(fs), 1):.0%}, CI {lo:.0%}–{hi:.0%})")
        lines += ["| **all fits** | " + " | ".join(pooled) + " |", ""]
    lines += ["## seqNMF K x L grid (fits with p < 0.05; median ρ)", "", "| condition | " + " | ".join(f"K={K}, L={L}" for K in (3, 5, 10) for L in ("half", "full")) + " |",
              "|---" * 7 + "|"]
    for cond in CONDS:
        row = []
        for K in (3, 5, 10):
            for L in ("half", "full"):
                fs = [x for n in names for x in fits(rec, n, cond, "seqnmf", f"K{K}_L{L}")]
                k = sum(np.isfinite(x["p"]) and x["p"] < 0.05 for x in fs)
                row.append(f"{k}/{len(fs)}; {np.median([rho_of(x) for x in fs]) if fs else np.nan:.2f}")
        lines.append(f"| {cond} | " + " | ".join(row) + " |")
    lines += ["", "## Outputs per fit (median over fits) and run time", "", "| method | outputs, aligned | outputs, unaligned | members of best output / cells with a field, aligned | median run time per fit |", "|---|---|---|---|---|"]
    for _, label, f, var, _, _ in METHODS:
        fa = [x for n in names for x in fits(rec, n, "aligned", f, var)]
        fu = [x for n in names for x in fits(rec, n, "unaligned", f, var)]
        rt = [x["runtime_s"] for n in names for c in CONDS for x in fits(rec, n, c, f, var)]
        lines.append(f"| {label} | {np.median([x['n_outputs'] for x in fa]) if fa else np.nan:.0f} | {np.median([x['n_outputs'] for x in fu]) if fu else np.nan:.0f} | "
                     f"{np.median([x['n'] for x in fa]) if fa else np.nan:.0f} / {np.median([x['n_key'] for x in fa]) if fa else np.nan:.0f} | {fmt_time(np.median(rt)) if rt else '—'} |")
    Rs = [r["variants"]["type_rule"]["best_setting"]["num_sequence_types"] for k, r in raw.items() if k[4] == "ppseq" and k[1] != "shuffled"]
    if Rs:
        lines += ["", f"PP-Seq number of sequence types chosen by the held-out sweep (aligned and unaligned fits): " +
                  ", ".join(f"R={R}: {Rs.count(R)}" for R in sorted(set(Rs))) + "."]
    (RES / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    rec, raw = load()
    if not rec:
        print("no results yet")
        return
    summary(rec, raw)
    figure_sequences(rec, raw)
    figure_benchmark(rec)
    figure_sensitivity(rec)
    print(f"{len(raw)} result files -> {RES / 'summary.md'}, figures/hc5_*.png")


if __name__ == "__main__":
    main()
