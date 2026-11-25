"""Figures and table for forward and backward plays (results of scripts/run_direction.py).

    figures/direction_synthetic.png / .pdf   why time reversal is conjugation; the split on planted data
    figures/direction_hc11.png / .pdf        direction labels on hc-11 against the waking map, POST vs PRE
    results/direction/summary.md
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from scipy.stats import beta, binomtest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_direction import HC11_ORDER, NO_ORDER, synthetic_conditions  # noqa: E402

RES = ROOT / "results" / "direction"
FIG = ROOT / "figures"
EXAMPLE = "Achilles_10252013"        # fixed in advance: the session with the most place cells
INK, SOFT, GRID = "#1d2430", "#6b7380", "#e9ecf1"
POST_C, PRE_C = "#1f4e79", "#9fb3c8"
CEN_C, UNC_C, BEFORE_C = "#1f4e79", "#3b8bba", "#b3bac4"
FWD_C, BWD_C = "#1f4e79", "#d4652f"
POS_CMAP = plt.get_cmap("viridis")

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8, "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK,
    "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})

BOTH = ["B", "C", "F", "G", "H", "L0", "L2", "L3", "L4", "L5"]
FORWARD_ONLY = ["A", "D", "E", "L1"]
CONTROL = ["S0"]


def load(family):
    return [pickle.load(open(p, "rb")) for p in sorted((RES / family).glob("*.pkl"))]


def short(name):
    rat, day = name.split("_")
    return f"{rat} {day[:2]}/{day[2:4]}"


def cond_key(cond):
    return cond.split(" ")[0]


def cond_label(cond):
    k, rest = cond.split(" ", 1)
    rest = rest.replace("L0 with each event's spike trains shuffled among the cells that fired (no order)", "L0, cells shuffled in each event")
    return f"{k}  {rest}"


def clopper(k, n):
    if n == 0:
        return np.nan, np.nan
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def hit(m):
    return m is not None and np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"]


def panel_letter(ax, letter, x=-0.12, y=1.10):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")


# ----------------------------------------------------------------------------- synthetic
def synthetic_table(S):
    """condition -> convention -> lists over (draw, route)."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in S:
        k = cond_key(r["condition"])
        for conv, v in r["variants"].items():
            d = by[k][conv]
            for row in v["routes"]:
                d["before"].append(row["before"])
                d["after"].append(row["after"])
                if "accuracy" in row:
                    d["accuracy"].append(row["accuracy"])
                    d["true_minority"].append(row["true_minority"])
                    d["minority"].append(row["minority"])
                    d["accepted"].append(row["accepted"])
            for c in v["components"]:
                if c is None:
                    continue
                d["comp_accepted"].append(c["accepted"])
                d["comp_heldout"].append(c["heldout_share"])
                d["comp_passes"].append(bool(c["heldout_p"] < 0.05))
            d["n_kept"].append(v["n_kept"])
    return by


SHORT = {"B": "one route, both ways, 25 ms", "C": "one route, both ways, 80 ms", "F": "two routes, 15% replay",
         "G": "realistic, 15% replay", "H": "realistic, 40% replay", "L0": "one map on 60 cells", "L2": "two maps, same cells",
         "L3": "L2 + partial replays", "L4": "L2 + assemblies", "L5": "L2 + variable pacing", "A": "one route in every event",
         "D": "two routes, different cells", "E": "two routes, shared cells", "L1": "two maps, same cells", "S0": "L0, cells shuffled per event"}


def intuition_panel(ax):
    """Forward event, the same order played backward, and the backward event conjugated."""
    rng = np.random.default_rng(3)
    t = np.sort(np.linspace(0.08, 0.92, 7) + rng.normal(0, 0.015, 7))
    colours = POS_CMAP(np.linspace(0.08, 0.92, 7))
    centres = [0.0, 3.0, 6.0]
    angles = [np.pi * t, np.pi * (1 - t), -np.pi * (1 - t)]
    titles = ["forward play", "backward play", "backward play, conjugated"]
    notes = ["spike at time t  →  angle πt / T", "time t  →  T − t", "angles flip sign: the forward\norder, turned by half a circle"]
    th = np.linspace(0, 2 * np.pi, 200)
    for c, ang, title, note in zip(centres, angles, titles, notes):
        ax.plot(c + np.cos(th), np.sin(th), color=GRID, lw=1.0, zorder=1)
        ax.plot([c - 1.15, c + 1.15], [0, 0], color=GRID, lw=0.8, zorder=1)
        for a, col in zip(ang, colours):
            ax.add_patch(FancyArrowPatch((c, 0), (c + 0.93 * np.cos(a), 0.93 * np.sin(a)), arrowstyle="-|>", mutation_scale=9,
                                         color=col, lw=1.6, zorder=3, shrinkA=0, shrinkB=0))
        ax.plot(c, 0, "o", ms=3, color=INK, zorder=4)
        ax.text(c, 1.22, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.text(c, -1.25, note, ha="center", va="top", fontsize=8.5, color=SOFT, linespacing=1.25)
    for x0 in (1.3, 4.3):
        ax.annotate("", xy=(x0 + 0.4, 0.0), xytext=(x0, 0.0), arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=1.0))
    for i, col in enumerate(colours):
        ax.plot(-0.9 + 0.16 * i, -2.25, "s", ms=6, color=col)
    ax.text(-0.9 + 0.16 * 7, -2.25, "cells, in route order", va="center", fontsize=8.5, color=SOFT)
    ax.text(7.75, 0.95, "Why one template covers both directions", fontsize=10.5, fontweight="bold", va="top")
    ax.text(7.75, 0.45,
            "A lag between two cells is a difference of their angles. Conjugating the\n"
            "backward play gives back the forward play's lags, so a single template q\n"
            "matches forward events through |⟨s, q⟩| and backward events through\n"
            "|⟨s, q̄⟩|; both are unchanged by when the event started.\n\n"
            "After centring, the two directions of one route cancel each other's\n"
            "phases on a shared component (the 'before' values in b). The split\n"
            "labels every event by the larger of the two matches, conjugates the\n"
            "backward ones, and refits the template until the labels stop changing.",
            fontsize=8.8, va="top", color=INK, linespacing=1.35)
    ax.set_xlim(-1.3, 14.2)
    ax.set_ylim(-2.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")


def recovery_panel(ax, by, conv, rows, show_labels):
    col = CEN_C if conv == "centred" else UNC_C
    y = np.arange(len(rows))[::-1]
    for yi, (k, label) in zip(y, rows):
        if k is None:
            ax.axhline(yi, color=GRID, lw=0.8, zorder=0)
            if show_labels:
                ax.text(-0.015, yi, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=SOFT,
                        fontstyle="italic")
            continue
        d = by[k][conv]
        b, a = np.array(d["before"], float), np.array(d["after"], float)
        if not np.isfinite(a).any():
            ax.text(0.02, yi, "no component kept: centring removes a route that is in every event", va="center", fontsize=8.3, color=SOFT)
            continue
        mb, ma = np.nanmedian(b), np.nanmedian(a)
        ax.plot(np.nanpercentile(a, [25, 75]), [yi, yi], color=col, lw=6, alpha=0.18, solid_capstyle="round", zorder=1)
        ax.plot([mb, ma], [yi, yi], color=SOFT, lw=1.0, zorder=2)
        ax.plot(mb, yi, "o", ms=7, mfc="white", mec=BEFORE_C, mew=1.6, zorder=3)
        ax.plot(ma, yi, "o", ms=7, color=col, zorder=4)
    ticks = [(yi, k, lab) for yi, (k, lab) in zip(y, rows) if k is not None]
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([f"{k}   {lab}" for _, k, lab in ticks] if show_labels else [], fontsize=9)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("order recovered, |Spearman ρ| with the planted route")


def figure_synthetic(S):
    by = synthetic_table(S)
    rows = ([(None, "both directions")] + [(k, SHORT[k]) for k in BOTH] + [(None, "forward only")] +
            [(k, SHORT[k]) for k in FORWARD_ONLY] + [(None, "no order")] + [(k, SHORT[k]) for k in CONTROL])
    fig = plt.figure(figsize=(14.2, 14.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62, 1.2, 0.95], hspace=0.42, wspace=0.34, left=0.05, right=0.985, top=0.935, bottom=0.075)
    fig.suptitle("Forward and backward plays are one template and its conjugate", x=0.035, y=0.985, ha="left", fontsize=15,
                 fontweight="bold")
    fig.text(0.035, 0.958, "planted data: 10 fresh draws per condition, 80 cells, 3000 events; complex PCA, then the direction split "
             "of every kept component", fontsize=9.8, color=SOFT)

    ax = fig.add_subplot(gs[0, :])
    intuition_panel(ax)
    ax.set_anchor("W")
    fig.text(0.035, ax.get_position().y1 - 0.005, "a", fontsize=14, fontweight="bold", va="top")

    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.05)
    axes = [fig.add_subplot(sub[0, 0]), None]
    pos = axes[0].get_position()
    axes[0].set_position([0.235, pos.y0, 0.36, pos.height])
    axes[1] = fig.add_axes([0.62, pos.y0, 0.36, pos.height])
    for i, (axr, conv) in enumerate(zip(axes, ("centred", "uncentred"))):
        recovery_panel(axr, by, conv, rows, show_labels=i == 0)
        axr.set_title(f"{conv} complex PCA", fontsize=10.5, loc="center", fontweight="normal", color=SOFT)
    fig.text(0.055, pos.y1 + 0.03, "Order recovered for each planted route: the component's order, then the order after the split",
             fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(0.035, pos.y1 + 0.028, "b", fontsize=14, fontweight="bold", va="bottom")
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=BEFORE_C, mew=1.6, ms=7, label="component's order"),
               Line2D([], [], marker="o", ls="", color=CEN_C, ms=7, label="after the split (guarded), median"),
               Line2D([], [], color=CEN_C, lw=6, alpha=0.18, label="middle half of draws")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, pos.y1 + 0.026), ncol=3, frameon=False, fontsize=8.8,
               handletextpad=0.4, columnspacing=1.2)

    # c: direction accuracy
    ax = fig.add_subplot(gs[2, 0])
    rng = np.random.default_rng(0)
    for i, k in enumerate(BOTH):
        for conv, dx, col in (("centred", -0.17, CEN_C), ("uncentred", 0.17, UNC_C)):
            acc = np.array(by[k][conv]["accuracy"], float)
            if acc.size == 0:
                continue
            ax.scatter(i + dx + rng.uniform(-0.07, 0.07, acc.size), acc, s=11, color=col, alpha=0.45, lw=0, zorder=2)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [np.median(acc)] * 2, color=col, lw=2.2, zorder=3)
    ax.axhline(0.5, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.text(len(BOTH) - 0.45, 0.51, "chance", ha="right", va="bottom", fontsize=8.5, color=SOFT)
    ax.set_xticks(range(len(BOTH)))
    ax.set_xticklabels(BOTH)
    ax.set_ylim(0.42, 1.04)
    ax.set_xlim(-0.6, len(BOTH) - 0.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("events labelled in the planted direction")
    ax.set_title("Direction of each event", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CEN_C, ms=5, label="centred"),
                       Line2D([], [], marker="o", ls="", color=UNC_C, ms=5, label="uncentred"),
                       Line2D([], [], color=INK, lw=2.2, label="median")], loc="lower left", frameon=False, fontsize=8.5, ncol=3,
              bbox_to_anchor=(0.0, 0.03))
    ax.text(0, -0.2, "conditions with both directions; G, H and L3 replay 40% segments,\nwhich one per-cell template cannot fit",
            transform=ax.transAxes, fontsize=8.5, color=SOFT, va="top")
    panel_letter(ax, "c", x=-0.17, y=1.03)

    # d: estimated against planted reverse share
    ax = fig.add_subplot(gs[2, 1])
    ax.plot([0, 0.55], [0, 0.55], color=SOFT, lw=0.8, ls=(0, (3, 3)), zorder=1)
    for group, col, mk, name in ((BOTH, CEN_C, "o", "both directions planted"), (FORWARD_ONLY, BWD_C, "s", "forward only planted")):
        x = np.concatenate([by[k]["centred"]["true_minority"] for k in group]).astype(float)
        y = np.concatenate([by[k]["centred"]["minority"] for k in group]).astype(float)
        ax.scatter(x + rng.uniform(-0.004, 0.004, x.size), y, s=14, color=col, alpha=0.5, lw=0, marker=mk, zorder=2, label=name)
    l1 = [(t, m) for t, m in zip(by["L1"]["centred"]["true_minority"], by["L1"]["centred"]["minority"]) if m > 0.3]
    if l1:
        ax.annotate("L1: the return map is close to the\noutbound map reversed, so the\ntwo maps pair up as one route\nplayed both ways",
                    xy=(0.006, l1[0][1] - 0.006), xytext=(0.04, 0.44), fontsize=8, color=BWD_C, va="top",
                    arrowprops=dict(arrowstyle="-", color=BWD_C, lw=0.8))
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("planted share of reverse plays")
    ax.set_ylabel("estimated share")
    ax.set_title("How many plays run backwards", pad=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.3)
    ax.text(0, -0.2, "centred; the smaller direction among the route's\nown expressing events", transform=ax.transAxes,
            fontsize=8.5, color=SOFT, va="top")
    panel_letter(ax, "d", x=-0.22, y=1.03)

    # e: controls
    ax = fig.add_subplot(gs[2, 2])
    groups = [("both\ndirections", BOTH), ("forward\nonly", FORWARD_ONLY), ("no order", CONTROL)]
    metrics = [("comp_heldout", "held-out events that express", "#7a8aa0"), ("comp_passes", "components passing the held-out test", CEN_C),
               ("comp_accepted", "splits accepted by the guard", "#1f8a65")]
    w = 0.26
    for gi, (gname, keys) in enumerate(groups):
        for mi, (m, mname, col) in enumerate(metrics):
            vals = np.concatenate([np.array(by[k]["centred"][m], float) for k in keys])
            v = np.nanmean(vals) if vals.size else np.nan
            x = gi + (mi - 1) * w
            ax.bar(x, v, w * 0.88, color=col, label=mname if gi == 0 else None)
            ax.text(x, (v if np.isfinite(v) else 0) + 0.015, f"{v:.0%}" if np.isfinite(v) else "–", ha="center", va="bottom",
                    fontsize=7.8, color=INK)
    ax.plot([-0.5, 2.5], [0.05, 0.05], color=INK, lw=0.9, ls=(0, (2, 2)), zorder=0)
    ax.text(2.5, 0.12, "5%: chance for\nheld-out events", fontsize=7.5, color=INK, ha="right", va="bottom", linespacing=1.15)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Controls: does the split invent order?", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8.2, bbox_to_anchor=(1.0, 1.02), handlelength=1.2)
    ax.text(0, -0.2, "centred, all kept components; held-out = the template is\nfitted on the other half of the events",
            transform=ax.transAxes, fontsize=8.5, color=SOFT, va="top")
    panel_letter(ax, "e", x=-0.13, y=1.03)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"direction_synthetic.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- hc-11
def candidates(r):
    """Every template of a session that matches the map: (name, labels, groups, margin over its null)."""
    out = []
    p = r["population"]
    if p and p["labels"] and p["labels"]["n_significant_expressing"]:
        s = p["labels"]["map"]
        out.append(("population", p["labels"], p.get("groups"), abs(p["map"][s]["rho"]) - p["map"][s]["null95"], True))
    for c in r["components"]:
        if c.get("labels") and c["labels"]["n_significant_expressing"]:
            s = c["labels"]["map"]
            m = c["map"][s]["template"]
            out.append((f"component {c['index'] + 1}", c["labels"], c.get("groups"), abs(m["rho"]) - m["null95"], False))
    return out


def best_template(r):
    c = candidates(r)
    return max(c, key=lambda x: x[3]) if c else None


def example_events(r, n_each=3):
    lab = r["population"]["labels"]
    ok = lab["event_expresses"] & lab["event_correct"]
    picks = []
    for want in (True, False):
        idx = np.flatnonzero(ok & (lab["event_forward"] == want))
        idx = idx[np.argsort(-np.abs(lab["event_rho"][idx]))][:n_each]
        picks += [(i, want) for i in idx]
    return picks


def raster(ax, r, i, forward):
    lab = r["population"]["labels"]
    cen = r["centres"][lab["map"]]
    times = lab["event_times_ms"][i]
    T = lab["event_duration_ms"][i]
    cells = np.flatnonzero(np.isfinite(times).any(axis=1) & np.isfinite(cen))
    order = cells[np.argsort(cen[cells])]
    lo, hi = np.nanmin(cen), np.nanmax(cen)
    for rank, c in enumerate(order):
        t = times[c][np.isfinite(times[c])]
        ax.scatter(t, np.full(t.size, rank), s=10, color=POS_CMAP((cen[c] - lo) / (hi - lo)), lw=0, zorder=2)
    ax.set_xlim(-0.03 * T, 1.03 * T)
    ax.set_ylim(-1, max(len(order), 1))
    ax.set_xticks([0, T])
    ax.set_xticklabels(["0", f"{T:.0f} ms"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    col = FWD_C if forward else BWD_C
    ax.set_title(f"labelled {'forward' if forward else 'backward'}", fontsize=9.5, color=col, loc="center", fontweight="bold", pad=17)
    ax.text(0.5, 1.02, f"ρ with the map {lab['event_rho'][i]:+.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2, color=SOFT)


def row_letter(fig, ax, letter, top=0.452):
    fig.text(ax.get_position().x0 - 0.035, top, letter, fontsize=14, fontweight="bold", va="bottom")


def session_axis(ax, names, labels=True):
    ypos = np.arange(len(names))[::-1]
    ax.set_yticks(ypos)
    ax.set_yticklabels([short(n) for n in names] if labels else [], fontsize=8.8)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_ylim(-0.6, len(names) - 0.4)
    return ypos


def figure_hc11(H):
    by = {(r["name"], r["epoch"]): r for r in H}
    names = [n for n in HC11_ORDER if (n, "post") in by]
    fig = plt.figure(figsize=(14.2, 9.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 1.0], hspace=0.6, wspace=0.5, left=0.085, right=0.985, top=0.835, bottom=0.115)
    fig.suptitle("hc-11: direction labels from complex PCA, checked against the waking map", x=0.035, y=0.99, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.035, 0.947, "templates fitted to population bursts without the map (one on all cells, and one per kept component); "
             "the waking place map is used only to score", fontsize=9.8, color=SOFT)

    # a: example rasters
    r = by[(EXAMPLE, "post")]
    sub = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0, :], wspace=0.16)
    for j, (i, fwd) in enumerate(example_events(r)):
        ax = fig.add_subplot(sub[0, j])
        raster(ax, r, i, fwd)
        if j == 0:
            ax.set_ylabel("cells, by place-field position", fontsize=9)
    fig.text(0.055, 0.893, f"Replays labelled by the whole-population template, {short(EXAMPLE)} POST "
             "(the strongest correct ones in each direction)", fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(0.035, 0.891, "a", fontsize=14, fontweight="bold", va="bottom")
    cax = fig.add_axes([0.875, 0.9, 0.11, 0.011])
    cax.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=POS_CMAP)
    cax.set_xticks([0, 99])
    cax.set_xticklabels(["track start", "end"], fontsize=8)
    cax.tick_params(length=0, pad=2)
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)

    epochs = (("post", 0.17, POST_C), ("pre", -0.17, PRE_C))
    epoch_legend = [Line2D([], [], marker="o", color=POST_C, ms=5, lw=0, label="POST"),
                    Line2D([], [], marker="o", color=PRE_C, ms=5, lw=0, label="PRE")]

    # b: held-out test on the whole-population template
    ax = fig.add_subplot(gs[1, 0])
    ypos = session_axis(ax, names)
    for epoch, dy, col in epochs:
        for y, n in zip(ypos, names):
            p = by.get((n, epoch), {}).get("population")
            if not p:
                continue
            sig = p["heldout_p"] < 0.05
            ax.plot(p["heldout_share"], y + dy, "o", ms=6.5, mfc=col if sig else "white", mec=col, mew=1.4)
    ax.axvline(0.05, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0, 0.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("held-out events expressing the template")
    ax.set_title("Is there a reproducible order?", pad=24)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=epoch_legend + [Line2D([], [], marker="o", ls="", mfc="white", mec=SOFT, mew=1.4, ms=6, label="p ≥ 0.05")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.3, borderaxespad=0.1, handletextpad=0.2,
              columnspacing=0.8)
    ax.text(0.052, -0.55, "chance", fontsize=7.8, color=SOFT, va="center")
    ax.text(0, -0.2, "whole-population template, fitted on half the events,\nscored on the other half (filled: p < 0.05)",
            transform=ax.transAxes, fontsize=8.3, color=SOFT, va="top")
    row_letter(fig, ax, "b")

    # c: labels of the best-matching template
    ax = fig.add_subplot(gs[1, 1])
    session_axis(ax, names, labels=False)
    for epoch, dy, col in epochs:
        for y, n in zip(ypos, names):
            b = best_template(by[(n, epoch)]) if (n, epoch) in by else None
            if b is None:
                ax.plot(0.5, y + dy, "x", ms=4, color=col, mew=1.0)
                continue
            lab = b[1]
            k, m = lab["correct_expressing"], lab["n_significant_expressing"]
            lo, hi = clopper(k, m)
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.6, solid_capstyle="round")
            ax.plot(k / m, y + dy, "D" if b[4] else "o", ms=6, color=col, mec="white", mew=0.6)
            ax.text(1.06, y + dy, f"{k}/{m}", fontsize=7.6, color=col, va="center", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0.5, color=SOFT, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("labels matching the map's direction")
    ax.set_title("Are the labels right?", pad=24)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=5, label="population"),
                       Line2D([], [], marker="o", ls="", color=INK, ms=5, label="a component"),
                       Line2D([], [], marker="x", ls="", color=INK, ms=5, label="none matches")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.3, borderaxespad=0.1, handletextpad=0.2,
              columnspacing=0.8)
    ax.text(0, -0.2, "template that best matches the map; its expressing\nevents that are significant replays (95% interval)",
            transform=ax.transAxes, fontsize=8.3, color=SOFT, va="top")
    row_letter(fig, ax, "c")

    # d: forward and backward groups have opposite signs
    ax = fig.add_subplot(gs[1, 2])
    lim = 0.85
    ax.fill_between([0, lim], [-lim, -lim], [0, 0], color=FWD_C, alpha=0.06, lw=0)
    ax.axhline(0, color=SOFT, lw=0.7)
    ax.axvline(0, color=SOFT, lw=0.7)
    ax.text(0.04, -lim + 0.04, "one route, played both ways", ha="left", va="bottom", fontsize=8.0, color=FWD_C)
    for epoch, col, face in (("post", POST_C, POST_C), ("pre", PRE_C, "white")):
        for n in names:
            if (n, epoch) not in by:
                continue
            for name, lab, groups, _, is_pop in candidates(by[(n, epoch)]):
                if not groups:
                    continue
                g, s = groups[lab["map"]], np.sign(lab["template_rho"])
                if is_pop:
                    ax.plot(g["forward"]["rho"] * s, g["backward"]["rho"] * s, "D", ms=7.5, mfc=face, mec=col, mew=1.5, zorder=3)
                else:
                    ax.plot(g["forward"]["rho"] * s, g["backward"]["rho"] * s, "o", ms=4.5, mfc=face, mec=col, mew=1.0, alpha=0.8, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    box = ax.get_position()
    W, Hh = fig.get_size_inches()
    side = min(box.width * W, box.height * Hh)
    ax.set_position([box.x0, box.y1 - side / Hh, side / W, side / Hh])
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel("forward events' template vs map, ρ")
    ax.set_ylabel("backward events' template vs map, ρ")
    ax.set_title("Two groups, opposite ways?", pad=24)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=INK, ms=5.5, label="population"),
                       Line2D([], [], marker="o", ls="", color=INK, ms=4.5, label="component"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=PRE_C, mew=1.2, ms=5, label="PRE (open)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.3, borderaxespad=0.1, handletextpad=0.2,
              columnspacing=0.8)
    ax.text(0, -0.2, "every template that matches the map; sign set so\nthat the template's own ρ is positive",
            transform=ax.transAxes, fontsize=8.3, color=SOFT, va="top")
    row_letter(fig, ax, "d")

    # e: the expressing events are enriched for replays
    ax = fig.add_subplot(gs[1, 3])
    for epoch, _, col in epochs:
        for n in names:
            b = best_template(by[(n, epoch)]) if (n, epoch) in by else None
            if b is None or not b[1]["n_other_scored"]:
                continue
            lab = b[1]
            e = lab["n_significant_expressing"] / max(lab["n_expressing"], 1)
            o = (lab["n_significant"] - lab["n_significant_expressing"]) / lab["n_other_scored"]
            ax.plot([0, 1], [o, e], "-", color=col, lw=1.2, alpha=0.9)
            ax.plot([0, 1], [o, e], "o", color=col, ms=5, mec="white", mew=0.6)
            ax.text(1.06, e, short(n), fontsize=7.6, color=col, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other\nevents", "expressing\nevents"])
    ax.set_xlim(-0.25, 1.75)
    ax.set_ylim(0, None)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("significant replays of the map")
    ax.set_title("Does the gate pick out replays?", pad=24)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=epoch_legend, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.3, borderaxespad=0.1,
              handletextpad=0.2)
    ax.text(0, -0.2, "template of c; up to 300 other events scored", transform=ax.transAxes, fontsize=8.3, color=SOFT, va="top")
    row_letter(fig, ax, "e")

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"direction_hc11.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def summary(S, H):
    by = synthetic_table(S)
    labels = {cond_key(c): cond_label(c) for c in synthetic_conditions()}
    out = ["# Forward and backward plays: results", "",
           "Produced by `scripts/summarize_direction.py` from `results/direction/` (`scripts/run_direction.py`).", "",
           "## Planted data (10 fresh draws per condition)", "",
           "Order recovered per planted route (|Spearman ρ| of the best component, median over draws and routes), "
           "before and after the split; direction accuracy and reverse share for the best component after the split.", "",
           "| condition | centred before | centred after | uncentred before | uncentred after | accuracy (centred) | reverse share planted / estimated | held-out events expressing | components passing held-out test | splits accepted |",
           "|---|---|---|---|---|---|---|---|---|---|"]

    def med(v):
        v = np.array(v, float)
        return f"{np.nanmedian(v):.2f}" if np.isfinite(v).any() else "–"

    for k in BOTH + FORWARD_ONLY + CONTROL:
        c, u = by[k]["centred"], by[k]["uncentred"]
        tail = (f"{np.mean(c['comp_heldout']):.1%} | {np.mean(c['comp_passes']):.0%} | {np.mean(c['comp_accepted']):.0%}"
                if c["comp_accepted"] else "no component | – | –")
        out.append(f"| {labels[k]} | {med(c['before'])} | {med(c['after'])} | {med(u['before'])} | {med(u['after'])} | {med(c['accuracy'])} | "
                   f"{med(c['true_minority'])} / {med(c['minority'])} | {tail} |")
    out += ["", "## hc-11: one template on the whole population", "",
            "Map ρ is the template's order against each running direction's place map (null 95th percentile in brackets). "
            "Labels: expressing events that are significant replays of the template's map, labelled in the map's direction. "
            "Groups: templates of the forward and the backward events against that map.", "",
            "| session | epoch | events | expressing | reverse share | map ρ (+) | map ρ (−) | labels correct | groups ρ fwd / bwd | held-out expressing (p) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(lambda: [0, 0])
    for r in sorted(H, key=lambda r: (HC11_ORDER.index(r["name"]), r["epoch"] != "post")):
        p = r["population"]
        if p is None:
            out.append(f"| {short(r['name'])} | {r['epoch'].upper()} | {r['n_events']} | – | – | – | – | – | – | – |")
            continue
        m = p["map"]
        lab = p["labels"]
        labtxt, gtxt = "no map match", "–"
        if lab:
            labtxt = f"{lab['correct_expressing']}/{lab['n_significant_expressing']}"
            tot[r["epoch"]][0] += lab["correct_expressing"]
            tot[r["epoch"]][1] += lab["n_significant_expressing"]
            if p.get("groups"):
                g = p["groups"][lab["map"]]
                gtxt = f"{g['forward']['rho']:+.2f} / {g['backward']['rho']:+.2f}"
        out.append(f"| {short(r['name'])} | {r['epoch'].upper()} | {p['n_events']} | {p['n_expressing']} | {p['minority_fraction']:.2f} | "
                   f"{m[1.0]['rho']:+.2f} ({m[1.0]['null95']:.2f}) | {m[-1.0]['rho']:+.2f} ({m[-1.0]['null95']:.2f}) | {labtxt} | {gtxt} | "
                   f"{p['heldout_share']:.1%} ({p['heldout_p']:.0e}) |")
    out += [""]
    for e in ("post", "pre"):
        k, n = tot[e]
        if n:
            out.append(f"- {e.upper()}: {k}/{n} labels correct ({k / n:.1%}; binomial p against 50%: {binomtest(k, n, 0.5, alternative='greater').pvalue:.1e}).")
    out += ["", "## hc-11: the split of each component", "",
            "| session | epoch | kept | pass held-out test | split accepted | map hits before | map hits after | template hits (unguarded) | template labels correct |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(H, key=lambda r: (HC11_ORDER.index(r["name"]), r["epoch"] != "post")):
        passes = sum(1 for c in r["components"] if c["split"] and c["split"]["heldout_p"] < 0.05)
        acc = sum(1 for c in r["components"] if c["split"] and c["split"]["accepted"])
        hb = sum(1 for c in r["components"] if any(hit(c["map"][s]["before"]) for s in c["map"]))
        ha = sum(1 for c in r["components"] if any(hit(c["map"][s].get("after", c["map"][s]["before"])) for s in c["map"]))
        ht = sum(1 for c in r["components"] if any(hit(c["map"][s].get("template")) for s in c["map"]))
        k = sum(c["labels"]["correct_expressing"] for c in r["components"] if c.get("labels"))
        n = sum(c["labels"]["n_significant_expressing"] for c in r["components"] if c.get("labels"))
        out.append(f"| {short(r['name'])} | {r['epoch'].upper()} | {r['n_kept']} | {passes} | {acc} | {hb} | {ha} | {ht} | {k}/{n} |")
    (RES / "summary.md").write_text("\n".join(out) + "\n")


def main():
    S, H = load("synthetic"), load("hc11")
    print(f"{len(S)} synthetic, {len(H)} hc-11 results")
    if S:
        figure_synthetic(S)
    if H:
        figure_hc11(H)
    if S and H:
        summary(S, H)


if __name__ == "__main__":
    main()
