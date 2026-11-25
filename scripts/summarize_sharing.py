"""Figures and table for sequences that share cells (results of scripts/run_sharing.py).

    figures/sharing_synthetic.png / .pdf   what PCA returns for two maps on the same cells, and what the templates return
    figures/sharing_ladder.png / .pdf      the difficulty ladder: complex PCA with Step 3 against PP-Seq and seqNMF on the same events
    figures/sharing_hc11.png / .pdf        hc-11, POST and PRE
    results/sharing/summary.md
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
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_direction import (BEFORE_C, CEN_C, GRID, INK, POS_CMAP, POST_C, PRE_C, SHORT, SOFT, UNC_C,  # noqa: E402
                                         panel_letter, short)
from scripts.run_sharing import HC11_ORDER, MATCH  # noqa: E402

RES = ROOT / "results" / "sharing"
LADDER_RES = ROOT / "results" / "ladder"
FIG = ROOT / "figures"
EXAMPLE = ("L2 two maps on the same cells, both directions", 0)     # fixed in advance: first fresh draw of L2
HC11_EXAMPLE = "Achilles_10252013"
PP_C, NMF_C, S3_C = "#b8901a", "#d4652f", "#1f4e79"
FALSE_C = "#c9a3a3"

SHARED = ["E", "L1", "L2", "L4", "L5", "L5s"]
NOT_SHARED = ["A", "B", "C", "D", "F", "L0"]
PARTIAL = ["G", "H", "L3"]
CONTROL = ["S0"]
LABEL = dict(SHORT, L5s="L2 + variable pacing, per-event scale", L1="two maps, same cells, forward only", E="two routes sharing half their cells")
RUNGS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
RUNG_LABEL = {"L0": "one map, both ways", "L1": "two maps, same cells, forward", "L2": "two maps, same cells, both ways",
              "L3": "L2 + partial replays", "L4": "L2 + assemblies", "L5": "L2 + variable pacing", "L6": "L2 + partial + assemblies (= H)"}


def load(family):
    return [pickle.load(open(p, "rb")) for p in sorted((RES / family).glob("*.pkl"))]


def med(v):
    v = np.array(v, float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


# ----------------------------------------------------------------------------- tables
def synthetic_table(S):
    """key (condition, or L5s for L5 with the per-event scale) -> convention -> lists."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in S:
        k = r["condition"].split(" ")[0]
        for v, x in r["variants"].items():
            key, conv = ("L5s", "centred") if v == "centred, per-event scale" else (k, v)
            d = by[key][conv]
            for a, b in zip(x["step2_maps"], x["step3_maps"]):
                d["step2"].append(a[0])
                d["step3"].append(b[0])
                d["coverage"].append(b[1])
            d["n_valid"].append(x["step3"]["n_valid"] if x["step3"] else 0)
            d["n_false"].append(x["n_false"])
            d["n_maps"].append(r["n_maps"])
            d["runtime"].append(x["runtime_step3_s"])
            for e in x["events"]:
                if e:
                    d["recall"].append(e["recall"])
                    d["precision"].append(e["precision"])
                    d["direction"].append(e.get("direction_accuracy", np.nan))
    return by


def ladder_table(L):
    """rung -> method -> list of per-map (rho, p); plus run times."""
    out = defaultdict(lambda: defaultdict(list))
    rt = defaultdict(list)
    for r in L:
        k = r["rung"].split(" ")[0]
        for conv, x in r["variants"].items():
            n_maps = len(x["step2"])
            for m in x["step2"]:
                out[k][f"step2 {conv}"].append((m["rho"], m["p"]))
            for i in range(n_maps):
                m = x["step3"][i] if x["step3"] else {"rho": np.nan, "p": 1.0}
                out[k][f"step3 {conv}"].append((m["rho"], m["p"]))
            rt[f"step3 {conv}"].append(x["runtime_pca_s"] + x["runtime_step3_s"])
    for method, variant in (("cpca", "centred"), ("cpca", "uncentred"), ("ppseq", "type_rule"), ("seqnmf", "K15_L120")):
        for p in sorted(LADDER_RES.glob(f"*.{method}.pkl")):
            r = pickle.load(open(p, "rb"))
            k = r["rung"].split(" ")[0]
            name = f"components {variant}" if method == "cpca" else method
            for m in r["variants"][variant]["maps"]:
                out[k][name].append((m["rho"], m["p"]))
            if method != "cpca" or variant == "centred":
                rt[method].append(r["runtime_s"] if method != "cpca" else r["variants"]["centred"]["runtime_s"])
    return out, rt


# ----------------------------------------------------------------------------- figure 1
def scatter_panel(ax, order, cen, title=None, col=INK, show_x=False, show_y=False):
    m = np.isfinite(order) & np.isfinite(cen)
    if m.sum() >= 3:
        from scipy.stats import rankdata, spearmanr
        y = rankdata(order[m]) / m.sum()
        rho = spearmanr(order[m], cen[m]).statistic
        ax.scatter(cen[m], y, s=7, color=POS_CMAP(cen[m]), lw=0)
        ax.text(0.03, 0.97, f"|ρ| {abs(rho):.2f}", transform=ax.transAxes, fontsize=8.5, va="top", color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=5)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xticks([0, 1] if show_x else [])
    ax.set_yticks([])
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, loc="center", fontweight="bold", color=col, pad=4)


def example_panel(fig, spec):
    from cpca.direction import order_of
    from scripts.run_sharing import example_fit
    data, res, s2, sq = example_fit(*EXAMPLE)
    K = res["n_kept"]
    comps = [order_of(res["pca_components"][:, j]) for j in range(K)]
    valid = np.flatnonzero(sq["check"]["keep"])
    temps = [order_of(sq["templates"][:, j]) for j in valid]
    ncol = K + len(temps)
    sub = GridSpecFromSubplotSpec(2, ncol + 1, subplot_spec=spec, wspace=0.18, hspace=0.18,
                                  width_ratios=[1] * K + [0.35] + [1] * len(temps))
    for row, cen in enumerate(data["centres"]):
        for j, o in enumerate(comps):
            ax = fig.add_subplot(sub[row, j])
            scatter_panel(ax, o, cen, title=f"component {j + 1}" if row == 0 else None, col=BEFORE_C if row == 0 else INK,
                          show_x=row == 1)
            if j == 0:
                ax.set_ylabel(f"map {'AB'[row]}\nread-out rank", fontsize=8.5)
        for i, o in enumerate(temps):
            ax = fig.add_subplot(sub[row, K + 1 + i])
            scatter_panel(ax, o, cen, title=f"template {i + 1}" if row == 0 else None, col=S3_C, show_x=row == 1)
    fig.text(sub[1, 0].get_position(fig).x0, sub[1, 0].get_position(fig).y0 - 0.032, "x: planted position on the map; y: rank of the "
             "cell's read-out time; one dot per member cell", fontsize=8.5, color=SOFT)
    return data, res


def recovery_rows():
    rows = [(None, "two orders on the same cells")] + [(k, LABEL[k]) for k in SHARED]
    rows += [(None, "no shared cells")] + [(k, LABEL[k]) for k in NOT_SHARED]
    rows += [(None, "partial replays")] + [(k, LABEL[k]) for k in PARTIAL]
    rows += [(None, "no order")] + [(k, LABEL[k]) for k in CONTROL]
    return rows


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
        d = by[k].get(conv)
        if d is None:
            ax.text(0.02, yi, "centred only", va="center", fontsize=8.3, color=SOFT)
            continue
        b, a = np.array(d["step2"], float), np.array(d["step3"], float)
        if not np.isfinite(b).any():
            ax.text(0.02, yi, "no component kept: centring removes a route that is in every event", va="center", fontsize=8.3, color=SOFT)
            continue
        mb = np.nanmedian(b)
        ax.plot(mb, yi, "o", ms=7, mfc="white", mec=BEFORE_C, mew=1.6, zorder=3)
        if not np.isfinite(a).any():
            ax.text(mb + 0.04, yi, f"no template validated ({len(a)}/{len(a)} draws)", va="center", fontsize=8.3, color=SOFT)
            continue
        ma = np.nanmedian(np.nan_to_num(a, nan=0.0))
        ax.plot(np.nanpercentile(np.nan_to_num(a, nan=0.0), [25, 75]), [yi, yi], color=col, lw=6, alpha=0.18, solid_capstyle="round", zorder=1)
        ax.plot([mb, ma], [yi, yi], color=SOFT, lw=1.0, zorder=2)
        ax.plot(ma, yi, "o", ms=7, color=col, zorder=4)
        if k in PARTIAL and conv == "centred":
            ax.text(0.02, yi, f"templates cover {med(d['coverage']):.0%} of the map", va="center", fontsize=8, color=SOFT)
    ticks = [(yi, k, lab) for yi, (k, lab) in zip(y, rows) if k is not None]
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([f"{'L5' if k == 'L5s' else k}   {lab}" for _, k, lab in ticks] if show_labels else [], fontsize=9)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("order recovered, |Spearman ρ| with the planted map")


def figure_synthetic(S):
    by = synthetic_table(S)
    rows = recovery_rows()
    fig = plt.figure(figsize=(14.2, 15.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 1.25, 0.8], hspace=0.5, wspace=0.34, left=0.05, right=0.975, top=0.905, bottom=0.06)
    fig.suptitle("Two sequences on the same cells: one template each, every event assigned to one of them", x=0.035, y=0.985, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.035, 0.96, "planted data: 10 fresh draws per condition, 80 cells, 3000 events; complex PCA, then templates started from "
             "its components and fitted by assigning events (Step 3), kept only if they hold up on held-out events", fontsize=9.8, color=SOFT)

    # a: example
    example_panel(fig, gs[0, :])
    top = gs[0, :].get_position(fig).y1
    fig.text(0.055, top + 0.012, "Two maps on the same 60 cells (L2, first fresh draw): PCA returns their sum and differences, "
             "the templates return the maps", fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(0.035, top + 0.01, "a", fontsize=14, fontweight="bold", va="bottom")

    # b: recovery
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.05)
    ax0 = fig.add_subplot(sub[0, 0])
    pos = ax0.get_position()
    ax0.set_position([0.27, pos.y0, 0.33, pos.height])
    ax1 = fig.add_axes([0.63, pos.y0, 0.33, pos.height])
    for i, (axr, conv) in enumerate(zip((ax0, ax1), ("centred", "uncentred"))):
        recovery_panel(axr, by, conv, rows, show_labels=i == 0)
        axr.set_title(f"{conv} complex PCA", fontsize=10.5, loc="center", fontweight="normal", color=SOFT)
    fig.text(0.055, pos.y1 + 0.03, "Order recovered for each planted map: Step 2 (direction split), then Step 3 (validated templates)",
             fontsize=11.5, fontweight="bold", va="bottom")
    fig.text(0.035, pos.y1 + 0.028, "b", fontsize=14, fontweight="bold", va="bottom")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=BEFORE_C, mew=1.6, ms=7, label="Step 2: best component order, median"),
                        Line2D([], [], marker="o", ls="", color=CEN_C, ms=7, label="Step 3: best validated template, median"),
                        Line2D([], [], color=CEN_C, lw=6, alpha=0.18, label="Step 3: middle half of maps and draws")],
               loc="upper center", bbox_to_anchor=(0.615, pos.y0 - 0.032), ncol=3, frameon=False, fontsize=8.8, handletextpad=0.4,
               columnspacing=1.4)

    keys = SHARED + NOT_SHARED + PARTIAL + CONTROL
    xs = np.arange(len(keys))
    names = ["L5\n(own\nscale)" if k == "L5s" else k for k in keys]
    bounds = np.cumsum([len(SHARED), len(NOT_SHARED), len(PARTIAL)]) - 0.5

    # c: templates against planted maps
    ax = fig.add_subplot(gs[2, :2])
    for i, k in enumerate(keys):
        d = by[k]["centred"]
        valid, false, maps = np.mean(d["n_valid"]), np.mean(d["n_false"]), np.mean(d["n_maps"])
        ax.bar(i, valid - false, 0.62, color=CEN_C)
        ax.bar(i, false, 0.62, bottom=valid - false, color=FALSE_C)
        ax.plot([i - 0.38, i + 0.38], [maps, maps], color=INK, lw=1.6)
    for b in bounds:
        ax.axvline(b, color=GRID, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8.8)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel("templates per fit (mean of 10 draws)")
    ax.set_title("How many sequences are found", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CEN_C, label=f"validated, matches a planted map (|ρ| ≥ {MATCH})"),
                       plt.Rectangle((0, 0), 1, 1, color=FALSE_C, label="validated, matches no planted map"),
                       Line2D([], [], color=INK, lw=1.6, label="planted maps")], loc="upper left", frameon=False, fontsize=8.5)
    ax.text(0, -0.2, "centred; groups as in b. With partial replays one map is found as several segments, and a few templates mix "
            "segments of the two maps", transform=ax.transAxes, fontsize=8.5, color=SOFT, va="top")
    panel_letter(ax, "c", x=-0.07, y=1.03)

    # d: event assignment
    ax = fig.add_subplot(gs[2, 2])
    ev_keys = [k for k in SHARED + NOT_SHARED if k != "L5s"]
    for i, k in enumerate(ev_keys):
        d = by[k]["centred"]
        for m, dx, col, mk in (("recall", -0.15, CEN_C, "o"), ("precision", 0.15, PP_C, "s")):
            ax.plot(i + dx, med(d[m]), mk, color=col, ms=6.5, mec="white", mew=0.6)
    ax.set_xticks(range(len(ev_keys)))
    ax.set_xticklabels(ev_keys, fontsize=8.8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Which events go to which template", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CEN_C, ms=6, label="map's replays assigned to its template"),
                       Line2D([], [], marker="s", ls="", color=PP_C, ms=6, label="template's events that are the map's replays")],
              loc="lower left", frameon=False, fontsize=8.2)
    ax.text(0, -0.2, "centred, medians; the rest of a template's events are\nvolleys that pass the gate; direction labels: 100% correct",
            transform=ax.transAxes, fontsize=8.5, color=SOFT, va="top")
    panel_letter(ax, "d", x=-0.13, y=1.03)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"sharing_synthetic.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 2
LADDER_METHODS = [("components centred", "complex PCA components", BEFORE_C, "o", "white"),
                  ("step2 centred", "+ direction split (Step 2)", "#7a8aa0", "o", None),
                  ("step3 centred", "+ templates (Step 3)", S3_C, "D", None),
                  ("ppseq", "PP-Seq", PP_C, "s", None),
                  ("seqnmf", "seqNMF (K = 15, L = 120 ms)", NMF_C, "^", None)]


def figure_ladder(L):
    tab, rt = ladder_table(L)
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.3, left=0.17, right=0.985, top=0.79, bottom=0.2)
    fig.suptitle("The difficulty ladder: with Step 3, complex PCA matches PP-Seq on whole routes and leads on partial replays",
                 x=0.035, y=0.975, ha="left", fontsize=14.5, fontweight="bold")
    fig.text(0.035, 0.925, "the ladder's own events (1500 per draw, 3 draws per rung, centred complex PCA); every method scored the same "
             "way: best output per planted map, against a null that takes the best of the method's outputs", fontsize=9.6, color=SOFT)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(RUNGS))[::-1]
    offs = np.linspace(0.28, -0.28, len(LADDER_METHODS))
    rng = np.random.default_rng(0)
    for (key, label, col, mk, face), dy in zip(LADDER_METHODS, offs):
        for yi, k in zip(y, RUNGS):
            v = np.array(tab[k][key], float)
            if v.size == 0:
                continue
            rho = np.nan_to_num(v[:, 0], nan=0.0)
            ax.scatter(rho, yi + dy + rng.uniform(-0.025, 0.025, rho.size), s=9, color=col, alpha=0.35, lw=0, zorder=2)
            ax.plot(np.median(rho), yi + dy, mk, ms=7.5, color=col, mfc=face or col, mec=col, mew=1.4, zorder=3)
            ax.text(1.03, yi + dy, f"{int(np.sum(v[:, 1] < 0.05))}/{len(v)}", fontsize=7.4, color=col, va="center",
                    transform=ax.get_yaxis_transform())
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=GRID, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}   {RUNG_LABEL[k]}" for k in RUNGS], fontsize=9.2)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_ylim(-0.6, len(RUNGS) - 0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("order recovered per planted map, |Spearman ρ| (large: median; small: each map and draw)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(1.03, len(RUNGS) - 0.38, "maps\nfound", transform=ax.get_yaxis_transform(), fontsize=7.6, color=SOFT, va="bottom")
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", ms=7, color=col, mfc=face or col, mec=col, mew=1.4, label=label)
                       for _, label, col, mk, face in LADDER_METHODS], loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=5, frameon=False,
              fontsize=8.6, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
    panel_letter(ax, "a", x=-0.36, y=1.08)

    ax = fig.add_subplot(gs[0, 1])
    items = [("cpca", "complex\nPCA", BEFORE_C), ("step3 centred", "PCA +\nStep 3", S3_C),
             ("seqnmf", "seqNMF\n3 × 3 grid", NMF_C), ("ppseq", "PP-Seq\nsweep +\nrefit", PP_C)]
    for i, (key, label, col) in enumerate(items):
        v = np.array(rt[key], float)
        ax.scatter(np.full(v.size, i) + rng.uniform(-0.12, 0.12, v.size), v, s=14, color=col, alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=col, lw=2.2)
        m = np.median(v)
        txt = f"{m:.0f} s" if m < 120 else f"{m / 60:.0f} min"
        ax.text(i + 0.3, m, txt, ha="left", va="center", fontsize=8.3, color=col, fontweight="bold")
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 60, 600, 3600, 36000])
    ax.set_yticklabels(["1 s", "10 s", "1 min", "10 min", "1 h", "10 h"])
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([lab for _, lab, _ in items], fontsize=8.6)
    ax.set_xlim(-0.5, len(items) - 0.05)
    ax.set_title("Run time per fit", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0, -0.21, "centred fits, one laptop core each; seqNMF and\nPP-Seq include their settings searches", transform=ax.transAxes,
            fontsize=8.3, color=SOFT, va="top")
    panel_letter(ax, "b", x=-0.22, y=1.08)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"sharing_ladder.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 3
def hit(m):
    return np.isfinite(m["rho"]) and abs(m["rho"]) > m["null95"]


def figure_hc11(H):
    by = {(r["name"], r["epoch"]): r for r in H}
    names = [n for n in HC11_ORDER if (n, "post") in by]
    fig = plt.figure(figsize=(14.2, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42, left=0.09, right=0.985, top=0.76, bottom=0.17)
    fig.suptitle("hc-11: few templates hold up on held-out sleep events", x=0.035, y=0.975, ha="left", fontsize=14.5, fontweight="bold")
    fig.text(0.035, 0.912, "centred complex PCA on all population bursts, then Step 3; the waking place map is used only to score",
             fontsize=9.6, color=SOFT)

    ax = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(names))[::-1]
    for epoch, dy, col in (("post", 0.17, POST_C), ("pre", -0.17, PRE_C)):
        for y, n in zip(ypos, names):
            r = by.get((n, epoch))
            if r is None:
                continue
            T = r["templates"]
            n_t, n_v = len(T), sum(t["valid"] for t in T)
            n_h = sum(t["valid"] and any(hit(m) for m in t["map"].values()) for t in T)
            ax.barh(y + dy, n_t, 0.3, color="white", edgecolor=col, lw=1.0)
            ax.barh(y + dy, n_v, 0.3, color=col, alpha=0.45, lw=0)
            ax.barh(y + dy, n_h, 0.3, color=col, lw=0)
            ax.text(n_t + 0.15, y + dy, f"{n_v} valid, {n_h} on the map" if n_v else "none valid", fontsize=7.6, color=col, va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels([short(n) for n in names], fontsize=8.8)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0, 13.5)
    ax.set_xlabel("templates per session")
    ax.set_title("Templates fitted, validated, matching the map", pad=22)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, label="fitted"),
                       plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.45, label="validated"),
                       plt.Rectangle((0, 0), 1, 1, color=INK, label="validated and on the map"),
                       Line2D([], [], marker="s", ls="", color=POST_C, label="POST"), Line2D([], [], marker="s", ls="", color=PRE_C, label="PRE")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=7.8, handletextpad=0.3, columnspacing=0.8,
              borderaxespad=0.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", x=-0.3, y=1.1)

    # b, c: the validated template of the example session against the two direction maps
    r = by[(HC11_EXAMPLE, "post")]
    t = next(t for t in r["templates"] if t["valid"])
    cen = r["centres"]
    for i, (s, lab) in enumerate(((1.0, "running direction +"), (-1.0, "running direction −"))):
        ax = fig.add_subplot(gs[0, 1 + i])
        o = t["order"]
        m = np.isfinite(o) & np.isfinite(cen[s])
        from scipy.stats import rankdata
        ax.scatter(cen[s][m], rankdata(o[m]) / m.sum(), s=16, color=POST_C, alpha=0.8, lw=0)
        mm = t["map"][s]
        ax.set_title(f"Its order vs the map, {lab}", pad=10, fontsize=10.5)
        ax.text(0.03, 0.97, f"ρ {mm['rho']:+.2f}  (null 95%: {mm['null95']:.2f})\n{int(m.sum())} cells with a field", transform=ax.transAxes,
                fontsize=8.5, va="top", color=INK)
        ax.set_xlabel("place-field centre on the track (m)")
        if i == 0:
            ax.set_ylabel("rank of the cell's read-out time")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        panel_letter(ax, "bc"[i], x=-0.2, y=1.05)
    lab = t["labels"]
    fig.text(0.6, 0.035, f"{short(HC11_EXAMPLE)} POST, the one validated template ({t['n_events']} events, {t['n_members']} cells): its direction "
             f"labels match the map on {lab['correct_expressing']} of {lab['n_significant_expressing']} significant replays", fontsize=9,
             color=SOFT, ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"sharing_hc11.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- summary
def summary(S, L, H):
    by = synthetic_table(S)
    tab, rt = ladder_table(L)
    out = ["# Sequences that share cells: results", "",
           "Produced by `scripts/summarize_sharing.py` from `results/sharing/` (`scripts/run_sharing.py`). Acceptance criteria: `docs/sharing.md`.", "",
           "## Planted data (10 fresh draws per condition, 3000 events)", "",
           "Per planted map, median over maps and draws: |Spearman ρ| of the best Step 2 order (direction split of each component) "
           "and of the best validated Step 3 template, with the template's coverage of the map; templates per fit (mean) against planted maps; "
           "for the best template per map, the share of the map's replays assigned to it (recall), the share of its events that are the map's "
           "replays (precision), and the direction accuracy of those events.", "",
           "| condition | convention | Step 2 | Step 3 | coverage | validated / planted | matching no map (total) | recall | precision | direction | Step 3 time |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in SHARED + NOT_SHARED + PARTIAL + CONTROL:
        for conv in ("centred", "uncentred"):
            d = by[k].get(conv)
            if d is None:
                continue
            f = lambda v: f"{med(v):.2f}" if np.isfinite(med(v)) else "–"
            out.append(f"| {'L5 (own scale)' if k == 'L5s' else k} {LABEL[k]} | {conv} | {f(d['step2'])} | {f(d['step3'])} | {f(d['coverage'])} | "
                       f"{np.mean(d['n_valid']):.1f} / {np.mean(d['n_maps']):.0f} | {int(np.sum(d['n_false']))} | {f(d['recall'])} | {f(d['precision'])} | "
                       f"{f(d['direction'])} | {med(d['runtime']):.1f} s |")
    out += ["", "## Difficulty ladder (the ladder's own events; 1500 per draw, 3 draws)", "",
            "Median |ρ| per planted map and maps recovered (p < 0.05 against the best-of-outputs null).", "",
            "| rung | components (centred) | + Step 2 | + Step 3 (centred) | + Step 3 (uncentred) | PP-Seq | seqNMF K = 15, L = 120 ms |",
            "|---|---|---|---|---|---|---|"]
    for k in RUNGS:
        cells = []
        for key in ("components centred", "step2 centred", "step3 centred", "step3 uncentred", "ppseq", "seqnmf"):
            v = np.array(tab[k][key], float)
            cells.append(f"{np.median(np.nan_to_num(v[:, 0], nan=0.0)):.2f} ({int(np.sum(v[:, 1] < 0.05))}/{len(v)})" if v.size else "–")
        out.append(f"| {k} {RUNG_LABEL[k]} | " + " | ".join(cells) + " |")
    out += ["", "Run time per fit (median): " + ", ".join(f"{k} {np.median(v):.1f} s" for k, v in rt.items()) + ".", "",
            "## hc-11 (centred, all population bursts)", "",
            "| session | epoch | components | templates | validated | validated and matching a map | map ρ (+ / −) of validated templates | labels correct |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(H, key=lambda r: (HC11_ORDER.index(r["name"]), r["epoch"] != "post")):
        T = r["templates"]
        valid = [t for t in T if t["valid"]]
        hits = [t for t in valid if any(hit(m) for m in t["map"].values())]
        rhos = "; ".join(f"{t['map'][1.0]['rho']:+.2f} ({t['map'][1.0]['null95']:.2f}) / {t['map'][-1.0]['rho']:+.2f} ({t['map'][-1.0]['null95']:.2f})"
                         for t in valid) or "–"
        labs = "; ".join(f"{t['labels']['correct_expressing']}/{t['labels']['n_significant_expressing']}" for t in valid if t["labels"]) or "–"
        out.append(f"| {short(r['name'])} | {r['epoch'].upper()} | {r['n_kept']} | {len(T)} | {len(valid)} | {len(hits)} | {rhos} | {labs} |")
    comp_hits = {e: sum(any(hit(m) for m in c.values()) for r in H if r["epoch"] == e for c in r["components"]) for e in ("post", "pre")}
    comp_n = {e: sum(len(r["components"]) for r in H if r["epoch"] == e) for e in ("post", "pre")}
    out += ["", f"Components matching a map (either direction), for reference: POST {comp_hits['post']}/{comp_n['post']}, PRE {comp_hits['pre']}/{comp_n['pre']}."]
    (RES / "summary.md").write_text("\n".join(out) + "\n")


def main():
    S, L, H = load("synthetic"), load("ladder"), load("hc11")
    print(f"{len(S)} synthetic, {len(L)} ladder, {len(H)} hc-11 results")
    figure_synthetic(S)
    figure_ladder(L)
    figure_hc11(H)
    summary(S, L, H)


if __name__ == "__main__":
    main()
