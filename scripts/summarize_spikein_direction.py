"""Tables and figure for check C (docs/segments.md, addendum): forward-only against balanced planted replays on real
hc-11 events, read out three ways, with the direction oracles.

Usage: python scripts/summarize_spikein_direction.py
Output: results/spikein_direction/summary.md, figures/spikein_direction.png/.pdf
"""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_spikein_direction import FRACTIONS, HC11_ORDER, READOUTS  # noqa: E402

OUT = ROOT / "results" / "spikein_direction"
FIG = ROOT / "figures"
LARGE = ["Achilles_10252013", "Achilles_11012013"]
LABELS = {"standard": "standard", "pairs_adjacent": "adjacent pairs", "pairs_all": "all pairs", "oracle_uncentred": "oracle, uncentred", "oracle_centred": "oracle, centred"}


def load():
    return {n: pickle.load(open(OUT / f"{n}.pkl", "rb")) for n in HC11_ORDER if (OUT / f"{n}.pkl").exists()}


def count(rows, key):
    return sum(np.isfinite(r[key]["p"]) and r[key]["p"] < 0.05 for r in rows)


def threshold(R, names, kind, key, need=4):
    """Lowest share at which at least `need` of 5 draws pass in every named session (None if never)."""
    for f in FRACTIONS[1:]:
        ok = all(count([r for r in R[n]["rows"] if r["kind"] == kind and r["f"] == f], key) >= need for n in names)
        if ok:
            return f
    return None


def summary(R):
    lines = ["# Forward-only against balanced planted replays on real hc-11 events", "",
             "Produced by `scripts/summarize_spikein_direction.py` from `scripts/run_spikein_direction.py`; design in `docs/segments.md` (addendum). "
             "Matched 1500 PRE events per session; ideal replays of the waking map planted at share f with the published seeds (balanced reproduces "
             "`results/spikein/` event for event) or all forward. Entries: draws of 5 whose best output matches the map (p < 0.05).", ""]
    for kind in ("balanced", "forward"):
        lines += [f"## {kind.capitalize()} replays", "",
                  "| session | read-out | " + " | ".join(f"{f:.0%}" for f in FRACTIONS if f > 0 or kind == "balanced") + " |",
                  "|---|---|" + "---|" * len([f for f in FRACTIONS if f > 0 or kind == "balanced"])]
        for n, r in R.items():
            for key in READOUTS:
                cells = [f"{count([x for x in r['rows'] if x['kind'] == kind and x['f'] == f], key)}/5" for f in FRACTIONS if f > 0 or kind == "balanced"]
                lines.append(f"| {n} | {LABELS[key]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Pooled over sessions (draws passing of 10 for the two largest sessions, of 30 for the other six)", "",
              "| sessions | kind | read-out | " + " | ".join(f"{f:.0%}" for f in FRACTIONS) + " |", "|---|---|---|" + "---|" * len(FRACTIONS)]
    for group, names in (("two largest", [n for n in LARGE if n in R]), ("other six", [n for n in HC11_ORDER if n in R and n not in LARGE])):
        for kind in ("balanced", "forward"):
            for key in READOUTS:
                cells = []
                for f in FRACTIONS:
                    k0 = "balanced" if f == 0 else kind
                    rows = [x for n in names for x in R[n]["rows"] if x["kind"] == k0 and x["f"] == f]
                    cells.append(f"{count(rows, key)}/{len(rows)}" if rows else "–")
                lines.append(f"| {group} | {kind} | {LABELS[key]} | " + " | ".join(cells) + " |")
    lines += ["", "## Thresholds (lowest share with at least 4 of 5 draws passing in both of the two largest sessions)", "",
              "| read-out | balanced | forward-only |", "|---|---|---|"]
    names = [n for n in LARGE if n in R]
    for key in READOUTS:
        tb, tf = threshold(R, names, "balanced", key), threshold(R, names, "forward", key)
        lines.append(f"| {LABELS[key]} | {tb:.0%} | {tf:.0%} |".replace("None%", "> 20%") if tb is not None and tf is not None else
                     f"| {LABELS[key]} | {'> 20%' if tb is None else f'{tb:.0%}'} | {'> 20%' if tf is None else f'{tf:.0%}'} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, SOFT = "#1d2430", "#6b7380"
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": "white"})
    colours = {"standard": "#1f4e79", "pairs_adjacent": "#d4652f", "pairs_all": "#f0a070", "oracle_uncentred": "#9fd3b2", "oracle_centred": "#2e8b57"}
    groups = [("two largest sessions", [n for n in LARGE if n in R]), ("other six sessions", [n for n in HC11_ORDER if n in R and n not in LARGE])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    xs = [100 * f for f in FRACTIONS]
    for i, (gname, names) in enumerate(groups):
        for j, kind in enumerate(("balanced", "forward")):
            ax = axes[i, j]
            for key in READOUTS:
                ys = []
                for f in FRACTIONS:
                    k0 = "balanced" if f == 0 else kind
                    rows = [x for n in names for x in R[n]["rows"] if x["kind"] == k0 and x["f"] == f]
                    ys.append(count(rows, key) / len(rows) if rows else np.nan)
                ax.plot(xs, ys, "-o", color=colours[key], label=LABELS[key], lw=2 if key in ("standard", "pairs_adjacent") else 1.4, ms=5)
            ax.axhline(0.05, color="#c9ced6", lw=0.8, ls="--")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(f"{kind} planted replays")
            if j == 0:
                ax.set_ylabel(f"{gname}\nshare of draws recovering the map", color=INK)
            if i == 1:
                ax.set_xlabel("planted replay share (%)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Real hc-11 events: what direction balance costs, and what the pair read-out and the oracles recover", x=0.01, y=0.99, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, "Matched 1500 PRE events per session with ideal replays of the waking map planted at share f (5 draws per session and share).",
             fontsize=8.8, color=SOFT)
    fig.text(0.01, 0.92, "Best output against the map, best-of-outputs null, p < 0.05. Oracles use each planted replay's sense; other events get a random sense.",
             fontsize=8.8, color=SOFT)
    fig.tight_layout(rect=(0, 0.06, 1, 0.895))
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"spikein_direction.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    R = load()
    if R:
        summary(R)
        figure(R)
        print("figures/spikein_direction.png")
