"""How far apart one cell's spikes are inside windows of different lengths.

Uses the session mats already extracted. Does not modify them or the library.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hc11_window import MAT_DIR, analyze_session, load_session

OUT = Path(__file__).resolve().parents[1] / "figures" / "hc11_gap_distance.png"
WINDOWS_S = np.array([0.04, 0.08, 0.15, 0.30, 0.50])


def collect_gaps(by_cell, peaks, width):
    half = width / 2
    gaps = []
    for peak in peaks:
        start = peak - half
        stop = start + width
        for times in by_cell.values():
            chosen = times[(times >= start) & (times < stop)]
            if chosen.size < 2:
                continue
            gaps.append(np.diff(np.sort(chosen)).max())
    return np.array(gaps)


def main():
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    analyzed = [analyze_session(load_session(path)) for path in paths]
    pooled = []
    for width in WINDOWS_S:
        gaps = [collect_gaps(item["by_cell"], item["peaks"], width) for item in analyzed]
        pooled.append(np.concatenate(gaps))
        print(f"{width*1000:.0f} ms  n={pooled[-1].size}  median={np.median(pooled[-1])*1000:.1f}  p90={np.percentile(pooled[-1], 90)*1000:.1f}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    x = WINDOWS_S * 1000
    percentiles = [10, 50, 90]
    colors = ("#9aa4b2", "#1f4e79", "#b42318")
    labels = ("10th percentile", "median", "90th percentile")

    ax = axes[0]
    for pct, color, label in zip(percentiles, colors, labels):
        values = [np.percentile(gaps, pct) * 1000 for gaps in pooled]
        ax.plot(x, values, color=color, lw=2.2, marker="o", label=label)
    ax.plot(x, x, color="#d0d5dd", lw=1.2, label="the whole window")
    ax.set_xlabel("Window centered on the population peak (ms)")
    ax.set_ylabel("Largest gap inside one cell (ms)")
    ax.set_title("The extra spikes stay close to the burst")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    for pct, color, label in zip(percentiles, colors, labels):
        values = [100 * np.percentile(gaps, pct) / width for gaps, width in zip(pooled, WINDOWS_S)]
        ax.plot(x, values, color=color, lw=2.2, marker="o", label=label)
    ax.set_xlabel("Window centered on the population peak (ms)")
    ax.set_ylabel("Largest gap, as a percent of the window")
    ax.set_title("The same gap covers less of a longer window")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "A cell's spikes inside a rest population event stay in one cluster",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
