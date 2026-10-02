"""Repeating time gaps between pyramidal cells in hc-11 rest events.

Uses the same 100 ms windows as the PCA run. Does not modify the library
or the recordings. A pair is kept when the two cells share at least 50
events and the gap repeats with agreement at least 0.5.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_hc11_window import MAT_DIR, analyze_session, load_session
from cpca import encode
from runs.hc11_t100.run_cpca import WINDOW_MS, build_times

OUT = Path(__file__).resolve().parent / "cell_gaps.png"
NAVY = "#1f4e79"
MIN_SHARED = 50
MIN_AGREEMENT = 0.5


def short_label(name):
    rat, day = name.split("_")
    return f"{rat} {day[:2]}/{day[2:4]}"


def pair_gaps(path):
    item = analyze_session(load_session(path))
    cell_ids = np.array(sorted(item["by_cell"]))
    times, _, _ = build_times(item["by_cell"], item["bursts"], item["peaks"], cell_ids)
    active = np.isfinite(times).any(axis=-1)
    encoded = encode(times, WINDOW_MS, how="mean", t0=0.0)
    shared = active.T.astype(float) @ active.astype(float)
    # sum_m s_k conj(s_l): positive angle means cell k fires later than cell l
    total = encoded.T @ np.conj(encoded)
    both = np.triu_indices(len(cell_ids), k=1)
    count = shared[both]
    mean = total[both] / np.maximum(count, 1)
    gap_ms = np.angle(mean) * WINDOW_MS / np.pi
    agreement = np.abs(mean)
    return cell_ids, both, count, gap_ms, agreement


def main():
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    kept_gaps = []
    records = []
    for path in paths:
        cell_ids, both, count, gap_ms, agreement = pair_gaps(path)
        keep = (count >= MIN_SHARED) & (agreement >= MIN_AGREEMENT)
        kept_gaps.append(np.abs(gap_ms[keep]))
        earlier = np.where(gap_ms >= 0, both[1], both[0])
        later = np.where(gap_ms >= 0, both[0], both[1])
        for index in np.flatnonzero(keep):
            records.append((
                short_label(path.stem.replace("_sessInfo", "")),
                int(cell_ids[earlier[index]]),
                int(cell_ids[later[index]]),
                float(abs(gap_ms[index])),
                float(agreement[index]),
                int(count[index]),
            ))
        print(
            short_label(path.stem.replace("_sessInfo", "")),
            "pairs", int(keep.sum()),
            "median gap", round(float(np.median(np.abs(gap_ms[keep]))), 1),
            "gap>=10", int(np.sum(np.abs(gap_ms[keep]) >= 10)),
        )

    pooled = np.concatenate(kept_gaps)
    longest = sorted(records, key=lambda row: row[3], reverse=True)[:12]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), gridspec_kw={"width_ratios": [1.05, 1.15]})

    ax = axes[0]
    bins = np.arange(0, 31, 1)
    ax.hist(pooled, bins=bins, color=NAVY, width=0.72, zorder=2)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Time gap between the two cells (ms)")
    ax.set_ylabel("Cell pairs")
    ax.set_title("Gaps that repeat")
    ax.yaxis.grid(True, color="#eceff3", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5)
    ax.text(
        0.97,
        0.96,
        f"{len(pooled):,} pairs\n"
        "both cells fired in the\n"
        "same event ≥ 50 times\n"
        "agreement ≥ 0.5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#333333",
    )

    ax = axes[1]
    longest = longest[::-1]
    y = np.arange(len(longest))
    ax.barh(y, [row[3] for row in longest], color=NAVY, height=0.62, zorder=2)
    labels = [
        f"{row[0]}   {row[1]} then {row[2]}"
        for row in longest
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    for ypos, row in zip(y, longest):
        ax.text(row[3] + 0.4, ypos, f"{row[3]:.0f} ms", va="center", ha="left", fontsize=9, color="#333333")
    ax.set_xlabel("Time gap (ms)")
    ax.set_xlim(0, max(row[3] for row in longest) + 8)
    ax.set_title("The longest repeating gaps")
    ax.xaxis.grid(True, color="#eceff3", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3.5)
    fig.suptitle(
        "Cells that keep a time gap across rest events",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)
    print("pooled pairs", pooled.size, "median", round(float(np.median(pooled)), 2))
    print("under 5 ms", round(100 * np.mean(pooled < 5), 1), "at least 10 ms", round(100 * np.mean(pooled >= 10), 1))


if __name__ == "__main__":
    main()
