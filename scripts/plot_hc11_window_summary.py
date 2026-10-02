"""One figure for the window decision on hc-11 rest events.

Reads extracted session mats. Does not modify them or the library.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hc11_window import MAT_DIR, analyze_session, load_session

OUT = Path(__file__).resolve().parents[1] / "figures" / "hc11_window_summary.png"
CACHE = OUT.with_suffix(".npz")
WINDOWS_S = np.array([0.04, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50])
BURST_GAP_S = 0.006


def cell_gaps(by_cell, peaks, width):
    half = width / 2
    gaps = []
    lengths = []
    for peak in peaks:
        start = peak - half
        stop = start + width
        for times in by_cell.values():
            chosen = times[(times >= start) & (times < stop)]
            if chosen.size < 2:
                continue
            ordered = np.sort(chosen)
            gaps.append(np.diff(ordered).max())
            centered = (ordered - start) / width
            lengths.append(np.abs(np.exp(1j * np.pi * centered).mean()))
    return np.array(gaps), np.array(lengths)


def second_event_rate(peak_lists, width):
    hit = 0
    total = 0
    half = width / 2
    for peaks in peak_lists:
        total += len(peaks)
        if len(peaks) < 2:
            continue
        for time in peaks:
            lo = np.searchsorted(peaks, time - half, side="left")
            hi = np.searchsorted(peaks, time + half, side="right")
            if hi - lo > 1:
                hit += 1
    return 100 * hit / total if total else np.nan


def spike_structure():
    """Spike counts inside each detected event, and which repeats are bursts.

    A burst means every gap between that cell's spikes is at most 6 ms.
    A longer gap means separate spikes inside the same event.
    """
    single = 0
    burst = np.zeros(3, dtype=np.int64)
    separate = np.zeros(3, dtype=np.int64)
    for path in sorted(MAT_DIR.glob("*_sessInfo.mat")):
        item = analyze_session(load_session(path))
        arrays = [np.asarray(times) for times in item["by_cell"].values()]
        for start, end in item["bursts"]:
            for times in arrays:
                i0 = int(np.searchsorted(times, start, side="left"))
                i1 = int(np.searchsorted(times, end, side="left"))
                n = i1 - i0
                if n <= 0:
                    continue
                if n == 1:
                    single += 1
                    continue
                idx = min(n, 4) - 2
                if np.diff(times[i0:i1]).max() <= BURST_GAP_S:
                    burst[idx] += 1
                else:
                    separate[idx] += 1
    total = single + burst.sum() + separate.sum()
    return {
        "single_pct": np.array([100 * single / total]),
        "burst_pct": 100 * burst / total,
        "separate_pct": 100 * separate / total,
    }


def collect():
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    analyzed = [analyze_session(load_session(path)) for path in paths]
    durations = np.concatenate([
        np.diff(item["bursts"], axis=1).ravel()
        for item in analyzed
        if len(item["bursts"])
    ])
    peak_lists = [np.sort(item["peaks"]) for item in analyzed if len(item["peaks"])]
    next_gap = np.concatenate([np.diff(peaks) for peaks in peak_lists if len(peaks) > 1])

    median = []
    p90 = []
    arrow = []
    weak = []
    second = []
    for width in WINDOWS_S:
        gaps, lengths = zip(*(
            cell_gaps(item["by_cell"], item["peaks"], width) for item in analyzed
        ))
        gaps = np.concatenate(gaps)
        lengths = np.concatenate(lengths)
        median.append(np.median(gaps))
        p90.append(np.percentile(gaps, 90))
        arrow.append(lengths.mean())
        weak.append(100 * np.mean(lengths < 0.5))
        second.append(second_event_rate(peak_lists, width))
        print(
            f"{width * 1000:.0f} ms  median gap {median[-1] * 1000:.1f}  "
            f"p90 {p90[-1] * 1000:.1f}  arrow {arrow[-1]:.3f}  "
            f"arrow<0.5 {weak[-1]:.1f}%  second event {second[-1]:.1f}%"
        )

    payload = {
        "durations": durations,
        "next_gap": next_gap,
        "windows": WINDOWS_S,
        "median": np.array(median),
        "p90": np.array(p90),
        "arrow": np.array(arrow),
        "weak": np.array(weak),
        "second": np.array(second),
    }
    np.savez(CACHE, **payload)
    return payload


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.5, labelsize=10)
    ax.yaxis.grid(True, color="#eceff3", zorder=0)
    ax.set_axisbelow(True)


def draw(data):
    durations_ms = np.round(data["durations"] * 1000).astype(int)
    next_gap = data["next_gap"]
    x = data["windows"] * 1000
    median_ms = data["median"] * 1000
    share = 100 * data["median"] / data["windows"]
    navy = "#1f4e79"
    red = "#b42318"
    amber = "#c2410c"
    green = "#1f8a65"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "regular",
        "axes.titlesize": 12,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.0))
    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.11, hspace=0.62, wspace=0.42)

    ax = axes[0, 0]
    centers = np.arange(30, 160, 10)
    counts = np.array([(durations_ms == center).sum() for center in centers])
    ax.bar(centers, counts, width=6.4, color=navy, zorder=2)
    median_count = counts[centers == 40][0]
    ax.bar([40], [median_count], width=6.4, color=red, zorder=3)
    ax.set_xlim(22, 158)
    ax.set_xticks(np.arange(30, 160, 10))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Length of one population event (ms)")
    ax.set_ylabel("Events")
    ax.set_title("One population event lasts about 40 ms")
    ax.text(
        0.98,
        0.96,
        "Red bar: median, 40 ms\n90% end by 70 ms\n95% end by 90 ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#333333",
    )
    _style_axes(ax)

    ax = axes[0, 1]
    edges = np.array([0.0, 0.1, 0.2, 0.5, 1, 2, 5, 10, np.inf])
    labels = ["< 0.1 s", "0.1–0.2 s", "0.2–0.5 s", "0.5–1 s", "1–2 s", "2–5 s", "5–10 s", "≥ 10 s"]
    gap_counts = np.array([
        np.sum((next_gap >= lo) & (next_gap < hi)) for lo, hi in zip(edges[:-1], edges[1:])
    ])
    gap_pct = 100 * gap_counts / gap_counts.sum()
    positions = np.arange(len(labels))
    ax.bar(positions, gap_pct, width=0.62, color=navy, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=32, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Time from one event to the next")
    ax.set_ylabel("Successive events (%)")
    ax.set_title("The next event is usually seconds away")
    ax.text(
        0.03,
        0.96,
        "10th percentile 190 ms\nmedian 2.0 s\n90th percentile 9.0 s",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#333333",
    )
    _style_axes(ax)

    ax = axes[1, 0]
    gap_line, = ax.plot(x, median_ms, color=navy, lw=2.4, marker="o", ms=7, label="Median gap (ms)", zorder=3)
    ax.set_xlabel("Window centered on one event (ms)")
    ax.set_ylabel("Median gap (ms)", color=navy)
    ax.tick_params(axis="y", colors=navy)
    ax.set_ylim(0, 80)
    ax.set_title("The typical gap inside one cell")
    ax2 = ax.twinx()
    share_line, = ax2.plot(
        x, share, color=amber, lw=2.4, marker="o", ms=7, label="Share of the window (%)", zorder=3
    )
    ax2.set_ylabel("Share of the window (%)", color=amber)
    ax2.tick_params(axis="y", colors=amber)
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(amber)
    ax.spines["left"].set_color(navy)
    _style_axes(ax)
    ax.legend(
        handles=[gap_line, share_line],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    ax = axes[1, 1]
    spike_x = np.arange(4)
    burst = np.asarray(data["burst_pct"], dtype=float)
    separate = np.asarray(data["separate_pct"], dtype=float)
    single = float(np.asarray(data["single_pct"]).ravel()[0])
    ax.bar(spike_x[0], single, width=0.62, color=navy, zorder=2)
    ax.bar(spike_x[1:], burst, width=0.62, color=green, zorder=2)
    ax.bar(spike_x[1:], separate, width=0.62, bottom=burst, color=amber, zorder=2)
    for xpos, height in zip(spike_x, [single, *(burst + separate)]):
        ax.text(xpos, height + 1.6, f"{height:.0f}%", ha="center", va="bottom", fontsize=10, color="#333333")
    ax.set_xticks(spike_x)
    ax.set_xticklabels(["1 spike", "2 spikes", "3 spikes", "4 or more"])
    ax.set_ylim(0, 84)
    ax.set_xlabel("Spikes from one cell in the event")
    ax.set_ylabel("Cells that fired in the event (%)")
    ax.set_title("One spike, a burst, or separate spikes")
    _style_axes(ax)
    ax.legend(
        handles=[
            Patch(facecolor=navy, label="One spike"),
            Patch(facecolor=green, label="Burst, every gap ≤ 6 ms"),
            Patch(facecolor=amber, label="Separate spikes, a longer gap"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=1,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "hc-11 rest: events are brief, and extra spikes are often separate firings",
        fontsize=13,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)


def main():
    if CACHE.exists():
        loaded = np.load(CACHE)
        data = {key: loaded[key] for key in loaded.files}
    else:
        data = collect()
    if "burst_pct" not in data:
        data.update(spike_structure())
        np.savez(CACHE, **data)
    draw(data)


if __name__ == "__main__":
    main()
