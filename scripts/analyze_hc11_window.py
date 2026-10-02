"""How long a window can be before one cell's spikes split into two times.

Reads extracted hc-11 session mats. Does not modify them or the library.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

MAT_DIR = Path(__file__).resolve().parents[1] / "data" / "hc11_sessinfo" / "NoveltySessInfoMatFiles"
OUT = Path(__file__).resolve().parents[1] / "figures" / "hc11_window_tuning.png"

WINDOWS_S = np.array([0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50])
BURST_GAP_S = 0.015


def load_session(path):
    with h5py.File(path, "r") as handle:
        epochs = handle["sessInfo/Epochs"]
        spikes = handle["sessInfo/Spikes"]
        position = handle["sessInfo/Position"]
        maze = "".join(chr(c) for c in np.array(position["MazeType"]).ravel())
        def interval(name):
            values = np.array(epochs[name])
            if values.shape[0] != 2:
                values = values.T
            return values.T.copy()
        return {
            "name": path.name.replace("_sessInfo.mat", ""),
            "maze": maze,
            "pre": interval("PREEpoch").ravel(),
            "maze_epoch": interval("MazeEpoch").ravel(),
            "post": interval("POSTEpoch").ravel(),
            "nrem": interval("NREM"),
            "times": np.array(spikes["SpikeTimes"]).ravel(),
            "ids": np.array(spikes["SpikeIDs"]).ravel(),
            "pyr": np.array(spikes["PyrIDs"]).ravel(),
            "interneurons": np.array(spikes["IntIDs"]).ravel(),
        }


def in_intervals(times, intervals):
    keep = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        keep |= (times >= start) & (times < end)
    return keep


def population_bursts(spike_times, nrem, bin_s=0.01):
    if spike_times.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    start = max(0.0, spike_times.min() - 1)
    stop = spike_times.max() + 1
    edges = np.arange(start, stop, bin_s)
    counts, _ = np.histogram(spike_times, bins=edges)
    centers = edges[:-1] + bin_s / 2
    kernel = np.exp(-0.5 * (np.arange(-3, 4)) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    in_nrem = in_intervals(centers, nrem)
    baseline = smooth[in_nrem]
    threshold = baseline.mean() + 3 * baseline.std()
    above = (smooth >= threshold) & in_nrem
    if not above.any():
        return np.zeros((0, 2)), np.zeros(0)
    changes = np.diff(above.astype(int))
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, len(above)]
    events = []
    peaks = []
    for begin, finish in zip(starts, ends):
        t0 = centers[begin] - bin_s / 2
        t1 = centers[finish - 1] + bin_s / 2
        if events and t0 - events[-1][1] <= 0.03:
            events[-1][1] = t1
            if smooth[begin:finish].max() > smooth[int((peaks[-1] - start) / bin_s)]:
                peaks[-1] = centers[begin + np.argmax(smooth[begin:finish])]
        else:
            events.append([t0, t1])
            peaks.append(centers[begin + np.argmax(smooth[begin:finish])])
    events = np.array(events)
    peaks = np.array(peaks)
    duration = np.diff(events, axis=1).ravel()
    keep = (duration >= 0.03) & (duration <= 0.40)
    return events[keep], peaks[keep]


def cell_gaps(spike_lists):
    gaps = []
    counts = []
    for times in spike_lists:
        counts.append(times.size)
        if times.size >= 2:
            gaps.append(np.diff(np.sort(times)).max())
    return np.array(counts), np.array(gaps)


def spikes_in_window(cell_times, start, end):
    found = []
    for times in cell_times:
        chosen = times[(times >= start) & (times < end)]
        if chosen.size:
            found.append(chosen)
    return found


def analyze_session(session):
    pyr = set(session["pyr"].tolist())
    pyr_mask = np.isin(session["ids"], session["pyr"])
    times = session["times"][pyr_mask]
    ids = session["ids"][pyr_mask]
    nrem = session["nrem"]
    pre_post = nrem[
        (nrem[:, 1] <= session["maze_epoch"][0] + 1)
        | (nrem[:, 0] >= session["maze_epoch"][1] - 1)
    ]
    in_rest = in_intervals(times, pre_post)
    times = times[in_rest]
    ids = ids[in_rest]
    order = np.argsort(times)
    times = times[order]
    ids = ids[order]
    bursts, peaks = population_bursts(times, pre_post)
    by_cell = {cell: times[ids == cell] for cell in pyr}
    return {
        "session": session,
        "n_pyr_spikes_rest": int(times.size),
        "bursts": bursts,
        "peaks": peaks,
        "by_cell": by_cell,
    }


def window_stats(by_cell, peaks, windows):
    """Among cells with 2 or more spikes, how often the mean arrow collapses."""
    split = []
    weak = []
    resultants = []
    for width in windows:
        n_split = 0
        n_weak = 0
        n_multi = 0
        resultant_sum = 0.0
        half = width / 2
        for peak in peaks:
            start = peak - half
            stop = start + width
            for times in by_cell.values():
                chosen = times[(times >= start) & (times < stop)]
                if chosen.size < 2:
                    continue
                n_multi += 1
                ordered = np.sort(chosen)
                if np.diff(ordered).max() > BURST_GAP_S:
                    n_split += 1
                centered = (ordered - start) / width
                length = np.abs(np.exp(1j * np.pi * centered).mean())
                resultant_sum += length
                if length < 0.5:
                    n_weak += 1
        split.append(n_split / n_multi if n_multi else np.nan)
        weak.append(n_weak / n_multi if n_multi else np.nan)
        resultants.append(resultant_sum / n_multi if n_multi else np.nan)
    return np.array(split), np.array(weak), np.array(resultants)


def main():
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    sessions = [load_session(path) for path in paths]
    analyzed = [analyze_session(session) for session in sessions]

    rows = []
    all_durations = []
    all_natural_gaps = []
    fraction_rows = []
    weak_rows = []
    resultant_rows = []
    for item in analyzed:
        session = item["session"]
        duration_h = {
            "pre": (session["pre"][1] - session["pre"][0]) / 3600,
            "maze": (session["maze_epoch"][1] - session["maze_epoch"][0]) / 60,
            "post": (session["post"][1] - session["post"][0]) / 3600,
        }
        bursts = item["bursts"]
        durations = np.diff(bursts, axis=1).ravel() if len(bursts) else np.array([])
        all_durations.append(durations)
        natural_gaps = []
        if len(bursts):
            for start, end in bursts:
                for times in item["by_cell"].values():
                    chosen = times[(times >= start) & (times < end)]
                    if chosen.size >= 2:
                        natural_gaps.append(np.diff(np.sort(chosen)).max())
        all_natural_gaps.append(np.array(natural_gaps))
        fractions, weak, resultants = window_stats(item["by_cell"], item["peaks"], WINDOWS_S)
        fraction_rows.append(fractions)
        weak_rows.append(weak)
        resultant_rows.append(resultants)
        rows.append(
            [
                session["name"].split("_")[0],
                session["maze"].strip(),
                len(session["pyr"]),
                len(session["interneurons"]),
                f"{duration_h['pre']:.1f} h",
                f"{duration_h['maze']:.0f} min",
                f"{duration_h['post']:.1f} h",
                f"{len(bursts)}",
                f"{np.median(durations)*1000:.0f} ms" if len(durations) else "—",
                f"{100 * np.mean(np.array(natural_gaps) > BURST_GAP_S):.0f}%" if len(natural_gaps) else "—",
            ]
        )
        print(
            session["name"],
            session["maze"],
            "pyr", len(session["pyr"]),
            "bursts", len(bursts),
            "median ms", np.median(durations) * 1000 if len(durations) else None,
            "split", np.mean(np.array(natural_gaps) > BURST_GAP_S) if len(natural_gaps) else None,
        )

    fractions = np.vstack(fraction_rows)
    weak = np.vstack(weak_rows)
    resultants = np.vstack(resultant_rows)
    durations = np.concatenate(all_durations)
    natural_gaps = np.concatenate([g for g in all_natural_gaps if g.size])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(12.2, 8.6))
    grid = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    ax.hist(durations * 1000, bins=np.linspace(30, 400, 38), color="#1f4e79", alpha=0.9)
    ax.axvline(np.median(durations) * 1000, color="#b42318", lw=1.5, label=f"median {np.median(durations)*1000:.0f} ms")
    ax.set_xlabel("Population-burst duration (ms)")
    ax.set_ylabel("Events")
    ax.set_title("Detected rest bursts, all 8 sessions")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[0, 1])
    ax.hist(natural_gaps * 1000, bins=np.linspace(0, 200, 40), color="#1f8a65", alpha=0.9)
    ax.axvline(BURST_GAP_S * 1000, color="#b42318", lw=1.5, label="15 ms, longer than one burst")
    ax.set_xlabel("Largest gap between one cell's spikes in a burst (ms)")
    ax.set_ylabel("Cell participations with 2 or more spikes")
    ax.set_title("Do spikes inside one burst already split?")
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    x = WINDOWS_S * 1000
    for row, item in zip(fractions, analyzed):
        ax.plot(x, 100 * row, color="#d0d5dd", lw=1)
    ax.plot(x, 100 * np.nanmean(fractions, axis=0), color="#b42318", lw=2.4, marker="o", label="gap over 15 ms")
    ax.plot(x, 100 * np.nanmean(weak, axis=0), color="#1f4e79", lw=2.4, marker="o", label="mean arrow shorter than 0.5")
    ax.set_xlabel("Window centered on the burst peak (ms)")
    ax.set_ylabel("Multi-spike cell participations (%)")
    ax.set_title("A gap is common. Cancellation is not.")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)

    ax = fig.add_subplot(grid[1, 1])
    for row in resultants:
        ax.plot(x, row, color="#d0d5dd", lw=1)
    ax.plot(x, np.nanmean(resultants, axis=0), color="#1f4e79", lw=2.4, marker="o")
    ax.axhline(0.7, color="#b42318", lw=1, ls="--", label="resultant 0.7")
    ax.set_xlabel("Window length T, mapped onto a half-circle (ms)")
    ax.set_ylabel("Mean arrow length of one cell")
    ax.set_title("Mean encoding stays concentrated inside burst-length windows")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)

    fig.suptitle(
        "hc-11 rest bursts: one cell's spikes are one burst, until the window gets long",
        fontsize=14,
        y=0.98,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)
    print("sessions", len(rows))
    print("pooled bursts", durations.size, "median ms", np.median(durations) * 1000)
    print("natural split fraction", np.mean(natural_gaps > BURST_GAP_S))
    print("windows", x.tolist())
    print("mean split %", np.round(100 * np.nanmean(fractions, axis=0), 1))
    print("mean weak %", np.round(100 * np.nanmean(weak, axis=0), 1))
    print("mean resultant", np.round(np.nanmean(resultants, axis=0), 3))
    for row in rows:
        print(" | ".join(str(item) for item in row))


if __name__ == "__main__":
    main()
