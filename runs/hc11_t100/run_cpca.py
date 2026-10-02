"""Run the current complex PCA on hc-11 rest events.

Reads extracted session mats and imports cpca. Does not modify either.
Window is 100 ms centered on each population-event peak. Encoding is the
mean phasor. Columns are not centered.

Outputs stay in this directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_hc11_window import MAT_DIR, analyze_session, load_session
from cpca import (
    coactivation_null,
    decompose,
    encode,
    readout,
    split_by_direction,
    split_half,
    timing_centered,
    timing_excess,
)
from cpca.theory import lag_matrix

OUT_DIR = Path(__file__).resolve().parent
RESULT = OUT_DIR / "results.npz"
FIGURE = OUT_DIR / "hc11_t100_results.png"

WINDOW_MS = 100.0
MAX_SPIKES = 24
N_COMPONENTS = 6
N_SHUFFLES = 40
MIN_WEIGHT = 0.1
RNG_SEED = 0


def build_times(by_cell, bursts, peaks, cell_ids):
    half_s = (WINDOW_MS / 2) / 1000
    n_events = len(peaks)
    n_cells = len(cell_ids)
    times = np.full((n_events, n_cells, MAX_SPIKES), np.nan)
    truncated = 0
    in_event = 0
    outside = 0
    starts = peaks - half_s
    ends = peaks + half_s
    for cell, cell_id in enumerate(cell_ids):
        spikes = np.asarray(by_cell[cell_id])
        if spikes.size == 0:
            continue
        left = np.searchsorted(spikes, starts, side="left")
        right = np.searchsorted(spikes, ends, side="left")
        event_left = np.searchsorted(spikes, bursts[:, 0], side="left")
        event_right = np.searchsorted(spikes, bursts[:, 1], side="left")
        for event in range(n_events):
            count = int(right[event] - left[event])
            if count > 0:
                take = min(count, MAX_SPIKES)
                if count > MAX_SPIKES:
                    truncated += 1
                chosen = spikes[left[event] : left[event] + take]
                times[event, cell, :take] = (chosen - starts[event]) * 1000
            start = int(event_left[event])
            stop = int(event_right[event])
            if stop > start:
                inside = spikes[start:stop]
                in_event += inside.size
                outside += int(np.sum((inside < starts[event]) | (inside >= ends[event])))
    clipped = outside / in_event if in_event else np.nan
    return times, truncated, clipped


def leading_delta_eigenvalue(encoded, active):
    lag = lag_matrix(encoded)
    delta = timing_centered(lag, coactivation_null(encoded, active))
    hermitian = (delta + delta.conj().T) / 2
    return float(np.linalg.eigvalsh(hermitian)[-1])


def permute_arrows(encoded, active, rng):
    shuffled = np.array(encoded, copy=True)
    for row, mask in zip(shuffled, active):
        index = np.flatnonzero(mask)
        if index.size < 2:
            continue
        row[index] = row[index][rng.permutation(index.size)]
    return shuffled


def component_excess(encoded, components, active):
    excess = []
    for component in range(components.shape[1]):
        excess.append(float(np.mean(timing_excess(encoded, components[:, component], active))))
    return np.array(excess)


def safe_spearman(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return np.nan
    coefficient, _ = spearmanr(left[mask], right[mask])
    return float(coefficient)


def run_session(path, rng):
    item = analyze_session(load_session(path))
    session = item["session"]
    cell_ids = np.array(sorted(item["by_cell"]))
    bursts = item["bursts"]
    peaks = item["peaks"]
    times, truncated, clipped = build_times(item["by_cell"], bursts, peaks, cell_ids)
    active = np.isfinite(times).any(axis=-1)
    encoded = encode(times, WINDOW_MS, how="mean", t0=0.0)
    eigenvalues, components, scores, lag = decompose(encoded, n_components=N_COMPONENTS)
    trace = float(np.real(np.trace(lag)))
    excess = component_excess(encoded, components, active)
    observed_delta = leading_delta_eigenvalue(encoded, active)
    null_delta = np.empty(N_SHUFFLES)
    for draw in range(N_SHUFFLES):
        null_delta[draw] = leading_delta_eigenvalue(
            permute_arrows(encoded, active, rng), active
        )
    overlap, held_out = split_half(encoded, rng)
    weights, order, _, included, span = readout(components[:, 0], WINDOW_MS, min_weight=MIN_WEIGHT)
    column_mean = encoded.mean(axis=0)
    mean_norm = np.linalg.norm(column_mean)
    alignment = float(np.abs(np.vdot(components[:, 0], column_mean)) / mean_norm) if mean_norm else np.nan
    mean_time = (WINDOW_MS / np.pi) * np.angle(column_mean)
    mean_time[np.abs(column_mean) == 0] = np.nan
    order_vs_mean = safe_spearman(order, mean_time)

    maze_start = float(session["maze_epoch"][0])
    maze_end = float(session["maze_epoch"][1])
    pre = peaks < maze_start
    post = peaks >= maze_end
    pre_post = np.nan
    if int(pre.sum()) >= 2 and int(post.sum()) >= 2:
        _, pre_components, _, _ = decompose(encoded[pre], n_components=1)
        _, post_components, _, _ = decompose(encoded[post], n_components=1)
        pre_post = float(np.abs(np.vdot(pre_components[:, 0], post_components[:, 0])))

    split = split_by_direction(encoded)
    signs = split["signs"]
    positive = float(np.mean(signs > 0))
    direction_corr = np.nan
    groups = split["groups"]
    if 1.0 in groups and -1.0 in groups:
        _, forward, _, forward_included, _ = readout(groups[1.0]["component"], WINDOW_MS, MIN_WEIGHT)
        _, reverse, _, reverse_included, _ = readout(groups[-1.0]["component"], WINDOW_MS, MIN_WEIGHT)
        both = forward_included & reverse_included
        direction_corr = safe_spearman(forward, reverse) if both.any() else np.nan
        forward[~both] = np.nan
        reverse[~both] = np.nan
    else:
        forward = np.full(cell_ids.shape, np.nan)
        reverse = np.full(cell_ids.shape, np.nan)

    included_times = np.where(included, order, np.nan)
    return {
        "name": path.name.replace("_sessInfo.mat", ""),
        "n_events": int(len(peaks)),
        "n_cells": int(len(cell_ids)),
        "n_pre": int(pre.sum()),
        "n_post": int(post.sum()),
        "participation": float(active.mean()),
        "truncated": int(truncated),
        "clipped": float(clipped),
        "eigenvalues": eigenvalues,
        "trace": trace,
        "excess": excess,
        "delta_eigenvalue": observed_delta,
        "null_delta": null_delta,
        "split_overlap": float(overlap),
        "held_out": float(held_out),
        "n_included": int(included.sum()),
        "span": float(span),
        "template_times": included_times,
        "weights": weights,
        "alignment": alignment,
        "order_vs_mean": order_vs_mean,
        "pre_post": pre_post,
        "positive_fraction": positive,
        "direction_corr": direction_corr,
        "cell_ids": cell_ids,
    }


def pack(rows):
    max_cells = max(row["n_cells"] for row in rows)
    names = np.array([row["name"] for row in rows])
    template = np.full((len(rows), max_cells), np.nan)
    weights = np.full((len(rows), max_cells), np.nan)
    for index, row in enumerate(rows):
        count = row["n_cells"]
        template[index, :count] = row["template_times"]
        weights[index, :count] = row["weights"]
    scalar_keys = [
        "n_events", "n_cells", "n_pre", "n_post", "participation", "truncated",
        "clipped", "trace", "delta_eigenvalue", "split_overlap", "held_out",
        "n_included", "span", "alignment", "order_vs_mean", "pre_post",
        "positive_fraction", "direction_corr",
    ]
    payload = {
        "names": names,
        "eigenvalues": np.vstack([row["eigenvalues"] for row in rows]),
        "excess": np.vstack([row["excess"] for row in rows]),
        "null_delta": np.vstack([row["null_delta"] for row in rows]),
        "template_times": template,
        "weights": weights,
    }
    for key in scalar_keys:
        payload[key] = np.array([row[key] for row in rows])
    return payload


def short_label(name):
    rat, day = name.split("_")
    return f"{rat}\n{day[:2]}/{day[2:4]}"


def draw(data):
    names = [short_label(name) for name in data["names"]]
    x = np.arange(len(names))
    navy = "#1f4e79"
    red = "#b42318"
    green = "#1f8a65"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))

    ax = axes[0, 0]
    fractions = data["eigenvalues"] / data["trace"][:, None]
    for index in range(fractions.shape[0]):
        ax.plot(np.arange(1, N_COMPONENTS + 1), 100 * fractions[index], color="#c5d0dc", lw=1)
    ax.plot(np.arange(1, N_COMPONENTS + 1), 100 * fractions.mean(axis=0), color=navy, lw=2.4, marker="o")
    ax.set_xlabel("Component")
    ax.set_ylabel("Share of lag-matrix trace (%)")
    ax.set_title("Share of the uncentered lag matrix")
    ax.set_xticks(np.arange(1, N_COMPONENTS + 1))

    ax = axes[0, 1]
    null = data["null_delta"]
    observed = data["delta_eigenvalue"]
    ax.scatter(np.repeat(x, null.shape[1]), null.ravel(), s=8, color="#c5d0dc", label="arrow reassignment", zorder=2)
    ax.scatter(x, observed, s=36, color=red, label="observed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Largest eigenvalue of C − C0")
    ax.set_title("Timing eigenvalue against reassigned arrows")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    for index in range(data["template_times"].shape[0]):
        times = data["template_times"][index]
        times = np.sort(times[np.isfinite(times)])
        if times.size == 0:
            continue
        ax.plot(times, np.full(times.shape, index), "|", color=navy, ms=8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Time relative to the strongest cell (ms)")
    ax.set_title("First component, cells above 10% of its largest weight")
    ax.set_xlim(-WINDOW_MS, WINDOW_MS)

    ax = axes[1, 1]
    width = 0.36
    ax.bar(x - width / 2, data["split_overlap"], width=width, color=navy, label="two halves of the events")
    ax.bar(x + width / 2, data["pre_post"], width=width, color=green, label="PRE template vs POST template")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Absolute overlap of the first component")
    ax.set_title("Does the first component repeat?")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Complex PCA on hc-11 rest events, 100 ms window, mean encoding", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(FIGURE)


def report(data):
    fractions = data["eigenvalues"] / data["trace"][:, None]
    null_p99 = np.percentile(data["null_delta"], 99, axis=1)
    for index, name in enumerate(data["names"]):
        print(
            f"{name}  events {data['n_events'][index]:.0f}  cells {data['n_cells'][index]:.0f}  "
            f"PRE {data['n_pre'][index]:.0f} POST {data['n_post'][index]:.0f}"
        )
        print(
            f"  participation {100 * data['participation'][index]:.1f}%  "
            f"clipped {100 * data['clipped'][index]:.1f}%  truncated {data['truncated'][index]:.0f}"
        )
        print(
            f"  trace share {np.array2string(100 * fractions[index], precision=1)}  "
            f"excess {np.array2string(data['excess'][index], precision=3)}"
        )
        print(
            f"  C-C0 eig {data['delta_eigenvalue'][index]:.4f}  "
            f"null 99th {null_p99[index]:.4f}  "
            f"null max {data['null_delta'][index].max():.4f}"
        )
        print(
            f"  included {data['n_included'][index]:.0f}  span {data['span'][index]:.2f} rad  "
            f"align {data['alignment'][index]:.3f}  "
            f"order vs mean time {data['order_vs_mean'][index]:.3f}"
        )
        print(
            f"  split-half {data['split_overlap'][index]:.3f}  held-out {data['held_out'][index]:.4f}  "
            f"in-sample λ1 {data['eigenvalues'][index, 0]:.4f}"
        )
        print(
            f"  PRE-POST {data['pre_post'][index]:.3f}  "
            f"+ fraction {data['positive_fraction'][index]:.3f}  "
            f"direction corr {data['direction_corr'][index]:.3f}"
        )


def main():
    rng = np.random.default_rng(RNG_SEED)
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    rows = []
    for path in paths:
        print("running", path.name, flush=True)
        rows.append(run_session(path, rng))
    payload = pack(rows)
    np.savez(RESULT, **payload)
    report(payload)
    draw(payload)
    print(RESULT)


if __name__ == "__main__":
    main()
