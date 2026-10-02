"""Temporal patterns from waking and from rest, extracted the same way.

Mean of the complex spikes, then PCA. A waking event is one run along the
track. A rest event is a 100 ms window on a population burst. Patterns are
compared by the order of their cell times.

Does not modify the library, the recordings, or the earlier figures.
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_hc11_window import MAT_DIR, analyze_session, load_session
from cpca import decompose, encode, readout, split_by_direction
from runs.hc11_t100.run_cpca import WINDOW_MS, build_times

OUT_DIR = Path(__file__).resolve().parent
NAVY = "#1f4e79"
GREEN = "#1f8a65"
MIN_SPEED = 0.05
MIN_COVER = 0.6
MAX_SPIKES = 300
N_REST = 6
N_WAKE = 3
MIN_CELLS = 8
NULL_DRAWS = 200


def short_label(name):
    rat, day = name.split("_")
    return f"{rat} {day[:2]}/{day[2:4]}"


def load_position(path):
    with h5py.File(path, "r") as handle:
        return (
            np.array(handle["sessInfo/Position/TimeStamps"]).ravel(),
            np.array(handle["sessInfo/Position/OneDLocation"]).ravel(),
        )


def track_passes(ts, position, circular):
    ok = np.isfinite(position)
    pos_t = ts[ok]
    pos_x = position[ok]
    if pos_t.size < 10:
        return []
    length = float(np.nanmax(position))
    dt = np.diff(pos_t)
    dx = np.diff(pos_x)
    if circular:
        dx = (dx + length / 2) % length - length / 2
    direction = np.sign(dx)
    gap = np.zeros(len(dx), dtype=bool)
    gap[0] = True
    gap[1:] = dt[1:] > 0.5
    turn = np.zeros(len(dx), dtype=bool)
    turn[0] = True
    turn[1:] = (direction[1:] != direction[:-1]) | (direction[1:] == 0)
    cuts = np.flatnonzero(gap | turn)
    bounds = np.r_[cuts, len(dx)]
    passes = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop <= start:
            continue
        travel = float(dx[start:stop].sum())
        if abs(travel) < MIN_COVER * length:
            continue
        passes.append((float(pos_t[start]), float(pos_t[stop]), float(np.sign(travel))))
    return passes


def waking_tensor(by_cell, cell_ids, passes):
    chosen = [row for row in passes]
    n_events = len(chosen)
    n_cells = len(cell_ids)
    times = np.full((n_events, n_cells, MAX_SPIKES), np.nan)
    for event, (start, stop, _) in enumerate(chosen):
        width = stop - start
        if width <= 0:
            continue
        for cell, cell_id in enumerate(cell_ids):
            spikes = np.asarray(by_cell[cell_id])
            if spikes.size == 0:
                continue
            left = int(np.searchsorted(spikes, start, side="left"))
            right = int(np.searchsorted(spikes, stop, side="left"))
            count = right - left
            if count <= 0:
                continue
            take = min(count, MAX_SPIKES)
            stamps = spikes[left : left + take]
            times[event, cell, :take] = (stamps - start) / width * WINDOW_MS
    return times


def pattern_times(encoded, n_components):
    if encoded.shape[0] < 2:
        return []
    _, components, _, _ = decompose(encoded, n_components=n_components)
    times = []
    for component in range(components.shape[1]):
        _, order, _, _, _ = readout(components[:, component], WINDOW_MS, min_weight=0.0)
        times.append(order)
    return times


def rank_match(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < MIN_CELLS:
        return np.nan
    coefficient, _ = spearmanr(left[mask], right[mask])
    return float(coefficient)


def rest_patterns(item, cell_ids):
    by_cell = item["by_cell"]
    times, _, _ = build_times(by_cell, item["bursts"], item["peaks"], cell_ids)
    encoded = encode(times, WINDOW_MS, how="mean", t0=0.0)
    patterns = pattern_times(encoded, N_REST)
    labels = [f"rest {index + 1}" for index in range(len(patterns))]
    split = split_by_direction(encoded)
    for sign, name in ((1.0, "rest group +"), (-1.0, "rest group −")):
        group = split["groups"].get(sign)
        if group is None:
            continue
        _, order, _, _, _ = readout(group["component"], WINDOW_MS, min_weight=0.0)
        patterns.append(order)
        labels.append(name)
    return patterns, labels


def waking_patterns(session, cell_ids, passes):
    by_cell = {}
    pyr = set(np.asarray(cell_ids).tolist())
    maze = session["maze_epoch"]
    on_maze = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    times = session["times"][on_maze]
    ids = session["ids"][on_maze]
    for cell in cell_ids:
        by_cell[cell] = times[ids == cell]
    grouped = {1.0: [], -1.0: []}
    for row in passes:
        grouped[row[2]].append(row)
    patterns = []
    labels = []
    counts = {}
    for sign, name in ((1.0, "waking one way"), (-1.0, "waking the other way")):
        counts[sign] = len(grouped[sign])
        tensor = waking_tensor(by_cell, cell_ids, grouped[sign])
        encoded = encode(tensor, WINDOW_MS, how="mean", t0=0.0) if len(grouped[sign]) else None
        if encoded is None or encoded.shape[0] < 2:
            continue
        for index, order in enumerate(pattern_times(encoded, N_WAKE)):
            patterns.append(order)
            labels.append(f"{name}, pattern {index + 1}")
    return patterns, labels, counts


def match_matrix(waking, rest):
    matrix = np.full((len(waking), len(rest)), np.nan)
    for i, left in enumerate(waking):
        for j, right in enumerate(rest):
            matrix[i, j] = rank_match(left, right)
    return matrix


def null_best(waking, rest, rng):
    """95th percentile of the best absolute correlation after re-pairing cells."""
    if not waking or not rest:
        return np.nan
    scores = []
    for _ in range(NULL_DRAWS):
        order = rng.permutation(waking[0].shape[0])
        best = 0.0
        for left in waking:
            shuffled = left[order]
            for right in rest:
                coefficient = rank_match(shuffled, right)
                if np.isfinite(coefficient) and abs(coefficient) > abs(best):
                    best = abs(coefficient)
        scores.append(best)
    return float(np.percentile(scores, 95))


def analyze(path, rng):
    session = load_session(path)
    item = analyze_session(session)
    cell_ids = np.array(sorted(item["by_cell"]))
    position_t, position_x = load_position(path)
    circular = "ircular" in session["maze"]
    passes = track_passes(position_t, position_x, circular)
    waking, waking_labels, counts = waking_patterns(session, cell_ids, passes)
    rest, rest_labels = rest_patterns(item, cell_ids)
    matrix = match_matrix(waking, rest) if waking and rest else np.zeros((0, 0))
    observed = np.nanmax(np.abs(matrix)) if matrix.size else np.nan
    cutoff = null_best(waking, rest, rng)
    return {
        "name": short_label(path.stem.replace("_sessInfo", "")),
        "maze": session["maze"].strip(),
        "n_passes": counts,
        "waking": waking,
        "waking_labels": waking_labels,
        "rest": rest,
        "rest_labels": rest_labels,
        "matrix": matrix,
        "observed": observed,
        "cutoff": cutoff,
    }


def draw_example(example):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    matrix = example["matrix"]
    ax = axes[0]
    if matrix.size:
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(example["rest_labels"], rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(example["waking_labels"], fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Order correlation, {example['name']}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    if matrix.size and np.isfinite(matrix).any():
        flat = np.nanargmax(np.abs(matrix))
        row, col = np.unravel_index(flat, matrix.shape)
        waking = example["waking"][row]
        rest = example["rest"][col]
        mask = np.isfinite(waking) & np.isfinite(rest)
        ax.scatter(waking[mask], rest[mask], s=16, color=NAVY)
        ax.set_xlabel("Time in the waking pattern (ms)")
        ax.set_ylabel("Time in the rest pattern (ms)")
        ax.set_title(
            f"{example['waking_labels'][row]}\nvs {example['rest_labels'][col]}"
            f"\ncorrelation {matrix[row, col]:.2f}, shuffle 95th {example['cutoff']:.2f}"
        )
    ax.tick_params(length=3)
    fig.suptitle("Same cell order in waking and in rest?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shared_order.png", dpi=200, bbox_inches="tight")
    print(OUT_DIR / "shared_order.png")


def draw_summary(rows):
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(rows))
    width = 0.36
    observed = [row["observed"] for row in rows]
    cutoff = [row["cutoff"] for row in rows]
    ax.bar(x - width / 2, observed, width=width, color=NAVY, label="best match of a waking pattern to a rest pattern")
    ax.bar(x + width / 2, cutoff, width=width, color="#d0d5dd", label="95th percentile if cells are re-paired")
    ax.set_xticks(x)
    ax.set_xticklabels([row["name"] for row in rows], rotation=30, ha="right")
    ax.set_ylabel("Absolute correlation of cell order")
    ax.set_ylim(0, 1)
    ax.set_title("Do waking and rest share a cell order?")
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(length=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shared_summary.png", dpi=200, bbox_inches="tight")
    print(OUT_DIR / "shared_summary.png")


def main():
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(MAT_DIR.glob("*_sessInfo.mat")):
        print("running", path.name, flush=True)
        result = analyze(path, rng)
        rows.append(result)
        print(
            result["name"], result["maze"],
            "passes", result["n_passes"],
            "best", None if np.isnan(result["observed"]) else round(result["observed"], 3),
            "shuffle", None if np.isnan(result["cutoff"]) else round(result["cutoff"], 3),
        )
        if result["matrix"].size:
            print(np.round(result["matrix"], 2))
    example = next(row for row in rows if row["name"].startswith("Achilles 10"))
    draw_example(example)
    draw_summary(rows)


if __name__ == "__main__":
    main()
