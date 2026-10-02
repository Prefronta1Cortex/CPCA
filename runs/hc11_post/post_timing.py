"""POST-only timing templates, pairwise replay check, and a wider event cut.

Reads session mats and imports cpca. Does not modify the recordings or the
window-tuning scripts. Templates follow the plain note: when the top
eigenvector of C is a small arc, read C - C0, keep events with positive
timing excess, and recompute C on those events.
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

from analyze_hc11_window import MAT_DIR, in_intervals, load_session, population_bursts
from cpca import (
    coactivation_null,
    decompose,
    direction_split_null,
    encode,
    readout,
    timing_centered,
    timing_excess,
)
from cpca.stats import shuffle_phases
from cpca.theory import eigh_desc, lag_matrix
from runs.hc11_t100.run_cpca import WINDOW_MS, build_times

OUT = Path(__file__).resolve().parent
NAVY = "#1f4e79"
GREEN = "#1f8a65"
MIN_SPEED = 0.05
MIN_SPIKES = 40
N_BINS = 40
MIN_CELLS = 8
NULL_DRAWS = 80


def short_label(name):
    rat, day = name.split("_")
    return f"{rat}\n{day[:2]}/{day[2:4]}"


def load_position(path):
    with h5py.File(path, "r") as handle:
        return (
            np.array(handle["sessInfo/Position/TimeStamps"]).ravel(),
            np.array(handle["sessInfo/Position/OneDLocation"]).ravel(),
        )


def wide_bursts(spike_times, nrem, bin_s=0.01):
    """Onset at the mean rate, kept only if the peak exceeds mean + 3 SD."""
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
    peak_line = baseline.mean() + 3 * baseline.std()
    edge_line = baseline.mean()
    above = (smooth >= edge_line) & in_nrem
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
        if smooth[begin:finish].max() < peak_line:
            continue
        t0 = centers[begin] - bin_s / 2
        t1 = centers[finish - 1] + bin_s / 2
        if events and t0 - events[-1][1] <= 0.03:
            events[-1][1] = t1
            if smooth[begin:finish].max() > smooth[int((peaks[-1] - start) / bin_s)]:
                peaks[-1] = centers[begin + np.argmax(smooth[begin:finish])]
        else:
            events.append([t0, t1])
            peaks.append(centers[begin + np.argmax(smooth[begin:finish])])
    if not events:
        return np.zeros((0, 2)), np.zeros(0)
    events = np.array(events)
    peaks = np.array(peaks)
    duration = np.diff(events, axis=1).ravel()
    keep = (duration >= 0.05) & (duration <= 0.50)
    return events[keep], peaks[keep]


def post_spikes(session):
    pyr_mask = np.isin(session["ids"], session["pyr"])
    times = session["times"][pyr_mask]
    ids = session["ids"][pyr_mask]
    maze_end = float(session["maze_epoch"][1])
    post = session["nrem"][session["nrem"][:, 0] >= maze_end - 1]
    keep = in_intervals(times, post)
    times = times[keep]
    ids = ids[keep]
    order = np.argsort(times)
    by_cell = {cell: times[order][ids[order] == cell] for cell in session["pyr"]}
    return by_cell, post, times[order]


def running_track(ts, position, circular):
    ok = np.isfinite(position)
    pos_t = ts[ok]
    pos_x = position[ok]
    length = float(np.nanmax(position)) if np.isfinite(position).any() else 1.0
    dt = np.diff(pos_t)
    dx = np.diff(pos_x)
    if circular:
        dx = (dx + length / 2) % length - length / 2
    step = np.zeros(pos_t.shape)
    speed = np.zeros(pos_t.shape)
    step[1:] = dx
    speed[1:] = np.abs(dx) / np.maximum(dt, 1e-6)
    continuous = np.zeros(pos_t.shape, dtype=bool)
    continuous[1:] = dt < 0.05
    running = continuous & (speed >= MIN_SPEED)
    direction = np.sign(step)
    direction[~running] = 0
    return pos_t, pos_x, direction, length


def place_fields(session, cell_ids, pos_t, pos_x, direction, length, circular):
    maze = session["maze_epoch"]
    on_maze = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    spike_t = session["times"][on_maze]
    spike_id = session["ids"][on_maze]
    if pos_t.size == 0:
        return {1.0: {}, -1.0: {}}, length, circular
    index = np.searchsorted(pos_t, spike_t)
    index = np.clip(index, 1, len(pos_t) - 1)
    use_left = np.abs(pos_t[index - 1] - spike_t) <= np.abs(pos_t[index] - spike_t)
    nearest = np.where(use_left, index - 1, index)
    close = np.abs(pos_t[nearest] - spike_t) < 0.04
    keep = close & (direction[nearest] != 0)
    spike_id = spike_id[keep]
    spike_x = pos_x[nearest[keep]]
    spike_dir = direction[nearest[keep]]
    bins = np.linspace(0, length, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fields = {}
    for sign in (1.0, -1.0):
        occupancy = np.histogram(pos_x[direction == sign], bins=bins)[0] * 0.0256
        peaks = {}
        for cell in cell_ids:
            chosen = (spike_id == cell) & (spike_dir == sign)
            if int(chosen.sum()) < MIN_SPIKES:
                continue
            counts = np.histogram(spike_x[chosen], bins=bins)[0]
            rate = np.zeros_like(centers)
            enough = occupancy > 0.2
            rate[enough] = counts[enough] / occupancy[enough]
            if rate.max() <= 0 or rate.max() < 3 * np.mean(rate[enough]):
                continue
            peaks[cell] = float(centers[np.argmax(rate)])
        fields[sign] = peaks
    return fields, length, circular


def encode_windows(by_cell, cell_ids, peaks):
    bursts = np.column_stack([peaks - WINDOW_MS / 2000, peaks + WINDOW_MS / 2000])
    times, _, _ = build_times(by_cell, bursts, peaks, cell_ids)
    active = np.isfinite(times).any(axis=-1)
    encoded = encode(times, WINDOW_MS, how="mean", t0=0.0)
    return encoded, active


def rank_match(times, places, length, circular):
    mask = np.isfinite(times) & np.isfinite(places)
    if int(mask.sum()) < MIN_CELLS:
        return np.nan
    if not circular:
        coefficient, _ = spearmanr(times[mask], places[mask])
        return float(coefficient)
    best = 0.0
    for shift in range(18):
        moved = (places[mask] + shift * length / 18) % length
        coefficient, _ = spearmanr(times[mask], moved)
        if np.isfinite(coefficient) and abs(coefficient) > abs(best):
            best = float(coefficient)
    return best


def field_vector(peaks, cell_ids):
    place = np.full(len(cell_ids), np.nan)
    index = {cell: i for i, cell in enumerate(cell_ids)}
    for cell, peak in peaks.items():
        place[index[cell]] = peak
    return place


def templates(encoded, active):
    _, components, _, lag = decompose(encoded, n_components=1)
    from_c = components[:, 0]
    delta_matrix = timing_centered(lag, coactivation_null(encoded, active))
    _, delta_vectors = eigh_desc(delta_matrix)
    from_delta = delta_vectors[:, 0]
    excess = timing_excess(encoded, from_delta, active)
    keep = excess > 0
    if int(keep.sum()) >= 2:
        _, selected, _, _ = decompose(encoded[keep], n_components=1)
        from_selected = selected[:, 0]
    else:
        from_selected = np.full(encoded.shape[1], np.nan, dtype=complex)
    return from_c, from_delta, from_selected, int(keep.sum()), len(excess)


def template_time(component):
    if not np.isfinite(component).all():
        return np.full(component.shape, np.nan)
    _, order, _, _, span = readout(component, WINDOW_MS, min_weight=0.1)
    return order, span


def pairwise(encoded, active, place, circular, length, rng):
    cells = np.flatnonzero(np.isfinite(place))
    if cells.size < MIN_CELLS:
        return np.nan, np.nan, 0
    lag = lag_matrix(encoded)
    shared = active.T.astype(float) @ active.astype(float)

    def agreement(matrix):
        same = 0
        total = 0
        for a in range(cells.size):
            for b in range(a + 1, cells.size):
                i = cells[a]
                j = cells[b]
                if shared[i, j] < 20:
                    continue
                lag_sign = np.sign(np.angle(matrix[i, j]))
                gap = place[i] - place[j]
                if circular:
                    gap = (gap + length / 2) % length - length / 2
                place_sign = np.sign(gap)
                if lag_sign == 0 or place_sign == 0:
                    continue
                total += 1
                same += int(lag_sign == place_sign)
        if total == 0:
            return np.nan
        return 100 * same / total

    observed = agreement(lag)
    nulls = []
    for _ in range(30):
        nulls.append(agreement(lag_matrix(shuffle_phases(encoded, rng))))
    nulls = np.array(nulls, dtype=float)
    if not np.isfinite(observed) or not np.isfinite(nulls).any():
        return observed, np.nan, 0
    z = (observed - np.nanmean(nulls)) / max(np.nanstd(nulls), 1e-6)
    return observed, float(z), int(np.isfinite(nulls).sum())


def bayes_replay(encoded_times, peaks_place, cell_ids, pos_x, direction, length, sign):
    """Weighted correlation of a decoded trajectory. Returns scores per event."""
    bins = np.linspace(0, length, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    occupancy = np.histogram(pos_x[direction == sign], bins=bins)[0].astype(float)
    occupancy = np.maximum(occupancy, 1.0)
    rate = np.zeros((len(cell_ids), N_BINS))
    index = {cell: i for i, cell in enumerate(cell_ids)}
    # encoded_times: (events, cells, spikes) in ms from window start
    used = [cell for cell in peaks_place if cell in index]
    if len(used) < MIN_CELLS:
        return np.array([])
    # rebuild rates from the stored peaks only as a flat field is too weak;
    # caller passes rate rows. Here peaks_place is the peak dict; use a
    # Gaussian bump so the decoder has a field shape.
    for cell, peak in peaks_place.items():
        rate[index[cell]] = np.exp(-0.5 * ((centers - peak) / (0.08 * length)) ** 2)
    scores = []
    edges = np.linspace(0, WINDOW_MS, 6)
    for event in encoded_times:
        counts = np.zeros((5, len(cell_ids)))
        for cell in range(len(cell_ids)):
            stamps = event[cell]
            stamps = stamps[np.isfinite(stamps)]
            if stamps.size == 0:
                continue
            counts[:, cell] = np.histogram(stamps, bins=edges)[0]
        if counts.sum() < MIN_CELLS:
            scores.append(np.nan)
            continue
        log_rate = np.log(rate + 1e-3)
        log_post = np.log(occupancy) + counts @ log_rate - 0.02 * rate.sum(axis=0)
        log_post -= log_post.max(axis=1, keepdims=True)
        posterior = np.exp(log_post)
        posterior /= posterior.sum(axis=1, keepdims=True)
        scores.append(weighted_correlation(posterior, centers))
    return np.array(scores)


def weighted_correlation(posterior, centers):
    times = np.arange(posterior.shape[0])[:, None]
    position = centers[None, :]
    weight = posterior
    total = weight.sum()
    if total <= 0:
        return np.nan
    time_mean = np.sum(weight * times) / total
    pos_mean = np.sum(weight * position) / total
    covariance = np.sum(weight * (times - time_mean) * (position - pos_mean))
    time_var = np.sum(weight * (times - time_mean) ** 2)
    pos_var = np.sum(weight * (position - pos_mean) ** 2)
    if time_var <= 0 or pos_var <= 0:
        return np.nan
    return float(covariance / np.sqrt(time_var * pos_var))


def one_detector(by_cell, cell_ids, bursts, peaks, fields, length, circular, rng):
    if len(peaks) < 20:
        return None
    encoded, active = encode_windows(by_cell, cell_ids, peaks)
    from_c, from_delta, from_selected, n_kept, n_events = templates(encoded, active)
    times = {}
    spans = {}
    for name, component in (("C", from_c), ("CΔ", from_delta), ("selected", from_selected)):
        order, span = template_time(component)
        times[name] = order
        spans[name] = span
    spearman = {name: {} for name in times}
    for sign, peaks_m in fields.items():
        place = field_vector(peaks_m, cell_ids)
        for name, order in times.items():
            spearman[name][sign] = rank_match(order, place, length, circular)
    pairwise_z = {}
    for sign, peaks_m in fields.items():
        _, z, _ = pairwise(encoded, active, field_vector(peaks_m, cell_ids), circular, length, rng)
        pairwise_z[sign] = z
    observed, nulls = direction_split_null(encoded, rng, n_draws=12)
    nulls = nulls[np.isfinite(nulls)]
    split_inside = bool(nulls.size and np.isfinite(observed) and np.nanpercentile(nulls, 5) <= observed <= np.nanpercentile(nulls, 95))
    return {
        "n": n_events,
        "kept": n_kept,
        "span": spans,
        "spearman": spearman,
        "pairwise_z": pairwise_z,
        "split": observed,
        "split_inside_null": split_inside,
    }


def rank_match(times, places, length, circular):
    return _rank(times, places, length, circular)


def _rank(times, places, length, circular):
    mask = np.isfinite(times) & np.isfinite(places)
    if int(mask.sum()) < MIN_CELLS:
        return np.nan
    if not circular:
        coefficient, _ = spearmanr(times[mask], places[mask])
        return float(coefficient)
    best = 0.0
    for shift in range(12):
        moved = (places[mask] + shift * length / 12) % length
        coefficient, _ = spearmanr(times[mask], moved)
        if np.isfinite(coefficient) and abs(coefficient) > abs(best):
            best = float(coefficient)
    return best


def analyze_session_post(path, rng):
    session = load_session(path)
    by_cell, post_nrem, spike_times = post_spikes(session)
    cell_ids = np.array(sorted(by_cell))
    strict_b, strict_p = population_bursts(spike_times, post_nrem)
    wide_b, wide_p = wide_bursts(spike_times, post_nrem)
    pos_t, pos_x = load_position(path)
    circular = "ircular" in session["maze"]
    pos_t, pos_x, direction, length = running_track(pos_t, pos_x, circular)
    fields, length, circular = place_fields(session, cell_ids, pos_t, pos_x, direction, length, circular)
    # place_fields already returned fields; running_track consumed position.
    strict = one_detector(by_cell, cell_ids, strict_b, strict_p, fields, length, circular, rng)
    wide = one_detector(by_cell, cell_ids, wide_b, wide_p, fields, length, circular, rng)
    return {
        "name": short_label(path.stem.replace("_sessInfo", "")),
        "fields": {sign: len(peaks) for sign, peaks in fields.items()},
        "strict": strict,
        "wide": wide,
    }


def bayes_on_achilles(path, rng):
    """Upper bound: keep POST events whose decoded path beats a time shuffle."""
    session = load_session(path)
    if "ircular" in session["maze"] or "Linear" not in session["maze"]:
        return None
    by_cell, post_nrem, spike_times = post_spikes(session)
    cell_ids = np.array(sorted(by_cell))
    _, peaks = wide_bursts(spike_times, post_nrem)
    if len(peaks) < 30:
        return None
    pos_t, pos_x = load_position(path)
    pos_t, pos_x, direction, length = running_track(pos_t, pos_x, False)
    fields, length, _ = place_fields(session, cell_ids, pos_t, pos_x, direction, length, False)
    sign = max(fields, key=lambda item: len(fields[item]))
    bursts = np.column_stack([peaks - WINDOW_MS / 2000, peaks + WINDOW_MS / 2000])
    times, _, _ = build_times(by_cell, bursts, peaks, cell_ids)
    scores = bayes_replay(times, fields[sign], cell_ids, pos_x, direction, length, sign)
    null = np.empty((len(scores), 12))
    for draw in range(12):
        rolled = np.array(times, copy=True)
        for event in rolled:
            shift = int(rng.integers(1, 5))
            finite = np.isfinite(event)
            # shift spike times by one bin width, wrapping inside the window
            event[finite] = (event[finite] + shift * 20) % WINDOW_MS
        null[:, draw] = bayes_replay(rolled, fields[sign], cell_ids, pos_x, direction, length, sign)
    cutoff = np.nanpercentile(np.abs(null), 95, axis=1)
    keep = np.isfinite(scores) & (np.abs(scores) > cutoff) & (np.abs(scores) > 0.5)
    if int(keep.sum()) < 20:
        return {"kept": int(keep.sum()), "spearman": np.nan}
    place = field_vector(fields[sign], cell_ids)
    out = {"kept": int(keep.sum()), "of": int(len(scores))}
    for label, mask in (("forward", scores > 0), ("reverse", scores < 0)):
        chosen = keep & mask
        if int(chosen.sum()) < 20:
            out[label] = np.nan
            continue
        encoded = encode(times[chosen], WINDOW_MS, how="mean", t0=0.0)
        active = np.isfinite(times[chosen]).any(axis=-1)
        _, from_delta, _, _, _ = templates(encoded, active)
        order, _ = template_time(from_delta)
        out[label] = rank_match(order, place, length, False)
        out[label + "_n"] = int(chosen.sum())
    return out


def synthetic(rng):
    n_cells = 80
    n_events = 1500
    order = np.arange(n_cells) * (80.0 / (n_cells - 1))
    rows = []
    for event in range(n_events):
        carry = rng.random() < 0.40
        reverse = carry and rng.random() < 0.5
        times = np.full((n_cells, 1), np.nan)
        for cell in range(n_cells):
            if rng.random() > 0.15:
                continue
            if carry:
                base = order[cell] if not reverse else order[-1] - order[cell]
                times[cell, 0] = np.clip(base + rng.normal(0, 8), 0, 100)
            else:
                times[cell, 0] = np.clip(rng.normal(50, 8), 0, 100)
        rows.append(times[:, 0])
    encoded_times = np.array(rows)[:, :, None]
    active = np.isfinite(encoded_times[:, :, 0])
    encoded = encode(encoded_times, 100.0, how="mean", t0=0.0)
    from_c, from_delta, from_selected, n_kept, _ = templates(encoded, active)
    truth = order

    def score(component):
        order_time, _ = template_time(component)
        mask = np.isfinite(order_time)
        if int(mask.sum()) < MIN_CELLS:
            return np.nan
        coefficient, _ = spearmanr(truth[mask], order_time[mask])
        return float(coefficient)

    return {
        "C": score(from_c),
        "CΔ": score(from_delta),
        "selected": score(from_selected),
        "kept": n_kept,
        "events": n_events,
    }


def draw(rows, sim):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    labels = [row["name"] for row in rows]
    x = np.arange(len(rows))

    ax = axes[0]
    strict_z = [np.nanmax(np.abs(list(row["strict"]["pairwise_z"].values()))) if row["strict"] else np.nan for row in rows]
    wide_z = [np.nanmax(np.abs(list(row["wide"]["pairwise_z"].values()))) if row["wide"] else np.nan for row in rows]
    ax.bar(x - 0.18, strict_z, width=0.36, color=NAVY, label="crest only, mean + 3 SD")
    ax.bar(x + 0.18, wide_z, width=0.36, color=GREEN, label="edges at the mean, peak at 3 SD")
    ax.axhline(1.96, color="#b42318", lw=1, label="z = 1.96")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Pairwise replay z, larger direction")
    ax.set_title("Is there replay timing to extract?")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    width = 0.25
    for offset, key, color, label in (
        (-width, "C", "#98a8b8", "template from C"),
        (0, "CΔ", NAVY, "template from C − C0"),
        (width, "selected", GREEN, "C on events with excess > 0"),
    ):
        values = []
        for row in rows:
            block = row["wide"]["spearman"] if row["wide"] else {}
            pair = block.get(key, {})
            if not pair:
                values.append(np.nan)
            else:
                values.append(np.nanmax(np.abs(list(pair.values()))))
        ax.bar(x + offset, values, width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Absolute correlation with place")
    ax.set_ylim(0, 1)
    ax.set_title("POST, wider events")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.bar(np.arange(3), [abs(sim["C"]), abs(sim["CΔ"]), abs(sim["selected"])], color=["#98a8b8", NAVY, GREEN])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["C", "C − C0", "C after\nexcess > 0"])
    ax.set_ylim(0, 1)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_ylabel("Correlation with the planted order")
    ax.set_title("Synthetic: 15% firing, 40% sequenced")
    fig.tight_layout()
    fig.savefig(OUT / "post_timing.png", dpi=200, bbox_inches="tight")
    print(OUT / "post_timing.png")


def main():
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(MAT_DIR.glob("*_sessInfo.mat")):
        print("running", path.name, flush=True)
        result = analyze_session_post(path, rng)
        rows.append(result)
        print(result["name"].replace("\n", " "), "fields", result["fields"])
        for kind in ("strict", "wide"):
            block = result[kind]
            if block is None:
                print(" ", kind, "none")
                continue
            print(
                f"  {kind} n={block['n']} kept={block['kept']} "
                f"spans { {k: round(v, 2) for k, v in block['span'].items()} }"
            )
            print("   spearman", {k: {s: None if not np.isfinite(r) else round(r, 3) for s, r in v.items()} for k, v in block["spearman"].items()})
            print("   pairwise z", {s: None if not np.isfinite(z) else round(z, 2) for s, z in block["pairwise_z"].items()})
            print("   split", None if not np.isfinite(block["split"]) else round(block["split"], 3), "inside null", block["split_inside_null"])
    sim = synthetic(rng)
    print("synthetic", {k: round(v, 3) if isinstance(v, float) else v for k, v in sim.items()})
    achilles = next(path for path in sorted(MAT_DIR.glob("*_sessInfo.mat")) if "Achilles_1025" in path.name)
    upper = bayes_on_achilles(achilles, rng)
    print("bayes upper bound", upper)
    draw(rows, sim)


if __name__ == "__main__":
    main()
