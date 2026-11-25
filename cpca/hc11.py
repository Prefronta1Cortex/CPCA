"""Loading the CRCNS hc-11 session files (Grosmark & Buzsaki, 2016).

Reads the `*_sessInfo.mat` files (MATLAB v7.3 / HDF5): spike times and ids,
pyramidal/interneuron lists, epoch boundaries, sleep states, and position on
the maze. Also computes the answer key: each cell's place-field centre per
running direction, from the maze epoch only.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from cpca.events import in_intervals

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hc11_sessinfo" / "NoveltySessInfoMatFiles"


def session_paths(data_dir=DATA_DIR):
    return sorted(Path(data_dir).glob("*_sessInfo.mat"))


def load_session(path):
    with h5py.File(path, "r") as handle:
        epochs = handle["sessInfo/Epochs"]
        spikes = handle["sessInfo/Spikes"]
        position = handle["sessInfo/Position"]

        def interval(name):
            values = np.array(epochs[name])
            if values.shape[0] != 2:
                values = values.T
            return values.T.copy()

        return {
            "name": Path(path).name.replace("_sessInfo.mat", ""),
            "maze": "".join(chr(c) for c in np.array(position["MazeType"]).ravel()).strip(),
            "pre": interval("PREEpoch").ravel(),
            "maze_epoch": interval("MazeEpoch").ravel(),
            "post": interval("POSTEpoch").ravel(),
            "nrem": interval("NREM"),
            "times": np.array(spikes["SpikeTimes"]).ravel(),
            "ids": np.array(spikes["SpikeIDs"]).ravel(),
            "pyr": np.array(spikes["PyrIDs"]).ravel(),
            "interneurons": np.array(spikes["IntIDs"]).ravel(),
            "pos_t": np.array(position["TimeStamps"]).ravel(),
            "pos_x": np.array(position["OneDLocation"]).ravel(),
        }


def nrem_intervals(session, epoch="post"):
    """NREM intervals inside the PRE or POST epoch."""
    maze = session["maze_epoch"]
    nrem = session["nrem"]
    if epoch == "post":
        return nrem[nrem[:, 0] >= maze[1] - 1]
    if epoch == "pre":
        return nrem[nrem[:, 1] <= maze[0] + 1]
    raise ValueError("epoch must be 'pre' or 'post'")


def pyramidal_spikes(session, intervals):
    """Spike times of pyramidal cells inside the intervals: (by_cell dict, cell ids, pooled sorted times)."""
    mask = np.isin(session["ids"], session["pyr"]) & in_intervals(session["times"], intervals)
    times, ids = session["times"][mask], session["ids"][mask]
    order = np.argsort(times)
    times, ids = times[order], ids[order]
    cell_ids = np.array(sorted(session["pyr"]))
    by_cell = {cell: times[ids == cell] for cell in cell_ids}
    return by_cell, cell_ids, times


def running(session, min_speed=0.05, max_gap_s=0.05):
    """Tracked position samples with a running direction (+1, -1, or 0 when still or untracked)."""
    ok = np.isfinite(session["pos_x"])
    pos_t, pos_x = session["pos_t"][ok], session["pos_x"][ok]
    circular = "ircular" in session["maze"]
    length = float(np.nanmax(pos_x)) if pos_x.size else 1.0
    dt = np.diff(pos_t)
    dx = np.diff(pos_x)
    if circular:
        dx = (dx + length / 2) % length - length / 2
    step = np.zeros(pos_t.shape)
    speed = np.zeros(pos_t.shape)
    step[1:] = dx
    speed[1:] = np.abs(dx) / np.maximum(dt, 1e-6)
    continuous = np.zeros(pos_t.shape, dtype=bool)
    continuous[1:] = dt < max_gap_s
    direction = np.sign(step)
    direction[~(continuous & (speed >= min_speed))] = 0
    return {"t": pos_t, "x": pos_x, "direction": direction, "length": length, "circular": circular}


def laps(run, min_cover=0.6, max_pause_s=0.5):
    """Waking events: (start, end, direction) of continuous runs in one direction (pauses up to max_pause_s bridged)
    that travel at least min_cover of the track; on circular tracks a longer run is cut into pieces of one track length.

    On a linear track these are the laps from one end to the other.
    """
    t, x, d, L = run["t"], run["x"], run["direction"], run["length"]
    moving = np.flatnonzero(d != 0)
    if moving.size == 0:
        return []
    segments = []
    start = last = moving[0]
    for k in moving[1:]:
        if d[k] == d[last] and t[k] - t[last] <= max_pause_s:
            last = k
            continue
        segments.append((start, last))
        start = last = k
    segments.append((start, last))
    out = []
    for a, b in segments:
        dx = np.diff(x[a:b + 1])
        if run["circular"]:
            dx = (dx + L / 2) % L - L / 2
        travelled = np.concatenate([[0.0], np.cumsum(np.abs(dx))])
        if travelled[-1] < min_cover * L:
            continue
        cuts = [0]
        if run["circular"]:
            for m in range(1, int(travelled[-1] // L) + 1):
                cuts.append(int(np.searchsorted(travelled, m * L)))
        cuts.append(len(travelled) - 1)
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            if travelled[i1] - travelled[i0] >= min_cover * L:
                out.append((float(t[a + i0]), float(t[a + i1]), float(d[a])))
    return out


def place_centres(session, cell_ids, run, n_bins=40, min_spikes=40, min_peak_hz=3.0, min_occupancy_s=0.2):
    """Place-field centre (metres) per cell and running direction; NaN where no field. The answer key."""
    maps, n_spikes, mid = place_maps(session, cell_ids, run, n_bins=n_bins, min_occupancy_s=min_occupancy_s, smoothing_mode="nearest")
    out = {}
    for d, sign in enumerate((1.0, -1.0)):
        has = (n_spikes[:, d] >= min_spikes) & (maps[:, d].max(axis=1) >= min_peak_hz)
        out[sign] = np.where(has, mid[np.argmax(maps[:, d], axis=1)], np.nan)
    return out


def place_maps(session, cell_ids, run, n_bins=40, min_occupancy_s=0.2, smoothing_bins=1.0, smoothing_mode=None):
    """Direction-resolved rate maps from the maze epoch: (cells, 2, n_bins) in Hz, direction +1 first.

    Also returns the number of spikes per cell and direction (cells, 2) and the bin centres (m).
    Bins visited for less than min_occupancy_s are 0. Smoothing wraps on circular tracks unless
    smoothing_mode is given.
    """
    maze = session["maze_epoch"]
    on = (session["times"] >= maze[0]) & (session["times"] < maze[1])
    st, sid = session["times"][on], session["ids"][on]
    pos_t, pos_x, direction, length = run["t"], run["x"], run["direction"], run["length"]
    j = np.clip(np.searchsorted(pos_t, st), 1, pos_t.size - 1)
    j = np.where(np.abs(pos_t[j - 1] - st) <= np.abs(pos_t[j] - st), j - 1, j)
    keep = (np.abs(pos_t[j] - st) < 0.04) & (direction[j] != 0)
    sid, sx, sdir = sid[keep], pos_x[j[keep]], direction[j[keep]]
    dt = float(np.median(np.diff(pos_t)))
    edges = np.linspace(0, length, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    mode = smoothing_mode or ("wrap" if run["circular"] else "nearest")
    maps = np.zeros((len(cell_ids), 2, n_bins))
    n_spikes = np.zeros((len(cell_ids), 2), dtype=int)
    for d, sign in enumerate((1.0, -1.0)):
        occupancy = np.histogram(pos_x[direction == sign], bins=edges)[0] * dt
        valid = occupancy > min_occupancy_s
        for k, cell in enumerate(cell_ids):
            m = (sid == cell) & (sdir == sign)
            n_spikes[k, d] = int(m.sum())
            rate = np.zeros(n_bins)
            rate[valid] = np.histogram(sx[m], bins=edges)[0][valid] / occupancy[valid]
            maps[k, d] = gaussian_filter1d(rate, smoothing_bins, mode=mode)
    return maps, n_spikes, mid
