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


def place_centres(session, cell_ids, run, n_bins=40, min_spikes=40, min_peak_hz=3.0, min_occupancy_s=0.2):
    """Place-field centre (metres) per cell and running direction; NaN where no field. The answer key."""
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
    out = {}
    for sign in (1.0, -1.0):
        occupancy = np.histogram(pos_x[direction == sign], bins=edges)[0] * dt
        valid = occupancy > min_occupancy_s
        centres = np.full(len(cell_ids), np.nan)
        for k, cell in enumerate(cell_ids):
            m = (sid == cell) & (sdir == sign)
            if m.sum() < min_spikes:
                continue
            rate = np.zeros(n_bins)
            rate[valid] = np.histogram(sx[m], bins=edges)[0][valid] / occupancy[valid]
            rate = gaussian_filter1d(rate, 1.0, mode="nearest")
            if rate.max() >= min_peak_hz:
                centres[k] = mid[int(np.argmax(rate))]
        out[sign] = centres
    return out
