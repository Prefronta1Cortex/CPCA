"""Loading the processed NWB sessions of DANDI 000552 (Huszar, Zhang, Blockus & Buzsaki, 2022).

Mouse CA1, silicon probes, sorted units; sessions of home-cage sleep around a figure-eight alternation
task (left/right, 5 s delay). Read with h5py from the files in data/huszar/ (scripts/download_huszar.py).
Units are not typed in the files; putative pyramidal cells are taken by rate. The answer key is each
cell's place-field centre on the linearized maze, separately for left-arm and right-arm trials (the
linearization puts both arms on the same coordinates).
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "huszar"
LENGTH_CM = 222.0
ARMS = ("left", "right")


def session_paths(data_dir=DATA_DIR):
    return sorted(Path(data_dir).glob("*_behavior+ecephys.nwb"))


def _strings(values):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_session(path):
    with h5py.File(path, "r") as f:
        idx = f["units/spike_times_index"][:]
        spikes = np.split(f["units/spike_times"][:], idx[:-1])
        ep = f["intervals/epochs"] if "intervals/epochs" in f else None
        ss = f["processing/behavior/SleepStates"]
        labels = _strings(ss["label"][:])
        lp = f["processing/behavior/LinearizedPosition/LinearizedSpatialSeries"]
        xy = f["processing/behavior/SubjectPosition/SpatialSeries"]
        tr = f["intervals/trials"]
        name = Path(path).name.replace("_behavior+ecephys.nwb", "")
        return {
            "name": name, "subject": name.split("_")[0],
            "spikes": [np.sort(s) for s in spikes],
            "epochs": np.column_stack([ep["start_time"][:], ep["stop_time"][:]]) if ep is not None else np.zeros((0, 2)),
            "end": float(max(np.max(s[-1:], initial=0) for s in spikes)),
            "nrem": np.column_stack([ss["start_time"][:], ss["stop_time"][:]])[labels == "Non-REM"],
            "lin_t": lp["timestamps"][:], "lin_x": lp["data"][:],
            "xy_t": xy["timestamps"][:], "xy": xy["data"][:],
            "trials": np.column_stack([tr["start_time"][:], tr["stop_time"][:]]),
            "arm": _strings(tr["visited_arm"][:]),
            "correct": tr["visited_matched_expected"][:].astype(float),
        }


def task_window(session):
    """(PRE window, task window, POST window) as (start, end) pairs.

    From the epoch table when it lists a sleep epoch before and after the epoch of the first trial;
    otherwise (some files list only the first epoch, one lists none) from the trials: the task is the
    span of the trials of the first maze run, PRE everything before it, POST everything after it.
    """
    ep = session["epochs"]
    first = np.nanmin(session["trials"][:, 0])
    if len(ep) >= 3:
        k = int(np.searchsorted(ep[:, 0], first, side="right") - 1)
        if 1 <= k < len(ep) - 1:
            return ep[k - 1], ep[k], ep[k + 1]
    last = np.nanmax(session["trials"][:, 1])
    return np.array([0.0, first]), np.array([first, last]), np.array([last, session["end"]])


def nrem_in(session, window):
    a, b = window
    iv = session["nrem"]
    out = np.column_stack([np.maximum(iv[:, 0], a), np.minimum(iv[:, 1], b)])
    return out[out[:, 1] > out[:, 0]]


def putative_pyramidal(session, max_rate_hz=8.0):
    span = session["end"] - (session["epochs"][0, 0] if len(session["epochs"]) else 0.0)
    rates = np.array([len(s) / span for s in session["spikes"]])
    return np.flatnonzero((rates < max_rate_hz) & (rates > 0))


def task_trials(session, window):
    tr = session["trials"]
    ok = np.all(np.isfinite(tr), axis=1) & (tr[:, 0] >= window[0]) & (tr[:, 1] <= window[1]) & np.isin(session["arm"], ARMS)
    return np.flatnonzero(ok)


def arm_maps(session, cells, window, n_bins=40, min_speed=5.0, min_occupancy_s=0.2, smoothing_bins=1.0):
    """Rate maps (cells, 2, n_bins) in Hz on the linearized maze for left-arm and right-arm trials, spikes per map, bin centres (cm)."""
    t, x = session["lin_t"], session["lin_x"]
    st, xy = session["xy_t"], session["xy"]
    speed = np.r_[0, np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(st), 1e-6)]
    speed = np.interp(t, st, speed)
    edges = np.linspace(0, LENGTH_CM, n_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dt = float(np.median(np.diff(t)))
    trials = task_trials(session, window)
    maps = np.zeros((len(cells), 2, n_bins))
    n_spikes = np.zeros((len(cells), 2), dtype=int)
    for d, arm in enumerate(ARMS):
        sel = np.zeros(t.size, bool)
        for i in trials[session["arm"][trials] == arm]:
            a, b = session["trials"][i]
            sel |= (t >= a) & (t <= b)
        use = sel & (speed > min_speed) & np.isfinite(x)
        occ = np.histogram(x[use], bins=edges)[0] * dt
        valid = occ > min_occupancy_s
        for k, c in enumerate(cells):
            sp = session["spikes"][c]
            sp = sp[(sp >= window[0]) & (sp < window[1])]
            j = np.clip(np.searchsorted(t, sp), 1, t.size - 1)
            j = np.where(np.abs(t[j - 1] - sp) <= np.abs(t[j] - sp), j - 1, j)
            on = use[j] & (np.abs(t[j] - sp) < 0.05)
            n_spikes[k, d] = int(on.sum())
            rate = np.zeros(n_bins)
            rate[valid] = np.histogram(x[j[on]], bins=edges)[0][valid] / occ[valid]
            maps[k, d] = gaussian_filter1d(rate, smoothing_bins, mode="nearest")
    return maps, n_spikes, mid


def place_centres(maps, n_spikes, mid, min_spikes=40, min_peak_hz=3.0):
    """Field centre (cm) per cell for the left-arm (+1.0) and right-arm (-1.0) maps; NaN where no field (hc-11 criteria)."""
    out = {}
    for d, key in enumerate((1.0, -1.0)):
        has = (n_spikes[:, d] >= min_spikes) & (maps[:, d].max(axis=1) >= min_peak_hz)
        out[key] = np.where(has, mid[np.argmax(maps[:, d], axis=1)], np.nan)
    return out
