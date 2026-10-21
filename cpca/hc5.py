"""Loading the CRCNS hc-5 wheel-running sessions (Pastalkova et al., 2008).

One rat runs a figure-eight maze with a running wheel in the delay area. Each lap
is: start area, one of two arms, then the wheel area, where the rat runs in the
wheel before the next lap's choice. Read from `*_BehavElectrData.mat`:

- wheel runs: `Laps.WhlTimeStartCW` / `WhlTimeEndCW` (first segment of each run);
- arms: laps visiting maze sections 3-5-7-9 are arm A, 4-6-8-10 arm B. The
  `TrialType` codes agree with this in every session (1 and 4 are arm B, 2 and 3
  arm A; 3 and 4 are alternation errors), so the code is used where tracking is
  ambiguous;
- cells: principal cells (`Clu.isIntern == 0`) on shanks 5 and above, which the
  hc-5 channel map places in CA1 (shanks 1-4 are entorhinal cortex).

The wheel run at the end of lap k precedes the choice made on lap k + 1.

Every run is cut to the same window [onset, onset + W] so that all runs share one
clock; W is the 15th percentile of first-segment durations (segments under 10 s
excluded), and shorter runs are dropped. Spikes are also kept from PRE_FRAC * W
before onset, for windows that start before the run (the unaligned condition of
the benchmark); cell selection and time fields use [0, W) only. The answer key is
each cell's time field on the wheel: the peak of its trial-averaged firing rate
against time since wheel onset, computed on runs a method did not see.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hc5"
CACHE_DIR = DATA_DIR / "cache"
FS = 1250.0
ARM_OF_TRIAL_TYPE = {1: "B", 4: "B", 2: "A", 3: "A"}
MIN_SHANK = 5
MIN_SEGMENT_S = 10.0
WINDOW_PERCENTILE = 15
PRE_FRAC = 1 / 3


def session_names(data_dir=DATA_DIR):
    return sorted(p.name.replace("_BehavElectrData.mat", "") for p in Path(data_dir).glob("*_BehavElectrData.mat"))


def load_session(name, data_dir=DATA_DIR, cache_dir=CACHE_DIR, min_spikes=10, min_runs_active=3):
    """Wheel runs and CA1 principal-cell spikes for one session (cached as .npz after the first read).

    Returns a dict: name, W_s (window length), pre_s, onsets_s, durations_s, upcoming and previous
    arm ('A'/'B') per kept run, cell_ids (Clu.totClu), shanks, and `spikes`, a list over runs of
    lists over cells of spike times in seconds since that run's onset, within [-pre_s, W).
    Cells are kept if they fire at least `min_spikes` spikes in total in [0, W) over the runs and
    are active there in at least `min_runs_active` runs.
    """
    cache = Path(cache_dir) / f"{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: (z[k].item() if z[k].shape == () else z[k]) for k in z.files} | {"spikes": list(z["spikes"])}
    import scipy.io as sio

    d = sio.loadmat(Path(data_dir) / f"{name}_BehavElectrData.mat", squeeze_me=True, struct_as_record=False)
    par, spk, clu, laps = d["Par"], d["Spike"], d["Clu"], d["Laps"]
    lap_of_sample = np.asarray(laps.NLap)
    n_laps = int(par.NLap)
    trial_type = np.array([int(laps.TrialType[np.flatnonzero(lap_of_sample == k)[0]]) for k in range(1, n_laps + 1)])
    arm = _arms(laps, lap_of_sample, n_laps, trial_type)
    start = np.atleast_2d(laps.WhlTimeStartCW)
    end = np.atleast_2d(laps.WhlTimeEndCW)
    runs = []
    for k in range(start.shape[0]):
        if start[k, 0] <= 0:
            continue
        lap = int(lap_of_sample[int(start[k, 0])])
        if lap < 1 or lap >= n_laps:
            continue
        runs.append((start[k, 0] / FS, (end[k, 0] - start[k, 0]) / FS, arm[lap], arm[lap - 1], trial_type[lap]))
    durations = np.array([r[1] for r in runs])
    W = float(np.floor(np.percentile(durations[durations >= MIN_SEGMENT_S], WINDOW_PERCENTILE)))
    runs = [r for r in runs if r[1] >= W]
    onsets = np.array([r[0] for r in runs])

    principal = np.flatnonzero((np.asarray(clu.isIntern) == 0) & (np.asarray(clu.shank) >= MIN_SHANK))
    ids = np.asarray(clu.totClu)[principal]
    t = np.asarray(spk.res20kHz, float) / 20000.0 if hasattr(spk, "res20kHz") else np.asarray(spk.res, float) / FS
    tot = np.asarray(spk.totclu)
    order = np.argsort(t)
    t, tot = t[order], tot[order]
    pre = PRE_FRAC * W
    per_run = []
    for on in onsets:
        lo, hi = np.searchsorted(t, on - pre), np.searchsorted(t, on + W)
        tt, cc = t[lo:hi] - on, tot[lo:hi]
        per_run.append([tt[cc == c] for c in ids])
    counts = np.array([[int((s >= 0).sum()) for s in run] for run in per_run])
    keep = (counts.sum(0) >= min_spikes) & ((counts > 0).sum(0) >= min_runs_active)
    spikes = np.empty(len(per_run), dtype=object)
    for i, run in enumerate(per_run):
        cells = np.empty(int(keep.sum()), dtype=object)
        for j, k in enumerate(np.flatnonzero(keep)):
            cells[j] = run[k]
        spikes[i] = cells
    out = {"name": name, "W_s": W, "pre_s": pre, "onsets_s": onsets, "durations_s": np.array([r[1] for r in runs]),
           "upcoming": np.array([r[2] for r in runs]), "previous": np.array([r[3] for r in runs]),
           "trial_type": np.array([r[4] for r in runs]), "cell_ids": ids[keep],
           "shanks": np.asarray(clu.shank)[principal][keep], "n_principal_ca1": int(principal.size)}
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cache, spikes=spikes, **out)
    return out | {"spikes": list(spikes)}


def _arms(laps, lap_of_sample, n_laps, trial_type):
    """Arm ('A' or 'B') per lap (1-based index; index 0 unused) from maze sections, falling back on TrialType."""
    sections = np.asarray(laps.MazeSection)
    arm = [""]
    for k in range(1, n_laps + 1):
        visited = set(np.unique(sections[lap_of_sample == k]).tolist())
        a, b = bool(visited & {3, 5, 7, 9}), bool(visited & {4, 6, 8, 10})
        arm.append("A" if a and not b else "B" if b and not a else ARM_OF_TRIAL_TYPE.get(int(trial_type[k - 1]), "?"))
    return arm


def window_times(session, runs, offsets_s=None, length_s=None, max_spikes=None):
    """Spike times (ms, from each window's start) as (runs, cells, spikes), NaN padded.

    offsets_s: per-run start of the window relative to wheel onset (default 0; negative values
    start before onset, down to -pre_s); length_s: window length (default W - max offset).
    Spikes outside [offset, offset + length) are dropped.
    """
    runs = np.asarray(runs)
    offsets = np.zeros(len(runs)) if offsets_s is None else np.asarray(offsets_s, float)
    length = session["W_s"] - offsets.max() if length_s is None else float(length_s)
    cells = len(session["cell_ids"])
    clipped = [[(s[(s >= o) & (s < o + length)] - o) * 1000.0 for s in session["spikes"][r]] for r, o in zip(runs, offsets)]
    n = max_spikes or max(1, max(len(s) for run in clipped for s in run))
    out = np.full((len(runs), cells, n), np.nan)
    for i, run in enumerate(clipped):
        for k, s in enumerate(run):
            out[i, k, :min(len(s), n)] = s[:n]
    return out, length * 1000.0


def rate_maps(session, runs, bin_s=0.25, smooth_s=0.5):
    """Trial-averaged firing rate (Hz) against time since wheel onset: (cells, bins), and bin centres (s)."""
    edges = np.arange(0, session["W_s"] + 1e-9, bin_s)
    maps = np.zeros((len(session["cell_ids"]), len(edges) - 1))
    for r in runs:
        for k, s in enumerate(session["spikes"][r]):
            maps[k] += np.histogram(s, edges)[0]
    maps /= max(len(runs), 1) * bin_s
    maps = gaussian_filter1d(maps, smooth_s / bin_s, axis=1, mode="nearest")
    return maps, 0.5 * (edges[:-1] + edges[1:])


def time_fields(session, runs, min_peak_hz=2.0, min_active_frac=1 / 3, **kwargs):
    """Answer key: time of each cell's peak rate on the wheel (s); NaN for cells without a field.

    A cell has a field if its trial-averaged peak is at least `min_peak_hz` and it fires in at
    least `min_active_frac` of the given runs.
    """
    maps, centres = rate_maps(session, runs, **kwargs)
    active = np.array([[np.any(session["spikes"][r][k] >= 0) for r in runs] for k in range(len(session["cell_ids"]))]).mean(1)
    peak = centres[np.argmax(maps, axis=1)]
    has = (maps.max(1) >= min_peak_hz) & (active >= min_active_frac)
    return np.where(has, peak, np.nan)
