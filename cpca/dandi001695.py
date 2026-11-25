"""Loading DANDI 001695 (Gonzalez & Voroslakos 2026): mouse Neuropixels 2.0 recordings of CA1, CA3 and
retrosplenial cortex, with rest/sleep before a novel linear track, the track, and rest/sleep after it.

The sessions with behaviour ("*_behavior+ecephys.nwb") are read into the same dictionary as cpca.hc11.load_session,
so the hc-11 code (running, place_centres, place_maps, nrem_intervals, pyramidal_spikes) applies unchanged:
  maze epoch   the span of the position tracking (position is recorded only on the track);
  pre / post   everything before / after it;
  nrem         the NREM intervals of the provided sleep states;
  pos_x        position along the track (the x coordinate; the track runs along x, about 1.1 m) in metres from
               its smallest value; pos_xy the 2-D position in metres (one file stores metres, the others
               centimetres; the unit is taken from the span);
  pyr          the cells used: CA1 pyramidal cells (areas="CA1"), or CA1 and CA3 pyramidal cells;
  ripples      (n, 2) start/stop of the provided sharp-wave ripples.
"""

from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dandi001695"


def session_paths(data_dir=DATA_DIR):
    return sorted(Path(data_dir).glob("*_behavior+ecephys.nwb"))


def _strings(d):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in d[:]])


def load_session(path, areas=("CA1",)):
    with h5py.File(path, "r") as f:
        u = f["units"]
        idx = u["spike_times_index"][:]
        st = u["spike_times"][:]
        counts = np.diff(np.r_[0, idx])
        ids = np.repeat(np.arange(len(idx)), counts)
        area, ctype = _strings(u["cell_area"]), _strings(u["cell_type"])
        pyr = np.flatnonzero(np.isin(area, areas) & np.char.startswith(ctype, "Pyramidal"))
        pos = f["processing/behavior/AnimalPosition/Position"]
        xy, t = pos["data"][:], pos["timestamps"][:]
        states = f["intervals/SleepStates"]
        name = _strings(states["state"])
        a, b = states["start_time"][:], states["stop_time"][:]
        stop = float(st.max())
    scale = 1.0 if np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0]) < 5 else 0.01
    xy = xy * scale
    order = np.argsort(st, kind="stable")
    return {
        "name": Path(path).name.split("_behavior")[0],
        "maze": "Linear track",
        "pre": np.array([0.0, t[0]]),
        "maze_epoch": np.array([t[0], t[-1]]),
        "post": np.array([t[-1], stop]),
        "nrem": np.column_stack([a[name == "NREM"], b[name == "NREM"]]),
        "ripples": np.column_stack([a[name == "Ripple"], b[name == "Ripple"]]),
        "times": st[order],
        "ids": ids[order],
        "pyr": pyr,
        "interneurons": np.flatnonzero(~np.char.startswith(ctype, "Pyramidal")),
        "area": area,
        "pos_t": t,
        "pos_x": xy[:, 0] - np.nanmin(xy[:, 0]),
        "pos_xy": xy.T.copy(),
    }
