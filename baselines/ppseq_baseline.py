"""PP-Seq (Williams, Degleris, Wang & Linderman, NeurIPS 2020), run through the authors' Julia package.

Data preparation. PP-Seq models a continuous recording. The population-burst events
are laid end to end with a silent gap between them, and time is expressed in units
of 50 ms (one unit is roughly one replay-compressed traversal step, which puts the
authors' hippocampal prior ranges, written for running-speed sequences, at a
comparable scale). Spike times keep their position inside each event.

Hyperparameters. Following Supplement F.2 of the paper: a random sweep over the
number of sequence types R, the mean sequence amplitude (variance = mean), the
background amplitude (variance = mean), the sequence rate, and the neuron response
width, each setting scored by the held-out log-likelihood of a speckled 7.5% mask;
the best setting is then re-fit on all spikes. The sweep is much smaller than the
authors' 2,000 runs (SWEEP_SIZE settings) and the MCMC schedule is shortened
(stated below); both are limits of a laptop, not of the method.

Scoring. Each sequence type r gives a cell order: the neuron offsets b_nr, with
members the neurons whose response proportion a_nr is at least 20% of the type's
largest (the threshold the package's own `sortperm_neurons` uses). Each saved
post-anneal sample gives latent events; an event window "contains a sequence" in
the fraction of samples in which a latent event falls inside it. This is the
per-event detection score (AUC against planted replays on synthetic data; fraction
of events with posterior probability above 0.5 on real data).
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JULIA_PROJECT = HERE / "ppseq"
JULIA = Path.home() / ".juliaup" / "bin" / "julia"
UNIT_MS = 50.0
GAP_UNITS = 2.0
SWEEP_SIZE = 4
MEMBER_FRAC = 0.2
MAX_EVENTS = 1500          # random subset of events for the final fit (same cap as seqNMF); run time grows faster than linearly
SWEEP_EVENTS = 600         # smaller subset for the held-out hyperparameter sweep (the masked sampler runs single-threaded)
THREADS = 4                # the package's DistributedSeqModel splits the timeline across threads
MIN_SPREAD_MS = 20.0       # a sequence type whose member offsets span less than this is a synchronous volley

# shorter MCMC schedule for the sweep runs only
SWEEP_SCHEDULE = {"num_anneals": 6, "samples_per_anneal": 15, "save_every_during_anneal": 15,
                  "num_spike_resamples_per_anneal": 3, "num_spike_resamples": 6, "samples_per_resample": 4}

# MCMC schedule (authors: 20 anneal stages x 100 sweeps from temperature 500, then 100 sweeps with
# 1000 split-merge moves each). Here: 8 x 25 sweeps from temperature 100, then 40 sweeps, 10 split-merge moves.
BASE_CONFIG = {
    "seq_type_conc_param": 3.0,            # gamma = 3 (paper)
    "neuron_response_conc_param": 1.0,     # phi = 1 (paper)
    "neuron_offset_pseudo_obs": 1.0,
    "neuron_width_pseudo_obs": 4.0,        # nu = 4 (paper)
    "num_warp_values": 1, "max_warp": 1.0, "warp_variance": 1.0,   # no time warping (no benefit on hippocampal data in the paper)
    "bkgd_spikes_conc_param": 0.3,
    "max_sequence_length": "Inf",
    "num_anneals": 8, "samples_per_anneal": 25, "max_temperature": 100.0, "save_every_during_anneal": 25,
    "samples_after_anneal": 40, "save_every_after_anneal": 2,
    "split_merge_moves_during_anneal": 10, "split_merge_moves_after_anneal": 10, "split_merge_window": 1.0,
    # masked validation (speckled hold-out, as in the paper)
    "mask_lengths": 1.0, "percent_masked": 7.5,
    "num_spike_resamples_per_anneal": 5, "num_spike_resamples": 10, "samples_per_resample": 4,
}


def build_timeline(times_ms, T_ms, unit_ms=UNIT_MS, gap_units=GAP_UNITS):
    """Concatenate events. Returns (spikes [neuron 1-based, time], event starts (units), event ends, max_time)."""
    n_events, n_cells, _ = times_ms.shape
    T_ms = np.broadcast_to(np.asarray(T_ms, float), (n_events,))
    starts = np.empty(n_events)
    rows = []
    t = gap_units
    for i in range(n_events):
        starts[i] = t
        for k in range(n_cells):
            s = times_ms[i, k]
            s = s[np.isfinite(s)]
            for v in s:
                rows.append((k + 1, t + v / unit_ms))
        t += T_ms[i] / unit_ms + gap_units
    ends = starts + T_ms / unit_ms
    return np.array(rows), starts, ends, float(t)


def _run_julia(spikes, n_cells, max_time, config, masked, workdir):
    sp = Path(workdir) / "spikes.tsv"
    np.savetxt(sp, spikes, fmt=["%d", "%.6f"], delimiter="\t")
    cp = Path(workdir) / "config.json"
    cp.write_text(json.dumps(config))
    out = Path(workdir) / ("masked.json" if masked else "full.json")
    cmd = [str(JULIA), f"-t{THREADS}", f"--project={JULIA_PROJECT}", str(HERE / "ppseq" / "run_ppseq.jl"), str(sp), str(n_cells), f"{max_time:.6f}", str(cp), str(out)]
    if masked:
        cmd.append("masked")
    env = dict(os.environ, PATH=f"{JULIA.parent}:{os.environ.get('PATH', '')}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"PP-Seq failed:\n{proc.stderr[-3000:]}")
    return json.loads(out.read_text())


def sample_settings(rng, n, spikes_per_unit, mean_spikes_per_event, R_options=(2, 3, 4)):
    """Random hyperparameter settings in the spirit of the paper's sweep, scaled to these data."""
    settings = []
    for _ in range(n):
        settings.append({
            "num_sequence_types": int(rng.choice(R_options)),
            "mean_event_amplitude": float(10 ** rng.uniform(np.log10(0.3 * mean_spikes_per_event), np.log10(3 * mean_spikes_per_event))),
            "mean_bkgd_spike_rate": float(10 ** rng.uniform(np.log10(0.2 * spikes_per_unit), np.log10(2 * spikes_per_unit))),
            "seq_event_rate": float(10 ** rng.uniform(-2, -0.5)),
            "neuron_width_prior": float(10 ** rng.uniform(-2, 0)),
        })
    return settings


def fit(times_ms, T_ms, rng, sweep_size=SWEEP_SIZE, R_options=(2, 3, 4), max_events=MAX_EVENTS, verbose=True):
    """Sweep hyperparameters by held-out likelihood, refit the best, return parsed results.

    Returns per-type orders (lag_ms over all cells, NaN for non-members), per-event posterior
    probability of containing a latent sequence (NaN for events not in the fitted subset), the
    sweep table and the chosen setting.
    """
    t_all = time.perf_counter()
    n_all = times_ms.shape[0]
    T_all = np.broadcast_to(np.asarray(T_ms, float), (n_all,))
    subset = np.sort(rng.choice(n_all, max_events, replace=False)) if n_all > max_events else np.arange(n_all)
    times_ms, T_ms = times_ms[subset], T_all[subset]
    spikes, starts, ends, max_time = build_timeline(times_ms, T_ms)
    n_cells = times_ms.shape[1]
    spikes_per_unit = spikes.shape[0] / max_time
    active_per_event = np.isfinite(times_ms).sum(axis=(1, 2))
    mean_spikes_per_event = float(active_per_event[active_per_event > 0].mean())
    settings = sample_settings(rng, sweep_size, spikes_per_unit, mean_spikes_per_event, R_options)
    sweep_idx = np.sort(rng.choice(len(subset), SWEEP_EVENTS, replace=False)) if len(subset) > SWEEP_EVENTS else np.arange(len(subset))
    sw_spikes, _, _, sw_max_time = build_timeline(times_ms[sweep_idx], T_ms[sweep_idx])
    sweep = []
    with tempfile.TemporaryDirectory() as wd:
        for i, s in enumerate(settings):
            cfg = dict(BASE_CONFIG, **SWEEP_SCHEDULE, **s, var_event_amplitude=s["mean_event_amplitude"], var_bkgd_spike_rate=s["mean_bkgd_spike_rate"],
                       seed=int(rng.integers(1 << 30)), num_threads=THREADS)
            try:
                res = _run_julia(sw_spikes, n_cells, sw_max_time, cfg, True, wd)
                score = res["heldout_log_like_per_spike"]
            except RuntimeError as err:
                score, res = -np.inf, {"error": str(err)[-300:], "runtime_s": np.nan}
            sweep.append({"setting": s, "heldout_log_like_per_spike": score, "runtime_s": res.get("runtime_s", np.nan)})
            if verbose:
                print(f"    PP-Seq sweep {i + 1}/{sweep_size}: R={s['num_sequence_types']} amp={s['mean_event_amplitude']:.0f} bkgd={s['mean_bkgd_spike_rate']:.1f} "
                      f"rate={s['seq_event_rate']:.3f} width={s['neuron_width_prior']:.3f} -> held-out {score:.4f} ({res.get('runtime_s', float('nan')):.0f}s)", flush=True)
        best = max(sweep, key=lambda r: r["heldout_log_like_per_spike"])["setting"]
        cfg = dict(BASE_CONFIG, **best, var_event_amplitude=best["mean_event_amplitude"], var_bkgd_spike_rate=best["mean_bkgd_spike_rate"],
                   seed=int(rng.integers(1 << 30)), num_threads=THREADS)
        full = _run_julia(spikes, n_cells, max_time, cfg, False, wd)
    parsed = parse(full, starts, ends, times_ms.shape[0])
    for key in ("p_event", "p_event_sequence_types"):
        p_all = np.full(n_all, np.nan)
        p_all[subset] = parsed[key]
        parsed[key] = p_all
    parsed["event_subset"] = subset
    return parsed | {"sweep": sweep, "best_setting": best, "runtime_s": time.perf_counter() - t_all,
                     "n_spikes": int(spikes.shape[0]), "n_events_fitted": int(len(subset)), "max_time_units": max_time}


def _n_by_r(value, n_cells):
    """JSON.jl writes a Julia matrix column by column; return the array as (neurons, types)."""
    a = np.array(value, float)
    return a if a.shape[0] == n_cells and (a.ndim == 1 or a.shape[1] != n_cells or a.shape[0] == a.shape[1]) else a.T


def parse(full, starts, ends, n_events):
    n_cells = int(full["num_neurons"])
    offsets = _n_by_r(full["neuron_offsets"], n_cells)                # N x R
    props = _n_by_r(full["neuron_response_proportions"], n_cells)     # N x R
    widths = _n_by_r(full["neuron_widths"], n_cells)
    if offsets.ndim == 1:
        offsets, props, widths = offsets[:, None], props[:, None], widths[:, None]
    R = offsets.shape[1]
    type_prop = np.array(full["seq_type_proportions"], float)
    spikes_per_type = np.array(full["spikes_per_type_final"], int)
    # per-event posterior probability of containing a latent sequence event, by type and overall
    saved = full["saved_events"]
    n_saved = len(saved)
    contains = np.zeros((n_events, R))
    for ev in saved:
        hit = np.zeros((n_events, R), bool)
        for e in ev:
            i = np.searchsorted(starts, e["t"], side="right") - 1
            if 0 <= i < n_events and starts[i] - GAP_UNITS / 2 <= e["t"] <= ends[i] + GAP_UNITS / 2:
                hit[i, e["type"] - 1] = True
        contains += hit
    contains /= max(n_saved, 1)
    types = []
    for r in range(R):
        members = props[:, r] >= MEMBER_FRAC * props[:, r].max() if props[:, r].max() > 0 else np.zeros(offsets.shape[0], bool)
        lag_ms = np.where(members, offsets[:, r] * UNIT_MS, np.nan)
        n_latent = float(np.mean([sum(1 for e in ev if e["type"] == r + 1) for ev in saved])) if n_saved else 0.0
        types.append({"r": r, "members": members, "lag_ms": lag_ms, "n_members": int(members.sum()),
                      "lag_span_ms": float(np.nanmax(lag_ms) - np.nanmin(lag_ms)) if members.sum() > 1 else np.nan,
                      "width_ms": float(np.sqrt(np.nanmedian(widths[members, r])) * UNIT_MS) if members.any() else np.nan,
                      "type_proportion": float(type_prop[r]), "spikes_final": int(spikes_per_type[r]), "mean_latent_events": n_latent})
    # PP-Seq models a synchronous volley as a sequence type whose offsets all coincide. Types whose
    # members spread over at least MIN_SPREAD_MS are "sequence" types; detection is reported for
    # any type and for sequence types only. Both use the model's own parameters, no behaviour.
    spread = np.array([t["lag_span_ms"] >= MIN_SPREAD_MS if np.isfinite(t["lag_span_ms"]) else False for t in types])
    p_seq = contains[:, spread].max(axis=1) if spread.any() else np.zeros(n_events)
    return {"types": types, "p_event": contains.max(axis=1), "p_event_sequence_types": p_seq, "sequence_types": spread,
            "p_event_by_type": contains, "n_saved_samples": n_saved,
            "n_background_final": int(full["n_background_final"]), "julia_runtime_s": full["runtime_s"]}
