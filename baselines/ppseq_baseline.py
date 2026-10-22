"""PP-Seq (Williams, Degleris, Wang & Linderman, NeurIPS 2020), run through the authors' Julia package.

Data preparation. PP-Seq models a continuous recording. The population-burst events
are laid end to end with a silent gap between them, and time is expressed in units
of 50 ms (one unit is roughly one replay-compressed traversal step, which puts the
authors' hippocampal prior ranges, written for running-speed sequences, at a
comparable scale). Spike times keep their position inside each event.

Hyperparameters, following Supplement F.2 of the paper: a random sweep over the
number of sequence types R (drawn independently of the other hyperparameters), the
mean sequence amplitude (variance = mean), the background amplitude (variance = mean),
the sequence rate, the neuron response width, and whether time warping is allowed
(warp factors up to max_warp); every candidate is scored by the package's held-out
statistic on ONE speckled 7.5% mask shared by all candidates (log-likelihood of the
masked spikes minus that of a homogeneous Poisson process on the masked volume), and
the best candidate is re-fit on all spikes. Candidates run as parallel single-threaded
Julia processes; the final fit uses the package's multi-threaded sampler.

Scoring. Two membership rules are reported for each sequence type r:
- "type": neurons whose response proportion a_nr is at least 20% of the type's largest;
- "package": the rule of the package's `sortperm_neurons` (a neuron's largest response
  proportion across types above the 20th percentile of those maxima, the neuron assigned
  to its preferred type).
The order is the neuron offsets b_nr of the members. Each saved post-anneal sample gives
latent events; an event window contains a sequence in the fraction of samples with a
latent event inside it. Because a synchronous volley is represented as a type whose
offsets coincide, detection is reported for any type and for "sequence types" (member
offsets spanning at least MIN_SPREAD_MS), both defined from the model alone.
"""

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JULIA_PROJECT = HERE / "ppseq"
JULIA = Path.home() / ".juliaup" / "bin" / "julia"
UNIT_MS = 50.0
GAP_UNITS = 2.0
SWEEP_SIZE = 20
SWEEP_WORKERS = 3          # parallel single-threaded Julia processes for the masked sweep
MEMBER_FRAC = 0.2
MAX_EVENTS = 1500          # events for the final fit (the comparison uses one fixed subset for every method)
SWEEP_EVENTS = 600         # subset for the held-out sweep (the masked sampler runs single-threaded)
THREADS = 4                # the package's DistributedSeqModel splits the timeline across threads (final fit)
MIN_SPREAD_MS = 20.0       # a sequence type whose member offsets span less than this is a synchronous volley
R_RANGE = (2, 8)           # number of sequence types, inclusive

# MCMC schedule for the final fit (authors: 20 anneal stages x 100 sweeps from temperature 500, then 100 sweeps
# with 1000 split-merge moves each). Here: 12 x 40 sweeps from temperature 200, then 60 sweeps, 20 split-merge moves.
BASE_CONFIG = {
    "seq_type_conc_param": 3.0,            # gamma = 3 (paper)
    "neuron_response_conc_param": 1.0,     # phi = 1 (paper)
    "neuron_offset_pseudo_obs": 1.0,
    "neuron_width_pseudo_obs": 4.0,        # nu = 4 (paper)
    "num_warp_values": 1, "max_warp": 1.0, "warp_variance": 100.0,   # overridden by the sweep
    "bkgd_spikes_conc_param": 0.3,
    "max_sequence_length": "Inf",
    "num_anneals": 12, "samples_per_anneal": 40, "max_temperature": 200.0, "save_every_during_anneal": 40,
    "samples_after_anneal": 60, "save_every_after_anneal": 3,
    "split_merge_moves_during_anneal": 20, "split_merge_moves_after_anneal": 20, "split_merge_window": 1.0,
    "mask_lengths": 1.0, "percent_masked": 7.5,
    "num_spike_resamples_per_anneal": 5, "num_spike_resamples": 10, "samples_per_resample": 4,
}
# shorter schedule for the sweep runs only
SWEEP_SCHEDULE = {"num_anneals": 6, "samples_per_anneal": 15, "save_every_during_anneal": 15,
                  "num_spike_resamples_per_anneal": 3, "num_spike_resamples": 6, "samples_per_resample": 4}


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


def _run_julia(spikes, n_cells, max_time, config, masked, workdir, threads):
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    sp = wd / "spikes.tsv"
    if not sp.exists():
        np.savetxt(sp, spikes, fmt=["%d", "%.6f"], delimiter="\t")
    cp = wd / "config.json"
    cp.write_text(json.dumps(config))
    out = wd / ("masked.json" if masked else "full.json")
    cmd = [str(JULIA), f"-t{threads}", f"--project={JULIA_PROJECT}", str(HERE / "ppseq" / "run_ppseq.jl"), str(sp), str(n_cells), f"{max_time:.6f}", str(cp), str(out)]
    if masked:
        cmd.append("masked")
    env = dict(os.environ, PATH=f"{JULIA.parent}:{os.environ.get('PATH', '')}")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"PP-Seq failed:\n{proc.stderr[-3000:]}")
    return json.loads(out.read_text())


def sample_settings(rng, n, spikes_per_unit, mean_spikes_per_event):
    """Random settings in the spirit of the paper's sweep, scaled to these data. R and warping are independent draws."""
    settings = []
    for _ in range(n):
        warp = bool(rng.random() < 0.5)
        settings.append({
            "num_sequence_types": int(rng.integers(R_RANGE[0], R_RANGE[1] + 1)),
            "mean_event_amplitude": float(10 ** rng.uniform(np.log10(0.3 * mean_spikes_per_event), np.log10(3 * mean_spikes_per_event))),
            "mean_bkgd_spike_rate": float(10 ** rng.uniform(np.log10(0.2 * spikes_per_unit), np.log10(2 * spikes_per_unit))),
            "seq_event_rate": float(10 ** rng.uniform(-2, -1)),            # paper range (1e-3, 1e-1); the upper end bounds run time
            "neuron_width_prior": float(10 ** rng.uniform(-2, 0)),
            "num_warp_values": 3 if warp else 1,
            "max_warp": float(rng.uniform(1.2, 1.5)) if warp else 1.0,    # paper range (1, 1.5)
        })
    return settings


def _config(setting, seed, mask_seed, threads, schedule=None):
    cfg = dict(BASE_CONFIG, **(schedule or {}), **setting,
               var_event_amplitude=setting["mean_event_amplitude"], var_bkgd_spike_rate=setting["mean_bkgd_spike_rate"],
               seed=int(seed), mask_seed=int(mask_seed), num_threads=threads)
    return cfg


def fit(times_ms, T_ms, rng, sweep_size=SWEEP_SIZE, sweep_events=SWEEP_EVENTS, verbose=True,
        unit_ms=UNIT_MS, gap_units=GAP_UNITS, sweep_workers=SWEEP_WORKERS, threads=THREADS,
        sweep_schedule=SWEEP_SCHEDULE, final_schedule=None, refits_if_empty=0):
    """Sweep hyperparameters on one shared mask, refit the best on all given events, return parsed results.

    times_ms: (events, cells, spikes); the caller passes the fixed event subset shared by all methods.
    unit_ms is the model time unit, gap_units the silent gap between concatenated events; threads=0
    runs the final fit single-threaded (for callers that parallelise over fits). sweep_schedule and
    final_schedule override BASE_CONFIG's sampler schedule for the sweep runs and the final fit.
    refits_if_empty: if the refit of the best candidate ends with no latent events (every spike
    background), refit the next-best candidate, up to this many times.
    Returns per-type orders under both membership rules, per-event posterior probabilities, the sweep table
    and the chosen setting.
    """
    t_all = time.perf_counter()
    plan = plan_fit(times_ms, T_ms, rng, sweep_size, sweep_events, unit_ms, gap_units)
    with tempfile.TemporaryDirectory() as wd:
        with ThreadPoolExecutor(max_workers=sweep_workers) as pool:
            sweep = list(pool.map(lambda i: sweep_candidate(plan, i, Path(wd) / f"sweep_{i}", sweep_schedule), range(sweep_size)))
        if verbose:
            for i, r in enumerate(sweep):
                s = r["setting"]
                print(f"    PP-Seq sweep {i + 1}/{sweep_size}: R={s['num_sequence_types']} warp={'yes' if s['num_warp_values'] > 1 else 'no'} amp={s['mean_event_amplitude']:.0f} "
                      f"bkgd={s['mean_bkgd_spike_rate']:.1f} rate={s['seq_event_rate']:.3f} width={s['neuron_width_prior']:.3f} -> held-out {r['heldout_log_like']:.4f} per masked neuron-unit ({r['runtime_s']:.0f}s)", flush=True)
        parsed, best, attempts = refit_best(plan, sweep, Path(wd), threads, final_schedule, refits_if_empty)
    return parsed | {"sweep": sweep, "best_setting": best, "runtime_s": time.perf_counter() - t_all, "refit_attempts": attempts,
                     "n_spikes": int(plan["spikes"].shape[0]), "n_events_fitted": int(plan["n_all"]), "max_time_units": plan["max_time"]}


def plan_fit(times_ms, T_ms, rng, sweep_size=SWEEP_SIZE, sweep_events=SWEEP_EVENTS, unit_ms=UNIT_MS, gap_units=GAP_UNITS):
    """Everything `fit` decides before running Julia: timelines, the candidate settings, the shared mask seed and the run seeds.

    Deterministic given rng, so a caller that runs the candidates as separate jobs can rebuild the
    same plan in every job (and `fit` is plan_fit + sweep_candidate for each candidate + refit_best).
    """
    n_all, n_cells, _ = times_ms.shape
    T_ms = np.broadcast_to(np.asarray(T_ms, float), (n_all,))
    spikes, starts, ends, max_time = build_timeline(times_ms, T_ms, unit_ms, gap_units)
    spikes_per_unit = spikes.shape[0] / max_time
    active_per_event = np.isfinite(times_ms).sum(axis=(1, 2))
    mean_spikes_per_event = float(active_per_event[active_per_event > 0].mean())
    settings = sample_settings(rng, sweep_size, spikes_per_unit, mean_spikes_per_event)
    sweep_idx = np.sort(rng.choice(n_all, sweep_events, replace=False)) if n_all > sweep_events else np.arange(n_all)
    sw_spikes, _, _, sw_max_time = build_timeline(times_ms[sweep_idx], T_ms[sweep_idx], unit_ms, gap_units)
    mask_seed = int(rng.integers(1 << 30))
    seeds = rng.integers(1 << 30, size=sweep_size + 1)
    return {"n_all": n_all, "n_cells": n_cells, "spikes": spikes, "starts": starts, "ends": ends, "max_time": max_time,
            "settings": settings, "sw_spikes": sw_spikes, "sw_max_time": sw_max_time, "mask_seed": mask_seed, "seeds": seeds,
            "unit_ms": unit_ms, "gap_units": gap_units}


def sweep_candidate(plan, i, workdir, sweep_schedule=SWEEP_SCHEDULE):
    """Run candidate i of the plan on the shared mask (single-threaded); returns its sweep-table row."""
    cfg = _config(plan["settings"][i], plan["seeds"][i], plan["mask_seed"], 0, sweep_schedule)
    try:
        res = _run_julia(plan["sw_spikes"], plan["n_cells"], plan["sw_max_time"], cfg, True, Path(workdir), 1)
        return {"setting": plan["settings"][i], "heldout_log_like": res["heldout_log_like"], "n_heldout_spikes": res["n_heldout_spikes"], "runtime_s": res["runtime_s"]}
    except RuntimeError as err:
        return {"setting": plan["settings"][i], "heldout_log_like": -np.inf, "error": str(err)[-300:], "runtime_s": np.nan}


def refit_best(plan, sweep, workdir, threads=THREADS, final_schedule=None, refits_if_empty=0):
    """Refit the best candidate on all events (the next-best ones if a refit has no latent events); returns (parsed, setting, attempts)."""
    ranked = sorted(range(len(sweep)), key=lambda i: -sweep[i]["heldout_log_like"])
    for attempt, i in enumerate(ranked[:1 + refits_if_empty]):
        best = sweep[i]["setting"]
        full = _run_julia(plan["spikes"], plan["n_cells"], plan["max_time"], _config(best, plan["seeds"][-1], plan["mask_seed"], threads, final_schedule),
                          False, Path(workdir) / f"final_{attempt}", max(threads, 1))
        parsed = parse(full, plan["starts"], plan["ends"], plan["n_all"], plan["unit_ms"], plan["gap_units"])
        if any(t["spikes_final"] > 0 or t["mean_latent_events"] > 0 for t in parsed["types"]):
            break
    return parsed, best, attempt + 1


def _n_by_r(value, n_cells):
    """JSON.jl writes a Julia matrix column by column; return the array as (neurons, types)."""
    a = np.array(value, float)
    if a.ndim == 1:
        return a[:, None]
    return a if a.shape[0] == n_cells and a.shape[1] != n_cells else (a.T if a.shape[1] == n_cells else a)


def membership_package(props):
    """The package's `sortperm_neurons` rule: largest response across types above the 20th percentile of those maxima; preferred type."""
    peak = props.max(axis=1)
    has = peak > np.quantile(peak, MEMBER_FRAC)
    preferred = props.argmax(axis=1)
    return has, preferred


def parse(full, starts, ends, n_events, unit_ms=UNIT_MS, gap_units=GAP_UNITS):
    n_cells = int(full["num_neurons"])
    offsets = _n_by_r(full["neuron_offsets"], n_cells)                # N x R
    props = _n_by_r(full["neuron_response_proportions"], n_cells)     # N x R
    widths = _n_by_r(full["neuron_widths"], n_cells)
    R = offsets.shape[1]
    type_prop = np.atleast_1d(np.array(full["seq_type_proportions"], float))
    spikes_per_type = np.atleast_1d(np.array(full["spikes_per_type_final"], int))
    saved = full["saved_events"]
    n_saved = len(saved)
    contains = np.zeros((n_events, R))
    for ev in saved:
        hit = np.zeros((n_events, R), bool)
        for e in ev:
            i = np.searchsorted(starts, e["t"], side="right") - 1
            if 0 <= i < n_events and starts[i] - gap_units / 2 <= e["t"] <= ends[i] + gap_units / 2:
                hit[i, e["type"] - 1] = True
        contains += hit
    contains /= max(n_saved, 1)
    has_pkg, preferred = membership_package(props)
    types = []
    for r in range(R):
        m_type = props[:, r] >= MEMBER_FRAC * props[:, r].max() if props[:, r].max() > 0 else np.zeros(n_cells, bool)
        m_pkg = has_pkg & (preferred == r)
        lag_type = np.where(m_type, offsets[:, r] * unit_ms, np.nan)
        lag_pkg = np.where(m_pkg, offsets[:, r] * unit_ms, np.nan)
        n_latent = float(np.mean([sum(1 for e in ev if e["type"] == r + 1) for ev in saved])) if n_saved else 0.0
        types.append({"r": r, "lag_ms": lag_type, "lag_ms_package": lag_pkg, "n_members": int(m_type.sum()), "n_members_package": int(m_pkg.sum()),
                      "lag_span_ms": float(np.nanmax(lag_type) - np.nanmin(lag_type)) if m_type.sum() > 1 else np.nan,
                      "width_ms": float(np.sqrt(np.nanmedian(widths[m_type, r])) * unit_ms) if m_type.any() else np.nan,
                      "type_proportion": float(type_prop[r]), "spikes_final": int(spikes_per_type[r]), "mean_latent_events": n_latent})
    spread = np.array([t["lag_span_ms"] >= MIN_SPREAD_MS if np.isfinite(t["lag_span_ms"]) else False for t in types])
    p_seq = contains[:, spread].max(axis=1) if spread.any() else np.zeros(n_events)
    return {"types": types, "p_event": contains.max(axis=1), "p_event_sequence_types": p_seq, "sequence_types": spread,
            "p_event_by_type": contains, "n_saved_samples": n_saved,
            "n_background_final": int(full["n_background_final"]), "julia_runtime_s": full["runtime_s"]}
