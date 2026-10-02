"""Run the pipeline on hc-11 sessions, POST and PRE sleep.

Events: population bursts in NREM (edges at the mean rate, peak above mean + 3 SD),
window = the event itself centred on its peak, T = the event's duration, so the
population peak sits at angle pi/2 in every event and each event's extent is the
half-circle. Variants: {uncentred, centred} x {T = event duration, T = spike spread}.

Label-free outputs per variant: eigenvalue shares, components kept (Marchenko-Pastur),
split-half match of the kept components, per-component member count, angular span,
timing-consistency z.
Answer key (used only here, at the end): per component and running direction, the
Spearman between members' relative times and their place-field centres against a
cell-permutation null, and the compactness of members' fields on the track.
Baseline: count PCA/ICA on the same events (assemblies, no order).

Usage: python scripts/run_hc11.py [session_prefix ...]   (default: all sessions)
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.count_pca_ica import assemblies, best_jaccard, event_counts  # noqa: E402
from baselines.count_pca_ica import members as count_members  # noqa: E402
from cpca import event_scale, event_spike_times, fit, population_bursts  # noqa: E402
from cpca.evaluate import match_components, member_field_compactness, member_order_vs_place, timing_consistency  # noqa: E402
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402

OUT = ROOT / "results" / "hc11"
VARIANTS = {
    "uncentred, T = event duration": dict(center=False, scale=False, T="duration"),
    "centred, T = event duration": dict(center=True, scale=True, T="duration"),
    "uncentred, T = spike spread": dict(center=False, scale=False, T="spread"),
    "centred, T = spike spread": dict(center=True, scale=True, T="spread"),
}
EPOCHS = ("post", "pre")
MIN_SPAN = np.pi / 3


def events_for(session, epoch):
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    starts, ends = peaks - duration / 2, peaks + duration / 2
    times, truncated = event_spike_times(by_cell, cell_ids, starts, ends)
    return {"by_cell": by_cell, "cell_ids": cell_ids, "times": times, "duration_ms": duration * 1000,
            "n_events": len(peaks), "truncated": truncated}


def analyse_variant(ev, settings, run, centres, rng):
    if settings["T"] == "duration":
        T, times = ev["duration_ms"], ev["times"]
    else:
        T, start = event_scale(ev["times"])
        times = ev["times"] - start[:, None, None]
    t0 = time.perf_counter()
    res = fit(times, T, rng=rng, center=settings["center"], scale=settings["scale"], ica=False)
    k = res["n_kept"]
    # split-half reproducibility of the kept components
    order = rng.permutation(times.shape[0])
    half = times.shape[0] // 2
    a = fit(times[order[:half]], np.asarray(T)[order[:half]] if np.ndim(T) else T, rng=rng, center=settings["center"], scale=settings["scale"], ica=False, n_components=max(k, 1))
    b = fit(times[order[half:]], np.asarray(T)[order[half:]] if np.ndim(T) else T, rng=rng, center=settings["center"], scale=settings["scale"], ica=False, n_components=max(k, 1))
    split_match = match_components(a["patterns"], b["patterns"])
    comps = []
    for j in range(k):
        p = res["patterns"][:, j]
        z, _, _ = timing_consistency(res["encoded"], p, rng, n_shuffles=30)
        row = {"index": j, "eigen_share": float(res["eigenvalues"][j] / res["eigenvalues"].sum()), "n_members": int(res["members"][j].sum()),
               "span_rad": res["readouts"][j]["span_rad"], "timing_z": float(z), "map": {}}
        for sign in (1.0, -1.0):
            rho, null95, n = member_order_vs_place(p, res["T_read_ms"], centres[sign], run["circular"], run["length"], rng, n_null=500)
            spread, spread05, spread_mean, n_c = member_field_compactness(p, centres[sign], run["circular"], run["length"], rng, n_null=300)
            row["map"][sign] = {"rho": rho, "null95": null95, "n_members_with_field": n, "field_spread": spread,
                                "field_spread_null05": spread05, "field_spread_null_mean": spread_mean}
        comps.append(row)
    return {"n_kept": k, "noise_bound_share": float(res["noise_bound"] / res["eigenvalues"].sum()) if np.isfinite(res["noise_bound"]) else np.nan,
            "eigen_share": (res["eigenvalues"][:10] / res["eigenvalues"].sum()).tolist(), "split_half_match": split_match,
            "components": comps, "runtime_s": time.perf_counter() - t0, "members_masks": res["members"]}


def analyse_session(path, rng):
    session = load_session(path)
    run = running(session)
    out = {"name": session["name"], "maze": session["maze"], "circular": run["circular"], "epochs": {}}
    for epoch in EPOCHS:
        ev = events_for(session, epoch)
        centres = place_centres(session, ev["cell_ids"], run)
        block = {"n_cells": len(ev["cell_ids"]), "n_events": ev["n_events"], "median_duration_ms": float(np.median(ev["duration_ms"])),
                 "n_place": {s: int(np.isfinite(c).sum()) for s, c in centres.items()}, "variants": {}}
        for vname, settings in VARIANTS.items():
            block["variants"][vname] = analyse_variant(ev, settings, run, centres, rng)
        # count PCA/ICA baseline on the same events
        counts = event_counts(ev["times"])
        weights, n_assemblies, _, _ = assemblies(counts, rng)
        count_sets = [count_members(weights[i]) for i in range(n_assemblies)]
        spreads = [member_field_compactness(np.nan_to_num(weights[i]) + 0j, centres[1.0], run["circular"], run["length"], rng, n_null=200) for i in range(n_assemblies)]
        complex_sets = block["variants"]["uncentred, T = event duration"]["members_masks"]
        block["count_baseline"] = {"n_assemblies": n_assemblies,
                                   "field_spread": [s[0] for s in spreads], "field_spread_null05": [s[1] for s in spreads],
                                   "jaccard_complex_to_count": best_jaccard(complex_sets, count_sets).tolist() if count_sets else []}
        for v in block["variants"].values():
            v.pop("members_masks")
        out["epochs"][epoch] = block
        print(f"[{session['name']}] {epoch}: {block['n_events']} events, {block['n_cells']} cells, place {block['n_place']} | " +
              " | ".join(f"{vn}: k={v['n_kept']}, split {v['split_half_match']:.2f}, map hits +{sum(c['map'][1.0]['rho'] is not None and np.isfinite(c['map'][1.0]['rho']) and abs(c['map'][1.0]['rho']) > c['map'][1.0]['null95'] for c in v['components'])}/-{sum(np.isfinite(c['map'][-1.0]['rho']) and abs(c['map'][-1.0]['rho']) > c['map'][-1.0]['null95'] for c in v['components'])}" for vn, v in block["variants"].items()) +
              f" | count assemblies {n_assemblies}", flush=True)
    return out


def main(prefixes):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in session_paths() if not prefixes or any(p.name.startswith(x) for x in prefixes)]
    for path in paths:
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        result = analyse_session(path, rng)
        with open(OUT / f"{result['name']}.pkl", "wb") as fh:
            pickle.dump(result, fh)
        print(f"[{result['name']}] saved ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
