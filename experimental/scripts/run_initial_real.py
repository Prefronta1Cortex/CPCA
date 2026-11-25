"""Initial real-data tests of the un-mixing step on local data (docs/unmixing.md, "Initial real-data tests").

Waking: DANDI 000552 figure-eight trials (stem as its own shared segment) and DANDI 001695 laps (two running
directions). Reactivations: hc-11 and DANDI 001695, POST and PRE NREM bursts and rest on the maze. No method
changes: waking mode as in scripts/run_waking_separation.py, reactivation mode as confirmed on planted data.

Usage: python scripts/run_initial_real.py [--workers=9]
Output: results/initial_real/rows.pkl, summary.md
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "initial_real"
STEM_CM = 85.0
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def halves_by(labels, rng):
    half = np.zeros(len(labels), bool)
    for v in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == v))
        half[idx[: idx.size // 2]] = True
    return half


# ----------------------------------------------------------------------------- figure-eight (000552)
def figure8_events(s, trials, cells, min_speed=5.0, max_spikes=200):
    """Trials as events: spikes while running only, timed on a running-time clock (ms)."""
    t = s["lin_t"]
    speed = np.r_[0, np.hypot(*np.diff(s["xy"], axis=0).T) / np.maximum(np.diff(s["xy_t"]), 1e-6)]
    moving = np.interp(t, s["xy_t"], speed) > min_speed
    dt = float(np.median(np.diff(t)))
    times = np.full((len(trials), len(cells), max_spikes), np.nan)
    T = np.zeros(len(trials))
    for e, i in enumerate(trials):
        a, b = s["trials"][i]
        lo, hi = np.searchsorted(t, a), np.searchsorted(t, b)
        ts, run = t[lo:hi], moving[lo:hi]
        if ts.size < 2:
            continue
        clock = np.cumsum(run) * dt * 1000
        T[e] = max(clock[-1], 1.0)
        for k, c in enumerate(cells):
            sp = s["spikes"][c]
            sp = sp[(sp >= a) & (sp < b)]
            j = np.clip(np.searchsorted(ts, sp), 0, ts.size - 1)
            keep = run[j]
            v = np.sort(clock[j[keep]])[:max_spikes]
            times[e, k, :v.size] = v
    return times, T


def judge_figure8(order, left, right, rng):
    from scripts.run_separation import spearman_p
    arm_l = np.isfinite(left) & (left >= STEM_CM) & ~(np.isfinite(right) & (right >= STEM_CM))
    arm_r = np.isfinite(right) & (right >= STEM_CM) & ~(np.isfinite(left) & (left >= STEM_CM))
    stem = ((np.isfinite(left) & (left < STEM_CM)) | (np.isfinite(right) & (right < STEM_CM))) & ~arm_l & ~arm_r
    mem = np.isfinite(order)
    nl, nr, ns = int((mem & arm_l).sum()), int((mem & arm_r).sum()), int((mem & stem).sum())
    if nl + nr >= 6 and max(nl, nr) / (nl + nr) >= 0.8:
        side, cen, on = ("left", left, mem & arm_l) if nl >= nr else ("right", right, mem & arm_r)
        if on.sum() >= 6 and spearman_p(order[on], cen[on], rng)[1] < 0.05:
            return side
        return "fail"
    if ns >= 6 and ns / max(nl + nr + ns, 1) >= 0.8:
        on = mem & stem
        pos = np.nanmean(np.vstack([np.where(left < STEM_CM, left, np.nan), np.where(right < STEM_CM, right, np.nan)]), axis=0)
        return "stem" if spearman_p(order[on], pos[on], rng)[1] < 0.05 else "fail"
    return "fail"


def figure8_job(path, fold):
    import cpca.huszar as hz
    from experimental.scripts.dev_waking import waking_outputs
    from experimental.scripts.run_waking_separation import SETTINGS
    s = hz.load_session(path)
    task_w = hz.task_window(s)[1]
    trials = hz.task_trials(s, task_w)
    arms = s["arm"][trials]
    cells = hz.putative_pyramidal(s)
    half = halves_by(arms, np.random.default_rng(seed("fig8-half", s["name"])))
    train = half if fold == 0 else ~half
    times, T = figure8_events(s, trials, cells)
    held = dict(s)
    held["arm"] = s["arm"].copy()
    held["arm"][trials[train]] = "none"
    maps, n_spikes, mid = hz.arm_maps(held, cells, task_w)
    cen = hz.place_centres(maps, n_spikes, mid)
    rng = np.random.default_rng(seed("fig8-fit", s["name"], fold))
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    outs = waking_outputs(times[tr], T[tr], times[te], T[te], rng, **SETTINGS)
    jrng = np.random.default_rng(seed("fig8-judge", s["name"], fold))
    verdicts = [judge_figure8(o, cen[1.0], cen[-1.0], jrng) for o, p, _ in outs if p < 0.05]
    return {"part": "figure-eight", "name": s["name"], "fold": fold, "n_trials": int(train.sum()),
            "fields": (int(np.isfinite(cen[1.0]).sum()), int(np.isfinite(cen[-1.0]).sum())), "verdicts": verdicts}


# ----------------------------------------------------------------------------- 001695 laps
def laps_job(path, fold):
    from cpca import event_spike_times
    from cpca.dandi001695 import load_session
    from cpca.hc11 import laps, place_centres, pyramidal_spikes, running
    from experimental.scripts.dev_waking import waking_outputs
    from scripts.run_laps_hc11 import heldout_run
    from experimental.scripts.run_waking_separation import SETTINGS, score
    s = load_session(path)
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    half = halves_by(dirs, np.random.default_rng(seed("dandi-laps-half", path.name)))
    train = half if fold == 0 else ~half
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    X, _ = event_spike_times(by_cell, cell_ids, starts[tr], ends[tr], max_spikes=400)
    Y, _ = event_spike_times(by_cell, cell_ids, starts[te], ends[te], max_spikes=400)
    TX, TY = (ends[tr] - starts[tr]) * 1000, (ends[te] - starts[te]) * 1000
    centres = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
    maps = [centres[sg] for sg in (1.0, -1.0) if np.isfinite(centres[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
    rng = np.random.default_rng(seed("dandi-laps-fit", path.name, fold))
    outs = waking_outputs(X, TX, Y, TY, rng, **SETTINGS)
    rows = score(outs, maps, run["circular"], run["length"], np.random.default_rng(seed("dandi-laps-score", path.name, fold)))
    return {"part": "001695 laps", "name": path.name, "fold": fold, "n_maps": len(maps), "outputs": rows,
            "n_laps": (int(np.sum(dirs[tr] == 1)), int(np.sum(dirs[tr] == -1)))}


# ----------------------------------------------------------------------------- reactivations
def reactivation_job(dataset, key, epoch, fold):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import laps, nrem_intervals, place_centres, pyramidal_spikes, running
    from cpca.replay import immobile_intervals
    from experimental.scripts.dev_unmixing import outputs
    from scripts.run_replay_hc11 import intersect
    from experimental.scripts.run_waking_separation import score
    if dataset == "hc-11":
        from cpca.hc11 import load_session, session_paths
        from scripts.run_replay_hc11 import two_d
        path = next(p for p in session_paths() if p.name.startswith(key))
        s = load_session(path)
        s["pos_xy"] = two_d(path)
    else:
        from cpca.dandi001695 import load_session
        s = load_session(key)
    run = running(s)

    def bursts(iv):
        by_cell, cell_ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv)
        return by_cell, cell_ids, pk, np.diff(b, axis=1).ravel()

    rng = np.random.default_rng(seed("react", dataset, str(key), epoch))
    if epoch == "maze":
        by_cell, cell_ids, pk, d = bursts(intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]))
    else:
        n = min(len(bursts(nrem_intervals(s, e))[2]) for e in ("pre", "post"))
        by_cell, cell_ids, pk, d = bursts(nrem_intervals(s, epoch))
        pick = np.sort(rng.choice(len(pk), n, replace=False))
        pk, d = pk[pick], d[pick]
    times, _ = event_spike_times(by_cell, cell_ids, pk - d / 2, pk + d / 2, max_spikes=60)
    T = d * 1000
    perm = rng.permutation(len(pk))
    halves = (perm[: perm.size // 2], perm[perm.size // 2:])
    tr, te = halves if fold == 0 else halves[::-1]
    dirs = np.array([x[2] for x in laps(run)])
    centres = place_centres(s, cell_ids, run)
    maps = [centres[sg] for sg in (1.0, -1.0) if np.isfinite(centres[sg]).sum() >= 6 and np.sum(dirs == sg) >= 3]
    outs = outputs("final_k_now", times[tr], T[tr], times[te], T[te], np.random.default_rng(seed("react-fit", dataset, str(key), epoch, fold)))
    rows = score(outs, maps, run["circular"], run["length"], np.random.default_rng(seed("react-score", dataset, str(key), epoch, fold)))
    name = key if dataset == "hc-11" else Path(key).name
    return {"part": f"{dataset} reactivations", "name": name, "epoch": epoch, "fold": fold, "n_maps": len(maps),
            "circular": bool(run["circular"]), "n_events": len(pk), "outputs": rows}


# ----------------------------------------------------------------------------- summary
def summarize(rows):
    lines = ["# Initial real-data tests of the un-mixing step", "", "Design: `docs/unmixing.md` (initial real-data tests).", ""]
    fig = [r for r in rows if r["part"] == "figure-eight"]
    if fig:
        v = [x for r in fig for x in r["verdicts"]]
        clean = sum({"left", "right"} <= set(r["verdicts"]) and "fail" not in r["verdicts"] for r in fig)
        both = sum({"left", "right"} <= set(r["verdicts"]) for r in fig)
        lines += ["## Waking, figure-eight (DANDI 000552)", "",
                  f"- fits: {len(fig)}; outputs: {len(v)} (left arm {v.count('left')}, right arm {v.count('right')}, stem {v.count('stem')}, fail {v.count('fail')})",
                  f"- fits with both arms as separate outputs: {both}/{len(fig)}; clean (both arms, no failing output): {clean}/{len(fig)}",
                  f"- fits with any output: {sum(len(r['verdicts']) > 0 for r in fig)}/{len(fig)}; median training trials {np.median([r['n_trials'] for r in fig]):.0f}", ""]
    for part in ("001695 laps",):
        rs = [r for r in rows if r["part"] == part]
        if not rs:
            continue
        lines += [f"## Waking, {part}", "", "| criterion | outputs passing | clean fits | fits with any output |", "|---|---|---|---|"]
        for crit, label in (("_partial", "partial correlation"), ("_share", "variance share"), ("", "original")):
            outs = [[o for o in r["outputs"] if o["passes_check"]] for r in rs]
            p = sum(o["passes" + crit] for oo in outs for o in oo)
            clean = sum(r["n_maps"] > 0 and len({o["map" + crit] for o in oo if o["passes" + crit]}) == r["n_maps"]
                        and all(o["passes" + crit] for o in oo) for r, oo in zip(rs, outs))
            lines.append(f"| {label} | {p}/{sum(len(o) for o in outs)} | {clean}/{len(rs)} | {sum(len(o) > 0 for o in outs)}/{len(rs)} |")
        lines.append("")
    for ds in ("hc-11", "001695"):
        rs = [r for r in rows if r["part"] == f"{ds} reactivations"]
        if not rs:
            continue
        lines += [f"## Reactivations, {ds}", "", "Outputs passing the latency check; matching exactly one running direction (partial / variance share / original); "
                  "chance per output about 9.5% with two maps, 5% with one.", "",
                  "| epoch | fits | outputs | partial | share | original | expected by chance | sessions with a match (partial) |", "|---|---|---|---|---|---|---|---|"]
        for ep in ("maze", "post", "pre"):
            sub = [r for r in rs if r["epoch"] == ep]
            outs = [(r, o) for r in sub for o in r["outputs"] if o["passes_check"]]
            exp = sum(0.095 if r["n_maps"] == 2 else 0.05 for r, _ in outs)
            sess = len({r["name"] for r, o in outs if o["passes_partial"]})
            lines.append(f"| {ep} | {len(sub)} | {len(outs)} | {sum(o['passes_partial'] for _, o in outs)} | {sum(o['passes_share'] for _, o in outs)} | "
                         f"{sum(o['passes'] for _, o in outs)} | {exp:.1f} | {sess}/{len({r['name'] for r in sub})} |")
        lines.append("")
    return "\n".join(lines)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.dandi001695 as dd
    import cpca.huszar as hz
    jobs = [(figure8_job, (p, f)) for p in hz.session_paths() for f in (0, 1)]
    jobs += [(laps_job, (p, f)) for p in dd.session_paths() for f in (0, 1)]
    jobs += [(reactivation_job, ("hc-11", n, e, f)) for n in HC11 for e in ("maze", "post", "pre") for f in (0, 1)]
    jobs += [(reactivation_job, ("001695", p, e, f)) for p in dd.session_paths() for e in ("maze", "post", "pre") for f in (0, 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc()[-1500:]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))
    text = summarize(rows)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
