"""Waking un-mixing on real data, run once after the planted bar (docs/unmixing.md, waking design).

Pipeline: uncentred complex PCA on the training half, then experimental.separate.separate in waking mode (forward only,
lap-clustering split start, members trimmed after a split, reproduction bar from the data's own event count,
explaining away, event floor 4), then the waking check on the held-out half (circularly shifted surrogates,
Bonferroni over outputs). The answer key, used only to score, is the held-out half's place (or time) fields.

Data: hc-11 laps (8 sessions x 2 folds; scored on every running direction with held-out laps and at least 6 fields),
hc-5 wheel (aligned, repeat 0, 6 sessions x 2 folds), and two controls: hc-5 with shuffled training runs, and hc-11
laps with every cell's spikes circularly shifted in every lap (both halves).

Scores: an output passes if it matches exactly one map under the partial-correlation criterion (fixed before the
planted run), the variance-share criterion (added after the planted development) and the original criterion. On the
circular tracks, with one map, an output passes if it matches the map with the rotation search of best_map_match.

Usage: python scripts/run_waking_separation.py [--workers=9]
Output: results/waking_separation/rows.pkl, summary.md
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

OUT = ROOT / "results" / "waking_separation"
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
SETTINGS = dict(center=False, explain=True, conjugate=False, min_events=4, event_starts=True, trim=True, adaptive=True)


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def score(outs, maps, circular, length, rng):
    """Per output: passes under the three criteria (and which map)."""
    from experimental.scripts.dev_waking import judge_both
    from scripts.plot_hc11_sequences import best_map_match
    rows = []
    for order, p, mask in outs:
        row = {"p": p, "passes_check": p < 0.05, "n_members": int(mask.sum())}
        if not maps:
            row.update(passes=False, passes_partial=False, passes_share=False, map=-1, map_partial=-1, map_share=-1)
        elif circular:
            ok = bool(np.sum(np.isfinite(order) & np.isfinite(maps[0])) >= 6 and
                      best_map_match([order], {1.0: maps[0]}, True, length, rng, n_null=500)["p"] < 0.05)
            row.update(passes=ok, passes_partial=ok, passes_share=ok, map=0 if ok else -1, map_partial=0 if ok else -1,
                       map_share=0 if ok else -1)
        else:
            ok, k, ok2, k2, ok3, k3 = judge_both(order, maps, rng)
            row.update(passes=ok, map=k, passes_partial=ok2, map_partial=k2, passes_share=ok3, map_share=k3)
        rows.append(row)
    return rows


def lap_job(name, fold, control=False):
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from experimental.scripts.dev_waking import circular_shift, waking_outputs
    from scripts.run_laps_hc11 import heldout_run
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng = np.random.default_rng(seed("waking-sep", name))
    half = np.zeros(len(L), bool)
    for d in np.unique(dirs):
        idx = rng.permutation(np.flatnonzero(dirs == d))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    X, _ = event_spike_times(by_cell, cell_ids, starts[tr], ends[tr], max_spikes=400)
    Y, _ = event_spike_times(by_cell, cell_ids, starts[te], ends[te], max_spikes=400)
    TX, TY = (ends[tr] - starts[tr]) * 1000, (ends[te] - starts[te]) * 1000
    frng = np.random.default_rng(seed("waking-sep-fit", name, fold, control))
    if control:
        X, Y = circular_shift(X, TX, frng), circular_shift(Y, TY, frng)
    centres = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
    maps = [] if control else [centres[sg] for sg in (1.0, -1.0) if np.isfinite(centres[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
    outs = waking_outputs(X, TX, Y, TY, frng, **SETTINGS)
    rows = score(outs, maps, run["circular"], run["length"], np.random.default_rng(seed("waking-sep-score", name, fold)))
    return {"data": "hc-11 laps" + (" (control)" if control else ""), "track": "circular" if run["circular"] else "linear",
            "name": name, "fold": fold, "n_maps": len(maps), "outputs": rows, "n_train": len(tr)}


def wheel_job(name, fold, control=False):
    import cpca.hc5 as hc5
    from experimental.scripts.dev_waking import waking_outputs
    from scripts.run_hc5 import training_data
    session = hc5.load_session(name)
    X, T, train, test, _ = training_data(session, "shuffled" if control else "aligned", 0, fold)
    Y, TY, _, _, _ = training_data(session, "aligned", 0, 1 - fold)
    TX = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
    TY = np.full(Y.shape[0], TY) if np.ndim(TY) == 0 else np.asarray(TY)
    fields = hc5.time_fields(session, test)
    maps = [fields] if np.isfinite(fields).sum() >= 6 else []
    frng = np.random.default_rng(seed("waking-sep-wheel", name, fold, control))
    outs = waking_outputs(X, TX, Y, TY, frng, **SETTINGS)
    from experimental.scripts.dev_waking import judge_both
    rows = []
    srng = np.random.default_rng(seed("waking-sep-wheel-score", name, fold))
    for order, p, mask in outs:
        ok, k, ok2, k2, ok3, k3 = judge_both(order, maps, srng) if maps else (False, -1, False, -1, False, -1)
        rows.append({"p": p, "passes_check": p < 0.05, "n_members": int(mask.sum()), "passes": ok, "map": k,
                     "passes_partial": ok2, "map_partial": k2, "passes_share": ok3, "map_share": k3})
    return {"data": "hc-5 wheel" + (" (control)" if control else ""), "track": "wheel", "name": name, "fold": fold,
            "n_maps": len(maps), "outputs": rows, "n_train": int(X.shape[0])}


def summarize(rows):
    lines = ["# Waking un-mixing on real data", "", "Design and bars: `docs/unmixing.md` (waking design). Outputs = templates passing "
             "the waking check on held-out events. Clean = every scored map represented and every output passing.", "",
             "| data | track | criterion | fits | outputs passing | clean fits | fits with any output |", "|---|---|---|---|---|---|---|"]
    groups = [("hc-11 laps", "linear"), ("hc-11 laps", "circular"), ("hc-5 wheel", "wheel"),
              ("hc-5 wheel (control)", "wheel"), ("hc-11 laps (control)", "linear"), ("hc-11 laps (control)", "circular")]
    for data, track in groups:
        rs = [r for r in rows if r["data"] == data and r["track"] == track]
        if not rs:
            continue
        for crit, label in (("_partial", "partial correlation (fixed)"), ("_share", "variance share"), ("", "original")):
            P, K = "passes" + crit, "map" + crit
            outs = [[o for o in r["outputs"] if o["passes_check"]] for r in rs]
            n = sum(len(o) for o in outs)
            p = sum(o[P] for oo in outs for o in oo)
            if "control" in data:
                clean = sum(len(o) == 0 for o in outs)
            else:
                clean = sum(r["n_maps"] > 0 and len({o[K] for o in oo if o[P]}) == r["n_maps"] and all(o[P] for o in oo)
                            for r, oo in zip(rs, outs))
            lines.append(f"| {data} | {track} | {label} | {len(rs)} | {p}/{n} | {clean}/{len(rs)} | {sum(len(o) > 0 for o in outs)}/{len(rs)} |")
            if "control" in data:
                break
    lines += ["", "## Per fit (partial-correlation criterion)", "", "| data | session | fold | maps scored | outputs (map or fail) |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["data"], r["name"], r["fold"])):
        outs = [o for o in r["outputs"] if o["passes_check"]]
        desc = ", ".join(str(o["map_partial"]) if o["passes_partial"] else "fail" for o in outs) or "none"
        lines.append(f"| {r['data']} | {r['name']} | {r['fold']} | {r['n_maps']} | {desc} |")
    return "\n".join(lines)


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 9)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import cpca.hc5 as hc5
    jobs = [(lap_job, (n, f, c)) for n in HC11 for f in (0, 1) for c in (False, True)]
    jobs += [(wheel_job, (n, f, c)) for n in hc5.session_names() for f in (0, 1) for c in (False, True)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))
    text = summarize(rows)
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
