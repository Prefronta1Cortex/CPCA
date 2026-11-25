"""Independent check of Gemini's pillars 1, 2 and 4 at the current arc (pi), before adopting them.

Fresh seeds ('verify'), the project's own scoring functions. Real data only:
  hc-5 wheel runs and hc-11 laps (waking regime), scored per output on held-out fields;
  hc-11 spike-in power (two largest sessions, ideal replays in the matched 1500 PRE events);
  hc-11 rest on the maze: order test (best output) and 20 timing-shuffled refits.
Usage: python exploration/verify_adoption.py [arc multiple of pi, default 1.0]
"""

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARC = float(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 1.0
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*p):
    return zlib.crc32("|".join(map(str, ("verify",) + p)).encode())


def cfg(regime):
    return dict(regime=regime, alpha=ARC * np.pi)


def clean_of(verdicts, n_maps):
    return bool(verdicts) and len({k for ok, k in verdicts if ok}) == n_maps and all(ok for ok, _ in verdicts)


def hc5(args):
    name, fold, cond = args
    import cpca.hc5 as hc5
    from exploration.cpca_v2 import fit_cpca_v2
    from scripts.run_hc5 import training_data
    from scripts.run_pure_capabilities import judge_share
    s = hc5.load_session(name)
    X, T, train, test, _ = training_data(s, cond, 0, fold)
    T = np.full(X.shape[0], T) if np.ndim(T) == 0 else np.asarray(T)
    f = hc5.time_fields(s, test)
    res = fit_cpca_v2(X, T, np.random.default_rng(seed("hc5", name, fold, cond)), **cfg("waking"))
    v = [judge_share(o, [f], np.random.default_rng(seed("hc5j", name, fold, cond, i))) for i, o in enumerate(res["orders"])]
    return ("hc-5 " + cond, clean_of(v, 1) if cond != "shuffled" else len(res["orders"]) == 0, len(v), sum(ok for ok, _ in v))


def hc11_laps(args):
    name, fold = args
    from cpca import event_spike_times
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from exploration.cpca_v2 import fit_cpca_v2
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_laps_hc11 import heldout_run
    from scripts.run_pure_capabilities import judge_share
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    st, en, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng0 = np.random.default_rng(seed("half", name))
    half = np.zeros(len(L), bool)
    for d in np.unique(dirs):
        idx = rng0.permutation(np.flatnonzero(dirs == d))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    by_cell, ids, _ = pyramidal_spikes(s, np.column_stack([st, en]))
    X, _ = event_spike_times(by_cell, ids, st[tr], en[tr], max_spikes=400)
    c = place_centres(s, ids, heldout_run(run, np.column_stack([st[te], en[te]])), min_spikes=20)
    maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs[te] == sg) >= 3]
    res = fit_cpca_v2(X, (en[tr] - st[tr]) * 1000, np.random.default_rng(seed("hc11", name, fold)), **cfg("waking"))
    jr = np.random.default_rng(seed("hc11j", name, fold))
    if run["circular"]:
        v = [(bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                   best_map_match([o], {1.0: maps[0]}, True, run["length"], jr, n_null=500)["p"] < 0.05), 0) for o in res["orders"]]
    else:
        v = [judge_share(o, maps, jr) for o in res["orders"]]
    return ("hc-11 laps " + ("circular" if run["circular"] else "linear"), clean_of(v, len(maps)), len(v), sum(ok for ok, _ in v))


def _pre_events(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths
    from scripts.run_rerun import ORDER_INDEX, fixed_subset
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    iv = nrem_intervals(s, "pre")
    by_cell, ids, pooled = pyramidal_spikes(s, iv)
    b, pk = population_bursts(pooled, iv)
    dur = np.diff(b, axis=1).ravel()
    sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[name] + 1)
    base, _ = event_spike_times(by_cell, ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
    return base, dur[sub] * 1000, place_centres(s, ids, run), run


def spikein(args):
    name, f, d = args
    from exploration.cpca_v2 import fit_cpca_v2
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_spikein_hc11 import ideal
    base, T, centres, run = _pre_events(name)
    rng = np.random.default_rng(seed("spike", name, f, d))
    k = int(round(f * len(T)))
    times = ideal(base, T, rng.choice(len(T), k, replace=False), centres, rng) if k else base
    res = fit_cpca_v2(times, T, np.random.default_rng(seed("spikefit", name, f, d)), **cfg("reactivation"))
    p = best_map_match(res["orders"], centres, run["circular"], run["length"], np.random.default_rng(seed("spikesc", name, f, d)),
                       n_null=500)["p"] if res["orders"] else np.nan
    return ("spike-in", name, f, bool(np.isfinite(p) and p < 0.05))


def maze_rest(args):
    name, = args
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, place_centres, pyramidal_spikes, running, session_paths
    from cpca.replay import immobile_intervals
    from exploration.cpca_v2 import fit_cpca_v2
    from scripts.plot_hc11_sequences import best_map_match
    from scripts.run_maze_rest import shuffle_timing
    from scripts.run_replay_hc11 import intersect, two_d
    path = next(p for p in session_paths() if p.name.startswith(name))
    s = load_session(path)
    s["pos_xy"] = two_d(path)
    run = running(s)
    iv = intersect(immobile_intervals(s), [tuple(s["maze_epoch"])])
    by_cell, ids, pooled = pyramidal_spikes(s, iv)
    b, pk = population_bursts(pooled, iv)
    dur = np.diff(b, axis=1).ravel()
    times, _ = event_spike_times(by_cell, ids, pk - dur / 2, pk + dur / 2, max_spikes=60)
    T = dur * 1000
    centres = place_centres(s, ids, run)

    def score(tt, tag):
        res = fit_cpca_v2(tt, T, np.random.default_rng(seed("maze", name, tag)), **cfg("reactivation"))
        if not res["orders"]:
            return np.nan, np.nan
        sc = best_map_match(res["orders"], centres, run["circular"], run["length"], np.random.default_rng(seed("mazesc", name, tag)), n_null=500)
        return sc["abs"], sc["p"]

    real_abs, real_p = score(times, "real")
    shuf = [score(shuffle_timing(times, np.random.default_rng(seed("mazesh", name, i))), f"sh{i}") for i in range(20)]
    s_abs = np.array([a for a, _ in shuf], float)
    fp = sum(np.isfinite(p) and p < 0.05 for _, p in shuf)
    p_tim = (1 + np.sum(np.nan_to_num(s_abs, nan=-1) >= real_abs)) / 21 if np.isfinite(real_abs) else 1.0
    return ("maze rest", name, real_p, p_tim, fp)


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    import cpca.hc5 as hc5m
    with ProcessPoolExecutor(9) as pool:
        waking = list(pool.map(hc5, [(n, f, c) for n in hc5m.session_names() for f in (0, 1) for c in ("aligned", "unaligned", "shuffled")]))
        waking += list(pool.map(hc11_laps, [(n, f) for n in HC11 for f in (0, 1)]))
        spikes = list(pool.map(spikein, [(n, f, d) for n in HC11[:2] for f in (0.0, 0.02, 0.03, 0.05, 0.10) for d in range(5)]))
        maze = list(pool.map(maze_rest, [(n,) for n in HC11]))
    print(f"arc {ARC} pi")
    for g in sorted({w[0] for w in waking}):
        rs = [w for w in waking if w[0] == g]
        print(f"  {g:24} clean fits {sum(r[1] for r in rs)}/{len(rs)}  outputs one route {sum(r[3] for r in rs)}/{sum(r[2] for r in rs)}")
    for f in (0.0, 0.02, 0.03, 0.05, 0.10):
        print(f"  spike-in {f:.0%}: {sum(s[3] for s in spikes if s[2] == f)}/10")
    for m in maze:
        print(f"  maze rest {m[1][:12]}: order p {m[2]:.3f}  timing p {m[3]:.3f}  shuffled fits passing order {m[4]}/20")
