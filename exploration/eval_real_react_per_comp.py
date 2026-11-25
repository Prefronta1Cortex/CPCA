"""Per-component evaluation of cpca_v2 (alpha=pi and 1.5*pi) on real reactivations
(hc-11 and DANDI 001695: maze rest, POST, PRE) using the EXACT same splits and scoring
as results/pure/ and results/initial_real/, PLUS real awake-replay spike-ins on hc-11.
"""

import pickle
import sys
import time
import zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exploration.cpca_v2 import fit_cpca_v2
from cpca import event_spike_times, population_bursts
from cpca.hc11 import laps, load_session as load_hc11, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths as hc11_paths
import cpca.dandi001695 as dd
from cpca.replay import immobile_intervals
from scripts.run_replay_hc11 import intersect, two_d
from scripts.plot_hc11_sequences import best_map_match
from scripts.run_pure_capabilities import judge_share, HC11
from scripts.run_separation import judge
from scripts.run_rerun import ORDER_INDEX, fixed_subset
from experimental.scripts.dev_unmixing import judge_partial

OUT = ROOT / "exploration" / "results"
REPLAY = ROOT / "results" / "replay"


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def score_orders(orders, maps, circular, length, rng):
    """Per-output verdicts matching scripts/run_pure_capabilities.py and experimental/scripts/run_initial_real.py."""
    res = []
    for o in orders:
        if not maps:
            res.append({"partial": False, "share": False, "orig": False})
        elif circular:
            ok = bool(np.sum(np.isfinite(o) & np.isfinite(maps[0])) >= 6 and
                      best_map_match([o], {1.0: maps[0]}, True, length, rng, n_null=500)["p"] < 0.05)
            res.append({"partial": ok, "share": ok, "orig": ok})
        elif len(maps) == 1:
            ok = judge(o, maps, True, rng)[0]
            res.append({"partial": ok, "share": ok, "orig": ok})
        else:
            ok_o = judge(o, maps, True, rng)[0]
            ok_s = judge_share(o, maps, rng)[0]
            ok_p = judge_partial(o, maps, rng)[0]
            res.append({"partial": ok_p, "share": ok_s, "orig": ok_o})
    return res


def eval_dataset(dataset, alpha):
    keys = HC11 if dataset == "hc-11" else list(dd.session_paths())
    out = {}
    for ep in ("maze", "post", "pre"):
        n_fits = 0
        n_out = 0
        p_part = p_share = p_orig = 0
        exp_chance = 0.0
        sess_hit = set()
        sess_all = set()
        best_pass = 0
        for key in keys:
            if dataset == "hc-11":
                path = next(p for p in hc11_paths() if p.name.startswith(key))
                s = load_hc11(path)
                s["pos_xy"] = two_d(path)
                sname = key
            else:
                s = dd.load_session(key)
                sname = key.name
            run = running(s)
            sess_all.add(sname)

            def bursts(iv):
                bc, ids, pl = pyramidal_spikes(s, iv)
                b, pk = population_bursts(pl, iv)
                return bc, ids, pk, np.diff(b, axis=1).ravel()

            rng = np.random.default_rng(seed("pure-react", dataset, str(key), ep))
            if ep == "maze":
                by_cell, cell_ids, pk, d = bursts(intersect(immobile_intervals(s), [tuple(s["maze_epoch"])]))
            else:
                n = min(len(bursts(nrem_intervals(s, e))[2]) for e in ("pre", "post"))
                by_cell, cell_ids, pk, d = bursts(nrem_intervals(s, ep))
                if len(pk) == 0 or n == 0:
                    continue
                pick = np.sort(rng.choice(len(pk), n, replace=False))
                pk, d = pk[pick], d[pick]
            if len(pk) < 40:
                continue
            times, _ = event_spike_times(by_cell, cell_ids, pk - d / 2, pk + d / 2, max_spikes=60)
            T = d * 1000
            perm = rng.permutation(len(pk))
            halves = (perm[: perm.size // 2], perm[perm.size // 2:])
            dirs = np.array([x[2] for x in laps(run)])
            c = place_centres(s, cell_ids, run)
            maps = [c[sg] for sg in (1.0, -1.0) if np.isfinite(c[sg]).sum() >= 6 and np.sum(dirs == sg) >= 3]

            for fold in (0, 1):
                tr = halves[fold]
                res = fit_cpca_v2(times[tr], T[tr], np.random.default_rng(seed("pc-fit", dataset, sname, ep, fold)),
                                  regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                                  n_shuffles=100, quantile=99.0, member_frac=0.2)
                n_fits += 1
                orders = res["orders"]
                vs = score_orders(orders, maps, bool(run["circular"]), run["length"],
                                  np.random.default_rng(seed("pc-sc", dataset, sname, ep, fold)))
                n_out += len(vs)
                p_part += sum(v["partial"] for v in vs)
                p_share += sum(v["share"] for v in vs)
                p_orig += sum(v["orig"] for v in vs)
                exp_chance += len(vs) * (0.095 if len(maps) == 2 else 0.05)
                if any(v["partial"] for v in vs):
                    sess_hit.add(sname)
                # Also check best-of-outputs on this fold
                if orders and maps:
                    bm = best_map_match(orders, c, bool(run["circular"]), run["length"],
                                        np.random.default_rng(seed("pc-bm", dataset, sname, ep, fold)), n_null=500)
                    best_pass += bool(np.isfinite(bm["p"]) and bm["p"] < 0.05)

        out[ep] = {
            "fits": n_fits, "outputs": n_out,
            "partial": p_part, "share": p_share, "orig": p_orig,
            "expected": exp_chance, "sess_hit": len(sess_hit), "sess_tot": len(sess_all),
            "best_pass": best_pass,
        }
    return out


def eval_awake_spikein(alpha):
    """Real awake replays from hc-11 maze rest spiked into matched 1500 PRE events (docs/replay.md)."""
    fracs = [0.01, 0.02, 0.03, 0.05]
    out = {}
    for sname in ("Achilles_10252013", "Achilles_11012013", "Cicero_09012014", "Cicero_09172014", "Gatsby_08022013"):
        rep_path = REPLAY / f"{sname}.pkl"
        if not rep_path.exists():
            continue
        rep = pickle.load(open(rep_path, "rb"))["paper"]["maze"]
        sig = (rep["p"] < 0.05) & (rep["p_time"] < 0.05)
        if sig.sum() < 15:
            continue
        path = next(p for p in hc11_paths() if p.name.startswith(sname))
        s = load_hc11(path); run = running(s)
        iv = nrem_intervals(s, "pre")
        by_cell, cell_ids, pooled = pyramidal_spikes(s, iv)
        b, pk = population_bursts(pooled, iv); dur = np.diff(b, axis=1).ravel()
        sub = fixed_subset(len(pk), 555 + 2 * ORDER_INDEX[sname] + 1)
        base, _ = event_spike_times(by_cell, cell_ids, (pk - dur / 2)[sub], (pk + dur / 2)[sub], max_spikes=60)
        T = dur[sub] * 1000; centres = place_centres(s, cell_ids, run)
        maze_cells = {c: np.sort(s["times"][s["ids"] == c]) for c in cell_ids}
        awake, _ = event_spike_times(maze_cells, cell_ids, rep["start"][sig], rep["end"][sig], max_spikes=60)
        awake_T = (rep["end"][sig] - rep["start"][sig]) * 1000
        row = {"n_awake": int(len(awake_T)), "hits": {}}
        for f in fracs:
            k = int(round(f * len(T)))
            if k > len(awake_T):
                continue
            hits = 0
            for d in range(5):
                rng = np.random.default_rng(seed("spikein", sname, "awake", f, d))
                chosen = rng.choice(len(T), k, replace=False)
                pick = rng.choice(len(awake_T), k, replace=False)
                times = base.copy(); Te = T.copy()
                times[chosen] = awake[pick]; Te[chosen] = awake_T[pick]
                res = fit_cpca_v2(times, Te, np.random.default_rng(seed("aw-fit", sname, f, d)),
                                  regime="reactivation", alpha=alpha, power=1.5, scale="global_rms",
                                  n_shuffles=100, quantile=99.0, member_frac=0.2)
                sc = best_map_match(res["orders"], centres, run["circular"], run["length"],
                                    np.random.default_rng(seed("aw-sc", sname, f, d)), n_null=500) if res["orders"] else {"p": np.nan}
                hits += bool(np.isfinite(sc["p"]) and sc["p"] < 0.05)
            row["hits"][f] = hits
        out[sname] = row
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    all_res = {}
    for label, alpha in (("pi", np.pi), ("1.5pi", 1.5 * np.pi)):
        print(f"\n================ REAL REACTIVATIONS PER-COMPONENT: alpha={label} ================", flush=True)
        for ds in ("hc-11", "001695"):
            r = eval_dataset(ds, alpha)
            all_res[(ds, label)] = r
            print(f"  [{ds}]", flush=True)
            for ep in ("maze", "post", "pre"):
                d = r[ep]
                pct = 100 * d["partial"] / max(d["outputs"], 1)
                print(f"    {ep:<4}: fits={d['fits']:2d} outs={d['outputs']:3d} | partial={d['partial']:2d} ({pct:4.1f}%) share={d['share']:2d} orig={d['orig']:2d} (chance={d['expected']:4.1f}) | sess={d['sess_hit']}/{d['sess_tot']} | best_pass={d['best_pass']}/{d['fits']}", flush=True)
        aw = eval_awake_spikein(alpha)
        all_res[("awake_spikein", label)] = aw
        print(f"  [hc-11 real awake-replay spike-in]", flush=True)
        for sn, rw in aw.items():
            print(f"    {sn:<18} (n_awake={rw['n_awake']:3d}): " + " ".join(f"{f:.0%}:{h}/5" for f, h in rw["hits"].items()), flush=True)
    pickle.dump(all_res, open(OUT / "real_react_per_comp.pkl", "wb"))
    print(f"\nDone in {time.perf_counter()-t0:.1f}s", flush=True)
