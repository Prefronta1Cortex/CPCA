"""Un-mixing on waking sequences with a known answer (docs/unmixing.md, next step).

The confirmed pipeline (cpca.separate.separate: quartimax, refined templates, explaining away,
split into 2-4 orders kept only if each reproduces on the held-out half at similarity >= 0.97).
The latency check is not applied: on laps and wheel runs the sequence is the same in every event,
so that check rejects it (docs/readout_check.md). The map is used only to score.

Two event gates, both reported:
  confirmed   a template needs 20 expressing events and the split needs 80, as confirmed on planted data;
  open        both minima are 4 events, below every training half except one 3-run wheel fold,
              so the step is allowed to run on these recordings. The 0.97 reproduction cut is unchanged.

Score, fixed before the run. Fit on one half of the events, answer key from the other half.
An output passes when its order matches exactly one held-out map (p < 0.05) and not the other
(scripts.run_separation.judge). One map (circular laps, the wheel): it must match that map.
A fit is clean when every held-out map with at least 6 fields is represented and every output passes.
Plain components of the same fit are scored the same way.

hc-11 laps: split 0, both folds, uncentred (the sequence is in every lap) and centred.
hc-5 wheel: aligned windows, repeat 0, both folds, uncentred only.

Usage: python scripts/run_unmix_waking.py [--workers=8]
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

OUT = ROOT / "results" / "unmix_waking"
HC11 = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
        "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def orders_of(templates, T):
    from cpca.decode import readout
    out = []
    for t in templates:
        _, times, _, inc, _ = readout(t["template"], T, min_weight=1e-9)
        out.append((np.where(inc & t["members"], times, np.nan), "split_from" in t))
    return out


def component_orders(result):
    from cpca.decode import readout
    return [np.where(inc, t, np.nan) for _, t, _, inc, _ in
            (readout(result["pca_components"][:, j], result["T_read_ms"], min_weight=0.2) for j in range(result["n_kept"]))]


def score_orders(orders, maps, shared, rng):
    from scripts.run_separation import judge
    rows = []
    for order in orders:
        ok, k, det = judge(order, maps, shared, rng) if maps else (False, -1, {})
        rows.append({"passes": bool(ok), "map": k, **det})
    represented = {r["map"] for r in rows if r["passes"]}
    usable = list(range(len(maps)))
    clean = bool(maps) and all(r["passes"] for r in rows) and represented >= set(usable) and len(rows) > 0
    return rows, clean


def summarise(name, kind, convention, gate, n_train, maps, comp_rows, comp_clean, tmpl_rows, tmpl_clean, n_split, split_sim):
    return {"name": name, "kind": kind, "convention": convention, "gate": gate, "n_train": n_train,
            "n_maps": len(maps), "n_components": len(comp_rows), "components_pass": sum(r["passes"] for r in comp_rows),
            "components_clean": comp_clean, "n_templates": len(tmpl_rows), "templates_pass": sum(r["passes"] for r in tmpl_rows),
            "templates_clean": tmpl_clean, "n_from_split": n_split, "split_similarity": split_sim,
            "template_maps": [r["map"] for r in tmpl_rows]}


def split_similarity(temps):
    vals = [t.get("split_similarity", np.nan) for t in temps]
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


def lap_job(name, fold, center, min_events):
    from cpca import event_spike_times
    from cpca.encode import encode
    from cpca.hc11 import laps, load_session, place_centres, pyramidal_spikes, running, session_paths
    from experimental.separate import separate
    from cpca import fit
    s = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(s)
    L = laps(run)
    starts, ends, dirs = (np.array([x[i] for x in L]) for i in range(3))
    rng = np.random.default_rng(seed("unmix-laps", name))
    half = np.zeros(len(L), bool)
    for d in np.unique(dirs):
        idx = rng.permutation(np.flatnonzero(dirs == d))
        half[idx[: idx.size // 2]] = True
    train = half if fold == 0 else ~half
    by_cell, cell_ids, _ = pyramidal_spikes(s, np.column_stack([starts, ends]))
    tr, te = np.flatnonzero(train), np.flatnonzero(~train)
    times, _ = event_spike_times(by_cell, cell_ids, starts[tr], ends[tr], max_spikes=400)
    T = (ends[tr] - starts[tr]) * 1000
    te_times, _ = event_spike_times(by_cell, cell_ids, starts[te], ends[te], max_spikes=400)
    te_T = (ends[te] - starts[te]) * 1000
    from scripts.run_laps_hc11 import heldout_run
    centres = place_centres(s, cell_ids, heldout_run(run, np.column_stack([starts[te], ends[te]])), min_spikes=20)
    maps, usable = [], []
    for sign in (1.0, -1.0):
        c = centres[sign]
        if np.isfinite(c).sum() >= 6:
            maps.append(c)
            usable.append(sign)
    shared = len(maps) > 1
    frng = np.random.default_rng(seed("unmix-laps", name, fold, center, min_events))
    r = fit(times, T, rng=frng, ica=False, center=center, scale=center, n_shuffles=50)
    temps = []
    if r["n_kept"]:
        temps = separate(r, encode(times, T), frng, explain=True, split="k", validate=encode(te_times, te_T),
                         reproduce=0.97, min_events=min_events)
    srng = np.random.default_rng(seed("unmix-laps-score", name, fold, center, min_events))
    comp, comp_clean = score_orders(component_orders(r), maps, shared, srng)
    tmpl, tmpl_clean = score_orders([o for o, _ in orders_of(temps, r["T_read_ms"])], maps, shared, srng)
    return summarise(name, "laps", "centred" if center else "uncentred", min_events, len(tr), maps,
                     comp, comp_clean, tmpl, tmpl_clean, sum(1 for t in temps if "split_from" in t), split_similarity(temps))


def wheel_job(name, fold, min_events):
    import cpca.hc5 as hc5
    from cpca import fit
    from cpca.encode import encode
    from experimental.separate import separate
    from scripts.run_hc5 import training_data
    session = hc5.load_session(name)
    times, T, train, test, _ = training_data(session, "aligned", 0, fold)
    fields = hc5.time_fields(session, test)
    maps = [fields] if np.isfinite(fields).sum() >= 6 else []
    frng = np.random.default_rng(seed("unmix-hc5", name, fold, min_events))
    T_arr = np.full(times.shape[0], T) if np.ndim(T) == 0 else T
    r = fit(times, T_arr, rng=frng, ica=False, center=False, scale=False, n_shuffles=50)
    te_times, te_T, _, _, _ = training_data(session, "aligned", 0, 1 - fold)
    te_T = np.full(te_times.shape[0], te_T) if np.ndim(te_T) == 0 else te_T
    temps = []
    if r["n_kept"]:
        temps = separate(r, encode(times, T_arr), frng, explain=True, split="k", validate=encode(te_times, te_T),
                         reproduce=0.97, min_events=min_events)
    srng = np.random.default_rng(seed("unmix-hc5-score", name, fold, min_events))
    comp, comp_clean = score_orders(component_orders(r), maps, False, srng)
    tmpl, tmpl_clean = score_orders([o for o, _ in orders_of(temps, r["T_read_ms"])], maps, False, srng)
    return summarise(name, "wheel", "uncentred", min_events, len(train), maps, comp, comp_clean, tmpl, tmpl_clean,
                     sum(1 for t in temps if "split_from" in t), split_similarity(temps))


def line(r):
    return (f"{r['kind']:5} {r['name']:22} {r['convention']:10} gate {r['gate']:3}  train {r['n_train']:3}  "
            f"maps {r['n_maps']}  components {r['components_pass']}/{r['n_components']} clean {int(r['components_clean'])}  "
            f"templates {r['templates_pass']}/{r['n_templates']} clean {int(r['templates_clean'])}  "
            f"from-split {r['n_from_split']}  split-sim {r['split_similarity'] if np.isfinite(r['split_similarity']) else float('nan'):.2f}")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    jobs = []
    for gate in (80, 4):
        jobs += [(lap_job, (n, f, c, gate)) for n in HC11 for f in (0, 1) for c in (False, True)]
        jobs += [(wheel_job, (n, f, gate)) for n in hc5_names() for f in (0, 1)]
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): a for fn, a in jobs}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception:
                print(f"FAILED {futures[fut]}:\n{traceback.format_exc()}", flush=True)
                continue
            rows.append(row)
            print(line(row), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(OUT / "rows.pkl", "wb"))
    text = report(rows)
    (OUT / "summary.md").write_text(text + "\n")
    print("\n" + text)


def hc5_names():
    import cpca.hc5 as hc5
    return hc5.session_names()


def best_sim(rows):
    vals = [r["split_similarity"] for r in rows if np.isfinite(r["split_similarity"])]
    return float(max(vals)) if vals else float("nan")


def report(rows):
    lines = ["# Un-mixing on waking sequences", "",
             "Fit on one half, scored on the other half's fields. No latency check. "
             "Clean = every held-out map is represented and every output matches exactly one map.", ""]
    for kind in ("laps", "wheel"):
        for gate in (80, 4):
            sub = [r for r in rows if r["kind"] == kind and r["gate"] == gate]
            lines.append(f"## {kind}, split minimum {gate} events")
            for conv in sorted({r["convention"] for r in sub}):
                c = [r for r in sub if r["convention"] == conv]
                nt, pt = sum(r["n_templates"] for r in c), sum(r["templates_pass"] for r in c)
                nc, pc = sum(r["n_components"] for r in c), sum(r["components_pass"] for r in c)
                lines.append(f"- {conv}: components {pc}/{nc} pass, {sum(r['components_clean'] for r in c)}/{len(c)} fits clean; "
                             f"templates {pt}/{nt} pass, {sum(r['templates_clean'] for r in c)}/{len(c)} fits clean; "
                             f"outputs from a split {sum(r['n_from_split'] for r in c)}; "
                             f"best held-out split similarity {best_sim(c):.2f}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
