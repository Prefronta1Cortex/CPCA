"""Label-free enrichment of replay events on hc-11 (docs/segments.md, last section).

For every population burst that the paper's Bayesian analysis scored on our events (results/replay/<session>.pkl,
"ours", POST and PRE), label-free features are computed from the burst's spikes alone: duration, number of active
pyramidal cells, total spikes, spikes per active cell, spread of spike times (SD / duration), interquartile range of
spike times / duration, share of spikes in the central half of the window, and the largest 20 ms spike count / total.
Per feature and epoch: AUC for "clear replay" (p < 0.05 under both shuffles), and the clear-replay share in the top
30% and 50% of events by the feature against the share in all scored events. PRE is the control for biases of the
decoder's test itself (its power grows with active place cells and bins). The map enters only through the labels.

Usage: python scripts/check_enrichment.py [--workers=8]
Output: results/enrichment/<session>.pkl, results/enrichment/summary.md
"""

import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "enrichment"
REPLAY = ROOT / "results" / "replay"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
FEATURES = ["duration", "n_active", "n_spikes", "spikes_per_cell", "spread", "iqr", "central_share", "peak_share"]


def features_of(times, T):
    """Label-free features per event from spike times (events, cells, spikes) in ms and window durations T (ms)."""
    finite = np.isfinite(times)
    n_active = finite.any(-1).sum(1)
    n_spikes = finite.sum((1, 2))
    flat = times.reshape(times.shape[0], -1)
    rel = flat / T[:, None]
    rel = np.where(np.isfinite(rel), rel, np.nan)
    sd = np.nanstd(rel, axis=1)
    q75, q25 = np.nanpercentile(rel, 75, axis=1), np.nanpercentile(rel, 25, axis=1)
    central = np.nanmean((rel > 0.25) & (rel < 0.75), axis=1)
    peak = np.zeros(times.shape[0])
    for i in range(times.shape[0]):
        t = flat[i][np.isfinite(flat[i])]
        if t.size:
            edges = np.arange(0, T[i] + 20, 20)
            peak[i] = np.histogram(t, bins=edges)[0].max() / t.size
    return {"duration": T, "n_active": n_active.astype(float), "n_spikes": n_spikes.astype(float),
            "spikes_per_cell": n_spikes / np.maximum(n_active, 1), "spread": sd, "iqr": q75 - q25,
            "central_share": central, "peak_share": peak}


def auc(score, label):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    if label[ok].sum() < 3 or (~label[ok]).sum() < 3:
        return np.nan
    return float(roc_auc_score(label[ok], score[ok]))


def job(name):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, pyramidal_spikes, session_paths
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    rep = pickle.load(open(REPLAY / f"{name}.pkl", "rb"))["ours"]
    out = {"name": name, "epochs": {}}
    for epoch in ("post", "pre"):
        iv = nrem_intervals(session, epoch)
        by_cell, cell_ids, pooled = pyramidal_spikes(session, iv)
        bursts, peaks = population_bursts(pooled, iv)
        d = np.diff(bursts, axis=1).ravel()
        starts = peaks - d / 2
        scored = rep[epoch]
        if not scored:
            continue
        # match the scored events to the bursts by start time
        idx = np.searchsorted(starts, scored["start"])
        idx = np.clip(idx, 0, starts.size - 1)
        ok = np.abs(starts[idx] - scored["start"]) < 1e-6
        idx, keep = idx[ok], ok
        times, _ = event_spike_times(by_cell, cell_ids, starts[idx], starts[idx] + d[idx], max_spikes=60)
        feats = features_of(times, d[idx] * 1000)
        clear = (scored["p"][keep] < 0.05) & (scored["p_time"][keep] < 0.05)
        loose = scored["p"][keep] < 0.05
        block = {"n": int(idx.size), "n_clear": int(clear.sum()), "share": float(clear.mean()), "rZ": scored["rZ"][keep], "clear": clear,
                 "features": feats, "auc": {}, "top30": {}, "top50": {}, "auc_loose": {}, "rz_corr": {}}
        for f in FEATURES:
            x = feats[f]
            block["auc"][f] = auc(x, clear)
            block["auc_loose"][f] = auc(x, loose)
            order = np.argsort(-np.where(np.isfinite(x), x, -np.inf))
            for frac, key in ((0.3, "top30"), (0.5, "top50")):
                top = order[: max(1, int(round(frac * idx.size)))]
                block[key][f] = float(clear[top].mean())
            fin = np.isfinite(x) & np.isfinite(scored["rZ"][keep])
            block["rz_corr"][f] = float(np.corrcoef(x[fin], scored["rZ"][keep][fin])[0, 1]) if fin.sum() > 10 else np.nan
        out["epochs"][epoch] = block
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(OUT / f"{name}.pkl", "wb"))
    po = out["epochs"].get("post", {})
    return f"POST {po.get('n', 0)} events, clear {po.get('share', np.nan):.1%} | AUC " + ", ".join(f"{f} {po['auc'][f]:.2f}" for f in FEATURES) if po else "no POST"


def summary():
    R = {n: pickle.load(open(OUT / f"{n}.pkl", "rb")) for n in HC11_ORDER if (OUT / f"{n}.pkl").exists()}
    lines = ["# Label-free enrichment of clear replays on hc-11", "",
             "Produced by `scripts/check_enrichment.py`; design in `docs/segments.md`. Events: our POST and PRE bursts scored by the paper's "
             "Bayesian analysis (`results/replay/`); clear = p < 0.05 under both shuffles. Features use spikes only.", ""]
    for epoch in ("post", "pre"):
        lines += [f"## {epoch.upper()}: AUC of each feature for clear replay", "",
                  "| session | events | clear | " + " | ".join(FEATURES) + " |", "|---" * (len(FEATURES) + 3) + "|"]
        for n, r in R.items():
            b = r["epochs"].get(epoch)
            if not b:
                continue
            lines.append(f"| {n} | {b['n']} | {b['share']:.1%} | " + " | ".join(f"{b['auc'][f]:.2f}" for f in FEATURES) + " |")
        lines.append("")
    lines += ["## POST: clear-replay share in the top 50% of events by each feature (all events in brackets)", "",
              "| session | all | " + " | ".join(FEATURES) + " |", "|---" * (len(FEATURES) + 2) + "|"]
    for n, r in R.items():
        b = r["epochs"].get("post")
        if not b:
            continue
        lines.append(f"| {n} | {b['share']:.1%} | " + " | ".join(f"{b['top50'][f]:.1%}" for f in FEATURES) + " |")
    lines += ["", "## POST minus PRE: excess clear share, all events and in the top 50% by each feature (percentage points)", "",
              "| session | all | " + " | ".join(FEATURES) + " |", "|---" * (len(FEATURES) + 2) + "|"]
    for n, r in R.items():
        po, pr = r["epochs"].get("post"), r["epochs"].get("pre")
        if not po or not pr:
            continue
        lines.append(f"| {n} | {100 * (po['share'] - pr['share']):+.1f} | " +
                     " | ".join(f"{100 * (po['top50'][f] - pr['top50'][f]):+.1f}" for f in FEATURES) + " |")
    lines += ["", "## POST: Pearson correlation of each feature with rZ", "",
              "| session | " + " | ".join(FEATURES) + " |", "|---" * (len(FEATURES) + 1) + "|"]
    for n, r in R.items():
        b = r["epochs"].get("post")
        if b:
            lines.append(f"| {n} | " + " | ".join(f"{b['rz_corr'][f]:+.2f}" for f in FEATURES) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), 8)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    todo = [n for n in HC11_ORDER if not (OUT / f"{n}.pkl").exists()]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job, n): n for n in todo}
            for f in as_completed(futures):
                try:
                    print(f"{futures[f]}: {f.result()}", flush=True)
                except Exception:
                    print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
