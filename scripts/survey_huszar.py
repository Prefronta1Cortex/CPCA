"""Survey of DANDI 000552 (Huszar et al., 2022): what each processed session contains, read remotely (no download).

Per session: number of sorted units, epochs (sleep / maze / sleep), alternation trials and how many were
correct, detected ripples, and whether position is present. Units are not labelled by cell type in the
files; the count includes interneurons (the paper: 7,411 pyramidal cells and 2,060 interneurons over 92
sessions).

Usage: python scripts/survey_huszar.py [--threads=12]
Output: results/datasets/huszar_sessions.csv, printed summary.
"""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import remfile
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "datasets"
API = "https://api.dandiarchive.org/api"


def sessions():
    rows = requests.get(f"{API}/dandisets/000552/versions/draft/assets/?page_size=400", timeout=60).json()["results"]
    return sorted([a for a in rows if a["path"].endswith("behavior+ecephys.nwb")], key=lambda a: a["path"])


def read(asset):
    url = requests.head(f"{API}/assets/{asset['asset_id']}/download/", allow_redirects=True, timeout=60).url
    out = {"path": asset["path"], "size_mb": round(asset["size"] / 1e6)}
    try:
        with h5py.File(remfile.File(url), "r") as f:
            out["units"] = int(f["units/id"].shape[0]) if "units" in f else 0
            ep = f["intervals/epochs"] if "intervals/epochs" in f else None
            if ep is not None:
                dur = (ep["stop_time"][:] - ep["start_time"][:]) / 60
                out["epochs_min"] = " / ".join(f"{d:.0f}" for d in dur)
            if "intervals/trials" in f:
                t = f["intervals/trials"]
                out["trials"] = int(t["id"].shape[0])
                ok = t["visited_matched_expected"][:] if "visited_matched_expected" in t else None
                out["correct"] = int(np.nansum(np.asarray(ok, float) == 1)) if ok is not None else ""
                out["undetermined"] = int(np.sum(~np.isfinite(np.asarray(ok, float)))) if ok is not None else ""
                fam = t["is_familiar_maze"][:] if "is_familiar_maze" in t else None
                out["familiar_trials"] = int(np.nansum(np.asarray(fam, float) == 1)) if fam is not None else ""
            else:
                out["trials"] = 0
            rip = "processing/ecephys/Ripples"
            if rip in f:
                g = f[rip]
                inner = [k for k in g.keys()]
                n = None
                for k in inner:
                    obj = g[k]
                    if isinstance(obj, h5py.Group) and "start_time" in obj:
                        n = obj["start_time"].shape[0]
                    elif isinstance(obj, h5py.Dataset) and k in ("start_time", "id"):
                        n = obj.shape[0]
                out["ripples"] = n if n is not None else ""
            out["position"] = "processing/behavior/LinearizedPosition" in f
    except Exception as e:  # report and continue
        out["error"] = repr(e)[:120]
    return out


def main(argv):
    threads = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--threads=")), 12)
    assets = sessions()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        rows = list(pool.map(read, assets))
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["path", "size_mb", "units", "epochs_min", "trials", "correct", "undetermined", "familiar_trials", "ripples", "position", "error"]
    with open(OUT / "huszar_sessions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    units = np.array([r.get("units", 0) for r in rows])
    task = [r for r in rows if r.get("trials", 0) > 0]
    print(f"{len(rows)} processed sessions, {sum('error' in r for r in rows)} unreadable")
    print(f"units per session: median {np.median(units):.0f}, range {units.min()}-{units.max()}, total {units.sum()}")
    print(f"sessions with alternation trials: {len(task)}; trials per session median {np.median([r['trials'] for r in task]):.0f}; "
          f"units in those sessions median {np.median([r['units'] for r in task]):.0f} (range {min(r['units'] for r in task)}-{max(r['units'] for r in task)})")
    print(f"subjects with task sessions: {len({r['path'].split('/')[0] for r in task})}")
    good = [r for r in task if isinstance(r.get("correct"), int) and r["trials"]]
    if good:
        acc = np.array([r["correct"] / max(r["trials"] - (r.get("undetermined") or 0), 1) for r in good])
        print(f"choice accuracy per session: median {np.median(acc):.2f} (range {acc.min():.2f}-{acc.max():.2f}); "
              f"sessions with at least 3 errors: {sum((r['trials'] - (r.get('undetermined') or 0) - r['correct']) >= 3 for r in good)}")
    rip = [r["ripples"] for r in rows if isinstance(r.get("ripples"), int)]
    if rip:
        print(f"ripples per session: median {np.median(rip):.0f} (range {min(rip)}-{max(rip)}) in {len(rip)} sessions")


if __name__ == "__main__":
    main(sys.argv[1:])
