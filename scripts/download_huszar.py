"""Download the processed task sessions of DANDI 000552 (Huszar et al., 2022) to data/huszar/ (git-ignored).

Only `*_behavior+ecephys.nwb` files of sessions with alternation trials (the raw `*-raw_ecephys.nwb` files,
about 1.5 TB, are not needed). The list of task sessions is results/datasets/huszar_sessions.csv
(scripts/survey_huszar.py). Existing complete files are skipped.

Usage: python scripts/download_huszar.py [--threads=6]
"""

import csv
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "huszar"
API = "https://api.dandiarchive.org/api"


def fetch(asset):
    target = OUT / Path(asset["path"]).name
    if target.exists() and target.stat().st_size == asset["size"]:
        return target.name, "cached"
    tmp = target.with_suffix(".part")
    with requests.get(f"{API}/assets/{asset['asset_id']}/download/", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=1 << 22)
    tmp.replace(target)
    return target.name, f"{asset['size'] / 1e6:.0f} MB"


def main(argv):
    threads = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--threads=")), 6)
    task = {r["path"] for r in csv.DictReader(open(ROOT / "results" / "datasets" / "huszar_sessions.csv")) if r["trials"] and int(r["trials"]) > 0}
    assets = [a for a in requests.get(f"{API}/dandisets/000552/versions/draft/assets/?page_size=400", timeout=60).json()["results"]
              if a["path"] in task]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, (name, status) in enumerate(pool.map(fetch, assets), 1):
            print(f"[{i}/{len(assets)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
