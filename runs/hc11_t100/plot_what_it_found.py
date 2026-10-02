"""Picture of the pattern found by the hc-11 rest run.

Reads session mats and uses cpca only to form the same mean arrows as the run.
Does not modify the library or the data.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_hc11_window import MAT_DIR, analyze_session, load_session
from cpca import encode
from runs.hc11_t100.run_cpca import WINDOW_MS, build_times

OUT = Path(__file__).resolve().parent / "what_the_run_found.png"
NAVY = "#1f4e79"
PEAK = "#b42318"


def session_means(path):
    item = analyze_session(load_session(path))
    cell_ids = np.array(sorted(item["by_cell"]))
    times, _, _ = build_times(item["by_cell"], item["bursts"], item["peaks"], cell_ids)
    encoded = encode(times, WINDOW_MS, how="mean", t0=0.0)
    mean = encoded.mean(axis=0)
    # Window start is angle 0. The population peak sits at 50 ms, straight up.
    when = WINDOW_MS / np.pi * np.angle(mean) - WINDOW_MS / 2
    when[mean == 0] = np.nan
    return mean, when


def short_label(name):
    rat, day = name.split("_")
    return f"{rat} {day[:2]}/{day[2:4]}"


def main():
    paths = sorted(MAT_DIR.glob("*_sessInfo.mat"))
    stored = [(path, *session_means(path)) for path in paths]
    example_name, example_mean, _ = stored[0]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "regular",
    })
    fig = plt.figure(figsize=(12.4, 8.4))
    arrows = fig.add_axes([0.06, 0.46, 0.40, 0.42])
    full = fig.add_axes([0.56, 0.46, 0.40, 0.42])
    zoom = fig.add_axes([0.10, 0.08, 0.84, 0.26])

    for value in example_mean:
        if value == 0:
            continue
        arrows.annotate(
            "",
            xy=(np.real(value), np.imag(value)),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 0.9, "alpha": 0.45, "mutation_scale": 8},
            zorder=2,
        )
    tips_x = np.real(example_mean)
    tips_y = np.imag(example_mean)
    arrows.annotate(
        "",
        xy=(0, np.max(tips_y) * 1.08),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": PEAK, "lw": 1.6, "mutation_scale": 12},
        zorder=3,
    )
    arrows.text(0.004, np.max(tips_y) * 1.1, "population peak", color=PEAK, fontsize=10, ha="left", va="bottom")
    half = max(float(np.max(np.abs(tips_x))) * 1.6, 0.02)
    arrows.set_xlim(-half, half)
    arrows.set_ylim(-0.02, float(np.max(tips_y)) * 1.28)
    arrows.axis("off")
    arrows.set_title(f"One arrow for each cell, {short_label(example_name.stem.replace('_sessInfo', ''))}", pad=8)
    arrows.text(
        0.0,
        -0.02,
        "Up is the population peak.\nSideways spread is stretched so a fan\nof a few degrees can be seen.",
        transform=arrows.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#333333",
    )

    for row, (path, _, when) in enumerate(stored):
        full.plot(when, np.full(when.shape, row), "|", color=NAVY, ms=7, alpha=0.55)
    full.axvline(0, color=PEAK, lw=1.2, zorder=0)
    full.axvspan(-20, 20, color="#e7eef6", zorder=0)
    full.set_yticks(range(len(stored)))
    full.set_yticklabels([short_label(path.stem.replace("_sessInfo", "")) for path, _, _ in stored], fontsize=8)
    full.set_xlim(-50, 50)
    full.set_ylim(-0.8, len(stored) - 0.2)
    full.set_xlabel("Milliseconds from the population peak")
    full.set_title("The same times, on the full window")
    full.spines["top"].set_visible(False)
    full.spines["right"].set_visible(False)
    full.tick_params(length=3)
    full.legend(
        handles=[
            Line2D([0], [0], color=PEAK, lw=1.2, label="population peak"),
            Patch(facecolor="#e7eef6", edgecolor="none", label="40 ms, a typical event"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper right",
    )

    pooled = np.concatenate([when for _, _, when in stored])
    bins = np.arange(-15, 15.5, 1)
    zoom.hist(pooled, bins=bins, color=NAVY, width=0.72, zorder=2)
    zoom.axvline(0, color=PEAK, lw=1.2, zorder=3)
    zoom.set_xlim(-15, 15)
    zoom.set_xlabel("Milliseconds from the population peak")
    zoom.set_ylabel("Cells")
    zoom.set_title("Up close, the cells cover only a few milliseconds")
    zoom.spines["top"].set_visible(False)
    zoom.spines["right"].set_visible(False)
    zoom.tick_params(length=3)
    zoom.yaxis.grid(True, color="#eceff3", zorder=0)
    zoom.set_axisbelow(True)

    fig.suptitle(
        "The pattern kept from hc-11 rest: cells fire together at the population peak",
        fontsize=13,
        y=0.98,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    kept = pooled[np.isfinite(pooled)]
    print(OUT)
    print("cells", kept.size, "std", round(kept.std(), 2))
    print("percentiles", np.round(np.percentile(kept, [5, 10, 50, 90, 95]), 2))


if __name__ == "__main__":
    main()
