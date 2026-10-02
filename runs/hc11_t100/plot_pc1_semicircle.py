"""Unit semicircle of the first rest component and the mean arrows.

Achilles 10/25, 100 ms windows, mean encoding. Does not modify the library
or the recordings.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from analyze_hc11_window import MAT_DIR, analyze_session, load_session
from cpca import decompose, encode
from runs.hc11_t100.run_cpca import build_times

OUT = Path(__file__).resolve().parent / "pc1_semicircle.png"
NAVY = "#1f4e79"
RED = "#b42318"


def main():
    path = next(p for p in sorted(MAT_DIR.glob("*_sessInfo.mat")) if "Achilles_1025" in p.name)
    item = analyze_session(load_session(path))
    cell_ids = np.array(sorted(item["by_cell"]))
    times, _, _ = build_times(item["by_cell"], item["bursts"], item["peaks"], cell_ids)
    encoded = encode(times, 100.0, how="mean", t0=0.0)
    mean = encoded.mean(axis=0)
    _, components, _, _ = decompose(encoded, n_components=1)
    component = components[:, 0]
    # Put the component in the same clock as the mean arrows. decompose()
    # anchors the strongest cell at angle 0, which is the window start.
    component = component * np.exp(1j * np.angle(np.vdot(component, mean)))
    component = component * (np.max(np.abs(mean)) / np.max(np.abs(component)))

    angles = np.angle(mean[mean != 0])
    arc = np.degrees(np.percentile(angles, 95) - np.percentile(angles, 5))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    theta = np.linspace(0, np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#c5cdd6", lw=1.4, zorder=0)
    ax.plot([-1, 1], [0, 0], color="#c5cdd6", lw=1.0, zorder=0)
    for value, color, width, alpha in (
        (mean, NAVY, 1.15, 0.55),
        (component, RED, 0.7, 0.35),
    ):
        for z in value:
            if z == 0:
                continue
            ax.annotate(
                "",
                xy=(np.real(z), np.imag(z)),
                xytext=(0, 0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": width,
                    "alpha": alpha,
                    "mutation_scale": 8,
                },
                zorder=2,
            )
    ax.plot(0, 1, marker="o", color="#1f8a65", ms=5, zorder=3)
    ax.text(1.08, -0.02, "window start", ha="left", va="top", fontsize=10, color="#444444")
    ax.text(-1.08, -0.02, "window end", ha="right", va="top", fontsize=10, color="#444444")
    ax.text(0.0, 1.08, "population peak", ha="center", va="bottom", fontsize=10, color="#1f8a65")
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.45)
    ax.set_ylim(-0.22, 1.28)
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color=NAVY, lw=1.6, label="mean arrow of each cell"),
            Line2D([0], [0], color=RED, lw=1.2, label="first component, same clock"),
        ],
        frameon=False,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(-0.02, 0.42),
    )
    ax.set_title("Achilles 10/25. The half-circle is the 100 ms window.", pad=12)
    fig.text(
        0.5,
        0.02,
        f"The first component lies on the mean arrows. "
        f"90% of those arrows span {arc:.0f}° of the 180° half-circle.",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(OUT)
    print("arc degrees", round(arc, 1), "alignment", round(abs(np.vdot(components[:, 0], mean)) / np.linalg.norm(mean), 4))


if __name__ == "__main__":
    main()
