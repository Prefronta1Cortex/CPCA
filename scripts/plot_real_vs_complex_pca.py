"""Geometry of real PCA versus spike-phase PCA. Does not import the library."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "real_vs_complex_pca.png"

CELL_COLORS = ("#1f4e79", "#d4652f", "#1f8a65")
CELL_NAMES = ("cell 0  ·  0 ms", "cell 1  ·  20 ms", "cell 2  ·  40 ms")


def leading(A):
    C = (A.T @ A) / len(A)
    w, V = np.linalg.eigh(C)
    k = int(np.argmax(w))
    v = V[:, k]
    return w[k], v


def spike_matrix(shift, n_events, rng):
    truth = np.array([0.0, 20.0, 40.0])
    T = 100.0
    times = np.tile(truth, (n_events, 1)) + rng.normal(0, 4.0, size=(n_events, 3))
    times[rng.random(times.shape) < 0.25] = np.nan
    if shift:
        times = times + rng.uniform(0, shift, size=(n_events, 1))
    S = np.exp(1j * np.pi * times / T)
    S[~np.isfinite(times)] = 0
    return S


def component(S):
    C = (S.T @ np.conj(S)) / len(S)
    H = (C + C.conj().T) / 2
    w, V = np.linalg.eigh(H)
    p = V[:, int(np.argmax(w.real))]
    p = p * np.exp(-1j * np.angle(p[0]))
    return p


def draw_arrow(ax, start, end, color, width, alpha, zorder):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
            alpha=alpha,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def real_panel(ax, centered):
    rng = np.random.default_rng(2)
    x = rng.normal(8.0, 0.28, size=80)
    y = rng.normal(1.6, 1.5, size=80)
    pts = np.column_stack([x, y])
    if centered:
        pts = pts - pts.mean(axis=0)
    _, v = leading(pts)
    if centered and v[1] < 0:
        v = -v
    if not centered and v[0] < 0:
        v = -v
    length = 3.2 if centered else 8.3
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c="#1f4e79", alpha=0.75, linewidths=0, zorder=3)
    draw_arrow(ax, (0, 0), (v[0] * length, v[1] * length), "#b42318", 2.2, 1, 4)
    ax.axhline(0, color="#222", lw=0.8)
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("source 1")
    ax.set_ylabel("source 2")
    if centered:
        ax.set_title("Real data, mean removed\narrow follows the spread")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
    else:
        ax.set_title("Real data, mean kept\narrow points at the cloud")
        ax.set_xlim(-1.5, 10)
        ax.set_ylim(-4.2, 6.2)


def complex_panel(ax, S, title):
    T = 100.0
    for deg, label in ((0, "0 ms"), (36, "20 ms"), (72, "40 ms")):
        rad = np.deg2rad(deg)
        ax.plot([0, np.cos(rad)], [0, np.sin(rad)], color="#c8c8c8", lw=1, ls=(0, (2, 2)), zorder=1)
        ax.text(1.14 * np.cos(rad), 1.14 * np.sin(rad), label, ha="center", va="center", color="#666", fontsize=9)
    shown = S[:120]
    for event in shown:
        for cell in range(3):
            z = event[cell]
            if z == 0:
                continue
            draw_arrow(ax, (0, 0), (z.real, z.imag), CELL_COLORS[cell], 0.6, 0.18, 2)
    p = component(S)
    scale = 1.0 / np.max(np.abs(p))
    for cell in range(3):
        z = p[cell] * scale
        draw_arrow(ax, (0, 0), (z.real, z.imag), CELL_COLORS[cell], 2.4, 1, 5)
    circle = plt.Circle((0, 0), 1, fill=False, ec="#ddd", lw=0.8, zorder=1)
    ax.add_patch(circle)
    ax.axhline(0, color="#222", lw=0.8)
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title(title)


def main():
    rng = np.random.default_rng(0)
    shared = spike_matrix(0, 2000, rng)
    spun = spike_matrix(200, 2000, rng)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.4))
    real_panel(axes[0, 0], centered=False)
    real_panel(axes[0, 1], centered=True)
    complex_panel(axes[1, 0], shared, "Complex data, one shared start\nthin arrows are events, thick arrows are the component")
    complex_panel(axes[1, 1], spun, "Complex data, each event has its own start\nthe component still sits on 0, 20, 40 ms")

    handles = [
        plt.Line2D([0], [0], color=c, lw=3, label=name) for c, name in zip(CELL_COLORS, CELL_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("What one component looks like", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
