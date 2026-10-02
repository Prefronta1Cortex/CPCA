"""Why ordinary complex data is centered, and why spike-phase data is not.

Does not import or modify the library.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures" / "centering_contrast.png"
CELL_COLORS = ("#1f4e79", "#d4652f", "#1f8a65")


def arrow(ax, start, end, color, width, alpha, zorder, scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            color=color,
            alpha=alpha,
            zorder=zorder,
            shrinkA=0,
            shrinkB=0,
        )
    )


def leading_real(points):
    cov = (points.T @ points) / len(points)
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(np.argmax(values))]
    return float(values.max()), direction


def spike_matrix(shift, n_events, rng):
    truth = np.array([0.0, 20.0, 40.0])
    times = np.tile(truth, (n_events, 1)) + rng.normal(0.0, 4.0, size=(n_events, 3))
    times[rng.random(times.shape) < 0.25] = np.nan
    if shift:
        times = times + rng.uniform(0.0, shift, size=(n_events, 1))
    encoded = np.exp(1j * np.pi * times / 100.0)
    encoded[~np.isfinite(times)] = 0
    return encoded


def component(encoded):
    lag = (encoded.T @ np.conj(encoded)) / len(encoded)
    hermitian = (lag + lag.conj().T) / 2
    values, vectors = np.linalg.eigh(hermitian)
    weights = vectors[:, int(np.argmax(values.real))]
    return weights * np.exp(-1j * np.angle(weights[0]))


def style_axes(ax):
    ax.set_aspect("equal")
    ax.axhline(0, color="#222222", lw=0.8, zorder=0)
    ax.axvline(0, color="#222222", lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def panel_ordinary(ax):
    """One complex channel: a baseline away from 0, plus a smaller wobble."""
    rng = np.random.default_rng(1)
    wobble = rng.normal(size=(250, 2)) @ np.array([[0.15, 0.95], [0.05, 0.0]])
    points = wobble + np.array([2.4, 0.35])
    mean = points.mean(axis=0)
    _, toward_mean = leading_real(points)
    if np.dot(toward_mean, mean) < 0:
        toward_mean = -toward_mean
    _, along_wobble = leading_real(points - mean)
    if along_wobble[1] < 0:
        along_wobble = -along_wobble

    ax.scatter(points[:, 0], points[:, 1], s=18, c="#9aa4b2", linewidths=0, zorder=2)
    ax.scatter([mean[0]], [mean[1]], s=36, c="#111111", zorder=4)
    arrow(ax, (0, 0), toward_mean * 2.55, "#b42318", 2.4, 1, 5)
    arrow(ax, mean - along_wobble * 1.15, mean + along_wobble * 1.15, "#111111", 2.4, 1, 5)
    style_axes(ax)
    ax.set_xlim(-0.6, 3.4)
    ax.set_ylim(-1.6, 2.3)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title("Ordinary complex data\nred arrow misses the wobble")


def panel_spikes(ax, encoded, title):
    for deg, label in ((0, "0 ms"), (36, "20 ms"), (72, "40 ms")):
        rad = np.deg2rad(deg)
        ax.plot(
            [0, np.cos(rad)],
            [0, np.sin(rad)],
            color="#d0d0d0",
            lw=1,
            zorder=1,
        )
        ax.text(
            1.18 * np.cos(rad),
            1.18 * np.sin(rad),
            label,
            ha="center",
            va="center",
            color="#666666",
            fontsize=9,
        )
    colors = ("#1f4e79", "#d4652f", "#1f8a65")
    for event in encoded[:150]:
        for cell in range(3):
            value = event[cell]
            if value == 0:
                continue
            arrow(ax, (0, 0), (value.real, value.imag), colors[cell], 0.5, 0.16, 2, scale=8)
    pattern = component(encoded)
    scale = 0.98 / np.max(np.abs(pattern))
    for cell in range(3):
        value = pattern[cell] * scale
        arrow(ax, (0, 0), (value.real, value.imag), colors[cell], 2.6, 1, 5, scale=16)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#e4e4e4", lw=0.8, zorder=1))
    style_axes(ax)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title(title)


def main():
    rng = np.random.default_rng(0)
    shared = spike_matrix(0, 2000, rng)
    random_clock = spike_matrix(200, 2000, rng)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8))
    panel_ordinary(axes[0])
    panel_spikes(
        axes[1],
        shared,
        "Spike phases, shared start\nthe mean is already the order",
    )
    panel_spikes(
        axes[2],
        random_clock,
        "Spike phases, random start\nthere is no mean left to remove",
    )
    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=label)
        for color, label in zip(
            ("#1f4e79", "#d4652f", "#1f8a65"),
            ("cell 0", "cell 1", "cell 2"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Centering removes a baseline. It also removes a sequence stored in the arrows.",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.25)
    print(OUT)


if __name__ == "__main__":
    main()
