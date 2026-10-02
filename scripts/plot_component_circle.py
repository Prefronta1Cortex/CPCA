"""Components and data points on the unit semicircle in the complex plane.

Left: a planted route (synthetic). Each dot is one cell's arrow in one event
(angle = spike time within the event, radius 1); the thick arrows are the
component found by the decomposition, one per cell. A cell's component arrow
points where that cell's dots cluster.

Middle and right: a real component from hc-11 (Achilles 11/01, POST sleep,
centred variant, the component whose member order best matches the place
map). Middle: the component's member arrows, coloured by the cell's
place-field centre. Right: the raw arrows of those member cells in the events
that express the component most strongly, same colours.

Writes figures/component_on_the_circle.png.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpca import encode, event_spike_times, fit, population_bursts  # noqa: E402
from cpca.evaluate import member_order_vs_place, members  # noqa: E402
from cpca.hc11 import load_session, nrem_intervals, place_centres, pyramidal_spikes, running, session_paths  # noqa: E402
from sim.routes import generate, make_routes  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white"})


def semicircle(ax, T_ms, full_circle=False):
    theta = np.linspace(0, np.pi, 200)
    if full_circle:
        t2 = np.linspace(np.pi, 2 * np.pi, 200)
        ax.plot(np.cos(t2), np.sin(t2), color="#e4e7eb", lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.plot(np.cos(theta), np.sin(theta), color="#b8bfc9", lw=1.4, zorder=1)
    ax.plot([-1, 1], [0, 0], color="#b8bfc9", lw=1.4, zorder=1)
    ax.text(1.02, -0.06, "window start\n0 ms", ha="center", va="top", fontsize=9, color="#555555")
    ax.text(-1.02, -0.06, f"window end\n{T_ms:.0f} ms", ha="center", va="top", fontsize=9, color="#555555")
    ax.text(0, 1.06, f"population peak, {T_ms / 2:.0f} ms", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.15 if full_circle else -0.4, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, z, colour, lw=2.4, alpha=1.0):
    ax.add_patch(FancyArrowPatch((0, 0), (z.real, z.imag), arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=colour, alpha=alpha, shrinkA=0, shrinkB=0, zorder=4))


def align_to_data(p, data_means, mask):
    """Rotate a component by one global phase so its arrows sit where the data's mean arrows sit.

    An eigenvector is only defined up to a common rotation; this picks the rotation that
    best matches the cells' mean arrows (weighted by the component's own weights).
    """
    phi = np.angle(np.sum(np.abs(p[mask]) * data_means[mask] * np.conj(p[mask])))
    return p * np.exp(1j * phi)


def panel_synthetic(ax):
    rng = np.random.default_rng(0)
    n_cells = 8
    routes = make_routes(rng, n_cells=n_cells, n_routes=1)
    routes[0]["pos"] = np.linspace(0, 1, n_cells)              # evenly spaced along the route
    data = generate(rng, n_cells=n_cells, n_events=400, routes=routes, replay_frac=1.0, reverse_frac=0.0,
                    participation=0.5, seq_participation=0.6, jitter_ms=4.0, span_ms=80.0)
    T = data["event_ms"]
    S = encode(data["times"], T)
    res = fit(data["times"], T, center=False, scale=False, ica=False)
    active = S != 0
    means = np.array([S[active[:, k], k].mean() if active[:, k].any() else 0 for k in range(n_cells)])
    p = align_to_data(res["patterns"][:, 0], means, np.ones(n_cells, bool))
    colours = plt.cm.viridis(np.linspace(0, 1, n_cells))
    semicircle(ax, T)
    for k in range(n_cells):
        z = S[active[:, k], k][:60]
        ax.scatter(z.real, z.imag, s=16, color=colours[k], alpha=0.4, zorder=2, linewidths=0)
    scale = 0.95 / np.abs(p).max()
    for k in range(n_cells):
        arrow(ax, p[k] * scale, colours[k])
    ax.set_title("Synthetic: one planted route (8 cells)\nsmall dots: one cell's arrow in one event; thick arrows: the component", fontsize=11)


def real_component():
    path = next(p for p in session_paths() if p.name.startswith("Achilles_1101"))
    session = load_session(path)
    run = running(session)
    intervals = nrem_intervals(session, "post")
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    times, _ = event_spike_times(by_cell, cell_ids, peaks - duration / 2, peaks + duration / 2)
    T = duration * 1000
    res = fit(times, T, rng=np.random.default_rng(0), center=True, scale=True, ica=False)
    centres = place_centres(session, cell_ids, run)
    rng = np.random.default_rng(0)
    best = None
    for j in range(res["n_kept"]):
        for sign in (1.0, -1.0):
            rho, null95, n = member_order_vs_place(res["patterns"][:, j], res["T_read_ms"], centres[sign], run["circular"], run["length"], rng, n_null=300)
            if np.isfinite(rho) and (best is None or abs(rho) - null95 > best[0]):
                best = (abs(rho) - null95, j, sign, rho, null95, n)
    _, j, sign, rho, null95, n = best
    return {"S": res["encoded"], "p": res["patterns"][:, j], "scores": res["scores"][:, j], "j": j, "sign": sign, "rho": rho,
            "null95": null95, "n": n, "centres": centres[sign], "T_med": float(np.median(T)), "length": run["length"], "name": session["name"]}


def cell_means(S, scores, members_mask, n_events):
    """Each member cell's mean arrow over the events that express the component most strongly."""
    top = np.argsort(-np.abs(scores))[:n_events]
    means = np.zeros(S.shape[1], dtype=complex)
    for k in np.flatnonzero(members_mask):
        z = S[top, k]
        z = z[z != 0]
        if z.size:
            means[k] = np.mean(z / np.abs(z))                       # mean of unit arrows: angle = typical time, length = consistency
    return means


def panel_real_component(ax, rc, means, m, norm):
    p = align_to_data(rc["p"], means, m)
    semicircle(ax, rc["T_med"], full_circle=True)
    scale = 0.95 / np.abs(p[m]).max()
    for k in np.flatnonzero(m):
        arrow(ax, p[k] * scale, plt.cm.viridis(norm(rc["centres"][k])), lw=1.8)
    ax.set_title(f"{rc['name'].replace('_', ' ')}, POST sleep, component {rc['j'] + 1}: the component's arrows\n"
                 f"{m.sum()} member cells with a field; member order vs place: Spearman {rc['rho']:+.2f} (chance {rc['null95']:.2f})", fontsize=11)


def panel_real_data(ax, rc, means, m, norm, n_events):
    semicircle(ax, rc["T_med"])
    for k in np.flatnonzero(m):
        arrow(ax, means[k], plt.cm.viridis(norm(rc["centres"][k])), lw=1.8)
    ax.set_title(f"The data behind it: each member cell's mean arrow over the {n_events} events\n"
                 "that express the component most strongly (length = how consistent its timing is)", fontsize=11)


def main(n_events=60):
    rc = real_component()
    m = members(rc["p"], 0.2) & np.isfinite(rc["centres"])
    norm = plt.Normalize(0, rc["length"])
    means = cell_means(rc["S"], rc["scores"], m, n_events)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.2))
    panel_synthetic(axes[0])
    panel_real_component(axes[1], rc, means, m, norm)
    panel_real_data(axes[2], rc, means, m, norm, n_events)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1:], fraction=0.02, pad=0.02)
    cbar.set_label(f"place-field centre, running direction {'+' if rc['sign'] > 0 else '−'} (m)")
    fig.suptitle("Arrows on the half-circle: the angle is the time within the event (0 = window start, π = window end), "
                 "the length is how much the cell takes part", y=1.0, fontsize=13)
    fig.text(0.5, -0.02, "Middle panel: a centred component's angles are relative; the whole component was rotated by one angle so that it lies on the data "
             "in the right panel. Arrows below the axis mean the component's relative times span more than the half-circle (see docs/method.md, Section 8).",
             ha="center", va="top", fontsize=10, color="#444444", wrap=True)
    out = ROOT / "figures" / "component_on_the_circle.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
