"""Generate objective comparison figures for alpha = pi vs alpha = 1.5*pi in exploration/figures/."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "exploration" / "results"
FIG = ROOT / "exploration" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

C_PI = "#d4732a"
C_15 = "#1b5e8a"
C_GR = "#2a8f5e"
C_RD = "#b83b3b"


def fig_theory_and_sweeps(p1):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    # Panel A: Exact analytical curves vs alpha/pi
    ax = axes[0]
    m = np.linspace(0.5, 2.0, 300)
    a = m * np.pi
    # 1) Unrelated map overlap |E[exp(j a U)]|^2 = sinc^2(a/(2pi))
    overlap = (np.sin(a / 2) / (a / 2)) ** 2
    # 2) Active-centred minor (quadrature) component variance for uniform [0,1] centered at 0:
    #    cos_var = 0.5*(1 + sin(a)/a) - (sin(a/2)/(a/2))^2;  sin_var = 0.5*(1 - sin(a)/a)
    cos_var = 0.5 * (1.0 + np.sin(a) / a) - overlap
    sin_var = 0.5 * (1.0 - np.sin(a) / a)
    quad_min = np.minimum(cos_var, sin_var)
    # 3) Jitter power retention exp(-a^2 sigma^2) for tight (sigma=0.05) and broad (sigma=0.15)
    jit_tight = np.exp(-(a * 0.05) ** 2)
    jit_broad = np.exp(-(a * 0.15) ** 2)
    # 4) Empty unwrapping gap fraction (2pi - a) / (2pi)
    gap_frac = (2 * np.pi - a) / (2 * np.pi)

    ax.plot(m, overlap, lw=2.2, color=C_RD, label="Unrelated-map overlap $\\mathrm{sinc}^2(\\alpha/2\\pi)$")
    ax.plot(m, quad_min, lw=2.4, color=C_15, label="Minor quadrature variance (active-centred)")
    ax.plot(m, jit_broad, lw=2.0, ls="--", color=C_PI, label="Broad-jitter power ($\\sigma=0.15T$)")
    ax.plot(m, gap_frac, lw=1.8, ls=":", color=C_GR, label="Empty circle gap $(2\\pi-\\alpha)/2\\pi$")
    ax.axvline(1.0, color=C_PI, ls="-.", lw=1.1, alpha=0.8)
    ax.axvline(1.5, color=C_15, ls="-.", lw=1.1, alpha=0.8)
    ax.text(1.02, 0.96, "$\\pi$", color=C_PI, fontweight="bold", va="top")
    ax.text(1.52, 0.96, "$1.5\\pi$", color=C_15, fontweight="bold", va="top")
    ax.set_xlabel("Phase arc $\\alpha / \\pi$")
    ax.set_ylabel("Dimensionless magnitude")
    ax.set_ylim(-0.02, 1.32)
    ax.set_title("A. Analytical trade-offs vs arc length $\\alpha$")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=7.6, loc="upper right")

    # Panel B: Empirical fine alpha sweep (D, K, I)
    ax = axes[1]
    sw = p1["sweeps"]["alpha_sweep"]
    xs = [r["mult"] for r in sw]
    ax.plot(xs, [r["D_same_10"]["clean_s"] * 10 for r in sw], "o-", lw=2.3, color=C_15, label="D: same cells, 10%+10%")
    ax.plot(xs, [r["K_same_5"]["clean_s"] * 10 for r in sw], "s-", lw=2.3, color=C_GR, label="K: same cells, 5%+5%")
    ax.plot(xs, [r["I_diff_3"]["clean_s"] * 10 for r in sw], "^--", lw=2.0, color=C_PI, label="I: diff cells, 3%+3%")
    ax.axvline(1.0, color=C_PI, ls="-.", lw=1.0, alpha=0.6)
    ax.axvline(1.5, color=C_15, ls="-.", lw=1.0, alpha=0.6)
    ax.set_xlabel("Phase arc $\\alpha / \\pi$")
    ax.set_ylabel("Clean fits (%)")
    ax.set_ylim(-5, 110)
    ax.set_title("B. Empirical sweet spot & $2\\pi$ wrap-around cliff")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#e0e0e0", fontsize=8.2, loc="lower left")

    # Panel C: Mechanistic trade-offs — Jitter & Segment length (where pi vs 1.5pi wins!)
    ax = axes[2]
    jt = p1["sweeps"]["jitter"]
    sg = p1["sweeps"]["segment"]
    j_vals = [r["jitter_ms"] for r in jt]
    ax.plot(j_vals, [r["pi"]["clean"] * 10 for r in jt], "o--", lw=2.2, color=C_PI, label="$\\alpha=\\pi$ (jitter sweep, Cond. A)")
    ax.plot(j_vals, [r["1.5pi"]["clean"] * 10 for r in jt], "o-", lw=2.4, color=C_15, label="$\\alpha=1.5\\pi$ (jitter sweep, Cond. A)")
    ax.set_xlabel("Within-event spike jitter SD (ms)")
    ax.set_ylabel("Clean fits (%)")
    ax.set_ylim(-5, 112)
    ax.set_title("C. High-jitter stress test (Condition A, 8–40 ms)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="#e0e0e0", fontsize=8.2, loc="lower left")

    fig.suptitle("Why arc length matters: Geometry, sweet spot, and failure modes of $\\pi$ vs $1.5\\pi$", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "arc_theory_and_sweeps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_head_to_head(p1, p2):
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.2))

    # Panel A: Planted A-F + Stress G-M (clean fits, share criterion)
    ax = axes[0, 0]
    pi_all = p1["AF_pi"] + p1["STRESS_pi"]
    a15_all = p1["AF_1.5pi"] + p1["STRESS_1.5pi"]
    labs = [r["name"][0] for r in pi_all]
    x = np.arange(len(labs)); w = 0.38
    ax.bar(x - w / 2, [r["cs"] * 10 for r in pi_all], w, color=C_PI, label=f"$\\alpha=\\pi$ (A–F: {sum(r['cs'] for r in p1['AF_pi'])}/60, G–M: {sum(r['cs'] for r in p1['STRESS_pi'])}/70)")
    ax.bar(x + w / 2, [r["cs"] * 10 for r in a15_all], w, color=C_15, label=f"$\\alpha=1.5\\pi$ (A–F: {sum(r['cs'] for r in p1['AF_1.5pi'])}/60, G–M: {sum(r['cs'] for r in p1['STRESS_1.5pi'])}/70)")
    ax.axvline(5.5, color="#999999", ls=":", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylim(0, 122); ax.set_ylabel("Clean fits (%)")
    ax.set_title("A. Planted reactivations A–F & Stress G–M (clean fits)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8.3, loc="upper left")

    # Panel B: Segment-length sweep + G1 short partial replays
    ax = axes[0, 1]
    sg = p1["sweeps"]["segment"]
    s_labs = [f"L={s['segment'][0]:.1f}–{s['segment'][1]:.2f}\n(Cond. E)" for s in sg] + ["G1 grid\nL=0.3–0.5\n(90 draws)"]
    v_pi = [s["pi"]["clean"] * 10 for s in sg] + [100 * p1["G1_pi"]["tot"] / 90]
    v_15 = [s["1.5pi"]["clean"] * 10 for s in sg] + [100 * p1["G1_1.5pi"]["tot"] / 90]
    x = np.arange(len(s_labs)); w = 0.36
    ax.bar(x - w / 2, v_pi, w, color=C_PI, label="$\\alpha = \\pi$")
    ax.bar(x + w / 2, v_15, w, color=C_15, label="$\\alpha = 1.5\\pi$")
    for i, (a, b) in enumerate(zip(v_pi, v_15)):
        ax.text(i - w / 2, a + 1.5, f"{a:.0f}%", ha="center", va="bottom", fontsize=8, color=C_PI, fontweight="bold")
        ax.text(i + w / 2, b + 1.5, f"{b:.0f}%", ha="center", va="bottom", fontsize=8, color=C_15, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(s_labs, fontsize=8.5)
    ax.set_ylim(0, 118); ax.set_ylabel("Clean fits / draws passing (%)")
    ax.set_title("B. Partial-replay segment length (wrap-around test)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8.5, loc="upper right")

    # Panel C: Waking benchmarks (planted + real)
    ax = axes[1, 0]
    wp, w15 = p2["waking_pi"], p2["waking_1.5pi"]
    w_labs = ["W1–W5\n(50)", "hc-11\nlin (10)", "hc-11\ncirc (6)", "hc-5\nalgn (12)", "hc-5\nunal (12)", "001695\nlaps (30)", "000552\nk≥2 (65)"]
    f8_pi = sum(np.isfinite(x["acc"]) for x in wp["f8"])
    f8_15 = sum(np.isfinite(x["acc"]) for x in w15["f8"])
    y_pi = [
        100 * sum(r["cs"] for r in wp["planted_laps"]) / 50,
        100 * wp["hc11_linear"]["cs"] / wp["hc11_linear"]["nf"],
        100 * wp["hc11_circular"]["cs"] / wp["hc11_circular"]["nf"],
        100 * wp["hc5_aligned"]["cs"] / wp["hc5_aligned"]["nf"],
        100 * wp["hc5_unaligned"]["cs"] / wp["hc5_unaligned"]["nf"],
        100 * wp["dd_laps"]["cs"] / wp["dd_laps"]["nf"],
        100 * f8_pi / len(wp["f8"]),
    ]
    y_15 = [
        100 * sum(r["cs"] for r in w15["planted_laps"]) / 50,
        100 * w15["hc11_linear"]["cs"] / w15["hc11_linear"]["nf"],
        100 * w15["hc11_circular"]["cs"] / w15["hc11_circular"]["nf"],
        100 * w15["hc5_aligned"]["cs"] / w15["hc5_aligned"]["nf"],
        100 * w15["hc5_unaligned"]["cs"] / w15["hc5_unaligned"]["nf"],
        100 * w15["dd_laps"]["cs"] / w15["dd_laps"]["nf"],
        100 * f8_15 / len(w15["f8"]),
    ]
    x = np.arange(len(w_labs)); w = 0.36
    ax.bar(x - w / 2, y_pi, w, color=C_PI, label="$\\alpha = \\pi$")
    ax.bar(x + w / 2, y_15, w, color=C_15, label="$\\alpha = 1.5\\pi$")
    ax.set_xticks(x); ax.set_xticklabels(w_labs, fontsize=8.3)
    ax.set_ylim(0, 120); ax.set_ylabel("Clean fits / sessions (%)")
    ax.set_title("C. Waking traversals (uncentred + circ-shift null + ICA)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=8.5, loc="upper right")

    # Panel D: Real hc-11 spike-in power (cpca_v2 AND legacy cpca)
    ax = axes[1, 1]
    fracs = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    pcts = [f * 100 for f in fracs]
    big_pi = [r for r in p2["hc11_pi"]["spikein"] if r["session"].startswith("Achilles")]
    big_15 = [r for r in p2["hc11_1.5pi"]["spikein"] if r["session"].startswith("Achilles")]
    sml_pi = [r for r in p2["hc11_pi"]["spikein"] if not r["session"].startswith("Achilles")]
    sml_15 = [r for r in p2["hc11_1.5pi"]["spikein"] if not r["session"].startswith("Achilles")]

    ax.plot(pcts, [100 * sum(r["hits"][f] for r in big_pi) / 10 for f in fracs], "o--", color=C_PI, lw=2.2, label="v2 (2 largest): $\\alpha=\\pi$")
    ax.plot(pcts, [100 * sum(r["hits"][f] for r in big_15) / 10 for f in fracs], "o-", color=C_15, lw=2.6, label="v2 (2 largest): $\\alpha=1.5\\pi$")
    ax.plot(pcts, [100 * sum(r["hits"][f] for r in sml_pi) / 30 for f in fracs], "s--", color=C_PI, lw=1.6, alpha=0.7, label="v2 (6 smaller): $\\alpha=\\pi$")
    ax.plot(pcts, [100 * sum(r["hits"][f] for r in sml_15) / 30 for f in fracs], "s-", color=C_15, lw=1.8, alpha=0.7, label="v2 (6 smaller): $\\alpha=1.5\\pi$")

    # Legacy cpca spike-in (2 largest)
    leg_f = [0.0, 0.03, 0.05, 0.10, 0.20]
    leg_pi = p1["legacy"]["spikein_legacy"]["pi"]
    leg_15 = p1["legacy"]["spikein_legacy"]["1.5pi"]
    ax.plot([f * 100 for f in leg_f], [100 * leg_pi[f] / 10 for f in leg_f], "^:", color="#888888", lw=1.5, label="Legacy cPCA (2 largest): $\\alpha=\\pi$")
    ax.plot([f * 100 for f in leg_f], [100 * leg_15[f] / 10 for f in leg_f], "v:", color="#444444", lw=1.5, label="Legacy cPCA (2 largest): $\\alpha=1.5\\pi$")

    ax.set_xlabel("Ideal replays spiked into real hc-11 PRE sleep (%)")
    ax.set_ylabel("Draws recovering map (%)")
    ax.set_ylim(-5, 108); ax.set_xlim(-0.5, 21)
    ax.set_title("D. Real hc-11 spike-in power (v2 & legacy cPCA)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="#e0e0e0", fontsize=7.6, loc="lower right")

    fig.suptitle("Matched head-to-head evaluation: $\\alpha = \\pi$ vs $\\alpha = 1.5\\pi$ (identical seeds & code)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "arc_head_to_head.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p1 = pickle.load(open(RES / "arc_eval_part1.pkl", "rb"))
    p2 = pickle.load(open(RES / "arc_eval_part2.pkl", "rb"))
    fig_theory_and_sweeps(p1)
    fig_head_to_head(p1, p2)
    print("Saved arc comparison figures to", FIG)
