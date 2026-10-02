"""Tables (results/summary.md) and figures (figures/*.png) from the cached results."""

import glob
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
FIGURES.mkdir(exist_ok=True)

ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
         "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
METHODS = [  # key, label, colour
    ("cpca_C-C0", "cPCA, one template (C − C⁰)", "#9aa4b2"),
    ("cpca_mixture", "cPCA, mixture of templates (new)", "#1f4e79"),
    ("cpca_mixture_phase", "cPCA, mixture, membership-blind (new)", "#6aa7d8"),
    ("seqnmf", "seqNMF", "#d4652f"),
    ("hmm", "Poisson HMM (label-free seriation)", "#1f8a65"),
]
REALISTIC_METHODS = [
    ("cpca_C-C0", "cPCA, one template (C − C⁰)", "#9aa4b2"),
    ("mixture_cosine", "cPCA, mixture of templates (new)", "#1f4e79"),
    ("mixture_phase", "cPCA, mixture, event arrows unit length", "#6aa7d8"),
    ("seqnmf", "seqNMF", "#d4652f"),
    ("hmm", "Poisson HMM (label-free seriation)", "#1f8a65"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def label(name):
    rat, day = name.split("_")
    return f"{rat}\n{day[:2]}/{day[2:4]}"


def star(v):
    return "*" if np.isfinite(v["best"]) and np.isfinite(v["null95"]) and v["best"] > v["null95"] else ""


def fmt(v):
    if not np.isfinite(v.get("best", np.nan)):
        return "—"
    return f"{v['best']:.2f} [{v['null95']:.2f}]{star(v)}"


def grouped_bars(ax, groups, methods, getter, ylabel, title, ylim=(0, 1.05)):
    """One group per item on x; one bar per method; black tick = matched null 95th percentile."""
    x = np.arange(len(groups))
    width = 0.8 / len(methods)
    for i, (key, name, colour) in enumerate(methods):
        vals = [getter(g, key) for g in groups]
        best = [v["best"] if v else np.nan for v in vals]
        null = [v["null95"] if v else np.nan for v in vals]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, best, width=width * 0.92, color=colour, label=name, zorder=2)
        ax.hlines(null, pos - width * 0.46, pos + width * 0.46, colors="black", linewidth=2.2, zorder=3,
                  label="chance level (95th percentile of cell-shuffle null)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
    ax.set_axisbelow(True)


def figure_synthetic(syn):
    methods = [m for m in METHODS if m[0] != "cpca_mixture_phase"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    names = [s["condition"] for s in syn]
    grouped_bars(ax, syn, methods, lambda s, k: s.get(k), "|Spearman| between recovered order and planted order",
                 "Clean synthetic events with a planted order: does each method recover it?")
    ax.set_xticklabels([n.replace(", ", ",\n") for n in names], fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthetic_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_realistic(rea):
    fig, ax = plt.subplots(figsize=(14, 6.4))
    grouped_bars(ax, rea, REALISTIC_METHODS, lambda s, k: s.get(k), "|Spearman| between recovered order and planted order",
                 "Realistic synthetic sleep: which property of real data removes the recoverable order?")
    def wrap(text, width=16):
        words, lines_, cur = text.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > width:
                lines_.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines_.append(cur)
        return "\n".join(lines_)
    ax.set_xticklabels([wrap(s["condition"]) for s in rea], fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "realistic_synthetic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_real(real):
    fig, ax = plt.subplots(figsize=(14, 6.4))
    grouped_bars(ax, real, METHODS, lambda r, k: r.get(k), "|Spearman| between recovered order and place-field order",
                 "hc-11 POST sleep: cell order recovered without labels, scored against the waking place map")
    ax.set_xticklabels([label(r["name"]) for r in real])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "real_sessions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_realistic(rea, lines):
    lines += ["", "## Realistic synthetic sleep (3000 events, 15% replay unless stated): which property removes the order?", "",
              "Each replay covers a 40% segment of the track in one of two direction-specific maps, forward or backward; 'assemblies' means which cells fire is structured (6 groups); "
              "'non-place' means half the recorded cells have no field and fire anyway.", "",
              "| condition | replay events | cPCA one template (C − C⁰) | cPCA mixture | cPCA mixture, unit event arrows | seqNMF | HMM |", "|---|---|---|---|---|---|---|"]
    for s in rea:
        lines.append(f"| {s['condition']} | {100*s['fraction_replay']:.0f}% | {fmt(s['cpca_C-C0'])} | {fmt(s['mixture_cosine'])} (K={s['mixture_cosine']['K']}) | "
                     f"{fmt(s['mixture_phase'])} (K={s['mixture_phase']['K']}) | {fmt(s['seqnmf'])} | {fmt(s['hmm'])} |")
    lines += ["", "Membership-blind mixture (event and template arrows both unit length), same conditions, K = 2 / 4: "
              "R4 0.25 [0.28] / 0.51 [0.28]*; R2 0.21 [0.27] / 0.39 [0.31]*; R0 0.22 [0.34] / 0.24 [0.34]; R5 0.39 [0.34]* / 0.34 [0.32]*."]


def figure_mixture_K(real):
    Ks = sorted({K for r in real for K in r["mixture_by_K"]})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    x = np.arange(len(real))
    width = 0.8 / len(Ks)
    palette = ["#c9d4e3", "#7f9cc0", "#3e6da3", "#1f4e79", "#102c47"]
    for i, K in enumerate(Ks):
        pos = x - 0.4 + width * (i + 0.5)
        best = [r["mixture_by_K"].get(K, {}).get("best", np.nan) for r in real]
        null = [r["mixture_by_K"].get(K, {}).get("null95", np.nan) for r in real]
        stab = [r["mixture_by_K"].get(K, {}).get("stability", np.nan) for r in real]
        axes[0].bar(pos, best, width=width * 0.92, color=palette[i % len(palette)], label=f"K = {K}", zorder=2)
        axes[0].hlines(null, pos - width * 0.46, pos + width * 0.46, colors="black", linewidth=2.2, zorder=3,
                       label="chance level (null 95th percentile)" if i == 0 else None)
        axes[1].bar(pos, stab, width=width * 0.92, color=palette[i % len(palette)], label=f"K = {K}", zorder=2)
    for ax, ylabel, title in ((axes[0], "|Spearman| vs place-field order", "Mixture cPCA: best template per number of templates K"),
                              (axes[1], "split-half template agreement", "Label-free choice of K: stability across halves of the events")):
        ax.set_xticks(x)
        ax.set_xticklabels([label(r["name"]) for r in real])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)
        ax.yaxis.grid(True, color="#e6e9ee", zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(Ks) + 1, frameon=False)
    for r, xi in zip(real, x):
        axes[1].annotate(f"K={r['mixture_K']}", (xi, 1.0), ha="center", va="bottom", fontsize=11, color="#1f4e79")
    fig.tight_layout()
    fig.savefig(FIGURES / "mixture_K_selection.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(syn, real, rea=()):
    lines = ["# Label-free comparison: results", "",
             "Score: best |Spearman| between the method's cell order and the answer key (planted order, or place-field centre per running direction), "
             "maximised over the method's templates/factors and the two directions. In brackets: 95th percentile of the same maximum after shuffling cell identities. "
             "A star marks a value above that chance level. The answer key is used for scoring only.", ""]
    lines += ["## Synthetic events with a planted order", "",
              "| condition | replay events | cPCA one template (C) | cPCA one template (C − C⁰) | cPCA mixture (K by stability) | seqNMF | HMM | count PCA/ICA |",
              "|---|---|---|---|---|---|---|---|"]
    for s in syn:
        lines.append(f"| {s['condition']} | {100*s['fraction_replay']:.0f}% | {fmt(s['cpca_C'])} | {fmt(s['cpca_C-C0'])} | {fmt(s['cpca_mixture'])} (K={s['mixture_K']}) | "
                     f"{fmt(s['seqnmf'])} | {fmt(s['hmm'])} (M={s['hmm']['n_states']}) | {s['count_ica']['n_assemblies']} assemblies, no order |")
    if rea:
        write_realistic(rea, lines)
    lines += ["", "## hc-11 POST NREM, wide events (edges at mean rate, peak above mean + 3 SD), pyramidal cells", "",
              "| session | maze | cells | place cells (+/−) | events | cPCA one template (C) | cPCA one template (C − C⁰) | cPCA mixture (K by stability) | cPCA mixture, membership-blind | seqNMF | HMM | count PCA/ICA |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    wins = {k: 0 for k, _, _ in METHODS}
    wins["cpca_C"] = 0
    for r in real:
        for k in wins:
            if k in r and star(r[k]):
                wins[k] += 1
        ica = r["count_ica"]
        mp = r.get("cpca_mixture_phase", {"best": np.nan, "null95": np.nan})
        lines.append(f"| {r['name']} | {r['maze']} | {r['n_cells']} | {r['n_place'][1.0]}/{r['n_place'][-1.0]} | {r['n_events']} | {fmt(r['cpca_C'])} | {fmt(r['cpca_C-C0'])} | "
                     f"{fmt(r['cpca_mixture'])} (K={r['mixture_K']}) | {fmt(mp)} (K={r.get('mixture_phase_K', '—')}) | {fmt(r['seqnmf'])} | {fmt(r['hmm'])} (M={r['hmm']['n_states']}) | "
                     f"{ica['n_assemblies']} assemblies; member field spread {ica.get('member_field_spread', np.nan):.2f} m vs random {ica.get('member_field_spread_null', np.nan):.2f} m |")
    lines.append(f"| **sessions above chance** | | | | | {wins['cpca_C']}/{len(real)} | {wins['cpca_C-C0']}/{len(real)} | {wins['cpca_mixture']}/{len(real)} | {wins['cpca_mixture_phase']}/{len(real)} | {wins['seqnmf']}/{len(real)} | {wins['hmm']}/{len(real)} | — |")
    lines += ["", "## Mixture cPCA per number of templates (real sessions)", "",
              "| session | " + " | ".join(f"K={K}" for K in sorted({K for r in real for K in r['mixture_by_K']})) + " | chosen K |", "|---|" + "---|" * (len({K for r in real for K in r['mixture_by_K']}) + 1)]
    for r in real:
        cells = []
        for K in sorted({K for rr in real for K in rr["mixture_by_K"]}):
            v = r["mixture_by_K"].get(K)
            cells.append(f"{fmt(v)} stab {v['stability']:.2f}" if v else "—")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + f" | {r['mixture_K']} |")
    lines += ["", "## Run time per session (seconds)", "", "| session | cPCA one template | cPCA mixture (4 values of K, with stability fits) | seqNMF (3 restarts) | HMM (2 state counts) | count PCA/ICA |", "|---|---|---|---|---|---|"]
    for r in real:
        t = r["runtimes"]
        lines.append(f"| {r['name']} | {t['cpca_single']:.2f} | {t['cpca_mixture']:.0f} | {t['seqnmf']:.0f} | {t['hmm']:.0f} | {t['count_ica']:.2f} |")
    lines += ["", "seqNMF factor lag spans (ms) and cells per factor, real sessions: " + "; ".join(f"{r['name'].split('_')[0]} {r['name'].split('_')[1][:4]}: spans {np.round(r['seqnmf']['spans_ms']).astype(int).tolist()} cells {r['seqnmf']['n_cells']}" for r in real)]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    syn = pickle.load(open(RESULTS / "synthetic.pkl", "rb")) if (RESULTS / "synthetic.pkl").exists() else []
    rea = pickle.load(open(RESULTS / "realistic_synthetic.pkl", "rb")) if (RESULTS / "realistic_synthetic.pkl").exists() else []
    real = [pickle.load(open(f, "rb")) for f in glob.glob(str(RESULTS / "real_*.pkl"))]
    real.sort(key=lambda r: ORDER.index(r["name"]))
    if syn:
        figure_synthetic(syn)
    if rea:
        figure_realistic(rea)
    if real:
        figure_real(real)
        figure_mixture_K(real)
    write_summary(syn, real, rea)


if __name__ == "__main__":
    main()
