"""hc-11: waking place fields sorted by each method's sleep order (what each method recovers from sleep).

Each method's outputs are those of the matched rerun (scripts/run_rerun.py): the same 1500-event
subset of a session's sleep bursts for every method; complex PCA centred (recomputed exactly as
the rerun did: same subset, same seed, same read-out), seqNMF at K = 15, L = 120 ms, PP-Seq's
sequence types. Per method, its best output against the waking map is shown: the output, running
direction and sense (forward or reversed: replays run both ways) that best match the place-field
centres. The p-value is against a null that permutes the field centres among the cells that have
one (separately per direction) and takes the same maximum over outputs, directions, senses (and
rotations on circular tracks), so a method gains nothing from returning more outputs.

The figure shows one session fixed in advance (Achilles 10/25, the one with the most place cells),
after and before the maze; the caption and results/hc11_sequences/summary.md give all sessions.

Usage: python scripts/plot_hc11_sequences.py [--workers=10]
Cache: results/hc11_sequences/<session>_<epoch>.pkl (existing files are reused).
"""

import os
import pickle
import sys
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "hc11_sequences"
FIG = ROOT / "figures"
RERUN = ROOT / "results" / "rerun"
EXAMPLE = "Achilles_10252013"
HC11_ORDER = ["Achilles_10252013", "Achilles_11012013", "Buddy_06272013", "Cicero_09012014",
              "Cicero_09102014", "Cicero_09172014", "Gatsby_08022013", "Gatsby_08282013"]
METHODS = [("cpca", "Complex PCA", "#1f4e79"), ("seqnmf", "seqNMF", "#d4652f"), ("ppseq", "PP-Seq", "#b8901a")]
N_NULL = 1000
MIN_MEMBERS = 6
N_ROTATIONS = 12


def seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ----------------------------------------------------------------------------- the test
def best_map_match(orders, centres, circular, length, rng, n_null=N_NULL, min_members=MIN_MEMBERS, n_rotations=N_ROTATIONS):
    """Best output against the map, over outputs, running directions, senses and (circular tracks) rotations.

    orders: list of per-cell times (NaN = not a member); centres: {+1.0, -1.0} -> per-cell field
    centre (NaN = no field). The statistic is |Spearman| between an output's members' times and
    their field centres in one direction (at least min_members). The null permutes each
    direction's centres among the cells with a field in that direction and keeps, per
    permutation, the best |Spearman| over everything the statistic maximises over.
    """
    from scipy.stats import rankdata

    def corr(a, b):
        a = a - a.mean(-1, keepdims=True)
        b = b - b.mean(-1, keepdims=True)
        d = np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))
        return (a * b).sum(-1) / np.where(d > 0, d, np.nan)

    shifts = [r * length / n_rotations for r in range(n_rotations)] if circular else [0.0]
    has = {s: np.flatnonzero(np.isfinite(c)) for s, c in centres.items()}
    perms = {s: np.array([rng.permutation(h) for _ in range(n_null)]) for s, h in has.items()}
    best = {"rho": np.nan, "abs": -np.inf, "index": -1, "direction": np.nan, "shift": 0.0, "n": 0, "n_key": 0}
    null_max = np.full(n_null, -np.inf)
    for j, o in enumerate(orders):
        o = np.asarray(o, float)
        for s, cen in centres.items():
            use = np.isfinite(o) & np.isfinite(cen)
            n = int(use.sum())
            if n < min_members:
                continue
            idx = np.flatnonzero(use)
            rank_t = rankdata(o[idx])
            P = cen[perms[s][:, np.searchsorted(has[s], idx)]]                     # (n_null, n)
            for sh in shifts:
                rho = float(corr(rank_t, rankdata((cen[idx] + sh) % length if circular else cen[idx])))
                if abs(rho) > best["abs"]:
                    best.update(rho=rho, abs=abs(rho), index=j, direction=s, shift=sh, n=n, n_key=int(has[s].size))
                C = (P + sh) % length if circular else P
                null_max = np.maximum(null_max, np.nan_to_num(np.abs(corr(rank_t[None, :], rankdata(C, axis=1))), nan=0.0))
    if best["index"] < 0:
        return {**best, "p": np.nan, "null95": np.nan}
    return {**best, "p": float((1 + np.sum(null_max >= best["abs"])) / (n_null + 1)), "null95": float(np.percentile(null_max, 95))}


# ----------------------------------------------------------------------------- one session and epoch
def job(name, epoch):
    from cpca import event_spike_times, population_bursts
    from cpca.hc11 import load_session, nrem_intervals, place_centres, place_maps, pyramidal_spikes, running, session_paths
    from scripts.run_rerun import ORDER_INDEX, fixed_subset, run_cpca
    session = load_session(next(p for p in session_paths() if p.name.startswith(name)))
    run = running(session)
    ei = ("post", "pre").index(epoch)
    intervals = nrem_intervals(session, epoch)
    by_cell, cell_ids, pooled = pyramidal_spikes(session, intervals)
    bursts, peaks = population_bursts(pooled, intervals)
    duration = np.diff(bursts, axis=1).ravel()
    centres = place_centres(session, cell_ids, run)
    maps, _, mid = place_maps(session, cell_ids, run)
    # the rerun's matched subset and complex PCA fit, reproduced call for call
    sub = fixed_subset(len(peaks), 555 + 2 * ORDER_INDEX[name] + ei)
    times, _ = event_spike_times(by_cell, cell_ids, (peaks - duration / 2)[sub], (peaks + duration / 2)[sub], max_spikes=60)
    cp = run_cpca(times, duration[sub] * 1000, np.random.default_rng(31 + ei))["centred"]
    stored = pickle.load(open(RERUN / f"{name}.cpca.pkl", "rb"))["epochs"][epoch]["cpca"]["centred"]
    assert cp["n_kept"] == stored["n_kept"], f"{name} {epoch}: recomputed {cp['n_kept']} components, rerun stored {stored['n_kept']}"
    seq = pickle.load(open(RERUN / f"{name}.seqnmf.pkl", "rb"))["epochs"][epoch]["seqnmf"]
    grid = next(g for g in seq["grid"] if g["K"] == 15 and abs(g["L_ms"] - 120) < 1e-6)
    pp = pickle.load(open(RERUN / f"{name}.ppseq.pkl", "rb"))["epochs"][epoch]["ppseq"]
    outputs = {"cpca": [np.asarray(o, float) for o in cp["orders"]],
               "seqnmf": [np.asarray(f["lag_ms"], float) for f in grid["factors"]],
               "ppseq": [np.asarray(t["lag_ms"], float) for t in pp["types"]]}
    place = np.isfinite(centres[1.0]) | np.isfinite(centres[-1.0])
    chance_rng = np.random.default_rng(seed("hc11-sequences-chance", name))
    chance = np.full(len(cell_ids), np.nan)
    chance[place] = chance_rng.permutation(int(place.sum())).astype(float)
    outputs["chance"] = [chance]
    rng = np.random.default_rng(seed("hc11-sequences", name, epoch))
    scores = {m: best_map_match(o, centres, run["circular"], run["length"], rng) for m, o in outputs.items()}
    result = {"name": name, "epoch": epoch, "circular": run["circular"], "length": run["length"], "n_events": int(len(sub)),
              "maps": maps.astype(np.float32), "mid": mid, "centres": centres, "place": place, "outputs": outputs, "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}_{epoch}.tmp"
    pickle.dump(result, open(tmp, "wb"))
    tmp.replace(OUT / f"{name}_{epoch}.pkl")
    return " | ".join(f"{m} {s['rho']:+.2f} p={s['p']:.3f}" for m, s in scores.items())


def compute(workers):
    todo = [(n, e) for n in HC11_ORDER for e in ("post", "pre") if not (OUT / f"{n}_{e}.pkl").exists()]
    if not todo:
        return
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    print(f"{len(todo)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, *a): a for a in todo}
        for f in as_completed(futures):
            try:
                print(f"{futures[f]}: {f.result()}", flush=True)
            except Exception:
                print(f"FAILED {futures[f]}:\n{traceback.format_exc()}", flush=True)


# ----------------------------------------------------------------------------- figure
def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    INK, SOFT = "#1d2430", "#6b7380"
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10.5, "axes.titlesize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa1ab", "axes.linewidth": 0.8,
        "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": INK, "text.color": INK, "axes.titlecolor": INK,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    cmap = LinearSegmentedColormap.from_list("fields", ["#ffffff", "#c9d1dc", "#5d6b80", "#1d2430"])
    post, pre = R[(EXAMPLE, "post")], R[(EXAMPLE, "pre")]
    maps, length = post["maps"], post["length"]
    n_bins = maps.shape[2]
    gap = 5                                                                         # empty bins between the two directions

    def strip(cells):
        m = maps[cells]
        m = m / np.maximum(m.max(axis=(1, 2), keepdims=True), 1e-9)
        out = np.full((len(cells), 2 * n_bins + gap), np.nan)
        out[:, :n_bins], out[:, n_bins + gap:] = m[:, 0], m[:, 1]
        return out

    def x_of(position, direction):
        return position / length * n_bins + (0 if direction > 0 else n_bins + gap)

    def heat(ax, cells, dots, title, colour, subtitle):
        ax.imshow(strip(cells), aspect="auto", cmap=cmap, vmin=0, vmax=1.15, interpolation="nearest",
                  extent=[0, 2 * n_bins + gap, len(cells), 0])
        ax.scatter([x_of(p, d) for p, d in dots], np.arange(len(cells)) + 0.5, s=14, color="white", edgecolor=colour,
                   linewidth=1.1, zorder=3)
        ax.set_xlim(0, 2 * n_bins + gap)
        ax.set_ylim(len(cells), 0)
        ax.set_yticks([])
        ax.set_xticks([0, n_bins, n_bins + gap, 2 * n_bins + gap])
        labels = ax.set_xticklabels(["0", f"{length:.1f} m", "0", f"{length:.1f} m"], fontsize=8.5)
        for lab, ha in zip(labels, ("left", "right", "left", "right")):
            lab.set_horizontalalignment(ha)
        ax.tick_params(axis="x", length=0, pad=3)
        ax.spines[["left", "bottom"]].set_visible(False)
        for x, lab in ((n_bins / 2, "running direction +"), (n_bins + gap + n_bins / 2, "running direction −")):
            ax.text(x, -0.11, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=SOFT)
        ax.set_title(title, color=colour, pad=18)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.8, color=SOFT, va="bottom")

    def sorted_cells(r, m):
        s = r["scores"][m]
        if s["index"] < 0:
            return None, s
        o = r["outputs"][m][s["index"]]
        cen = r["centres"][s["direction"]]
        use = np.flatnonzero(np.isfinite(o) & np.isfinite(cen))
        use = use[np.argsort(o[use] * np.sign(s["rho"]), kind="stable")]           # reversed outputs shown in the map's sense
        return use, s

    def subtitle(s, m):
        if s["index"] < 0:
            unit = {"cpca": "component", "seqnmf": "factor", "ppseq": "sequence type"}.get(m, "output")
            return f"no {unit} contains {MIN_MEMBERS} or more\nplace cells with a field"
        sense = ", reversed" if s["rho"] < 0 else ""
        p = f"p = {s['p']:.3f}" if s["p"] >= 0.001 else "p < 0.001"
        return f"|ρ| = {s['abs']:.2f} · {p} · {s['n']} of {s['n_key']} cells (direction {'+' if s['direction'] > 0 else '−'}{sense})"

    fig = plt.figure(figsize=(15.8, 10.2))
    gs = fig.add_gridspec(2, 4, left=0.035, right=0.985, top=0.765, bottom=0.16, wspace=0.12, hspace=0.62)
    fig.text(0.035, 0.962, "What each method recovers from sleep on hc-11", fontsize=16, fontweight="bold")
    fig.text(0.035, 0.928, f"Session {EXAMPLE.split('_')[0]} {EXAMPLE.split('_')[1][:2]}/{EXAMPLE.split('_')[1][2:4]} (fixed in advance: the most place "
             f"cells). Every method gets the same {post['n_events']} sleep bursts; no method sees the waking session.", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.905, "Each row is one place cell's waking firing rate along the track in both running directions (scaled to its peak; same colour scale "
             "in every panel), rows in the order of the method's best output.", fontsize=10.5, color=SOFT)
    fig.text(0.035, 0.882, "The dot is the cell's place-field centre in the direction scored: dots falling from top left to bottom right mean the "
             "method's sleep order is the waking map.", fontsize=10.5, color=SOFT)

    # answer key: cells sorted by their strongest direction, then field centre
    place = np.flatnonzero(post["place"])
    strongest = np.where(np.nan_to_num(maps[place, 0].max(1)) >= np.nan_to_num(maps[place, 1].max(1)), 1.0, -1.0)
    strongest = np.where(np.isfinite(post["centres"][1.0][place]) & ~np.isfinite(post["centres"][-1.0][place]), 1.0,
                         np.where(np.isfinite(post["centres"][-1.0][place]) & ~np.isfinite(post["centres"][1.0][place]), -1.0, strongest))
    centre = np.array([post["centres"][d][c] for d, c in zip(strongest, place)])
    key_order = np.lexsort((centre, -strongest))
    cells = place[key_order]
    ax = fig.add_subplot(gs[0, 0])
    heat(ax, cells, [(post["centres"][d][c], d) for d, c in zip(strongest[key_order], cells)], "Answer key", INK,
         f"cells sorted by their own field · {len(cells)} place cells")
    ax.set_ylabel("place cells", color=SOFT)

    # chance: a random order of all place cells, scored like any method
    use, s = sorted_cells(post, "chance")
    ax = fig.add_subplot(gs[1, 0])
    heat(ax, use, [(post["centres"][s["direction"]][c], s["direction"]) for c in use], "Chance: a random order", "#8a919c",
         subtitle(s, "chance"))
    ax.set_ylabel("place cells", color=SOFT)
    ax.text(0, -0.2, "what no recovery looks like: all place cells in a\nrandom order, scored the same way", transform=ax.transAxes,
            fontsize=8.6, color=SOFT, va="top", linespacing=1.3)

    for row, (r, label, note) in enumerate(((post, "Sleep after the maze (POST)", "the replays of the maze, if there are any"),
                                            (pre, "Sleep before the maze (PRE)", "the control: the maze has not been seen yet"))):
        for c, (m, mlabel, colour) in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, c + 1])
            use, s = sorted_cells(r, m)
            if use is None:
                ax.axis("off")
                ax.text(0.5, 0.5, subtitle(s, m), ha="center", va="center", transform=ax.transAxes, color=SOFT, fontsize=10, linespacing=1.4)
                ax.set_title(mlabel, color=colour, pad=18)
                continue
            heat(ax, use, [(r["centres"][s["direction"]][cc], s["direction"]) for cc in use], mlabel, colour, subtitle(s, m))
            if c == 0:
                pos = ax.get_position()
                fig.text(pos.x0, pos.y1 + 0.068, label, fontsize=12.5, fontweight="bold")
                fig.text(pos.x0 + 0.19, pos.y1 + 0.068, note, fontsize=10.5, color=SOFT)
                fig.add_artist(Line2D([pos.x0, 0.985], [pos.y1 + 0.062] * 2, color="#d5d9e0", lw=0.8))

    counts = {}
    for e in ("post", "pre"):
        counts[e] = {m: sum(np.isfinite(R[(n, e)]["scores"][m]["p"]) and R[(n, e)]["scores"][m]["p"] < 0.05 for n in HC11_ORDER)
                     for m, *_ in METHODS + [("chance",)]}
    fig.text(0.035, 0.045, "Best output per method: the output, running direction and sense (replays run both ways) that best match the field centres; "
             "p is against a null that permutes the centres and takes the same best over all of that method's outputs.", fontsize=9, color=SOFT)
    fig.text(0.035, 0.024, "All 8 sessions, best output with p < 0.05 — POST: " +
             ", ".join(f"{lab} {counts['post'][m]}/8" for m, lab, _ in METHODS) + f", random order {counts['post']['chance']}/8;  PRE: " +
             ", ".join(f"{lab} {counts['pre'][m]}/8" for m, lab, _ in METHODS) + f", random order {counts['pre']['chance']}/8 "
             "(per session: results/hc11_sequences/summary.md).", fontsize=9, color=SOFT)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"hc11_sequences.{ext}", dpi=200)
    plt.close(fig)


def summary(R):
    lines = ["# hc-11: each method's best output against the waking map", "",
             "Produced by `scripts/plot_hc11_sequences.py` (figure: `figures/hc11_sequences.png`). Matched 1500-event subsets of the rerun; "
             "|ρ| of the best output (over outputs, running directions, senses and, on circular tracks, rotations), p against the same maximum "
             "under permutations of the field centres, and the cells it covers.", "",
             "| session | epoch | " + " | ".join(lab for _, lab, _ in METHODS) + " | random order |", "|---" * (len(METHODS) + 3) + "|"]
    for n in HC11_ORDER:
        for e in ("post", "pre"):
            cells = []
            for m in [m for m, *_ in METHODS] + ["chance"]:
                s = R[(n, e)]["scores"][m]
                cells.append(f"{s['abs']:.2f} (p {s['p']:.3f}, {s['n']}/{s['n_key']} cells)" if s["index"] >= 0 else "–")
            lines.append(f"| {n} | {e.upper()} | " + " | ".join(cells) + " |")
    for e in ("post", "pre"):
        lines.append(f"| **sessions with p < 0.05** | {e.upper()} | " + " | ".join(
            f"{sum(np.isfinite(R[(n, e)]['scores'][m]['p']) and R[(n, e)]['scores'][m]['p'] < 0.05 for n in HC11_ORDER)}/8"
            for m in [m for m, *_ in METHODS] + ["chance"]) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv):
    workers = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--workers=")), os.cpu_count() or 2)
    compute(workers)
    R = {(n, e): pickle.load(open(OUT / f"{n}_{e}.pkl", "rb")) for n in HC11_ORDER for e in ("post", "pre")}
    FIG.mkdir(exist_ok=True)
    figure(R)
    summary(R)
    print("figures/hc11_sequences.png; results/hc11_sequences/summary.md")


if __name__ == "__main__":
    main(sys.argv[1:])
