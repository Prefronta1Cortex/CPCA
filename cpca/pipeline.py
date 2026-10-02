"""The method, end to end.

    spikes in events  ->  arrows (half-circle per event)  ->  centre and scale
    ->  complex PCA  ->  keep components above the noise bound  ->  rotate to
    independent components  ->  read each as cells with weights and relative times

Parameters that matter: how events are cut (upstream), and whether the
half-circle is each event's own duration (default) or a fixed T. Everything
else is a convention (mean arrow, centre, scale, Marchenko-Pastur bound, ICA).
"""

import numpy as np

from cpca.decode import readout
from cpca.decompose import decompose, n_significant, prepare
from cpca.encode import encode
from cpca.evaluate import members
from cpca.ica import unmix


def fit(times_ms, T, rng=None, how="mean", center=True, scale=True, n_components="mp", ica=True,
        member_frac=0.2, max_components=20):
    """Fit complex PCA (+ ICA) to spike times.

    times_ms: (events, cells, spikes) in ms from each event window's start; NaN = no spike.
    T: half-circle scale in ms, one number or one per event (use the event's duration
       so that each event's extent maps to [0, pi]).
    n_components: "mp" (Marchenko-Pastur bound) or an integer.
    Returns a dict with the encoded arrows, the prepared matrix, eigenvalues, the noise
    bound, the kept PCA components, the final patterns (ICA-rotated when ica=True and
    more than one component is kept), their event scores, member masks, and read-outs
    (weights, relative times in ms, delays) per pattern.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    encoded = encode(times_ms, T, how=how, t0=0.0)
    prepared = prepare(encoded, center=center, scale=scale)
    M, N = prepared.shape
    eigenvalues, components, pc_scores, cov = decompose(encoded, n_components=None, center=center, scale=scale)
    if n_components == "mp":
        k, bound = n_significant(eigenvalues, M, N)
    else:
        k, bound = int(n_components), np.nan
    k = int(min(max(k, 0), max_components))
    kept_values, kept = eigenvalues[:k], components[:, :k]
    if ica and k > 1:
        patterns, scores, rotation = unmix(prepared, kept_values, kept, rng)
    else:
        patterns, scores, rotation = kept, pc_scores[:, :k], np.eye(k)
    T_read = float(np.median(T)) if np.ndim(T) else float(T)
    reads = []
    for j in range(k):
        weights, rel_times, delays, included, span = readout(patterns[:, j], T_read, min_weight=member_frac)
        reads.append({"weights": weights, "times_ms": rel_times, "delays_ms": delays, "included": included, "span_rad": span})
    return {
        "encoded": encoded, "prepared": prepared, "eigenvalues": eigenvalues, "noise_bound": bound,
        "n_kept": k, "pca_components": kept, "patterns": patterns, "scores": scores, "rotation": rotation,
        "members": [members(patterns[:, j], member_frac) for j in range(k)], "readouts": reads,
        "T_read_ms": T_read, "settings": {"how": how, "center": center, "scale": scale, "ica": ica, "member_frac": member_frac},
    }
