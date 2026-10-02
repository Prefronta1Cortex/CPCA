"""Split forward and reverse events, then refit each group.

The sign of Im(a1 conj(a2)) separates the two orders. Each event's score
vector is divided by its length first; that does not change the sign.
The split is meaningful when components 1 and 2 are a forward/reverse pair.
direction_split_null compares it with phase-randomised data, where no
timing remains.
"""

import numpy as np

from cpca.decompose import decompose
from cpca.stats import shuffle_phases


def direction_signs(event_scores):
    """Sign of Im(a1 conj(a2)) after dividing each event by its score length."""
    event_scores = np.asarray(event_scores)
    length = np.linalg.norm(event_scores, axis=1)
    scaled = np.zeros_like(event_scores)
    np.divide(event_scores, length[:, None], out=scaled, where=length[:, None] > 0)
    return np.sign(np.imag(scaled[:, 0] * np.conj(scaled[:, 1]))).real


def split_by_direction(encoded):
    """Refit the top component inside each direction group.

    Returns the per-event signs and a dict keyed by +1 and -1. A key is
    omitted when that group has fewer than two events.
    """
    eigenvalues, components, event_scores, lag = decompose(encoded, n_components=2)
    signs = direction_signs(event_scores)
    groups = {}
    for sign in (1.0, -1.0):
        mask = signs == sign
        if int(mask.sum()) < 2:
            continue
        group_eigenvalues, group_components, _, group_lag = decompose(
            encoded[mask], n_components=1
        )
        groups[sign] = {
            "mask": mask,
            "eigenvalues": group_eigenvalues,
            "component": group_components[:, 0],
            "lag": group_lag,
        }
    return {
        "signs": signs,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": event_scores,
        "lag": lag,
        "groups": groups,
    }


def direction_correlation(encoded):
    """Correlation of the two refit templates. NaN if either group is missing.

    A forward/reverse pair has a negative correlation. Phase-randomised data
    can also land on either side, so the value needs direction_split_null.
    """
    groups = split_by_direction(encoded)["groups"]
    if 1.0 not in groups or -1.0 not in groups:
        return np.nan
    first = np.angle(groups[1.0]["component"])
    second = np.angle(groups[-1.0]["component"])
    if np.std(first) < 1e-8 or np.std(second) < 1e-8:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def direction_split_null(encoded, rng, n_draws=20):
    """Real direction correlation and the same statistic after phase randomisation.

    Returns the observed correlation and one value per draw. The split is
    not evidence of two orders when the observed value sits inside this null.
    """
    observed = direction_correlation(encoded)
    nulls = np.empty(n_draws)
    for draw in range(n_draws):
        nulls[draw] = direction_correlation(shuffle_phases(encoded, rng))
    return observed, nulls
