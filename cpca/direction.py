"""Forward and backward plays of one component.

In the half-circle encoding a spike at time t is the arrow exp(j pi t / T). Played backwards it
falls at T - t, whose arrow is -conj(exp(j pi t / T)): **time reversal is complex conjugation**,
up to a sign that is a common phase of the event and cancels in every lag. So an event s
matches a template q forward with strength F = |<s, q>| and backward with B = |<s, conj(q)>|,
both unchanged by the event's clock.

Why this is needed: after centring, forward and backward plays of one route are opposite
deviations from the same mean (with a 60/40 mix, 0.4 (s_f - s_b) and -0.6 (s_f - s_b)) and land
on one component, which for a balanced route is nearly real: its phases, the order, cancel.
Uncentred, they stay apart only when the route fills the half-circle.

The split, per kept component:
1. members: cells with weight at least 20% of the largest; events: those in which at least
   `min_active` members fire. The raw (uncentred) arrows of the members are used.
2. alternate until the labels stop changing: label each event forward if F >= B; conjugate the
   backward events; the template is the leading eigenvector of their lag matrix. Start from the
   leading eigenvector without flips.
3. an event expresses the template if max(F, B) exceeds the 95th percentile of the same quantity
   after shuffling which member got which arrow in that event (50 shuffles). The shuffle keeps who
   fired and the set of spike times, so a synchronous volley does not pass; only an event whose
   members fire in the template's order (either way round) does. The template is refitted on the
   expressing events, and directions are reported for those events only.
4. held-out test: a template fitted (steps 2-3) on one random half of the events must find
   events of the other half that express it, each judged by its own permutation p-value, more
   often than the 5% expected without order (binomial test; both halves in turn, Bonferroni).
   Everything that touches the held-out half is fixed beforehand, so the test stays valid
   although the template is chosen to fit. (A split-half agreement of direction labels on the
   expressing events was tried first and is not a test: the events are chosen by a template
   fitted to them, and orderless data scored as high as replays.)
5. guard: the template replaces the component's order only if the two orders agree (|Spearman|
   over shared members at least AGREEMENT, 0.6). The split may sharpen an order (a balanced
   centred component reads 0.8 of the route, its template 1.0) but may not overturn it: with
   partial replays a single per-cell template is the wrong model (a 40% segment stretched over the
   whole event puts its cells 2.5 times further apart in angle than the template does) and the fit
   converges to something else. The threshold was set on synthetic development draws (results on
   fresh draws in docs/direction.md).

The split separates one order played both ways. It does not separate two different orders on the
same cells (their template is a blend of the two); that is cpca/mixture.py.

The template's order is read out as for any component; forward and backward events play it in
opposite directions. Which direction is called "forward" is arbitrary (q and conj(q) describe
the same pair of orders).
"""

import numpy as np
from scipy.stats import binomtest, spearmanr

from cpca.decode import readout
from cpca.theory import eigh_desc, lag_matrix

AGREEMENT = 0.6
MIN_EXPRESSING = 40


def _leading(X):
    return eigh_desc(lag_matrix(X))[1][:, 0]


def order_of(component, member_frac=0.2):
    """Read-out order of a component: relative time of each member, NaN elsewhere (time scale irrelevant)."""
    _, times, _, included, _ = readout(component, 1.0, min_weight=member_frac)
    return np.where(included, times, np.nan)


def fit_template(S, n_iter=50):
    """Template q (members) and forward labels for the events in S (events x members, raw arrows)."""
    q = _leading(S)
    labels = None
    for it in range(n_iter):
        new = np.abs(S @ np.conj(q)) >= np.abs(S @ q)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        q = _leading(np.where(labels[:, None], S, np.conj(S)))
    F, B = np.abs(S @ np.conj(q)), np.abs(S @ q)
    return q, F >= B, F, B, it + 1


def permute_within_events(S, rng):
    """Shuffle which active member got which arrow, separately in every event.

    Keeps who fired and the set of arrows (so a synchronous volley is unchanged) and destroys
    only the order of the members relative to each other.
    """
    S = np.asarray(S)
    active = S != 0
    keys = np.where(active, rng.random(S.shape), np.inf)
    shuffled = np.argsort(keys, axis=1)                                  # active positions first, in random order
    in_place = np.argsort(np.where(active, np.arange(S.shape[1])[None, :], np.inf), axis=1)   # active positions, in order
    rank = np.arange(S.shape[1])[None, :] < active.sum(axis=1, keepdims=True)
    out = np.zeros_like(S)
    rows = np.broadcast_to(np.arange(S.shape[0])[:, None], S.shape)
    out[rows[rank], in_place[rank]] = S[rows[rank], shuffled[rank]]
    return out


def expression_null(S, q, rng, n_shuffles=50, quantile=95.0):
    """Per event, the given percentile of max(F, B) after shuffling which member got which arrow (permute_within_events)."""
    null = np.empty((n_shuffles, S.shape[0]))
    for s in range(n_shuffles):
        R = permute_within_events(S, rng)
        null[s] = np.maximum(np.abs(R @ np.conj(q)), np.abs(R @ q))
    return np.percentile(null, quantile, axis=0)


def expression_p(S, q, rng, n_shuffles=50):
    """Per event, the permutation p-value of max(F, B) against the within-event shuffle: (1 + #null >= observed) / (n + 1)."""
    observed = np.maximum(np.abs(S @ np.conj(q)), np.abs(S @ q))
    count = np.zeros(S.shape[0])
    for _ in range(n_shuffles):
        R = permute_within_events(S, rng)
        count += np.maximum(np.abs(R @ np.conj(q)), np.abs(R @ q)) >= observed - 1e-12
    return (1 + count) / (n_shuffles + 1)


def fit_and_gate(S, rng):
    """Steps 2-3: template, labels, F, B, iterations and the expressing events of S (events x members)."""
    q, forward, F, B, n_iter = fit_template(S)
    express = np.maximum(F, B) > expression_null(S, q, rng)
    if express.sum() >= MIN_EXPRESSING:
        # refit on the events that carry the order, then label and gate every active event with that template
        q = fit_template(S[express])[0]
        F, B = np.abs(S @ np.conj(q)), np.abs(S @ q)
        forward = F >= B
        express = np.maximum(F, B) > expression_null(S, q, rng)
    return q, forward, F, B, n_iter, express


def heldout_test(S, rng, alpha=0.05):
    """Step 4: (share of held-out events expressing at p <= alpha, p-value).

    The template (steps 2-3) is fitted on one random half of the events and every event of the
    other half gets its own permutation p-value against it (expression_p). Without order among the
    members, a held-out event reaches p <= alpha with probability at most alpha, independently
    given the template, so the count is tested against Binomial(n, alpha); both halves take a
    turn and the smaller p is doubled (Bonferroni).
    """
    perm = rng.permutation(S.shape[0])
    halves = perm[: S.shape[0] // 2], perm[S.shape[0] // 2:]
    k = n = 0
    ps = []
    for train, held in (halves, halves[::-1]):
        q = fit_and_gate(S[train], rng)[0]
        hits = int((expression_p(S[held], q, rng) <= alpha).sum())
        ps.append(binomtest(hits, held.size, alpha, alternative="greater").pvalue)
        k, n = k + hits, n + held.size
    return k / n, float(min(1.0, 2 * min(ps)))


def split_members(encoded, members, rng, min_active=3, test=True):
    """Steps 1-4 of the split for a given set of member cells (boolean, cells); None if too few events.

    The template is returned in cell space (zeros off the members) so it can be read out like a component.
    """
    S_all = encoded[:, members]
    active = (np.abs(S_all) > 0).sum(axis=1) >= min_active
    if active.sum() < 20 or members.sum() < 3:
        return None
    S = S_all[active]
    q, forward, F, B, n_iter, express = fit_and_gate(S, rng)
    template = np.zeros(encoded.shape[1], complex)
    template[members] = q
    backward = float(1 - forward[express].mean()) if express.any() else np.nan
    out = {"members": members, "template": template, "events": np.flatnonzero(active), "forward": forward,
           "index": (F - B) / np.maximum(F + B, 1e-12), "expresses": express, "n_iter": n_iter,
           "n_expressing": int(express.sum()), "backward_fraction": backward,
           "minority_fraction": float(min(backward, 1 - backward)) if np.isfinite(backward) else np.nan}
    if test:
        share, p = heldout_test(S, rng)
        out.update(heldout_share=share, heldout_p=p)
    return out


def split_component(encoded, component, rng, member_frac=0.2, min_active=3, test=True):
    """Direction split of one component (steps 1-5). Returns a dict (see module docstring), or None if too few events.

    encoded: raw arrows (events x cells); component: (cells,) complex.
    """
    weight = np.abs(component)
    out = split_members(encoded, weight >= member_frac * weight.max(), rng, min_active, test)
    if out is None:
        return None
    o_comp, o_temp = order_of(component, member_frac), order_of(out["template"], member_frac)
    both = np.isfinite(o_comp) & np.isfinite(o_temp)
    agreement = abs(float(spearmanr(o_comp[both], o_temp[both]).statistic)) if both.sum() >= 6 else np.nan
    accepted = bool(np.isfinite(agreement) and agreement >= AGREEMENT)
    out.update(agreement=agreement, accepted=accepted, order=o_temp if accepted else o_comp)
    return out


def group_templates(encoded, split, min_events=20):
    """Templates fitted on the forward and on the backward expressing events separately (cell space), or None."""
    S = encoded[split["events"]][:, split["members"]]
    out = {}
    for name, mask in (("forward", split["forward"] & split["expresses"]), ("backward", ~split["forward"] & split["expresses"])):
        if mask.sum() < min_events:
            out[name] = None
            continue
        t = np.zeros(encoded.shape[1], complex)
        t[split["members"]] = _leading(S[mask])
        out[name] = t
    return out
