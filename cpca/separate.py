"""Separation step of the cPCA pipeline: one ordered output per sequence, chosen without the map (docs/unmixing.md).

The kept components contain the sequences but mix them. This step turns them into one template per sequence:
1. rotate the kept components within their span so that each is concentrated on few cells (quartimax);
2. each rotated component's members (two-group split of its weights) seed a template, fitted on those cells' raw
   arrows; with `reversal=True` backward plays are matched as time-reversed (conjugated) versions of the same order;
3. refine: refit the template on all cells using only the events that express it, re-read its members, repeat;
4. merge templates that are alike, and let a smaller template give up cells claimed by a larger one (explaining away);
5. split a template into 2-4 orders on its cells (the template step of cpca.mixture) when every order is reproduced
   on the held-out events (the rules differ between reactivations and waking data; see `split`);
6. keep the templates whose expression on held-out events beats a timing null (Bonferroni over the templates):
   with `reversal=True` (reactivations) each cell's spike times are redrawn from its own timing; with
   `reversal=False` (waking traversals, where a sequence is never played backward) every cell's spikes are
   circularly shifted in every event.

Settings: `reversal` (reactivations or waking data) and `min_events` (events a template or a split needs).
Nothing here uses a map.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from cpca.direction import _leading, fit_and_gate, permute_within_events
from cpca.rotation import quartimax
from cpca.theory import eigh_desc, lag_matrix

REPRODUCE = 0.97
MERGE = 0.5
N_SURROGATES = 200
ALPHA = 0.05


def otsu_mask(a):
    """Members: the upper group of the weights, cut where the within-group variance is smallest."""
    s = np.sort(np.asarray(a, float))
    n = s.size
    c1, c2 = np.cumsum(s), np.cumsum(s ** 2)
    best, cut = np.inf, s[-1]
    for i in range(1, n):
        lo = c2[i - 1] - c1[i - 1] ** 2 / i
        hi = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (n - i)
        if lo + hi < best:
            best, cut = lo + hi, s[i]
    return np.asarray(a) >= cut


def _forward_gate(S, rng, min_refit=5):
    """Forward-only template and its expressing events (no event conjugated), gated as in cpca.direction.fit_and_gate."""
    q = _leading(S)
    for _ in range(3):
        F = np.abs(S @ np.conj(q))
        null = np.array([np.abs(permute_within_events(S, rng) @ np.conj(q)) for _ in range(50)])
        express = F > np.percentile(null, 95, axis=0)
        if _ == 2 or express.sum() < min_refit:
            break
        q = _leading(S[express])
    return q, np.ones(S.shape[0], bool), express


def _template_on(encoded, members, rng, reversal, min_events, min_active=3):
    S_all = encoded[:, members]
    active = np.flatnonzero((np.abs(S_all) > 0).sum(axis=1) >= min_active)
    if active.size < min_events or members.sum() < 3:
        return None
    if reversal:
        q, forward, _, _, _, express = fit_and_gate(S_all[active], rng)
    else:
        q, forward, express = _forward_gate(S_all[active], rng)
    t = np.zeros(encoded.shape[1], complex)
    t[members] = q
    return t, active[express], forward[express]


def refine(encoded, members, rng, reversal, min_events, allowed=None, max_iter=10):
    allowed = np.ones(encoded.shape[1], bool) if allowed is None else allowed
    members = members & allowed
    for _ in range(max_iter):
        fitted = _template_on(encoded, members, rng, reversal, min_events)
        if fitted is None or fitted[1].size < min_events:
            return None
        _, events, forward = fitted
        X = encoded[events].copy()
        X[~forward] = np.conj(X[~forward])
        new = otsu_mask(np.where(allowed, np.abs(eigh_desc(lag_matrix(X))[1][:, 0]), 0.0)) & allowed
        if np.array_equal(new, members):
            break
        members = new
    fitted = _template_on(encoded, members, rng, reversal, min_events)
    if fitted is None:
        return None
    return {"template": fitted[0], "members": members, "events": fitted[1]}


def similarity(a, b, reversal=True):
    a = a / max(np.linalg.norm(a), 1e-300)
    b = b / max(np.linalg.norm(b), 1e-300)
    return max(abs(np.vdot(a, b)), abs(np.vdot(a, np.conj(b)))) if reversal else abs(np.vdot(a, b))


def merge(templates, reversal):
    order = sorted(range(len(templates)), key=lambda i: -templates[i]["events"].size)
    kept = []
    for i in order:
        if all(similarity(templates[i]["template"], templates[j]["template"], reversal) < MERGE for j in kept):
            kept.append(i)
    return [templates[i] for i in kept]


def explain_away(templates, encoded, rng, reversal, min_events, min_members=6):
    order = sorted(range(len(templates)), key=lambda i: -int(templates[i]["members"].sum()))
    claimed = np.zeros(encoded.shape[1], bool)
    kept = []
    for i in order:
        t = templates[i]
        if (t["members"] & claimed).any():
            free = t["members"] & ~claimed
            if free.sum() < min_members:
                continue
            t = refine(encoded, free, rng, reversal, min_events, allowed=~claimed)
            if t is None or t["members"].sum() < min_members:
                continue
        kept.append(t)
        claimed |= t["members"]
    return kept


def event_groups_start(S, K, rng, reversal):
    """K starting templates from a spectral clustering of the events by their pairwise similarity (each group's
    leading eigenvector). Used where every event is a whole traversal (waking data)."""
    from sklearn.cluster import KMeans
    if S.shape[0] < 2 * K:
        return None
    U = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-300)
    G = np.abs(U @ U.conj().T)
    if reversal:
        G = np.maximum(G, np.abs(U @ U.T))
    np.fill_diagonal(G, 0)
    d = np.maximum(G.sum(axis=1), 1e-300)
    Y = np.linalg.eigh(G / np.sqrt(np.outer(d, d)))[1][:, -K:]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-300)
    labels = KMeans(K, n_init=10, random_state=int(rng.integers(2 ** 31 - 1))).fit_predict(Y)
    cols = [_leading(S[labels == g]) for g in range(K) if (labels == g).sum() >= 2]
    return np.column_stack(cols) if len(cols) >= 2 else None


def _orders(E, rng, K, reversal, min_events, floor, event_starts=False, min_active=3):
    from cpca.decompose import decompose, prepare
    from cpca.mixture import fit_templates
    act = (E != 0).sum(axis=1) >= min_active
    if act.sum() < floor:
        return None
    E = E[act]
    vals, comps, _, _ = decompose(E, n_components=K, center=True, scale=True)
    extra = {"events": lambda S: event_groups_start(S, K, rng, reversal)} if event_starts else None
    fitted = fit_templates(E, vals, comps, prepare(E, center=True, scale=True), rng, member_frac=1e-9,
                           min_active=min_active, min_events=min_events, conjugate=reversal, extra_starts=extra)
    return None if fitted is None or fitted["templates"].shape[1] < 2 else fitted


def _reference(E_tr, E_val, n, rng, reversal, n_sub=5, min_active=3):
    """How well a single template estimated from n events reproduces between the two halves (median of n_sub)."""
    from cpca.mixture import similarity as msim
    a = E_tr[(E_tr != 0).sum(axis=1) >= min_active]
    b = E_val[(E_val != 0).sum(axis=1) >= min_active]
    n = int(min(n, a.shape[0], b.shape[0]))
    if n < 2:
        return 1.0
    return float(np.median([msim(_leading(a[rng.choice(a.shape[0], n, replace=False)]),
                                 _leading(b[rng.choice(b.shape[0], n, replace=False)]), reversal) for _ in range(n_sub)]))


def split(encoded, validate, template, rng, reversal, min_events, max_k=4):
    """2-4 orders on the template's cells; the run with the most orders, all reproduced on `validate`, replaces it.

    Reactivations (reversal=True): events are sparse and partial, a split needs 4 x min_events active events, and an
    order is reproduced at similarity REPRODUCE or more. Waking data (reversal=False): every event is a whole
    traversal, so the events themselves, grouped by similarity, are also tried as a start; a split needs min_events
    active events; and an order is reproduced if it is at least as similar to its match as a single template
    estimated from the same number of events is (capped at REPRODUCE)."""
    waking = not reversal
    floor = min_events if waking else 4 * min_events
    from cpca.mixture import similarity as msim
    M = template["members"]
    if M.sum() < 6:
        return [template]
    best = None
    for K in range(2, max_k + 1):
        fitted = _orders(encoded[:, M], rng, K, reversal, min_events, floor, event_starts=waking)
        other = _orders(validate[:, M], rng, K, reversal, min_events, floor, event_starts=waking) if fitted is not None else None
        if fitted is None or other is None:
            continue
        T, U = fitted["templates"], other["templates"]
        sim = np.array([[msim(T[:, i], U[:, j], reversal) for j in range(U.shape[1])] for i in range(T.shape[1])])
        rows, cols = linear_sum_assignment(-sim)
        counts = np.bincount(fitted["label"][fitted["label"] >= 0], minlength=T.shape[1])
        bar = (lambda i: min(REPRODUCE, _reference(encoded[:, M], validate[:, M], counts[i], rng, reversal))) if waking \
            else (lambda i: REPRODUCE)
        good = [i for i, j in zip(rows, cols) if sim[i, j] >= bar(i)]
        if len(good) == T.shape[1] and (best is None or len(good) > best.shape[1]):
            best = T
    if best is None:
        return [template]
    out = []
    for j in range(best.shape[1]):
        q = np.zeros(encoded.shape[1], complex)
        q[M] = best[:, j]
        out.append({"template": q, "members": M.copy(), "events": template["events"]})
    return out


def _expression(Z, q, reversal):
    q = q / max(np.linalg.norm(q), 1e-300)
    F = np.abs(Z @ np.conj(q))
    return float(np.mean(np.maximum(F, np.abs(Z @ q)) ** 2 if reversal else F ** 2))


def _surrogate(times, T, rng, reversal):
    if reversal:
        from cpca.readout_check import redraw_latency
        return redraw_latency(times, T, rng)
    shift = rng.uniform(0, 1, times.shape[:2])[..., None] * np.asarray(T, float)[:, None, None]
    return np.sort(np.mod(times + shift, np.asarray(T, float)[:, None, None]), axis=-1)


def select(templates, te_times, te_T, rng, reversal, n_surrogates=N_SURROGATES, alpha=ALPHA, arc=np.pi):
    """Bonferroni-corrected p-value of each template's held-out expression against the timing null; keep p < alpha."""
    from cpca.encode import encode
    if not templates:
        return [], []
    Z = encode(te_times, te_T, arc=arc)
    real = np.array([_expression(Z, t["template"], reversal) for t in templates])
    ge = np.zeros(len(templates))
    for _ in range(n_surrogates):
        Zs = encode(_surrogate(te_times, te_T, rng, reversal), te_T, arc=arc)
        ge += np.array([_expression(Zs, t["template"], reversal) for t in templates]) >= real
    p = np.minimum(1.0, (1 + ge) / (1 + n_surrogates) * len(templates))
    return [t for t, pp in zip(templates, p) if pp < alpha], p.tolist()


def separate(result, tr_times, tr_T, te_times, te_T, rng, reversal=True, min_events=20, stages=None, rotation="quartimax",
             arc=np.pi):
    """The separation step on a cpca.fit result of the training events; held-out events validate splits and select
    the outputs. Returns (outputs, p-values). `stages` (for checks only) can switch stages off by name:
    'rotation', 'refine', 'explain', 'split', 'select'. rotation="ica" seeds the templates from the fit's ICA
    patterns (cpca.fit with ica=True) instead of rotating the kept components by quartimax."""
    from cpca.encode import encode
    off = set(stages or ())
    k = result["n_kept"]
    if not k:
        return [], []
    enc, val = encode(tr_times, tr_T, arc=arc), encode(te_times, te_T, arc=arc)
    V = result["pca_components"]
    if rotation == "ica":
        W = result["patterns"]
    else:
        W = quartimax(V)[0] if (k > 1 and "rotation" not in off) else V
    templates = []
    for w in W.T:
        seed = otsu_mask(np.abs(w))
        if "refine" in off:
            fitted = _template_on(enc, seed, rng, reversal, min_events)
            t = None if fitted is None else {"template": fitted[0], "members": seed, "events": fitted[1]}
        else:
            t = refine(enc, seed, rng, reversal, min_events)
        if t is not None:
            templates.append(t)
    templates = merge(templates, reversal)
    if "explain" not in off:
        templates = explain_away(templates, enc, rng, reversal, min_events)
    if "split" not in off:
        templates = [s for t in templates for s in split(enc, val, t, rng, reversal, min_events)]
    if "select" in off:
        return templates, [0.0] * len(templates)
    return select(templates, te_times, te_T, rng, reversal, arc=arc)
