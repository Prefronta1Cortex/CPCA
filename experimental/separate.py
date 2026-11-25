"""Un-mixing complex PCA's kept components into one ordered template per sequence (docs/unmixing.md).

The kept components span the sequences but cut them badly: near-equal eigenvalues mix them, and a sequence played
both ways splits into a real pair c, s with q = c + js. The steps here use the components only to find where each
sequence lives, then fit its order on the raw arrows:

1. sparse rotation of the kept components within their span (experimental.rotation.quartimax);
2. each rotated component's members (largest-gap split of the weights) seed a template, fitted on those cells' raw
   arrows with the direction split (cpca.direction.fit_and_gate: backward events conjugated, events gated against a
   within-event shuffle);
3. refinement: the template is refitted on all cells using only the events that express it (backward ones
   conjugated), its members re-read, and the template refitted on them, until the members stop changing;
4. templates alike up to conjugation are merged (the two halves of a split pair give the same template).

Nothing here uses a map.
"""

import numpy as np

from cpca.direction import fit_and_gate
from experimental.rotation import quartimax
from cpca.theory import eigh_desc, lag_matrix


def largest_gap_mask(a):
    """Members: cells above the widest relative gap in the sorted weights (ignores the bottom half of the cells)."""
    a = np.asarray(a, float)
    order = np.argsort(a)[::-1]
    s = a[order]
    if s[0] <= 0:
        return np.zeros(a.shape, bool)
    half = max(2, s.size // 2)
    ratios = s[1:half] / np.maximum(s[: half - 1], 1e-300)
    cut = int(np.argmin(ratios)) + 1
    mask = np.zeros(a.shape, bool)
    mask[order[:cut]] = True
    return mask


def otsu_mask(a):
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


def _forward_fit_and_gate(S, rng, min_refit=5):
    """Forward-only template (no event conjugated) and its expressing events, gated as in fit_and_gate."""
    from cpca.direction import _leading, permute_within_events
    q = _leading(S)
    for _ in range(2):
        F = np.abs(S @ np.conj(q))
        null = np.array([np.abs(permute_within_events(S, rng) @ np.conj(q)) for _ in range(50)])
        express = F > np.percentile(null, 95, axis=0)
        if express.sum() < min_refit:
            break
        q = _leading(S[express])
    F = np.abs(S @ np.conj(q))
    null = np.array([np.abs(permute_within_events(S, rng) @ np.conj(q)) for _ in range(50)])
    express = F > np.percentile(null, 95, axis=0)
    return q, np.ones(S.shape[0], bool), F, np.zeros_like(F), 1, express


def _template_on(encoded, members, rng, min_active=3, min_events=20, conjugate=True):
    """Direction-split template on the members' raw arrows (forward only if conjugate=False);
    (template in cell space, expressing events, forward) or None."""
    S_all = encoded[:, members]
    active = np.flatnonzero((np.abs(S_all) > 0).sum(axis=1) >= min_active)
    if active.size < min_events or members.sum() < 3:
        return None
    gate = fit_and_gate if conjugate else _forward_fit_and_gate
    q, forward, _, _, _, express = gate(S_all[active], rng)
    t = np.zeros(encoded.shape[1], complex)
    t[members] = q
    return t, active[express], forward[express]


def refine(encoded, members, rng, max_iter=10, min_events=20, rule=otsu_mask, allowed=None, conjugate=True):
    """Alternate (template on members, gated events) and (members from the all-cell template on those events).

    allowed: cells that may become members (default all)."""
    allowed = np.ones(encoded.shape[1], bool) if allowed is None else allowed
    members = members & allowed
    history = []
    for _ in range(max_iter):
        fitted = _template_on(encoded, members, rng, min_events=min_events, conjugate=conjugate)
        if fitted is None:
            return None
        t, events, forward = fitted
        if events.size < min_events:
            return None
        X = encoded[events].copy()
        X[~forward] = np.conj(X[~forward])
        full = eigh_desc(lag_matrix(X))[1][:, 0]
        new = rule(np.where(allowed, np.abs(full), 0.0)) & allowed
        history.append(int(new.sum()))
        if np.array_equal(new, members):
            break
        members = new
    fitted = _template_on(encoded, members, rng, min_events=min_events, conjugate=conjugate)
    if fitted is None:
        return None
    t, events, forward = fitted
    return {"template": t, "members": members, "events": events, "forward": forward, "history": history}


def similarity(a, b, conjugate=True):
    a = a / max(np.linalg.norm(a), 1e-300)
    b = b / max(np.linalg.norm(b), 1e-300)
    if not conjugate:
        return abs(np.vdot(a, b))
    return max(abs(np.vdot(a, b)), abs(np.vdot(a, np.conj(b))))


def merge(templates, threshold=0.5, conjugate=True):
    """Greedy merge of templates alike (up to conjugation if conjugate=True); the one with more expressing events is kept."""
    order = sorted(range(len(templates)), key=lambda i: -templates[i]["events"].size)
    kept = []
    for i in order:
        if all(similarity(templates[i]["template"], templates[j]["template"], conjugate) < threshold for j in kept):
            kept.append(i)
    return [templates[i] for i in kept]


def explain_away(templates, encoded, rng, rule=otsu_mask, min_members=6, refine_iter=10, min_events=20, conjugate=True):
    """Largest first; each template loses the cells claimed by larger kept ones and is refitted on the rest."""
    order = sorted(range(len(templates)), key=lambda i: -int(templates[i]["members"].sum()))
    claimed = np.zeros(encoded.shape[1], bool)
    kept = []
    for i in order:
        t = templates[i]
        if (t["members"] & claimed).any():
            free = t["members"] & ~claimed
            if free.sum() < min_members:
                continue
            t = refine(encoded, free, rng, max_iter=refine_iter, rule=rule, allowed=~claimed, min_events=min_events,
                       conjugate=conjugate)
            if t is None or t["members"].sum() < min_members:
                continue
        kept.append(t)
        claimed |= t["members"]
    return kept


def event_cluster_start(S, K, rng, conjugate=False):
    """K starting templates from a spectral clustering of the events by their pairwise similarity |<s_i, s_j>|
    (unit-norm events; with conjugate=True also |<s_i, conj s_j>|). Meant for waking data, where every event is a
    whole sequence. Each group's start is the leading eigenvector of its lag matrix."""
    from sklearn.cluster import KMeans
    from cpca.direction import _leading
    if S.shape[0] < 2 * K:
        return None
    U = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-300)
    G = np.abs(U @ U.conj().T)
    if conjugate:
        G = np.maximum(G, np.abs(U @ U.T))
    np.fill_diagonal(G, 0)
    d = G.sum(axis=1)
    A = G / np.sqrt(np.outer(np.maximum(d, 1e-300), np.maximum(d, 1e-300)))
    vals, vecs = np.linalg.eigh(A)
    Y = vecs[:, -K:]
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-300)
    labels = KMeans(K, n_init=10, random_state=int(rng.integers(2 ** 31 - 1))).fit_predict(Y)
    cols = [_leading(S[labels == g]) for g in range(K) if (labels == g).sum() >= 2]
    return np.column_stack(cols) if len(cols) >= 2 else None


def _two_orders(E, rng, K=2, min_active=3, min_events=80, template_events=20, conjugate=True, event_starts=False):
    """K orders on E (events x members) by the template step; None unless at least two survive.
    min_events: active events needed to try; template_events: events a template needs to survive."""
    from cpca.decompose import decompose, prepare
    from experimental.template_step import fit_templates
    act = (E != 0).sum(axis=1) >= min_active
    if act.sum() < min_events:
        return None
    E = E[act]
    vals, comps, _, _ = decompose(E, n_components=K, center=True, scale=True)
    extra = {"events": lambda S: event_cluster_start(S, K, rng, conjugate)} if event_starts else None
    fitted = fit_templates(E, vals, comps, prepare(E, center=True, scale=True), rng, member_frac=1e-9, min_active=min_active,
                           min_events=template_events, conjugate=conjugate, extra_starts=extra)
    return None if fitted is None or fitted["templates"].shape[1] < 2 else fitted


def _orders_from_starts(E, starts, rng, min_active=3):
    """cpca.mixture.alternate on E (events x members) from each start; the run capturing most energy."""
    from experimental.template_step import alternate
    act = (E != 0).sum(axis=1) >= min_active
    if act.sum() < 80:
        return None
    S = E[act]
    runs = [alternate(S, Q0.copy(), rng) for Q0 in starts]
    best = max(runs, key=lambda r: r["energy"])
    return best if best["templates"].shape[1] else None


def split_global(encoded, template, starts, rng, validate, reproduce=0.8, min_active=3):
    """Orders on one template's cells, from the global components (and their ICA rotations) restricted to its
    members, as in the template step of cpca.mixture. Templates are kept only if refitting on the independent
    `validate` events reproduces them one-to-one (similarity at least `reproduce`); fewer than two kept means
    no split."""
    from scipy.optimize import linear_sum_assignment
    from experimental.template_step import similarity as msim
    M = template["members"]
    if M.sum() < 6:
        return [template]
    local = [Q[M] for Q in starts]
    fitted = _orders_from_starts(encoded[:, M], local, rng, min_active)
    if fitted is None or fitted["templates"].shape[1] < 2:
        return [template]
    other = _orders_from_starts(validate[:, M], local, rng, min_active)
    if other is None:
        return [template]
    T, U = fitted["templates"], other["templates"]
    sim = np.array([[msim(T[:, i], U[:, j]) for j in range(U.shape[1])] for i in range(T.shape[1])])
    rows, cols = linear_sum_assignment(-sim)
    good = [i for i, j in zip(rows, cols) if sim[i, j] >= reproduce]
    if len(good) < 2:
        return [template]
    out = []
    for j in good:
        q = np.zeros(encoded.shape[1], complex)
        q[M] = T[:, j]
        out.append({"template": q, "members": M.copy(), "events": template["events"], "split_from": int(M.sum())})
    return out


def _reference_reproduction(E_tr, E_val, n, rng, conjugate, n_sub=5, min_active=3):
    """Median similarity of the single template (leading eigenvector) fitted on n random active events of each half."""
    from cpca.direction import _leading
    from experimental.template_step import similarity as msim
    a = E_tr[(E_tr != 0).sum(axis=1) >= min_active]
    b = E_val[(E_val != 0).sum(axis=1) >= min_active]
    n = int(min(n, a.shape[0], b.shape[0]))
    if n < 2:
        return 1.0
    sims = [msim(_leading(a[rng.choice(a.shape[0], n, replace=False)]), _leading(b[rng.choice(b.shape[0], n, replace=False)]), conjugate)
            for _ in range(n_sub)]
    return float(np.median(sims))


def split_orders_k(encoded, template, rng, validate, max_k=4, min_active=3, reproduce=0.97, min_events=80,
                   template_events=20, conjugate=True, event_starts=False, trim=False, adaptive=False):
    """As split_orders with `validate`, starting K = 2 .. max_k templates on the template's cells. cpca.mixture.alternate
    may merge or drop some, so a run ends with at most K. The run with the most templates (at least two) that are all
    reproduced one-to-one on the validation events is kept; none: the single template."""
    from scipy.optimize import linear_sum_assignment
    from experimental.template_step import similarity as msim
    M = template["members"]
    if M.sum() < 6:
        return [template]
    best = None
    template["split_similarity"] = np.nan
    for K in range(2, max_k + 1):
        fitted = _two_orders(encoded[:, M], rng, K, min_active, min_events, template_events, conjugate, event_starts)
        other = (_two_orders(validate[:, M], rng, K, min_active, min_events, template_events, conjugate, event_starts)
                 if fitted is not None else None)
        if fitted is None or other is None:
            continue
        T, U = fitted["templates"], other["templates"]
        if T.shape[1] == 0 or U.shape[1] == 0:
            continue
        sim = np.array([[msim(T[:, i], U[:, j], conjugate) for j in range(U.shape[1])] for i in range(T.shape[1])])
        template["split_similarity"] = np.nanmax([template["split_similarity"], np.nanmax(sim)])
        rows, cols = linear_sum_assignment(-sim)
        if adaptive:
            counts = np.bincount(fitted["label"][fitted["label"] >= 0], minlength=T.shape[1])
            bars = {i: min(reproduce, _reference_reproduction(encoded[:, M], validate[:, M], counts[i], rng, conjugate))
                    for i in rows}
            good = [i for i, j in zip(rows, cols) if sim[i, j] >= bars[i]]
        else:
            good = [i for i, j in zip(rows, cols) if sim[i, j] >= reproduce]
        if len(good) == T.shape[1] and (best is None or len(good) > len(best[1])):
            best = (T, good)
    if best is None:
        return [template]
    T, good = best
    out = []
    for j in good:
        q = np.zeros(encoded.shape[1], complex)
        q[M] = T[:, j]
        mem = M.copy()
        if trim:
            mem = M & otsu_mask(np.abs(q))
            q = np.where(mem, q, 0)
        out.append({"template": q, "members": mem, "events": template["events"], "split_from": int(M.sum())})
    return out


def split_orders(encoded, template, rng, K=2, min_active=3, validate=None, reproduce=0.8):
    """Two orders on the same cells: the template step of cpca.mixture run on the template's members only.

    validate: raw arrows of other, independent events (the held-out half). The split is refitted on them and kept
    only if both templates are reproduced one-to-one (similarity at least `reproduce`, up to conjugation).
    Without `validate`, the split is kept only if both templates pass cpca.mixture.heldout_check."""
    from scipy.optimize import linear_sum_assignment
    from experimental.template_step import heldout_check, similarity as msim
    M = template["members"]
    if M.sum() < 6:
        return [template]
    E = encoded[:, M]
    fitted = _two_orders(E, rng, K, min_active)
    if fitted is None:
        return [template]
    if validate is None:
        act = (E != 0).sum(axis=1) >= min_active
        check = heldout_check(E[act], fitted, K, rng, center=True, scale=True, member_frac=1e-9, min_active=min_active)
        if check["keep"].sum() < 2:
            return [template]
        keep = np.flatnonzero(check["keep"])
    else:
        other = _two_orders(validate[:, M], rng, K, min_active)
        if other is None:
            return [template]
        T, U = fitted["templates"], other["templates"]
        sim = np.array([[msim(T[:, i], U[:, j]) for j in range(U.shape[1])] for i in range(T.shape[1])])
        rows, cols = linear_sum_assignment(-sim)
        good = [i for i, j in zip(rows, cols) if sim[i, j] >= reproduce]
        if len(good) < 2:
            return [template]
        keep = np.array(good)
    out = []
    for j in keep:
        q = np.zeros(encoded.shape[1], complex)
        q[M] = fitted["templates"][:, j]
        out.append({"template": q, "members": M.copy(), "events": template["events"], "split_from": int(M.sum())})
    return out


def separate(result, encoded, rng, seed_rule=otsu_mask, rule=otsu_mask, merge_threshold=0.5, refine_iter=10,
             explain=False, split=False, validate=None, reproduce=0.97, min_events=80, conjugate=True, event_starts=False,
             trim=False, adaptive=False):
    """Templates (list of dicts) from a complex PCA fit (`result`, kept components) and the raw arrows of the same events."""
    k = result["n_kept"]
    if not k:
        return []
    W, _ = quartimax(result["pca_components"]) if k > 1 else (result["pca_components"], None)
    template_min = min(20, min_events)
    templates = []
    for w in W.T:
        seed = seed_rule(np.abs(w))
        out = refine(encoded, seed, rng, max_iter=refine_iter, rule=rule, min_events=template_min,
                     conjugate=conjugate) if refine_iter else None
        if refine_iter == 0:
            fitted = _template_on(encoded, seed, rng, min_events=template_min, conjugate=conjugate)
            out = None if fitted is None else {"template": fitted[0], "members": seed, "events": fitted[1], "forward": fitted[2]}
        if out is not None:
            templates.append(out)
    templates = merge(templates, merge_threshold, conjugate)
    if explain:
        templates = explain_away(templates, encoded, rng, rule=rule, refine_iter=refine_iter, min_events=template_min,
                                 conjugate=conjugate)
    if split == "k" and validate is not None:
        templates = [s for t in templates for s in split_orders_k(encoded, t, rng, validate, reproduce=reproduce,
                                                                 min_events=min_events, template_events=template_min,
                                                                 conjugate=conjugate, event_starts=event_starts,
                                                                 trim=trim, adaptive=adaptive)]
    elif split == "global" and validate is not None and k > 1:
        from cpca.ica import unmix as ica_unmix
        V = result["pca_components"]
        starts = [V] + [ica_unmix(result["prepared"], result["eigenvalues"][:k], V, rng)[0] for _ in range(3)]
        templates = [s for t in templates for s in split_global(encoded, t, starts, rng, validate)]
    elif split:
        templates = [s for t in templates for s in split_orders(encoded, t, rng, validate=validate, reproduce=reproduce)]
    return templates


def global_with_sparse_starts(result, encoded, rng, sparse_templates, member_frac=0.2, min_active=3, n_ica=3):
    """The template step of cpca.mixture (all kept components' members, several templates at once, events assigned
    to the template they match either way round), with the sparse templates as one more start. The start whose
    templates capture the most energy on their gated events is kept (as in cpca.mixture.fit_templates)."""
    from cpca.ica import unmix as ica_unmix
    from experimental.template_step import alternate, members as mmembers
    k = result["n_kept"]
    if not k:
        return np.zeros((encoded.shape[1], 0), complex)
    V = result["pca_components"]
    U = np.any([mmembers(V[:, j], member_frac) for j in range(k)], axis=0)
    for t in sparse_templates:
        U |= t["members"]
    S_all = encoded[:, U]
    act = (S_all != 0).sum(axis=1) >= min_active
    if act.sum() < 40:
        return np.zeros((encoded.shape[1], 0), complex)
    S = S_all[act]
    starts = {"components": V[U]}
    if k > 1:
        for s in range(n_ica):
            starts[f"ica{s}"] = ica_unmix(result["prepared"], result["eigenvalues"][:k], V, rng)[0][U]
    if sparse_templates:
        starts["sparse"] = np.column_stack([t["template"][U] for t in sparse_templates])
    runs = {n: alternate(S, Q0.copy(), rng) for n, Q0 in starts.items()}
    best = max(runs, key=lambda n: runs[n]["energy"])
    T = np.zeros((encoded.shape[1], runs[best]["templates"].shape[1]), complex)
    T[U] = runs[best]["templates"]
    return T, best
