import numpy as np

from cpca.containment import basis, best_match, containment


def test_containment_of_a_component_in_its_own_span_and_of_noise():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(40, 2)) + 1j * rng.normal(size=(40, 2))
    inside = W @ np.array([1.0, 0.5j])
    reversed_ = np.conj(W[:, 0])
    noise = rng.normal(size=40) + 1j * rng.normal(size=40)
    c = containment(np.column_stack([inside, reversed_, noise]), basis(W))
    assert c[0] > 0.999 and c[1] > 0.999 and c[2] < 0.3
    assert best_match(reversed_[:, None], W)[0] > 0.999
