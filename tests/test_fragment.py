import numpy as np

from cpca.fragment import best_fragment, timing_p


def test_an_ordered_fragment_is_found_and_unordered_outputs_are_calibrated():
    rng = np.random.default_rng(0)
    cen = rng.uniform(0, 100, 60)
    seg = np.where((cen > 20) & (cen < 60), cen + rng.normal(0, 3, 60), np.nan)
    res = best_fragment([seg], {1.0: cen}, 100.0, rng, n_null=300)
    assert res["abs"] > 0.9 and res["p"] < 0.01 and res["start"] <= 40          # a stretch covering the ordered cells
    ps = [best_fragment([np.where(rng.random(60) < 0.5, rng.random(60), np.nan) for _ in range(3)], {1.0: cen}, 100.0, rng, n_null=200)["p"]
          for _ in range(40)]
    assert np.mean(np.array(ps) < 0.05) <= 0.12


def test_too_few_cells_in_any_stretch_gives_no_test():
    rng = np.random.default_rng(1)
    cen = np.linspace(0, 100, 10)
    res = best_fragment([np.arange(10.0)], {1.0: cen}, 100.0, rng, n_null=50)
    assert res["index"] == -1 and np.isnan(res["p"])


def test_timing_p_counts_shuffles_at_least_as_good():
    assert timing_p(0.8, [0.1, 0.9, 0.5, 0.8]) == (1 + 2) / 5

