"""stats.py: empirical p-values, paired-delta bootstrap CIs, ranking stability and the repo's t-test (Table 10)."""
import numpy as np
import pytest

from valueheads_pythia.stats import empirical_p, paired_delta_ci, spearman_jaccard, summarize_random, ttest_paper_mode


def test_empirical_p_definition():
    r = [0.50, 0.60, 0.70]
    assert empirical_p(r, 0.55) == (1 + 1) / (1 + 3)                 # one random set at most as low as TopK
    assert empirical_p(r, 0.55, "ge") == (1 + 2) / (1 + 3)
    assert empirical_p(r, 0.40) == 1 / 4 and empirical_p(r, 0.90) == 1.0
    assert empirical_p(r + [float("nan")], 0.55) == 0.5               # NaN controls are dropped from R
    assert np.isnan(empirical_p([], 0.5)) and np.isnan(empirical_p(r, float("nan")))
    with pytest.raises(ValueError):
        empirical_p(r, 0.5, "lt")


def test_paired_delta_ci_item_cluster_and_weights():
    rng = np.random.RandomState(0)
    d = rng.normal(0.5, 1.0, 400)
    res = paired_delta_ci(d, B=500, seed=1)
    assert res["n"] == 400 and res["n_groups"] == 400 and res["B"] == 500
    assert np.isclose(res["mean"], d.mean()) and res["lo"] < res["mean"] < res["hi"]
    assert res["lo"] < 0.5 < res["hi"] and (res["hi"] - res["lo"]) < 0.4   # ~ +-2 * 1/sqrt(400)
    # constant deltas -> degenerate CI
    c = paired_delta_ci(np.ones(10), B=100)
    assert c["mean"] == c["lo"] == c["hi"] == 1.0
    # cluster bootstrap: strongly cluster-correlated deltas give a wider CI than the item bootstrap
    groups = np.repeat(np.arange(20), 20)
    clustered = rng.normal(0, 1, 20)[groups] + rng.normal(0, 0.1, 400)
    item = paired_delta_ci(clustered, B=500, seed=2)
    cluster = paired_delta_ci(clustered, B=500, seed=2, groups=groups)
    assert cluster["n_groups"] == 20 and (cluster["hi"] - cluster["lo"]) > 2 * (item["hi"] - item["lo"])
    # weights: token-weighted mean of perplexity windows
    w = np.array([1000, 1000, 10])
    v = np.array([1.0, 1.0, 100.0])
    res_w = paired_delta_ci(v, B=200, weights=w)
    assert np.isclose(res_w["mean"], np.average(v, weights=w))
    # NaN items are dropped together with their weight / group
    res_nan = paired_delta_ci([1.0, float("nan"), 3.0], B=50, groups=[0, 1, 2], weights=[1, 1, 1])
    assert res_nan["n"] == 2 and res_nan["mean"] == 2.0
    assert np.isnan(paired_delta_ci([], B=10)["mean"])
    with pytest.raises(ValueError):
        paired_delta_ci([1, 2], weights=[1])
    with pytest.raises(ValueError):
        paired_delta_ci([1, 2], groups=[1])


def test_spearman_jaccard():
    a = [3, 1, 2, 0]
    assert spearman_jaccard(a, a, 2) == {"spearman": 1.0, "jaccard": 1.0, "overlap": 2, "k": 2, "n": 4}
    rev = spearman_jaccard(a, a[::-1], 2)
    assert rev["spearman"] == -1.0 and rev["jaccard"] == 0.0 and rev["overlap"] == 0
    half = spearman_jaccard([0, 1, 2, 3], [0, 2, 1, 3], 2)
    assert half["overlap"] == 1 and half["jaccard"] == 1 / 3 and 0 < half["spearman"] < 1
    # score vectors (index = flat_idx) and {flat_idx: score} dicts
    s_a, s_b = np.array([0.5, 0.9, 0.7, 0.1]), np.array([0.4, 0.95, 0.8, 0.2])
    res = spearman_jaccard(s_a, s_b, 2, scores=True)
    assert res["spearman"] == 1.0 and res["jaccard"] == 1.0
    res_d = spearman_jaccard({i: v for i, v in enumerate(s_a)}, {i: v for i, v in enumerate(s_b)}, 1)
    assert res_d["overlap"] == 1 and res_d["spearman"] == 1.0
    with pytest.raises(ValueError):
        spearman_jaccard([0, 1, 2], [0, 1, 3], 1)


def _five(mean, std):
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / np.sqrt(2.5)      # mean 0, std(ddof=1) = 1
    return mean + std * z


@pytest.mark.parametrize("mean,std,topk,p_expected", [
    (0.346, 0.097, 0.200, 0.014),   # Llama-3.1-8B harmless (paper Table 10)
    (0.403, 0.084, 0.140, 0.001),   # Llama-3.1-8B honesty
    (0.704, 0.046, 0.620, 0.008),   # Qwen3-8B complexity
    (0.353, 0.116, 0.370, 0.620),   # Yi-1.5-9B harmless (TopK above the random mean)
    (0.921, 0.018, 0.960, 0.996),   # Llama-3.1-8B verbosity
])
def test_ttest_paper_mode_reproduces_table_10(mean, std, topk, p_expected):
    x = _five(mean, std)
    assert np.isclose(x.mean(), mean) and np.isclose(x.std(ddof=1), std)
    res = ttest_paper_mode(x, topk)
    assert res["n_random"] == 5 and np.isclose(res["random_mean"], mean) and np.isclose(res["random_std"], std)
    assert round(res["p"], 3) == p_expected
    assert np.isnan(ttest_paper_mode([0.3], 0.2)["p"])


def test_summarize_random():
    s = summarize_random([0.1, 0.3, float("nan")])
    assert s["n"] == 2 and np.isclose(s["mean"], 0.2) and np.isclose(s["std"], np.std([0.1, 0.3], ddof=1))
    assert s["min"] == 0.1 and s["max"] == 0.3
    assert summarize_random([])["n"] == 0 and np.isnan(summarize_random([0.5])["std"])
