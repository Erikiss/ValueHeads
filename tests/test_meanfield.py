"""Mean-field signatures of value-direction angles (valueheads_pythia/analysis/meanfield.py, docs/MEANFIELD.md).

Synthetic cases with known answers: the exact isotropic null N1 against Monte Carlo, isotropic directions (mean-field-like,
slope -1/2, location 0), a component shared by every head (systematic), layer-clustered residuals (collective), a
two-population angle distribution (bistable), persistence across checkpoints, predictability of the residual (g: iid
noise vs a structured term that averages out like noise, a depth profile, a magnitude that depends on the DoM norm, the
false-positive rate of the permutation test), and the end-to-end run on a synthetic Drive tree (discovery of a root at
the base and in a subfolder, outputs, Table-17 numbers = direction_angles, idempotency).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from valueheads_pythia.analysis import direction_angles as ang
from valueheads_pythia.analysis import meanfield as mf

B = 1000          # bootstrap draws / permutations in the unit cases (Holm floor 9 / 1001 < 0.05 in a one-row family)


def judged(row, model="m", revision="step143000", headset="all"):
    """One analyze_set row through Holm + verdict (a one-row family, not primary)."""
    row = dict(row, model=model, revision=revision, pair="a x b", headset=headset, prev_revision="step100000")
    return mf.judge(pd.DataFrame([row]), primary_revision="none").iloc[0]


def isotropic_r(rng, n, d):
    return mf.head_cosines(rng.standard_normal((n, d)), rng.standard_normal((n, d)))


def head_covariates(rng, n):
    """Synthetic (g) covariates as analyze_model passes them (log DoM norms, probe scores), independent of everything."""
    return {"log_norm_a": rng.normal(0, 0.3, n), "log_norm_b": rng.normal(0, 0.3, n), "score_a": rng.uniform(0.5, 0.9, n),
            "score_b": rng.uniform(0.5, 0.9, n)}


def standardized(x):
    return (x - x.mean()) / x.std()


def test_n1_exact_quantiles_match_monte_carlo():
    rng = np.random.default_rng(0)
    for d, sd_deg in ((64, 7.2), (128, 5.1)):
        r = isotropic_r(rng, 20000, d)
        assert np.var(r) == pytest.approx(1 / d, rel=0.05)
        assert np.degrees(np.arccos(r)).std() == pytest.approx(sd_deg, abs=0.15)      # the numbers quoted for Table 17
        assert stats.kstest(r, lambda x: mf.n1_cdf(x, d)).pvalue > 0.01
        q = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
        assert np.allclose(np.quantile(r, q), mf.n1_quantile(q, d), atol=0.006)
        ang_q = mf.n1_angle_quantile(q, d)
        assert np.allclose(np.quantile(np.degrees(np.arccos(r)), q), ang_q, atol=0.4)
        assert ang_q[2] == pytest.approx(90.0) and ang_q[0] + ang_q[-1] == pytest.approx(180.0)
        assert (np.degrees(np.arccos(r)) > 100).mean() == pytest.approx(mf.n1_frac_beyond(100, d), abs=0.006)
        assert mf.n1_cdf(mf.n1_quantile(q, d), d) == pytest.approx(q)
        xs = np.linspace(-0.999, 0.999, 4001)
        assert np.trapezoid(mf.n1_pdf(xs, d), xs) == pytest.approx(1.0, abs=1e-3)
    # d = 3: r is uniform on [-1, 1] (Archimedes), so the quantiles are 2q - 1 exactly
    assert mf.n1_quantile([0.1, 0.5, 0.9], 3) == pytest.approx([-0.8, 0.0, 0.8])
    assert mf.n1_excess_kurtosis(128) == pytest.approx(-6 / 130)
    lo, hi = mf.n1_mean_interval(100, 64)
    assert hi == pytest.approx(-lo) and hi == pytest.approx(1.959964 / 80, rel=1e-5)


def test_head_cosines_reuse_direction_angles():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((30, 16)).astype(np.float32), rng.standard_normal((30, 16)).astype(np.float32)
    a[3] = 0.0
    r = mf.head_cosines(a, b)
    for h in range(30):
        if h == 3:
            assert np.isnan(r[h]) and np.isnan(ang.angle_deg(a[h], b[h]))
        else:
            assert np.degrees(np.arccos(r[h])) == pytest.approx(ang.angle_deg(a[h], b[h]), abs=1e-6)
    x, y = mf.pair_with_cosines(np.array([0.3, -0.5, 0.0]), 64, rng)
    assert mf.head_cosines(x, y) == pytest.approx([0.3, -0.5, 0.0], abs=1e-12)


def test_holm():
    adj = mf.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06]) and np.isnan(adj[3])
    assert mf.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_isotropic_directions_are_mean_field(seed):
    rng = np.random.default_rng(seed)
    L, H, d = 24, 16, 128
    r = isotropic_r(rng, L * H, d)
    row = judged(mf.analyze_set(r, np.repeat(np.arange(L), H), d, np.random.default_rng(seed), n_boot=B, n_perm=B))
    assert row["verdict"] == "mean-field-artig", row["drivers"]
    assert abs(row["mean_r"]) < 3 / np.sqrt(L * H * d) and row["loc_in_n1_interval"]
    assert row["slope"] == pytest.approx(-0.5, abs=0.1) and row["slope_lo"] <= -0.5 <= row["slope_hi"]
    assert row["slope_null_mean"] == pytest.approx(-0.5, abs=0.05)      # random subsets: -1/2 after the correction
    assert row["sd_ratio"] == pytest.approx(1.0, abs=0.12) and row["neff_ratio"] == pytest.approx(1.0, abs=0.35)
    assert not row["bimodal"] and row["p_icc"] > 0.05
    assert row["kurt"] == pytest.approx(mf.n1_excess_kurtosis(d), abs=0.6)


def test_shared_component_across_heads_is_systematic():
    rng = np.random.default_rng(3)
    L, H, d = 24, 16, 128
    a, b, c = (rng.standard_normal((L * H, d)) for _ in range(3))
    r = mf.head_cosines(a + 0.6 * c, b + 0.6 * c)                     # every head shares c: r ~ 0.26 on all heads
    r0 = isotropic_r(rng, L * H, d)
    row = judged(mf.analyze_set(r, np.repeat(np.arange(L), H), d, np.random.default_rng(3), r0=r0, n_boot=B, n_perm=B))
    labels = row["verdict"].split("+")
    assert "systematisch" in labels and "mean-field-artig" not in labels
    assert row["mean_r"] > 0.2 and row["p_loc_n1_holm"] < 1e-6 and row["delta_r"] > 0.2 and row["p_loc_n2_holm"] < 1e-6
    assert row["verdict_code"].startswith("S") and "Lage≠0 (N1)" in row["drivers"]
    # within one realization a component shared by all heads is a constant shift: the averaging law (c) cannot see it
    # (the slope stays at -1/2), only the location test (a) does
    assert row["slope_lo"] <= -0.5 <= row["slope_hi"] and row["p_slope"] > 0.05


def test_layer_clustered_residuals_are_collective():
    rng = np.random.default_rng(4)
    L, H, d = 24, 16, 128
    layers = np.repeat(np.arange(L), H)
    offset = rng.normal(0, 0.12, L)
    offset -= offset.mean()                                            # no net location shift: only the clustering
    a, b = mf.pair_with_cosines(np.clip(offset[layers] + rng.normal(0, 1 / np.sqrt(d), L * H), -1, 1), d, rng)
    row = judged(mf.analyze_set(mf.head_cosines(a, b), layers, d, np.random.default_rng(4), n_boot=B, n_perm=B))
    labels = row["verdict"].split("+")
    assert "kollektiv" in labels and "mean-field-artig" not in labels
    assert row["icc"] > 0.4 and row["p_icc_holm"] < 0.05 and row["p_icc_F"] < 1e-10
    assert row["slope"] > -0.35 and row["slope_lo"] > -0.5 and row["p_slope_holm"] < 0.05
    assert row["neff_ratio"] < 0.5                                     # far fewer independent heads than heads
    assert "Layer-Häufung" in row["drivers"]


def test_random_subsets_cannot_see_correlation():
    """sd of random-subset means = S sqrt((1 - n/N) / n) for ANY fixed values (slope -1/2 by construction), while the
    windows in layer order expose the clustering: the reason why (c) uses contiguous windows against this null."""
    rng = np.random.default_rng(5)
    L, H = 16, 16
    layers = np.repeat(np.arange(L), H)
    x = rng.normal(0, 0.15, L)[layers] + rng.normal(0, 0.05, L * H)
    for n, sd_sim, sd_fpc in mf.random_subset_sd(x, [2, 8, 32, 128], rng, draws=3000):
        assert sd_sim == pytest.approx(sd_fpc, rel=0.06), n
    perms = np.argsort(rng.random((500, len(x))), axis=1)
    res = mf.averaging_law(x, layers, rng, perms, n_boot=300)
    assert res["slope"] > -0.3 and res["p_slope"] < 0.01


def test_two_population_angles_are_bistable():
    rng = np.random.default_rng(6)
    L, H, d = 24, 16, 128
    n = L * H
    minor = rng.random(n) < 0.15                                       # 15 % of the heads in directional conflict
    r = np.where(minor, np.cos(np.radians(rng.normal(110, 3, n))), isotropic_r(rng, n, d))
    row = judged(mf.analyze_set(r, np.repeat(np.arange(L), H), d, np.random.default_rng(6), n_boot=B, n_perm=B))
    assert row["bimodal"] and "bistabil" in row["verdict"].split("+") and "mean-field-artig" not in row["verdict"]
    assert row["gmm_angle_minor"] == pytest.approx(110, abs=3) and row["gmm_w_minor"] == pytest.approx(0.15, abs=0.05)
    assert row["gmm_angle_major"] == pytest.approx(90, abs=2) and row["gmm_ashman_d"] > 3 and row["gmm_dbic"] > 50
    assert row["gmm_dbic_skew"] > 50 and row["gmm_modes"] == 2       # also against a skewed one-component fit
    assert row["frac_gt100"] > 3 * row["frac_gt100_n1"] and row["p_conflict_n1"] < 1e-6
    assert "bimodal" in row["drivers"]
    # a single extreme head is not a population
    r1 = isotropic_r(rng, n, d)
    r1[7] = np.cos(np.radians(150))
    assert not mf.bimodality(r1)["bimodal"]


def old_gaussian_rule(b):
    """The former bimodality rule (2 Gaussians against a single Gaussian only)."""
    return b["gmm_dbic"] > mf.BIMODAL_DBIC and b["gmm_ashman_d"] > mf.BIMODAL_ASHMAN and b["gmm_w_minor"] >= mf.BIMODAL_MIN_WEIGHT


def test_skewed_unimodal_r_is_not_bistable():
    """A skewed but unimodal r (skew-normal a = 8, gamma k = 2; plausible for harmless x helpful) is split by a
    2-component mixture into a narrow core and a wide tail with D just above 2: against a single Gaussian it was flagged
    'bistabil' in ~15-55 % of runs at n = 144-384; against the skew-normal alternative it is not."""
    n, d = 384, 128
    old = 0
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        for x in (stats.skewnorm.rvs(8, size=n, random_state=rng), rng.gamma(2.0, size=n)):
            r = 2 * (x - x.mean()) / x.std() / np.sqrt(d)                # twice the isotropic sd, as in Table 17
            assert stats.skew(r) > 0.5
            b = mf.bimodality(r)
            assert not b["bimodal"], (seed, b)
            assert b["gmm_dbic_skew"] <= b["gmm_dbic"]
            old += old_gaussian_rule(b)
    assert old >= 2                     # the cases the former rule got wrong are among them
    # the full row: not bistable
    rng = np.random.default_rng(9)
    x = stats.skewnorm.rvs(8, size=n, random_state=rng)
    r = (x - x.mean()) / x.std() / np.sqrt(d)
    row = judged(mf.analyze_set(r, np.repeat(np.arange(24), 16), d, np.random.default_rng(9), n_boot=B, n_perm=B))
    assert not row["bimodal"] and "bistabil" not in row["verdict"]


def test_sparse_set_without_layer_test_is_not_mean_field():
    """18 heads of a 16 x 8 model, almost one head per layer, strong layer offsets (ICC 0.5): the layer test cannot be
    run (p_icc NaN), which must not count as 'no clustering'; the slope CI of 18 heads is also too wide."""
    rng = np.random.default_rng(11)
    L, H, d = 16, 8, 256
    off = rng.normal(0, np.sqrt(0.5 / d), L)
    idx = np.unique(np.concatenate([np.arange(L) * H + rng.integers(0, H, L), [H + 7, 5 * H + 7]]))
    layers = idx // H
    r = off[layers] + rng.normal(0, np.sqrt(0.5 / d), len(idx))
    row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(11), n_boot=B, n_perm=B), headset="shared_top15")
    labels = row["verdict"].split("+")
    assert len(idx) >= mf.MIN_HEADS and np.isnan(row["p_icc"])
    assert "mean-field-artig" not in labels and "unterbestimmt" in labels
    assert "Layer-Häufung nicht testbar" in row["drivers"] and "keine Layer-Häufung" not in row["drivers"]
    assert row["slope_hi"] - row["slope_lo"] > mf.MAX_SLOPE_CI_WIDTH and "Steigungs-KI zu breit" in row["drivers"]
    assert row["verdict_code"] == mf.UNDERDETERMINED_MF_CODE         # location, slope and shape fit, the guards do not


def test_verdict_guards():
    base = {"model": "m", "revision": "step143000", "pair": "a x b", "headset": "all", "n_heads": 384, "n_layers": 24,
            "mean_r": 0.001, "loc_sign_ok": True, "delta_sign_ok": True, "slope": -0.49, "slope_lo": -0.57,
            "slope_hi": -0.41, "slope_ci_has_half": True, "icc": 0.01, "bimodal": False, "rho_prev": 0.0,
            "prev_revision": "step100000"}
    for t in mf.TESTS:
        base[f"p_{t}"] = 0.5
    ok = mf.judge(pd.DataFrame([base]), primary_revision="none").iloc[0]
    assert ok["verdict"] == "mean-field-artig" and ok["verdict_code"] == "M" and "keine Layer-Häufung (ICC=" in ok["drivers"]
    untestable = mf.judge(pd.DataFrame([dict(base, p_icc=np.nan)]), primary_revision="none").iloc[0]
    assert untestable["verdict"] == "unterbestimmt" and untestable["verdict_code"] == "m"
    assert "Layer-Häufung nicht testbar (384 Heads in 24 Layern" in untestable["drivers"]
    wide = mf.judge(pd.DataFrame([dict(base, slope_lo=-0.9, slope_hi=-0.2)]), primary_revision="none").iloc[0]
    assert wide["verdict"] == "unterbestimmt" and "Steigungs-KI zu breit (0.70 > 0.5)" in wide["drivers"]
    # other labels stay, the untestable layer test is still named
    shifted = dict(base, p_icc=np.nan, mean_r=0.2, p_loc_n1=1e-8)
    s = mf.judge(pd.DataFrame([shifted]), primary_revision="none").iloc[0]
    assert s["verdict"] == "systematisch" and s["verdict_code"] == "S" and "Layer-Häufung nicht testbar" in s["drivers"]
    # (g) not computable: "not computable" must not count as "not predictable"
    for missing in ("p_pred_r", "p_pred_absr"):
        nopred = mf.judge(pd.DataFrame([dict(base, **{missing: np.nan})]), primary_revision="none").iloc[0]
        assert nopred["verdict"] == "unterbestimmt" and nopred["verdict_code"] == "m"
        assert "Vorhersagbarkeit des Rests (g) nicht berechenbar" in nopred["drivers"]
    # (g) significant: vorhersagbar instead of mean-field-artig, the drivers name R^2 and the leading covariate
    for key, name in (("r", "(r)"), ("absr", "(|r|)")):
        pred = dict(base, **{f"p_pred_{key}": 1e-6, f"pred_{key}_r2": 0.82, f"pred_{key}_r2_null": -0.01,
                             f"pred_{key}_top": "depth"})
        v = mf.judge(pd.DataFrame([pred]), primary_revision="none").iloc[0]
        assert v["verdict"] == "vorhersagbar" and v["verdict_code"] == "V", v["drivers"]
        assert f"Rest vorhersagbar {name}: R²=+0.82" in v["drivers"] and "stärkste Kovariate Tiefe" in v["drivers"]
        assert "aber der Rest ist vorhersagbar" in v["drivers"]
    # (g) is a descriptive family: not part of the primary Holm family, Holm over all tests of the row
    assert "pred_r" in mf.TESTS and "pred_absr" in mf.TESTS and not set(mf.PRIMARY_TESTS) & {"pred_r", "pred_absr"}
    prim = mf.judge(pd.DataFrame([dict(base, headset="shared_top15", p_pred_r=0.004)]), primary_revision="step143000").iloc[0]
    assert prim["primary"] and "p_pred_r_primary_holm" not in prim
    assert prim["p_pred_r_holm"] == pytest.approx(min(1.0, len(mf.TESTS) * 0.004))
    assert prim["verdict"] == "vorhersagbar"


def test_widened_confidence_intervals():
    """Slope CI: each half >= 1.96 x the sd of the permutation null; var / sd_ratio CI contains the chi^2 interval."""
    rng = np.random.default_rng(12)
    L, H, d = 12, 12, 64
    layers = np.repeat(np.arange(L), H)
    r = isotropic_r(rng, L * H, d)
    perms = np.argsort(rng.random((300, L * H)), axis=1)
    res = mf.averaging_law(r, layers, rng, perms, n_boot=300)
    z = stats.norm.ppf(0.975)
    assert res["slope_hi"] - res["slope"] >= z * res["slope_null_sd"] - 1e-12
    assert res["slope"] - res["slope_lo"] >= z * res["slope_null_sd"] - 1e-12
    n = 20
    x = r[:n]
    sp = mf.spread_test(x, d, rng.integers(0, n, size=(500, n)))
    lo_c, hi_c = (n - 1) * x.var(ddof=1) * d / stats.chi2.ppf([0.975, 0.025], n - 1)
    assert sp["sd_ratio_lo"] <= np.sqrt(lo_c) + 1e-12 and sp["sd_ratio_hi"] >= np.sqrt(hi_c) - 1e-12
    assert sp["excess_var_lo"] == pytest.approx(sp["sd_ratio_lo"] ** 2 / d - 1 / d)


def test_persistence_identical_vs_independent():
    rng = np.random.default_rng(7)
    L, H, d = 16, 16, 64
    layers = np.repeat(np.arange(L), H)
    r = isotropic_r(rng, L * H, d)
    perms = np.argsort(rng.random((B, L * H)), axis=1)
    rho, p = mf.persistence_test(r, r.copy(), rng, perms)
    assert rho == pytest.approx(1.0) and p == pytest.approx(1 / (B + 1))
    rho, p = mf.persistence_test(r, r + rng.normal(0, 0.03, L * H), rng, perms)
    assert rho > 0.8 and p < 0.01
    rho, p = mf.persistence_test(r, isotropic_r(rng, L * H, d), rng, perms)
    assert abs(rho) < 0.15 and p > 0.05
    # through the verdict: identical table at the previous checkpoint -> systematisch by persistence only ("P")
    row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(7), r_prev=r.copy(), n_boot=B, n_perm=B))
    assert "systematisch" in row["verdict"] and "P" in row["verdict_code"] and "Persistenz" in row["drivers"]
    row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(7), r_prev=isotropic_r(rng, L * H, d), n_boot=B, n_perm=B))
    assert "systematisch" not in row["verdict"] and row["p_persist_prev"] > 0.05


def naive_ridge_cv_r2(y, Z, folds, k, alphas):
    """Reference for ridge_cv_r2: explicit nested CV with one fit per (outer fold, inner fold, penalty)."""
    def fit(train, a):
        mz, my = Z[train].mean(axis=0), y[train].mean()
        zc, yc = Z[train] - mz, y[train] - my
        beta = np.linalg.solve(zc.T @ zc + a * train.sum() * np.eye(Z.shape[1]), zc.T @ yc)
        return beta, my - mz @ beta

    res = tot = 0.0
    for f in range(k):
        inner = []
        for a in alphas:
            err = 0.0
            for j in set(range(k)) - {f}:
                beta, c = fit((folds != f) & (folds != j), a)
                err += ((y[folds == j] - Z[folds == j] @ beta - c) ** 2).sum()
            inner.append(err)
        beta, c = fit(folds != f, alphas[int(np.argmin(inner))])
        res += ((y[folds == f] - Z[folds == f] @ beta - c) ** 2).sum()
        tot += ((y[folds == f] - y[folds != f].mean()) ** 2).sum()
    return 1 - res / tot


def test_ridge_cv_and_block_permutations():
    rng = np.random.default_rng(20)
    layers = np.repeat(np.arange(12), 12)
    folds = mf.pred_folds(layers)
    assert sorted(np.unique(folds)) == list(range(mf.PRED_FOLDS))
    for f in range(mf.PRED_FOLDS):                                     # whole layers, spread over the depth
        held = np.unique(layers[folds == f])
        assert all((folds[layers == g] == f).all() for g in held) and held.max() - held.min() >= 5
    assert mf.pred_folds(np.arange(4)) is None
    Z = rng.standard_normal((len(layers), 5))
    Y = np.vstack([0.4 * Z[:, 0] + rng.standard_normal(len(layers)) + 2.0, rng.standard_normal(len(layers))])
    fast = mf.ridge_cv_r2(Y, Z, folds)["r2"]
    for row, value in zip(Y, fast):
        assert value == pytest.approx(naive_ridge_cv_r2(row, Z, folds, mf.PRED_FOLDS, mf.PRED_ALPHAS), abs=1e-10)
    assert fast[0] > 0.05 and fast[1] < 0.02
    # block permutations: true permutations; a layer's heads land together on one layer of the same size
    lay = np.array([0, 0, 1, 2, 2, 3, 4, 4, 4, 5, 6, 6])
    perms = mf.layer_block_perms(lay, rng, 300)
    assert (np.sort(perms, axis=1) == np.arange(len(lay))).all()
    for p in perms[:50]:
        for g in np.unique(lay):
            src = np.unique(lay[p[lay == g]])
            assert len(src) == 1 and (lay == src[0]).sum() == (lay == g).sum()
    assert len({tuple(p) for p in perms}) > 100 and (perms[:, 6:9] >= 6).all() and (perms[:, 6:9] <= 8).all()


def test_iid_residuals_are_not_predictable():
    """iid residuals, covariates independent of them: held-out R^2 <= ~0, (g) not significant, still mean-field-artig."""
    L, H, d = 24, 16, 128
    layers = np.repeat(np.arange(L), H)
    mean_field, sig = 0, 0
    for seed in range(6):
        rng = np.random.default_rng(30 + seed)
        r, r0 = isotropic_r(rng, L * H, d), isotropic_r(rng, L * H, d)
        row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(seed), r0=r0, n_boot=500, n_perm=500,
                                    covariates=head_covariates(rng, L * H), n_layers_model=L))
        assert row["pred_r_r2"] < 0.02 and row["pred_absr_r2"] < 0.02
        assert row["pred_n_cov"] == 7 and row["pred_covariates"].split() == ["depth", "depth2", "log_norm_a", "log_norm_b",
                                                                              "score_a", "score_b", "r_step0"]
        sig += (row["p_pred_r_holm"] < 0.05) + (row["p_pred_absr_holm"] < 0.05)
        mean_field += "mean-field-artig" in row["verdict"].split("+")
    assert sig <= 1 and mean_field >= 5


def test_structured_residual_averages_like_noise_but_is_predictable():
    """The demo behind (g): a residual driven by a head property (here the probe score, not tied to the layer order)
    shrinks like 1/sqrt(n) exactly as iid noise does - same averaging slope, no layer clustering, unimodal, location 0 -
    so (a)-(f) call it mean-field-artig; only its out-of-sample predictability (R^2 ~ 0.8 vs ~ 0) separates it."""
    L, H, d = 24, 16, 128
    n = L * H
    layers = np.repeat(np.arange(L), H)
    rng = np.random.default_rng(40)
    cov = head_covariates(rng, n)
    e = rng.standard_normal(n)
    rows = {}
    for name, r in (("noise", e / np.sqrt(d)), ("structured", (0.9 * standardized(cov["score_a"]) + np.sqrt(0.19) * e) / np.sqrt(d))):
        rows[name] = judged(mf.analyze_set(r, layers, d, np.random.default_rng(41), n_boot=B, n_perm=B, covariates=cov,
                                           n_layers_model=L))
    noise, structured = rows["noise"], rows["structured"]
    for row in rows.values():                                          # the averaging law cannot tell them apart
        assert row["slope_lo"] <= -0.5 <= row["slope_hi"] and row["slope"] == pytest.approx(-0.5, abs=0.1)
        assert row["p_icc_holm"] > 0.05 and not row["bimodal"] and row["p_loc_n1_holm"] > 0.05
    assert noise["verdict"] == "mean-field-artig" and noise["pred_r_r2"] < 0.02 and noise["p_pred_r_holm"] > 0.05
    assert structured["pred_r_r2"] > 0.7 and structured["p_pred_r_holm"] < 0.05 and structured["pred_r_top"] == "score_a"
    assert structured["verdict"] == "vorhersagbar" and structured["verdict_code"] == "V"
    assert "aber der Rest ist vorhersagbar" in structured["drivers"] and "Probe-Score a" in structured["drivers"]


def test_depth_profile_is_predictable():
    """r = f(depth) + noise (a smooth depth profile with mean 0): predictable, not mean-field-artig."""
    L, H, d = 24, 16, 128
    layers = np.repeat(np.arange(L), H)
    rng = np.random.default_rng(50)
    f = standardized(np.cos(np.pi * layers / (L - 1)))
    r = (0.6 * f + 0.8 * rng.standard_normal(L * H)) / np.sqrt(d)
    row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(51), r0=isotropic_r(rng, L * H, d), n_boot=B,
                                n_perm=B, covariates=head_covariates(rng, L * H), n_layers_model=L))
    labels = row["verdict"].split("+")
    assert "vorhersagbar" in labels and "mean-field-artig" not in labels and "V" in row["verdict_code"]
    assert row["pred_r_r2"] > 0.2 and row["p_pred_r_holm"] < 0.05 and row["pred_r_top"] in ("depth", "depth2")
    assert row["pred_r_r2"] > row["pred_r_r2_null"] + 0.2


def test_magnitude_depending_on_dom_norm_is_predictable():
    """Heteroscedastic residual: sign random, |r| grows with the DoM norm of value a -> (g) on |r| significant."""
    L, H, d = 24, 16, 128
    n = L * H
    layers = np.repeat(np.arange(L), H)
    rng = np.random.default_rng(60)
    cov = head_covariates(rng, n)
    s = np.exp(2.0 * cov["log_norm_a"])
    r = rng.standard_normal(n) * s / np.sqrt((s ** 2).mean()) / np.sqrt(d)
    row = judged(mf.analyze_set(r, layers, d, np.random.default_rng(61), n_boot=B, n_perm=B, covariates=cov,
                                n_layers_model=L))
    assert row["p_pred_absr_holm"] < 0.05 and row["pred_absr_r2"] > 0.1 and row["pred_absr_top"] == "log_norm_a"
    assert row["p_pred_r_holm"] > 0.05 and abs(row["pred_r_r2"]) < 0.05       # the signed residual is not predictable
    assert "vorhersagbar" in row["verdict"].split("+") and "mean-field-artig" not in row["verdict"]
    assert "Rest vorhersagbar (|r|)" in row["drivers"]


def test_frozen_init_residual_counts_as_predictable():
    """r(h) = 0.8 r_step0(h) + noise: predictable from the init reference (signed via r_step0, magnitude via |r_step0|)."""
    rng = np.random.default_rng(70)
    L, H = 12, 12
    layers = np.repeat(np.arange(L), H)
    r0 = rng.standard_normal(L * H)
    r = 0.8 * r0 + 0.6 * rng.standard_normal(L * H)
    out = mf.predictability(r, layers, {"depth": layers / (L - 1), "r_step0": r0}, rng, n_perm=500)
    assert out["pred_r_r2"] > 0.5 and out["p_pred_r"] < 0.01 and out["pred_r_top"] == "r_step0"
    assert out["pred_absr_top"] == "|r_step0|" and "|r_step0|:" in out["pred_absr_coefs"]


def test_predictability_permutation_false_positive_rate():
    """~0.05 false positives of (g) at alpha = 0.05 over 120 null simulations each: iid residuals, and layer offsets
    (ICC 0.3) that do not depend on the covariates while the DoM norms do grow with depth - the layer-block
    permutations keep the clustering, so the test stays at its level there (plain shuffles: ~0.10, docs/MEANFIELD.md);
    the same for a sparse head set with unevenly filled layers."""
    L, H, d, n_sim = 12, 12, 64, 120
    layers = np.repeat(np.arange(L), H)
    n = L * H
    depth = layers / (L - 1)
    for icc in (0.0, 0.3):
        hits = np.zeros(2)
        for s in range(n_sim):
            rng = np.random.default_rng(1000 + s)
            r = rng.normal(0, np.sqrt(icc / d), L)[layers] + rng.normal(0, np.sqrt((1 - icc) / d), n)
            cov = {"depth": depth, "depth2": (depth - 0.5) ** 2, **head_covariates(rng, n), "r_step0": rng.standard_normal(n)}
            if icc:
                cov["log_norm_a"] = cov["log_norm_a"] + 2.0 * depth
            out = mf.predictability(r, layers, cov, rng, n_perm=199)
            hits += [out["p_pred_r"] <= 0.05, out["p_pred_absr"] <= 0.05]
        fpr = hits / n_sim
        assert (fpr >= 0.01).all() and (fpr <= 0.10).all(), (icc, fpr)
    # sparse set with unevenly filled layers (like the shared top-15 % heads: 36 of 24 x 16, layer weights ~ Dirichlet):
    # most layers have a head count of their own and cannot be moved, the permutation group is smaller but still exact
    L, H, k = 24, 16, 36
    hits, ok = np.zeros(2), 0
    for s in range(n_sim):
        rng = np.random.default_rng(2000 + s)
        idx = np.sort(rng.choice(L * H, k, replace=False, p=np.repeat(rng.dirichlet(np.ones(L)), H) / H))
        lay, dep = idx // H, idx // H / (L - 1)
        r = rng.normal(0, np.sqrt(0.3), L)[lay] + rng.normal(0, np.sqrt(0.7), k)
        cov = {"depth": dep, "depth2": (dep - 0.5) ** 2, **head_covariates(rng, k), "r_step0": rng.standard_normal(k)}
        cov["log_norm_a"] = cov["log_norm_a"] + 2.0 * dep
        out = mf.predictability(r, lay, cov, rng, n_perm=199)
        if np.isfinite(out["p_pred_r"]):
            ok += 1
            hits += [out["p_pred_r"] <= 0.05, out["p_pred_absr"] <= 0.05]
    assert ok >= 0.9 * n_sim and (hits / ok >= 0.01).all() and (hits / ok <= 0.10).all(), hits / ok


def test_layer_blocked_cv_does_not_leak_layer_offsets():
    """Strong layer offsets (ICC 0.8) unrelated to the covariates, two covariates constant within a layer (they identify
    the layer): CV over single heads learns a layer's offset from its other heads (held-out R^2 > 0, leakage); folds of
    whole layers (pred_folds) cannot, held-out R^2 <= 0."""
    L, H = 24, 16
    layers = np.repeat(np.arange(L), H)
    n, depth = L * H, layers / (L - 1)
    blocked, by_head = [], []
    for s in range(30):
        rng = np.random.default_rng(900 + s)
        y = standardized(rng.normal(0, np.sqrt(0.8), L)[layers] + rng.normal(0, np.sqrt(0.2), n))
        Z = np.column_stack([depth, (depth - 0.5) ** 2, rng.standard_normal(L)[layers], rng.standard_normal(L)[layers]])
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        blocked.append(mf.ridge_cv_r2(y, Z, mf.pred_folds(layers))["r2"][0])
        by_head.append(mf.ridge_cv_r2(y, Z, rng.permutation(n) % mf.PRED_FOLDS)["r2"][0])
    assert np.mean(blocked) < 0 and np.mean(by_head) > 0.05, (np.mean(blocked), np.mean(by_head))


def test_predictability_not_computable():
    rng = np.random.default_rng(80)
    cov = {"depth": np.repeat(np.arange(12), 2) / 11.0}
    out = mf.predictability(rng.standard_normal(10), np.arange(10), {"depth": np.arange(10) / 9}, rng, n_perm=50)
    assert np.isnan(out["p_pred_r"]) and np.isnan(out["pred_r_r2"])               # n < MIN_HEADS
    out = mf.predictability(rng.standard_normal(40), np.repeat(np.arange(4), 10), {"depth": np.repeat(np.arange(4), 10) / 3},
                            rng, n_perm=50)
    assert np.isnan(out["p_pred_absr"])                                           # fewer layers than folds
    r, layers = rng.standard_normal(24), np.repeat(np.arange(12), 2)
    out = mf.predictability(r, layers, {"score_a": np.full(24, 0.7), "score_b": None, "bad": np.r_[np.nan, np.ones(23)]},
                            rng, n_perm=50)
    assert np.isnan(out["p_pred_r"]) and out["pred_n_cov"] == 0                   # no usable covariate
    out = mf.predictability(r, layers, cov, rng, n_perm=50)
    assert np.isfinite(out["p_pred_r"]) and out["pred_covariates"] == "depth" and out["pred_folds"] == mf.PRED_FOLDS
    # through analyze_set: a sparse set in 4 layers has no (g); with an otherwise mean-field pattern it is "m"
    rng = np.random.default_rng(81)
    row = mf.analyze_set(isotropic_r(rng, 64, 64), np.repeat(np.arange(4), 16), 64, rng, n_boot=200, n_perm=200)
    assert np.isnan(row["p_pred_r"]) and np.isnan(row["p_pred_absr"])


def test_small_and_sparse_sets():
    rng = np.random.default_rng(8)
    r = isotropic_r(rng, 10, 64)
    row = judged(mf.analyze_set(r, np.arange(10), 64, rng, n_boot=200, n_perm=200), headset="shared_top15")
    assert "unterbestimmt" in row["verdict"] and np.isnan(row["slope"]) and np.isnan(row["icc"])
    # one head per layer: no within-layer degrees of freedom, no clustering test
    assert np.isnan(mf.layer_structure(isotropic_r(rng, 30, 64), np.arange(30), rng, np.argsort(rng.random((50, 30)), axis=1))["p_icc"])
    row = mf.analyze_set(np.array([0.1, np.nan]), np.array([0, 1]), 64, rng)
    assert row["n_heads"] == 1 and row["n_dropped_nan"] == 1 and np.isnan(row["p_loc_n1"])
    empty = judged(mf.analyze_set(np.array([]), np.array([], dtype=int), 64, rng), headset="shared_top15")
    assert empty["n_heads"] == 0 and empty["verdict"] == "unterbestimmt"


def test_synthetic_tree_end_to_end(tmp_path):
    base = str(tmp_path / "ValueHeadsPythia")
    roots = mf.make_synthetic_tree(base)
    # noise the discovery must ignore: other values / paper mode in the directions folder, a features folder with a
    # nested results/directions, an empty root
    open(os.path.join(base, "results", "directions", "pythia-160m_step0_harmless_pooled.npz"), "w").close()
    open(os.path.join(base, "results", "directions", "pythia-160m_step0_honesty_paper.npz"), "w").close()
    os.makedirs(os.path.join(base, "features", "x", "results", "directions"))
    os.makedirs(os.path.join(base, "empty", "results", "directions"))
    old = os.path.join(base, "old_copy", "results", "directions")          # a stale partial copy of pythia-1b
    os.makedirs(old)
    name = "pythia-1b_step0_honesty.npz"
    with open(os.path.join(roots[1], "results", "directions", name), "rb") as f, open(os.path.join(old, name), "wb") as g:
        g.write(f.read())
    found = mf.discover(base)
    assert list(found) == ["pythia-160m", "pythia-1b"]
    assert found["pythia-160m"]["root"] == os.path.abspath(roots[0]) and found["pythia-1b"]["root"] == os.path.abspath(roots[1])
    assert any("old_copy" in n for n in found["pythia-1b"]["notes"])
    assert list(found["pythia-160m"]["revisions"]) == ["step0", "step1000", "step143000"]
    assert all(f["heads"] for f in found["pythia-1b"]["revisions"]["step0"].values())
    table = mf.discovery_table(found)
    assert list(table["d"]) == [64, 256] and list(table["directions"]) == [9, 9]

    logs = []
    res = mf.run_all(base, n_boot=200, n_perm=200, say=logs.append)
    df = res["models"]["pythia-160m"]["df"]
    assert len(df) == 3 * 3 * 2 and set(df["headset"]) == set(mf.HEADSETS)
    for model, r in res["models"].items():
        for key in ("csv", "md", "json"):
            assert os.path.exists(r["paths"][key])
        pngs = sorted(os.listdir(r["paths"]["figures"]))
        assert len(pngs) == 5 and any(p.endswith("_trajektorie.png") for p in pngs)
    assert res["combined"] and os.path.exists(res["combined"]) and os.path.exists(res["width_figure"])
    assert len(pd.read_csv(res["combined"])) == 36

    fin = df[df["revision"] == "step143000"].set_index(["pair", "headset"])
    hh = fin.loc[("harmless x helpful", "all")]
    assert hh["mean_r"] > 0.2 and "systematisch" in hh["verdict"] and hh["angle_mean_deg"] < 80
    for pair in ("harmless x honesty", "helpful x honesty"):
        assert abs(fin.loc[(pair, "all"), "mean_r"]) < 0.05
    step0 = df[df["revision"] == "step0"]
    assert step0["p_loc_n2"].isna().all() and step0["p_persist_prev"].isna().all() and step0["p_persist_final"].notna().all()
    prim = df[df["primary"]]
    assert len(prim) == 3 and set(prim["headset"]) == {"shared_top15"} and set(prim["revision"]) == {"step143000"}
    assert prim["p_loc_n1_primary_holm"].notna().all() and (df.loc[~df["primary"], "status"] == "deskriptiv").all()

    # Table-17 numbers are the ones of analysis/direction_angles on the same directions and top-15 % heads
    data, _ = mf.load_model_data("pythia-160m", found["pythia-160m"])
    d143 = data["step143000"]
    t17 = ang.angle_table({v: d143["dirs"][v] for v in ("harmless", "helpful")},
                          {v: d143["top"][v] for v in ("harmless", "helpful")}).iloc[0]
    row = fin.loc[("harmless x helpful", "shared_top15")]
    assert row["angle_n"] == t17["n_shared"] and row["angle_mean_deg"] == pytest.approx(t17["mean_deg"])
    assert row["angle_std_deg"] == pytest.approx(t17["std_deg"]) and row["heads"] == " ".join(map(str, sorted(t17["shared_heads"])))

    # (g): covariates from the directions (DoM norms) and the head tables (stored AUC), r_step0 except at step0
    assert df.loc[df["headset"] == "all", "p_pred_r"].notna().all() and df.loc[df["headset"] == "all", "p_pred_absr"].notna().all()
    for _, row in df[df["headset"] == "all"].iterrows():
        names = row["pred_covariates"].split()
        assert {"depth", "depth2", "log_norm_a", "log_norm_b", "score_a", "score_b"} <= set(names)
        assert ("r_step0" in names) == (row["revision"] != "step0")
    written = pd.read_csv(res["models"]["pythia-160m"]["paths"]["csv"])
    assert {"p_pred_r_holm", "p_pred_absr_holm", "pred_r_r2", "pred_absr_r2", "pred_r_coefs"} <= set(written)
    with open(found["pythia-160m"]["revisions"]["step143000"]["helpful"]["heads"]) as f:
        heads = json.load(f)["heads"]
    scores = data["step143000"]["scores"]["helpful"]               # data: load_model_data of the Table-17 check above
    assert np.isfinite(scores).all()
    assert np.allclose(scores[[h["flat_idx"] for h in heads]], [h["auc"] for h in heads])

    md = open(res["models"]["pythia-160m"]["paths"]["md"]).read()
    for text in ("Primärbefund", "Tabelle 17", "Verlauf über das Training", "mean-field-artig", "keine** Aussage",
                 "Vorhersagbarkeit des Rests (g)", "V = vorhersagbar"):
        assert text in md

    # idempotent: unchanged inputs -> skipped; changed input -> recomputed, deterministic
    csv = res["models"]["pythia-160m"]["paths"]["csv"]
    mtime = os.path.getmtime(csv)
    res2 = mf.run_all(base, n_boot=200, n_perm=200, say=logs.append)
    assert all(r["skipped"] for r in res2["models"].values()) and os.path.getmtime(csv) == mtime
    res3 = mf.run_all(base, models=["pythia-160m"], n_boot=200, n_perm=200, force=True, say=logs.append)
    again = res3["models"]["pythia-160m"]["df"]
    assert np.allclose(again["mean_r"], df["mean_r"]) and np.allclose(again["p_loc_n1"], df["p_loc_n1"], equal_nan=True)
    assert list(again["verdict"]) == list(df["verdict"])
    merged = mf.merge_summaries([res["combined"], csv])
    assert len(merged) == 36
