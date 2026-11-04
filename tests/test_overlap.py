"""Overlap analysis value heads x ICL roles (valueheads_pythia.overlap, `overlap` command): null construction, exact vs
Monte-Carlo p-values, Holm, layer-stratified Spearman, verdict rules and the CLI on synthetic head tables. Offline except
for the optional checks against a local checkout of the ICL repo."""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom, spearmanr

from valueheads_pythia import cli, config, icl_sets as I, overlap as O
from valueheads_pythia.probing import _finish_table, save_head_scores

ICL_REPO = os.environ.get(I.ENV_REPO) or "/home/user/andrewgcodes/in-context-learning-circuit"
HAVE_REPO = os.path.exists(os.path.join(ICL_REPO, I.CENSUS["qwen3-0.6b-base"].path))
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="ICL repo checkout not present")


def _table(value, topk, L, H, scores=None, ranking="pythia"):
    scores = np.random.default_rng(0).random(L * H) if scores is None else scores
    return O.ValueTable(value, ranking, O.RANKINGS[ranking]["score"], L, H, tuple(topk), np.asarray(scores, float),
                        f"/x/{value}.json", 5000, 1000, {})


def _role(name, heads, layers, **kw):
    return I.RoleSet(name, tuple(sorted(heads)), "full", len(heads), tuple(layers), tuple(layers), ("test",), **kw)


# ------------------------------------------------------------------------------------------------- null construction

def test_layer_matched_draws_keep_per_layer_counts():
    topk = [(0, 1), (0, 5), (2, 3), (3, 0), (3, 1), (3, 2)]
    M = O.layer_matched_draws(topk, 4, 8, 500, np.random.default_rng(0))
    per_layer = M.reshape(500, 4, 8).sum(axis=2)
    assert (per_layer == np.array([2, 0, 1, 3])).all()
    # every head of a layer is equally likely: hit rate n_l / H
    rate = M.reshape(500, 4, 8).mean(axis=0)
    assert abs(rate[3].mean() - 3 / 8) < 1e-9 and rate[3].min() > 0.25 and rate[3].max() < 0.5


def test_kv_null_keeps_group_clustering():
    # three value heads in KV group 1 (heads 4-7) and one in group 0 of an 8-head layer with 2 heads per group... use 4
    topk = [(1, 4), (1, 5), (1, 6), (1, 0)]
    M = O.layer_matched_draws(topk, 2, 8, 400, np.random.default_rng(1), heads_per_kv=4).reshape(400, 2, 8)
    assert (M[:, 0].sum(1) == 0).all() and (M[:, 1].sum(1) == 4).all()
    groups = M[:, 1].reshape(400, 2, 4).sum(axis=2)
    assert all(sorted(g) == [1, 3] for g in groups.tolist())         # the (3, 1) clustering survives every draw
    assert (groups[:, 0] == 3).any() and (groups[:, 1] == 3).any()   # but its location is random
    with pytest.raises(ValueError):
        O.layer_matched_draws([(0, 0)], 1, 6, 10, np.random.default_rng(0), heads_per_kv=4)


def test_mc_null_matches_exact_hypergeometric():
    L, H = 3, 8
    topk = [(0, 0), (0, 1), (0, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 7)]
    role = [(0, 0), (0, 5), (1, 3), (1, 4), (2, 6)]
    hist = O.null_histograms(topk, [[l * H + h for l, h in role]], L, H, 40000, np.random.default_rng(2))[0]
    pmf = O.exact_overlap_pmf([(3, 2), (1, 2), (4, 1)], H)
    assert abs(pmf.sum() - 1) < 1e-12
    assert np.allclose(hist[:len(pmf)] / hist.sum(), pmf, atol=0.01)
    # single-layer case equals scipy's hypergeometric
    assert np.allclose(O.exact_overlap_pmf([(4, 3)], 10), hypergeom.pmf(np.arange(4), 10, 3, 4))


def test_kv_layer_support_matches_the_null():
    rng = np.random.default_rng(4)
    for _ in range(25):
        H, s = 16, 4
        V = sorted(rng.choice(H, int(rng.integers(1, 9)), replace=False).tolist())
        R = sorted(rng.choice(H, int(rng.integers(1, 9)), replace=False).tolist())
        M = O.layer_matched_draws([(0, h) for h in V], 1, H, 20000, np.random.default_rng(0), heads_per_kv=s)
        o = M[:, R].sum(axis=1)
        assert (int(o.min()), int(o.max())) == O.kv_layer_support(V, R, H, s), (V, R)
        assert O.layer_support(len(V), len(R), H) == (max(0, len(V) + len(R) - H), min(len(V), len(R)))


def test_min_attainable_p_and_tarone_holm():
    hist = np.array([10, 80, 9, 1])                          # O* in 0..3, N = 100
    assert O.min_attainable_p(hist, 0, 3) == pytest.approx(2 * 2 / 101)
    assert O.min_attainable_p(hist, 1, 2) == pytest.approx(min(1, 2 * min(11, 91) / 101))
    assert math.isnan(O.min_attainable_p(hist, 2, 2))
    p = np.array([0.001, 0.02, 0.6, 0.9, 0.4, np.nan])
    p_min = np.array([0.001, 0.001, 0.5, 0.5, 0.3, 0.001])
    adj, fam, K, m = O.tarone_holm(p, p_min, 0.05)
    # k = 1: two tests with p_min <= 0.05 > 1; k = 2: two with p_min <= 0.025 <= 2 -> K = 2, family = the first two
    assert K == 2 and m == 5 and fam.tolist() == [True, True, False, False, False, False]
    assert np.allclose(adj[:2], [0.002, 0.02]) and np.isnan(adj[2:]).all()
    # without dead tests Tarone is plain Holm
    adj2, fam2, K2, _ = O.tarone_holm([0.01, 0.04, 0.03], [1e-6] * 3, 0.05)
    assert fam2.all() and K2 == 3 and np.allclose(adj2, O.holm([0.01, 0.04, 0.03]))


def test_cutoff_ties():
    s = np.array([0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.1])
    assert O.cutoff_ties(s, [0, 1, 2]) == {"ties_at_cutoff": 4, "ties_in_topk": 1}
    assert O.cutoff_ties(s, [0, 1]) == {"ties_at_cutoff": 1, "ties_in_topk": 1}
    # stable ranking: bal_acc breaks AUC ties before the index
    bal = np.array([0.9, 0.8, 0.70, 0.70, 0.65, 0.60, 0.1])
    assert O.cutoff_ties(s, [0, 1, 2], secondary=bal) == {"ties_at_cutoff": 2, "ties_in_topk": 1}


def test_holm_known_values():
    adj = O.holm([0.01, 0.04, 0.03, np.nan, 0.005])
    assert np.allclose(adj[[4, 0, 2, 1]], [0.02, 0.03, 0.06, 0.06]) and np.isnan(adj[3])
    assert np.allclose(O.holm([0.5, 0.9]), [1.0, 1.0])


def test_stratified_spearman_rho_and_null():
    rng = np.random.default_rng(3)
    layers = np.repeat(np.arange(6), 20)
    y = rng.normal(size=120)
    x = 0.8 * y + rng.normal(size=120)
    rho, null = O.stratified_spearman(x, y, layers, 2000, np.random.default_rng(0))
    assert math.isclose(rho, spearmanr(x, y)[0], abs_tol=1e-12)
    assert (null < rho).all()
    # association only between layers (layer means shared, within-layer noise independent): the null keeps it
    lm = rng.normal(size=6)
    x2 = lm[layers] * 3 + rng.normal(size=120) * 0.1
    y2 = lm[layers] * 3 + rng.normal(size=120) * 0.1
    rho2, null2 = O.stratified_spearman(x2, y2, layers, 2000, np.random.default_rng(0))
    assert rho2 > 0.7 and abs(np.mean(null2) - rho2) < 0.1
    p_ge = (1 + (null2 >= rho2).sum()) / 2001
    assert p_ge > 0.01


# ------------------------------------------------------------------------------------------------- analyze

def _fake_census(tmp_path, L=4, H=8):
    rows = []
    for l in (2, 3):
        for h in range(H):
            reader = h < 4
            rows.append({"L": l, "H": h, "dret": -0.3 if reader else 0.0, "dret_lo": -0.4 if reader else -0.1,
                         "dret_hi": -0.2 if reader else 0.1, "dR": -0.1 * reader, "dP": 0.01 * h, "dP_lo": -0.1, "dP_hi": 0.1})
    data = json.dumps(rows).encode()
    path = tmp_path / "census.json"
    path.write_bytes(data)
    import hashlib
    spec = I.CensusSpec("fake", "fake/fake", "lbFAKE", "data/x.json", hashlib.sha256(data).hexdigest(), L, H, 2, (2, 3),
                        {}, {})
    return I.census_sets(spec, source=str(path), verify=False)


def test_analyze_planted_overlap_and_degenerate(tmp_path):
    L, H = 4, 8
    census = _fake_census(tmp_path)
    roles = census.role_sets()
    roles["empty_band"] = _role("empty_band", [(0, 1)], (0,))           # no value head in layer 0 -> degenerate
    roles["partial_x"] = I.RoleSet("partial_x", ((2, 0),), "partial", 3, (2, 3), (), ("t",))
    readers = set(census.roles["readers"])
    assert len(readers) == 8
    planted = sorted(readers) + [(1, 0), (1, 1)]                      # all 8 readers + 2 heads outside the band
    other = [(2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7), (1, 0), (1, 1)]   # avoids the readers
    s = np.zeros(L * H)
    for l, h in readers:
        s[l * H + h] = 1.0
    tables = [_table("planted", planted, L, H, s), _table("avoid", other, L, H, -s)]
    res = O.analyze(tables, roles, census, draws=10000, perms=2000, seed=0, heads_per_kv=1, verbose=False)
    tk = res["topk"].set_index(["value", "role"])
    r = tk.loc[("planted", "census_readers")]
    assert r.overlap == 8 and r.k_in_band == 8 and r.role_n == 8 and math.isclose(r.null_mean, 4.0, abs_tol=0.1)
    assert r.null == "layer" and r.p_enr == r.p_enr_layer and "p_enr_kv" not in tk
    assert r.p_enr < 1e-3 and r.p_enr_exact < 1e-3 and r.verdict == "angereichert"
    assert r.p_enr_jaccard == r.p_enr and r.jaccard == 1.0
    a = tk.loc[("avoid", "census_readers")]
    assert a.overlap == 0 and a.p_dep < 1e-3 and a.verdict == "abgereichert"
    d = tk.loc[("planted", "empty_band")]
    assert bool(d.degenerate) and math.isnan(d.p_two) and d.verdict.startswith("kein Test")
    fam = res["families"]["a"]["pythia"]
    # tested: census_readers + empty_band (the fake census has no brakes / P-roles: empty lists are not testable)
    assert fam["roles"] == ["census_readers", "empty_band"] and fam["m"] == 2 and fam["tarone_K"] == 2
    assert sorted(fam["excluded_degenerate"]) == ["avoid/empty_band", "planted/empty_band"]
    # GQA (4 heads per KV group): the 8 readers are KV group 0 (heads 0-3) of both layers and the value heads fill
    # exactly these groups. The KV null (primary under GQA) keeps whole groups together, so the overlap is 0 or 4 per
    # layer and O = 8 has probability 1/4: a head-level enrichment that is only a group-level coincidence is no finding;
    # its smallest attainable p (2 * 1/4) is above alpha, so Tarone drops it from the family.
    res_kv = O.analyze(tables, roles, census, draws=10000, perms=0, seed=0, heads_per_kv=4, verbose=False)
    rk = res_kv["topk"].set_index(["value", "role"]).loc[("planted", "census_readers")]
    assert rk.null == "kv" and rk.p_enr == rk.p_enr_kv
    assert rk.p_enr_kv == pytest.approx(0.25, abs=0.02) and rk.null_mean_kv == pytest.approx(4.0, abs=0.15)
    assert rk.p_enr_layer < 1e-3 and rk.p_min == pytest.approx(0.5, abs=0.05)
    assert rk.verdict.startswith("nicht testbar") and not bool(rk.in_family)
    assert not res_kv["topk"]["verdict"].isin(O.VERDICTS_SIGNIFICANT).any()
    assert res_kv["families"]["a"]["pythia"]["null"] == "kv"
    assert "census_brakes" in res["skipped_roles"]
    assert "partial_x" in res["skipped_roles"] and "census_r_readers" in res["skipped_roles"]
    assert "partial_x" not in fam["roles"]
    sh = res["shares"].set_index(["value", "role"])
    assert bool(sh.loc[("planted", "partial_x")].lower_bound) and sh.loc[("planted", "census_readers")].share_band == 1.0
    sp = res["spearman"].set_index(["value", "effect"])
    assert sp.loc[("planted", "dret")].rho < -0.8 and sp.loc[("planted", "abs_dret")].rho > 0.8
    md = O.markdown_summary(res, {"model": "fake", "rankings": ["pythia"], "k": 10, "icl": {"model": "fake"},
                                  "tables": [{"value": "planted", "ranking": "pythia", "score": "auc", "path": "/x/a.json",
                                              "n_train": 50, "n_val": 20, "k": 10}]})
    assert "angereichert" in md and "Kleine Stichproben" in md and "Holm-Familie" in md
    width = None                                                     # no stray pipes inside table cells
    for line in md.splitlines():
        if line.startswith("|"):
            width = width or line.count("|")
            assert line.count("|") == width, line
        else:
            width = None


def test_analyze_random_value_heads_give_no_verdict(tmp_path):
    L, H = 4, 8
    census = _fake_census(tmp_path)
    rng = np.random.default_rng(5)
    tables = []
    for i in range(6):
        idx = rng.choice(L * H, 6, replace=False)
        tables.append(_table(f"v{i}", [(int(j) // H, int(j) % H) for j in idx], L, H))
    res = O.analyze(tables, census.role_sets(), census, draws=10000, perms=500, seed=1, verbose=False)
    assert not res["topk"]["verdict"].isin(["angereichert", "abgereichert"]).any()
    tk = res["topk"][res["topk"]["p_holm"].notna()]
    assert (tk["p_holm"] >= tk["p_two"]).all()


def _kv_group_table(name, groups, L, H, s=4, ranking="pythia"):
    return _table(name, sorted((l, h) for l, g in groups for h in range(g * s, (g + 1) * s)), L, H, ranking=ranking)


def test_kv_clustered_value_sets_give_no_false_verdict():
    """Planted KV-clustered value sets (whole KV groups) that coincide with group-clustered roles by chance: the old
    head-exchangeable null calls them enriched, the KV-preserving primary null (and the group-level test of
    value_kv_groups) does not."""
    L, H, s = 36, 32, 4
    n = I.NAMED["qwen3-4b-base"]
    clustered = _role("clustered_role", [(l, h) for l in range(16, 30) for h in range(0, 4)], range(16, 36))   # group 0
    roles = {"value_kv_groups": n["value_kv_groups"], "clustered_role": clustered}
    assert n["value_kv_groups"].unit == "kv_group"
    # value set A: whole groups 22.5 and 24.5 (two of the three value KV groups) + 29.0, plus groups outside 21-29
    a = _kv_group_table("A", [(22, 5), (22, 1), (24, 5), (29, 0), (10, 3), (12, 2)], L, H)
    # value set B: group 0 in layers 16-19 (4 of the 14 role groups) and group 1 in layers 20-35
    b = _kv_group_table("B", [(l, 0) for l in range(16, 20)] + [(l, 1) for l in range(20, 36)], L, H)
    old = O.analyze([a, b], roles, None, draws=20000, perms=0, seed=0, heads_per_kv=s, kv_null=False, verbose=False)
    new = O.analyze([a, b], roles, None, draws=20000, perms=0, seed=0, heads_per_kv=s, verbose=False)
    to, tn = old["topk"].set_index(["value", "role"]), new["topk"].set_index(["value", "role"])
    assert to.loc[("B", "clustered_role")].verdict == "angereichert"          # false positive of the exchangeable null
    assert tn.loc[("B", "clustered_role")].null == "kv" and tn.loc[("B", "clustered_role")].p_two > 0.05
    g = tn.loc[("A", "value_kv_groups")]
    assert g.null == "kv_group" and g.unit == "KV-Gruppe" and g.overlap == 2 and g.overlap_heads_n == 8
    assert g.k_in_band == 4 and g.role_n == 3 and g.p_two_layer < 1e-3     # head level: astronomically "significant"
    # group level: P(O >= 2) for independent hits with 2/8, 1/8, 1/8 = 0.0547 + 0.0273 + ... (exact pmf)
    pmf = O.exact_overlap_pmf([(2, 1), (1, 1), (1, 1)], 8)
    assert g.p_enr_exact == pytest.approx(pmf[2:].sum()) and g.p_enr == pytest.approx(pmf[2:].sum(), abs=0.01)
    assert not new["topk"]["verdict"].isin(O.VERDICTS_SIGNIFICANT).any()
    with pytest.raises(ValueError):                                          # kv_group roles must be whole groups
        O.analyze([a], {"x": _role("x", [(22, 20), (22, 21)], (22,), unit="kv_group")}, None, draws=10, perms=0,
                  heads_per_kv=s, verbose=False)


def test_kv_null_is_calibrated_for_clustered_value_sets():
    """Type-I rate at 0.05 over random whole-KV-group value sets placed independently of a group-clustered role:
    the KV-preserving null holds the level, the head-exchangeable null does not (review: 0.29-0.36 vs <= 0.02)."""
    L, H, s = 12, 16, 4
    rng = np.random.default_rng(11)
    role = _role("clustered", [(l, h) for l, g in [(4, 0), (5, 1), (6, 2), (7, 0), (8, 3), (9, 1), (10, 0), (11, 2)]
                               for h in range(g * s, (g + 1) * s)], range(4, 12))
    tables = []
    for i in range(150):
        cells = set()
        while len(cells) < 6:
            cells.add((int(rng.integers(3, 12)), int(rng.integers(0, H // s))))
        tables.append(_kv_group_table(f"v{i}", sorted(cells), L, H))
    res = O.analyze(tables, {"clustered": role}, None, draws=2000, perms=0, seed=3, heads_per_kv=s, verbose=False)
    tk = res["topk"][~res["topk"]["degenerate"]]
    rate_kv, rate_layer = float((tk.p_two < 0.05).mean()), float((tk.p_two_layer < 0.05).mean())
    assert rate_kv <= 0.08, rate_kv
    assert rate_layer >= 0.15, rate_layer


def test_descriptive_roles_get_no_verdict(tmp_path):
    L, H = 4, 8
    census = _fake_census(tmp_path)
    roles = census.role_sets()
    readers = sorted(census.roles["readers"])
    roles["desc"] = _role("desc", readers, (2, 3), family="descriptive")
    s = np.zeros(L * H)
    res = O.analyze([_table("planted", readers + [(1, 0), (1, 1)], L, H, s)], roles, census, draws=10000, perms=0,
                    verbose=False)
    tk = res["topk"].set_index("role")
    assert tk.loc["census_readers"].verdict == "angereichert"
    assert tk.loc["desc"].verdict.startswith("deskriptiv") and math.isnan(tk.loc["desc"].p_holm)
    assert tk.loc["desc"].p_enr < 1e-3                                       # p reported, never a verdict
    fam = res["families"]["a"]["pythia"]
    assert "desc" not in fam["roles"] and fam["descriptive_roles"] == ["desc"] and fam["m"] == 1


def test_analyze_rejects_geometry_mismatch(tmp_path):
    census = _fake_census(tmp_path)
    with pytest.raises(ValueError):
        O.analyze([_table("v", [(0, 0)], 5, 8)], census.role_sets(), census, draws=10, perms=0, n_layers=4, n_heads=8,
                  verbose=False)


# ------------------------------------------------------------------------------------------------- CLI

def _write_table(root, model, value, mode, L, H, scores, tag=None):
    rows = [{"layer": l, "head": h, "flat_idx": l * H + h, "auc": float(scores[l * H + h]), "acc": float(scores[l * H + h]),
             "bal_acc": 0.5, "majority": 0.5, "pos_rate": 0.5, "n_train": 2000, "n_val": 500} for l in range(L) for h in range(H)]
    score = "auc" if mode == "pythia" else "acc"
    ranking = "stable" if mode == "pythia" else "paper"
    df = _finish_table(rows, {"layers": L, "heads": H, "score": score, "ranking": ranking, "feature_mode": "head"}, score, ranking)
    path = config.heads_path(config.paths(str(root)), model, None, value, mode, tag)
    save_head_scores(df, path, meta={"model": model, "mode": mode})
    return path


@needs_repo
def test_overlap_cli_end_to_end(tmp_path, capsys):
    spec = I.CENSUS["qwen3-0.6b-base"]
    L, H = spec.n_layers, spec.n_heads
    census = I.census_sets("qwen3-0.6b-base", source=ICL_REPO)
    s = np.random.default_rng(0).random(L * H) * 0.1
    for l, h in census.roles["readers"]:
        s[l * H + h] += 1.0                                          # value heads = census readers
    for model in ("qwen3-0.6b-base", "qwen3-0.6b"):
        _write_table(tmp_path, model, "honesty", "pythia", L, H, s)
        _write_table(tmp_path, model, "honesty", "paper", L, H, s, tag="pyfeat")
    args = ["overlap", "--root", str(tmp_path), "--values", "honesty", "--paper_tag", "pyfeat", "--draws", "10000",
            "--perms", "500", "--icl_source", ICL_REPO, "--no_download", "--quiet"]
    assert cli.main(args + ["--model", "Qwen/Qwen3-0.6B-Base"]) == 0
    base = os.path.join(tmp_path, "results", "overlap", "qwen3-0.6b-base_main_overlap")
    for ext in (".json", ".md", "_topk.csv", "_spearman.csv", "_shares.csv"):
        assert os.path.exists(base + ext)
    payload = json.load(open(base + ".json"))
    assert payload["info"]["icl"]["transferred"] is False and payload["info"]["icl"]["pinned"] is True
    assert payload["info"]["k"] == config.MODELS["qwen3-0.6b-base"]["k15"] == 67
    tk = pd.read_csv(base + "_topk.csv")
    r = tk[(tk.role == "census_readers")]
    assert set(r.ranking) == {"pythia", "paper"} and (r.verdict == "angereichert").all()
    assert set(tk.role) == {"census_readers", "census_brakes", "census_p_brakes", "census_p_carriers",
                            "shared_top2_brakes_0.6b_1.7b"}
    assert (tk[tk.role == "shared_top2_brakes_0.6b_1.7b"].verdict.str.startswith("deskriptiv")).all()
    assert set(tk.null) == {"kv"} and payload["families"]["a"]["pythia"]["null"] == "kv"
    out = capsys.readouterr().out
    assert cli.main(args + ["--model", "Qwen/Qwen3-0.6B-Base"]) == 0          # idempotent
    assert "skip" in capsys.readouterr().out and "skip" not in out
    assert cli.main(args + ["--model", "qwen3-0.6b"]) == 0                     # post-trained -> base sets, transferred
    p2 = json.load(open(os.path.join(tmp_path, "results", "overlap", "qwen3-0.6b_main_overlap.json")))
    assert p2["info"]["icl"]["transferred"] is True and p2["info"]["icl"]["model"] == "qwen3-0.6b-base"
    assert "Übertragen" in open(os.path.join(tmp_path, "results", "overlap", "qwen3-0.6b_main_overlap.md")).read()


@needs_repo
def test_overlap_cli_paper_harmless_is_the_pooled_set(tmp_path):
    spec = I.CENSUS["qwen3-0.6b-base"]
    L, H = spec.n_layers, spec.n_heads
    rng = np.random.default_rng(1)
    _write_table(tmp_path, "qwen3-0.6b-base", "harmless", "pythia", L, H, rng.random(L * H))
    _write_table(tmp_path, "qwen3-0.6b-base", "harmless", "paper", L, H, rng.random(L * H), tag="pyfeat")
    _write_table(tmp_path, "qwen3-0.6b-base", "harmless_pooled", "paper", L, H, rng.random(L * H))
    args = ["overlap", "--root", str(tmp_path), "--model", "qwen3-0.6b-base", "--values", "harmless", "--paper_tag", "pyfeat",
            "--draws", "10000", "--perms", "0", "--icl_source", ICL_REPO, "--no_download", "--quiet"]
    assert cli.main(args) == 0
    payload = json.load(open(os.path.join(tmp_path, "results", "overlap", "qwen3-0.6b-base_main_overlap.json")))
    got = {(t["ranking"], t["value"], os.path.basename(t["path"])) for t in payload["info"]["tables"]}
    assert got == {("pythia", "harmless", "qwen3-0.6b-base_main_harmless_pythia.json"),
                   ("paper", "harmless_pooled", "qwen3-0.6b-base_main_harmless_pooled_paper.json")}
    assert payload["info"]["paper_harmless"] == "harmless_pooled"
    assert cli.main(args + ["--paper_harmless", "harmless", "--out_tag", "base"]) == 0
    p2 = json.load(open(os.path.join(tmp_path, "results", "overlap", "qwen3-0.6b-base_main_base_overlap.json")))
    assert {(t["ranking"], t["value"]) for t in p2["info"]["tables"]} == {("pythia", "harmless"), ("paper", "harmless")}
    from valueheads_pythia.commands.overlap import ranking_values
    assert ranking_values(["harmless", "harmless_pooled", "honesty"], "pythia") == ["harmless", "honesty"]
    assert ranking_values(["harmless", "harmless_pooled", "honesty"], "paper") == ["harmless_pooled", "honesty"]


def test_overlap_cli_guards(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["overlap", "--root", str(tmp_path), "--model", "qwen3-0.6b-base", "--draws", "500"])   # < 10,000 draws
    with pytest.raises(SystemExit):
        cli.main(["overlap", "--root", str(tmp_path), "--model", "pythia-160m"])                      # no ICL sets for Pythia
    assert cli.main(["overlap", "--root", str(tmp_path), "--model", "qwen3-0.6b-base", "--no_download"]) == 1   # no tables
    assert "overlap" in cli.COMMANDS


# ------------------------------------------------------------------------------------------------- run notebook

def test_overlap_notebook_is_generated_reproducibly(tmp_path):
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(root, "scripts", "make_overlap_notebook.py"), "--out", str(tmp_path)],
                   check=True, capture_output=True)
    fresh = (tmp_path / "qwen3-4b-overlap.ipynb").read_text()
    committed = os.path.join(root, "notebooks", "runs", "qwen3-4b-overlap.ipynb")
    assert fresh == open(committed).read(), "notebooks/runs/qwen3-4b-overlap.ipynb is stale: run scripts/make_overlap_notebook.py"
    nb = json.loads(fresh)
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "nb", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert '"Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Base"' in code and 'MyDrive/ValueHeadsPythia/qwen3-4b"' in code
    assert all(f'"{v}"' in code for v in ["honesty", "harmless", "helpful", "helpfulness", "complexity", "verbosity", "coherence"])
    assert '"harmless_pooled"' in code and '"--mode", "paper", "--dataset"' in code      # reference harmless set, paper ranking
    assert 'DRIVE_PROJECT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"' in code
    assert 'run("overlap"' not in code and '"overlap", "--model"' in code and '"probe-stats"' in code
    for banned in ('"eval-likelihood"', '"eval-generate"', 'run("judge"', '"judge", "--'):   # no likelihood evals / judges
        assert banned not in code
    assert "def run(" in code and "HF_TOKEN" in code and "git" in code and nb["metadata"]["colab"]["gpuType"] == "A100"
