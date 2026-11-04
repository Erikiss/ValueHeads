"""ICL-circuit head sets (valueheads_pythia.icl_sets): offline tests on a tiny fake census, plus checks against the
real ICL repo (andrewgcodes/in-context-learning-circuit) when a checkout is present. No network access."""
import hashlib
import json
import math
import os

import pytest

from valueheads_pythia import icl_sets as I

ICL_REPO = os.environ.get(I.ENV_REPO) or "/home/user/andrewgcodes/in-context-learning-circuit"
HAVE_REPO = os.path.exists(os.path.join(ICL_REPO, I.CENSUS["qwen3-4b-base"].path))
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="ICL repo checkout not present")


def _row(L, H, dret=0.0, lo=-0.1, hi=0.1, dP=0.0, plo=-0.1, phi=0.1, dR=0.0):
    return {"L": L, "H": H, "dret": dret, "dret_lo": lo, "dret_hi": hi, "dR": dR, "dP": dP, "dP_lo": plo, "dP_hi": phi}


def _fake_rows():
    return [
        _row(2, 0, -0.2, -0.3, -0.1),                        # reader
        _row(2, 1, -0.2, -0.3, 0.0),                         # hi == 0 -> not a reader (strict rule)
        _row(2, 2, 0.2, 0.1, 0.3, dP=0.2, plo=0.1, phi=0.3),  # brake and P-brake
        _row(2, 3, 0.2, 0.0, 0.3),                           # lo == 0 -> not a brake
        _row(3, 0, dP=math.nan, plo=0.1, phi=0.3),           # NaN dP -> neither P-brake nor P-carrier
        _row(3, 1, dP=-0.2, plo=-0.3, phi=-0.1),             # P-carrier
        _row(3, 2, -0.5, -0.6, -0.4, dP=-0.2, plo=-0.3, phi=-0.1),  # reader and P-carrier
        _row(3, 3),
    ]


def _fake_spec(data, expected=None):
    return I.CensusSpec("fake", "fake/fake", "lbFAKE", "data/D188_heads_fake.json", hashlib.sha256(data).hexdigest(),
                        4, 4, 2, (2, 3), expected or {"readers": 2, "brakes": 1, "p_brakes": 1},
                        {"readers": -0.7, "brakes": 0.2, "p_brakes": 0.2})


def test_classify_census_exact_rule():
    roles = I.classify_census(_fake_rows())
    assert roles == {"readers": ((2, 0), (3, 2)), "brakes": ((2, 2),), "p_brakes": ((2, 2),),
                     "p_carriers": ((3, 1), (3, 2))}


def test_validate_rejects_bad_rows():
    with pytest.raises(ValueError):
        I.validate_census_rows([_row(5, 0)], 4, 4)                 # layer out of range
    with pytest.raises(ValueError):
        I.validate_census_rows([_row(2, 0), _row(2, 0)], 4, 4)     # duplicate head
    with pytest.raises(ValueError):
        I.validate_census_rows([_row(2, 0, lo=0.2, hi=0.1)], 4, 4)  # lo > hi
    with pytest.raises(ValueError):
        I.validate_census_rows([{"L": 2, "H": 0}], 4, 4)           # missing keys
    with pytest.raises(ValueError):
        I.validate_census_rows([dict(_row(2, 0), dret="x")], 4, 4)  # non-numeric
    rows = I.validate_census_rows([dict(_row(2, 0), dP=None)], 4, 4)
    assert math.isnan(rows[0]["dP"])


def test_census_sets_local_file_and_checks(tmp_path):
    data = json.dumps(_fake_rows()).encode()
    f = tmp_path / "census.json"
    f.write_bytes(data)
    res = I.census_sets(_fake_spec(data), source=str(f))
    assert res.pinned and res.screened_layers == (2, 3) and len(res.screened_heads) == 8
    assert res.roles["readers"] == ((2, 0), (3, 2))
    assert res.effects("dret")[(3, 2)] == -0.5
    sets = res.role_sets()
    assert sets["census_readers"].overlap_ready and sets["census_readers"].test_layers == (2, 3)
    assert sets["census_r_readers"].explicit == "none" and not sets["census_r_readers"].overlap_ready
    # wrong expected count -> the self-check refuses
    with pytest.raises(RuntimeError):
        I.census_sets(_fake_spec(data, {"readers": 3, "brakes": 1, "p_brakes": 1}), source=str(f))
    # bytes that do not match the pinned hash are refused unless verify=False
    other = tmp_path / "other.json"
    other.write_bytes(json.dumps(_fake_rows()[:-1]).encode())
    with pytest.raises(ValueError):
        I.census_sets(_fake_spec(data), source=str(other))
    res2 = I.census_sets(_fake_spec(data), source=str(other), verify=False)
    assert not res2.pinned and len(res2.screened_heads) == 7


def test_incomplete_screened_layer_is_refused(tmp_path):
    rows = [r for r in _fake_rows() if (r["L"], r["H"]) != (3, 3)]
    data = json.dumps(rows).encode()
    f = tmp_path / "census.json"
    f.write_bytes(data)
    with pytest.raises(ValueError):
        I.census_sets(_fake_spec(data), source=str(f))


def test_offline_resolution_and_cache(tmp_path, monkeypatch):
    monkeypatch.delenv(I.ENV_REPO, raising=False)
    data = json.dumps(_fake_rows()).encode()
    spec = _fake_spec(data)
    with pytest.raises(FileNotFoundError):
        I.census_sets(spec, cache_dir=str(tmp_path), allow_download=False)
    calls = []

    def fake_download(url, timeout):
        calls.append(url)
        return data
    monkeypatch.setattr(I, "_download", fake_download)
    res = I.census_sets(spec, cache_dir=str(tmp_path))
    assert calls == [spec.url] and spec.url.startswith(I.RAW_BASE) and I.ICL_COMMIT in spec.url
    assert res.origin == spec.url
    res2 = I.census_sets(spec, cache_dir=str(tmp_path), allow_download=False)   # served from the cache now
    assert len(calls) == 1 and res2.origin.startswith(str(tmp_path)) and res2.roles == res.roles
    # a download with the wrong hash is refused and not cached
    monkeypatch.setattr(I, "_download", lambda url, timeout: data + b" ")
    with pytest.raises(ValueError):
        I.census_sets(spec, cache_dir=str(tmp_path / "other"))
    assert not (tmp_path / "other").exists()


def test_resolve_and_kv_groups():
    assert I.resolve_icl_model("Qwen/Qwen3-4B-Base") == ("qwen3-4b-base", False)
    assert I.resolve_icl_model("Qwen/Qwen3-4B") == ("qwen3-4b-base", True)
    assert I.resolve_icl_model("qwen3-0.6b-base") == ("qwen3-0.6b-base", False)
    with pytest.raises(KeyError):
        I.resolve_icl_model("pythia-160m")
    assert I.kv_group_query_heads(24, 5) == ((24, 20), (24, 21), (24, 22), (24, 23))
    assert I.kv_group_query_heads(3, 7, n_heads=16, n_kv_heads=8) == ((3, 14), (3, 15))
    with pytest.raises(ValueError):
        I.kv_group_query_heads(3, 8)


@pytest.mark.parametrize("model", sorted(I.NAMED))
def test_named_constants_consistent(model):
    spec = I.CENSUS[model]
    for name, rs in I.NAMED[model].items():
        assert rs.role == name and rs.sources
        assert all(0 <= l < spec.n_layers and 0 <= h < spec.n_heads for l, h in rs.heads)
        assert list(rs.heads) == sorted(set(rs.heads))
        assert all(0 <= l < spec.n_layers for l in rs.layers + rs.screened_layers)
        if rs.explicit == "full" and isinstance(rs.n_reported, int):
            assert len(rs.heads) == rs.n_reported, name
        if rs.explicit == "partial" and isinstance(rs.n_reported, int):
            assert len(rs.heads) < rs.n_reported, name
        if rs.explicit == "full":
            assert set(l for l, _ in rs.heads) <= set(rs.test_layers), name
        assert rs.restrict(rs.heads) == sorted(h for h in rs.heads if h[0] in rs.test_layers)


def test_named_qwen3_4b_structure():
    n = I.NAMED["qwen3-4b-base"]
    assert n["query_heads"].heads == ((22, 10), (23, 26))
    assert len(I.NAMED_CIRCUIT_L20_27_QWEN3_4B) == 33
    assert {l for l, _ in I.NAMED_CIRCUIT_L20_27_QWEN3_4B} <= set(range(20, 28))
    # query heads and the readers / brakes the paper names in L20-27 are in the named set; screen additions are not
    assert set(n["query_heads"].heads) <= set(I.NAMED_CIRCUIT_L20_27_QWEN3_4B)
    assert {(22, 13), (24, 22), (22, 20), (24, 21), (24, 8), (22, 2), (22, 8)} <= set(I.NAMED_CIRCUIT_L20_27_QWEN3_4B)
    ext = set(n["screen_additions_L20_27"].heads)
    assert not ext & set(I.NAMED_CIRCUIT_L20_27_QWEN3_4B)
    assert ext == set(n["query_type_screen_L20_27"].heads) | set(n["readers_screen_L20_27"].heads) \
        | set(n["arrow_readers_screen_L20_27"].heads)
    assert 67 + len(ext) + len(n["screen_additions_L0_19"].heads) == 79      # main.tex:312
    for role in ("duplicate_token_heads", "pooling_heads", "writers"):
        assert n[role].explicit == "none" and not n[role].overlap_ready
    assert n["value_kv_groups"].heads == I.kv_group_query_heads(22, 5) + I.kv_group_query_heads(24, 5) \
        + I.kv_group_query_heads(29, 2)
    assert n["value_kv_groups"].unit == "kv_group"
    # pre-specified confirmatory family vs descriptive fragments (overlap.py: only `confirmatory` roles get a verdict)
    assert set(n["query_type_heads_L20_27"].heads) == set(n["query_heads"].heads) | set(n["query_type_screen_L20_27"].heads)
    assert n["query_type_heads_L20_27"].test_layers == tuple(range(20, 28))
    primary = {r for r, rs in n.items() if rs.confirmatory}
    assert primary == {"query_type_heads_L20_27", "screen_additions_L0_19", "named_circuit_L20_27", "value_kv_groups"}
    assert {r for r, rs in n.items() if rs.overlap_ready and rs.family == "descriptive"} == {
        "query_heads", "query_type_screen_L20_27", "readers_screen_L20_27", "arrow_readers_screen_L20_27",
        "screen_additions_L20_27", "shared_top5_readers_4b_8b", "shared_top5_brakes_4b_8b"}
    assert not I.NAMED["qwen3-0.6b-base"]["shared_top2_brakes_0.6b_1.7b"].confirmatory
    with pytest.raises(ValueError):
        I.RoleSet("x", ((0, 0),), "full", 1, (0,), (0,), ("t",), family="maybe")
    sets = I.icl_role_sets("Qwen/Qwen3-4B", census=False)
    assert "census_readers" not in sets and set(n) <= set(sets)
    table = I.role_table(sets)
    assert {r["role"] for r in table} == set(sets)
    assert next(r for r in table if r["role"] == "value_kv_groups")["layers"] == "22,24,29"


def test_parse_d174_screen():
    text = ("256 heads screened; thr -0.03\n"
            "mean_both  [CI]           rare    sym   head   named  att_rep att_fresh n\n"
            "-0.114 [-0.158,-0.071] -0.168 -0.061  23:26  N     0.048  0.005  144\n"
            "-0.076 [-0.105,-0.046] -0.097 -0.054  20:5   -     0.060  0.001  144\n"
            "-0.037 [-0.060,-0.015] -0.036 -0.038   8:23  -     nan  nan  144\n"
            "\nEXT (2 heads, CI<0 and mean<-0.03): 20:5,8:23\n")
    rows, ext = I.parse_d174_screen(text)
    assert [(r["L"], r["H"], r["named"]) for r in rows] == [(23, 26, True), (20, 5, False), (8, 23, False)]
    assert rows[0]["lo"] == -0.158 and rows[0]["hi"] == -0.071
    assert ext == ((20, 5), (8, 23))


# ---------------------------------------------------------------------------------------------------------------
# against the real ICL repo (local checkout only)
# ---------------------------------------------------------------------------------------------------------------

@needs_repo
@pytest.mark.parametrize("model", sorted(I.CENSUS))
def test_real_census_matches_summary(model):
    res = I.census_sets(model, source=ICL_REPO)            # verify=True: pinned sha256 + D188_summary self-check
    spec = I.CENSUS[model]
    summ = {d["name"]: d for d in json.load(open(os.path.join(ICL_REPO, "data", "D188_summary.json")))}
    s = summ[f"{os.path.basename(spec.path)[len('D188_heads_'):-len('.json')]}"]
    assert len(res.roles["readers"]) == s["n_readers"]
    assert len(res.roles["brakes"]) == s["n_brakes"]
    assert len(res.roles["p_brakes"]) == s["n_Pbrakes"]
    eff = res.effects("dret")
    assert math.isclose(sum(eff[h] for h in res.roles["readers"]), s["sum_readers"], rel_tol=1e-12)
    assert math.isclose(sum(eff[h] for h in res.roles["brakes"]), s["sum_brakes"], rel_tol=1e-12)
    assert res.screened_layers == spec.screened_layers
    assert len(res.screened_heads) == len(spec.screened_layers) * spec.n_heads


@needs_repo
def test_real_counts_4b_and_06b():
    r4 = I.census_sets("qwen3-4b-base", source=ICL_REPO)
    assert (len(r4.roles["readers"]), len(r4.roles["brakes"]), len(r4.roles["p_brakes"])) == (89, 56, 54)
    assert r4.screened_layers == tuple(range(16, 36)) and len(r4.screened_heads) == 640
    r06 = I.census_sets("qwen3-0.6b-base", source=ICL_REPO)
    assert (len(r06.roles["readers"]), len(r06.roles["brakes"]), len(r06.roles["p_brakes"])) == (70, 13, 23)
    assert r06.screened_layers == tuple(range(10, 28)) and len(r06.screened_heads) == 288
    # paper statements (main.tex:462) hold on the recomputed census
    n4 = I.NAMED["qwen3-4b-base"]
    eff4 = r4.effects("dret")
    top_readers = sorted(r4.roles["readers"], key=lambda h: eff4[h])
    top_brakes = sorted(r4.roles["brakes"], key=lambda h: -eff4[h])
    assert set(n4["shared_top5_readers_4b_8b"].heads) <= set(top_readers[:6])   # 4B ranks 21:11 5th
    assert set(n4["shared_top5_brakes_4b_8b"].heads) <= set(top_brakes[:6])     # 4B ranks 21:31 4th
    top2 = sorted(r06.roles["brakes"], key=lambda h: -r06.effects()[h])[:2]
    assert tuple(sorted(top2)) == I.NAMED["qwen3-0.6b-base"]["shared_top2_brakes_0.6b_1.7b"].heads
    sets = I.icl_role_sets("qwen3-4b-base", source=ICL_REPO)
    assert sets["census_readers"].heads == r4.roles["readers"] and sets["census_readers"].n_reported == 89


@needs_repo
def test_real_d174_screens_match_constants():
    with open(os.path.join(ICL_REPO, "data", "D174_screen_full.txt")) as f:
        rows, ext = I.parse_d174_screen(f.read())
    assert len(rows) == 256 and {r["L"] for r in rows} == set(range(20, 28))
    named = tuple(sorted((r["L"], r["H"]) for r in rows if r["named"]))
    assert named == I.NAMED_CIRCUIT_L20_27_QWEN3_4B                     # "the 33 named heads in the band"
    assert sum(r["named"] and r["mean"] < -0.02 for r in rows) == 11    # main.tex:738 "11 cost more than 0.02"
    assert tuple(sorted(ext)) == I.NAMED["qwen3-4b-base"]["screen_additions_L20_27"].heads
    with open(os.path.join(ICL_REPO, "data", "D174_screen_L0-19_full.txt")) as f:
        rows0, ext0 = I.parse_d174_screen(f.read())
    assert len(rows0) == 640 and not any(r["named"] for r in rows0)   # no named-set member in L0-19 in that run
    assert ext0 == ((8, 23),)
    ranked = [(r["L"], r["H"]) for r in sorted(rows0, key=lambda r: r["mean"])]
    assert set(I.NAMED["qwen3-4b-base"]["screen_additions_L0_19"].heads) == {ranked[0], *ranked[2:5]}


@needs_repo
def test_real_gate_logits_heads():
    with open(os.path.join(ICL_REPO, "data", "D66b_gate_logits.json")) as f:
        g = json.load(f)
    gated = {tuple(int(x) for x in k.split("|")[1].split(":")) for k in g if k.startswith("radd23x1.0_q|")
             and g[k][0] < 0}
    assert gated == set(I.NAMED["qwen3-4b-base"]["gate_released_readers"].heads)
