"""Part B of the causal ablation test (docs/ABLATION_DESIGN.md 6.7): HH split, named / random / frozen head sets
(valueheads_pythia.ablation_sets) and the statistics + German verdict texts (valueheads_pythia.ablation_stats).

Offline: synthetic ASCII items and synthetic head tables only, no model, no network. The `needs_repo` tests read the
local checkout of the ICL repo (census files) and are skipped without it. The hook exactness checks (own-reference
mean ablation == clean logits, empty set == no hook) belong to part A (tests/test_ablation_hooks.py, 5.7/5.8)."""
import json
import math
import os

import numpy as np
import pytest

from valueheads_pythia import ablation_sets as S
from valueheads_pythia import ablation_stats as T
from valueheads_pythia import icl_sets as I
from valueheads_pythia import overlap as O
from valueheads_pythia.probing import _finish_table, save_head_scores
from valueheads_pythia.probing_stats import WeightedAUC, auc_matrix
from valueheads_pythia.stats import paired_delta_ci

ICL_REPO = os.environ.get(I.ENV_REPO) or "/home/user/andrewgcodes/in-context-learning-circuit"
HAVE_REPO = os.path.exists(os.path.join(ICL_REPO, I.CENSUS["qwen3-4b-base"].path))
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="ICL repo checkout not present")


def _hs(text):
    return [tuple(int(x) for x in t.split(":")) for t in text.split()]


# ================================================================================================= 1. HH split

def _item(g, ctx, pos, neg):
    return {"group": g, "context": ctx, "pos": [pos], "neg": [neg], "meta": {"value": "helpful"}}


def _items():
    """Synthetic HH-style items: 80 ids, some prompts shared (also with different whitespace), planted drops."""
    items = []
    for g in range(80):
        p = g if g < 60 else g - 30                       # ids 60-79 repeat the prompts of ids 30-49 (2-item clusters)
        ctx = f"\n\nHuman: question number {p} about things\n\nAssistant:"
        if g in (61, 63):
            ctx = ctx.replace(" about", "   about")       # same prompt key after whitespace normalisation
        items.append(_item(g, ctx, f" answer A{g}", f" answer B{g}"))
    items[20]["pos"] = ["   "]                            # empty_response
    items[21]["neg"] = [""]                               # empty_response
    items[22]["neg"] = list(items[22]["pos"])             # identical
    items[23]["pos"] = [" "]                              # empty, but its prompt is in probe-train -> first reason wins
    items[24]["neg"] = list(items[24]["pos"])             # identical, but prompt in probe-val -> first reason wins
    return items


def _keys(items, ids):
    return {S.prompt_key(items[g]["context"]) for g in ids}


def _split(items=None, seed=42, n=(6, 5, 10)):
    items = items or _items()
    train_groups = list(range(10))                        # probe-train pairs
    train_keys = _keys(items, range(10)) | _keys(items, [23, 25])   # 25: same prompt as a probe-train dialogue
    val_keys = _keys(items, [24, 26])
    return S.build_split(items, train_groups, train_keys, val_keys, n_means_ref=n[0], n_means_probe=n[1],
                         n_eval=n[2], seed=seed)


def test_build_split_drops_roles_and_determinism():
    items = _items()
    sp = _split(items)
    c = sp["counts"]
    assert c["items"] == 80 and c["not_in_probe_train"] == 70
    assert c["dropped"] == {"prompt_in_probe_train": 2, "prompt_in_probe_val": 2, "empty_response": 2, "identical": 1}
    assert list(c["dropped"]) == list(S.DROP_REASONS)
    assert c["pool"] == 63 and c["clusters"] == 43                    # 20 two-item clusters (30-49 + 60-79) + 23 singles
    assert (c["means_ref"], c["means_probe"], c["eval"]) == (6, 5, 10) and c["reserve"] == 63 - 21
    roles = sp["roles"]
    all_ids = [g for r in S.SPLIT_ROLES for g in roles[r]]
    assert len(all_ids) == len(set(all_ids)) == 21
    assert not set(all_ids) & set(range(10)) and not set(all_ids) & {20, 21, 22, 23, 24, 25, 26}
    # no prompt cluster spans two roles, and no role item shares a prompt with probe-train / probe-val
    by_role = {r: {sp["cluster"][str(g)] for g in roles[r]} for r in S.SPLIT_ROLES}
    assert not (by_role["means_ref"] & by_role["means_probe"]) and not (by_role["means_ref"] & by_role["eval"])
    assert not (by_role["means_probe"] & by_role["eval"])
    assert not set().union(*by_role.values()) & (_keys(items, range(10)) | _keys(items, [24, 26]))
    for g in all_ids:
        assert sp["cluster"][str(g)] == S.prompt_key(items[g]["context"])
        assert sp["item_sha"][str(g)] == S.item_sha(items[g])
    # clusters stay together and keep (cluster order, group id) inside a role
    for r in S.SPLIT_ROLES:
        keys = [sp["cluster"][str(g)] for g in roles[r]]
        for k in set(keys):
            pos = [i for i, x in enumerate(keys) if x == k]
            assert pos == list(range(pos[0], pos[0] + len(pos)))
            ids = [roles[r][i] for i in pos]
            assert ids == sorted(ids) and len(ids) == sum(S.prompt_key(it["context"]) == k for it in items
                                                            if it["group"] not in {20, 21, 22})
    # whitespace-normalised prompt keys: 61 / 63 share their cluster with 31 / 33
    assert S.prompt_key(items[61]["context"]) == S.prompt_key(items[31]["context"])
    # deterministic for a fixed seed, different for another seed
    assert _split(items) == sp
    assert _split(items, seed=7)["roles"] != sp["roles"]
    # no text of an item in the split
    blob = json.dumps(sp)
    assert "answer" not in blob and "question" not in blob and "Human" not in blob


def test_split_prefix_errors_and_files(tmp_path):
    items = _items()
    sp = _split(items)
    by_id = {it["group"]: it for it in items}
    ev = S.split_items(sp, by_id, "eval")
    assert [it["group"] for it in ev] == sp["roles"]["eval"]
    assert S.split_items(sp, by_id, "eval", n=4) == ev[:4]
    with pytest.raises(ValueError):
        S.split_items(sp, by_id, "eval", n=11)
    with pytest.raises(ValueError):
        S.split_items(sp, by_id, "train")
    assert list(S.cluster_ids(sp, sp["roles"]["eval"][:3])) == [sp["cluster"][str(g)] for g in sp["roles"]["eval"][:3]]
    with pytest.raises(ValueError):                       # pool of 63 cannot hold 30 + 30 + 10
        _split(items, n=(30, 30, 10))
    with pytest.raises(ValueError):                       # only 2-item clusters left would overshoot an odd target
        S.build_split([_item(g, f"p{g // 2}", " a", " b") for g in range(8)], [], set(), set(), n_means_ref=3,
                      n_means_probe=0, n_eval=0)
    # write twice -> identical bytes; a different split refuses unless forced
    path = str(tmp_path / "ablation" / "hh_helpful_split.json")
    sha1 = S.write_split(path, sp)
    raw = open(path, "rb").read()
    sha2 = S.write_split(path, _split(items))
    assert sha1 == sha2 and open(path, "rb").read() == raw and sha1 == O.file_sha256(path)
    with pytest.raises(S.SplitMismatch):
        S.write_split(path, _split(items, seed=7))
    assert S.write_split(path, _split(items, seed=7), force=True) != sha1
    S.write_split(path, sp, force=True)
    loaded = S.load_split(path)
    assert loaded["roles"] == sp["roles"] and loaded["counts"] == sp["counts"]
    S.verify_split(loaded, by_id)
    changed = dict(by_id)
    g = sp["roles"]["eval"][0]
    changed[g] = {**by_id[g], "neg": [by_id[g]["neg"][0] + "x"]}
    with pytest.raises(ValueError):
        S.verify_split(loaded, changed)
    missing = {k: v for k, v in by_id.items() if k != sp["roles"]["means_ref"][0]}
    with pytest.raises(ValueError):
        S.verify_split(loaded, missing)


# ================================================================================================= 2. named sets

L8, H8, HPK2 = 8, 8, 2
BAND8 = (4, 7)
C33_SYN = _hs("4:0 4:1 5:2 5:3 5:6 6:0 6:4 7:1 7:5 7:7")
TOPK_SYN = _hs("4:0 5:2 5:3 6:4 4:3 5:0 5:7 6:6 1:1")       # 4 in C33 (O), 4 band candidates, 1 outside the band


def _write_table(path, L, H, top):
    s = np.random.default_rng(0).random(L * H) * 0.3 + 0.5
    for i, (l, h) in enumerate(top):
        s[l * H + h] = 0.99 - 0.01 * i
    rows = [{"layer": l, "head": h, "flat_idx": l * H + h, "auc": float(s[l * H + h]), "acc": float(s[l * H + h]),
             "bal_acc": 0.5, "majority": 0.5, "pos_rate": 0.5, "n_train": 20000, "n_val": 1400}
            for l in range(L) for h in range(H)]
    df = _finish_table(rows, {"layers": L, "heads": H, "score": "auc", "ranking": "stable", "feature_mode": "head"},
                       "auc", "stable")
    save_head_scores(df, str(path), meta={"model": "syn", "mode": "pythia"})
    return str(path)


def _census_syn():
    roles = {"readers": _hs("4:1 5:2 6:0 6:1 6:2 7:3"), "brakes": _hs("5:6 7:7 6:5")}
    eff = {lh: -0.1 * (i + 1) for i, lh in enumerate(roles["readers"])}
    eff.update({lh: 0.2 * (i + 1) for i, lh in enumerate(roles["brakes"])})
    return roles, eff


def _syn_sets(tmp_path, model="syn-base", dom=None):
    table = O.load_value_table(_write_table(tmp_path / f"{model}.json", L8, H8, TOPK_SYN), "helpful", "pythia")
    assert len(table.topk) == int(0.15 * L8 * H8) == 9 and set(table.topk) == set(TOPK_SYN)
    roles, eff = _census_syn()
    if dom is None:
        dom = np.linspace(0.1, 1.0, L8 * H8)
    return S.named_sets(topk=table.topk, c33=C33_SYN, census_roles=roles, census_effects=eff, dom_norm=dom,
                        n_layers=L8, n_heads=H8, band=BAND8, seed=42, model_identity=model, icl_identity="icl-base")


def test_named_sets_relations(tmp_path):
    sets = _syn_sets(tmp_path)
    assert list(sets) == [n for n in S.SET_ORDER if n in sets] and len(sets) == len(S.SET_ORDER)
    c33, o, cmo, top = (set(sets[n]["heads"]) for n in ("C33", "O", "CminusO", "TopK_all"))
    assert c33 == set(C33_SYN) and top == set(TOPK_SYN)
    assert o == c33 & top == set(_hs("4:0 5:2 5:3 6:4")) and cmo == c33 - o and not (o & cmo)
    for name, e in sets.items():
        assert e["n"] == len(e["heads"]) == len(e["flat"]) and e["flat"] == [l * H8 + h for l, h in e["heads"]]
        assert e["identity"] == ("icl-base" if name in S.ICL_SETS else "syn-base")
    pb = set(sets["P_band"]["heads"])
    band_cand = {lh for lh in top if 4 <= lh[0] <= 7} - c33
    assert pb <= band_cand and len(pb) == len(o) and S.layer_profile(pb) == S.layer_profile(o)
    assert sets["P_band"]["info"]["short_by"] == 0 and sets["P_band"]["info"]["moves"] == []
    assert set(sets["C33_readers"]["heads"]) == set(_hs("4:1 5:2 6:0"))
    assert set(sets["C33_brakes"]["heads"]) == set(_hs("5:6 7:7"))
    assert sets["WholeBand"]["n"] == 4 * H8 and {l for l, _ in sets["WholeBand"]["heads"]} == {4, 5, 6, 7}
    assert sets["CR20"]["heads"] == sorted(_census_syn()[0]["readers"])          # fewer than 20 readers: all of them
    assert sets["CR20"]["info"]["short_by"] == 14 and sets["CB24"]["info"]["short_by"] == 21
    assert set(sets["CensusReaders"]["heads"]) == set(_census_syn()[0]["readers"])
    # P_DoM: the top-|O| band heads by norm outside C33 (norm increases with the flat index -> layer 7 first)
    pdom = sets["P_DoM"]["heads"]
    assert pdom == sorted(_hs("7:6 7:4 7:3 7:2")) and not set(pdom) & c33
    # no census / no DoM -> those sets are left out
    s2 = S.named_sets(topk=TOPK_SYN, c33=C33_SYN, census_roles=None, census_effects=None, dom_norm=None, n_layers=L8,
                      n_heads=H8, band=BAND8, model_identity="m", icl_identity="i")
    assert set(s2) == {"C33", "O", "CminusO", "P_band", "WholeBand", "TopK_all"}


def test_named_sets_smoke_standin():
    sets = S.named_sets(topk=TOPK_SYN, c33=None, census_roles=_census_syn()[0], census_effects=_census_syn()[1],
                        dom_norm=None, n_layers=L8, n_heads=H8, band=BAND8, seed=42, model_identity="m",
                        icl_identity="qwen3-0.6b-base")
    c = sets["C33"]
    assert c["synthetic"] and c["rule"] == S.SMOKE_RULE and c["identity"] == "smoke" and c["n"] == S.SMOKE_C33_SIZE
    assert all(4 <= l <= 7 for l, _ in c["heads"])
    assert c["heads"] == S.smoke_c33(L8, H8, BAND8, 42) and S.smoke_c33(L8, H8, BAND8, 43) != c["heads"]
    assert sets["C33_readers"].get("synthetic") and sets["C33_readers"]["identity"] == "smoke"
    assert sets["WholeBand"]["identity"] == "qwen3-0.6b-base" and not sets["WholeBand"].get("synthetic")
    assert S.named_c33("qwen3-0.6b-base") is None and len(S.named_c33("qwen3-4b-base")) == 33


def test_p_band_nearest_layer_fill_and_shortage():
    c33 = _hs("5:0 5:1 5:2 5:3")
    o = _hs("5:0 5:1 5:2")                                                  # target 3 heads in layer 5
    topk = o + _hs("5:7 4:4 4:5 6:4 6:5")                                  # 1 candidate in 5, 2 in 4, 2 in 6
    heads, info = S.select_p_band(topk, c33, o, (4, 7), seed=1, identity="x")
    assert (5, 7) in heads and len(heads) == 3 and info["short_by"] == 0
    assert info["moves"] == [[5, 4], [5, 4]]                                # tie |4-5| == |6-5| -> the lower layer
    assert {lh for lh in heads if lh[0] == 4} == set(_hs("4:4 4:5"))
    assert info["candidates"] == 5 and info["per_layer_candidates"]["5"] == 1
    # deterministic, and the identity keys the subsample
    assert S.select_p_band(topk, c33, o, (4, 7), seed=1, identity="x") == (heads, info)
    big_o = _hs("5:0 5:1 5:2 5:3")
    many = big_o + [(5, h) for h in range(4, 8)]
    picks = {tuple(S.select_p_band(many, big_o, big_o[:2], (4, 7), seed=1, identity=k)[0]) for k in "abcdefgh"}
    assert len(picks) > 1
    # candidates run out -> smaller than |O|
    o5 = _hs("5:0 5:1 5:2 5:3")
    heads, info = S.select_p_band(o5 + _hs("6:1 3:0"), o5, o5, (4, 7), seed=1, identity="x")
    assert heads == [(6, 1)] and info["short_by"] == 3 and info["moves"] == [[5, 6]]


def test_p_dom_ties_and_exclusion():
    dom = np.zeros(L8 * H8)
    dom[4 * H8 + 0] = 9.0                                                   # in C33 -> excluded
    dom[5 * H8 + 1] = 5.0
    dom[6 * H8 + 3] = 4.0
    dom[7 * H8 + 2] = 4.0                                                   # tie with 6:3 -> lower flat index first
    dom[2 * H8 + 0] = 99.0                                                  # outside the band
    dom[4 * H8 + 5] = np.nan
    heads, info = S.select_p_dom(dom, H8, BAND8, [(4, 0)], 2)
    assert heads == [(5, 1), (6, 3)] and info["cutoff"] == 4.0
    assert info["dom_norm"] == {str(5 * H8 + 1): 5.0, str(6 * H8 + 3): 4.0} and "scale" in info["note"]
    heads3, _ = S.select_p_dom(dom, H8, BAND8, [(4, 0)], 3)
    assert heads3 == [(5, 1), (6, 3), (7, 2)]
    assert S.select_p_dom(dom, H8, BAND8, [], 0)[0] == []


def test_census_ranked_order():
    eff = {(21, 5): -0.3, (20, 9): -0.3, (22, 1): -0.5, (23, 0): float("nan"), (19, 2): 0.4, (18, 1): 0.4}
    assert S.census_ranked(eff, eff, 3) == [(22, 1), (20, 9), (21, 5)]
    assert S.census_ranked(eff, eff, 10)[-1] == (23, 0)
    assert S.census_ranked(eff, eff, 2, largest=True) == [(18, 1), (19, 2)]
    assert S.census_ranked(eff, [(21, 5), (19, 2)], 5) == [(21, 5), (19, 2)]


def test_profiles_and_layer_parsing():
    assert S.parse_layers("20-27") == tuple(range(20, 28)) and S.parse_layers("28,30-31") == (28, 30, 31)
    assert S.parse_layers([3, 1, 3]) == (1, 3) and S.band_layers((20, 22)) == (20, 21, 22)
    assert S.band_layers("1-2") == (1, 2)
    hs = _hs("1:0 1:1 1:3 2:5")
    assert S.from_flat(S.to_flat(hs, 8), 8) == hs
    assert S.layer_profile(hs) == {1: 3, 2: 1}
    assert S.kv_profile(hs, 2) == {1: (2, 1), 2: (1,)} and S.kv_profile(hs, 4) == {1: (3,), 2: (1,)}
    assert S.same_profile(hs, _hs("1:4 1:5 1:6 2:0"), 2)
    assert not S.same_profile(hs, _hs("1:0 1:2 1:4 2:0"), 2)               # (1, 1, 1) != (2, 1)
    assert S.layer_profile({1: [0, 1]}) == {1: 2}


@needs_repo
def test_real_census_sets_match_the_spec():
    census = I.census_sets("qwen3-4b-base", source=ICL_REPO)
    eff = census.effects("dret")
    cr20 = S.census_ranked(eff, census.roles["readers"], 20)
    assert cr20 == _hs("29:11 24:29 28:0 24:30 21:11 24:22 22:10 23:26 24:31 30:7 27:19 35:26 27:18 21:18 24:23 "
                       "22:0 26:26 30:5 29:14 17:4")
    assert math.isclose(sum(eff[x] for x in cr20), -2.547, abs_tol=5e-4)
    cb24 = S.census_ranked(eff, census.roles["brakes"], 24, largest=True)
    assert cb24 == _hs("25:30 28:23 31:30 21:31 25:14 33:21 22:8 23:28 25:28 24:28 24:21 22:2 27:16 17:7 28:2 34:29 "
                       "20:7 25:31 29:9 21:26 22:20 27:15 24:12 23:31")
    c33 = S.named_c33("qwen3-4b-base")
    top = O.ValueTable("helpful", "pythia", "auc", 36, 32, tuple(c33[:21]) + tuple((0, h) for h in range(32)),
                       np.zeros(36 * 32)).topk
    sets = S.named_sets(topk=top, c33=c33, census_roles=census.roles, census_effects=eff, dom_norm=None, n_layers=36,
                        n_heads=32, model_identity="qwen3-4b-base", icl_identity="qwen3-4b-base")
    assert sets["C33"]["n"] == 33 and S.layer_profile(c33) == {20: 1, 21: 4, 22: 12, 23: 6, 24: 5, 25: 2, 27: 3}
    groups = [c for t in S.kv_profile(c33, 4).values() for c in t]
    assert len(groups) == 22 and groups.count(1) == 13 and groups.count(2) == 8 and groups.count(4) == 1
    assert sets["C33_readers"]["heads"] == sorted(_hs("21:11 21:18 22:0 22:10 22:11 22:13 22:21 23:3 23:26 24:22 "
                                                      "24:29 27:19"))
    assert sets["C33_brakes"]["heads"] == sorted(_hs("22:2 22:8 22:9 22:20 23:28 24:21 24:28 25:28 27:15 27:16"))
    assert sets["CensusReaders"]["n"] == 89 and sets["WholeBand"]["n"] == 256 and sets["O"]["n"] == 21
    small = I.census_sets("qwen3-0.6b-base", source=ICL_REPO)
    assert S.census_ranked(small.effects("dret"), small.roles["readers"], 20) == _hs(
        "21:8 20:14 23:6 18:7 17:2 27:15 21:13 17:12 19:6 15:6 21:12 18:5 19:2 21:1 16:8 17:6 19:5 21:9 25:5 12:5")
    assert len(S.census_ranked(small.effects("dret"), small.roles["brakes"], 24, largest=True)) == 13


# ================================================================================================= 3. random draws

def test_random_draws_keep_layer_and_kv_profile():
    heads = _hs("1:0 1:1 1:2 1:5 3:7 3:6 3:0")                 # layer 1: groups (3, 1); layer 3: (2, 1) with hpk 4
    kw = dict(n_layers=4, n_heads=8, heads_per_kv=4, seed=42, identity="m", set_name="X")
    draws = S.random_draws(heads, R=60, **kw)
    assert len(draws) == 60
    for d in draws:
        hd = S.from_flat(d, 8)
        assert d == sorted(d) and len(set(d)) == len(heads)
        assert S.layer_profile(hd) == S.layer_profile(heads) and S.kv_profile(hd, 4) == S.kv_profile(heads, 4)
    assert len({tuple(d) for d in draws}) > 20                  # the locations vary
    # prefix property: R = 5 is the start of R = 10; start= continues the stream
    assert S.random_draws(heads, R=10, **kw)[:5] == S.random_draws(heads, R=5, **kw)
    assert S.random_draws(heads, R=5, start=5, **kw) == S.random_draws(heads, R=10, **kw)[5:]
    assert S.random_draws(heads, R=10, **{**kw, "set_name": "Y"}) != S.random_draws(heads, R=10, **kw)
    assert S.random_draws(heads, R=10, **{**kw, "identity": "other"}) != S.random_draws(heads, R=10, **kw)
    assert S.random_draws([], R=3, **kw) == [[], [], []]
    # head-exchangeable fallback (MHA, hpk 1) keeps the layer counts
    for d in S.random_draws(heads, R=10, **{**kw, "heads_per_kv": 1}):
        assert S.layer_profile(S.from_flat(d, 8)) == S.layer_profile(heads)


def test_random_table_identity_sharing_and_reuse(tmp_path):
    base = _syn_sets(tmp_path, "syn-base")
    inst = _syn_sets(tmp_path, "syn")
    R = S.r_table(4, 4, 3, 2)
    kw = dict(n_layers=L8, n_heads=H8, heads_per_kv=HPK2, seed=42)
    rb, ri = S.random_table(base, R, **kw), S.random_table(inst, R, **kw)
    assert rb["C33"] == ri["C33"] and rb["C33"]["identity"] == "icl-base" and rb["C33"]["R"] == 4
    assert rb["CR20"] == ri["CR20"]
    assert rb["O"]["draws"] != ri["O"]["draws"]                 # table sets are keyed by the defining model
    assert rb["O"]["draws"] != rb["C33"]["draws"][:4]
    assert rb["WholeBand"]["R"] == 0 and rb["WholeBand"]["draws"] == []
    # P_band has O's profile here -> F12 reuse; otherwise its own draws
    same = S.same_profile(base["P_band"]["heads"], base["O"]["heads"], HPK2)
    assert S.pband_reuses_o(base, HPK2) == same
    assert (rb["P_band"] == {"reuse": "O"}) == same
    other = dict(base)
    other["P_band"] = {**base["P_band"], "heads": [(4, 3)], "flat": [35], "n": 1}
    rt = S.random_table(other, R, **kw)
    assert rt["P_band"]["R"] == 3 and all(S.layer_profile(S.from_flat(d, H8)) == {4: 1} for d in rt["P_band"]["draws"])


# ================================================================================================= 4. frozen file

def _frozen(tmp_path, R=None, sets=None):
    sets = sets or _syn_sets(tmp_path)
    geo = {"layers": L8, "heads": H8, "kv_heads": H8 // HPK2, "heads_per_kv": HPK2, "head_dim": 4}
    rnd = S.random_table(sets, R or S.r_table(4, 4, 3, 2), n_layers=L8, n_heads=H8, heads_per_kv=HPK2, seed=42)
    return S.build_frozen(model_short="syn-base", icl_identity="icl-base", geometry=geo, band=BAND8, seed=42,
                          sets=sets, randoms=rnd, inputs={"heads_table": {"path": "x.json", "sha256": "0" * 64}})


def test_frozen_roundtrip_mismatch_force_and_extend(tmp_path):
    fz = _frozen(tmp_path)
    assert fz["band"] == [4, 7] and fz["sets"]["C33"]["layers"] == {"4": 2, "5": 3, "6": 2, "7": 3}
    assert fz["sets"]["C33"]["kv"]["5"] == [2, 1]
    path = str(tmp_path / "ablation" / "syn-base_frozen_sets.json")
    sha = S.write_frozen(path, fz)
    loaded = S.load_frozen(path)
    assert loaded["sets"]["O"]["heads"] == [list(x) for x in _hs("4:0 5:2 5:3 6:4")]
    assert S.set_heads(loaded, "O") == _hs("4:0 5:2 5:3 6:4")
    assert S.random_heads(loaded, "C33", 3) == S.from_flat(fz["random"]["C33"]["draws"][3], H8)
    if loaded["random"]["P_band"].get("reuse") == "O":
        assert S.random_heads(loaded, "P_band", 1) == S.random_heads(loaded, "O", 1)
    with pytest.raises(IndexError):
        S.random_heads(loaded, "C33", 4)
    # same sets again (a later Stage 0, new timestamp) -> the stored file is kept byte for byte
    raw = open(path, "rb").read()
    assert S.write_frozen(path, _frozen(tmp_path)) == sha and open(path, "rb").read() == raw
    # changed sets -> FrozenMismatch, unless forced
    changed = _syn_sets(tmp_path)
    changed["O"] = {**changed["O"], "heads": _hs("4:0 5:2 5:3")}
    with pytest.raises(S.FrozenMismatch):
        S.write_frozen(path, _frozen(tmp_path, sets=changed))
    other_draws = _frozen(tmp_path)
    other_draws["random"]["C33"]["draws"][0] = other_draws["random"]["C33"]["draws"][1]
    with pytest.raises(S.FrozenMismatch):
        S.write_frozen(path, other_draws)
    assert S.write_frozen(path, _frozen(tmp_path, sets=changed), force=True) != sha
    assert S.set_heads(S.load_frozen(path), "O") == _hs("4:0 5:2 5:3")
    # extension keeps the stored draws (prefix property) and is accepted without force
    S.write_frozen(path, fz, force=True)
    stored = S.load_frozen(path)
    ext = S.extend_random(stored, "C33", 7)
    assert ext["random"]["C33"]["draws"][:4] == stored["random"]["C33"]["draws"] and ext["random"]["C33"]["R"] == 7
    assert ext["random"]["C33"]["extended"] == [{"from": 4, "to": 7}]
    assert stored["random"]["C33"]["R"] == 4                                 # the input is not modified
    assert ext["random"]["C33"]["draws"] == _frozen(tmp_path, R=S.r_table(7, 4, 3, 2))["random"]["C33"]["draws"]
    S.write_frozen(path, ext)
    assert S.random_R(S.load_frozen(path), "C33") == 7
    assert S.extend_random(ext, "C33", 5) == ext                             # nothing to add
    # a payload with fewer draws than stored is compatible: the stored (longer) file stays
    sha_ext = O.file_sha256(path)
    assert S.write_frozen(path, fz) == sha_ext


def test_build_frozen_rejects_foreign_draws(tmp_path):
    sets = _syn_sets(tmp_path)
    rnd = S.random_table(sets, S.r_table(2, 2, 2, 2), n_layers=L8, n_heads=H8, heads_per_kv=HPK2, seed=42)
    rnd["O"]["draws"][0] = rnd["C33"]["draws"][0]                          # wrong layer profile for O
    geo = {"layers": L8, "heads": H8, "kv_heads": 4, "heads_per_kv": HPK2, "head_dim": 4}
    with pytest.raises(ValueError):
        S.build_frozen(model_short="m", icl_identity="i", geometry=geo, band=BAND8, seed=42, sets=sets, randoms=rnd,
                       inputs={})


# ================================================================================================= 5. bootstrap

def _clustered(seed=3, G=120):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, G)
    cl = np.repeat(np.arange(G), sizes)
    d = rng.normal(0.3, 1.0, G)[cl] + rng.normal(0, 0.5, len(cl))
    return d, cl


def test_cluster_bootstrap_matches_paired_delta_ci():
    d, cl = _clustered()
    ref = paired_delta_ci(d, B=4000, seed=1, groups=cl)
    got = T.cluster_bootstrap(d, cl, B=4000, seed=0)
    assert got["mean"] == ref["mean"]
    sd = float(np.std(d))
    assert abs(got["lo"] - ref["lo"]) < 0.02 * sd and abs(got["hi"] - ref["hi"]) < 0.02 * sd
    assert abs(got["se"] - ref["se"]) < 0.02 * sd
    assert got["n"] == len(d) and got["n_clusters"] == 120 and got["B"] == 4000 and got["ci"] == 0.95
    # chunk size does not change the result for a fixed seed
    for chunk in (1, 7, 500, 10000):
        assert T.cluster_bootstrap(d, cl, B=1000, seed=5, chunk=chunk) == T.cluster_bootstrap(d, cl, B=1000, seed=5)
    assert T.cluster_bootstrap(d, cl, B=1000, seed=6) != T.cluster_bootstrap(d, cl, B=1000, seed=5)
    # string clusters, 2-D (concept, seq) clusters and item bootstrap
    s = T.cluster_bootstrap(d, np.array([f"k{c}" for c in cl]), B=1000, seed=5)
    two = T.cluster_bootstrap(d, np.stack([cl // 10, cl % 10], axis=1), B=1000, seed=5)
    assert s["mean"] == two["mean"] == got["mean"] and s["n_clusters"] == two["n_clusters"] == 120
    item = T.cluster_bootstrap(d, None, B=1000, seed=5)
    assert item["n_clusters"] == len(d) and item["hi"] - item["lo"] < s["hi"] - s["lo"]


def test_cluster_bootstrap_weights_nan_and_degenerate():
    d, cl = _clustered(4, 60)
    w = np.random.default_rng(1).integers(1, 50, len(d)).astype(float)
    d2 = d.copy()
    d2[cl == 0] = np.nan                                                   # a whole cluster disappears
    d2[5] = np.nan
    got = T.cluster_bootstrap(d2, cl, weights=w, B=2000, seed=0)
    ref = paired_delta_ci(d2, B=2000, seed=0, groups=cl, weights=w)
    keep = np.isfinite(d2)
    assert got["mean"] == ref["mean"] and math.isclose(got["mean"], np.sum(w[keep] * d2[keep]) / np.sum(w[keep]))
    assert got["n"] == int(keep.sum()) and got["n_clusters"] == ref["n_groups"] == 59
    assert abs(got["lo"] - ref["lo"]) < 0.05 and abs(got["hi"] - ref["hi"]) < 0.05
    # a constant change has a degenerate CI at that value; one cluster / B = 0 / empty -> NaN CI
    c = T.cluster_bootstrap(np.full(30, -0.25), np.arange(30) // 3, B=500)
    assert c["mean"] == c["lo"] == c["hi"] == -0.25
    one = T.cluster_bootstrap([1.0, 2.0, 3.0], [0, 0, 0], B=100)
    assert one["mean"] == 2.0 and math.isnan(one["lo"]) and one["n_clusters"] == 1
    assert math.isnan(T.cluster_bootstrap(d, cl, B=0)["lo"])
    e = T.cluster_bootstrap([np.nan, np.nan], [0, 1], B=10)
    assert e["n"] == 0 and math.isnan(e["mean"])
    with pytest.raises(ValueError):
        T.cluster_bootstrap([1.0, 2.0], [0], B=10)


def test_cluster_bootstrap_ratio():
    rng = np.random.default_rng(0)
    G = 100
    cl = np.repeat(np.arange(G), 2)
    den = rng.normal(-1.0, 0.2, 2 * G)
    num = 0.6 * den + rng.normal(0, 0.05, 2 * G)
    r = T.cluster_bootstrap_ratio(num, den, cl, B=2000, seed=0)
    assert r["defined"] and math.isclose(r["ratio"], num.sum() / den.sum()) and r["lo"] < 0.6 < r["hi"]
    assert r["den"]["hi"] < 0 and r["num"]["lo"] < r["num"]["mean"] < r["num"]["hi"]
    # SRI = 1 - TE / DE
    assert 0.3 < 1 - r["ratio"] < 0.5
    flat = T.cluster_bootstrap_ratio(num, rng.normal(0, 1, 2 * G), cl, B=500)
    assert not flat["defined"] and math.isnan(flat["ratio"]) and math.isnan(flat["lo"])
    assert flat["den"]["lo"] <= 0 <= flat["den"]["hi"]


# ================================================================================================= 6. AUC deltas

def test_paired_auc_delta():
    rng = np.random.default_rng(0)
    N, K = 200, 5
    y = np.tile([1, 0], N // 2)
    cl = np.repeat(np.arange(N // 2), 2)                                    # (chosen, rejected) of one item
    ref = rng.normal(0, 1, (K, N)) + 1.2 * y
    cond = ref * 0.3 + rng.normal(0, 1, (K, N))
    out = T.paired_auc_delta(ref, cond, y, cl, B=400, seed=0)
    exact = float(np.mean(auc_matrix(cond, y) - auc_matrix(ref, y)))
    assert math.isclose(out["delta"], exact, abs_tol=1e-12) and out["k"] == K and out["n"] == N
    assert math.isclose(out["auc_ref"], float(np.mean(auc_matrix(ref, y)))) and out["delta"] < 0
    assert out["lo"] <= out["delta"] <= out["hi"] and out["n_clusters"] == N // 2
    # excluded (ablated) readout heads = all-NaN rows are dropped from both matrices
    ref2, cond2 = ref.copy(), cond.copy()
    cond2[1] = np.nan
    o2 = T.paired_auc_delta(ref2, cond2, y, cl, B=50)
    keep = [0, 2, 3, 4]
    assert o2["k"] == 4 and math.isclose(o2["delta"], float(np.mean(auc_matrix(cond[keep], y) - auc_matrix(ref[keep], y))))
    one = T.paired_auc_delta(ref[0], ref[0], y, cl, B=50)
    assert one["delta"] == 0.0 and one["lo"] == 0.0 == one["hi"] and one["k"] == 1


def test_paired_auc_delta_equals_weighted_auc_bootstrap():
    """The quadratic-form bootstrap equals mean_k WeightedAUC(cond) - WeightedAUC(ref) on the same cluster counts,
    including ties and unequal cluster sizes / label mixes."""
    rng = np.random.default_rng(5)
    N, K, B = 90, 3, 300
    y = rng.integers(0, 2, N)
    cl = rng.integers(0, 25, N)
    ref = np.round(rng.normal(0, 1, (K, N)) + y, 1)                         # rounding -> many ties
    cond = np.round(rng.normal(0, 1, (K, N)) + 0.5 * y, 1)
    out = T.paired_auc_delta(ref, cond, y, cl, B=B, seed=3)
    _, inv = np.unique(cl, return_inverse=True)
    G = inv.max() + 1
    counts = np.random.RandomState(3).multinomial(G, np.full(G, 1.0 / G), size=B)
    wr, wc = WeightedAUC(ref, y), WeightedAUC(cond, y)
    boots = np.array([np.mean(wc.auc(c[inv].astype(float)) - wr.auc(c[inv].astype(float))) for c in counts])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    assert math.isclose(out["lo"], lo, abs_tol=1e-12) and math.isclose(out["hi"], hi, abs_tol=1e-12)
    assert math.isclose(out["se"], np.std(boots, ddof=1), abs_tol=1e-12)
    assert math.isclose(WeightedAUC(ref, y).auc(np.ones(N)).mean(), out["auc_ref"], abs_tol=1e-12)


# ================================================================================================= 7. vs_random

def test_vs_random_hand_computed():
    r = [1.0, 2.0, 3.0, 4.0, np.nan]
    v = T.vs_random(2.0, r, "le")
    assert v["R"] == 4 and v["p_le"] == 3 / 5 and v["p_ge"] == 4 / 5 and v["p_one"] == 3 / 5 and v["p_two"] == 1.0
    assert math.isclose(v["z"], -0.5 / np.std([1, 2, 3, 4], ddof=1)) and v["rand_mean"] == 2.5
    assert v["rand_min"] == 1.0 and v["rand_max"] == 4.0 and v["n_le"] == 2 and v["n_ge"] == 3
    assert math.isclose(v["rand_q025"], np.quantile([1, 2, 3, 4], 0.025))
    low = T.vs_random(0.0, r, "le")
    assert low["p_le"] == 1 / 5 and low["p_ge"] == 1.0 and math.isclose(low["p_two"], 0.4)
    hi = T.vs_random(9.0, r, "ge")
    assert hi["p_one"] == hi["p_ge"] == 1 / 5 and hi["p_le"] == 1.0
    two = T.vs_random(0.0, r, "two")
    assert math.isnan(two["p_one"]) and math.isclose(two["p_two"], 0.4)
    none = T.vs_random(0.5, [], "le")
    assert none["R"] == 0 and math.isnan(none["p_le"]) and math.isnan(none["p_two"]) and math.isnan(none["z"])
    assert math.isnan(T.vs_random(0.5, [np.nan, np.nan], "le")["p_one"])
    assert math.isnan(T.vs_random(float("nan"), r, "le")["p_le"])
    flat = T.vs_random(1.0, [1.0, 1.0, 1.0], "le")
    assert math.isnan(flat["z"]) and flat["p_le"] == 1.0
    with pytest.raises(ValueError):
        T.vs_random(0.0, r, "lt")


# ================================================================================================= 8. Holm, floor, gate, terciles, sign

def test_primary_holm():
    rows = T.primary_holm([{"set": "C33", "p_one": 0.01, "observed": -0.2}, {"set": "O", "p_one": 0.04, "observed": -0.1}])
    assert [r["p_holm"] for r in rows] == [0.02, 0.04] and all(r["significant"] for r in rows)
    assert rows[0]["observed"] == -0.2 and rows[0]["set"] == "C33"
    rows = T.primary_holm([{"set": "C33", "p_one": 0.03}, {"set": "O", "p_one": 0.04}])
    assert [r["p_holm"] for r in rows] == [0.06, 0.06] and not any(r["significant"] for r in rows)
    rows = T.primary_holm([{"set": "C33", "p_one": float("nan")}, {"set": "O", "p_one": 0.04}])
    assert math.isnan(rows[0]["p_holm"]) and not rows[0]["significant"] and rows[1]["p_holm"] == 0.04
    assert T.primary_holm([]) == []


def test_floor_check():
    cl = np.arange(400) // 2
    rng = np.random.default_rng(0)

    def run(rate):
        win = np.zeros(400)
        win[:int(rate * 400)] = 1
        rng.shuffle(win)
        margin = np.where(win > 0, 0.1, -0.1) + rng.normal(0, 0.01, 400)
        return T.floor_check(win, margin, rng.integers(0, 2, 400), cl, B=1000)

    f = run(0.5)
    assert f["floor"] and not f["inverted"] and f["acc_lo"] <= 0.5 <= f["acc_hi"] and "**Boden:**" in f["text_de"]
    assert "nur M2 und M3" in f["text_de"] and f["acc"] == 0.5 and f["n"] == 400 and f["n_clusters"] == 200
    ok = run(0.8)
    assert not ok["floor"] and not ok["inverted"] and ok["acc_lo"] > 0.5 and "Kein Boden" in ok["text_de"]
    assert ok["margin"] > 0 and 0.3 < ok["len_baseline"] < 0.7
    inv = run(0.2)
    assert inv["inverted"] and not inv["floor"] and "invertiert" in inv["text_de"]
    assert T.floor_check([1.0], [0.1], [1], [0], B=100)["floor"]           # CI undefined -> floor (conservative)


def _gate(mean=-0.6, sd=0.3, randoms=(-0.1, 0.0, 0.05, 0.1), calibration=True, **kw):
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(48), 4)
    d = mean + rng.normal(0, sd, len(cl))
    cal = {"ok": calibration, "diffs": {"symrare": 0.01, "nat": 0.4 if not calibration else 0.02}, "tol": 0.15} \
        if calibration is not None else None
    return T.gate_decision(d, cl, list(randoms), calibration=cal, B=1000, **kw)


def test_gate_decision_criteria():
    g = _gate()
    assert g["passed"] and g["proceed"] and not g["advisory"] and g["hi"] < 0
    assert g["criteria"] == {"calibration": True, "ci_below_zero": True, "below_all_random": True}
    assert g["random"]["R"] == 4 and g["random"]["p_le"] == 1 / 5 and "bestanden" in g["reason_de"]
    assert "Kalibrierung ok" in g["reason_de"]
    cal = _gate(calibration=False)
    assert not cal["passed"] and cal["criteria"]["calibration"] is False and cal["criteria"]["ci_below_zero"]
    assert "Kalibrierung" in cal["reason_de"] and "nat +0.400" in cal["reason_de"]
    ci = _gate(mean=-0.05, sd=1.0, randoms=(0.5, 0.6))
    assert not ci["passed"] and not ci["criteria"]["ci_below_zero"] and ci["criteria"]["below_all_random"]
    assert "nicht vollständig unter 0" in ci["reason_de"]
    rnd = _gate(randoms=(-2.0, 0.0, 0.1))
    assert not rnd["passed"] and rnd["criteria"]["ci_below_zero"] and not rnd["criteria"]["below_all_random"]
    assert "nicht unter allen 3 Zufallsmengen" in rnd["reason_de"]
    empty = _gate(randoms=())
    assert not empty["passed"] and not empty["criteria"]["below_all_random"]
    skipped = _gate(calibration=None)
    assert skipped["passed"] and skipped["criteria"]["calibration"] is None and "übersprungen" in skipped["reason_de"]
    adv = _gate(randoms=(-2.0,), advisory=True)
    assert not adv["passed"] and adv["advisory"] and adv["proceed"] and "beratend" in adv["reason_de"]


def test_terciles_and_sign_change():
    lab, edges = T.tercile_labels(np.arange(9.0))
    assert lab.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2] and np.allclose(edges, [8 / 3, 16 / 3])
    lab, edges = T.tercile_labels([1, 1, 1, 1, 1, 2])
    assert edges.tolist() == [1.0, 1.0] and lab.tolist() == [2] * 6       # ties at an edge go up ('right')
    lab, _ = T.tercile_labels([0.0, 1.0, np.nan, 2.0, 1.0, 1.0])
    assert lab[2] == -1 and lab.tolist() == [0, 2, -1, 2, 2, 2]
    lab, edges = T.tercile_labels([np.nan])
    assert lab.tolist() == [-1] and np.isnan(edges).all()
    a = {"mean": -0.2, "lo": -0.3, "hi": -0.1}
    b = {"mean": 0.2, "lo": 0.1, "hi": 0.3}
    z = {"mean": 0.05, "lo": -0.1, "hi": 0.2}
    assert T.sign_change(a, b) == "Vorzeichenwechsel" and T.sign_change(a, a) == "gleiche Richtung"
    assert T.sign_change(a, z) == "unentschieden" and T.sign_change(None, a) == "unentschieden"
    assert T.sign_change(a, {"mean": 1.0, "lo": float("nan"), "hi": float("nan")}) == "unentschieden"


def test_prereg_directions():
    assert T.direction("M1", "C33") == "le" and T.direction("M1", "O") == "le" and T.direction("M1", "O_base") == "le"
    assert T.direction("M2", "P_band") == "le" and T.direction("M4", "TopK_all") == "ge"
    assert T.direction("M3", "CR20") == "le" and T.direction("M3", "CB24") == "ge" and T.direction("M3", "C33") == "le"
    for s in ("O", "P_band", "TopK_all", "WholeBand"):
        assert T.direction("M3", s) == "two"
    for m in ("M1", "M2", "M3", "M4"):
        assert T.direction(m, "C33_readers") == "two" == T.direction(m, "C33_brakes")
    assert T.direction("M7", "C33") == "two" and set(T.PREREG.values()) <= {"le", "ge", "two"}


# ================================================================================================= 9. verdict texts

def _primary(p_c33=0.01, p_o=0.5):
    tests = [{"set": "C33", "observed": -0.12, "lo": -0.18, "hi": -0.06, "p_one": p_c33, "R": 100, "n_le": 0},
             {"set": "O", "observed": -0.01, "lo": -0.04, "hi": 0.02, "p_one": p_o, "R": 100, "n_le": 49}]
    return T.primary_holm(tests)


def test_verdict_lines_key_phrases():
    gate_ok = {"passed": True, "advisory": False, "reason_de": "bestanden"}
    gate_bad = {"passed": False, "advisory": False, "reason_de": "CI nicht unter 0"}
    floor_no = {"floor": False, "inverted": False, "acc": 0.62, "acc_lo": 0.59, "acc_hi": 0.65}
    floor_yes = {"floor": True, "inverted": False, "acc": 0.51, "acc_lo": 0.48, "acc_hi": 0.54}
    stop = T.verdict_lines(_primary(), floor_no, gate_bad, model="qwen3-4b-base")
    assert len(stop) == 1 and stop[0].startswith("**STOPP:**") and "CI nicht unter 0" in stop[0]
    assert "STOPP" in T.verdict_lines(_primary(), floor_no, None, model="m")[0]
    lines = T.verdict_lines(_primary(), floor_no, gate_ok, model="qwen3-4b-base")
    text = "\n".join(lines)
    assert "**C33: Abhängigkeit nachgewiesen**" in text and "stärker als 100 von 100" in text
    assert "senkt die Helpful-Marge um 0.120 Nats" in text
    assert "**O: keine Abhängigkeit nachgewiesen**" in text and "Expression ohne nachgewiesene Reliance" in text
    assert "STOPP" not in text and "Boden" not in text and "SMOKE" not in text
    both = "\n".join(T.verdict_lines(_primary(0.01, 0.01), floor_no, gate_ok, model="m"))
    assert "**O: Abhängigkeit nachgewiesen**" in both and "(Expression + Reliance)" in both
    fl = "\n".join(T.verdict_lines(_primary(), floor_yes, gate_ok, model="m"))
    assert "**Boden:**" in fl and fl.count("nicht interpretierbar (Boden)") == 2 and "nachgewiesen**" not in fl
    tr = T.verdict_lines(_primary(), floor_no, gate_ok, model="qwen3-4b", transferred=True, smoke=True)
    assert tr[0] == T.SMOKE_LINE and sum(T.TRANSFER_MARK in line for line in tr) == 1
    c33_line = next(line for line in tr if line.startswith("**C33"))
    o_line = next(line for line in tr if line.startswith("**O"))
    assert T.TRANSFER_MARK in c33_line                                  # C33 = Base identities
    assert T.TRANSFER_MARK not in o_line and T.TRANSFER_MARK_O in o_line   # O = C33 ∩ the model's own top-k
    adv = T.verdict_lines(_primary(), floor_no, {"passed": False, "advisory": True, "reason_de": "x"}, model="m",
                          smoke=True)
    assert adv[0] == T.SMOKE_LINE and "beratenden" in adv[1] and not any("STOPP" in line for line in adv)
