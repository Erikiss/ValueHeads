"""ICL battery and query knockout (valueheads_pythia.icl_battery, ABLATION_DESIGN part C, section 7.5).

Offline. The battery tests with the real tokenizer use the cached Qwen/Qwen3-0.6B-Base tokenizer (tokenizer only) and
are skipped when it is not in the HF cache; the census tests read the local ICL checkout (`needs_repo`). The knockout
mechanics run on a tiny random Qwen3 (GQA, head_dim decoupled from hidden) with a test-local lossless tokenizer.

Battery equivalence with the original (recorded 2026-11-05): the original builder
`icl/tasks.py build_battery(tok, ("concept12n",), smoke)` of andrewgcodes/in-context-learning-circuit @ 26c44caf8f9b was
run once in a scratch process (`python3 -I -B`, sys.path -> the checkout's icl/, env BATTERY_SEED=1 / 2, CONCEPT_ORDER and
CONCEPT_PERM unset, cwd = the scratch dir, Qwen3-0.6B-Base tokenizer only, target / alt ids via icl/tokspans.py
target_ids) and fingerprinted with the canonical encoding of `battery_fingerprint`. It gave exactly the pinned values
below: seed 1 f637629192f5..., seed 2 3804b91949ee..., smoke (seed 2) 15bc382a7eac... The same run measured prompt
lengths 118-181 tokens for both seeds and 337 (seed 1) / 340 (seed 2) single-token alts, not the scouting values
115-183 / 341 of the spec (section 7.5); the measured values are pinned here.
"""
import collections
import dataclasses
import hashlib
import math
import os
import shutil

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from valueheads_pythia import icl_battery as IB
from valueheads_pythia import icl_sets as I
from valueheads_pythia.hooks import HeadHooks

ICL_REPO = os.environ.get(I.ENV_REPO) or "/home/user/andrewgcodes/in-context-learning-circuit"
HAVE_REPO = os.path.exists(os.path.join(ICL_REPO, I.CENSUS["qwen3-4b-base"].path))
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="ICL repo checkout not present")

FP_SEED1 = "f637629192f587a9abf2cd94661d31bada1c07011b63b4b369227318986b22fc"
FP_SEED2 = "3804b91949ee6afb59d328ded27dbd8743627f60f5979f93f99a36b261ce998a"
FP_SMOKE_SEED2 = "15bc382a7eacadcca42ae5454766bfa46d7b719c60bfff051706dddc030da81b"
SINGLE_ALTS = {1: 337, 2: 340}

CR20_4B = "29:11 24:29 28:0 24:30 21:11 24:22 22:10 23:26 24:31 30:7 27:19 35:26 27:18 21:18 24:23 22:0 26:26 30:5 29:14 17:4"
CB24_4B = ("25:30 28:23 31:30 21:31 25:14 33:21 22:8 23:28 25:28 24:28 24:21 22:2 27:16 17:7 28:2 34:29 20:7 25:31 29:9 "
           "21:26 22:20 27:15 24:12 23:31")
CR20_06B = "21:8 20:14 23:6 18:7 17:2 27:15 21:13 17:12 19:6 15:6 21:12 18:5 19:2 21:1 16:8 17:6 19:5 21:9 25:5 12:5"


def _lh(text):
    return [tuple(int(v) for v in t.split(":")) for t in text.split()]


# ------------------------------------------------------------------------------------------------ helpers

class GreedyTokenizer:
    """Lossless test tokenizer: greedy longest match over a few multi-character tokens (' foo', ' bar', ' good', ' bad':
    single-token labels) and single characters (vocabulary from the characters of the concept data). id 0 is padding."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 0
    bos_token_id = None
    MERGES = (" foo", " bar", " good", " bad")

    def __init__(self, texts):
        chars = sorted(set("".join(texts)) | set(" ->\n") | set("".join(self.MERGES)))
        self.vocab = ["<pad>"] + list(self.MERGES) + chars
        self.ids = {t: i for i, t in enumerate(self.vocab)}
        self.merges = sorted(self.MERGES, key=len, reverse=True)

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for m in self.merges:
                if text.startswith(m, i):
                    out.append(self.ids[m])
                    i += len(m)
                    break
            else:
                out.append(self.ids[text[i]])
                i += 1
        return out

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self.encode(text)}

    def decode(self, ids):
        return "".join(self.vocab[int(i)] for i in ids if int(i) != 0)


def tiny_qwen3(vocab, heads=4, kv_heads=2, head_dim=16, hidden=48, layers=3, seed=0):
    """Random Qwen3 (copied from tests/test_qwen_port.py, vocabulary and context enlarged): H * D = 64 != hidden 48, GQA
    with 2 query heads per KV head, fp32 eager."""
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
                      max_position_embeddings=2048, attention_bias=False, tie_word_embeddings=True, pad_token_id=0,
                      eos_token_id=0, bos_token_id=None)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():                    # random init gives near-zero attention outputs; make them clearly non-zero
        for block in model.model.layers:
            for lin in (block.self_attn.v_proj, block.self_attn.q_proj, block.self_attn.k_proj):
                lin.weight.normal_(0, 0.3)
    return model


class Capture(HeadHooks):
    """Records the o_proj input of `layers` (registered after an edit hook, it sees the edited input)."""

    def __init__(self, model, layers):
        super().__init__(model, positions="all", decode_only=False, layers=layers)
        self.values = {}

    def edit(self, layer, x):
        self.values[layer] = x.detach().clone()
        return x


def short_concepts(per_class=8, seed=0):
    """Synthetic ASCII concept data (short two-word sentences) with the concept12n structure, so that the tiny-model
    mechanics run on ~200-token prompts instead of ~800 characters per real prompt."""
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))

    def sentence():
        return " ".join("".join(rng.choice(letters, int(rng.integers(2, 6)))) for _ in range(2))

    return {c: [[sentence() for _ in range(per_class)], [sentence() for _ in range(per_class)]] for c in IB.CONCEPTS_12N}


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """The tiny model's ops are far too small for intra-op threads (20x slower on a loaded 4-CPU box); restored after."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def test_tok():
    """Lossless tokenizer whose vocabulary covers the characters of the concept data file (and of the labels)."""
    concepts = IB.load_concepts()
    texts = [s for a, b in concepts.values() for s in a + b] + [w for p in IB.NATLAB.values() for w in p]
    return GreedyTokenizer(texts + ["foo bar zorp quil abcdefghijklmnopqrstuvwxyz"])


@pytest.fixture(scope="module")
def tiny(test_tok):
    model = tiny_qwen3(len(test_tok.vocab))
    items = IB.concept12n(test_tok, seed=IB.FRESH_SEED, n_seq=2, concepts=short_concepts())   # 32 items, 2 per kind
    return model, test_tok, items


@pytest.fixture(scope="module")
def qwen_tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except Exception as exc:   # not cached / offline
        pytest.skip(f"Qwen3-0.6B-Base tokenizer not cached: {type(exc).__name__}")


KW = dict(batch_tokens=4096, max_batch=8)


def _reference_margin(model, item, knock=None, ref=None):
    """Unbatched per-item reference with icl/score.py seq_logprob semantics (one row, no padding); optional knock with
    the per-head reference `ref` {layer: (n, D)} at T - 1."""
    T = len(item.ids)

    def run(cont):
        ids = torch.tensor([list(item.ids) + list(cont)])
        if knock is not None:
            knock.set_rows(torch.tensor([T - 1]), {l: r[None] for l, r in ref.items()})
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0].float(), -1)
        pos = torch.arange(T - 1, T - 1 + len(cont))
        return lp, float(lp[pos, torch.tensor(cont)].sum())

    lp, tlp = run(item.target_ids)
    alt = float(lp[T - 1, item.alt_ids[0]]) if len(item.alt_ids) == 1 else run(item.alt_ids)[1]
    return tlp, alt


# ------------------------------------------------------------------------------------------------ 1. data file

def test_concept_tasks_file_is_pinned_and_verified(tmp_path):
    with open(IB.CONCEPT_TASKS_PATH, "rb") as f:
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == IB.CONCEPT_TASKS_SHA256 and len(data) == 11115
    concepts = IB.load_concepts()
    assert list(concepts)[:3] == ["topic", "lang", "question"]
    assert [c for c in concepts if c in IB.CONCEPTS_12N] == list(IB.CONCEPTS_12N)   # JSON order = concept index order
    assert all(len(a) == 16 and len(b) == 16 for c, (a, b) in concepts.items() if c in IB.CONCEPTS_12N)

    bad = tmp_path / "concept_tasks.json"
    bad.write_bytes(data.replace(b"]", b" ]", 1))          # semantically equal JSON, different bytes
    with pytest.raises(ValueError, match="sha256"):
        IB.load_concepts(bad)
    assert IB.load_concepts(bad, verify=False) == concepts
    with pytest.raises(ValueError, match="concept12n concept"):
        IB.concept12n(GreedyTokenizer(["ab"]), concepts={"topic": [["a"] * 7, ["b"] * 7]})
    with pytest.raises(ValueError):
        IB.load_concepts(_write(tmp_path / "x.json", b'{"topic": ["only one class"]}'), verify=False)
    copy = tmp_path / "repo" / "batteries"
    copy.mkdir(parents=True)
    shutil.copy(IB.CONCEPT_TASKS_PATH, copy / "concept_tasks.json")
    assert IB.load_concepts(tmp_path / "repo") == concepts                # a checkout directory works too


def _write(path, data):
    path.write_bytes(data)
    return path


@needs_repo
def test_concept_tasks_equal_the_icl_checkout():
    assert IB.load_concepts(ICL_REPO) == IB.load_concepts()


# ------------------------------------------------------------------------------------------------ 2./3. battery

def _lines(tok, item):
    lines = tok.decode(list(item.ids)).split("\n")
    return [ln.rsplit(" -> ", 1) for ln in lines[:-1]], lines[-1]


@pytest.mark.parametrize("seed,fingerprint", [(1, FP_SEED1), (2, FP_SEED2)])
def test_concept12n_with_the_qwen_tokenizer(qwen_tok, seed, fingerprint):
    items = IB.concept12n(qwen_tok, seed=seed)
    assert len(items) == 384 and [it.index for it in items] == list(range(384))
    kinds = collections.Counter(it.kind for it in items)
    assert len(kinds) == 16 and set(kinds.values()) == {24}
    assert {it.kind for it in items} == {f"icl_concept_{c}_{ls}_12n" for c in IB.CONCEPTS_12N for ls in IB.LABELSETS}
    lengths = [len(it.ids) for it in items]
    assert (min(lengths), max(lengths)) == (118, 181)
    assert sum(len(it.alt_ids) == 1 for it in items) == SINGLE_ALTS[seed]
    assert IB.battery_fingerprint(items) == fingerprint
    assert all(qwen_tok.decode(list(it.ids)).endswith(" ->") for it in items)
    assert all(not IB.target_ids(qwen_tok, list(it.ids), it.target)[1] for it in items[:32])   # no tokenisation fallback

    by_draw = collections.defaultdict(dict)
    for it in items:
        by_draw[(it.concept, it.seq)][it.labelset] = it
    assert len(by_draw) == 96
    clusters = collections.Counter(it.cluster for it in items)
    assert len(clusters) == 96 and set(clusters.values()) == {4}
    for (concept, seq), group in by_draw.items():
        assert set(group) == set(IB.LABELSETS)
        assert len({it.qpos for it in group.values()}) == 1 and len({it.cluster for it in group.values()}) == 1
        parsed = {ls: _lines(qwen_tok, it) for ls, it in group.items()}
        sentences = {ls: tuple(x for x, _ in demos) for ls, (demos, _) in parsed.items()}
        queries = {q for _, q in parsed.values()}
        assert len(set(sentences.values())) == 1 and len(queries) == 1         # same sentences, demo order and query
        assert len(sentences["sym"]) == 2 * IB.CONCEPT_PER
        for ls, (demos, _) in parsed.items():
            lp, ln = (IB.SYMBOL_LABELS.get(ls) or (IB.NATLAB[concept] if ls == "nat" else IB.NATLAB[concept][::-1]))
            assert collections.Counter(y for _, y in demos) == {lp: IB.CONCEPT_PER, ln: IB.CONCEPT_PER}
            it = group[ls]
            assert (it.target, it.alt) == ((lp, ln) if it.qpos else (ln, lp))
        assert group["nat"].target == group["flip"].alt and group["nat"].alt == group["flip"].target


def test_census_and_fresh_draws_differ_and_smoke(qwen_tok):
    a = IB.concept12n(qwen_tok, seed=IB.CENSUS_SEED)
    b = IB.concept12n(qwen_tok, seed=IB.FRESH_SEED)
    assert IB.battery_fingerprint(a) != IB.battery_fingerprint(b)
    assert [it.kind for it in a] == [it.kind for it in b]        # same kind order; different draws
    smoke = IB.concept12n(qwen_tok, seed=IB.FRESH_SEED, smoke=True)
    assert len(smoke) == 16 and IB.battery_fingerprint(smoke) == FP_SMOKE_SEED2
    assert len({it.kind for it in smoke}) == 16


def test_smoke_and_n_seq_with_the_test_tokenizer(tiny):
    _, tok, items = tiny
    smoke = IB.concept12n(tok, seed=2, smoke=True)                   # real data file, lossless test tokenizer
    assert len(smoke) == 16 and len({it.kind for it in smoke}) == 16
    assert all(tok.decode(list(it.ids)).endswith(" ->") for it in smoke)
    assert len(items) == 32 and set(collections.Counter(it.kind for it in items).values()) == {2}
    again = IB.concept12n(tok, seed=2, n_seq=2, concepts=short_concepts())
    assert IB.battery_fingerprint(again) == IB.battery_fingerprint(items)          # deterministic
    assert IB.battery_fingerprint(IB.concept12n(tok, seed=1, n_seq=2, concepts=short_concepts())) != \
        IB.battery_fingerprint(items)
    assert {len(it.alt_ids) == 1 for it in items} == {True, False}        # both alt paths are exercised
    with pytest.raises(ValueError):
        IB.concept12n(tok, n_seq=0)


# ------------------------------------------------------------------------------------------------ 4. mechanics

def test_batched_scores_equal_unbatched_reference(tiny):
    model, tok, items = tiny
    out = IB.score_items(model, items, tok.pad_token_id, batch_tokens=3000, max_batch=5, logit_chunk=7)
    assert out["n_rows"] == len(items) + sum(len(it.alt_ids) > 1 for it in items)
    target_rows = sum(len(it.ids) + len(it.target_ids) for it in items)
    alt_rows = sum(len(it.ids) + len(it.alt_ids) for it in items if len(it.alt_ids) > 1)
    assert out["tokens_processed"] == target_rows + alt_rows
    assert out["tokens_padded"] >= out["tokens_processed"] and out["seconds"] >= 0
    ref = np.array([_reference_margin(model, it) for it in items])
    np.testing.assert_allclose(out["tlp"], ref[:, 0], atol=2e-5, rtol=0)
    np.testing.assert_allclose(out["alt"], ref[:, 1], atol=2e-5, rtol=0)
    np.testing.assert_allclose(out["margin"], ref[:, 0] - ref[:, 1], atol=4e-5, rtol=0)
    multi = [n for n, it in enumerate(items) if len(it.alt_ids) > 1]
    assert multi and all(np.isfinite(out["alt"][multi]))
    # a different batch plan gives the same values
    other = IB.score_items(model, items, tok.pad_token_id, batch_tokens=100000, max_batch=64)
    np.testing.assert_allclose(other["margin"], out["margin"], atol=4e-5, rtol=0)


def test_empty_knock_is_identical_to_intact(tiny):
    model, tok, items = tiny
    intact = IB.score_items(model, items, tok.pad_token_id, **KW)
    for heads in ([], {}, {3: []}):
        empty = IB.score_items(model, items, tok.pad_token_id, heads=heads, kind_means=None, **KW)
        assert np.array_equal(empty["margin"], intact["margin"])
    assert IB.QueryKnock(model, []).layers == []


def test_knock_with_own_query_value_is_identity(tiny):
    model, tok, items = tiny
    own = [dataclasses.replace(it, kind=f"own{n:02d}") for n, it in enumerate(items)]
    means = IB.query_means(model, own, tok.pad_token_id, **KW)
    assert list(means) == [it.kind for it in own] and means["own00"].shape == (3, 64) and means["own00"].dtype == np.float32
    every = [(l, h) for l in range(3) for h in range(4)]
    intact = IB.score_items(model, own, tok.pad_token_id, **KW)
    knocked = IB.score_items(model, own, tok.pad_token_id, heads=every, kind_means=means, **KW)
    np.testing.assert_allclose(knocked["margin"], intact["margin"], atol=1e-5, rtol=0)
    # the real per-kind mean (2 prompts per kind) does move the margins
    kind_means = IB.query_means(model, items, tok.pad_token_id, **KW)
    moved = IB.score_items(model, items, tok.pad_token_id, heads=every, kind_means=kind_means, **KW)
    assert np.max(np.abs(moved["margin"] - intact["margin"])) > 1e-3


def test_batched_knock_equals_unbatched_knock(tiny):
    model, tok, items = tiny
    heads = {1: [0, 3], 2: [2]}
    means = IB.query_means(model, items, tok.pad_token_id, **KW)
    out = IB.score_items(model, items, tok.pad_token_id, heads=heads, kind_means=means, batch_tokens=3000, max_batch=5)
    knock = IB.QueryKnock(model, heads)
    ref = []
    with knock:
        for it in items:
            m = means[it.kind].reshape(3, 4, 16)
            ref.append(_reference_margin(model, it, knock, {l: torch.from_numpy(m[l, hs]) for l, hs in heads.items()}))
    ref = np.array(ref)
    np.testing.assert_allclose(out["tlp"], ref[:, 0], atol=2e-5, rtol=0)
    np.testing.assert_allclose(out["alt"], ref[:, 1], atol=2e-5, rtol=0)
    assert not knock.handles


def test_knock_changes_only_the_query_position_of_the_set_heads(tiny):
    model, tok, items = tiny
    rows = [items[0], items[5], items[13]]
    seqs = [list(it.ids) + list(it.target_ids) for it in rows]
    input_ids, mask = IB._pad(seqs, list(range(3)), tok.pad_token_id)
    q = torch.tensor([len(it.ids) - 1 for it in rows])
    heads = {1: [0, 2]}                              # one head of each KV group; heads 1 and 3 are their KV siblings
    ref = {1: 1000.0 + torch.arange(3 * 2 * 16, dtype=torch.float32).reshape(3, 2, 16)}

    clean = Capture(model, [0, 1, 2])
    with clean, torch.no_grad():
        model(input_ids=input_ids, attention_mask=mask)
    knock = IB.QueryKnock(model, heads).set_rows(q, ref)
    cap = Capture(model, [0, 1, 2])
    with knock, cap, torch.no_grad():                # knock registered first: the capture sees the edited input
        model(input_ids=input_ids, attention_mask=mask)
    assert torch.equal(cap.values[0], clean.values[0])
    a = clean.values[1].view(3, -1, 4, 16)
    b = cap.values[1].view(3, -1, 4, 16)
    changed = (a != b).any(-1)                       # (B, S, H)
    expected = torch.zeros_like(changed)
    for r in range(3):
        expected[r, int(q[r]), [0, 2]] = True
    assert torch.equal(changed, expected)
    for r in range(3):
        assert torch.equal(b[r, int(q[r]), [0, 2]], ref[1][r])
    assert not torch.equal(cap.values[2], clean.values[2])          # the edit propagates downstream


def test_query_means_equal_the_target_row_value_at_the_query(tiny):
    model, tok, items = tiny
    own = [dataclasses.replace(it, kind=f"own{n:02d}") for n, it in enumerate(items)]
    means = IB.query_means(model, own, tok.pad_token_id, batch_tokens=2500, max_batch=3)
    cap = Capture(model, [0, 1, 2])
    for it in own:
        with cap, torch.no_grad():
            model(torch.tensor([list(it.ids) + list(it.target_ids)]))
        got = torch.stack([cap.values[l][0, len(it.ids) - 1] for l in range(3)]).numpy()
        np.testing.assert_allclose(means[it.kind], got, atol=1e-5, rtol=0)
    pooled = IB.query_means(model, items, tok.pad_token_id, **KW)
    for kind in IB.kinds_of(items):
        members = [f"own{n:02d}" for n, it in enumerate(items) if it.kind == kind]
        np.testing.assert_allclose(pooled[kind], np.mean([means[m] for m in members], 0), atol=1e-6, rtol=0)


def test_query_knock_guards(tiny):
    model, tok, items = tiny
    with pytest.raises(ValueError, match="outside"):
        IB.QueryKnock(model, [(3, 0)])
    with pytest.raises(ValueError, match="outside"):
        IB.QueryKnock(model, {0: [4]})
    knock = IB.QueryKnock(model, [(1, 2), (1, 0), (1, 2)])
    assert knock.heads == {1: [0, 2]} and knock.layers == [1]
    with pytest.raises(ValueError, match="shape"):
        knock.set_rows(torch.tensor([3]), {1: torch.zeros(1, 1, 16)})
    with pytest.raises(ValueError, match="lack layer"):
        knock.set_rows(torch.tensor([3]), {})
    ids = torch.tensor([list(items[0].ids)])
    with knock, pytest.raises(RuntimeError, match="set_rows"):
        model(ids)
    knock.set_rows(torch.tensor([ids.shape[1]]), {1: torch.zeros(1, 2, 16)})
    with knock, pytest.raises(ValueError, match="does not fit"):
        model(ids)
    with pytest.raises(ValueError, match="kind_means"):
        IB.score_items(model, items[:2], tok.pad_token_id, heads=[(0, 0)])
    with pytest.raises(ValueError, match="lack the kinds"):
        IB.score_items(model, items[:2], tok.pad_token_id, heads=[(0, 0)], kind_means={})


# ------------------------------------------------------------------------------------------------ 5. summarize / effects

def _synthetic_items(concepts=("topic", "lang"), n_seq=3):
    out = []
    for concept in concepts:
        for seq in range(n_seq):
            for ls in IB.LABELSETS:
                out.append(IB.ICLItem(len(out), f"icl_concept_{concept}_{ls}_12n", concept, ls, seq, bool(seq % 2),
                                      (1, 2, 3), "a", "b", (4,), (5,), IB.CONCEPTS_12N.index(concept) * 100 + seq))
    return out


def test_summarize_and_effects_on_synthetic_margins():
    items = _synthetic_items()
    base = {"sym": 1.0, "rare": 2.0, "nat": 5.0, "flip": 0.5}
    m = np.array([base[it.labelset] + 0.1 * it.seq + (0.3 if it.concept == "lang" else 0.0) for it in items])
    s = IB.summarize(items, m)
    assert s["n"] == 24 and s["n_pairs"] == 6
    assert math.isclose(s["sym"], 1.0 + 0.1 + 0.15) and math.isclose(s["rare"], 2.25)
    assert math.isclose(s["symrare"], (s["sym"] + s["rare"]) / 2)
    assert math.isclose(s["nat"], 5.25) and math.isclose(s["flip"], 0.75)
    assert math.isclose(s["R"], (5.25 + 0.75) / 2) and math.isclose(s["P"], (5.0 - 0.5) / 2)

    delta = np.array([{"sym": -0.2, "rare": -0.4, "nat": -1.0, "flip": 0.6}[it.labelset] * (1 + it.seq) for it in items])
    eff = IB.effects(items, m, m + delta)
    dret, cl = eff["dret"]
    ret = [n for n, it in enumerate(items) if it.labelset in ("sym", "rare")]
    np.testing.assert_allclose(dret, delta[ret])
    assert list(cl) == [items[n].cluster for n in ret]
    dR, clR = eff["dR"]
    dP, clP = eff["dP"]
    assert list(clR) == list(clP) == [0, 1, 2, 100, 101, 102]
    k = np.array([1, 2, 3, 1, 2, 3])
    np.testing.assert_allclose(dR, (-1.0 + 0.6) / 2 * k)
    np.testing.assert_allclose(dP, (-1.0 - 0.6) / 2 * k)

    # pairing by meta, not by order: a shuffled battery gives the same pair effects and summary
    perm = np.random.default_rng(3).permutation(len(items))
    shuffled = [items[i] for i in perm]
    eff2 = IB.effects(shuffled, m[perm], (m + delta)[perm])
    np.testing.assert_allclose(eff2["dR"][0], dR)
    np.testing.assert_allclose(eff2["dP"][0], dP)
    assert list(eff2["dR"][1]) == list(clR)
    assert sorted(eff2["dret"][0]) == pytest.approx(sorted(dret))
    s2 = IB.summarize(shuffled, m[perm])
    assert all(math.isclose(s2[k_], s[k_]) for k_ in ("sym", "rare", "symrare", "nat", "flip", "R", "P"))

    with pytest.raises(ValueError, match="duplicate"):
        IB.summarize(items + [items[0]], np.append(m, 0.0))
    with pytest.raises(ValueError):
        IB.effects(items, m[:-1], m)
    lonely = [it for it in items if it.labelset != "flip"]                # nat without flip: no pairs
    s3 = IB.summarize(lonely, np.ones(len(lonely)))
    assert s3["n_pairs"] == 0 and math.isnan(s3["R"]) and math.isnan(s3["flip"])


# ------------------------------------------------------------------------------------------------ 6. census ranking

def _census(rows):
    rows = [dict(L=l, H=h, dret=d, dret_lo=lo, dret_hi=hi, dR=0.0, dP=p, dP_lo=plo, dP_hi=phi)
            for l, h, d, lo, hi, p, plo, phi in rows]
    roles = I.classify_census(rows)
    return I.CensusResult("synthetic", roles, frozenset((r["L"], r["H"]) for r in rows), (0, 1, 2), tuple(rows),
                          "test", "0" * 64, False)


def test_census_ranked_on_a_synthetic_census():
    c = _census([
        (2, 1, -0.5, -0.9, -0.1, 0.0, -0.1, 0.1),
        (0, 3, -0.5, -0.8, -0.2, 0.2, 0.1, 0.3),     # tie with 2:1 -> lower layer first
        (1, 0, -0.9, -1.2, -0.6, 0.0, -0.1, 0.1),
        (1, 1, -0.7, -0.9, 0.05, 0.0, -0.1, 0.1),    # CI contains 0: not a reader
        (0, 0, 0.8, 0.3, 1.2, 0.5, 0.2, 0.9),
        (2, 2, 0.8, 0.1, 1.5, 0.0, -0.1, 0.1),       # tie with 0:0 -> lower layer first
        (1, 2, 1.1, 0.4, 1.9, -0.3, -0.5, -0.1),
    ])
    assert IB.census_ranked(c, "readers", 20) == [(1, 0), (0, 3), (2, 1)]
    assert IB.census_ranked(c, "readers", 2) == [(1, 0), (0, 3)]
    assert IB.census_ranked(c, "brakes", 24) == [(1, 2), (0, 0), (2, 2)]
    assert IB.census_ranked(c, "brakes", None) == [(1, 2), (0, 0), (2, 2)]
    assert IB.census_ranked(c, "p_brakes", 5) == [(0, 0), (0, 3)]
    assert IB.census_ranked(c, "p_carriers", 5) == [(1, 2)]
    with pytest.raises(KeyError):
        IB.census_ranked(c, "r_readers")


@needs_repo
def test_census_ranked_equals_the_pinned_gate_sets():
    c4 = I.census_sets("qwen3-4b-base", source=ICL_REPO, allow_download=False)
    cr20 = IB.census_ranked(c4, "readers", 20)
    assert cr20 == _lh(CR20_4B)
    assert math.isclose(sum(c4.effects("dret")[lh] for lh in cr20), -2.547, abs_tol=5e-4)
    assert IB.census_ranked(c4, "brakes", 24) == _lh(CB24_4B)
    assert len(IB.census_ranked(c4, "readers", None)) == 89
    c06 = I.census_sets("qwen3-0.6b-base", source=ICL_REPO, allow_download=False)
    assert IB.census_ranked(c06, "readers", 20) == _lh(CR20_06B)
    assert len(IB.census_ranked(c06, "brakes", 24)) == 13          # fewer brakes than k: all of them


# ------------------------------------------------------------------------------------------------ calibration / targets

def test_calibration_check_and_targets():
    ref = IB.CALIBRATION["qwen3-4b-base"]
    ok = IB.calibration_check(dict(ref), "Qwen/Qwen3-4B-Base")
    assert ok["ok"] is True and all(v == 0 for v in ok["diffs"].values()) and ok["reference"] == "data/D188_summary.json"
    near = dict(ref, flip=ref["flip"] + 0.149)
    assert IB.calibration_check(near, "qwen3-4b-base")["ok"] is True
    far = dict(ref, P=ref["P"] - 0.151)
    res = IB.calibration_check(far, "qwen3-4b-base")
    assert res["ok"] is False and math.isclose(res["diffs"]["P"], -0.151)
    assert IB.calibration_check(dict(ref, nat=math.nan), "qwen3-4b-base")["ok"] is False
    skipped = IB.calibration_check(dict(ref), "Qwen/Qwen3-4B")
    assert skipped["ok"] is None and "skipped" in skipped
    assert IB.CALIBRATION["qwen3-4b-base"]["R"] == pytest.approx((ref["nat"] + ref["flip"]) / 2)
    assert IB.CALIBRATION["qwen3-4b-base"]["P"] == pytest.approx((ref["nat"] - ref["flip"]) / 2)
    reader20 = [t for t in IB.POSITIVE_CONTROL_TARGETS if t["quantity"].startswith("reader-20 joint knock")]
    assert reader20 and reader20[0]["value"] == -0.74 and reader20[0]["ci"] == [-0.95, -0.55]
    assert reader20[0]["random"]["value"] == 0.02 and reader20[0]["random"]["ci"] == [-0.03, 0.06]
    by_id = {t["id"]: t for t in IB.POSITIVE_CONTROL_TARGETS}
    assert len(by_id) == len(IB.POSITIVE_CONTROL_TARGETS)
    assert by_id["cr20_llama"]["value"] == -0.56 and by_id["cr20_llama"]["random"]["value"] == 0.0
    inst = by_id["instruct_removal"]                       # main.tex:605: 144-prompt symbol / rare battery, not concept12n
    assert inst["values"] == {"symbol": -0.61, "rare": -1.00} and inst["all_heads"] == {"symbol": -0.60, "rare": -1.00}
    assert "144 prompts" in inst["battery"] and "concept12n" not in inst["battery"]
    assert inst["source"] == "main.tex:470, :605"
