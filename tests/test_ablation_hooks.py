"""Part A of docs/ABLATION_DESIGN.md: head interventions, batched scoring, collectors, references, direct effect and the
hook self-test (`valueheads_pythia.ablation`). CPU only, fp32 eager, tiny random models (no download): a Qwen3 with
GQA (4 query heads / 2 KV heads) and head_dim decoupled from hidden (H * D = 64 != 48), and a GPT-NeoX. All texts are
synthetic ASCII strings.

`tiny_qwen3` and `CharTokenizer` are copied from tests/test_qwen_port.py (no cross-test imports)."""
import os

import numpy as np
import pytest
import torch
from transformers import BatchEncoding, GPTNeoXConfig, GPTNeoXForCausalLM, Qwen3Config, Qwen3ForCausalLM

from valueheads_pythia import ablation as A
from valueheads_pythia.eval_likelihood import encode_pair, loglik_batched
from valueheads_pythia.models import model_geometry, o_proj

VOCAB = 128


class CharTokenizer:
    """Minimal HF-like tokenizer (one id per character) for the tiny random models: __call__ with return_tensors /
    padding / add_special_tokens / truncation, padding_side, truncation_side, pad_token_id."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 0
    bos_token_id = None

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"

    def encode(self, text):
        return [1 + (ord(c) % (VOCAB - 1)) for c in text]

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t) for t in texts]
        if return_tensors is None:
            return {"input_ids": ids if not isinstance(text, str) else ids[0]}
        width = max(len(s) for s in ids)
        if not padding and any(len(s) != width for s in ids):
            raise ValueError("unequal lengths need padding=True")
        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, s in enumerate(ids):
            sl = slice(0, len(s)) if self.padding_side == "right" else slice(width - len(s), width)
            input_ids[r, sl] = torch.tensor(s)
            mask[r, sl] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": mask})


def tiny_qwen3(heads=4, kv_heads=2, head_dim=16, hidden=48, layers=3, bias=False, attn="eager", seed=0):
    """Random Qwen3 whose head_dim is decoupled from hidden (H * D = 64 != 48) and with GQA (2 query heads per KV head)."""
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, max_position_embeddings=256,
                      attention_bias=bias, tie_word_embeddings=True, pad_token_id=0, eos_token_id=0, bos_token_id=None)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():                    # random init gives near-zero attention outputs; make them clearly non-zero
        for block in model.model.layers:
            for lin in (block.self_attn.v_proj, block.self_attn.q_proj, block.self_attn.k_proj):
                lin.weight.normal_(0, 0.3)
                if lin.bias is not None:
                    lin.bias.normal_(0, 0.3)
    model.generation_config.pad_token_id = 0
    return model


def tiny_neox(seed=0):
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=32, num_attention_heads=4, intermediate_size=64, num_hidden_layers=2,
                        max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = GPTNeoXForCausalLM(cfg).eval()
    with torch.no_grad():
        for block in model.gpt_neox.layers:
            block.attention.query_key_value.weight.normal_(0, 0.3)
    return model


@pytest.fixture(scope="module")
def qwen():
    return tiny_qwen3()


@pytest.fixture(scope="module")
def tok():
    return CharTokenizer()


WORDS = ["alpha", "beta", "gamma", "delta", "omega", "kappa", "sigma", "theta", "lambda", "zeta"]


def synth_pairs(n, seed=0, pos_extra=0):
    """Synthetic HH-style (context, chosen, rejected) triples of mixed length (ASCII only)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = " ".join(rng.choice(WORDS, int(rng.integers(1, 6))))
        ctx = f"\n\nHuman: {q}?\n\nAssistant:"
        pos = " " + " ".join(rng.choice(WORDS, int(rng.integers(1, 5)) + pos_extra))
        neg = " " + " ".join(rng.choice(WORDS, int(rng.integers(1, 5))))
        out.append((ctx, pos, neg))
    return out


def pair_seqs(tok, triples, items=None, unrel=False):
    """(pos_seqs, neg_seqs[, unrel_seqs]) via eval_likelihood.encode_pair (no truncation)."""
    items = list(range(len(triples))) if items is None else items
    pos, neg, unr = [], [], []
    for k, (i, (ctx, p, q)) in enumerate(zip(items, triples)):
        for cont, role, lst in ((p, "pos", pos), (q, "neg", neg)):
            ids, b, _, cut = encode_pair(tok, ctx, cont, max_length=4096)
            assert cut == 0
            lst.append(A.Seq(tuple(ids), b, i, role))
        if unrel:
            ids, b, _, _ = encode_pair(tok, ctx, triples[(k + 1) % len(triples)][1], max_length=4096)
            unr.append(A.Seq(tuple(ids), b, i, "unrel"))
    return (pos, neg, unr) if unrel else (pos, neg)


def all_seqs(tok, n=6, seed=0):
    pos, neg = pair_seqs(tok, synth_pairs(n, seed))
    return pos + neg


def rand_ref(model, seed=1, scale=1.0):
    L, H, D = model_geometry(model.config)
    return (scale * np.random.default_rng(seed).standard_normal((L, H * D))).astype(np.float32)


class FullCapture:
    """Test helper: full o_proj input (B, S, H*D) per layer of the last forward, registered after any edit hooks."""

    def __init__(self, model, layers):
        self.model, self.layers, self.x, self.handles = model, list(layers), {}, []

    def install(self):
        for l in self.layers:
            def pre(module, args, kwargs, l=l):
                self.x[l] = (args[0] if args else kwargs["input"]).detach().clone()
            self.handles.append(o_proj(self.model, l).register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def capture_batch(model, batch, hooks, layers):
    """Runs one batch through the A machinery with `hooks` and a trailing FullCapture; returns {layer: (B,S,H*D)}."""
    cap = FullCapture(model, layers)
    with A._installed(list(hooks) + [cap]):
        A._forward_batch(model, batch, list(hooks))
    return cap.x


def n_hooks(model):
    L, _, _ = model_geometry(model.config)
    return sum(len(o_proj(model, l)._forward_pre_hooks) for l in range(L)) + len(A.final_norm(model)._forward_pre_hooks)


# ----------------------------------------------------------------------------------------------------------------
# 1. runner == eval_likelihood.loglik_batched
# ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["qwen3", "neox"])
def test_runner_equals_loglik_batched(arch, qwen, tok):
    model = qwen if arch == "qwen3" else tiny_neox()
    seqs = all_seqs(tok, 7) + [A.Seq((5, 6), 2, 99, "ppl"), A.Seq((5, 6, 7), 1, 98, "ppl")]   # b == ln (nothing scored), b == 1
    lengths = [len(s.ids) for s in seqs]
    assert len(set(lengths)) > 3
    rr = A.run_sequences(model, seqs, 0, batch_tokens=96, max_batch=3)
    sums, counts = loglik_batched(model, [list(s.ids) for s in seqs], [s.boundary for s in seqs], 0, 96, 3)
    # loglik_batched sums the token log-probs in float32, the runner in float64: rtol = float32 epsilon on long sums
    np.testing.assert_allclose(rr.sums, sums, atol=1e-5, rtol=1e-7)
    np.testing.assert_array_equal(rr.counts, counts)
    assert rr.sums.dtype == np.float64 and rr.counts.dtype == np.int64
    assert rr.tokens_processed == sum(lengths) and rr.tokens_scored == int(counts.sum())
    assert rr.tokens_padded >= rr.tokens_processed and rr.tokens_per_second > 0
    assert rr.sums[-2] == 0.0 and rr.counts[-2] == 0
    # score=False: no lm_head, NaN sums, same counts; collected values identical to the scoring pass
    c1 = A.LastTokenCollector(model, None, len(seqs))
    c2 = A.LastTokenCollector(model, None, len(seqs))
    A.run_sequences(model, seqs, 0, [c1], batch_tokens=96, max_batch=3, score=True)
    r2 = A.run_sequences(model, seqs, 0, [c2], batch_tokens=96, max_batch=3, score=False)
    assert np.isnan(r2.sums).all()
    np.testing.assert_array_equal(r2.counts, counts)
    np.testing.assert_array_equal(c1.values, c2.values)
    assert n_hooks(model) == 0 and not model.training


def test_runner_chunked_lm_head_and_training_mode(qwen, tok):
    seqs = all_seqs(tok, 5)
    r1 = A.run_sequences(qwen, seqs, 0, batch_tokens=4096, max_batch=64, logit_chunk=2048)
    r2 = A.run_sequences(qwen, seqs, 0, batch_tokens=4096, max_batch=64, logit_chunk=3)
    np.testing.assert_allclose(r1.sums, r2.sums, atol=1e-5, rtol=0)
    qwen.train()
    try:
        A.run_sequences(qwen, seqs[:2], 0)
        assert qwen.training
    finally:
        qwen.eval()


def test_seq_validation():
    with pytest.raises(ValueError):
        A.Seq((1, 2, 3), 0, 0, "pos")
    with pytest.raises(ValueError):
        A.Seq((1, 2, 3), 4, 0, "pos")
    with pytest.raises(ValueError):
        A.Seq((1, 2, 3), 1, 0, "chosen")
    s = A.Seq([1, 2, 3], 2, np.int64(7), "neg")
    assert s.ids == (1, 2, 3) and isinstance(s.item, int)


# ----------------------------------------------------------------------------------------------------------------
# 2. hook self-test
# ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arch", ["qwen3", "neox"])
def test_hook_selftest_passes(arch, qwen, tok):
    model = qwen if arch == "qwen3" else tiny_neox()
    seqs = all_seqs(tok, 4)
    heads = [(0, 1), (1, 2), (1, 3)] if arch == "qwen3" else [(0, 1), (1, 3)]
    res = A.hook_selftest(model, seqs, 0, heads, rand_ref(model), batch_tokens=128, max_batch=3)
    assert res["ok"], res
    assert set(res["checks"]) == {"empty_set", "own_reference", "locality", "prompt_untouched", "first_token",
                                  "causality_last", "dir_projection", "gauss_norm"}
    assert res["checks"]["first_token"]["value"] > 0 and res["checks"]["prompt_untouched"]["value"] == 0
    assert res["checks"]["empty_set"]["value"] == 0.0 and res["checks"]["empty_set"]["tol"] == 0.0
    assert res["checks"]["own_reference"]["value"] <= 1e-5
    assert res["checks"]["causality_last"]["value"] == 0.0
    assert res["dtype"] == "float32" and res["n_seqs"] == len(seqs)
    assert n_hooks(model) == 0


def test_hook_selftest_detects_broken_hooks(qwen, tok, monkeypatch):
    seqs = all_seqs(tok, 3)
    ref = rand_ref(qwen)
    orig = A.OwnReplace.replacement
    monkeypatch.setattr(A.OwnReplace, "replacement", lambda self, l, hs, z, b: orig(self, l, hs, z, b) + 1e-2)
    res = A.hook_selftest(qwen, seqs, 0, [(1, 2)], ref, batch_tokens=128, max_batch=3)
    assert not res["ok"] and not res["checks"]["own_reference"]["ok"]
    monkeypatch.undo()
    # a GQA bug that also writes the KV sibling (head 3 shares KV group 1 with head 2) breaks locality
    orig_idx = A.MaskedHeadEdit._head_index

    def leaky(self, layer, device):
        idx = orig_idx(self, layer, device)
        return torch.cat([idx, idx ^ 1])
    monkeypatch.setattr(A.MaskedHeadEdit, "_head_index", leaky)
    res = A.hook_selftest(qwen, seqs, 0, [(1, 2)], ref, batch_tokens=128, max_batch=3)
    assert not res["checks"]["locality"]["ok"] and res["checks"]["locality"]["value"] > 0


@pytest.mark.parametrize("bug", ["starts_at_b", "covers_prompt", "covers_pads"])
def test_hook_selftest_detects_a_wrong_edit_mask(bug, qwen, tok, monkeypatch):
    """The expected positions of the self-test are rebuilt from lengths / boundaries, so a wrong response mask in
    `edit_mask` (used by every intervention through Batch.mask) fails the gate instead of passing it."""
    seqs = all_seqs(tok, 4, seed=1)
    ref = rand_ref(qwen)
    heads = [(1, 0), (1, 3), (2, 1)]
    orig = A.edit_mask

    def buggy(policy, lengths, boundaries, width, device=None):
        if policy != "response":
            return orig(policy, lengths, boundaries, width, device)
        if bug == "starts_at_b":                                   # first response token's logit never edited
            return orig("response", lengths, torch.as_tensor(boundaries) + 1, width, device)
        if bug == "covers_prompt":                                 # prompt positions edited
            return orig("all", lengths, boundaries, width, device)
        t = torch.arange(int(width))[None]                         # right padding edited as well
        return t >= torch.as_tensor(boundaries).reshape(-1, 1) - 1

    monkeypatch.setattr(A, "edit_mask", buggy)
    res = A.hook_selftest(qwen, seqs, 0, heads, ref, batch_tokens=128, max_batch=3)
    failing = {k for k, v in res["checks"].items() if not v["ok"]}
    assert not res["ok"]
    if bug == "starts_at_b":
        assert {"locality", "first_token"} <= failing
        assert res["checks"]["first_token"]["value"] == 0.0         # log p of token b is bitwise unchanged
    else:
        assert "prompt_untouched" in failing and "dir_projection" in failing
    monkeypatch.undo()
    assert A.hook_selftest(qwen, seqs, 0, heads, ref, batch_tokens=128, max_batch=3)["ok"]


def test_hook_selftest_needs_heads(qwen, tok):
    with pytest.raises(ValueError):
        A.hook_selftest(qwen, all_seqs(tok, 1), 0, [], rand_ref(qwen))


# ----------------------------------------------------------------------------------------------------------------
# 3. edit_mask
# ----------------------------------------------------------------------------------------------------------------

def test_edit_mask_policies():
    lengths, bounds, width = [5, 3, 6, 1], [3, 1, 6, 1], 6
    exp = {
        "response": [[0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]],
        "last":     [[0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]],
        "all":      [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]],
        "query":    [[0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]],
    }
    for policy, m in exp.items():
        got = A.edit_mask(policy, lengths, bounds, width)
        assert got.dtype == torch.bool and got.shape == (4, 6)
        assert got.int().tolist() == m, policy
        assert A.edit_mask(policy, torch.tensor(lengths), torch.tensor(bounds), width).int().tolist() == m
    with pytest.raises(ValueError):
        A.edit_mask("prompt", lengths, bounds, width)
    batch = A.make_batch([A.Seq((1,) * 5, 3, 0, "pos"), A.Seq((1,) * 3, 1, 1, "neg")], [0, 1], 0, "cpu")
    assert batch.mask("response").int().tolist() == [row[:5] for row in exp["response"][:2]]     # width = longest row
    assert batch.mask("response") is batch.mask("response")          # cached


# ----------------------------------------------------------------------------------------------------------------
# 4. locality and GQA
# ----------------------------------------------------------------------------------------------------------------

def test_locality_gqa_siblings_untouched(qwen, tok):
    seqs = all_seqs(tok, 3)
    L, H, D = model_geometry(qwen.config)
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    clean = capture_batch(qwen, batch, [], range(L))
    sentinel = np.full((L, H * D), 7.0, dtype=np.float32)
    for h in range(H):
        edit = A.MeanReplace(qwen, [(1, h)], sentinel)
        assert edit.layers == [1]
        x = capture_batch(qwen, batch, [edit], range(L))
        E = batch.mask("response")
        assert torch.equal(x[0], clean[0])                                       # upstream layer untouched
        xv, cv = x[1].view(*x[1].shape[:2], H, D), clean[1].view(*x[1].shape[:2], H, D)
        for other in range(H):
            if other != h:                                                       # including the KV sibling h ^ 1
                assert torch.equal(xv[:, :, other], cv[:, :, other])
        assert torch.all(xv[:, :, h][E] == 7.0)
        assert torch.equal(xv[:, :, h][~E], cv[:, :, h][~E])                     # prompt and pad positions bitwise
    # empty set: nothing hooked at all
    empty = A.MeanReplace(qwen, [], sentinel)
    assert empty.layers == [] and empty.n_heads == 0
    with A._installed([empty]):
        assert n_hooks(qwen) == 0
    with pytest.raises(ValueError):
        A.MeanReplace(qwen, [(0, H)], sentinel)
    with pytest.raises(ValueError):
        A.MeanReplace(qwen, [(L, 0)], sentinel)
    with pytest.raises(ValueError):
        A.MeanReplace(qwen, [(0, 0)], sentinel[:, :-1])
    # (L*H, D) layout of the reference is the same as (L, H*D)
    ref = rand_ref(qwen, seed=12)
    a, b = A.MeanReplace(qwen, {1: [2], 2: [0, 3]}, ref), A.MeanReplace(qwen, {1: [2], 2: [3, 0]}, ref.reshape(L * H, D))
    assert all(torch.equal(a.ref[l], b.ref[l]) for l in (1, 2)) and a.heads == {1: [2], 2: [0, 3]}
    assert torch.equal(a.ref[2][1], torch.as_tensor(ref[2, 3 * D:4 * D]))


# ----------------------------------------------------------------------------------------------------------------
# 5. causality
# ----------------------------------------------------------------------------------------------------------------

def test_policy_last_cannot_change_scored_logprobs(qwen, tok):
    seqs = all_seqs(tok, 5)
    heads = [(0, 0), (1, 1), (2, 3)]
    big = rand_ref(qwen, scale=50.0)
    clean = A.run_sequences(qwen, seqs, 0, batch_tokens=96, max_batch=4)
    last = A.run_sequences(qwen, seqs, 0, [A.MeanReplace(qwen, heads, big, policy="last")], batch_tokens=96, max_batch=4)
    np.testing.assert_array_equal(last.sums, clean.sums)                     # bitwise
    resp = A.run_sequences(qwen, seqs, 0, [A.MeanReplace(qwen, heads, big)], batch_tokens=96, max_batch=4)
    assert np.abs(resp.sums - clean.sums).min() > 1e-4                       # every item has edited decision positions
    # the query policy (b - 1 only) edits the first decision position: changes sums, but not when nothing is scored
    q = A.run_sequences(qwen, seqs, 0, [A.MeanReplace(qwen, heads, big, policy="query")], batch_tokens=96, max_batch=4)
    assert np.abs(q.sums - clean.sums).min() > 0


def test_prompt_positions_untouched_by_response_policy(qwen, tok):
    """Editing at E_i leaves the clean hidden state at every prompt position [0, b - 2] (all layers) bitwise equal."""
    seqs = all_seqs(tok, 4)
    L, _, _ = model_geometry(qwen.config)
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    clean = capture_batch(qwen, batch, [], range(L))
    x = capture_batch(qwen, batch, [A.MeanReplace(qwen, [(0, 0), (0, 3), (1, 1)], rand_ref(qwen, scale=20))], range(L))
    prompt = batch.mask("all") & ~batch.mask("response")
    for l in range(L):
        assert torch.equal(x[l][prompt], clean[l][prompt])
    assert not torch.equal(x[2][batch.mask("response")], clean[2][batch.mask("response")])


# ----------------------------------------------------------------------------------------------------------------
# 6. compute_refs vs brute force
# ----------------------------------------------------------------------------------------------------------------

def brute_force_values(model, seqs):
    """Per sequence, unbatched: o_proj input (L, S, H*D) at every position."""
    L, _, _ = model_geometry(model.config)
    out = []
    for s in seqs:
        cap = FullCapture(model, range(L)).install()
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([s.ids]), use_cache=False)
        finally:
            cap.remove()
        out.append(np.stack([cap.x[l][0].double().numpy() for l in range(L)]))
    return out


def test_compute_refs_equals_brute_force(qwen, tok):
    # imbalanced token counts: the chosen responses are much longer, so per-token pooling != per-sequence 0.5 / 0.5
    triples = synth_pairs(5, seed=3, pos_extra=6)
    pos, neg = pair_seqs(tok, triples, items=[10, 11, 12, 13, 14])
    refs = A.compute_refs(qwen, pos, neg, 0, batch_tokens=128, max_batch=3)
    L, H, D = model_geometry(qwen.config)
    vp, vn = brute_force_values(qwen, pos), brute_force_values(qwen, neg)
    p_pos = np.stack([v[:, s.boundary - 1:len(s.ids)].mean(1) for v, s in zip(vp, pos)])     # (n, L, H*D)
    p_neg = np.stack([v[:, s.boundary - 1:len(s.ids)].mean(1) for v, s in zip(vn, neg)])
    z_pos = np.stack([v[:, -1] for v in vp])
    z_neg = np.stack([v[:, -1] for v in vn])
    mu_resp = 0.5 * p_pos.mean(0) + 0.5 * p_neg.mean(0)
    tok_all = np.concatenate([v[:, s.boundary - 1:].transpose(1, 0, 2) for v, s in zip(vp + vn, pos + neg)])
    np.testing.assert_allclose(refs.mu_response, mu_resp, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(refs.mu_response_tokens, tok_all.mean(0), atol=1e-5, rtol=1e-5)
    assert np.abs(mu_resp - tok_all.mean(0)).max() > 1e-3                   # the pooling choice matters here
    np.testing.assert_allclose(refs.mu_last, 0.5 * z_pos.mean(0) + 0.5 * z_neg.mean(0), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(refs.dom, (z_pos.mean(0) - z_neg.mean(0)).reshape(L * H, D), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(refs.donor_means.astype(np.float64), 0.5 * (p_pos + p_neg), atol=2e-3, rtol=2e-3)
    assert refs.donor_means.dtype == np.float16 and refs.mu_response.dtype == np.float32 and refs.dom.dtype == np.float32
    assert refs.items.tolist() == [10, 11, 12, 13, 14]
    assert refs.meta["n_pairs"] == 5 and refs.meta["policy"] == "response" and refs.meta["dtype"] == "float32"
    np.testing.assert_allclose(A.dom_norms(refs.dom), np.linalg.norm(refs.dom.astype(np.float64), axis=1))
    with pytest.raises(ValueError):
        A.compute_refs(qwen, pos, neg[::-1], 0)
    with pytest.raises(ValueError):
        A.compute_refs(qwen, pos, neg[:-1], 0)


def test_pooled_collector_token_sums_and_roles(qwen, tok):
    pos, neg = pair_seqs(tok, synth_pairs(4, seed=5))
    seqs = pos + neg
    col = A.PooledCollector(qwen, len(seqs), policy="response", store_dtype=np.float32)
    A.run_sequences(qwen, seqs, 0, [col], batch_tokens=64, max_batch=3, score=False)
    vals = brute_force_values(qwen, seqs)
    for role, group in (("pos", pos), ("neg", neg)):
        idx = [seqs.index(s) for s in group]
        exp = sum(vals[i][:, seqs[i].boundary - 1:].sum(1) for i in idx)
        np.testing.assert_allclose(col.token_sum[role], exp, atol=1e-4, rtol=1e-5)
        assert col.token_count[role] == sum(len(s.ids) - s.boundary + 1 for s in group)
    last = A.PooledCollector(qwen, len(seqs), policy="last", layers=[2], store_dtype=np.float32)
    A.run_sequences(qwen, seqs, 0, [last], score=False)
    np.testing.assert_allclose(last.values[:, 0], np.stack([v[2, -1] for v in vals]), atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# 7. DonorReplace
# ----------------------------------------------------------------------------------------------------------------

def test_donor_replace_same_donor_per_item_and_order_free(qwen, tok):
    L, H, D = model_geometry(qwen.config)
    triples = synth_pairs(4, seed=7)
    items = [3, 8, 21, 40]
    pos, neg, unr = pair_seqs(tok, triples, items=items, unrel=True)
    seqs = pos + neg + unr
    donor_means = np.random.default_rng(2).standard_normal((5, L, H * D)).astype(np.float16)
    donor_of = A.donor_assignment(items, 5, seed=42)
    assert donor_of == A.donor_assignment(items[::-1], 5, seed=42)
    assert donor_of[8] == int(np.random.default_rng([42, A._crc("donor"), 8]).integers(5))
    assert A.donor_assignment(range(50), 5, 42) != A.donor_assignment(range(50), 5, 43)
    heads = [(1, 0), (1, 3), (2, 2)]
    # the replaced value at E_i is the item's donor mean, for every role of the item
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    x = capture_batch(qwen, batch, [A.DonorReplace(qwen, heads, donor_means, donor_of)], [1, 2])
    E = batch.mask("response")
    dm = donor_means.astype(np.float32).reshape(5, L, H, D)
    for r in range(len(seqs)):
        d = donor_of[int(batch.items[r])]
        for l, h in heads:
            got = x[l][r].view(-1, H, D)[E[r], h].numpy()
            np.testing.assert_array_equal(got, np.broadcast_to(dm[d, l, h], got.shape))
    # batch order / composition does not change per-sequence results
    r1 = A.run_sequences(qwen, seqs, 0, [A.DonorReplace(qwen, heads, donor_means, donor_of)], batch_tokens=4096, max_batch=64)
    perm = np.random.default_rng(0).permutation(len(seqs))
    shuffled = [seqs[i] for i in perm]
    r2 = A.run_sequences(qwen, shuffled, 0, [A.DonorReplace(qwen, heads, donor_means, donor_of)], batch_tokens=40, max_batch=2)
    np.testing.assert_allclose(r2.sums, r1.sums[perm], atol=1e-5, rtol=0)
    with pytest.raises(KeyError):
        A.run_sequences(qwen, [A.Seq(seqs[0].ids, seqs[0].boundary, 999, "pos")], 0,
                        [A.DonorReplace(qwen, heads, donor_means, donor_of)])
    with pytest.raises(ValueError):
        A.DonorReplace(qwen, heads, donor_means[:, :2], donor_of)


# ----------------------------------------------------------------------------------------------------------------
# 8. GaussReplace
# ----------------------------------------------------------------------------------------------------------------

def test_gauss_noise_seeded_and_prefix_stable():
    a = A.gauss_noise(42, 7, "pos", 3, 1, 10, 16)
    assert a.shape == (10, 16) and a.dtype == np.float32
    np.testing.assert_array_equal(a, A.gauss_noise(42, 7, 0, 3, 1, 10, 16))
    np.testing.assert_array_equal(a[:4], A.gauss_noise(42, 7, "pos", 3, 1, 4, 16))      # row t = absolute position t
    for other in [(43, 7, "pos", 3, 1), (42, 8, "pos", 3, 1), (42, 7, "neg", 3, 1), (42, 7, "pos", 4, 1), (42, 7, "pos", 3, 2)]:
        assert not np.array_equal(a, A.gauss_noise(*other, 10, 16))


def test_gauss_replace_norm_determinism_and_batch_independence(qwen, tok):
    L, H, D = model_geometry(qwen.config)
    pos, neg = pair_seqs(tok, synth_pairs(4, seed=9), items=[5, 6, 7, 8])
    seqs = pos + neg
    heads = [(0, 2), (2, 1), (2, 3)]
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    clean = capture_batch(qwen, batch, [], [0, 2])
    edited = capture_batch(qwen, batch, [A.GaussReplace(qwen, heads, seed=11)], [0, 2])
    E = batch.mask("response")
    for r, s in enumerate(seqs):
        for l, h in heads:
            got = edited[l][r].view(-1, H, D)[:, h].double().numpy()
            zc = clean[l][r].view(-1, H, D)[:, h].double().numpy()
            g = A.gauss_noise(11, s.item, s.role, l, h, len(s.ids), D).astype(np.float64)
            exp = g * (np.linalg.norm(zc[:len(s.ids)], axis=1) / np.linalg.norm(g, axis=1))[:, None]
            e = E[r, :len(s.ids)].numpy()
            np.testing.assert_allclose(got[:len(s.ids)][e], exp[e], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(got[:len(s.ids)][e], axis=1), np.linalg.norm(zc[:len(s.ids)][e], axis=1),
                                       rtol=1e-5)
            prompt = np.arange(len(s.ids)) < s.boundary - 1                   # prompt positions stay clean
            np.testing.assert_array_equal(got[:len(s.ids)][prompt], zc[:len(s.ids)][prompt])
    # independent of batch composition: each sequence alone gives the same edited values (and sums)
    r_all = A.run_sequences(qwen, seqs, 0, [A.GaussReplace(qwen, heads, seed=11)], batch_tokens=4096, max_batch=64)
    r_one = A.run_sequences(qwen, seqs, 0, [A.GaussReplace(qwen, heads, seed=11)], batch_tokens=4096, max_batch=1)
    np.testing.assert_allclose(r_all.sums, r_one.sums, atol=1e-5, rtol=0)
    r_seed = A.run_sequences(qwen, seqs, 0, [A.GaussReplace(qwen, heads, seed=12)], batch_tokens=4096, max_batch=64)
    assert np.abs(r_seed.sums - r_all.sums).max() > 1e-4
    with pytest.raises(RuntimeError):                  # needs batch metadata even with policy="all"
        with A.GaussReplace(qwen, heads, seed=1, policy="all"):
            loglik_batched(qwen, [list(seqs[0].ids)], [seqs[0].boundary], 0)


# ----------------------------------------------------------------------------------------------------------------
# 9. DirectionReplace and random directions
# ----------------------------------------------------------------------------------------------------------------

def test_random_directions_unit_and_seeded():
    heads = {3: [1, 2], 5: [0]}
    d = A.random_directions(heads, 16, 42, "qwen3-4b-base", "C33", 0)
    assert set(d) == {(3, 1), (3, 2), (5, 0)}
    for v in d.values():
        assert v.dtype == np.float32 and v.shape == (16,) and abs(np.linalg.norm(v) - 1) < 1e-6
    same = A.random_directions([(3, 1), (3, 2), (5, 0)], 16, 42, "qwen3-4b-base", "C33", 0)
    assert all(np.array_equal(d[k], same[k]) for k in d)
    for args in [(43, "qwen3-4b-base", "C33", 0), (42, "qwen3-4b", "C33", 0), (42, "qwen3-4b-base", "O", 0),
                 (42, "qwen3-4b-base", "C33", 1)]:
        other = A.random_directions(heads, 16, *args)
        assert not np.array_equal(d[(3, 1)], other[(3, 1)])
    # F8 key: overlap._rng(seed, "ablation-dir-v1", identity, set, d, l, h)
    from valueheads_pythia.overlap import _rng
    raw = _rng(42, "ablation-dir-v1", "qwen3-4b-base", "C33", 0, 5, 0).standard_normal(16)
    np.testing.assert_allclose(d[(5, 0)], raw / np.linalg.norm(raw), atol=1e-7)


def test_direction_replace_projection_identities(qwen, tok):
    L, H, D = model_geometry(qwen.config)
    seqs = all_seqs(tok, 3)
    heads = [(1, 1), (1, 2)]
    dirs = A.random_directions(heads, D, 0, "t", "t", 0)
    ref = rand_ref(qwen, seed=4)
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    before = A._SliceCapture(qwen, heads)
    after = A._SliceCapture(qwen, heads)
    edit = A.DirectionReplace(qwen, heads, ref, dirs)
    with A._installed([before, edit, after]):
        A._forward_batch(qwen, batch, [before, edit, after])
    E = batch.mask("response")
    mu = ref.reshape(L, H, D)
    for j, h in enumerate([1, 2]):
        v = torch.as_tensor(dirs[(1, h)])
        zb, za = before.z[1][:, :, j], after.z[1][:, :, j]
        np.testing.assert_allclose((za[E] @ v).numpy(), np.full(int(E.sum()), float(mu[1, h] @ dirs[(1, h)])), atol=1e-5)
        diff = za[E] - zb[E]
        perp = diff - (diff @ v)[:, None] * v
        assert float(perp.abs().max()) < 1e-5
        assert torch.equal(za[~E], zb[~E])
    with pytest.raises(ValueError):
        A.DirectionReplace(qwen, heads, ref, {k: 2 * v for k, v in dirs.items()})
    with pytest.raises(KeyError):
        A.DirectionReplace(qwen, heads + [(2, 0)], ref, dirs)


# ----------------------------------------------------------------------------------------------------------------
# 10. MeanReplace(policy="all") without the runner
# ----------------------------------------------------------------------------------------------------------------

def test_mean_replace_all_inside_loglik_batched(qwen, tok):
    seqs = all_seqs(tok, 5)
    heads = [(0, 3), (2, 0)]
    ref = rand_ref(qwen, seed=6, scale=5)
    with A.MeanReplace(qwen, heads, ref, policy="all"):
        sums, counts = loglik_batched(qwen, [list(s.ids) for s in seqs], [s.boundary for s in seqs], 0, 96, 3)
    rr = A.run_sequences(qwen, seqs, 0, [A.MeanReplace(qwen, heads, ref, policy="all")], batch_tokens=96, max_batch=3)
    np.testing.assert_allclose(rr.sums, sums, atol=1e-5, rtol=1e-7)          # float32 vs float64 sums (see test 1)
    clean, _ = loglik_batched(qwen, [list(s.ids) for s in seqs], [s.boundary for s in seqs], 0, 96, 3)
    assert np.abs(clean - sums).min() > 1e-4
    for policy in ("response", "last", "query"):
        with pytest.raises(RuntimeError):
            with A.MeanReplace(qwen, heads, ref, policy=policy):
                loglik_batched(qwen, [list(seqs[0].ids)], [seqs[0].boundary], 0)
    assert n_hooks(qwen) == 0
    # run_condition's guard accepts the hook (positions="all", decode_only=False)
    h = A.MeanReplace(qwen, heads, ref, policy="all")
    assert h.positions == "all" and h.decode_only is False


# ----------------------------------------------------------------------------------------------------------------
# 11. clean_prepass and suspension
# ----------------------------------------------------------------------------------------------------------------

def test_clean_prepass_equals_full_forward_collectors(qwen, tok):
    seqs = all_seqs(tok, 4)
    L, H, D = model_geometry(qwen.config)
    heads = {0: [1], 1: [0, 3]}
    last = A.LastTokenCollector(qwen, [0, 1], len(seqs))
    pooled = A.PooledCollector(qwen, len(seqs), policy="response", layers=[0, 1], store_dtype=np.float32)
    A.run_sequences(qwen, seqs, 0, [last, pooled], batch_tokens=4096, max_batch=64, score=False)
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    # truncation: blocks above max(layer) never run
    calls = []
    hk = qwen.model.layers[2].register_forward_hook(lambda m, a, o: calls.append(1))
    try:
        z = A.clean_prepass(qwen, batch, heads, what="z")
        nrm = A.clean_prepass(qwen, batch, heads, what="norm")
    finally:
        hk.remove()
    assert calls == [] and set(z) == {0, 1} and z[1].shape == (len(seqs), batch.input_ids.shape[1], 2, D)
    rows = torch.arange(len(seqs))
    E = batch.mask("response").float()
    for li, l in enumerate([0, 1]):
        for j, h in enumerate(heads[l]):
            sl = slice(h * D, (h + 1) * D)
            np.testing.assert_allclose(z[l][rows, batch.lengths - 1, j].numpy(), last.values[batch.index, li, sl], atol=1e-6)
            pm = (z[l][:, :, j] * E[..., None]).sum(1) / E.sum(1, keepdim=True)
            np.testing.assert_allclose(pm.numpy(), pooled.values[batch.index, li, sl], atol=1e-5)
            np.testing.assert_allclose(nrm[l][:, :, j].numpy(), z[l][:, :, j].norm(dim=-1).numpy(), rtol=1e-6)
    assert A.clean_prepass(qwen, batch, [], what="z") == {}
    with pytest.raises(ValueError):
        A.clean_prepass(qwen, batch, heads, what="mean")


def test_prepass_is_clean_and_collectors_are_suspended(qwen, tok):
    seqs = all_seqs(tok, 3)
    L, H, D = model_geometry(qwen.config)
    batch = A.make_batch(seqs, list(range(len(seqs))), 0, "cpu")
    edit = A.MeanReplace(qwen, [(0, 1)], rand_ref(qwen, scale=30))
    pooled = A.PooledCollector(qwen, len(seqs), store_dtype=np.float32)
    lastc = A.LastTokenCollector(qwen, None, len(seqs))
    res = A.ResidualCapture(qwen, len(seqs))
    hooks = [edit, pooled, lastc, res]
    with A._installed(hooks):
        for h in hooks:
            h.begin_batch(batch)
        z = A.clean_prepass(qwen, batch, {1: [2]}, what="z")      # the edit at layer 0 is suspended
        for h in hooks:
            h.end_batch(batch)
    clean = capture_batch(qwen, batch, [], [1])
    assert torch.equal(z[1][:, :, 0], clean[1].view(*clean[1].shape[:2], H, D)[:, :, 2])
    assert not pooled.values.any() and not lastc.values.any() and not res.pre_last.any()
    assert all(not v.any() for v in pooled.token_sum.values())
    assert not A.is_suspended()
    with A.suspended():
        with A.suspended():
            assert A.is_suspended()
        assert A.is_suspended()
    assert not A.is_suspended()


def test_collectors_need_batch(qwen, tok):
    seqs = all_seqs(tok, 1)
    for hook in (A.LastTokenCollector(qwen, [0], 2), A.PooledCollector(qwen, 2, layers=[0]), A.ResidualCapture(qwen, 2),
                 A.OwnReplace(qwen, [(0, 0)])):
        with pytest.raises(RuntimeError):
            with hook:
                qwen(input_ids=torch.tensor([seqs[0].ids]), use_cache=False)
    assert n_hooks(qwen) == 0


# ----------------------------------------------------------------------------------------------------------------
# ResidualCapture (F16)
# ----------------------------------------------------------------------------------------------------------------

def test_residual_capture_pre_and_post_norm(qwen, tok):
    seqs = all_seqs(tok, 4)
    res = A.ResidualCapture(qwen, len(seqs))
    A.run_sequences(qwen, seqs, 0, [res], batch_tokens=64, max_batch=3)
    norm = A.final_norm(qwen)
    assert norm is qwen.model.norm
    for i, s in enumerate(seqs):
        with torch.no_grad():
            out = qwen(input_ids=torch.tensor([s.ids]), use_cache=False, output_hidden_states=True)
            post = out.hidden_states[-1][0, -1].numpy()
        np.testing.assert_allclose(res.post_last[i], post, atol=1e-5)
        np.testing.assert_allclose(norm(torch.tensor(res.pre_last[i])).detach().numpy(), res.post_last[i], atol=1e-5)
    with pytest.raises(ValueError):
        A.ResidualCapture(qwen, 2, positions="all")
    assert A.final_norm(tiny_neox()).__class__.__name__ == "LayerNorm"


# ----------------------------------------------------------------------------------------------------------------
# 12. direct_effect
# ----------------------------------------------------------------------------------------------------------------

def test_direct_effect_equals_true_effect_without_indirect_path(tok):
    model = tiny_qwen3(layers=1, seed=3)
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight.zero_()       # no MLP path: the head reaches the logits directly
    seqs = all_seqs(tok, 4)
    heads = [(0, 2)]
    ref = rand_ref(model, seed=8, scale=2)
    de = A.direct_effect(model, seqs, 0, heads, ref, batch_tokens=4096, max_batch=64)
    clean = A.run_sequences(model, seqs, 0)
    np.testing.assert_allclose(de["clean_sums"], clean.sums, atol=1e-5)
    np.testing.assert_array_equal(de["counts"], clean.counts)
    norm = A.final_norm(model)
    eps = norm.variance_epsilon
    for i, s in enumerate(seqs):                                 # true effect with the final-norm scale frozen at clean
        x = torch.tensor([s.ids])
        rms = {}
        h1 = norm.register_forward_pre_hook(lambda m, a: rms.__setitem__("v", torch.sqrt(a[0].pow(2).mean(-1, keepdim=True) + eps)))
        with torch.no_grad():
            model.model(input_ids=x, use_cache=False)
        h1.remove()
        h2 = norm.register_forward_hook(lambda m, a, o: m.weight * (a[0] / rms["v"]))
        try:
            te = A.run_sequences(model, [s], 0, [A.MeanReplace(model, heads, ref)]).sums[0] - \
                A.run_sequences(model, [s], 0).sums[0]
        finally:
            h2.remove()
        assert abs(te - de["de_sums"][i]) < 1e-5, (i, te, de["de_sums"][i])
    assert np.abs(de["de_sums"]).max() > 1e-3


def test_direct_effect_matches_numpy_formula(qwen, tok):
    L, H, D = model_geometry(qwen.config)
    seqs = all_seqs(tok, 3)
    heads = [(1, 1), (2, 2), (2, 3)]
    ref = rand_ref(qwen, seed=9, scale=2)
    de = A.direct_effect(qwen, seqs, 0, heads, ref, batch_tokens=64, max_batch=2, logit_chunk=5)
    norm = A.final_norm(qwen)
    g = norm.weight.detach().double().numpy()
    eps = norm.variance_epsilon
    W_U = qwen.get_output_embeddings().weight.detach().double().numpy()
    mu = ref.astype(np.float64).reshape(L, H, D)
    for i, s in enumerate(seqs):
        cap = FullCapture(qwen, range(L)).install()
        pre = {}
        hk = norm.register_forward_pre_hook(lambda m, a: pre.__setitem__("r", a[0].detach().clone()))
        try:
            with torch.no_grad():
                logits = qwen(input_ids=torch.tensor([s.ids]), use_cache=False).logits[0].double().numpy()
        finally:
            cap.remove()
            hk.remove()
        r = pre["r"][0].double().numpy()
        total = 0.0
        for t in range(s.boundary - 1, len(s.ids) - 1):
            d_resid = np.zeros(qwen.config.hidden_size)
            for l, h in heads:
                z = cap.x[l][0, t].double().numpy().reshape(H, D)[h]
                W_O = o_proj(qwen, l).weight.detach().double().numpy()[:, h * D:(h + 1) * D]
                d_resid += W_O @ (mu[l, h] - z)
            rms = np.sqrt(np.mean(r[t] ** 2) + eps)
            dl = W_U @ (g * d_resid / rms)
            y = s.ids[t + 1]

            def lsm(v):
                return v - (np.log(np.exp(v - v.max()).sum()) + v.max())
            total += lsm(logits[t] + dl)[y] - lsm(logits[t])[y]
        assert abs(total - de["de_sums"][i]) < 1e-5, (i, total, de["de_sums"][i])
    # policy "last": no scored position is edited, so the direct effect is zero
    de_last = A.direct_effect(qwen, seqs, 0, heads, ref, policy="last")
    assert np.all(de_last["de_sums"] == 0.0)
    with pytest.raises(ValueError):
        A.direct_effect(tiny_neox(), seqs, 0, [(0, 0)], np.zeros((2, 32), np.float32))   # o_proj bias / LayerNorm


# ----------------------------------------------------------------------------------------------------------------
# 13. save_refs / load_refs
# ----------------------------------------------------------------------------------------------------------------

def test_save_load_refs_round_trip(qwen, tok, tmp_path):
    pos, neg = pair_seqs(tok, synth_pairs(3, seed=11))
    refs = A.compute_refs(qwen, pos, neg, 0)
    path = tmp_path / "sub" / "refs.npz"
    sha = A.save_refs(path, refs)
    assert sorted(os.listdir(path.parent)) == ["refs.npz"]                      # atomic: no tmp file left
    assert sha == A._sha256_file(path)
    assert A.save_refs(path, refs) == sha                                         # stable bytes for identical content
    back = A.load_refs(path)
    for k in A.REF_ARRAYS:
        a, b = getattr(refs, k), getattr(back, k)
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), k
    assert back.meta == refs.meta
    with np.load(path, allow_pickle=False) as z:                                  # plain npz, no pickle needed
        assert set(z.files) == set(A.REF_ARRAYS) | {"meta"}
    refs.meta["n_pairs"] = 4
    assert A.save_refs(path, refs) != sha
