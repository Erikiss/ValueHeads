"""Head interventions for the causal ablation test (docs/ABLATION_DESIGN.md, part A).

Contents:
  * `Seq`, `Batch`, `edit_mask`, `run_sequences`: batched teacher-forced scoring with right padding and per-batch
    metadata (item ids, roles, boundaries) that the hooks below read. The runner calls the decoder, gathers the
    post-final-norm hidden states at the scored positions only and applies `lm_head` in chunks with an fp32 log-softmax.
  * Collectors (`LastTokenCollector`, `PooledCollector`, `ResidualCapture`): o_proj-input values at the last token,
    per-sequence means over a position policy, and the final residual (pre / post final norm) at the last token.
  * Interventions (`MaskedHeadEdit` and its subclasses `MeanReplace`, `DonorReplace`, `DirectionReplace`,
    `GaussReplace`, `OwnReplace`): replace the o_proj-input slice z_{l,h,t} of every set head at the positions of a
    policy, sequence-aware and right-padding safe. With grouped-query attention every query head keeps its own slice;
    editing head h never touches the other heads of its KV group.
  * Reference statistics (`compute_refs`, `RefStats`, `save_refs`, `load_refs`), the direct-effect helper for M5
    (`direct_effect`) and the hook self-test of the Stage-0 gate (`hook_selftest`).

Notation (ABLATION_DESIGN section 2): a sequence has length ln and boundary b (tokens [b, ln) are scored, token t from
position t - 1). R_i = [b - 1, ln - 2] are the scored ("decision") positions, E_i = [b - 1, ln - 1] the response edit
positions (F2). Prompt positions [0, b - 2] and padding are never edited by the "response" policy.

Hook order: pre-hooks run in registration order, so edit hooks must be installed before collectors (the runner installs
the `hooks` list in list order), and collectors then read the edited input.

`suspended()` is a module-level context manager under which every `MaskedHeadEdit` and every collector of this module is
a no-op. `clean_prepass` uses it for a truncated clean forward of the current batch (the clean head-output norms of
`GaussReplace`, the item's own clean values of `OwnReplace`).

Deviations from the spec (also reported to the orchestrator):
  * `run_sequences(score=False)` returns NaN sums (counts are still ln - b), so a collection pass can never be mistaken
    for a scoring pass.
  * `hook_selftest` check `causality_last` is bitwise (tolerance 0) in fp32 as specified, but uses the F4 tolerance
    1e-3 in half precision so that a GEMM-level nondeterminism on the GPU cannot fail the Stage-0 gate; the observed
    value is always reported.
  * Test 12 compares `direct_effect` on a 1-layer model with the "true" effect computed with the final-norm scale
    frozen at its clean value. This is the DE definition (frozen ln_final scale); with a recomputed norm the two differ
    by the norm's nonlinearity even without any indirect path.
  * `Seq` accepts len(ids) >= 1 (the spec's comment says >= 2); 1 <= boundary <= len(ids) is enforced.
  * The runner accumulates the per-sequence log-prob sums in float64; `eval_likelihood.loglik_batched` sums in float32.
    The equivalence test therefore uses atol 1e-5 plus rtol 1e-7 (float32 epsilon), since a 120-token fp32 sum alone
    is off by ~1e-5 absolute.
"""
import contextlib
import hashlib
import io
import json
import os
import time
import zipfile
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm.auto import tqdm

from .extract import plan_batches
from .hooks import HeadHooks
from .models import decoder_blocks, model_geometry, o_proj
from .overlap import _rng

ROLE_CODES = {"pos": 0, "neg": 1, "unrel": 2, "spos": 3, "sneg": 4, "ppl": 5}
POLICIES = ("response", "last", "all", "query")


# ----------------------------------------------------------------------------------------------------------------
# suspension (clean pre-passes)
# ----------------------------------------------------------------------------------------------------------------

_SUSPEND = [0]


@contextlib.contextmanager
def suspended():
    """Context manager under which every MaskedHeadEdit and every collector of this module is a no-op (re-entrant)."""
    _SUSPEND[0] += 1
    try:
        yield
    finally:
        _SUSPEND[0] -= 1


def is_suspended():
    return _SUSPEND[0] > 0


# ----------------------------------------------------------------------------------------------------------------
# sequences, masks, batches
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Seq:
    """One teacher-forced sequence: token ids without padding, boundary b (tokens [b, len) are scored), item, role."""
    ids: tuple
    boundary: int
    item: int
    role: str

    def __post_init__(self):
        ids = tuple(int(t) for t in self.ids)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "boundary", int(self.boundary))
        object.__setattr__(self, "item", int(self.item))
        if len(ids) < 1:
            raise ValueError("Seq needs at least one token")
        if not 1 <= self.boundary <= len(ids):
            raise ValueError(f"boundary {self.boundary} outside [1, {len(ids)}] (the first token can never be scored)")
        if self.role not in ROLE_CODES:
            raise ValueError(f"unknown role {self.role!r}; choose from {sorted(ROLE_CODES)}")


def edit_mask(policy, lengths, boundaries, width, device=None):
    """Bool mask (B, width) of the edited positions of `policy` for right-padded rows of the given lengths.

    response: b - 1 <= t <= ln - 1 (E_i, F2); last: t == ln - 1; all: t < ln; query: t == b - 1. Pads are always False.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")

    def col(x):
        if torch.is_tensor(x):
            return x.to(device=device if device is not None else x.device, dtype=torch.long).reshape(-1, 1)
        return torch.as_tensor(np.asarray(x), dtype=torch.long, device=device).reshape(-1, 1)

    ln, b = col(lengths), col(boundaries)
    if ln.shape != b.shape:
        raise ValueError(f"lengths {tuple(ln.shape)} and boundaries {tuple(b.shape)} differ in size")
    t = torch.arange(int(width), device=ln.device).unsqueeze(0)
    valid = t < ln
    if policy == "response":
        return (t >= b - 1) & valid
    if policy == "last":
        return t == ln - 1
    if policy == "all":
        return valid
    return (t == b - 1) & valid


@dataclass
class Batch:
    """A right-padded batch of `Seq`s plus the metadata the hooks read (set by the runner)."""
    index: np.ndarray
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor
    boundaries: torch.Tensor
    items: np.ndarray
    roles: list
    hidden: torch.Tensor = None
    _masks: dict = field(default_factory=dict, repr=False)

    def mask(self, policy):
        """Cached `edit_mask(policy, ...)` of this batch (on the batch device)."""
        if policy not in self._masks:
            self._masks[policy] = edit_mask(policy, self.lengths, self.boundaries, self.input_ids.shape[1],
                                            device=self.input_ids.device)
        return self._masks[policy]


class BatchHook:
    """Mixin: begin_batch(batch) before the forward, end_batch(batch) after it (both no-ops by default)."""

    def begin_batch(self, batch):
        pass

    def end_batch(self, batch):
        pass


def make_batch(seqs, idx, pad_id, device):
    """Batch of seqs[idx] (right padding with `pad_id`) on `device`."""
    idx = [int(i) for i in idx]
    lengths = [len(seqs[i].ids) for i in idx]
    width = max(lengths)
    input_ids = torch.full((len(idx), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for r, i in enumerate(idx):
        input_ids[r, :lengths[r]] = torch.as_tensor(seqs[i].ids, dtype=torch.long)
        mask[r, :lengths[r]] = 1
    return Batch(index=np.asarray(idx, dtype=np.int64), input_ids=input_ids.to(device), attention_mask=mask.to(device),
                 lengths=torch.as_tensor(lengths, dtype=torch.long, device=device),
                 boundaries=torch.as_tensor([seqs[i].boundary for i in idx], dtype=torch.long, device=device),
                 items=np.asarray([seqs[i].item for i in idx], dtype=np.int64), roles=[seqs[i].role for i in idx])


def iter_batches(seqs, pad_id, device, batch_tokens, max_batch):
    """Batches in the deterministic `extract.plan_batches` order (descending length, padded-width token budget)."""
    lengths = [len(s.ids) for s in seqs]
    for idx in plan_batches(lengths, max_batch, batch_tokens):
        yield make_batch(seqs, idx, pad_id, device)


def _decoder_forward(model, batch):
    """Post-final-norm hidden states (B, S, hidden) of the decoder for one batch (no lm_head)."""
    out = model.get_decoder()(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def scored_positions(batch):
    """(rows, cols, seg_rows) of the scored positions R_i = [b - 1, ln - 2] of every row (torch long, batch device).

    Position t predicts token t + 1; rows with b == ln have none."""
    lengths = batch.lengths.tolist()
    bounds = batch.boundaries.tolist()
    rows, cols = [], []
    for r, (ln, b) in enumerate(zip(lengths, bounds)):
        if b < ln:
            rows.append(np.full(ln - b, r, dtype=np.int64))
            cols.append(np.arange(b - 1, ln - 1, dtype=np.int64))
    dev = batch.input_ids.device
    if not rows:
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        return empty, empty
    return (torch.as_tensor(np.concatenate(rows), device=dev), torch.as_tensor(np.concatenate(cols), device=dev))


def _installable(h):
    return hasattr(h, "install") and hasattr(h, "remove")


@contextlib.contextmanager
def _installed(hooks):
    """Installs the installable `hooks` in list order (pre-hook order) and removes the ones installed here on exit."""
    mine = []
    try:
        for h in hooks:
            if not _installable(h) and not isinstance(h, BatchHook):
                raise TypeError(f"hook {type(h).__name__} has neither install()/remove() nor begin_batch()/end_batch()")
            if _installable(h) and not getattr(h, "handles", None):
                h.install()
                mine.append(h)
        yield
    finally:
        for h in reversed(mine):
            h.remove()


def _forward_batch(model, batch, hooks):
    """begin_batch -> decoder forward (hidden set on the batch) -> end_batch for already installed `hooks`."""
    bh = [h for h in hooks if isinstance(h, BatchHook)]
    started = []
    try:
        for h in bh:
            h.begin_batch(batch)
            started.append(h)
        batch.hidden = _decoder_forward(model, batch)
        return batch.hidden
    finally:
        for h in started:
            h.end_batch(batch)


# ----------------------------------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class RunResult:
    sums: np.ndarray          # float64 (N,): sum of log p over scored tokens [b, ln) (NaN with score=False)
    counts: np.ndarray        # int64 (N,): ln - b
    tokens_processed: int     # sum of unpadded lengths
    tokens_padded: int        # sum of B * S over batches
    tokens_scored: int
    seconds: float

    @property
    def tokens_per_second(self):
        return self.tokens_processed / self.seconds if self.seconds > 0 else float("nan")


def _check_logit_head(model):
    if getattr(model.config, "final_logit_softcapping", None):
        raise ValueError("final_logit_softcapping is not supported by the chunked lm_head scoring")


def _score_batch(lm_head, batch, hidden, logit_chunk):
    """Per-row float64 sums of log p(ids[t + 1]) over the scored positions of `batch` (fp32 log-softmax)."""
    rows, cols = scored_positions(batch)
    B = batch.input_ids.shape[0]
    sums = torch.zeros(B, dtype=torch.float64, device=hidden.device)
    if rows.numel() == 0:
        return sums.cpu().numpy()
    tgt = batch.input_ids[rows, cols + 1]
    for s in range(0, rows.numel(), int(logit_chunk)):
        e = s + int(logit_chunk)
        logits = lm_head(hidden[rows[s:e], cols[s:e]]).float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[s:e].unsqueeze(-1)).squeeze(-1)
        sums.index_add_(0, rows[s:e], lp.double())
    return sums.cpu().numpy()


@torch.no_grad()
def run_sequences(model, seqs, pad_id, hooks=(), *, batch_tokens=8192, max_batch=64, score=True, logit_chunk=2048,
                  progress=False, desc=None):
    """Teacher-forced forward of `seqs` (right padding, `extract.plan_batches`) under `hooks`; returns a RunResult.

    hooks: list; elements with install()/remove() are installed in list order (edit hooks first, collectors after) and
    removed on exit, BatchHook elements get begin_batch / end_batch around every forward. With score=False no lm_head
    is applied (pure collection pass) and `sums` is NaN.
    """
    seqs = list(seqs)
    hooks = list(hooks)
    if score:
        _check_logit_head(model)
    n = len(seqs)
    sums = np.full(n, np.nan) if not score else np.zeros(n)
    counts = np.array([len(s.ids) - s.boundary for s in seqs], dtype=np.int64)
    lengths = [len(s.ids) for s in seqs]
    batches = plan_batches(lengths, max_batch, batch_tokens) if n else []
    lm_head = model.get_output_embeddings() if score else None
    device = model.device
    was_training = model.training
    model.eval()
    t0 = time.time()
    tokens_padded = 0
    try:
        with _installed(hooks):
            for idx in tqdm(batches, desc=desc or "ablation", disable=not progress, leave=False):
                batch = make_batch(seqs, idx, pad_id, device)
                tokens_padded += int(batch.input_ids.numel())
                hidden = _forward_batch(model, batch, hooks)
                if score:
                    sums[batch.index] = _score_batch(lm_head, batch, hidden, logit_chunk)
                batch.hidden = None
    finally:
        model.train(was_training)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return RunResult(sums=sums, counts=counts, tokens_processed=int(sum(lengths)), tokens_padded=tokens_padded,
                     tokens_scored=int(counts.sum()), seconds=time.time() - t0)


# ----------------------------------------------------------------------------------------------------------------
# heads, norms
# ----------------------------------------------------------------------------------------------------------------

def heads_by_layer(heads, n_layers, n_heads):
    """{layer: sorted unique heads} from an iterable of (layer, head) or a {layer: [heads]} dict (empty -> {})."""
    if heads is None:
        return {}
    if isinstance(heads, dict):
        pairs = [(int(l), int(h)) for l, hs in heads.items() for h in hs]
    else:
        pairs = [(int(p[0]), int(p[1])) for p in heads]
    out = {}
    for l, h in pairs:
        if not 0 <= l < n_layers:
            raise ValueError(f"layer {l} out of range for {n_layers} layers")
        if not 0 <= h < n_heads:
            raise ValueError(f"head {h} out of range for {n_heads} heads")
        out.setdefault(l, set()).add(h)
    return {l: sorted(out[l]) for l in sorted(out)}


def final_norm(model):
    """The final norm before lm_head: Qwen3 / Llama `get_decoder().norm`, GPT-NeoX `gpt_neox.final_layer_norm`."""
    dec = model.get_decoder()
    for name in ("norm", "final_layer_norm", "ln_f"):
        mod = getattr(dec, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    raise AttributeError(f"no final norm found on decoder type {type(dec).__name__}")


def _rms_eps(norm):
    if isinstance(norm, torch.nn.LayerNorm):
        raise ValueError("direct_effect needs an RMSNorm final norm (Qwen3 / Llama); got LayerNorm")
    for name in ("variance_epsilon", "eps"):
        eps = getattr(norm, name, None)
        if eps is not None:
            return float(eps)
    raise ValueError(f"cannot read the epsilon of final norm {type(norm).__name__}")


def _ref_per_layer(ref, hb, n_layers, n_heads, head_dim, device):
    """{layer: fp32 tensor (n, D)} of the set heads from a (L, H*D) or (L*H, D) reference."""
    arr = ref.detach().float().cpu().numpy() if torch.is_tensor(ref) else np.asarray(ref, dtype=np.float32)
    if arr.shape not in ((n_layers, n_heads * head_dim), (n_layers * n_heads, head_dim)):
        raise ValueError(f"reference has shape {arr.shape}; expected ({n_layers}, {n_heads * head_dim}) or "
                         f"({n_layers * n_heads}, {head_dim})")
    arr = arr.reshape(n_layers, n_heads, head_dim)
    return {l: torch.as_tensor(np.ascontiguousarray(arr[l, hs]), dtype=torch.float32, device=device) for l, hs in hb.items()}


# ----------------------------------------------------------------------------------------------------------------
# collectors
# ----------------------------------------------------------------------------------------------------------------

class _BatchState:
    """Current batch bookkeeping shared by hooks."""

    _batch = None

    def _require_batch(self, what):
        if self._batch is None:
            raise RuntimeError(f"{type(self).__name__}: {what} needs begin_batch(batch) before the forward "
                               "(run it through ablation.run_sequences)")
        return self._batch


class LastTokenCollector(HeadHooks, BatchHook, _BatchState):
    """o_proj input at the last valid token ln - 1 of every sequence: values float32 (n_seqs, len(layers), H*D)."""

    def __init__(self, model, layers, n_seqs):
        L, _, _ = model_geometry(model.config)
        layers = list(range(L)) if layers is None else sorted(int(l) for l in layers)
        super().__init__(model, positions="all", decode_only=False, layers=layers)
        self._pos = {l: i for i, l in enumerate(self.layers)}
        self.values = np.zeros((int(n_seqs), len(self.layers), self.num_heads * self.head_dim), dtype=np.float32)
        self._batch = None

    def begin_batch(self, batch):
        self._batch = batch

    def end_batch(self, batch):
        self._batch = None

    def edit(self, layer, x):
        if is_suspended():
            return x
        b = self._require_batch("LastTokenCollector")
        rows = torch.arange(x.shape[0], device=x.device)
        self.values[b.index, self._pos[layer]] = x[rows, b.lengths - 1].float().cpu().numpy()
        return x


class PooledCollector(HeadHooks, BatchHook, _BatchState):
    """Per-sequence mean of the o_proj input over the positions of `policy` (computed in fp32 on the device, stored as
    `store_dtype`): values (n_seqs, len(layers), H*D). token_sum / token_count: per role, float64 sums over all policy
    tokens and their number (per-token pooling, diagnostic)."""

    def __init__(self, model, n_seqs, policy="response", layers=None, store_dtype=np.float16):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        L, _, _ = model_geometry(model.config)
        layers = list(range(L)) if layers is None else sorted(int(l) for l in layers)
        super().__init__(model, positions="all", decode_only=False, layers=layers)
        self.policy = policy
        self._pos = {l: i for i, l in enumerate(self.layers)}
        self.values = np.zeros((int(n_seqs), len(self.layers), self.num_heads * self.head_dim), dtype=store_dtype)
        self.token_sum, self.token_count = {}, {}
        self._batch = None
        self._role_rows = {}

    def begin_batch(self, batch):
        self._batch = batch
        cnt = batch.mask(self.policy).sum(1).cpu().numpy()
        self._role_rows = {}
        for r, role in enumerate(batch.roles):
            self._role_rows.setdefault(role, []).append(r)
        for role, rows in self._role_rows.items():
            self.token_count[role] = self.token_count.get(role, 0) + int(cnt[rows].sum())
            if role not in self.token_sum:
                self.token_sum[role] = np.zeros((len(self.layers), self.num_heads * self.head_dim), dtype=np.float64)

    def end_batch(self, batch):
        self._batch = None
        self._role_rows = {}

    def edit(self, layer, x):
        if is_suspended():
            return x
        b = self._require_batch("PooledCollector")
        m = b.mask(self.policy).to(torch.float32)
        sums = torch.bmm(m.unsqueeze(1), x.float()).squeeze(1)              # (B, H*D) fp32
        cnt = m.sum(1).clamp_min(1.0)
        li = self._pos[layer]
        self.values[b.index, li] = (sums / cnt[:, None]).cpu().numpy().astype(self.values.dtype)
        sums64 = sums.double().cpu().numpy()
        for role, rows in self._role_rows.items():
            self.token_sum[role][li] += sums64[rows].sum(0)
        return x


class ResidualCapture(BatchHook, _BatchState):
    """Final residual at ln - 1: pre_last = input of the final norm (pre-norm residual, M5 / F16), post_last =
    batch.hidden (post-norm state that lm_head reads, the M2 residual feature). Both float32 (n_seqs, hidden)."""

    def __init__(self, model, n_seqs, positions="last"):
        if positions != "last":
            raise ValueError("ResidualCapture supports positions='last' only")
        self.model = model
        self.positions = positions
        hidden = model.config.hidden_size
        self.pre_last = np.zeros((int(n_seqs), hidden), dtype=np.float32)
        self.post_last = np.zeros((int(n_seqs), hidden), dtype=np.float32)
        self.handles = []
        self._batch = None

    def _pre_hook(self, module, args):
        if is_suspended():
            return None
        b = self._require_batch("ResidualCapture")
        x = args[0]
        rows = torch.arange(x.shape[0], device=x.device)
        self.pre_last[b.index] = x[rows, b.lengths - 1].float().cpu().numpy()
        return None

    def install(self):
        if not self.handles:
            self.handles.append(final_norm(self.model).register_forward_pre_hook(self._pre_hook))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

    def begin_batch(self, batch):
        self._batch = batch

    def end_batch(self, batch):
        if batch.hidden is not None and not is_suspended():
            rows = torch.arange(batch.hidden.shape[0], device=batch.hidden.device)
            self.post_last[batch.index] = batch.hidden[rows, batch.lengths - 1].float().cpu().numpy()
        self._batch = None


class _SliceCapture(HeadHooks, BatchHook):
    """Internal: keeps the o_proj-input slices (B, S, n, D) of the given heads of the current batch (model dtype)."""

    def __init__(self, model, heads):
        L, H, _ = model_geometry(model.config)
        self.hb = heads_by_layer(heads, L, H)
        super().__init__(model, positions="all", decode_only=False, layers=list(self.hb))
        self.z = {}
        self._idx = {}

    def edit(self, layer, x):
        if is_suspended():
            return x
        if layer not in self._idx:
            self._idx[layer] = torch.as_tensor(self.hb[layer], device=x.device)
        B, S, _ = x.shape
        self.z[layer] = x.reshape(B, S, self.num_heads, self.head_dim)[:, :, self._idx[layer], :].detach().clone()
        return x


# ----------------------------------------------------------------------------------------------------------------
# clean pre-pass
# ----------------------------------------------------------------------------------------------------------------

class _StopForward(Exception):
    pass


@torch.no_grad()
def clean_prepass(model, batch, heads, what="z"):
    """Truncated clean forward of `batch`: {layer: (B, S, n, D) values (model dtype)} or, with what="norm",
    {layer: (B, S, n) fp32 L2 norms} of the set heads at every position.

    Runs under `suspended()` (no edits, no recording by this module's hooks); the forward is aborted by an exception
    raised in a forward hook on decoder block max(layer), so the upper layers and lm_head are skipped.
    """
    if what not in ("z", "norm"):
        raise ValueError("what must be 'z' or 'norm'")
    L, H, D = model_geometry(model.config)
    hb = heads_by_layer(heads, L, H)
    if not hb:
        return {}
    out, handles = {}, []

    def capture(layer, idx):
        def pre(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            B, S, _ = x.shape
            z = x.reshape(B, S, H, D)[:, :, idx.to(x.device), :]
            out[layer] = z.float().norm(dim=-1) if what == "norm" else z.detach().clone()
            return None
        return pre

    def stop(module, args, output):
        raise _StopForward()

    try:
        for layer, hs in hb.items():
            handles.append(o_proj(model, layer).register_forward_pre_hook(capture(layer, torch.as_tensor(hs)),
                                                                          with_kwargs=True))
        handles.append(decoder_blocks(model)[max(hb)].register_forward_hook(stop))
        with suspended():
            try:
                _decoder_forward(model, batch)
            except _StopForward:
                pass
    finally:
        for h in handles:
            h.remove()
    missing = set(hb) - set(out)
    if missing:
        raise RuntimeError(f"clean_prepass did not reach the o_proj of layers {sorted(missing)}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# interventions
# ----------------------------------------------------------------------------------------------------------------

class MaskedHeadEdit(HeadHooks, BatchHook, _BatchState):
    """Edits the o_proj-input slices of `heads` at the positions of `policy` (sequence-aware, right-padding safe).

    heads: iterable of (layer, head) or {layer: [heads]}; empty -> no layer hooked (identity). Only the set heads are
    written (every query head owns its slice, also under GQA); the edit clones x once per hooked layer and assigns
    through a (B, S, H, D) view with torch.where(mask, new, old). With policy="all" and no begin_batch the mask is
    "every position" (lets the hook run inside eval_likelihood.run_condition / loglik_batched); any other policy
    without begin_batch raises RuntimeError.
    """

    needs_batch = False      # subclasses that read batch metadata (items, roles, pre-pass) set True

    def __init__(self, model, heads, policy="response"):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
        L, H, _ = model_geometry(model.config)
        self.heads = heads_by_layer(heads, L, H)
        super().__init__(model, positions="all", decode_only=False, layers=list(self.heads))
        self.policy = policy
        self._batch = None
        self._mask = None
        self._idx = {}

    @property
    def n_heads(self):
        return sum(len(v) for v in self.heads.values())

    def begin_batch(self, batch):
        self._batch = batch
        self._mask = batch.mask(self.policy)

    def end_batch(self, batch):
        self._batch = None
        self._mask = None

    def _head_index(self, layer, device):
        idx = self._idx.get(layer)
        if idx is None or idx.device != device:
            idx = torch.as_tensor(self.heads[layer], dtype=torch.long, device=device)
            self._idx[layer] = idx
        return idx

    def edit(self, layer, x):
        if is_suspended() or layer not in self.heads:
            return x
        B, S, _ = x.shape
        if self._mask is None:
            if self.needs_batch or self.policy != "all":
                raise RuntimeError(f"{type(self).__name__}(policy={self.policy!r}) needs begin_batch(batch) before the "
                                   "forward (run it through ablation.run_sequences); only MeanReplace / DirectionReplace "
                                   "with policy='all' work without batch metadata")
            mask = None
        else:
            mask = self._mask
            if tuple(mask.shape) != (B, S):
                raise RuntimeError(f"batch mask {tuple(mask.shape)} does not match the o_proj input {(B, S)} of layer {layer}")
        idx = self._head_index(layer, x.device)
        out = x.clone(memory_format=torch.contiguous_format)
        view = out.view(B, S, self.num_heads, self.head_dim)
        z = view[:, :, idx, :]                                            # (B, S, n, D) copy of the current values
        new = self.replacement(layer, self.heads[layer], z, self._batch).to(dtype=z.dtype, device=z.device)
        if mask is None:
            view[:, :, idx, :] = new.expand_as(z)
        else:
            view[:, :, idx, :] = torch.where(mask[:, :, None, None], new, z)
        return out

    def replacement(self, layer, head_idx, z, batch):
        """New values for the set heads `head_idx` of `layer`; z: (B, S, n, D) current values. Returns a tensor
        broadcastable to z (only the policy positions are taken)."""
        raise NotImplementedError


class MeanReplace(MaskedHeadEdit):
    """z <- mu_{l,h} (fixed per-head reference). ref: (L, H*D) or (L*H, D) array (stored fp32, cast to the model dtype
    in the hook). Arms: mean (policy "response"), mean_last ("last"), M4 ("all")."""

    def __init__(self, model, heads, ref, policy="response"):
        super().__init__(model, heads, policy)
        self.ref = _ref_per_layer(ref, self.heads, self.num_layers, self.num_heads, self.head_dim, model.device)

    def replacement(self, layer, head_idx, z, batch):
        return self.ref[layer].to(device=z.device, dtype=z.dtype)[None, None]


class DonorReplace(MaskedHeadEdit):
    """z <- donor_means[donor_of[item]] (resample arm, F7): the label-neutral position-pooled response mean of the
    item's donor pair, the same for every sequence (role) of the item. donor_means: (n_donor, L, H*D), fp16 ok."""

    needs_batch = True

    def __init__(self, model, heads, donor_means, donor_of, policy="response"):
        super().__init__(model, heads, policy)
        dm = np.asarray(donor_means)
        if dm.ndim != 3 or dm.shape[1:] != (self.num_layers, self.num_heads * self.head_dim):
            raise ValueError(f"donor_means has shape {dm.shape}; expected (n_donor, {self.num_layers}, "
                             f"{self.num_heads * self.head_dim})")
        self.n_donor = dm.shape[0]
        self.donor_of = {int(k): int(v) for k, v in donor_of.items()}
        bad = [v for v in self.donor_of.values() if not 0 <= v < self.n_donor]
        if bad:
            raise ValueError(f"donor indices {bad[:5]} out of range for {self.n_donor} donors")
        view = dm.reshape(self.n_donor, self.num_layers, self.num_heads, self.head_dim)
        self.donor = {l: torch.as_tensor(np.ascontiguousarray(view[:, l, hs]), device=model.device)
                      for l, hs in self.heads.items()}                      # (n_donor, n, D), stored dtype
        self._rows = None

    def begin_batch(self, batch):
        super().begin_batch(batch)
        try:
            d = [self.donor_of[int(i)] for i in batch.items]
        except KeyError as e:
            raise KeyError(f"item {e.args[0]} has no donor in donor_of") from None
        self._rows = torch.as_tensor(d, dtype=torch.long, device=batch.input_ids.device)

    def end_batch(self, batch):
        super().end_batch(batch)
        self._rows = None

    def replacement(self, layer, head_idx, z, batch):
        return self.donor[layer].to(z.device)[self._rows].to(z.dtype)[:, None]        # (B, 1, n, D)


class DirectionReplace(MaskedHeadEdit):
    """Direction-level replacement (F8): z' = z - <z,v>v + <mu,v>v per set head, v = directions[(l, h)] (unit, (D,)).
    Computed in fp32 and cast to the model dtype."""

    def __init__(self, model, heads, ref, directions, policy="response"):
        super().__init__(model, heads, policy)
        mu = _ref_per_layer(ref, self.heads, self.num_layers, self.num_heads, self.head_dim, model.device)
        self.v, self.muv = {}, {}
        for l, hs in self.heads.items():
            rows = []
            for h in hs:
                if (l, h) not in directions:
                    raise KeyError(f"no direction for head ({l}, {h})")
                v = np.asarray(directions[(l, h)], dtype=np.float64).reshape(-1)
                if v.shape != (self.head_dim,):
                    raise ValueError(f"direction of ({l}, {h}) has shape {v.shape}; expected ({self.head_dim},)")
                nv = float(np.linalg.norm(v))
                if abs(nv - 1.0) > 1e-3:
                    raise ValueError(f"direction of ({l}, {h}) is not a unit vector (norm {nv:.6f})")
                rows.append(v / nv)
            v = torch.as_tensor(np.stack(rows), dtype=torch.float32, device=model.device)       # (n, D)
            self.v[l] = v
            self.muv[l] = (mu[l] * v).sum(-1)                                                  # (n,)

    def replacement(self, layer, head_idx, z, batch):
        v = self.v[layer].to(z.device)
        zf = z.float()
        coef = (zf * v).sum(-1, keepdim=True)
        return zf + (self.muv[layer].to(z.device)[:, None] - coef) * v


class GaussReplace(MaskedHeadEdit):
    """Norm-matched Gaussian replacement (F9): z' = g * ||z_clean|| / ||g|| with g = gauss_noise(seed, item, role, l, h)
    row t (absolute position) and the clean norm from a truncated clean pre-pass of the same batch (`clean_prepass`)."""

    needs_batch = True

    def __init__(self, model, heads, seed, policy="response"):
        super().__init__(model, heads, policy)
        self.seed = int(seed)
        self._new = {}

    def begin_batch(self, batch):
        super().begin_batch(batch)
        norms = clean_prepass(self.model, batch, self.heads, what="norm")
        B, S = batch.input_ids.shape
        lengths = batch.lengths.tolist()
        self._new = {}
        for l, hs in self.heads.items():
            g = np.zeros((B, S, len(hs), self.head_dim), dtype=np.float32)
            for r in range(B):
                for j, h in enumerate(hs):
                    g[r, :lengths[r], j] = gauss_noise(self.seed, int(batch.items[r]), batch.roles[r], l, h, lengths[r],
                                                       self.head_dim)
            gt = torch.as_tensor(g, device=batch.input_ids.device)
            gn = gt.norm(dim=-1).clamp_min(1e-30)
            self._new[l] = gt * (norms[l].float() / gn).unsqueeze(-1)

    def end_batch(self, batch):
        super().end_batch(batch)
        self._new = {}

    def replacement(self, layer, head_idx, z, batch):
        return self._new[layer]


class OwnReplace(MaskedHeadEdit):
    """Self-test only: replacement = the item's own clean z from a truncated clean pre-pass (must reproduce clean)."""

    needs_batch = True

    def __init__(self, model, heads, policy="response"):
        super().__init__(model, heads, policy)
        self._own = {}

    def begin_batch(self, batch):
        super().begin_batch(batch)
        self._own = clean_prepass(self.model, batch, self.heads, what="z")

    def end_batch(self, batch):
        super().end_batch(batch)
        self._own = {}

    def replacement(self, layer, head_idx, z, batch):
        return self._own[layer]


# ----------------------------------------------------------------------------------------------------------------
# seeded helpers
# ----------------------------------------------------------------------------------------------------------------

def _crc(s):
    return zlib.crc32(s.encode("utf-8"))


def donor_assignment(items, n_donor, seed):
    """{item: donor index} with donor(i) = default_rng([seed, crc32("donor"), item]).integers(n_donor) (F7)."""
    if int(n_donor) < 1:
        raise ValueError("need at least one donor")
    return {int(i): int(np.random.default_rng([int(seed), _crc("donor"), int(i)]).integers(int(n_donor)))
            for i in items}


def random_directions(heads, head_dim, seed, identity, set_name, index):
    """{(l, h): float32 unit vector (D,)} from overlap._rng(seed, "ablation-dir-v1", identity, set_name, index, l, h)
    (F8: direction `index` of head (l, h) of the named set)."""
    pairs = [(int(l), int(h)) for l, hs in heads.items() for h in hs] if isinstance(heads, dict) else \
        [(int(p[0]), int(p[1])) for p in heads]
    out = {}
    for l, h in pairs:
        v = _rng(seed, "ablation-dir-v1", identity, set_name, int(index), l, h).standard_normal(int(head_dim))
        out[(l, h)] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def gauss_noise(seed, item, role, layer, head, length, head_dim):
    """(length, D) float32 standard normals from default_rng([seed, crc32("gauss"), item, role_code, layer, head]);
    row t belongs to absolute position t (prefix-stable: row t does not depend on `length`)."""
    code = ROLE_CODES[role] if isinstance(role, str) else int(role)
    rng = np.random.default_rng([int(seed), _crc("gauss"), int(item), code, int(layer), int(head)])
    return rng.standard_normal((int(length), int(head_dim)), dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------------
# reference statistics
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class RefStats:
    mu_response: np.ndarray        # float32 (L, H*D), F3: 0.5 mean p+ + 0.5 mean p- (per-sequence pooling over E_i)
    mu_last: np.ndarray            # float32 (L, H*D), F4: 0.5 mean z+_{ln-1} + 0.5 mean z-_{ln-1}
    mu_response_tokens: np.ndarray # float32 (L, H*D), per-token pooled over all E_i tokens of both labels (diagnostic)
    dom: np.ndarray                # float32 (L*H, D): mean z+_{ln-1} - mean z-_{ln-1}
    donor_means: np.ndarray        # float16 (n_pairs, L, H*D): 0.5 * (p+_i + p-_i), F7
    items: np.ndarray              # int64 (n_pairs,)
    meta: dict


REF_ARRAYS = ("mu_response", "mu_last", "mu_response_tokens", "dom", "donor_means", "items")


@torch.no_grad()
def compute_refs(model, pos_seqs, neg_seqs, pad_id, *, batch_tokens=8192, max_batch=64, progress=False):
    """Reference statistics of the means_ref pairs (pos_seqs[i] / neg_seqs[i] = chosen / rejected of the same item):
    one collection pass with PooledCollector("response", fp32) + LastTokenCollector(all layers)."""
    pos_seqs, neg_seqs = list(pos_seqs), list(neg_seqs)
    if len(pos_seqs) != len(neg_seqs) or not pos_seqs:
        raise ValueError(f"need equally many (>0) chosen and rejected sequences, got {len(pos_seqs)} / {len(neg_seqs)}")
    for i, (p, q) in enumerate(zip(pos_seqs, neg_seqs)):
        if p.item != q.item:
            raise ValueError(f"pair {i}: chosen item {p.item} != rejected item {q.item}")
    n = len(pos_seqs)
    L, H, D = model_geometry(model.config)
    seqs = pos_seqs + neg_seqs
    pooled = PooledCollector(model, 2 * n, policy="response", store_dtype=np.float32)
    last = LastTokenCollector(model, None, 2 * n)
    run_sequences(model, seqs, pad_id, [pooled, last], batch_tokens=batch_tokens, max_batch=max_batch, score=False,
                  progress=progress, desc="refs")
    p_pos = pooled.values[:n].astype(np.float64)
    p_neg = pooled.values[n:].astype(np.float64)
    z_pos = last.values[:n].astype(np.float64)
    z_neg = last.values[n:].astype(np.float64)
    tok_sum = sum(pooled.token_sum.values())
    tok_cnt = sum(pooled.token_count.values())
    info = getattr(model, "vh_info", {}) or {}
    meta = {"model": info.get("model") or getattr(model.config, "name_or_path", None) or type(model).__name__,
            "dtype": str(model.dtype).replace("torch.", ""),
            "attn_impl": getattr(model.config, "_attn_implementation", None), "n_pairs": n, "policy": "response",
            "pooling": "per-sequence, 0.5 chosen + 0.5 rejected (F3)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return RefStats(
        mu_response=(0.5 * p_pos.mean(0) + 0.5 * p_neg.mean(0)).astype(np.float32),
        mu_last=(0.5 * z_pos.mean(0) + 0.5 * z_neg.mean(0)).astype(np.float32),
        mu_response_tokens=(tok_sum / max(1, tok_cnt)).astype(np.float32),
        dom=(z_pos.mean(0) - z_neg.mean(0)).reshape(L, H, D).reshape(L * H, D).astype(np.float32),
        donor_means=(0.5 * (p_pos + p_neg)).astype(np.float16),
        items=np.asarray([s.item for s in pos_seqs], dtype=np.int64),
        meta=meta,
    )


def dom_norms(dom):
    """(L*H,) float64 L2 norms of the per-head DoM vectors."""
    return np.linalg.norm(np.asarray(dom, dtype=np.float64), axis=-1)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_refs(path, refs):
    """Writes `refs` as an .npz (stored zip with fixed timestamps, so identical content gives identical bytes; meta as a
    JSON string entry, no pickle), atomically (tmp + os.replace). Returns the sha256 of the file."""
    path = str(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    arrays = {k: np.ascontiguousarray(getattr(refs, k)) for k in REF_ARRAYS}
    arrays["meta"] = np.array(json.dumps(refs.meta, sort_keys=True, default=str))
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                buf = io.BytesIO()
                np.lib.format.write_array(buf, arrays[name], allow_pickle=False)
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return _sha256_file(path)


def load_refs(path):
    """RefStats from a file written by `save_refs`."""
    with np.load(str(path), allow_pickle=False) as z:
        missing = [k for k in REF_ARRAYS + ("meta",) if k not in z.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        arrays = {k: z[k] for k in REF_ARRAYS}
        meta = json.loads(str(z["meta"][()]))
    return RefStats(meta=meta, **arrays)


# ----------------------------------------------------------------------------------------------------------------
# direct effect (M5)
# ----------------------------------------------------------------------------------------------------------------

class _NormInputRMS(BatchHook):
    """Internal: rms_t = sqrt(mean(r_t^2) + eps) (fp32) of the final-norm input r_t at every position."""

    def __init__(self, model, eps):
        self.model, self.eps = model, eps
        self.handles = []
        self.rms = None

    def _pre(self, module, args):
        if not is_suspended():
            x = args[0].float()
            self.rms = torch.sqrt(x.pow(2).mean(-1) + self.eps)
        return None

    def install(self):
        if not self.handles:
            self.handles.append(final_norm(self.model).register_forward_pre_hook(self._pre))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def direct_effect(model, seqs, pad_id, heads, ref, *, policy="response", batch_tokens=8192, max_batch=64,
                  logit_chunk=2048):
    """Direct (logit-attribution) effect of mean-replacing `heads`, with the final-norm scale frozen at its clean value.

    One clean forward per batch: z of the set heads, pre-norm residual r_t, rms_t = sqrt(mean(r_t^2) + eps) (fp32), clean
    logits l_t at the scored positions. For t in the policy positions:
      d_resid_t = sum_{(l,h) in set} (mu_{l,h} - z_{l,h,t}) @ W_O^{(l)}[:, slice_h].T
      d_logit_t = W_U (g * d_resid_t / rms_t)                         (g = final-norm weight)
      de_t = log_softmax(l_t + d_logit_t)[y_{t+1}] - log_softmax(l_t)[y_{t+1}]
    Returns {"de_sums": float64 (N,), "counts": int64 (N,), "clean_sums": float64 (N,)}. Raises if an o_proj has a bias
    (the formula assumes Qwen3's bias-free o_proj) or the final norm is not an RMSNorm.
    """
    seqs = list(seqs)
    _check_logit_head(model)
    L, H, D = model_geometry(model.config)
    hb = heads_by_layer(heads, L, H)
    for l in hb:
        if getattr(o_proj(model, l), "bias", None) is not None:
            raise ValueError(f"o_proj of layer {l} has a bias; direct_effect assumes a bias-free o_proj (Qwen3)")
    norm = final_norm(model)
    eps = _rms_eps(norm)
    g = norm.weight.detach().float()
    lm_head = model.get_output_embeddings()
    device = model.device
    mu = _ref_per_layer(ref, hb, L, H, D, device)
    w_o = {l: o_proj(model, l).weight.detach().float().view(-1, H, D)[:, hs, :] for l, hs in hb.items()}   # (hid, n, D)
    n = len(seqs)
    de_sums, clean_sums = np.zeros(n), np.zeros(n)
    counts = np.array([len(s.ids) - s.boundary for s in seqs], dtype=np.int64)
    cap = _SliceCapture(model, hb)
    rms_hook = _NormInputRMS(model, eps)
    was_training = model.training
    model.eval()
    try:
        with _installed([cap, rms_hook]):
            for batch in iter_batches(seqs, pad_id, device, batch_tokens, max_batch):
                hidden = _forward_batch(model, batch, [cap, rms_hook])
                rows, cols = scored_positions(batch)
                if rows.numel() == 0:
                    continue
                edited = batch.mask(policy)[rows, cols].float()                                       # (P,)
                d_resid = torch.zeros(rows.numel(), hidden.shape[-1], dtype=torch.float32, device=hidden.device)
                for l in hb:
                    z = cap.z[l][rows, cols].float()                                                  # (P, n, D)
                    delta = (mu[l][None] - z) * edited[:, None, None]
                    d_resid += torch.einsum("pnd,hnd->ph", delta, w_o[l])
                scaled = g * d_resid / rms_hook.rms[rows, cols][:, None]
                tgt = batch.input_ids[rows, cols + 1]
                B = batch.input_ids.shape[0]
                de_b = torch.zeros(B, dtype=torch.float64, device=hidden.device)
                cl_b = torch.zeros(B, dtype=torch.float64, device=hidden.device)
                for s in range(0, rows.numel(), int(logit_chunk)):
                    e = s + int(logit_chunk)
                    logits = lm_head(hidden[rows[s:e], cols[s:e]]).float()
                    d_logit = torch.nn.functional.linear(scaled[s:e].to(lm_head.weight.dtype), lm_head.weight).float()
                    t = tgt[s:e].unsqueeze(-1)
                    lp0 = torch.log_softmax(logits, -1).gather(-1, t).squeeze(-1)
                    lp1 = torch.log_softmax(logits + d_logit, -1).gather(-1, t).squeeze(-1)
                    de_b.index_add_(0, rows[s:e], (lp1 - lp0).double())
                    cl_b.index_add_(0, rows[s:e], lp0.double())
                de_sums[batch.index] = de_b.cpu().numpy()
                clean_sums[batch.index] = cl_b.cpu().numpy()
                batch.hidden = None
                cap.z = {}
    finally:
        model.train(was_training)
    return {"de_sums": de_sums, "counts": counts, "clean_sums": clean_sums}


# ----------------------------------------------------------------------------------------------------------------
# hook self-test (Stage-0 gate)
# ----------------------------------------------------------------------------------------------------------------

def _max_logit_diff(lm_head, batch, h0, h1, chunk=2048):
    """max |lm_head(h0) - lm_head(h1)| over all valid positions of `batch` (fp32)."""
    valid = batch.mask("all")
    rows, cols = valid.nonzero(as_tuple=True)
    out = 0.0
    for s in range(0, rows.numel(), chunk):
        e = s + chunk
        a = lm_head(h0[rows[s:e], cols[s:e]]).float()
        b = lm_head(h1[rows[s:e], cols[s:e]]).float()
        out = max(out, float((a - b).abs().max())) if a.numel() else out
    return out


def _check(value, tol, ok=None):
    value = float(value)
    return {"ok": bool(value <= tol if ok is None else ok), "value": value, "tol": float(tol)}


def _reference_positions(batch, policy="response"):
    """Expected edit positions of `policy`, built row by row from batch.lengths / batch.boundaries with explicit
    ranges (response: range(b - 1, ln); last: [ln - 1]). Deliberately independent of `edit_mask` / `Batch.mask`, which
    the interventions use, so the self-test can detect a wrong edit mask."""
    lengths = [int(x) for x in batch.lengths.tolist()]
    bounds = [int(x) for x in batch.boundaries.tolist()]
    out = torch.zeros(tuple(batch.input_ids.shape), dtype=torch.bool)
    for r, (ln, b) in enumerate(zip(lengths, bounds)):
        if policy == "response":
            cols = range(b - 1, ln)
        elif policy == "last":
            cols = [ln - 1]
        else:
            raise ValueError(f"no reference positions for policy {policy!r}")
        for t in cols:
            out[r, t] = True
    return out.to(batch.input_ids.device)


def _first_token_logprobs(lm_head, batch, hidden):
    """{row: log p(ids[b])} scored from position b - 1 for every row with b < ln (fp32 log-softmax)."""
    out = {}
    for r, (ln, b) in enumerate(zip(batch.lengths.tolist(), batch.boundaries.tolist())):
        if int(b) < int(ln):
            logits = lm_head(hidden[r, int(b) - 1]).float()
            out[r] = float(torch.log_softmax(logits, dim=-1)[int(batch.input_ids[r, int(b)])])
    return out


@torch.no_grad()
def hook_selftest(model, seqs, pad_id, heads, ref, *, batch_tokens=2048, max_batch=8):
    """Stage-0 hook self-test on `seqs` with the head set `heads` and reference `ref` (L, H*D).

    Checks: empty_set (MeanReplace with no heads == no hook, full logits, bitwise), own_reference (OwnReplace == clean),
    locality (sentinel on one head: the sentinel at every E_i position of that head, every other head untouched),
    prompt_untouched (that head's prompt positions [0, b - 2] and pads bitwise clean), first_token (the sentinel edit
    under policy "response" changes log p of the first response token, scored from b - 1, of every sequence),
    causality_last (policy "last" leaves the scored log-probs unchanged), dir_projection (<z',v> = <mu,v>, z' - z || v,
    nothing changed outside E_i), gauss_norm (||z'|| = ||z_clean||). The expected positions E_i = [b - 1, ln - 1] are
    rebuilt per row from the lengths and boundaries (`_reference_positions`), never taken from the edit mask itself.
    Returns {"ok", "dtype", "n_seqs", "checks": {name: {"ok", "value", "tol"}}}.
    """
    seqs = list(seqs)
    L, H, D = model_geometry(model.config)
    hb = heads_by_layer(heads, L, H)
    if not hb:
        raise ValueError("hook_selftest needs at least one head")
    if not seqs:
        raise ValueError("hook_selftest needs at least one sequence")
    _check_logit_head(model)
    half = model.dtype != torch.float32
    tol_own = 1e-3 if half else 1e-5
    tol_rel = 1e-2 if half else 1e-4
    tol_causal = 1e-3 if half else 0.0
    lm_head = model.get_output_embeddings()
    device = model.device
    batches = list(iter_batches(seqs, pad_id, device, batch_tokens, max_batch))
    checks = {}
    was_training = model.training
    model.eval()

    def fwd(batch, hooks):
        with _installed(hooks):
            return _forward_batch(model, batch, hooks)

    try:
        # empty_set and own_reference: full logits at every valid position
        d_empty = d_own = 0.0
        for batch in batches:
            h0 = fwd(batch, [])
            d_empty = max(d_empty, _max_logit_diff(lm_head, batch, h0, fwd(batch, [MeanReplace(model, [], ref)])))
            d_own = max(d_own, _max_logit_diff(lm_head, batch, h0, fwd(batch, [OwnReplace(model, hb)])))
        checks["empty_set"] = _check(d_empty, 0.0)
        checks["own_reference"] = _check(d_own, tol_own)

        # locality / prompt_untouched: sentinel reference on the first head; a capture after the edit sees the edited
        # o_proj input. Expected positions from the lengths and boundaries, not from the edit mask under test.
        l0 = min(hb)
        h0_ = hb[l0][0]
        sentinel = np.full((L, H * D), 1e3, dtype=np.float32)
        bad = bad_prompt = 0
        first_min, n_first = float("inf"), 0
        for batch in batches:
            cap_c = _SliceCapture(model, {l0: list(range(H))})
            h_clean = fwd(batch, [cap_c])
            lp_clean = _first_token_logprobs(lm_head, batch, h_clean)
            cap_e = _SliceCapture(model, {l0: list(range(H))})
            h_edit = fwd(batch, [MeanReplace(model, [(l0, h0_)], sentinel), cap_e])
            lp_edit = _first_token_logprobs(lm_head, batch, h_edit)
            E = _reference_positions(batch, "response")
            expected = cap_c.z[l0].clone()
            col = expected[:, :, h0_, :]
            col[E] = torch.tensor(1e3, dtype=expected.dtype, device=expected.device)
            expected[:, :, h0_, :] = col
            diff = cap_e.z[l0] != expected                                      # (B, S, H, D)
            outside = diff[:, :, h0_, :][~E]                                    # prompt [0, b - 2] and pads of h0
            bad_prompt += int(outside.sum())
            bad += int(diff.sum()) - int(outside.sum())
            for r, v in lp_clean.items():
                first_min = min(first_min, abs(lp_edit[r] - v))
                n_first += 1
        checks["locality"] = _check(bad, 0.0)
        checks["prompt_untouched"] = _check(bad_prompt, 0.0)
        # first_token: every sequence's first response token is scored from the edited position b - 1
        first_min = first_min if n_first else 0.0
        checks["first_token"] = _check(first_min, 0.0, ok=bool(n_first == 0 or first_min > 0.0))

        # causality_last: edits at ln - 1 cannot change any scored log-prob
        r0 = run_sequences(model, seqs, pad_id, [], batch_tokens=batch_tokens, max_batch=max_batch)
        r1 = run_sequences(model, seqs, pad_id, [MeanReplace(model, hb, ref, policy="last")],
                           batch_tokens=batch_tokens, max_batch=max_batch)
        checks["causality_last"] = _check(float(np.max(np.abs(r0.sums - r1.sums))) if len(seqs) else 0.0, tol_causal)

        # dir_projection: captures before and after the edit in the same forward
        dirs = random_directions(hb, D, 0, "selftest", "selftest", 0)
        mu = _ref_per_layer(ref, hb, L, H, D, device)
        err_dir = 0.0
        for batch in batches:
            before, after = _SliceCapture(model, hb), _SliceCapture(model, hb)
            edit = DirectionReplace(model, hb, ref, dirs)
            fwd(batch, [before, edit, after])
            E = _reference_positions(batch, "response")
            for l, hs in hb.items():
                zb, za = before.z[l].float(), after.z[l].float()
                if bool((before.z[l][~E] != after.z[l][~E]).any()):
                    err_dir = float("inf")
                zb, za = zb[E], za[E]                                                   # (P, n, D)
                v = edit.v[l]
                scale = zb.norm(dim=-1) + mu[l].norm(dim=-1)[None] + 1e-12
                e1 = ((za * v).sum(-1) - (mu[l] * v).sum(-1)[None]).abs() / scale
                d = za - zb
                perp = d - (d * v).sum(-1, keepdim=True) * v
                e2 = perp.norm(dim=-1) / scale
                if e1.numel():
                    err_dir = max(err_dir, float(e1.max()), float(e2.max()))
        checks["dir_projection"] = _check(err_dir, tol_rel)

        # gauss_norm: ||z'|| equals the clean norm at every E_i position
        err_g = 0.0
        for batch in batches:
            cap_c = _SliceCapture(model, hb)
            fwd(batch, [cap_c])
            cap_g = _SliceCapture(model, hb)
            fwd(batch, [GaussReplace(model, hb, seed=0), cap_g])
            E = _reference_positions(batch, "response")
            for l in hb:
                nc = cap_c.z[l].float().norm(dim=-1)[E]
                ng = cap_g.z[l].float().norm(dim=-1)[E]
                keep = nc > 1e-6
                if bool(keep.any()):
                    err_g = max(err_g, float((ng[keep] / nc[keep] - 1).abs().max()))
        checks["gauss_norm"] = _check(err_g, tol_rel)
    finally:
        model.train(was_training)
    return {"ok": all(c["ok"] for c in checks.values()), "dtype": str(model.dtype).replace("torch.", ""),
            "n_seqs": len(seqs), "checks": checks}
