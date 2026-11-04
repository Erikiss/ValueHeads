"""Head-level read / write access via forward pre-hooks on each layer's attention output projection.

Adapted from sihuo-design/valueheads `hooks.py` (MIT). The input of `o_proj` (GPT-NeoX: `attention.dense`)
has shape (batch, seq, num_heads * head_dim) and is the concatenation of the per-head attention outputs,
so head h of a layer occupies the slice [h * head_dim, (h + 1) * head_dim).

Extensions over the reference implementation (DESIGN 3.1):
  * `positions="last"` edits only the last position of every row (paper semantics, generation path), `positions="all"`
    edits every position (teacher-forced likelihood evaluations).
  * "last" means column S - 1, which is the last real token only without padding or with LEFT padding (the
    generation path); for a RIGHT-padded batch set `hook.last_index` to the per-row index of the last valid token
    (attention_mask.sum(1) - 1) before the forward, and reset it to None before KV-cache decoding steps.
  * `decode_only=True` skips every call whose sequence length is not 1 (prompt forward unmasked, every
    KV-cache decoding step edited). It defaults to True for positions="last" and False for "all". A prompt of a
    single token is indistinguishable from a decoding step (eval_generate refuses it).
  * `ActivationCollector` is batched: right padding plus a gather of the last valid token per row from the
    attention mask.
"""
import torch

from .models import model_geometry, o_proj


class HeadHooks:
    """Base class: registers one pre-hook per layer; use as a context manager (or install()/remove())."""

    def __init__(self, model, positions="last", decode_only=None, layers=None):
        if positions not in ("last", "all"):
            raise ValueError("positions must be 'last' or 'all'")
        self.model = model
        self.num_layers, self.num_heads, self.head_dim = model_geometry(model.config)
        self.positions = positions
        self.decode_only = (positions == "last") if decode_only is None else bool(decode_only)
        self.layers = list(range(self.num_layers)) if layers is None else sorted(layers)
        self.handles = []
        self.last_index = None      # optional (B,) per-row last valid position for positions="last" under right padding

    def edit(self, layer, x):
        """Returns the (possibly modified) o_proj input `x` of `layer`, shape (B, S, H * D)."""
        raise NotImplementedError

    def last_positions(self, x):
        """(rows, cols) index tensors of the edited position per row for positions="last": `last_index` when set
        (right padding), else column S - 1 of every row (no padding / left padding / KV-cache decoding step)."""
        B, S = x.shape[0], x.shape[1]
        rows = torch.arange(B, device=x.device)
        if self.last_index is None:
            return rows, torch.full((B,), S - 1, device=x.device, dtype=torch.long)
        cols = torch.as_tensor(self.last_index, device=x.device, dtype=torch.long).reshape(-1)
        if cols.numel() != B or int(cols.max()) >= S or int(cols.min()) < 0:
            raise ValueError(f"last_index {cols.tolist()} does not fit a batch of shape {tuple(x.shape)} (reset it to None for decoding steps)")
        return rows, cols

    def _hook(self, layer):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            if self.decode_only and x.shape[1] != 1:
                return None
            if args:
                return (self.edit(layer, args[0]),) + tuple(args[1:]), kwargs
            kwargs["input"] = self.edit(layer, kwargs["input"])
            return args, kwargs
        return pre_hook

    def install(self):
        if self.handles:
            return self
        for layer in self.layers:
            self.handles.append(o_proj(self.model, layer).register_forward_pre_hook(self._hook(layer), with_kwargs=True))
        return self

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

    def head_slice(self, head):
        return slice(head * self.head_dim, (head + 1) * self.head_dim)


class ActivationCollector(HeadHooks):
    """Records the last-valid-token head activations of every layer for a (right-padded) batch.

    The gather index per row is attention_mask.sum(1) - 1 (right padding), or S - 1 without a mask.
    """

    def __init__(self, model, layers=None):
        super().__init__(model, positions="last", decode_only=False, layers=layers)
        self.states = [None] * self.num_layers
        self._last = None

    def edit(self, layer, x):
        last = self._last if self._last is not None else torch.full((x.shape[0],), x.shape[1] - 1, device=x.device)
        self.states[layer] = x[torch.arange(x.shape[0], device=x.device), last.to(x.device)].detach().float().cpu()
        return x

    @torch.no_grad()
    def collect(self, input_ids, attention_mask=None):
        """Returns a float32 CPU tensor (B, num_layers, num_heads * head_dim) of last-token head activations.

        With `layers` restricted at construction, the un-collected layers are zero.
        """
        device = self.model.device
        input_ids = input_ids.to(device)
        if input_ids.shape[1] == 0:
            raise ValueError("empty batch (0 tokens); every row needs at least one token")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if not bool((attention_mask[:, 0] == 1).all()):
                raise ValueError("ActivationCollector expects right padding with at least one valid token per row "
                                 "(attention_mask[:, 0] must be 1 everywhere; empty or left-padded rows are not supported)")
            self._last = attention_mask.sum(1) - 1
        else:
            self._last = None
        self.states = [None] * self.num_layers
        self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        width = self.num_heads * self.head_dim
        out = [s if s is not None else torch.zeros(input_ids.shape[0], width) for s in self.states]
        return torch.stack(out, dim=1)


class HeadMask(HeadHooks):
    """Ablates heads by scaling their output with a small factor `eps` (paper: x_mask = softmax(.) * eps * v).

    heads: {layer: [head, ...]}. positions/decode_only as in HeadHooks (defaults: last-token, decode-only =
    paper semantics; use positions="all" for teacher-forced likelihood evaluations).
    """

    def __init__(self, model, heads, eps=1e-4, positions="last", decode_only=None):
        super().__init__(model, positions=positions, decode_only=decode_only, layers=[int(l) for l in heads if heads[l]])
        self.heads = {int(l): sorted(int(h) for h in hs) for l, hs in heads.items() if hs}
        self.eps = eps
        self._scale = {}
        for layer, hs in self.heads.items():
            scale = torch.ones(self.num_heads * self.head_dim)
            for h in hs:
                if not 0 <= h < self.num_heads:
                    raise ValueError(f"head {h} out of range for {self.num_heads} heads")
                scale[self.head_slice(h)] = eps
            self._scale[layer] = scale

    def edit(self, layer, x):
        if layer not in self._scale:
            return x
        scale = self._scale[layer].to(x.device, x.dtype)
        x = x.clone()
        if self.positions == "last":
            rows, cols = self.last_positions(x)
            x[rows, cols] = x[rows, cols] * scale
        else:
            x = x * scale
        return x


class HeadSteer(HeadHooks):
    """Shifts head activations along value directions: x <- x + alpha * vec (vec already scaled by sigma).

    vectors: {layer: tensor(num_heads * head_dim)}, zeros for untouched heads (directions.steering_vectors).
    """

    def __init__(self, model, vectors, alpha=1.0, positions="last", decode_only=None):
        super().__init__(model, positions=positions, decode_only=decode_only, layers=[int(l) for l in vectors])
        self.vectors = {int(l): torch.as_tensor(v).reshape(-1) for l, v in vectors.items()}
        for layer, v in self.vectors.items():
            if v.numel() != self.num_heads * self.head_dim:
                raise ValueError(f"vector of layer {layer} has {v.numel()} entries, expected {self.num_heads * self.head_dim}")
        self.alpha = alpha

    def edit(self, layer, x):
        if layer not in self.vectors:
            return x
        vec = self.alpha * self.vectors[layer].to(x.device, x.dtype)
        x = x.clone()
        if self.positions == "last":
            rows, cols = self.last_positions(x)
            x[rows, cols] = x[rows, cols] + vec
        else:
            x += vec
        return x
