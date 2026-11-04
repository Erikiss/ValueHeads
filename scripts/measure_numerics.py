"""Per-layer numerics of the selftest's batched-vs-single comparison for a Pythia model on CPU.

fp16/sdpa (the GPU policy dtype) and fp32/eager (reference). Reports per layer: activation scale (max|x|, ||x||),
batched-vs-single max-abs / max-rel / norm-rel, and the fp16-vs-fp32 error of the single text (the inherent fp16
noise level the GPU batch comparison is subject to). Usage: python scripts/measure_numerics.py MODEL REVISION  (writes numerics_MODEL_REVISION.json; docs/ holds the 2.8b run)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valueheads_pythia.commands.selftest import SHORT, TEXT, check_hook_layout  # noqa: E402
from valueheads_pythia.hooks import ActivationCollector  # noqa: E402
from valueheads_pythia.models import load_model  # noqa: E402

model_name, revision = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)


def collect(model, tok):
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        batch = tok([TEXT, SHORT], return_tensors="pt", padding=True)
    finally:
        tok.padding_side = old
    coll = ActivationCollector(model)
    with coll:
        long_ = coll.collect(tok(TEXT, return_tensors="pt").input_ids)[0]
        short = coll.collect(tok(SHORT, return_tensors="pt").input_ids)[0]
        both = coll.collect(batch.input_ids, batch.attention_mask)
    return long_.float(), short.float(), both.float()


def stats(a, ref):
    d = (a - ref)
    return {"maxabs": d.abs().amax(1), "maxrel": d.abs().amax(1) / ref.abs().amax(1).clamp(min=1e-6),
            "normrel": d.norm(dim=1) / ref.norm(dim=1).clamp(min=1e-6)}


out = {}
t = time.time()
model, tok = load_model(model_name, revision=revision, device="cpu", dtype="float16", attn_impl="sdpa")
print(f"loaded fp16 in {time.time() - t:.0f}s", flush=True)
res16 = check_hook_layout(model, tok)
print("selftest hook_layout fp16/sdpa CPU:", {k: v for k, v in res16.items() if "batch" in k or k in ("ok", "activation_scale_max")}, flush=True)
l16, s16, b16 = collect(model, tok)
del model
t = time.time()
model, tok = load_model(model_name, revision=revision, device="cpu", dtype="float32", attn_impl="eager")
print(f"loaded fp32 in {time.time() - t:.0f}s", flush=True)
l32, s32, b32 = collect(model, tok)

rows = []
L = l32.shape[0]
st_b16_l, st_b16_s = stats(b16[0], l16), stats(b16[1], s16)
st_b32_l, st_b32_s = stats(b32[0], l32), stats(b32[1], s32)
st_16v32_l, st_16v32_s = stats(l16, l32), stats(s16, s32)
for i in range(L):
    rows.append({"layer": i, "scale_long": float(l32[i].abs().max()), "scale_short": float(s32[i].abs().max()),
                 "norm_long": float(l32[i].norm()),
                 "b16_row0_maxabs": float(st_b16_l["maxabs"][i]), "b16_row1_maxabs": float(st_b16_s["maxabs"][i]),
                 "b16_row1_maxrel": float(st_b16_s["maxrel"][i]), "b16_row1_normrel": float(st_b16_s["normrel"][i]),
                 "b32_row1_maxabs": float(st_b32_s["maxabs"][i]),
                 "f16v32_long_maxabs": float(st_16v32_l["maxabs"][i]), "f16v32_long_maxrel": float(st_16v32_l["maxrel"][i]),
                 "f16v32_long_normrel": float(st_16v32_l["normrel"][i]),
                 "f16v32_short_maxrel": float(st_16v32_s["maxrel"][i]), "f16v32_short_normrel": float(st_16v32_s["normrel"][i])})
# wrong-token control: what a layout error (activation of a different position / token) looks like
wrong = stats(s32, l32)
out = {"model": model_name, "revision": revision, "layers": rows,
       "wrong_token_maxrel_min": float(wrong["maxrel"].min()), "wrong_token_normrel_min": float(wrong["normrel"].min()),
       "wrong_token_normrel_median": float(wrong["normrel"].median())}
json.dump(out, open(f"numerics_{model_name}_{revision}.json", "w"), indent=1)
print(f"{'L':>3} {'scale':>8} {'norm':>8} {'b16r1abs':>9} {'b16r1rel':>9} {'b16r1nrm':>9} {'16v32abs':>9} {'16v32rel':>9} {'16v32nrm':>9} {'16v32Srel':>9}")
for r in rows:
    print(f"{r['layer']:>3} {r['scale_long']:8.3f} {r['norm_long']:8.2f} {r['b16_row1_maxabs']:9.4f} {r['b16_row1_maxrel']:9.4f} "
          f"{r['b16_row1_normrel']:9.4f} {r['f16v32_long_maxabs']:9.4f} {r['f16v32_long_maxrel']:9.4f} {r['f16v32_long_normrel']:9.4f} "
          f"{r['f16v32_short_maxrel']:9.4f}")
print("wrong-token control: maxrel min", round(out["wrong_token_maxrel_min"], 3), "normrel min", round(out["wrong_token_normrel_min"], 3),
      "median", round(out["wrong_token_normrel_median"], 3))
