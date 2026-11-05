"""`ablate`: causal test of the value-head x ICL-circuit overlap on Qwen3-4B(-Base) (docs/ABLATION_DESIGN.md, part D).

Question: does the model RELY on the overlap heads O = C33 & topk(helpful) (and on the named ICL circuit C33) for its
own HH-helpful preference, or is "helpful" only DECODABLE there? Mean-ablation of the head sets at the response
positions, paired per item against the clean run of the same variant, compared with layer- and KV-group-matched random
head sets; primary family = M1 margin_mean under "mean" for C33 and O (Holm per model). A positive control on the ICL
battery (CR20 knocked at the query) must pass on the Base model before any helpfulness result is reported.

    python -m valueheads_pythia.cli ablate --model Qwen/Qwen3-4B-Base --stage 0          # inputs, split, self-test,
                                                                                       # refs, frozen sets, probes,
                                                                                       # floor, gate
    python -m valueheads_pythia.cli ablate --model Qwen/Qwen3-4B-Base --stage 1 --pilot  # 2 conditions -> ETA / CU
    python -m valueheads_pythia.cli ablate --model Qwen/Qwen3-4B-Base --stage 1          # primary (C33, O, 2 x 100 nulls)
    python -m valueheads_pythia.cli ablate --model Qwen/Qwen3-4B-Base --stage 2 --arms m3 mean_sets m4
    python -m valueheads_pythia.cli ablate --model Qwen/Qwen3-4B-Base --summary          # report only (no model)

Stages (8.4): 0 = check inputs, write / verify the HH split, hook self-test (exit 4), reference statistics
(refs.npz), M2 training features and probes, frozen head sets (written BEFORE the first EVAL forward; exit 5 on a
mismatch), clean__hh3 + floor check, positive-control gate (Base; the instruct run waits for the Base gate.json,
exit 3). 1 = the primary cells (requires the gate). 2 = secondary arms in the priority order m3 -> mean_sets -> m4 ->
mean_last -> m5 -> m7 -> resample -> dir_mean -> gauss -> fixed_identity.

Every condition is an idempotent file results/ablation/{short}/conditions/{name}.json (atomic; skipped when it exists
with the same key; recomputed on a key mismatch or with --force), so a dropped Colab session resumes where it stopped.
Exit codes: 0 ok, 1 error, 2 missing inputs or a changed --n_eval, 3 gate failed or missing, 4 self-test failed, 5 frozen
mismatch.

Nothing here prints or writes HH text: outputs hold ids, hashes and numbers only.
"""
import datetime as _dt
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .. import ablation as A
from .. import ablation_metrics as AM
from .. import ablation_report as AR
from .. import ablation_sets as AS
from .. import ablation_stats as ST
from .. import config, icl_sets, overlap
from ..cli import add_common_args
from ..eval_likelihood import condition_key

ARMS = ("m3", "mean_sets", "m4", "mean_last", "m5", "m7", "resample", "dir_mean", "gauss", "fixed_identity")
CU_PER_A100_HOUR = 6.0
COND_VERSION = 1
EXIT_OK, EXIT_ERROR, EXIT_INPUTS, EXIT_GATE, EXIT_SELFTEST, EXIT_FROZEN = 0, 1, 2, 3, 4, 5
POLICY = {"mean": "response", "mean_last": "last", "resample": "response", "dir_mean": "response", "gauss": "response",
          "dla": "response", "knock": "query"}
DEFAULTS = {"n_eval": 1000, "n_means_ref": 256, "n_means_probe": 256, "r_c33": 100, "r_o": 100, "r_small": 50,
            "r_sec": 20, "bootstrap": 10000, "ppl_tokens": 50000, "ppl_window": 1024, "ppl_stride": 512}
SMOKE = {"n_eval": 8, "n_means_ref": 8, "n_means_probe": 8, "r_c33": 2, "r_o": 2, "r_small": 2, "r_sec": 2,
         "bootstrap": 200, "ppl_tokens": 2048, "ppl_window": 256, "ppl_stride": 128}
SMOKE_ICL_NSEQ = 2          # see ablation_report.DESIGN_DEVIATIONS: 16-prompt smoke battery knocks are exact no-ops
STATIC_TOKS = 8000.0        # tok/s assumed before anything is measured (8.8)
ETA_FACTOR = {"gauss": 1.75, "dla": 1.2}
FIXED_IDENTITY_SETS = ("O", "CminusO", "P_band", "P_DoM", "TopK_all")
SELFTEST_PAIRS = 4
SELFTEST_VERSION = 2        # 2: positions rebuilt from lengths / boundaries, prompt_untouched + first_token checks


_KEY_RE = __import__("re").compile(r'^ "key": "([0-9a-f]+)",?$', __import__("re").M)
_NEVAL_RE = __import__("re").compile(r'^  "n_eval": (\d+),?$', __import__("re").M)      # inside "config"


class AblateExit(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = int(code)


def add_args(parser):
    add_common_args(parser, mode=False)
    parser.set_defaults(model="Qwen/Qwen3-4B-Base")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=None, help="stage (required unless --summary)")
    parser.add_argument("--pilot", action="store_true", help="stage 1: two conditions, ETA / CU, exit")
    parser.add_argument("--summary", action="store_true", help="write the report from the existing condition files only")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=None, help="stage-2 arms (default: all applicable)")
    parser.add_argument("--heads_table", default=None,
                        help="helpful probe table (default: results/heads/{short}_main_helpful_pythia.json)")
    parser.add_argument("--icl_model", default=None, help="identity of the ICL sets / gate (default: base model of --model)")
    parser.add_argument("--icl_source", default=None, help="census source (default: env VH_ICL_REPO, cache, pinned URL)")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--band", default="20-27", help="band of C33 / P_band / P_DoM / WholeBand")
    parser.add_argument("--readout_layers", default=None, help="M2 readout layers, e.g. 28-35 (default: the last 8)")
    parser.add_argument("--n_means_ref", type=int, default=None, help="means_ref prefix (default 256; smoke 8)")
    parser.add_argument("--n_means_probe", type=int, default=None, help="means_probe prefix (default 256; smoke 8)")
    parser.add_argument("--n_eval", type=int, default=None,
                        help="EVAL prefix (default 1000; smoke 8). Enters the key of every HH condition: a value other "
                             "than the stored one is refused (exit 2) unless --change_n_eval")
    parser.add_argument("--change_n_eval", action="store_true",
                        help="accept an --n_eval other than the stored one: every HH condition is recomputed when its "
                             "stage runs next (stale ones are left out of the report until then)")
    parser.add_argument("--r_c33", type=int, default=None, help="random sets for C33 (default 100; smoke 2)")
    parser.add_argument("--r_o", type=int, default=None, help="random sets for O (default 100; smoke 2)")
    parser.add_argument("--r_small", type=int, default=None, help="random sets for CminusO / P_band (default 50; smoke 2)")
    parser.add_argument("--r_sec", type=int, default=None, help="random sets of the secondary arms (default 20; smoke 2)")
    parser.add_argument("--threeway_all", action="store_true", help="every M1 condition in variant hh3 (F18)")
    parser.add_argument("--battery_seed", type=int, default=2, help="ICL battery draw (F21: fresh draw 2)")
    parser.add_argument("--calib_seed", type=int, default=1, help="ICL calibration draw (census draw 1)")
    parser.add_argument("--bootstrap", type=int, default=None, help="cluster-bootstrap B (default 10000; M2 min(B, 1000))")
    parser.add_argument("--batch_tokens", type=int, default=None, help="token budget per batch (CUDA 8192, CPU 2048)")
    parser.add_argument("--max_batch", type=int, default=None, help="rows per batch (CUDA 64, CPU 16)")
    parser.add_argument("--ppl_tokens", type=int, default=None, help="WikiText-2 tokens of M4 (default 50000)")
    parser.add_argument("--gate_wait_min", type=float, default=90.0, help="instruct: minutes to wait for the Base gate")
    parser.add_argument("--ignore_gate", action="store_true", help="debugging only; the report states 'Gate ignoriert'")
    parser.add_argument("--extend_random", action="store_true", help="allow R above the frozen R (appends draws)")
    parser.add_argument("--refreeze", action="store_true", help="overwrite the frozen file (refused once EVAL conditions exist)")
    parser.add_argument("--smoke", action="store_true", help="tiny sizes (F28), synthetic smoke-only table / C33, advisory gate")
    parser.add_argument("--note", default=None, help="line at the top of the report")
    parser.add_argument("--force", action="store_true", help="recompute existing conditions")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_progress", action="store_true")


def resolve_args(args):
    """Fills the size / budget defaults (smoke: F28) and the device-dependent batch plan (F19); idempotent."""
    table = SMOKE if args.smoke else DEFAULTS
    for k, v in table.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    from ..models import resolve_device
    args.device = resolve_device(args.device)
    cuda = str(args.device).startswith("cuda")
    if args.batch_tokens is None:
        args.batch_tokens = 8192 if cuda else 2048
    if args.max_batch is None:
        args.max_batch = 64 if cuda else 16
    if args.arms is None:
        args.arms = list(ARMS)
    return args


def resolve_icl_identity(icl_model, short):
    if icl_model:
        try:
            return icl_sets.resolve_icl_model(icl_model)[0]
        except KeyError:
            return config.model_short_name(icl_model)
    try:
        return icl_sets.resolve_icl_model(short)[0]
    except KeyError:
        return short


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(AR._clean_json(obj), f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Cond:
    """One cell of the condition grid (8.3)."""
    name: str
    arm: str
    variant: str
    set: str = None
    r: int = None
    d: int = None
    heads: tuple = ()
    stage: int = 0
    group: str = ""
    identity: str = None
    clean: str = None
    family: str = None
    randoms: list = field(default_factory=list)
    te: str = None
    te_clean: str = None

    @property
    def named(self):
        return self.arm != "clean" and self.r is None and self.d is None

    def spec(self):
        return {"name": self.name, "arm": self.arm, "variant": self.variant, "set": self.set, "clean": self.clean,
                "randoms": list(self.randoms), "stage": self.stage, "family": self.family,
                "kind": "named" if self.named else ("clean" if self.arm == "clean" else "random"),
                "te": self.te, "te_clean": self.te_clean}


def _clean(variant, stage, group):
    return Cond(f"clean__{variant}", "clean", variant, stage=stage, group=group)


# ----------------------------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------------------------

class Ctx:
    """Paths, inputs, geometry, sequence caches and the condition runner of one `ablate` invocation."""

    def __init__(self, args, model=None, tok=None, split=None, items_by_id=None, census=None, inject=None):
        import torch
        self.args = args
        self.inject = inject or {}
        self.P = config.paths(args.root)
        self.abl = self.P["ablation"]
        self.short, self.hf_id, self.reg = config.resolve_model(args.model)
        self.icl_identity = resolve_icl_identity(args.icl_model, self.short)
        self.is_base = self.short == self.icl_identity
        self.run_dir = os.path.join(self.abl, self.short)
        self.cond_dir = os.path.join(self.run_dir, "conditions")
        self.gate_path = os.path.join(self.abl, self.icl_identity, "gate.json")
        self.frozen_path = os.path.join(self.abl, f"{self.short}_frozen_sets.json")
        self.split_path = os.path.join(self.abl, "hh_helpful_split.json")
        self.smoke = bool(args.smoke)
        self.seed = int(args.seed)
        self.progress = not args.no_progress
        self._model = model if isinstance(model, torch.nn.Module) else None
        self._loader = model if (model is not None and self._model is None and callable(model)) else None
        self.tok = tok
        self.pad_id = getattr(tok, "pad_token_id", None) if tok is not None else None
        self.split = split
        self.items_by_id = items_by_id
        self.census = census
        self._cache = {}
        self._geometry()
        self.band = AS.parse_layers(args.band)
        self.band = (self.band[0], self.band[-1])
        self.readout = AS.parse_layers(args.readout_layers) if args.readout_layers else \
            tuple(range(max(0, self.L - 8), self.L))
        if any(not 0 <= l < self.L for l in self.readout):
            raise AblateExit(EXIT_INPUTS, f"readout layers {self.readout} outside the {self.L} layers")
        self._dtype_policy()
        self.commit_sha = None

    # ------------------------------------------------------------------------------------------------ setup
    def _geometry(self):
        if self._model is not None:
            cfg = self._model.config
            from ..models import model_geometry, num_kv_heads
            self.L, self.H, self.D = model_geometry(cfg)
            self.KV = num_kv_heads(cfg)
        elif os.path.exists(self.frozen_path):
            g = AS.load_frozen(self.frozen_path)["geometry"]
            self.L, self.H, self.D, self.KV = g["layers"], g["heads"], g["head_dim"], g["kv_heads"]
        elif self.reg and "layers" in self.reg:
            self.L, self.H, self.D = self.reg["layers"], self.reg["heads"], self.reg["head_dim"]
            self.KV = self.reg.get("kv_heads", self.H)
        else:
            from transformers import AutoConfig

            from ..models import model_geometry, num_kv_heads
            cfg = AutoConfig.from_pretrained(self.hf_id, revision=self.args.revision)
            self.L, self.H, self.D = model_geometry(cfg)
            self.KV = num_kv_heads(cfg)
        self.hpk = self.H // self.KV

    def _dtype_policy(self):
        if self._model is not None:
            self.dtype = str(self._model.dtype).replace("torch.", "")
            self.attn_impl = getattr(self._model.config, "_attn_implementation", None)
            return
        from ..models import dtype_name, to_dtype
        pol_dtype, pol_impl = config.dtype_policy(self.short, self.args.device)
        dtype = dtype_name(to_dtype(self.args.dtype) or to_dtype(pol_dtype))
        impl = self.args.attn_impl or (pol_impl if dtype == pol_dtype else
                                       config.DTYPE_POLICY["attn_impl_for_dtype"][dtype])
        self.dtype, self.attn_impl = dtype, impl

    @property
    def model(self):
        if self._model is None:
            if self._loader is None:
                raise AblateExit(EXIT_ERROR, "no model available (summary mode)")
            print(f"[ablate] lade {self.hf_id} ({self.dtype} / {self.attn_impl} auf {self.args.device}) ...", flush=True)
            t0 = time.time()
            self._model = self._loader()
            got = str(self._model.dtype).replace("torch.", "")
            if got != self.dtype:
                raise AblateExit(EXIT_ERROR, f"model loaded as {got}, the condition keys assume {self.dtype}")
            print(f"[ablate] Modell geladen in {time.time() - t0:.0f}s", flush=True)
        info = getattr(self._model, "vh_info", None) or {}
        self.commit_sha = info.get("commit_sha")
        return self._model

    def log(self, event, **kw):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "run_log.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.time(), "time": _now(), "event": event, **kw}, default=str) + "\n")

    def say(self, msg):
        print(f"[ablate] {msg}", flush=True)

    # ----------------------------------------------------------------------------------------------- items
    def role_items(self, role, n):
        key = ("items", role, n)
        if key not in self._cache:
            self._cache[key] = AS.split_items(self.split, self.items_by_id, role, n)
        return self._cache[key]

    @property
    def eval_items(self):
        return self.role_items("eval", self.args.n_eval)

    @property
    def eval_ids(self):
        return [int(it["group"]) for it in self.eval_items]

    @property
    def clusters(self):
        if "clusters" not in self._cache:
            self._cache["clusters"] = AS.cluster_ids(self.split, self.eval_ids)
        return self._cache["clusters"]

    def clusters_of(self, ids):
        return np.asarray([self.split["cluster"][str(int(i))] for i in ids])

    @property
    def partner(self):
        if "partner" not in self._cache:
            self._cache["partner"] = AM.derangement(self.clusters, self.seed)
        return self._cache["partner"]

    @property
    def enc(self):
        if "enc" not in self._cache:
            self._cache["enc"] = AM.Encoder(self.tok)
        return self._cache["enc"]

    def seqs(self, variant):
        key = ("seqs", variant)
        if key not in self._cache:
            if variant in AM.HH_VARIANTS:
                self._cache[key] = AM.hh_sequences(self.enc, self.eval_items, variant, self.partner)
            elif variant == "ppl":
                ids = self.inject.get("ppl_ids")
                if ids is None:
                    from ..data import wikitext
                    ids = wikitext(self.tok, self.args.ppl_tokens)
                ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)][:self.args.ppl_tokens]
                windows = AM.ppl_windows(len(ids), self.args.ppl_window, self.args.ppl_stride)
                self._cache[key] = AM.ppl_sequences(ids, windows)
            else:
                raise ValueError(variant)
        return self._cache[key]

    def icl_items(self, variant):
        key = ("icl", variant)
        if key not in self._cache:
            seed = self.args.battery_seed if variant == "icl" else self.args.calib_seed
            builder = self.inject.get("icl_items")
            if builder is not None:
                items = builder(self.tok, seed, self.smoke and variant == "icl")
            else:
                from .. import icl_battery as IB
                items = IB.concept12n(self.tok, seed=seed, smoke=False,
                                      n_seq=SMOKE_ICL_NSEQ if (self.smoke and variant == "icl") else None)
            self._cache[key] = items
        return self._cache[key]

    def fingerprint(self, variant):
        key = ("fp", variant)
        if key not in self._cache:
            from .. import icl_battery as IB
            self._cache[key] = IB.battery_fingerprint(self.icl_items(variant))
        return self._cache[key]

    # ------------------------------------------------------------------------------------- artefacts
    @property
    def refs(self):
        if "refs" not in self._cache:
            path = os.path.join(self.run_dir, "refs.npz")
            if not os.path.exists(path):
                raise AblateExit(EXIT_INPUTS, f"{path} fehlt: zuerst Stufe 0")
            self._cache["refs"] = A.load_refs(path)
            self._cache["refs_sha"] = overlap.file_sha256(path)
        return self._cache["refs"]

    @property
    def refs_sha(self):
        _ = self.refs
        return self._cache["refs_sha"]

    @property
    def probes(self):
        if "probes" not in self._cache:
            path = os.path.join(self.run_dir, "probes_m2.npz")
            if not os.path.exists(path):
                raise AblateExit(EXIT_INPUTS, f"{path} fehlt: zuerst Stufe 0")
            self._cache["probes"] = AM.load_npz(path)[0]
            self._cache["probes_sha"] = overlap.file_sha256(path)
        return self._cache["probes"]

    @property
    def probes_sha(self):
        _ = self.probes
        return self._cache["probes_sha"]

    @property
    def frozen(self):
        if "frozen" not in self._cache:
            if not os.path.exists(self.frozen_path):
                raise AblateExit(EXIT_INPUTS, f"{self.frozen_path} fehlt: zuerst Stufe 0")
            self._cache["frozen"] = AS.load_frozen(self.frozen_path)
        return self._cache["frozen"]

    @property
    def split_sha(self):
        if "split_sha" not in self._cache:
            self._cache["split_sha"] = overlap.file_sha256(self.split_path) if os.path.exists(self.split_path) else None
        return self._cache["split_sha"]

    def kind_means(self):
        if "kind_means" not in self._cache:
            path = os.path.join(self.run_dir, "icl_kind_means_s2.npz")
            arrays, meta = AM.load_npz(path) if os.path.exists(path) else ({}, {})
            fp = self.fingerprint("icl")
            if not arrays or meta.get("fingerprint") != fp or meta.get("dtype") != self.dtype:
                from .. import icl_battery as IB
                self.say("ICL: Query-Mittel je Art (Seed %d) ..." % self.args.battery_seed)
                means = IB.query_means(self.model, self.icl_items("icl"), self.pad_id,
                                       batch_tokens=self.args.batch_tokens, max_batch=self.args.max_batch)
                kinds = list(means)
                arrays = {f"k{i:02d}": means[k] for i, k in enumerate(kinds)}
                AM.save_npz_stable(path, arrays, {"fingerprint": fp, "kinds": kinds, "dtype": self.dtype,
                                                  "battery_seed": self.args.battery_seed})
                arrays, meta = AM.load_npz(path)
            self._cache["kind_means"] = {k: arrays[f"k{i:02d}"] for i, k in enumerate(meta["kinds"])}
            self._cache["kind_means_sha"] = overlap.file_sha256(path)
        return self._cache["kind_means"]

    # ------------------------------------------------------------------------------------ condition keys
    def variant_config(self, variant):
        key = ("vcfg", variant)
        if key in self._cache:
            return self._cache[key]
        a = self.args
        c = {"version": COND_VERSION, "hf_id": self.hf_id, "revision": a.revision or "main", "dtype": self.dtype,
             "attn_impl": self.attn_impl, "batch_tokens": a.batch_tokens, "max_batch": a.max_batch, "seed": self.seed}
        if variant in AM.HH_VARIANTS:
            c.update(split_sha=self.split_sha, n_eval=a.n_eval, eval_ids_sha=AM.ids_sha(self.eval_ids),
                     variant_seqs_sha=AM.seqs_sha(self.seqs(variant)))
            if variant in ("hh2", "hh3"):
                c.update(probes_sha=self.probes_sha, readout_layers=list(self.readout))
        elif variant == "ppl":
            c.update(ppl_tokens=a.ppl_tokens, window=a.ppl_window, stride=a.ppl_stride,
                     variant_seqs_sha=AM.seqs_sha(self.seqs("ppl")))
        elif variant in ("icl", "icl_s1"):
            c.update(battery_seed=a.battery_seed if variant == "icl" else a.calib_seed,
                     fingerprint=self.fingerprint(variant), smoke_nseq=SMOKE_ICL_NSEQ if self.smoke else None)
        self._cache[key] = c
        return c

    def cond_dict(self, cond):
        d = {"arm": cond.arm, "variant": cond.variant, "set": cond.set, "r": cond.r, "d": cond.d,
             "heads": sorted(AS.to_flat(cond.heads, self.H)), "policy": None}
        if cond.arm != "clean":
            d["policy"] = "all" if cond.variant == "ppl" else POLICY[cond.arm]
        if cond.arm in ("mean", "mean_last", "dir_mean", "dla", "resample"):
            d["ref_sha"] = self.refs_sha
        if cond.arm == "resample":
            d["donor_seed"] = self.seed
        if cond.arm == "dir_mean":
            d["dir_key"] = "dom" if cond.d is None else [self.seed, cond.identity, cond.set, cond.d]
        if cond.arm == "gauss":
            d["gauss_seed"] = self.seed
        if cond.arm == "knock":
            self.kind_means()
            d["kind_means_sha"] = self._cache["kind_means_sha"]
        return d

    def key(self, cond):
        return condition_key(self.variant_config(cond.variant), self.cond_dict(cond))

    def cond_path(self, name):
        return os.path.join(self.cond_dir, name + ".json")

    def load_cond(self, name):
        return _read_json(self.cond_path(name))

    def stored_n_eval(self, name):
        """config.n_eval of a stored HH condition (from the file head, else the full file), or None."""
        path = self.cond_path(name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            head = f.read(65536)
        m = _NEVAL_RE.search(head)
        if m:
            return int(m.group(1))
        n = ((_read_json(path) or {}).get("config") or {}).get("n_eval")
        return None if n is None else int(n)

    def stored_key(self, name):
        """The "key" of a stored condition from the file head (the header fields are written first), else None."""
        path = self.cond_path(name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            head = f.read(4096)
        m = _KEY_RE.search(head)
        if m:
            return m.group(1)
        payload = _read_json(path)
        return (payload or {}).get("key")

    # ------------------------------------------------------------------------------------ randoms
    def random_R(self, frozen, set_name, R, extend=False):
        """Number of usable draws (<= R); extends the own frozen file with --extend_random, else refuses R > stored."""
        stored = AS.random_R(frozen, set_name)
        if R <= stored:
            return R
        if frozen is not self._cache.get("frozen"):
            return stored                                    # Base frozen file of the fixed-identity arm (F14)
        if not extend:
            return stored
        if not self.args.extend_random:
            raise AblateExit(EXIT_INPUTS, f"{set_name}: R = {R} über dem eingefrorenen R = {stored} "
                                          "(--extend_random hängt Ziehungen an; Präfix-Eigenschaft)")
        new = AS.extend_random(frozen, set_name, R)
        AS.write_frozen(self.frozen_path, new)
        self._cache["frozen"] = AS.load_frozen(self.frozen_path)
        self.say(f"Zufallsmengen von {set_name} auf R = {R} erweitert")
        return R

    # ------------------------------------------------------------------------------------ runner
    def make_edit(self, cond):
        m = self.model
        heads = list(cond.heads)
        refs = self.refs
        if cond.arm == "mean":
            return A.MeanReplace(m, heads, refs.mu_response, policy="all" if cond.variant == "ppl" else "response")
        if cond.arm == "mean_last":
            return A.MeanReplace(m, heads, refs.mu_last, policy="last")
        if cond.arm == "resample":
            donors = A.donor_assignment(self.eval_ids, len(refs.items), self.seed)
            return A.DonorReplace(m, heads, refs.donor_means, donors, policy="response")
        if cond.arm == "dir_mean":
            if cond.d is None:
                dirs = {}
                for l, h in heads:
                    v = refs.dom[l * self.H + h].astype(np.float64)
                    nv = float(np.linalg.norm(v))
                    dirs[(l, h)] = (v / nv if nv > 0 else np.eye(self.D)[0]).astype(np.float32)
            else:
                dirs = A.random_directions(heads, self.D, self.seed, cond.identity, cond.set, cond.d)
            return A.DirectionReplace(m, heads, refs.mu_response, dirs, policy="response")
        if cond.arm == "gauss":
            return A.GaussReplace(m, heads, self.seed, policy="response")
        raise ValueError(f"no edit for arm {cond.arm}")

    def run_cond(self, cond, load=False):
        """Computes and stores one condition unless it exists with the same key (or --force). Returns (payload, new);
        for a skipped condition the payload is read only with load=True."""
        key = self.key(cond)
        if not self.args.force and self.stored_key(cond.name) == key:
            return (self.load_cond(cond.name) if load else None), False
        _ = self.model                          # lazy load outside the timer: wall_seconds is per-condition overhead only
        t0 = time.time()
        if cond.variant in AM.HH_VARIANTS:
            body = self._run_hh(cond)
        elif cond.variant == "ppl":
            body = self._run_ppl(cond)
        else:
            body = self._run_icl(cond)
        payload = {"name": cond.name, "version": COND_VERSION, "key": key, "model": self.short, "hf_id": self.hf_id,
                   "commit_sha": self.commit_sha, "dtype": self.dtype, "attn_impl": self.attn_impl, "arm": cond.arm,
                   "variant": cond.variant, "set": cond.set,
                   "random": ({"kind": "heads", "index": cond.r} if cond.r is not None else
                              {"kind": "directions", "index": cond.d} if cond.d is not None else None),
                   "heads": sorted(AS.to_flat(cond.heads, self.H)), "n_heads": len(cond.heads),
                   "config": {**self.variant_config(cond.variant), **{k: v for k, v in self.cond_dict(cond).items()
                                                                       if k not in ("heads",)}},
                   "created": _now(), **body}
        payload["stats"]["wall_seconds"] = round(time.time() - t0, 3)
        _write_json(self.cond_path(cond.name), payload)
        self.log("condition", name=cond.name, seconds=payload["stats"].get("seconds"))
        self._report_line(cond, payload)
        return payload, True

    def _report_line(self, cond, payload):
        if self.args.quiet and cond.r is not None:
            return
        s, st = payload.get("summary", {}), payload["stats"]
        if "margin_mean" in s:
            val = f"margin_mean {s['margin_mean']:+.4f} acc_mean {s['acc_mean']:.3f}"
            if "m2" in s:
                val += f" | M2 L1 {s['m2']['head_auc_mean']:.3f} resid {s['m2']['resid_auc']:.3f}"
        elif "nll" in s:
            val = f"nll {s['nll']:.4f}"
        elif "symrare" in s:
            val = f"symrare {s['symrare']:+.3f}" + (f" dret {s['dret_mean']:+.3f}" if "dret_mean" in s else "")
        elif "de_mean" in s:
            val = f"DE {s['de_mean']:+.4f}"
        else:
            val = ""
        tps = st.get("tokens_per_second")
        print(f"[ablate] {cond.name}: {val} ({st.get('seconds', 0):.1f}s, "
              f"{tps:.0f} tok/s)" if tps else f"[ablate] {cond.name}: {val} ({st.get('seconds', 0):.1f}s)", flush=True)

    def _run_hh(self, cond):
        seqs = self.seqs(cond.variant)
        n = len(self.eval_items)
        m = self.model
        if cond.arm == "dla":
            t0 = time.time()
            de = A.direct_effect(m, seqs, self.pad_id, list(cond.heads), self.refs.mu_response, policy="response",
                                 batch_tokens=self.args.batch_tokens, max_batch=self.args.max_batch)
            sec = time.time() - t0
            items = {"id": self.eval_ids, "de_pos": de["de_sums"][:n], "de_neg": de["de_sums"][n:2 * n],
                     "pos_n": de["counts"][:n], "neg_n": de["counts"][n:2 * n], "clean_pos": de["clean_sums"][:n],
                     "clean_neg": de["clean_sums"][n:2 * n]}
            toks = int(sum(len(s.ids) for s in seqs))
            return {"summary": {"de_mean": float(np.nanmean(AM.dla_values(items)))}, "items": items,
                    "stats": {"seconds": sec, "tokens_processed": toks, "tokens_per_second": toks / sec if sec else None}}
        hooks = [] if cond.arm == "clean" else [self.make_edit(cond)]
        m2 = cond.variant in ("hh2", "hh3")
        if m2:
            last = A.LastTokenCollector(m, self.readout, len(seqs))
            resid = A.ResidualCapture(m, len(seqs))
            hooks += [last, resid]
        res = A.run_sequences(m, seqs, self.pad_id, hooks, batch_tokens=self.args.batch_tokens,
                              max_batch=self.args.max_batch, progress=self.progress, desc=cond.name)
        items = AM.hh_item_arrays(seqs, res.sums, res.counts, cond.variant, n)
        summary = AM.m1_summary(items)
        payload = {"summary": summary, "items": items,
                   "stats": {"seconds": res.seconds, "tokens_processed": res.tokens_processed,
                             "tokens_padded": res.tokens_padded, "tokens_scored": res.tokens_scored,
                             "tokens_per_second": res.tokens_per_second}}
        if m2:
            head_x, post = last.values[:2 * n], resid.post_last[:2 * n]
            del last
            labels = np.r_[np.ones(n, np.int64), np.zeros(n, np.int64)]
            groups = np.r_[self.clusters, self.clusters]
            exclude = set(AS.to_flat(cond.heads, self.H))
            head, rs, mlp = AM.probe_scores(self.probes, head_x, post, exclude)
            summary["m2"] = AM.m2_summary(head, rs, mlp, labels)
            if cond.r is None and cond.d is None:
                dev = "cuda" if str(self.args.device).startswith("cuda") else None
                summary["m2"].update(AM.refit_m2(head_x, post, labels, groups, self.readout, self.H, exclude,
                                                 device=dev))
                AM.save_npz_stable(os.path.join(self.cond_dir, cond.name + ".m2.npz"),
                                   {"head": head.astype(np.float32), "resid": rs.astype(np.float32),
                                    "mlp": mlp.astype(np.float32), "labels": labels, "clusters": groups})
        if cond.arm == "clean" and cond.variant == "hh3":
            payload["covariates"] = AM.reuse_covariates(self.tok, self.eval_items)
        return payload

    def _run_ppl(self, cond):
        seqs = self.seqs("ppl")
        hooks = [] if cond.arm == "clean" else [self.make_edit(cond)]
        res = A.run_sequences(self.model, seqs, self.pad_id, hooks, batch_tokens=self.args.batch_tokens,
                              max_batch=self.args.max_batch, progress=self.progress, desc=cond.name)
        return {"summary": AM.ppl_summary(res.sums, res.counts),
                "items": {"id": list(range(len(seqs))), "sum": res.sums, "n": res.counts},
                "stats": {"seconds": res.seconds, "tokens_processed": res.tokens_processed,
                          "tokens_padded": res.tokens_padded, "tokens_per_second": res.tokens_per_second}}

    def _run_icl(self, cond):
        from .. import icl_battery as IB
        items = self.icl_items(cond.variant)
        if cond.arm == "clean":
            r = IB.score_items(self.model, items, self.pad_id, batch_tokens=self.args.batch_tokens,
                               max_batch=self.args.max_batch)
        else:
            r = IB.score_items(self.model, items, self.pad_id, heads=list(cond.heads), kind_means=self.kind_means(),
                               batch_tokens=self.args.batch_tokens, max_batch=self.args.max_batch)
        summary = IB.summarize(items, r["margin"])
        payload = {"summary": summary,
                   "items": {"index": [it.index for it in items], "tlp": r["tlp"], "alt": r["alt"], "margin": r["margin"]},
                   "stats": {"seconds": r["seconds"], "tokens_processed": r["tokens_processed"],
                             "tokens_padded": r.get("tokens_padded"),
                             "tokens_per_second": r["tokens_processed"] / r["seconds"] if r["seconds"] else None}}
        if cond.arm == "clean":
            payload["meta"] = {"kind": [it.kind for it in items], "concept": [it.concept for it in items],
                               "labelset": [it.labelset for it in items], "seq": [it.seq for it in items],
                               "cluster": [it.cluster for it in items], "fingerprint": self.fingerprint(cond.variant)}
        else:
            clean = self.load_cond(cond.clean or f"clean__{cond.variant}")
            if clean is not None:
                eff = IB.effects(items, clean["items"]["margin"], r["margin"])
                summary.update({f"{q}_mean": float(np.mean(eff[q][0])) for q in ("dret", "dR", "dP")})
        return payload

    def execute(self, conds):
        seen, n_new = set(), 0
        for c in conds:
            if c.name in seen:
                continue
            seen.add(c.name)
            _, new = self.run_cond(c)
            n_new += bool(new)
        return n_new


# ----------------------------------------------------------------------------------------------------------------
# grid (8.3)
# ----------------------------------------------------------------------------------------------------------------

def _set_ok(frozen, name):
    s = frozen["sets"].get(name)
    return s is not None and int(s["n"]) > 0


def named_with_null(ctx, arm, variant, set_name, R, stage, group, *, frozen=None, suffix="", extend=False,
                    directions=False):
    """[named condition, its random conditions] of one cell; [] for a missing / empty set. Random conditions are
    named after the set that owns the draws (F12 reuse), random directions (dir_mean) after the set itself."""
    frozen = frozen if frozen is not None else ctx.frozen
    if not _set_ok(frozen, set_name):
        return []
    heads = tuple(AS.set_heads(frozen, set_name))
    disp = set_name + suffix
    identity = frozen["sets"][set_name]["identity"]
    named = Cond(f"{arm}__{variant}__{disp}", arm, variant, disp, heads=heads, stage=stage, group=group,
                 identity=identity, clean=f"clean__{variant}")
    out = [named]
    if R <= 0:
        return out
    if directions:
        for d in range(R):
            c = Cond(f"{named.name}__d{d:03d}", arm, variant, disp, d=d, heads=heads, stage=stage, group=group,
                     identity=identity, clean=named.clean)
            named.randoms.append(c.name)
            out.append(c)
        return out
    owner = AS._random_entry(frozen, set_name)[0]
    R_eff = ctx.random_R(frozen, set_name, R, extend=extend)
    for r in range(R_eff):
        c = Cond(f"{arm}__{variant}__{owner}{suffix}__r{r:03d}", arm, variant, owner + suffix, r=r,
                 heads=tuple(AS.random_heads(frozen, set_name, r)), stage=stage, group=group,
                 identity=frozen["random"][owner]["identity"], clean=named.clean)
        named.randoms.append(c.name)
        out.append(c)
    return out


def plan_stage1(ctx, extend=False):
    a = ctx.args
    c33 = named_with_null(ctx, "mean", "hh3", "C33", a.r_c33, 1, "1 primär", extend=extend)
    o = named_with_null(ctx, "mean", "hh3", "O", a.r_o, 1, "1 primär", extend=extend)
    for lst in (c33, o):
        if lst:
            lst[0].family = "primary"
    out = [x[0] for x in (c33, o) if x]
    rc, ro = c33[1:], o[1:]
    for i in range(max(len(rc), len(ro))):
        out += ([rc[i]] if i < len(rc) else []) + ([ro[i]] if i < len(ro) else [])
    return out


def _m1_variant(ctx):
    return "hh3" if ctx.args.threeway_all else "hh2"


def plan_arm(ctx, arm, extend=False):
    a = ctx.args
    fr = ctx.frozen
    g = f"2 {arm}"
    out = []

    def add(*args, **kw):
        out.extend(named_with_null(ctx, *args, stage=2, group=g, extend=extend, **kw))

    if arm == "m3":
        out.append(_clean("icl", 2, g))
        for s, R in (("C33", a.r_sec), ("O", a.r_sec), ("P_band", a.r_sec), ("TopK_all", a.r_sec), ("WholeBand", 0),
                     ("CR20", a.r_sec), ("CensusReaders", a.r_sec), ("CB24", a.r_sec)):
            add("knock", "icl", s, R)
    elif arm == "mean_sets":
        v = _m1_variant(ctx)
        out.append(_clean(v, 2, g))
        reuse = fr["random"].get("P_band", {}).get("reuse") == "O"
        for s, R in (("CminusO", a.r_small), ("P_band", a.r_small), ("P_DoM", a.r_sec), ("C33_readers", a.r_sec),
                     ("C33_brakes", a.r_sec), ("TopK_all", a.r_sec), ("WholeBand", 0)):
            if s == "P_band" and reuse:                  # F12: hh3, compared with O's Stage-1 randoms
                add("mean", "hh3", "P_band", a.r_o)
                continue
            add("mean", v, s, R)
    elif arm == "m4":
        out.append(_clean("ppl", 2, g))
        for s, R in (("C33", a.r_sec), ("O", a.r_sec), ("CminusO", a.r_sec), ("P_band", a.r_sec), ("P_DoM", 0),
                     ("C33_readers", 0), ("C33_brakes", 0), ("TopK_all", a.r_sec), ("WholeBand", 0)):
            add("mean", "ppl", s, R)
    elif arm == "mean_last":
        v = _m1_variant(ctx)
        out.append(_clean(v, 2, g))
        for s in ("C33", "O", "P_band"):
            add("mean_last", v, s, a.r_sec)
    elif arm == "m5":
        for s in ("C33", "O"):
            if _set_ok(fr, s):
                out.append(Cond(f"dla__hh2__{s}", "dla", "hh2", s, heads=tuple(AS.set_heads(fr, s)), stage=2, group=g,
                                identity=fr["sets"][s]["identity"], clean="clean__hh2", te=f"mean__hh3__{s}",
                                te_clean="clean__hh3"))
    elif arm == "m7":
        out.append(_clean("hhshuf", 2, g))
        for s in ("C33", "O"):
            add("mean", "hhshuf", s, 0)
    elif arm in ("resample", "gauss"):
        v = _m1_variant(ctx)
        out.append(_clean(v, 2, g))
        for s in ("C33", "O"):
            add(arm, v, s, a.r_sec)
    elif arm == "dir_mean":
        v = _m1_variant(ctx)
        out.append(_clean(v, 2, g))
        for s in ("C33", "O", "P_band"):
            add("dir_mean", v, s, a.r_sec, directions=True)
    elif arm == "fixed_identity":
        base = load_base_frozen(ctx)
        if base is None:
            return []
        v = _m1_variant(ctx)
        out.append(_clean(v, 2, g))
        for s in FIXED_IDENTITY_SETS:
            add("mean", v, s, a.r_sec, frozen=base, suffix="_base")
    return out


def load_base_frozen(ctx):
    """Base frozen file of the fixed-identity arm (F14), or None (instruct only)."""
    if ctx.is_base:
        return None
    path = os.path.join(ctx.abl, f"{ctx.icl_identity}_frozen_sets.json")
    if not os.path.exists(path):
        return None
    if "base_frozen" not in ctx._cache:
        ctx._cache["base_frozen"] = AS.load_frozen(path)
    return ctx._cache["base_frozen"]


def gate_conds(ctx, extend=False):
    out = [_clean("icl", 0, "0 Gate")]
    out += named_with_null(ctx, "knock", "icl", "CR20", ctx.args.r_sec, 0, "0 Gate", extend=extend)
    return out


def all_specs(ctx):
    """Specs of every planned named condition (stage 0 gate, stage 1, all stage-2 arms) for the report."""
    conds = []
    try:
        conds += gate_conds(ctx)
        conds += plan_stage1(ctx)
        for arm in ARMS:
            if arm == "fixed_identity" and ctx.is_base:
                continue
            conds += plan_arm(ctx, arm)
    except AblateExit:
        pass
    seen, out = set(), []
    for c in conds:
        if c.name not in seen:
            seen.add(c.name)
            out.append(c.spec())
    return out


# ----------------------------------------------------------------------------------------------------------------
# stage 0
# ----------------------------------------------------------------------------------------------------------------

def smoke_table_path(ctx):
    return os.path.join(ctx.abl, "_smoke", f"{ctx.short}_main_helpful_pythia_SMOKE.json")


def write_smoke_table(ctx, path):
    """SMOKE-ONLY synthetic helpful table (9): uniform AUCs 0.55-0.60, +0.3 for 4 of the C33 heads and for 16 further
    band heads, so O, CminusO and P_band are non-empty. Never written under results/heads/."""
    from ..probing import _finish_table, save_head_scores
    L, H = ctx.L, ctx.H
    rng = overlap._rng(ctx.seed, "ablation-smoke-table")
    auc = 0.55 + 0.05 * rng.random(L * H)
    c33 = AS.named_c33(ctx.icl_identity) or AS.smoke_c33(L, H, ctx.band, ctx.seed)
    c33 = [(l, h) for l, h in c33 if l < L]
    pick = [c33[i] for i in rng.choice(len(c33), min(4, len(c33)), replace=False)]
    others = [(l, h) for l in AS.band_layers(ctx.band) if l < L for h in range(H) if (l, h) not in set(c33)]
    pick += [others[i] for i in rng.choice(len(others), min(16, len(others)), replace=False)]
    for l, h in pick:
        auc[l * H + h] += 0.3
    rows = [{"layer": l, "head": h, "flat_idx": l * H + h, "auc": float(auc[l * H + h]), "acc": float(auc[l * H + h]),
             "bal_acc": float(auc[l * H + h]), "majority": 0.5, "pos_rate": 0.5, "n_train": 20000, "n_val": 1400}
            for l in range(L) for h in range(H)]
    df = _finish_table(rows, {"layers": L, "heads": H, "score": "auc", "ranking": "stable", "feature_mode": "head",
                              "mode": "pythia"}, "auc", "stable")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_head_scores(df, path, meta={"model": ctx.short, "mode": "pythia", "synthetic": True,
                                     "rule": "SMOKE-ONLY synthetic table (ablate --smoke): not a probe result"})


def check_heads_table(ctx):
    """(ValueTable, path, synthetic) of the helpful probe table; F6 checks (exit 2)."""
    from ..probing import load_head_scores
    a = ctx.args
    path = a.heads_table or config.heads_path(ctx.P, ctx.short, None, "helpful", "pythia")
    synthetic = False
    if not os.path.exists(path):
        if a.heads_table or not ctx.smoke:
            raise AblateExit(EXIT_INPUTS, f"Probe-Tabelle fehlt: {path} – zuerst das Overlap-Notebook (helpful, Ranking "
                                          "pythia) für dieses Modell laufen lassen")
        path = smoke_table_path(ctx)
        if not os.path.exists(path):
            write_smoke_table(ctx, path)
            ctx.say(f"SMOKE: synthetische Probe-Tabelle {path} (nur Funktionsprüfung)")
    _, meta = load_head_scores(path)
    synthetic = bool(meta.get("synthetic"))
    table = overlap.load_value_table(path, "helpful", "pythia")
    if (table.layers, table.heads) != (ctx.L, ctx.H):
        raise AblateExit(EXIT_INPUTS, f"{path}: Geometrie {table.layers} x {table.heads} != Modell {ctx.L} x {ctx.H}")
    bad = [f"{k}={meta.get(k)!r}" for k, want in (("mode", "pythia"), ("score", "auc")) if meta.get(k) != want]
    if meta.get("feature_mode") not in (None, "head"):
        bad.append(f"feature_mode={meta.get('feature_mode')!r}")
    if bad:
        raise AblateExit(EXIT_INPUTS, f"{path}: erwartet Ranking pythia (Head-only-AUC), gefunden {', '.join(bad)}")
    if table.n_train != 20000:
        msg = f"{path}: n_train = {table.n_train}, die Selektion lief mit 20000 Texten (F6)"
        if not (ctx.smoke or synthetic):
            raise AblateExit(EXIT_INPUTS, msg + " – STOPP")
        ctx.say("SMOKE: " + msg)
    return table, path, synthetic


def run_selftest(ctx, c33_heads, band_heads):
    path = os.path.join(ctx.run_dir, "selftest.json")
    items = ctx.role_items("means_ref", min(SELFTEST_PAIRS, ctx.args.n_means_ref))
    seqs = AM.hh_sequences(ctx.enc, items, "hh2")
    n = len(items)
    key = {"checks": SELFTEST_VERSION, "items": AM.ids_sha([it["group"] for it in items]), "c33": AS.to_flat(c33_heads, ctx.H),
           "band": list(ctx.band), "dtype": ctx.dtype, "attn_impl": ctx.attn_impl, "hf_id": ctx.hf_id}
    old = _read_json(path)
    if old and old.get("key") == key and old.get("ok") and not ctx.args.force:
        ctx.say("Hook-Selbsttest: vorhanden (ok), übersprungen")
        return old
    ctx.say(f"Hook-Selbsttest auf {n} means_ref-Paaren (C33 und WholeBand) ...")
    m = ctx.model
    ref = A.compute_refs(m, seqs[:n], seqs[n:], ctx.pad_id, batch_tokens=ctx.args.batch_tokens,
                         max_batch=ctx.args.max_batch).mu_response
    out = {"key": key, "created": _now()}
    for name, heads in (("C33", c33_heads), ("WholeBand", band_heads)):
        out[name] = A.hook_selftest(m, seqs, ctx.pad_id, heads, ref, batch_tokens=min(2048, ctx.args.batch_tokens),
                                    max_batch=8)
        for k, v in out[name]["checks"].items():
            print(f"[ablate]   {name:9s} {k:15s} {'ok ' if v['ok'] else 'FEHLER'} value {v['value']:.3g} "
                  f"(tol {v['tol']:.0g})", flush=True)
    out["ok"] = all(out[k]["ok"] for k in ("C33", "WholeBand"))
    _write_json(path, out)
    ctx.log("selftest", ok=out["ok"])
    if not out["ok"]:
        raise AblateExit(EXIT_SELFTEST, "Hook-Selbsttest fehlgeschlagen (selftest.json) – STOPP")
    return out


def ensure_refs(ctx):
    path = os.path.join(ctx.run_dir, "refs.npz")
    items = ctx.role_items("means_ref", ctx.args.n_means_ref)
    ids = [int(it["group"]) for it in items]
    if os.path.exists(path) and not ctx.args.force:
        refs = A.load_refs(path)
        if refs.items.tolist() == ids and refs.meta.get("dtype") == ctx.dtype:
            ctx.say(f"Referenzen: {path} vorhanden ({len(ids)} Paare), übersprungen")
            return refs
    ctx.say(f"Referenzen (μ_resp, μ_last, DoM, Spender) auf {len(ids)} means_ref-Paaren ...")
    seqs = AM.hh_sequences(ctx.enc, items, "hh2")
    n = len(items)
    refs = A.compute_refs(ctx.model, seqs[:n], seqs[n:], ctx.pad_id, batch_tokens=ctx.args.batch_tokens,
                          max_batch=ctx.args.max_batch, progress=ctx.progress)
    refs.meta.update({"model": ctx.short, "items_sha": AM.ids_sha(ids)})
    sha = A.save_refs(path, refs)
    ctx._cache.pop("refs", None)
    ctx.log("refs", sha=sha)
    return refs


def ensure_m2_train(ctx):
    path = os.path.join(ctx.run_dir, "m2_train.npz")
    items = ctx.role_items("means_probe", ctx.args.n_means_probe)
    ids = [int(it["group"]) for it in items]
    meta = {"items_sha": AM.ids_sha(ids), "readout_layers": list(ctx.readout), "dtype": ctx.dtype, "hf_id": ctx.hf_id}
    if os.path.exists(path) and not ctx.args.force:
        _, old = AM.load_npz(path)
        if old == json.loads(json.dumps(meta)):
            ctx.say(f"M2-Trainingsmerkmale: {path} vorhanden, übersprungen")
            return path
    ctx.say(f"M2-Trainingsmerkmale (Layer {ctx.readout[0]}-{ctx.readout[-1]} + Residuum) auf {len(ids)} "
            "means_probe-Paaren ...")
    seqs = AM.hh_sequences(ctx.enc, items, "hh2")
    n = len(items)
    m = ctx.model
    last = A.LastTokenCollector(m, ctx.readout, len(seqs))
    resid = A.ResidualCapture(m, len(seqs))
    A.run_sequences(m, seqs, ctx.pad_id, [last, resid], batch_tokens=ctx.args.batch_tokens,
                    max_batch=ctx.args.max_batch, score=False, progress=ctx.progress, desc="m2_train")
    AM.save_npz_stable(path, {"head": last.values, "post": resid.post_last,
                              "labels": np.r_[np.ones(n, np.int64), np.zeros(n, np.int64)],
                              "items": np.asarray(ids + ids, np.int64)}, meta)
    ctx.log("m2_train")
    return path


def ensure_probes(ctx, train_path):
    path = os.path.join(ctx.run_dir, "probes_m2.npz")
    train_sha = overlap.file_sha256(train_path)
    if os.path.exists(path) and not ctx.args.force:
        _, meta = AM.load_npz(path)
        if meta.get("train_sha") == train_sha:
            ctx.say("M2-Probes: vorhanden, übersprungen")
            return
    ctx.say("M2-Probes trainieren (logistisch je Lese-Head, Residuum, MLP) ...")
    t0 = time.time()
    arrays, _ = AM.load_npz(train_path)
    probes = AM.train_probes(arrays["head"], arrays["post"], arrays["labels"], ctx.readout, ctx.H, ctx.D)
    AM.save_npz_stable(path, probes, {"train_sha": train_sha, "readout_layers": list(ctx.readout),
                                      "n_train": int(len(arrays["labels"]))})
    ctx._cache.pop("probes", None)
    ctx.log("probes", seconds=time.time() - t0)
    ctx.say(f"M2-Probes trainiert in {time.time() - t0:.0f}s")


def stored_n_eval(ctx):
    """n_eval of the existing HH conditions: that of clean__hh3 (the anchor of the primary family and of the floor),
    else the most frequent one among the HH condition files; None if there is none."""
    n = ctx.stored_n_eval("clean__hh3")
    if n is not None or not os.path.isdir(ctx.cond_dir):
        return n
    vals = [ctx.stored_n_eval(f[:-5]) for f in sorted(os.listdir(ctx.cond_dir))
            if f.endswith(".json") and ".tmp" not in f and AR.parse_name(f[:-5])["variant"] in AM.HH_VARIANTS]
    vals = [v for v in vals if v is not None]
    return max(set(vals), key=vals.count) if vals else None


def check_n_eval(ctx):
    """Exit 2 when --n_eval differs from the n_eval the existing HH conditions were computed with. N_EVAL is not a
    read-time prefix: it fixes the item ids, the hh3 partner derangement and the sequences, i.e. the key of every HH
    condition, so a change recomputes all of them. Allowed with --change_n_eval (or --force)."""
    old = stored_n_eval(ctx)
    a = ctx.args
    if old is None or old == a.n_eval:
        return
    if a.change_n_eval or a.force:
        ctx.say(f"WARNUNG: N_EVAL {old} -> {a.n_eval} (--change_n_eval): jede HH-Bedingung wird beim nächsten Lauf "
                "ihrer Stufe neu gerechnet; bis dahin lässt der Bericht die veralteten Bedingungen aus")
        return
    raise AblateExit(EXIT_INPUTS, f"--n_eval {a.n_eval} weicht von den vorhandenen HH-Bedingungen ab (gerechnet mit "
                                  f"n_eval = {old}, {ctx.cond_dir}). N_EVAL ist kein Präfix: es legt Items, "
                                  "Partner-Antworten und damit den Schlüssel jeder HH-Bedingung fest, eine Änderung "
                                  f"rechnet alle HH-Bedingungen neu. N_EVAL auf {old} zurücksetzen oder die Änderung "
                                  "bewusst mit --change_n_eval (Notebook: CHANGE_N_EVAL = True) erlauben")


def eval_condition_exists(ctx):
    if not os.path.isdir(ctx.cond_dir):
        return False
    return any(f.endswith(".json") and AR.parse_name(f[:-5])["variant"] in AM.HH_VARIANTS
               for f in os.listdir(ctx.cond_dir))


def freeze(ctx, table, table_path, synthetic):
    """Named sets + random draws -> {short}_frozen_sets.json (before any EVAL forward); exit 5 on a mismatch."""
    a = ctx.args
    refs = ctx.refs
    c33 = AS.named_c33(ctx.icl_identity)
    if c33 is None and not ctx.smoke:
        raise AblateExit(EXIT_INPUTS, f"kein benannter Schaltkreis für {ctx.icl_identity}; nur mit --smoke "
                                      "(synthetischer Ersatz)")
    census = ctx.census
    sets = AS.named_sets(topk=table.topk, c33=c33, census_roles=census.roles if census else None,
                         census_effects=census.effects("dret") if census else None,
                         dom_norm=A.dom_norms(refs.dom), n_layers=ctx.L, n_heads=ctx.H, band=ctx.band, seed=ctx.seed,
                         model_identity=ctx.short, icl_identity=ctx.icl_identity)
    R = AS.r_table(a.r_c33, a.r_o, a.r_small, a.r_sec)
    geometry = {"layers": ctx.L, "heads": ctx.H, "kv_heads": ctx.KV, "heads_per_kv": ctx.hpk, "head_dim": ctx.D}
    randoms = AS.random_table(sets, R, n_layers=ctx.L, n_heads=ctx.H, heads_per_kv=ctx.hpk, seed=ctx.seed)
    rel = lambda p: os.path.relpath(p, ctx.P["root"])               # noqa: E731
    inputs = {"heads_table": {"path": rel(table_path), "sha256": overlap.file_sha256(table_path), "k": table.k,
                              "n_train": table.n_train, "n_val": table.n_val, "synthetic": synthetic,
                              "meta": {"mode": table.meta.get("mode"), "score": table.score,
                                       "feature_mode": table.meta.get("feature_mode")}},
              "refs": {"path": rel(os.path.join(ctx.run_dir, "refs.npz")), "sha256": ctx.refs_sha},
              "split": {"path": rel(ctx.split_path), "sha256": ctx.split_sha}}
    if census is not None:
        inputs["census"] = {"file": getattr(census.spec, "path", None), "sha256": census.sha256, "pinned": census.pinned,
                            "origin": "local" if not str(census.origin).startswith("http") else census.origin}
    payload = AS.build_frozen(model_short=ctx.short, icl_identity=ctx.icl_identity, geometry=geometry, band=ctx.band,
                              seed=ctx.seed, sets=sets, randoms=randoms, inputs=inputs, smoke=ctx.smoke)
    force = False
    if os.path.exists(ctx.frozen_path):
        status, reason = AS._compare_frozen(AS.load_frozen(ctx.frozen_path), payload)
        if status == "mismatch":
            if not a.refreeze:
                raise AblateExit(EXIT_FROZEN, f"{ctx.frozen_path}: {reason}; die Sätze wurden vor der Auswertung "
                                              "eingefroren (--refreeze nur, solange keine EVAL-Bedingung existiert)")
            if eval_condition_exists(ctx):
                raise AblateExit(EXIT_FROZEN, "--refreeze verweigert: es existieren bereits EVAL-Bedingungen")
            force = True
        elif status == "extends" and not a.extend_random:
            raise AblateExit(EXIT_INPUTS, "R über dem eingefrorenen R: --extend_random hängt Ziehungen an")
    try:
        sha = AS.write_frozen(ctx.frozen_path, payload, force=force)
    except AS.FrozenMismatch as e:
        raise AblateExit(EXIT_FROZEN, str(e))
    ctx._cache.pop("frozen", None)
    fr = ctx.frozen
    ctx.log("frozen", sha=sha)
    ctx.say("eingefrorene Sätze (" + ("SMOKE, C33 = synthetischer Smoke-Ersatz; " if fr["sets"]["C33"].get("synthetic")
                                       else "") + ", ".join(f"{k} {v['n']}" for k, v in fr["sets"].items())
            + f"); P_band-Null: {'O (F12)' if fr['random'].get('P_band', {}).get('reuse') == 'O' else 'eigene Ziehungen'}"
            + f"; sha256 {sha[:12]}")
    return fr


def floor_from(ctx, clean):
    v = AM.m1_values(clean["items"])
    fl = ST.floor_check(v["win_mean"], v["margin_mean"], v["len_hi_gt_lo"], ctx.clusters_of(clean["items"]["id"]),
                        B=ctx.args.bootstrap, seed=0)
    fl.update({"condition": clean["name"], "key": clean["key"], "created": _now(), "model": ctx.short})
    _write_json(os.path.join(ctx.run_dir, "floor.json"), fl)
    ctx.say(fl["text_de"])
    return fl


def run_gate(ctx):
    """Base: calibration (seed 1; skipped in smoke), CR20 + R_SEC random knocks on the fresh draw -> gate.json."""
    from .. import icl_battery as IB
    calib = None
    if not ctx.smoke:
        clean_s1, _ = ctx.run_cond(_clean("icl_s1", 0, "0 Gate"), load=True)
        calib = IB.calibration_check(clean_s1["summary"], ctx.short)
        ctx.say("Kalibrierung Seed %d: " % ctx.args.calib_seed + ", ".join(
            f"{k} {v:+.3f}" for k, v in calib.get("diffs", {}).items()) + f" -> ok={calib.get('ok')}")
    conds = gate_conds(ctx, extend=True)
    if len(conds) < 2:
        raise AblateExit(EXIT_INPUTS, "CR20 fehlt (Census?) – keine Positivkontrolle möglich")
    ctx.execute(conds)
    clean = ctx.load_cond("clean__icl")
    items = AR.icl_meta_items(clean)
    cr = ctx.load_cond(conds[1].name)
    delta, cl = IB.effects(items, clean["items"]["margin"], cr["items"]["margin"])["dret"]
    rand = [float(np.mean(IB.effects(items, clean["items"]["margin"], ctx.load_cond(n)["items"]["margin"])["dret"][0]))
            for n in conds[1].randoms]
    gate = ST.gate_decision(delta, cl, rand, calibration=calib, B=ctx.args.bootstrap, seed=0, advisory=ctx.smoke)
    gate.update({"model": ctx.short, "icl_identity": ctx.icl_identity, "battery_seed": ctx.args.battery_seed,
                 "smoke": ctx.smoke, "created": _now(), "cr20": AS.to_flat(conds[1].heads, ctx.H),
                 "clean_symrare": clean["summary"].get("symrare"), "fingerprint": clean["meta"].get("fingerprint")})
    _write_json(ctx.gate_path, gate)
    ctx.log("gate", passed=gate["passed"], proceed=gate["proceed"])
    ctx.say(("Gate PASS: " if gate["passed"] else "Gate nicht bestanden, beratend (Smoke): " if gate["advisory"]
             else "Gate STOPP: ") + gate["reason_de"])
    return gate


def wait_gate(ctx):
    """Instruct: polls the Base gate.json every 60 s for at most --gate_wait_min minutes (exit 3 when missing)."""
    deadline = time.time() + 60.0 * float(ctx.args.gate_wait_min)
    announced = False
    while not os.path.exists(ctx.gate_path):
        left = deadline - time.time()
        if left <= 0:
            raise AblateExit(EXIT_GATE, f"Gate des Base-Modells fehlt ({ctx.gate_path}) nach {ctx.args.gate_wait_min} "
                                        "min – Base-Notebook Stufe 0 abwarten und Stufe 0 erneut starten")
        if not announced:
            ctx.say(f"warte auf das Base-Gate {ctx.gate_path} (höchstens {ctx.args.gate_wait_min} min) ...")
            announced = True
        time.sleep(min(60.0, left))
    return _read_json(ctx.gate_path)


def gate_problem(ctx, gate):
    """None if this run may proceed on `gate`, else the German reason. A full run needs a passed gate of a full run;
    an advisory (not passed) gate is accepted only by a smoke run (F28), never by a full one."""
    status, geff = AR.gate_status(gate, smoke=ctx.smoke)
    if status in ("passed", "advisory"):
        return None
    if gate is None:
        return f"Gate fehlt ({ctx.gate_path}): zuerst Stufe 0" + ("" if ctx.is_base else " des Base-Modells")
    if gate.get("smoke") and not ctx.smoke:
        return (f"{AR.SMOKE_GATE_REASON} ({ctx.gate_path}) – Stufe 0 des Base-Modells ohne --smoke rechnen "
                "(oder diesen Lauf mit --smoke starten)")
    return "Gate nicht bestanden: " + ST.stop_text(geff.get("reason_de", ""))


def require_gate(ctx):
    gate = _read_json(ctx.gate_path)
    problem = gate_problem(ctx, gate)
    if problem is None:
        return gate
    if ctx.args.ignore_gate:
        ctx.say(f"WARNUNG: Gate ignoriert (--ignore_gate) – kein gültiger Befund ({problem})")
        return gate
    raise AblateExit(EXIT_GATE, problem)


def require_stage0(ctx):
    for name in ("refs.npz", "probes_m2.npz"):
        if not os.path.exists(os.path.join(ctx.run_dir, name)):
            raise AblateExit(EXIT_INPUTS, f"{os.path.join(ctx.run_dir, name)} fehlt: zuerst Stufe 0")
    _ = ctx.frozen
    if ctx.load_cond("clean__hh3") is None:
        raise AblateExit(EXIT_INPUTS, "clean__hh3 fehlt: zuerst Stufe 0")


def stage0(ctx):
    a = ctx.args
    ctx.log("stage0_start")
    ctx.say(f"Stufe 0: {ctx.short} ({'Base' if ctx.is_base else 'Instruct'}), ICL-Identität {ctx.icl_identity}"
            + (" [SMOKE]" if ctx.smoke else ""))
    table, table_path, synthetic = check_heads_table(ctx)
    if ctx.split is None or ctx.items_by_id is None:
        raise AblateExit(EXIT_INPUTS, "HH-Split / Items fehlen")
    try:
        AS.write_split(ctx.split_path, ctx.split)
    except AS.SplitMismatch as e:
        raise AblateExit(EXIT_INPUTS, str(e))
    AS.verify_split(ctx.split, ctx.items_by_id)
    ctx._cache.pop("split_sha", None)
    for role, n in (("means_ref", a.n_means_ref), ("means_probe", a.n_means_probe), ("eval", a.n_eval)):
        have = len(ctx.split["roles"][role])
        if n > have:
            raise AblateExit(EXIT_INPUTS, f"--n_{role} {n} ist größer als die Split-Rolle ({have})")
    check_n_eval(ctx)
    c = ctx.split["counts"]
    ctx.say(f"Split: means_ref {a.n_means_ref}/{c['means_ref']}, means_probe {a.n_means_probe}/{c['means_probe']}, "
            f"eval {a.n_eval}/{c['eval']} (Prompt-Cluster-disjunkt; sha256 {ctx.split_sha[:12]})")
    if os.path.exists(ctx.frozen_path):
        c33_heads = AS.set_heads(ctx.frozen, "C33")
    else:
        c33_heads = AS.named_c33(ctx.icl_identity) or AS.smoke_c33(ctx.L, ctx.H, ctx.band, ctx.seed)
    band_heads = [(l, h) for l in AS.band_layers(ctx.band) if l < ctx.L for h in range(ctx.H)]
    run_selftest(ctx, c33_heads, band_heads)
    ensure_refs(ctx)
    train_path = ensure_m2_train(ctx)
    freeze(ctx, table, table_path, synthetic)
    ensure_probes(ctx, train_path)
    ctx.log("first_eval")
    clean, _ = ctx.run_cond(_clean("hh3", 0, "0 clean"), load=True)
    floor_from(ctx, clean)
    if ctx.is_base:
        gate = run_gate(ctx)
    else:
        try:
            gate = wait_gate(ctx)
        except AblateExit:
            if not a.ignore_gate:
                raise
            gate = None
            ctx.say("WARNUNG: Base-Gate fehlt, ignoriert (--ignore_gate) – kein gültiger Befund")
        if gate is not None:
            status = AR.gate_status(gate, smoke=ctx.smoke)[0]
            ctx.say(f"Base-Gate gelesen: {'PASS' if status == 'passed' else ('beratend' if status == 'advisory' else 'STOPP')}"
                    f" – {gate.get('reason_de', '')}")
    ctx.log("stage0_done")
    problem = gate_problem(ctx, gate) if gate is not None else None
    if problem is not None and not a.ignore_gate:
        print(problem if problem.startswith(AR.SMOKE_GATE_REASON) else
              ST.stop_text(AR.gate_status(gate, smoke=ctx.smoke)[1].get("reason_de", "")), flush=True)
        try:                                     # the STOPP report (8.7) must reach the run folder, not only the output
            write_summary(ctx)
        except Exception as e:                   # noqa: BLE001 - a report problem must not hide the gate result
            ctx.say(f"WARNUNG: STOPP-Bericht nicht geschrieben ({type(e).__name__}: {e})")
        return EXIT_GATE
    return EXIT_OK


# ----------------------------------------------------------------------------------------------------------------
# pilot / ETA
# ----------------------------------------------------------------------------------------------------------------

def cond_kind(random):
    """"random" for a random-set / random-direction condition, else "named" (named sets and clean runs: M2 refit and
    sidecar on top of the forward)."""
    return "random" if random else "named"


def measured_rates(ctx, refresh=False):
    """(tok/s per variant, {"named", "random"}: median per-condition overhead in s) from the existing condition files
    (cached per call of the command; refresh=True re-reads, e.g. after the pilot). The overhead (wall - compute
    seconds) is estimated per kind, so the named-set refits are not spread over the random conditions."""
    if "rates" in ctx._cache and not refresh:
        return ctx._cache["rates"]
    by, over = {}, {"named": [], "random": []}
    if os.path.isdir(ctx.cond_dir):
        for fn in os.listdir(ctx.cond_dir):
            if not fn.endswith(".json") or ".tmp" in fn:
                continue
            c = _read_json(os.path.join(ctx.cond_dir, fn))
            st = (c or {}).get("stats") or {}
            if not st.get("seconds") or not st.get("tokens_processed"):
                continue
            v = c["variant"]
            by.setdefault(v, [0, 0.0])
            by[v][0] += st["tokens_processed"] * (1.0 / ETA_FACTOR.get(c["arm"], 1.0))
            by[v][1] += st["seconds"]
            if st.get("wall_seconds"):
                over[cond_kind(c.get("random"))].append(max(0.0, st["wall_seconds"] - st["seconds"]))
    rates = {v: t / s for v, (t, s) in by.items() if s > 0}
    hh = rates.get("hh3") or rates.get("hh2")
    for v in ("hh3", "hh2", "hhshuf", "ppl"):
        rates.setdefault(v, hh or STATIC_TOKS)
    rates.setdefault("icl", rates.get("icl_s1") or hh or STATIC_TOKS)
    rates.setdefault("icl_s1", rates["icl"])
    med = {k: float(np.median(v)) for k, v in over.items() if v}
    overhead = {k: med.get(k, med.get("named" if k == "random" else "random", 1.0)) for k in ("named", "random")}
    ctx._cache["rates"] = (rates, overhead)
    return ctx._cache["rates"]


def variant_tokens(ctx, variant):
    cache = ctx._cache.setdefault("variant_tokens", {})
    if variant not in cache:
        cache[variant] = _variant_tokens(ctx, variant)
    return cache[variant]


def _variant_tokens(ctx, variant):
    c = ctx.load_cond(f"clean__{variant}")
    if c and c.get("stats", {}).get("tokens_processed"):
        return float(c["stats"]["tokens_processed"])
    hh3 = ctx.load_cond("clean__hh3")
    base = float(hh3["stats"]["tokens_processed"]) if hh3 else 219.0 * ctx.args.n_eval
    if variant in ("hh2", "hhshuf"):
        return base * 146.0 / 219.0
    if variant == "ppl":
        return 2.0 * ctx.args.ppl_tokens
    if variant in ("icl", "icl_s1"):
        return 65000.0 if not ctx.smoke else 5000.0
    return base


def eta(ctx, conds, rates, overhead):
    """Rows (stage / arm, pending conditions, tokens, hours, CU) for the conditions without a file. overhead: seconds
    per condition, a number or {"named", "random"} (measured_rates)."""
    if not isinstance(overhead, dict):
        overhead = {"named": float(overhead), "random": float(overhead)}
    rows, seen = {}, set()
    for c in conds:
        if c.name in seen:
            continue
        seen.add(c.name)
        if os.path.exists(ctx.cond_path(c.name)):
            continue
        tok = variant_tokens(ctx, c.variant) * ETA_FACTOR.get(c.arm, 1.0)
        sec = tok / rates.get(c.variant, STATIC_TOKS) + overhead[cond_kind(c.r is not None or c.d is not None)]
        r = rows.setdefault(c.group, {"group": c.group, "n": 0, "tokens": 0.0, "seconds": 0.0})
        r["n"] += 1
        r["tokens"] += tok
        r["seconds"] += sec
    out = list(rows.values())
    for r in out:
        r["hours"] = r["seconds"] / 3600.0
        r["cu_a100"] = r["hours"] * CU_PER_A100_HOUR
    return out


def print_eta(ctx, rows, rates):
    cuda = str(ctx.args.device).startswith("cuda")
    print("[ablate] ETA der ausstehenden Bedingungen (gemessene Raten: " + ", ".join(
        f"{k} {v:.0f}" for k, v in sorted(rates.items())) + " tok/s):", flush=True)
    print(f"  {'Stufe / Arm':18s} {'n':>5s} {'M Tokens':>9s} {'Stunden':>8s} {'CU (A100)':>10s}")
    tot = {"n": 0, "tokens": 0.0, "hours": 0.0, "cu_a100": 0.0}
    for r in rows:
        print(f"  {r['group']:18s} {r['n']:5d} {r['tokens'] / 1e6:9.2f} {r['hours']:8.2f} {r['cu_a100']:10.1f}")
        for k in tot:
            tot[k] += r[k]
    print(f"  {'Summe':18s} {tot['n']:5d} {tot['tokens'] / 1e6:9.2f} {tot['hours']:8.2f} {tot['cu_a100']:10.1f}"
          + ("" if cuda else "   (CPU: CU-Spalte nur als A100-Umrechnung bedeutsam)"), flush=True)
    return tot


def full_plan(ctx, extend=False):
    out = plan_stage1(ctx, extend=extend)
    for arm in ARMS:
        if arm in ctx.args.arms and not (arm == "fixed_identity" and ctx.is_base):
            out += plan_arm(ctx, arm, extend=extend)
    return out


# ----------------------------------------------------------------------------------------------------------------
# stages 1 and 2, summary
# ----------------------------------------------------------------------------------------------------------------

def stage1(ctx):
    require_stage0(ctx)
    check_n_eval(ctx)
    require_gate(ctx)
    conds = plan_stage1(ctx, extend=True)
    if not conds:
        raise AblateExit(EXIT_INPUTS, "C33 und O sind leer – nichts zu tun")
    conds = [_clean("hh3", 1, "1 primär")] + conds          # recomputed on a key change (e.g. another batch plan)
    if ctx.args.pilot:
        pilot = [c for c in conds if c.name in ("mean__hh3__C33", "mean__hh3__C33__r000")]
        ctx.say("Pilot: " + ", ".join(c.name for c in pilot))
        ctx.execute(pilot)
        rates, overhead = measured_rates(ctx, refresh=True)
        rows = eta(ctx, full_plan(ctx), rates, overhead)
        tot = print_eta(ctx, rows, rates)
        gpu = None
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        _write_json(os.path.join(ctx.run_dir, "pilot.json"),
                    {"created": _now(), "device": ctx.args.device, "gpu": gpu, "dtype": ctx.dtype, "rates": rates,
                     "overhead_seconds": overhead, "measured_from": [c.name for c in pilot], "eta": rows,
                     "total": tot, "cu_per_a100_hour": CU_PER_A100_HOUR, "batch_tokens": ctx.args.batch_tokens,
                     "max_batch": ctx.args.max_batch})
        return EXIT_OK
    rates, overhead = measured_rates(ctx)
    print_eta(ctx, eta(ctx, conds, rates, overhead), rates)
    t0 = time.time()
    n = ctx.execute(conds)
    ctx.say(f"Stufe 1: {n} neue Bedingungen in {time.time() - t0:.0f}s")
    return write_summary(ctx)


def stage2(ctx):
    require_stage0(ctx)
    check_n_eval(ctx)
    require_gate(ctx)
    arms = [a for a in ARMS if a in ctx.args.arms]
    if ctx.is_base and "fixed_identity" in arms:
        arms.remove("fixed_identity")
        if ctx.args.arms != list(ARMS):
            ctx.say("fixed_identity: nur im Instruct-Lauf (übersprungen)")
    t0 = time.time()
    for arm in arms:
        if arm == "fixed_identity" and load_base_frozen(ctx) is None:
            ctx.say(f"fixed_identity übersprungen: {ctx.icl_identity}_frozen_sets.json fehlt (Base-Stufe 0 zuerst; "
                    "der nächste Lauf holt den Arm nach)")
            continue
        conds = plan_arm(ctx, arm, extend=True)
        if not conds:
            ctx.say(f"Arm {arm}: keine Bedingungen (leere Sätze)")
            continue
        rates, overhead = measured_rates(ctx)
        pend = eta(ctx, conds, rates, overhead)
        ctx.say(f"Arm {arm}: {len({c.name for c in conds})} Bedingungen, ausstehend "
                f"{sum(r['n'] for r in pend)} (~{sum(r['hours'] for r in pend):.2f} h)")
        ctx.execute(conds)
    ctx.say(f"Stufe 2 fertig nach {time.time() - t0:.0f}s")
    return write_summary(ctx)


def other_short(ctx):
    if ctx.is_base:
        return ctx.short[:-5] if ctx.short.endswith("-base") else None
    return ctx.icl_identity


def write_summary(ctx):
    """Report from the condition files only (no model): {short}_ablation.{json,md} and the CSVs."""
    if ctx.split is None:
        if not os.path.exists(ctx.split_path):
            raise AblateExit(EXIT_INPUTS, f"{ctx.split_path} fehlt: zuerst Stufe 0")
        ctx.split = AS.load_split(ctx.split_path)
    conds = AR.load_conditions(ctx.cond_dir)
    if not conds:
        raise AblateExit(EXIT_INPUTS, f"keine Bedingungen unter {ctx.cond_dir}")
    floor = _read_json(os.path.join(ctx.run_dir, "floor.json"))
    if "clean__hh3" in conds and (floor is None or floor.get("key") != conds["clean__hh3"].get("key")):
        floor_from(ctx, conds["clean__hh3"])                   # the floor follows the current clean__hh3
    specs = all_specs(ctx) if os.path.exists(ctx.frozen_path) else []
    frozen = ctx.frozen if os.path.exists(ctx.frozen_path) else None
    commit = next((c.get("commit_sha") for c in conds.values() if c.get("commit_sha")), None)
    dtype = next((c.get("dtype") for c in conds.values() if c.get("dtype")), ctx.dtype)          # as computed, not the
    attn_impl = next((c.get("attn_impl") for c in conds.values() if c.get("attn_impl")), ctx.attn_impl)   # local policy
    sel = _read_json(os.path.join(ctx.run_dir, "selftest.json")) or {}
    oshort = other_short(ctx)
    other = _read_json(os.path.join(ctx.abl, f"{oshort}_ablation.json")) if oshort else None
    a = ctx.args
    n_eval = ctx.stored_n_eval("clean__hh3") or a.n_eval                 # as computed, not the local argument
    if n_eval != a.n_eval:
        ctx.say(f"Hinweis: clean__hh3 wurde mit n_eval = {n_eval} gerechnet (Argument {a.n_eval}); der Bericht nennt "
                f"{n_eval}")
    info = {"model": ctx.short, "hf_id": ctx.hf_id, "revision": a.revision or "main", "commit_sha": commit,
            "dtype": dtype, "attn_impl": attn_impl, "is_base": ctx.is_base, "icl_identity": ctx.icl_identity,
            "transferred": not ctx.is_base, "smoke": ctx.smoke, "note": a.note, "ignore_gate": a.ignore_gate,
            "n_eval": n_eval, "n_means_ref": a.n_means_ref, "n_means_probe": a.n_means_probe, "B": a.bootstrap,
            "seed": ctx.seed, "R": {"C33": a.r_c33, "O": a.r_o, "small": a.r_small, "sec": a.r_sec},
            "band": list(ctx.band), "readout_layers": list(ctx.readout), "threeway_all": a.threeway_all,
            "batch_tokens": a.batch_tokens, "max_batch": a.max_batch, "cond_dir": ctx.cond_dir,
            "c33_synthetic": bool(frozen and frozen["sets"].get("C33", {}).get("synthetic")),
            "other_model": oshort, "split_sha": ctx.split_sha,
            "refs_sha": overlap.file_sha256(os.path.join(ctx.run_dir, "refs.npz"))
            if os.path.exists(os.path.join(ctx.run_dir, "refs.npz")) else None,
            "probes_sha": overlap.file_sha256(os.path.join(ctx.run_dir, "probes_m2.npz"))
            if os.path.exists(os.path.join(ctx.run_dir, "probes_m2.npz")) else None}
    t0 = time.time()
    payload = AR.build(conds, specs, ctx.clusters_of, info=info, gate=_read_json(ctx.gate_path),
                       floor=_read_json(os.path.join(ctx.run_dir, "floor.json")),
                       selftest={k: v for k, v in sel.items() if k in ("C33", "WholeBand")}, frozen=frozen,
                       B=a.bootstrap, seed=0, other=other)
    paths = AR.write_report(os.path.join(ctx.abl, f"{ctx.short}_ablation"), payload)
    ctx.say(f"Bericht ({time.time() - t0:.0f}s): " + ", ".join(paths.values()))
    if payload.get("stale"):
        ctx.say(f"WARNUNG: {len(payload['stale'])} Bedingungen übersprungen (Items weichen vom sauberen Lauf ab, z. B. "
                "andere N_EVAL): " + ", ".join(x["condition"] for x in payload["stale"][:10])
                + (" …" if len(payload["stale"]) > 10 else ""))
    for line in payload["verdict"]:
        print("  " + line, flush=True)
    return EXIT_OK


# ----------------------------------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------------------------------

def run_pipeline(args, model, tok, split, items_by_id, census, inject=None):
    """Stage / summary dispatch on given inputs (the CLI builds the real split, items and census and passes a lazy model
    loader; tests inject synthetic ones). model: a loaded torch model or a zero-argument loader. inject: optional
    {"ppl_ids": token ids for M4, "icl_items": callable(tok, seed, smoke) -> ICLItems}. Returns the exit code."""
    resolve_args(args)
    try:
        ctx = Ctx(args, model, tok, split, items_by_id, census, inject)
        if args.summary:
            return write_summary(ctx)
        if args.stage is None:
            raise AblateExit(EXIT_ERROR, "--stage 0|1|2 oder --summary angeben")
        if args.pilot and args.stage != 1:
            raise AblateExit(EXIT_ERROR, "--pilot gehört zu --stage 1")
        os.makedirs(ctx.cond_dir, exist_ok=True)
        if args.stage == 0:
            return stage0(ctx)
        if args.stage == 1:
            return stage1(ctx)
        return stage2(ctx)
    except AblateExit as e:
        print(f"[ablate] {e} (exit {e.code})", flush=True)
        return e.code


def load_split_and_items(abl_dir, verbose=True):
    """(split, {id: item}): the stored split file (verified against the data) or a freshly built one."""
    from .. import data
    path = os.path.join(abl_dir, "hh_helpful_split.json")
    if os.path.exists(path):
        split = AS.load_split(path)
        items = data.eval_pairs("hh_pairwise", "helpful", split="train")
        by_id = {int(it["group"]): it for it in items}
        AS.verify_split(split, by_id)
        if verbose:
            print(f"[ablate] Split {path} gelesen und verifiziert ({len(by_id)} Items)", flush=True)
        return split, by_id
    return AS.hh_helpful_split(verbose=verbose)


def run(args):
    resolve_args(args)
    try:
        if args.summary:
            return run_pipeline(args, None, None, None, None, None)
        if args.stage is None:
            raise AblateExit(EXIT_ERROR, "--stage 0|1|2 oder --summary angeben")
        short, hf_id, _ = config.resolve_model(args.model)
        icl_identity = resolve_icl_identity(args.icl_model, short)
        try:
            census = icl_sets.census_sets(icl_identity, source=args.icl_source, allow_download=not args.no_download)
        except (KeyError, FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            raise AblateExit(EXIT_INPUTS, f"ICL-Census für {icl_identity} nicht verfügbar: {e}")
        print(f"[ablate] Census {icl_identity}: pinned {census.pinned}, " +
              ", ".join(f"{k} {len(v)}" for k, v in census.roles.items()), flush=True)
        from .. import icl_battery as IB
        try:
            IB.load_concepts(verify=True)
        except (ValueError, OSError) as e:
            raise AblateExit(EXIT_INPUTS, f"data/icl/concept_tasks.json: {e}")
        from ..models import load_model, load_tokenizer
        tok, _ = load_tokenizer(hf_id, args.revision)
        abl = config.paths(args.root)["ablation"]
        split, items = load_split_and_items(abl, verbose=not args.quiet)

        def loader():
            return load_model(args.model, revision=args.revision, device=args.device, dtype=args.dtype,
                              attn_impl=args.attn_impl)[0]
        return run_pipeline(args, loader, tok, split, items, census)
    except AblateExit as e:
        print(f"[ablate] {e} (exit {e.code})", flush=True)
        return e.code
