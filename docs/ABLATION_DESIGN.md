# ABLATION_DESIGN — Kausaler Test: Wert-Heads × ICL-Schaltkreis (Qwen3-4B)

Implementation spec, status 2026-11-05. It turns the fixed design of the orchestrator ("FIXED DESIGN", decided in
advance) into modules, signatures, file formats, CLI flags, notebook cells, tests and a CPU smoke recipe. Every open
choice is fixed here (section 3). The headings are German; the technical content is English. Notebook prose is German
(like the existing notebooks). Code comments and docstrings are English.

References: `file.py:N` = this repository; `icl/…`, `main.tex:N`, `data/…`, `figures/…` = the ICL repository
andrewgcodes/in-context-learning-circuit at commit `26c44caf8f9b5a5a197988095f1f76b0562bbaa4` (MIT, read-only checkout at
`/home/user/andrewgcodes/in-context-learning-circuit`; `main.tex` = `paper/main.tex`).

---

## 0. Frage und Kurzfassung

**Question.** The top-15 % HH-"helpful" probe heads of Qwen3-4B(-Base) overlap the 33-head "named ICL circuit L20-27" of
arXiv 2611.05344. The overlap run found 21 of 33 for the Base model and 23 of 33 for the instruct model, both enriched
after Holm. Is this overlap causal? Does the model **rely** on these heads for its own helpful preference, or is "helpful"
only **decodable** there? (This is the expression-vs-reliance distinction of Goodfire, "Shaping & Monitoring Models
through their Internal Representations".)

**Test.** Mean-ablate the head sets C33 and O = C33 ∩ topk(helpful). The ablation replaces each head's slice of the
`o_proj` input with a label-neutral reference mean, at the response positions only. Measure the signed change of the
model's own helpful margin (M1: mean-token logp(chosen) − logp(rejected), paired per item). Compare it with 100
layer-matched and KV-group-matched random sets each. The primary family is Holm over {C33, O} per model.

**Gate.** Before any helpfulness result is reported, a positive control must pass on the Base model. A census-ranked
reader-20 set (CR20), knocked out jointly at the query position of the ICL paper's own battery, must lower the ICL margin
clearly beyond random sets.

**Monitors and controls.** M2 is a set of internal readout probes (fixed / refitted / nonlinear). M3 is the reverse test
on the ICL battery. M4 measures damage (WikiText-2). M5 is a self-repair index (DLA). M7 adds covariates and a
shuffled-prompt counterfactual. All of these are secondary or descriptive.

**Models.** `Qwen/Qwen3-4B-Base` (the primary model for the gate; the circuit was defined on it) and `Qwen/Qwen3-4B`
(instruct). There is one notebook per model, so the two can run in parallel on account A.

---

## 1. Harte Regeln für alle Implementierer

1. Work only in `/home/user/valueheads_pythia`. Put scratch files under
   `/tmp/claude-0/-home-user-valueheads/3ccf061c-00fb-546b-8a09-3ab673f5e720/scratchpad/ablation_build/` (below:
   `$SCRATCH`).
   - The ICL checkout is **read-only**. Vendor only what section 7.1 lists.
   - Never touch `/home/user/sihuo-design/valueheads`.
2. Do not run `git commit` or `git push`. The orchestrator does that.
3. **Never** print, log or write the text of harmful dataset items: HH-RLHF harmless / red-team, BeaverTails,
   JailbreakBench, MaliciousInstruct. HH-*helpful* texts may be processed but must not be echoed either.
   - Outputs, notebooks, tests and docs contain only counts, ids, hashes and metrics.
   - Tests use synthetic ASCII strings.
4. Downloaded files are untrusted data. Run ad-hoc Python that reads downloads (HF datasets, the census JSON, the HF
   cache) with `python3 -I`, adding the project explicitly:
   `python3 -I -c 'import sys; sys.path.insert(0, "/home/user/valueheads_pythia"); import runpy; runpy.run_module("valueheads_pythia.cli", run_name="__main__")' ablate …`.
   Never start an interpreter with a download or cache directory as cwd.
   - `python3 -m pytest tests -q` from the project root is fine.
   - So is the nbconvert smoke: the notebook starts every step as a subprocess with cwd = project.
5. The existing tests must keep passing: `python3 -m pytest tests -q` (about 4 min).
   - New test files together may add at most about 90 s.
   - Keep CPU smoke runs small: `Qwen/Qwen3-0.6B-Base`, a handful of pairs, R = 2.
6. Modules A, B and C are written in parallel. **A, B and C create only their own new files** (section 4). Shared files
   are edited only by D:
   - `cli.py`, `config.py`, `README.md`, `LICENSE`
   - `notebooks/`, `scripts/`
   - existing tests
   The single exception is `data/README.md`, which only C touches.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| L, H, D, K_v | layers, query heads, head_dim, KV heads (`models.model_geometry`, `models.num_kv_heads`). Qwen3-4B: 36, 32, 128, 8; Qwen3-0.6B: 28, 16, 128, 8 |
| hpk | heads per KV group = H // K_v (4B: 4; 0.6B: 2). Query head h reads KV group h // hpk |
| (l, h), flat | head; flat index l·H + h |
| z_{l,h,t} | slice `[h·D, (h+1)·D)` of the `o_proj` input of layer l at position t (`hooks.HeadHooks.head_slice`) |
| item i | one HH helpful pair; id = `group` of `data.eval_pairs('hh_pairwise','helpful',split='train')` = index in `data.hh_pairs('helpful','train')` |
| cluster(i) | prompt cluster: `ablation_sets.prompt_key(context_i)`; the bootstrap and the split are clustered by it |
| sequence | context + continuation, tokenised jointly by `eval_likelihood.encode_pair(tok, ctx, cont, max_length=4096)` (no truncation; assert `cut == 0`) |
| b, ln | boundary (= length of the context tokenisation) and sequence length. Token t ∈ [b, ln) is scored from position t − 1 |
| R_i | scored ("decision") positions [b − 1, ln − 2] |
| E_i | **response edit positions** [b − 1, ln − 1] = R_i ∪ {ln − 1} (decision F2) |
| mean-token logp | LL̄(y\|x) = Σ_{t∈[b,ln)} log p(y_t \| y_<t) / (ln − b), as `PairwiseMetric.item_values` `margin_mean` |
| Δ | paired per-item change, condition − clean, on identical items with an identical batch plan |

---

## 3. Festlegungen (alle offenen Wahlen)

Binding for A to D. A deviation needs a written reason in the docstring and in section 10.

| ID | Topic | Decision | Reason |
|---|---|---|---|
| F1 | Item source and formatting | `data.eval_pairs('hh_pairwise','helpful',split='train')`, raw HH text, no chat template, no BOS, for both models. Context = text up to and including the last `"\n\nAssistant:"`; continuation keeps its leading space. Tokenise with `encode_pair(..., max_length=4096)`. M1 sums and counts must equal `eval_likelihood.loglik_batched` on the same sequences | identical to the helpful probe text and to `hh_pairwise` scoring |
| F2 | Response positions | The edit mask for "response" policies is E_i = [b−1, ln−1]. b−1 is the last prompt token (`:` of `Assistant:`), which predicts the first response token. ln−1 is the probe read position: it changes no M1 log-prob, but it does feed M2. Prompt positions [0, b−2] are never edited | every scored log-prob is produced at an edited position; M2 at ln−1 sees the edit |
| F3 | Reference pooling ("mean") | μ_resp(l,h) = ½·mean_{i∈MR} p⁺_i + ½·mean_{i∈MR} p⁻_i, where p^±_i = mean of z over E_i of item i's chosen / rejected sequence. This is per-sequence pooling: each sequence counts once, whatever its length. The per-token pooled mean is stored as a diagnostic only | label-neutral; long responses (max 613 tokens) do not dominate |
| F4 | `mean_last` reference | μ_last = ½·mean z⁺_{ln−1} + ½·mean z⁻_{ln−1} over MR, applied only at ln−1. On M1 it is **structurally zero** (causal attention). It is reported as a causality sanity check (\|Δ\| must be ≤ 1e-6 in fp32 and ≤ 1e-3 in bf16, else flagged), never as a finding. Its real use is M2 | resolves the "mean_last on M1" gap |
| F5 | Splits | The MEANS split (512 pairs) = means_ref (256: references, DoM, P_DoM, donors) + means_probe (256: M2 probe training). EVAL = 1000 pairs. The three roles are disjoint by prompt cluster and disjoint from probe-train (groups **and** prompt text) and from probe-val (prompt text). The algorithm is in section 6.1. N_EVAL < 1000 takes a prefix of the stored EVAL walk order | design: three disjoint roles |
| F6 | Probe-train reconstruction | groups of `data.probing_set('helpful','train','pythia',max_samples=20000,seed=42)` (= 10 000 whole pairs). Verify that the heads table has `n_train == 20000`, else STOP | the overlap run used tier a100, seed 42; its feature dirs were ephemeral |
| F7 | Donor (resample) | donor d(i) = `np.random.default_rng([seed, crc32("donor"), item]).integers(n_means_ref)` (256; smoke 8). The donor is drawn from means_ref only and is shared by all sequences of item i. Donor vector = ½(p⁺_d + p⁻_d) (label-neutral), applied at E_i | seeded per item, independent of batch order |
| F8 | dir_mean | v_{l,h} = unit DoM, DoM = mean z⁺_{ln−1} − mean z⁻_{ln−1} over means_ref. z' = z − ⟨z,v⟩v + ⟨μ_resp,v⟩v at E_i. Controls: the same operation along R_SEC = 20 random unit directions per head; direction d of head (l,h) comes from `overlap._rng(seed, "ablation-dir-v1", identity, set, d, l, h).standard_normal(D)`, normalised | design (4) |
| F9 | gauss | z' = g·‖z^clean_t‖/‖g‖ at E_i. g ~ N(0, I_D) from `np.random.default_rng([seed, crc32("gauss"), item, role_code, l, h])`, one row per absolute position t. The **clean** norm comes from a truncated clean pre-pass on the same batch (section 5.4, `clean_prepass`) | exact "clean head-output norm at that position" |
| F10 | Random sets | Each named set gets its **own** profile null: `overlap.layer_matched_draws(heads, L, H, 1, rng_r, heads_per_kv=hpk)` with rng_r = `overlap._rng(seed, "ablation-random-v1", identity, set, r)`. Draw r depends only on (seed, identity, set, r): there is a prefix property, and the instruct run shares the C33 draws of the Base run. Draws may repeat or overlap named sets. R table: section 8.3 | design; KV-matched like the overlap null |
| F11 | Identity keys | `icl` sets (C33, C33_readers, C33_brakes, WholeBand, CR20, CensusReaders, CB24) use identity = the ICL base model short name (`qwen3-4b-base` for both 4B models). Table/activation sets (TopK_all, O, CminusO, P_band, P_DoM) use the short name of the model that defines them | same identities on both models (two-point trajectory) |
| F12 | P_band reuse | If P_band has the same per-layer counts **and** the same per-layer KV-count multisets as O, P_band is scored in variant `hh3` and compared with O's 100 Stage-1 random conditions (`"reuse": "O"`). Otherwise it gets its own R_SMALL = 50 draws in `hh2` | the design's "R_SMALL when sizes differ", made exact |
| F13 | P_DoM | Selected per model from that model's means_ref DoM: the top-\|O\| heads of the band by raw ‖DoM‖₂ that are not in C33 (ties broken by flat index). The scale dependence is documented. The fixed-identity arm also evaluates P_DoM of the Base model on the instruct model | Goodfire comparator of O; the design leaves per-model selection for pattern sets open |
| F14 | Fixed-identity arm (instruct) | evaluates O_base, CminusO_base, P_band_base, P_DoM_base and TopK_all_base, read from `results/ablation/qwen3-4b-base_frozen_sets.json` (randoms: the first R_SEC draws stored there), under `mean`/`hh2` with the instruct model's own references. The arm **requires** that file. If it is missing, the arm is skipped with a message and the next run fills it in (idempotent). The sets are never recomputed in the instruct run | no Base activations in the instruct notebook |
| F15 | M2 probes | Fixed head probes = `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)` on 128-dim head features, no scaling (as in `probing._fit_one`, mode pythia), one per readout head (default: the last 8 layers, 28-35). Fixed residual probe: the same on the **post-final-norm** hidden state at ln−1. MLP (L3): `StandardScaler` + `MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-2, learning_rate_init=1e-3, max_iter=500, random_state=42)` on the same residual. Training data: means_probe, clean. Refit (L2): 5-fold `StratifiedGroupKFold(shuffle=True, random_state=42)` on the ablated EVAL texts, groups = cluster, `probing_stats.torch_batched_logreg(C=1.0, class_weight="balanced", max_iter=200)`, out-of-fold AUC; computed for named sets and clean only. AUC = `probing_stats.auc_matrix` on decision scores (MLP: probability). Readout heads that belong to the ablated set are excluded | same objective as `probing.py`; "never read out at ablated heads" |
| F16 | Final residual | M2 reads the post-norm state (`decoder(...).last_hidden_state`, what `lm_head` reads; = `hidden_states[-1]` in transformers 5.x). M5 uses the pre-norm residual (input of `decoder.norm`) and its RMS | both are documented |
| F17 | Unrelated response / shuffled prompt | π = a seeded derangement of the active EVAL items with cluster(π(i)) ≠ cluster(i). It is drawn as `overlap._rng(seed, "ablation-derangement-v1", n_eval).permutation(n_eval)`, repeated until valid (at most 1000 tries). Unrelated response for prompt i = the chosen response of π(i). Shuffled prompt: chosen_i and rejected_i are scored after the context of π(i) | design: fixed seeded pairing |
| F18 | Sequence variants | `hh3` = (pos, neg, unrel) per item; `hh2` = (pos, neg); `hhshuf` = (spos, sneg); `ppl` = WikiText windows; `icl` = ICL target/alt rows. Every condition is paired with the **clean run of the same variant**, so batch composition is identical. The three-way margin exists only in `hh3`. That variant covers Stage 1 (C33, O and their randoms, plus clean) and P_band under F12; `--threeway_all` puts every M1 condition in `hh3` | bf16 numerics are batch-dependent; budget |
| F19 | Batching | `extract.plan_batches(lengths, max_batch, batch_tokens)`, right padding. Defaults: CUDA 8192 / 64, CPU 2048 / 16. Both values enter the condition key, so changing them forces a recompute. Logits only at scored positions, `lm_head` in chunks of 2048 positions, fp32 log-softmax | identical plans across conditions |
| F20 | Numerics | `config.dtype_policy`: bf16 + sdpa on CUDA, fp32 + eager on CPU. References are stored in fp32 and cast to the model dtype in the hook; donor means are stored in fp16 | repository policy |
| F21 | ICL battery | concept12n (4 concepts × 24 draws × 4 label sets = 384 prompts, 16 kinds). The **fresh draw is BATTERY_SEED = 2** (`random.Random(15)`). Seed 1 (`Random(14)`) reproduces the census draw and is used for calibration only | the census was selected on seed 1 (0.6B calibration within 0.02 nats) |
| F22 | Gate set | CR20 = the 20 census readers with the most negative dret (`readers` = dret_hi < 0, sorted by (dret, l, h)) of the identity census. It substitutes for the unreleased reader-20. Section 7.4 has the list | the paper's own census-ranked protocol (Llama, main.tex:394) |
| F23 | Gate rule | PASS ⇔ (a) the calibration on seed 1 is within 0.15 nats of D188_summary for symrare, nat, flip, R and P (4B Base; skipped in smoke), **and** (b) the CR20 Δdret 95 % cluster-bootstrap CI (B = 10 000, clusters = (concept, seq)) lies entirely below 0, **and** (c) Δdret(CR20) < every one of the R_SEC = 20 random layer/KV-matched 20-sets (one-sided p = 1/21 ≈ 0.048). The gate is evaluated on Base only; instruct waits for the Base `gate.json` | "clearly beyond random", stated explicitly |
| F24 | Primary family | per model: M1 Δmargin_mean under `mean`/`hh3` for C33 and O; one-sided empirical p in direction "le" (`stats.empirical_p`); Holm (`overlap.holm`), α = 0.05. Verdict "Abhängigkeit nachgewiesen" ⇔ Holm-p < 0.05 | design |
| F25 | Floor | clean acc_mean (mean-token pair accuracy, ties count as losses) with a 95 % cluster CI (B = 10 000). If lo ≤ 0.5 ≤ hi, the floor is reached: M1 cannot show reduced reliance, and only M2 and M3 are interpretable. If hi < 0.5, the clean preference is inverted (warning) | design |
| F26 | Bootstrap | vectorised cluster bootstrap (multinomial cluster counts, percentile CI), B = 10 000 for M1/M3/M4/M5/floor/gate and B = 1000 for M2 AUC CIs (cost). Bootstrap seed 0 everywhere. Smoke: B = 200 | design B = 10 000; M2 is descriptive |
| F27 | Seeds | `--seed 42` everywhere. All keys are listed in F7-F10, F17 and 6.1 | reproducible |
| F28 | Budget defaults | N_EVAL = 1000, R_C33 = 100, R_O = 100, R_SMALL = 50, R_SEC = 20, WikiText 50 000 tokens (window 1024 / stride 512, `min_context` default), B as F26. Smoke: 8 EVAL, 8 + 8 MEANS, every R = 2, B = 200, ICL smoke battery (16 prompts), WikiText 2048 tokens with window 256 / stride 128 | design |
| F29 | Directions (preregistered) | M1 (all arms, all quantities): "le"; M2 AUCs: "le"; M4 Δnll: "ge"; M3: CR20 / CensusReaders / C33 "le", CB24 "ge", O / P_band / TopK_all / WholeBand two-sided (exploratory reverse test); C33_readers / C33_brakes: two-sided in every metric | design |
| F30 | M4 | `mean` reference μ_resp applied at **all** positions (`policy="all"`) on WikiText-2 test windows (`eval_likelihood.PPLMetric(tok, max_tokens=50000)`). Statistic: token-weighted Δnll. The HH-harmless margin is optional and **off** (not required) | design |

---

## 4. Module, Zuständigkeiten, Abhängigkeiten

| Part | New files (owner only) | May import | Must not edit |
|---|---|---|---|
| **A** | `valueheads_pythia/ablation.py`, `tests/test_ablation_hooks.py` | `hooks`, `models`, `extract.plan_batches`, `overlap._rng`, `eval_likelihood` (tests only), torch, numpy | any existing file |
| **B** | `valueheads_pythia/ablation_sets.py`, `valueheads_pythia/ablation_stats.py`, `tests/test_ablation_sets_stats.py` | `overlap` (`layer_matched_draws`, `_rng`, `holm`, `two_sided`, `load_value_table`, `file_sha256`), `stats.empirical_p`, `probing_stats` (`auc_matrix`, `WeightedAUC`), `data` (only inside `hh_helpful_split`), `icl_sets` (types only), numpy, pandas | any existing file |
| **C** | `valueheads_pythia/icl_battery.py`, `data/icl/concept_tasks.json` (copied unchanged), `tests/test_icl_battery.py`, an entry in `data/README.md` | `hooks.HeadHooks`, `hooks.ActivationCollector`, `models`, `extract.plan_batches`, `icl_sets`, torch, numpy | any other existing file |
| **D** (after A-C) | `valueheads_pythia/commands/ablate.py`, `valueheads_pythia/ablation_metrics.py`, `valueheads_pythia/ablation_report.py`, `scripts/make_ablation_notebook.py`, `notebooks/runs/qwen3-4b-base-ablation.ipynb`, `notebooks/runs/qwen3-4b-ablation.ipynb`, `tests/test_ablate.py` | everything | — |
| **D** (shared edits) | `cli.py` (COMMANDS entry), `config.py` (`paths()["ablation"]`), `README.md` (new section, Projektstruktur, Attribution), `LICENSE` (ICL attribution, section 8.10) | | |

C is independent of A on purpose: the ICL knock (one position per row, per-kind reference) is small and gets its own
`QueryKnock`. D calls only the public functions listed in sections 5-7 (the "contract"). If an implementer needs to
change a signature, the change must be reported back to the orchestrator.

```
A ablation.py ─┐
B ablation_sets.py, ablation_stats.py ─┼─> D commands/ablate.py + ablation_metrics.py + ablation_report.py ─> notebooks, README
C icl_battery.py ─┘
```

---

## 5. Modul A — `valueheads_pythia/ablation.py`

Head interventions for the causal test, batched teacher-forced scoring with per-batch metadata, collectors, reference
statistics, direct-effect helper and hook self-tests. No data loading, no file layout logic beyond `save_refs` and
`load_refs`.

### 5.1 Sequenzen und Batches

```python
ROLE_CODES = {"pos": 0, "neg": 1, "unrel": 2, "spos": 3, "sneg": 4, "ppl": 5}
POLICIES = ("response", "last", "all", "query")

@dataclass(frozen=True)
class Seq:
    ids: tuple            # token ids (no padding), len >= 2
    boundary: int         # b, 1 <= b <= len(ids); tokens [b, len) are scored
    item: int             # item id (HH group id; PPL window index)
    role: str             # key of ROLE_CODES

def edit_mask(policy, lengths, boundaries, width, device=None) -> torch.BoolTensor   # (B, width)
    # response: b-1 <= t <= ln-1 ; last: t == ln-1 ; all: t < ln ; query: t == b-1 ; pads always False

@dataclass
class Batch:
    index: np.ndarray            # (B,) positions in the `seqs` list of run_sequences
    input_ids: torch.Tensor      # (B, S) long, on model.device, right padded
    attention_mask: torch.Tensor # (B, S)
    lengths: torch.Tensor        # (B,) long (device)
    boundaries: torch.Tensor     # (B,) long (device)
    items: np.ndarray            # (B,) int64
    roles: list                  # B role strings
    hidden: torch.Tensor = None  # post-final-norm hidden (B, S, hidden); set by the runner before end_batch
    def mask(self, policy) -> torch.Tensor   # cached edit_mask(policy, ...)

class BatchHook:
    """Mixin: begin_batch(batch) before the forward, end_batch(batch) after it (both no-ops by default)."""
```

### 5.2 Runner

```python
@dataclass
class RunResult:
    sums: np.ndarray          # float64 (N,): sum of log p over scored tokens [b, ln)
    counts: np.ndarray        # int64 (N,): ln - b
    tokens_processed: int     # sum of unpadded lengths
    tokens_padded: int        # sum of B * S over batches
    tokens_scored: int
    seconds: float
    @property
    def tokens_per_second(self) -> float   # tokens_processed / seconds

@torch.no_grad()
def run_sequences(model, seqs, pad_id, hooks=(), *, batch_tokens=8192, max_batch=64, score=True, logit_chunk=2048,
                  progress=False, desc=None) -> RunResult
```

- `hooks` is a list. Elements with `install()` / `remove()` (`HeadHooks` subclasses and `ResidualCapture`) are installed
  in list order on entry and removed on exit (pre-hook order = list order: **edit hooks first, collectors after**, so
  collectors read the edited input). Every element that
  is a `BatchHook` gets `begin_batch` / `end_batch`.
- The runner calls `model.get_decoder()(input_ids=…, attention_mask=…, use_cache=False)`. It gathers the
  `last_hidden_state` rows at R_i, applies `model.get_output_embeddings()` in chunks of `logit_chunk` positions,
  takes an fp32 `log_softmax` and gathers `ids[t+1]`. With `score=False` there is no `lm_head` (pure collection passes).
- It sets `model.eval()` and restores the training mode afterwards.
- Equivalence (test): `sums` and `counts` equal `eval_likelihood.loglik_batched(model, [s.ids for s in seqs],
  [s.boundary for s in seqs], pad_id, batch_tokens, max_batch)` within 1e-5 in fp32. The batch plan is the same
  `plan_batches`.

### 5.3 Collectors

```python
def final_norm(model) -> torch.nn.Module     # Qwen3/Llama: get_decoder().norm ; GPT-NeoX: gpt_neox.final_layer_norm

class LastTokenCollector(HeadHooks, BatchHook):
    def __init__(self, model, layers, n_seqs)          # positions="all", decode_only=False
    values: np.ndarray                                 # float32 (n_seqs, len(layers), H*D): o_proj input at ln-1

class PooledCollector(HeadHooks, BatchHook):
    def __init__(self, model, n_seqs, policy="response", layers=None, store_dtype=np.float16)
    values: np.ndarray                                 # (n_seqs, len(layers), H*D): per-sequence mean over policy positions (fp32 on device)
    token_sum: dict                                    # role -> float64 (len(layers), H*D) sum over all policy tokens
    token_count: dict                                  # role -> int

class ResidualCapture(BatchHook):
    def __init__(self, model, n_seqs, positions="last")     # pre-hook on final_norm(model); install()/remove() like HeadHooks
    post_last: np.ndarray     # float32 (n_seqs, hidden): batch.hidden at ln-1 (M2 residual feature, F16)
    pre_last: np.ndarray      # float32 (n_seqs, hidden): pre-norm residual at ln-1
```

### 5.4 Interventionen

All of these share one base class. The constructor arguments are binding.

```python
class MaskedHeadEdit(HeadHooks, BatchHook):
    """Edits the o_proj-input slices of `heads` at the positions of `policy` (sequence-aware, right-padding safe)."""
    def __init__(self, model, heads, policy="response")
        # heads: iterable of (layer, head) or {layer: [heads]}; empty -> no layer hooked (identity)
        # super().__init__(model, positions="all", decode_only=False, layers=<layers of heads>)
    def begin_batch(self, batch)          # stores batch and batch.mask(policy)
    def end_batch(self, batch)            # clears them
    def edit(self, layer, x)              # view (B,S,H,D); new = replacement(...); torch.where(mask, new, old) on the set heads only
    def replacement(self, layer, head_idx, z, batch) -> torch.Tensor   # z: (B,S,n,D) current values; abstract

class MeanReplace(MaskedHeadEdit):
    def __init__(self, model, heads, ref, policy="response")
        # ref: (L, H*D) or (L*H, D) array -> per-layer (n, D) tensors; arms mean ("response"), mean_last ("last"), M4 ("all")

class DonorReplace(MaskedHeadEdit):
    def __init__(self, model, heads, donor_means, donor_of, policy="response")
        # donor_means: (n_donor, L, H*D) (fp16 ok); donor_of: dict item -> donor index (F7)

class DirectionReplace(MaskedHeadEdit):
    def __init__(self, model, heads, ref, directions, policy="response")
        # directions: {(l, h): unit vector (D,)}; z' = z - <z,v>v + <mu,v>v (F8)

class GaussReplace(MaskedHeadEdit):
    def __init__(self, model, heads, seed, policy="response")
        # begin_batch runs clean_prepass(..., what="norm"); z' = g * ||z_clean|| / ||g|| (F9)

class OwnReplace(MaskedHeadEdit):
    def __init__(self, model, heads, policy="response")
        # self-test only: begin_batch runs clean_prepass(..., what="z"); replacement = the item's own clean z
```

Rules:

- With `policy="all"` and no `begin_batch`, the mask is "every position". This lets `MeanReplace(policy="all")` run
  inside `eval_likelihood.run_condition` / `PPLMetric.evaluate`, which accept `positions="all"` and
  `decode_only=False` hooks. Any other policy without `begin_batch` raises `RuntimeError` (no silent no-op).
- Width check and GQA are as in `HeadHooks`: every query head owns its slice, and editing head h never touches the other
  heads of its KV group.
- The edit clones `x` once per hooked layer and assigns through a view. Layers without set heads are not hooked.

Helpers:

```python
def donor_assignment(items, n_donor, seed) -> dict          # F7
def random_directions(heads, head_dim, seed, identity, set_name, index) -> dict   # F8; float32 unit vectors
def gauss_noise(seed, item, role, layer, head, length, head_dim) -> np.ndarray   # (length, D) float32 (F9)
def clean_prepass(model, batch, heads, what="z") -> dict    # {layer: (B,S,n,D) or (B,S,n)}
    # runs under `suspended()`: a module-level context manager under which every MaskedHeadEdit and every collector of
    # this module is a no-op (no edits, no recording, no double-counted token sums); the forward is aborted by an
    # exception raised in a forward hook on decoder block max(layer) (saves the upper layers + lm_head)
def suspended() -> contextlib.AbstractContextManager
```

### 5.5 Referenzstatistik

```python
@dataclass
class RefStats:
    mu_response: np.ndarray        # float32 (L, H*D), F3
    mu_last: np.ndarray            # float32 (L, H*D), F4
    mu_response_tokens: np.ndarray # float32 (L, H*D), per-token pooled (diagnostic)
    dom: np.ndarray                # float32 (L*H, D): mean z+_{ln-1} - mean z-_{ln-1}
    donor_means: np.ndarray        # float16 (n_pairs, L, H*D): 0.5 * (p+_i + p-_i), F7
    items: np.ndarray              # int64 (n_pairs,) item ids, aligned with donor_means
    meta: dict                     # model, dtype, attn_impl, n_pairs, policy, created

def compute_refs(model, pos_seqs, neg_seqs, pad_id, *, batch_tokens=8192, max_batch=64, progress=False) -> RefStats
    # pos_seqs[i] / neg_seqs[i] belong to the same item (asserted); one pass with PooledCollector("response") + LastTokenCollector(all layers)
def dom_norms(dom) -> np.ndarray        # (L*H,) float64
def save_refs(path, refs) -> str        # npz (meta as a JSON string entry), atomic tmp + os.replace; returns sha256
def load_refs(path) -> RefStats
```

### 5.6 Direkter Effekt (Hilfe für M5)

```python
def direct_effect(model, seqs, pad_id, heads, ref, *, policy="response", batch_tokens=8192, max_batch=64,
                  logit_chunk=2048) -> dict
    # one clean forward: z of the set heads at the policy positions, pre-norm residual r_t and rms_t = sqrt(mean(r_t^2) + eps)
    # (fp32, eps = final_norm(model).variance_epsilon) at the scored positions, clean logits l_t.
    # d_resid_t = sum_{(l,h) in set} (mu_{l,h} - z_{l,h,t}) @ W_O^{(l)}[:, slice_h].T     (only t in policy positions)
    # d_logit_t = W_U (g * d_resid_t / rms_t)         (g = final_norm weight; frozen clean norm scale)
    # de_t = log_softmax(l_t + d_logit_t)[y_{t+1}] - log_softmax(l_t)[y_{t+1}]
    # returns {"de_sums": float64 (N,), "counts": int64 (N,), "clean_sums": float64 (N,)}
    # raises if o_proj has a bias (the formula assumes Qwen3's bias-free o_proj)
```

### 5.7 Hook-Selbsttest (Stage-0-Gate)

```python
def hook_selftest(model, seqs, pad_id, heads, ref, *, batch_tokens=2048, max_batch=8) -> dict
    # returns {"ok": bool, "dtype": str, "n_seqs": int, "checks": {name: {"ok": bool, "value": float, "tol": float}}}
```

| Check | Procedure | Pass |
|---|---|---|
| `empty_set` | `MeanReplace(model, [], ref)` vs no hook, full logits of all valid positions | max abs diff == 0.0 |
| `own_reference` | `OwnReplace(model, heads)` (the reference is the item's own clean z at E_i) vs clean | ≤ 1e-5 (fp32), ≤ 1e-3 (bf16/fp16) |
| `locality` | `MeanReplace` with a sentinel ref (1e3) on one head; a collector registered after it captures the edited input of that layer | only that head's slice changes, and the sentinel sits at every E_i position; E_i is rebuilt per row as `range(b - 1, ln)` from lengths and boundaries, never taken from `edit_mask` / `Batch.mask` (the mask under test) |
| `prompt_untouched` | same forward as `locality` | that head's prompt positions [0, b − 2] and pads are bitwise equal to clean |
| `first_token` | same forward as `locality`, scored | log p of the first response token (scored from b − 1) changes for every sequence (\|Δ\| > 0) |
| `causality_last` | `MeanReplace(policy="last")` | `RunResult.sums` bitwise equal to clean |
| `dir_projection` | `DirectionReplace` with random v; capture edited z' | ⟨z',v⟩ = ⟨μ,v⟩ and z' − z ∥ v (rel 1e-4 fp32 / 1e-2 bf16) |
| `gauss_norm` | `GaussReplace`; capture edited z' | ‖z'‖ / ‖z_clean‖ − 1 within 1e-4 fp32 / 1e-2 bf16 |

### 5.8 Tests `tests/test_ablation_hooks.py` (CPU, < 30 s, no download)

The file copies the `tiny_qwen3` / `CharTokenizer` helpers from `tests/test_qwen_port.py` (no cross-test imports).
Use fp32 eager, tiny models with GQA (4 heads / 2 KV) and `hidden != H·D`.

1. The runner equals `loglik_batched` (sums and counts) on right-padded mixed-length sequences, with and without
   `score`.
2. Every row of `hook_selftest` passes on the tiny model, including `empty_set` (bitwise) and `own_reference`.
3. `edit_mask`: all four policies on hand-built lengths and boundaries, including pads and b = 1.
4. Locality and GQA: editing head h leaves every other slice, including the KV siblings of h, bitwise equal.
5. Causality: under `policy="last"` the scored log-probs are bitwise equal to clean.
6. `compute_refs` equals a brute-force loop for μ_resp (per-sequence pooling with 0.5/0.5 weights on an imbalanced
   label count), μ_last, dom and donor means.
7. `DonorReplace`: the same donor for every role of an item, and the result does not depend on batch order (shuffled
   seqs give identical per-seq results).
8. `GaussReplace`: deterministic per (item, role), independent of batch composition, norm preserved.
9. `DirectionReplace` projection identities; `random_directions` are unit vectors and seeded.
10. `MeanReplace(policy="all")` works inside `eval_likelihood.loglik_batched` (no `begin_batch`) and equals the masked
    variant with `policy="all"` driven by the runner. Another policy without a batch raises `RuntimeError`.
11. `clean_prepass` (truncated) equals the values from a full forward with `LastTokenCollector` / `PooledCollector`.
12. `direct_effect`: on a 1-layer tiny model with the MLP zeroed, the direct effect of mean-replacing one head equals the
    true effect (no indirect path) within 1e-5. On the 3-layer model, it equals a manual numpy recomputation of the
    formula.
13. `save_refs` / `load_refs` round trip (atomic, sha256 stable).

**Done when:** every test passes, the full suite is green, and nothing outside the two files has changed.

---

## 6. Modul B — `ablation_sets.py` und `ablation_stats.py`

### 6.1 HH-Split (in `ablation_sets.py`)

```python
SPLIT_VERSION = 1
SPLIT_KEY = "hh_helpful_split_v1"

def prompt_key(context) -> str      # sha256(" ".join(context.split()).encode()).hexdigest()[:16]
def item_sha(item) -> str           # sha256(context + "\x1e" + pos + "\x1e" + neg)[:16]  (verification, no text stored)

def build_split(items, probe_train_groups, probe_train_prompt_keys, probe_val_prompt_keys, *, n_means_ref=256,
                n_means_probe=256, n_eval=1000, seed=42) -> dict           # pure, testable offline
def hh_helpful_split(*, n_means_ref=256, n_means_probe=256, n_eval=1000, seed=42, probe_max_samples=20000,
                     probe_seed=42) -> tuple                                # (split dict, {item id: item}); loads data (online)
def write_split(path, split) -> str          # atomic; returns sha256 of the file; identical content if written twice
def load_split(path) -> dict
def verify_split(split, items_by_id) -> None # recomputes item_sha for every listed id; raises on mismatch
def split_items(split, items_by_id, role, n=None) -> list    # role in means_ref | means_probe | eval; walk-order prefix
```

Algorithm of `build_split`:

1. Pool = items whose `group` is not in `probe_train_groups`. Expected: 12 822 items, 2 836 in the pool (scouting).
2. Drop items for the first matching reason, in this order, and count each reason:
   - `prompt_in_probe_train`: `prompt_key(context)` ∈ keys of the probe-train pairs (≈ 23). The context of a pair is
     its chosen text up to and including the last `"\n\nAssistant:"`, built for the 10 000 probe-train groups.
   - `prompt_in_probe_val`: the key is among the 700 HF-test contexts of `eval_pairs(..., split='val')` (≈ 1).
   - `empty_response`: `pos.strip() == ""` or `neg.strip() == ""` (≈ 3).
   - `identical`: `pos == neg`.
   About 2 809 items remain.
3. Clusters = items grouped by `prompt_key`. Order = `np.random.default_rng([seed, zlib.crc32(SPLIT_KEY.encode())])`
   permutation of the sorted unique keys.
4. Fill the roles one after another (means_ref, then means_probe, then eval). For each role, walk the cluster order:
   take every not-yet-used cluster that fits (count + size ≤ target) and skip the ones that do not fit, until the target
   is reached exactly. Raise `ValueError` if a target cannot be reached. Items inside a role keep walk order (cluster
   order, then group id).
5. The rest is `reserve` (count only).

`hh_helpful_split` builds the inputs:

- `data.eval_pairs('hh_pairwise','helpful',split='train')`
- `data.probing_set('helpful','train','pythia',max_samples=20000,seed=42)` groups
- `data.hh_pairs('helpful','train')` for the probe-train contexts
- `data.eval_pairs('hh_pairwise','helpful',split='val')` for the probe-val keys

It asserts 20 000 texts / 10 000 groups, prints counts only, and returns the items (texts stay in memory, never written).

Split file `results/ablation/hh_helpful_split.json` (model-independent, shared by both notebooks; written atomically, so
two writers produce the same bytes):

```json
{"version": 1, "key": "hh_helpful_split_v1", "seed": 42, "created": "...",
 "source": {"dataset": "Anthropic/hh-rlhf", "subset": "helpful-base", "hf_split": "train",
            "items": "data.eval_pairs('hh_pairwise','helpful',split='train')"},
 "probe_train": {"rule": "data.probing_set('helpful','train','pythia',max_samples=20000,seed=42)", "n_texts": 20000,
                 "n_groups": 10000, "groups_sha256": "..."},
 "probe_val": {"rule": "data.eval_pairs('hh_pairwise','helpful',split='val') (HF test)", "n_items": 700, "keys_sha256": "..."},
 "counts": {"items": 12822, "not_in_probe_train": 2836, "dropped": {"prompt_in_probe_train": 23, "prompt_in_probe_val": 1,
            "empty_response": 3, "identical": 0}, "pool": 2809, "clusters": 0, "means_ref": 256, "means_probe": 256,
            "eval": 1000, "reserve": 1297},
 "roles": {"means_ref": [1234, ...], "means_probe": [...], "eval": [...]},
 "cluster": {"1234": "a1b2c3d4e5f60718", ...},
 "item_sha": {"1234": "0f1e2d3c4b5a6978", ...}}
```

(The numbers above are scouting values. The implementation writes the actual counts.)

### 6.2 Kopf-Sätze (in `ablation_sets.py`)

```python
BAND = (20, 27)
SET_ORDER = ("C33", "O", "CminusO", "P_band", "P_DoM", "C33_readers", "C33_brakes", "WholeBand", "TopK_all",
             "CR20", "CensusReaders", "CB24")
ICL_SETS = ("C33", "C33_readers", "C33_brakes", "WholeBand", "CR20", "CensusReaders", "CB24")   # identity = ICL base model

def parse_layers(spec) -> tuple                     # "20-27" -> (20, ..., 27); "28,30-31" allowed
def to_flat(heads, n_heads) -> list ; def from_flat(flat, n_heads) -> list
def layer_profile(heads) -> dict                    # {layer: count}
def kv_profile(heads, heads_per_kv) -> dict         # {layer: tuple of per-group counts, descending, zeros dropped}
def same_profile(a, b, heads_per_kv) -> bool
def census_ranked(effects, members, k, largest=False) -> list   # sort by (dret, l, h) ascending (largest: by (-dret, l, h))
def select_p_band(topk, c33, o, band, *, seed, identity) -> tuple   # (heads, info) — rule below
def select_p_dom(dom_norm, n_heads, band, exclude, size) -> tuple   # (heads, info) — F13
def named_sets(*, topk, c33, census_roles, census_effects, dom_norm, n_layers, n_heads, band=BAND, seed=42,
               model_identity, icl_identity) -> dict                # name -> {"heads", "flat", "n", "identity", "rule", "info"}
```

| Set | Rule |
|---|---|
| C33 | `icl_sets.NAMED["qwen3-4b-base"]["named_circuit_L20_27"].heads` (33 heads; layers 20:1, 21:4, 22:12, 23:6, 24:5, 25:2, 27:3; 22 KV groups: 13 singletons, 8 pairs, 1 quad) |
| TopK_all | `overlap.load_value_table(table, "helpful", "pythia").topk` (k = 172 for 36×32) |
| O | C33 ∩ TopK_all (Base 21, instruct 23 per the overlap report) |
| CminusO | C33 − O (Base 12, instruct 10) |
| P_band | see the rule below (candidates: Base 42, instruct 53) |
| P_DoM | F13 |
| C33_readers / C33_brakes | C33 ∩ census readers / brakes of the identity census (12 / 10 heads, listed in Anhang B) |
| WholeBand | all H heads of every band layer (256) |
| CR20 | `census_ranked(dret, readers, 20)` (F22; list in 7.4) |
| CensusReaders | all census readers (89 on 4B, 70 on 0.6B) |
| CB24 | `census_ranked(dret, brakes, 24, largest=True)` (fewer if the census has fewer brakes: 0.6B has 13) |

P_band rule (deterministic):

1. Target counts t_l = `layer_profile(O)`.
2. Candidates cand_l = sorted((TopK_all ∩ layer l) − C33) for l in the band.
3. rng = `overlap._rng(seed, "ablation-pband-v1", identity)`.
4. Pass 1 (ascending l): take min(t_l, |cand_l|) heads from cand_l via `rng.choice(len, k, replace=False)`.
5. Pass 2: for every layer l with a deficit (ascending l) and every missing slot, pick from the nearest layer l' that
   still has candidates, by (|l' − l|, l'), with a uniform `rng.integers`. Record each move (l → l').
6. If the candidates run out, P_band is smaller than |O|; record `info["short_by"]`.

Smoke (models without a named circuit, e.g. `qwen3-0.6b-base`): the set **keeps the name `C33`** (so every code path is
the same) but holds a synthetic stand-in of 8 band heads, drawn with `overlap._rng(seed, "ablation-smoke-c33")`,
identity `"smoke"`. Its frozen entry has `"synthetic": true` and `"rule": "SMOKE-ONLY synthetic stand-in (no named
circuit for this model)"`. The frozen file and every report carry `"smoke": true` and print "C33 = synthetischer
Smoke-Ersatz".

### 6.3 Zufallsmengen

```python
def random_draws(heads, *, n_layers, n_heads, heads_per_kv, seed, identity, set_name, R, start=0) -> list
    # [sorted flat list for r in range(start, start + R)], F10
```

### 6.4 Frozen-Set-Datei `results/ablation/{short}_frozen_sets.json`

It is written in Stage 0 **before** the first EVAL forward, after `compute_refs` (P_DoM needs the DoM).

```python
def build_frozen(*, model_short, icl_identity, geometry, band, seed, sets, randoms, inputs, smoke=False) -> dict
def write_frozen(path, payload, force=False) -> str
    # atomic; if the file exists with different "sets"/"random" content -> FrozenMismatch unless force; returns sha256
def load_frozen(path) -> dict
def set_heads(frozen, name) -> list                 # [(l, h)]
def random_heads(frozen, name, r) -> list           # [(l, h)] of draw r (follows "reuse")
def extend_random(frozen, name, R_new, ...) -> dict # appends draws r >= stored R (prefix property); records "extended"
```

```json
{"version": 1, "model": "qwen3-4b-base", "icl_identity": "qwen3-4b-base", "smoke": false, "created": "...", "seed": 42,
 "geometry": {"layers": 36, "heads": 32, "kv_heads": 8, "heads_per_kv": 4, "head_dim": 128}, "band": [20, 27],
 "inputs": {"heads_table": {"path": "results/heads/qwen3-4b-base_main_helpful_pythia.json", "sha256": "...", "k": 172,
                            "n_train": 20000, "n_val": 1400, "meta": {"mode": "pythia", "score": "auc", "feature_mode": "head"}},
            "census": {"file": "data/D188_heads_Qwen3-4B-Base_lbD188SC4.json", "sha256": "...", "pinned": true},
            "refs": {"path": "results/ablation/qwen3-4b-base/refs.npz", "sha256": "..."},
            "split": {"path": "results/ablation/hh_helpful_split.json", "sha256": "..."}},
 "sets": {"C33": {"heads": [[20, 26], ...], "flat": [666, ...], "n": 33, "identity": "qwen3-4b-base",
                  "rule": "icl_sets named_circuit_L20_27", "layers": {"20": 1, ...}, "kv": {"20": [1], ...}, "info": {}},
          "P_band": {"...": "...", "info": {"candidates": 42, "moves": [[26, 25]], "short_by": 0}},
          "P_DoM": {"...": "...", "info": {"dom_norm": {"651": 3.21, ...}, "note": "raw norm, scale dependent"}}},
 "random": {"C33": {"R": 100, "identity": "qwen3-4b-base", "draws": [[...], ...]},
            "P_band": {"reuse": "O"},
            "CminusO": {"R": 50, "identity": "qwen3-4b-base", "draws": [...]}}}
```

### 6.5 Statistik `valueheads_pythia/ablation_stats.py`

```python
ALPHA = 0.05

def cluster_bootstrap(values, clusters, *, B=10000, seed=0, weights=None, ci=0.95, chunk=500) -> dict
    # statistic sum(w*d)/sum(w); cluster counts ~ Multinomial(G, 1/G) drawn in chunks (RandomState(seed)); percentile CI;
    # NaN items dropped with their weight/cluster; returns {"mean", "lo", "hi", "se", "n", "n_clusters", "B", "ci"}
    # (same semantics as stats.paired_delta_ci, vectorised)
def cluster_bootstrap_ratio(num, den, clusters, *, B=10000, seed=0, ci=0.95) -> dict   # ratio of means (M5 SRI); NaN if den CI contains 0
def paired_auc_delta(scores_ref, scores_cond, labels, clusters, *, B=1000, seed=0, ci=0.95) -> dict
    # scores (K, N) or (N,); statistic = mean over rows of AUC(cond) - AUC(ref) (probing_stats.WeightedAUC);
    # returns {"auc_ref", "auc_cond", "delta", "lo", "hi", "B", "k"}
def vs_random(observed, random_values, direction) -> dict
    # {"p_le", "p_ge" (stats.empirical_p), "p_one" (in `direction`; two-sided designation -> NaN), "p_two" (overlap.two_sided),
    #  "z" ((obs - mean) / std ddof=1), "rand_mean", "rand_std", "rand_min", "rand_max", "rand_q025", "rand_q975", "R"}
def primary_holm(tests, alpha=ALPHA) -> list         # tests: [{"set", "p_one", "observed"}]; adds "p_holm", "significant"
def floor_check(win, margin, len_hi_gt_lo, clusters, *, B=10000, seed=0) -> dict
    # {"acc", "acc_lo", "acc_hi", "margin", "margin_lo", "margin_hi", "len_baseline", "floor": bool, "inverted": bool, "text_de"}
def gate_decision(delta_dret, clusters, random_means, *, calibration=None, B=10000, seed=0, advisory=False) -> dict
    # F23: {"passed", "advisory", "criteria": {"calibration", "ci_below_zero", "below_all_random"}, "delta", "lo", "hi",
    #       "random": vs_random(...), "reason_de"}
def tercile_labels(x) -> tuple                       # (labels 0/1/2 via np.quantile [1/3, 2/3], searchsorted 'right'; edges)
def sign_change(a, b) -> str                         # a, b: {"mean","lo","hi"} -> "Vorzeichenwechsel" | "gleiche Richtung" | "unentschieden"
PREREG = {...}                                       # F29 as {(metric, set): "le" | "ge" | "two"}
def verdict_lines(primary, floor, gate, *, model, transferred=False, smoke=False) -> list   # German markdown (6.6)
```

### 6.6 Befundtexte (German, generated by `verdict_lines`)

1. Gate failed → only:
   "**STOPP:** Die Positivkontrolle (CR20, ICL-Marge) ist nicht bestanden ({Grund}). Ohne bestandene Positivkontrolle
   wird kein Helpfulness-Befund berichtet."
2. Floor → "**Boden:** Die saubere Paargenauigkeit ({acc} [{lo}, {hi}]) ist nicht von 0,5 unterscheidbar. M1 kann eine
   verringerte Abhängigkeit nicht zeigen; interpretierbar sind nur M2 und M3." Every primary row is then marked "nicht
   interpretierbar (Boden)".
3. For each of C33 and O:
   - Holm-p < 0.05 → "**{Satz}: Abhängigkeit nachgewiesen** – mean-Ablation senkt die Helpful-Marge um {Δ}
     [{lo}, {hi}] Nats, stärker als {k} von {R} layer- und KV-gematchten Zufallsmengen (Holm-p {p})."
   - Otherwise → "**{Satz}: keine Abhängigkeit nachgewiesen** (Δ {Δ} [{lo}, {hi}], Holm-p {p})."
4. A summary sentence for O:
   - significant → "Helpful ist in den Überlappungs-Heads dekodierbar **und** das Modell stützt seine Präferenz kausal
     darauf (Expression + Reliance)."
   - not significant → "Helpful ist in den Überlappungs-Heads dekodierbar, eine Abhängigkeit der Präferenz ist nicht
     nachgewiesen (Expression ohne nachgewiesene Reliance)."
5. A transferred identity (instruct) is marked "(Kopf-Identitäten aus dem Base-Modell übertragen)". Smoke runs get
   "SMOKE: nur Funktionsprüfung, kein Befund."

### 6.7 Tests `tests/test_ablation_sets_stats.py` (offline, < 20 s)

1. `build_split` on synthetic items (ASCII; duplicated prompts, empty responses, overlaps with fake probe-train and
   probe-val keys):
   - the drop counts follow the reason order;
   - roles are disjoint by group **and** prompt cluster, and no cluster spans two roles;
   - the sizes are exact; the result is deterministic for a fixed seed and changes with the seed;
   - the prefix of `split_items(..., n)` equals the first n items;
   - `ValueError` when the pool is too small;
   - `write_split` twice gives identical bytes;
   - `verify_split` detects a changed item.
2. Named sets from a synthetic head table (`probing._finish_table` + `save_head_scores`, as `tests/test_overlap.py`
   `_write_table` does) with C33 and a synthetic census:
   - O / CminusO / TopK_all relations;
   - P_band ⊂ TopK_all ∩ band − C33 with |P_band| = |O| and per-layer counts equal to O's when candidates suffice;
   - nearest-layer fill on a built shortage (tie → lower layer), and `short_by` when the candidates run out;
   - P_DoM = the top-|O| by norm outside C33 in the band, with tie order;
   - `census_ranked` order;
   - the CR20 list on the real 4B census (`needs_repo` marker as in `tests/test_overlap.py`; it equals the list in 7.4).
3. `random_draws`:
   - per-layer counts and per-layer KV multisets equal the named set's;
   - prefix property (R = 5 vs R = 10);
   - the identity key yields the same C33 draws for two model names;
   - draws differ across set names.
4. `write_frozen`: round trip; `FrozenMismatch` on changed sets; `force`; `extend_random` keeps the stored draws.
5. `cluster_bootstrap`:
   - the mean equals `stats.paired_delta_ci` exactly, and lo/hi agree within 0.02·sd (B = 4000, synthetic clustered
     data);
   - chunk size does not change the result for a fixed seed;
   - weights and NaN handling.
6. `paired_auc_delta`: the point delta equals the `auc_matrix` difference, and the CI contains it.
7. `vs_random`: hand-computed p_le / p_ge / p_two / z, including NaN randoms and R = 0 (NaN).
8. `primary_holm` (2 tests), `floor_check` (0.5 inside / outside / inverted), `gate_decision` (every criterion failing
   separately, and advisory mode), `sign_change` and `tercile_labels` with ties.
9. `verdict_lines`: the German key phrases appear ("STOPP", "Boden", "Abhängigkeit nachgewiesen", "keine Abhängigkeit
   nachgewiesen").

**Done when:** every test passes; there are no model loads and no network (except `needs_repo`, which reads the local
checkout).

---

## 7. Modul C — `valueheads_pythia/icl_battery.py` (Positivkontrolle M3)

### 7.1 Vendoring (MIT)

The module docstring starts with:
"Portions adapted from andrewgcodes/in-context-learning-circuit @ 26c44caf8f9b (MIT License, Copyright (c) 2026 Andrew
Gao, William Gao; see LICENSE)". The source files themselves carry no per-file header; the full licence text is already
in `LICENSE`.

Vendored verbatim in logic (renamed only where noted):

- `icl/tokspans.py:2-20`: `is_bare_ws`, `label_spans`, `target_ids`
- `icl/tasks.py:24-31`: `build`, `mk` → returns an `ICLItem`
- `icl/tasks.py:279`: `NATLAB`
- the concept-battery loop of `icl/tasks.py:294-322` as `fam_concept` with the `fam_concept12n` settings of `:280-287`:
  `CONCEPT_PER = 6`, suffix `_12n`, only {topic, lang, question, implied}, `LS` = sym (foo/bar), rare (zorp/quil), nat,
  flip

Changes:

- `random.seed(13 + BATTERY_SEED)` becomes a local `rng = random.Random(13 + seed)` with an **identical call order**:
  per seq, up to 50 × (`rng.sample(A, 7)`, `rng.sample(Bs, 7)`, the token-length check), then `rng.random()`, then
  `rng.shuffle(demos)`. Concepts are iterated in JSON order. Skipped concepts (parity, vowel3, symset, cvc, digrule)
  consume no randomness.
- Dropped: `add_twins` (it makes no RNG calls), `SENTPOOL`, the environment hooks `CONCEPT_ORDER` / `CONCEPT_PERM`
  (they must stay unset), module globals, and the `concept_tasks.json` lookup at import time.
- Tokeniser calls stay `tok(text)['input_ids']` with default special tokens (Qwen3 adds no BOS). This is byte-identical.

`data/icl/concept_tasks.json` is copied unchanged from `batteries/concept_tasks.json` (11 115 bytes, sha256
`bf242720283bdc9967ffdb22da277f2fc0237d374de2b8214c45790187e60df0`). The content is harmless (topic / language /
question / implied sentences). C adds a section "ICL-Batterie (`data/icl/`)" to `data/README.md` with source, commit,
sha256 and licence.

Do **not** vendor or import `icl/run.py` (it parses argv and loads a model at import), `icl/hooks.py` (batch size 1;
hooks everything), `icl/modal_app.py`, the loupes, `figures/*` or scipy.

### 7.2 API

```python
CONCEPT_TASKS_PATH = os.path.join(DATA_DIR, "icl", "concept_tasks.json")
CONCEPT_TASKS_SHA256 = "bf242720283bdc9967ffdb22da277f2fc0237d374de2b8214c45790187e60df0"
CONCEPTS_12N = ("topic", "lang", "question", "implied")
NATLAB = {...}; LABELSETS = ("sym", "rare", "nat", "flip")
FRESH_SEED = 2; CENSUS_SEED = 1

@dataclass(frozen=True)
class ICLItem:
    index: int; kind: str; concept: str; labelset: str; seq: int; qpos: bool
    ids: tuple              # prompt ids, ends with the query arrow ' ->' (query position T-1 = len(ids)-1)
    target: str; alt: str
    target_ids: tuple       # target_ids(tok, ids, target)[0]
    alt_ids: tuple
    cluster: int            # concept_index * 100 + seq

def load_concepts(path=None, verify=True) -> dict       # path may also be an ICL repo checkout dir (batteries/concept_tasks.json)
def concept12n(tok, seed=FRESH_SEED, smoke=False, concepts=None) -> list   # 384 items (smoke: 16), order as in the original
def battery_fingerprint(items) -> str                   # sha256 over (kind, ids, target_ids, alt_ids) of all items

class QueryKnock(HeadHooks):
    """Replaces the set heads' o_proj-input slices at one position per row (the query arrow) with a per-row reference."""
    def __init__(self, model, heads)                    # positions="all", decode_only=False
    def set_rows(self, query_index, refs)               # query_index (B,) long; refs {layer: (B, n, D)} -> set before each forward

def query_means(model, items, pad_id, *, batch_tokens=8192, max_batch=64) -> dict
    # {kind: float32 (L, H*D)}: mean over the items of that kind of the o_proj input at T-1 from a clean prompt-only pass
    # (hooks.ActivationCollector, right padding); the item itself included (icl/run.py:513-521)
def score_items(model, items, pad_id, *, heads=None, kind_means=None, batch_tokens=8192, max_batch=64) -> dict
    # rows: prompt + target_ids for every item; prompt + alt_ids only when len(alt_ids) > 1 (single-token alt: read
    # log p(alt) at T-1 of the target row, icl/run.py:657-660); the knock (heads given) is active in every row at T-1 with
    # the row's kind mean; logits only at the scored positions
    # returns {"tlp": (N,), "alt": (N,), "margin": (N,) float64, "tokens_processed": int, "seconds": float}
def summarize(items, margin) -> dict       # {"sym", "rare", "symrare", "nat", "flip", "R", "P", "n"} (means)
def effects(items, m_clean, m_cond) -> dict
    # {"dret": (per sym+rare item delta, clusters), "dR": (per (concept, seq) pair delta of (nat+flip)/2, clusters),
    #  "dP": (... (nat-flip)/2 ...)}; pairing by explicit meta (concept, seq), not by order
def census_ranked(census, role="readers", k=20) -> list     # icl_sets.CensusResult -> [(l, h)] (F22; brakes: largest first)
CALIBRATION = {"qwen3-4b-base": {"symrare": 1.5991635416666667, "nat": 5.398438541666667, "flip": 0.3502604166666667,
                                 "R": 2.8743494791666664, "P": 2.5240890625},
               "qwen3-0.6b-base": {"symrare": 0.4707578125, "nat": 2.17578125, "flip": -0.033854166666666664,
                                   "R": 1.0709635416666667, "P": 1.1048177083333333}}   # data/D188_summary.json
def calibration_check(summary, model, tol=0.15) -> dict      # {"ok", "diffs": {...}, "tol", "reference": "data/D188_summary.json"}
POSITIVE_CONTROL_TARGETS = [...]                             # Anhang B as data (for the report)
```

### 7.3 Protokoll (as in the census, main.tex:460 and icl/run.py:513-533)

- Knock at T − 1 only (the query arrow, the last prompt token). Label slots and every other position stay intact
  (main.tex:308).
- Reference = the per-kind mean at the query (same concept **and** label set, 24 prompts, item included). This is
  exact for the target/alt rows, because causal attention leaves position T − 1 independent of the appended tokens.
- Margin = Σ log p(target tokens) − Σ log p(alt tokens), both under the knock (`icl/score.py:3-6`).
- dret = mean Δmargin over sym + rare (192 prompts). ΔR / ΔP use the 96 nat/flip pairs. Clusters are (concept, seq).
  Each cluster holds the four label sets of one draw.
- dtype: bf16 + sdpa on CUDA. The census itself used bf16 + eager (run.py:27). The difference is bf16 rounding.

### 7.4 Gate-Satz CR20 und Zielwerte

CR20 (Qwen3-4B-Base census, dret ascending; sum of single-head effects −2.547):
`29:11 24:29 28:0 24:30 21:11 24:22 22:10 23:26 24:31 30:7 27:19 35:26 27:18 21:18 24:23 22:0 26:26 30:5 29:14 17:4`
(layers 17:1, 21:2, 22:2, 23:1, 24:5, 26:1, 27:2, 28:1, 29:2, 30:2, 35:1).

CB24 (census brakes, dret descending; sum +3.031):
`25:30 28:23 31:30 21:31 25:14 33:21 22:8 23:28 25:28 24:28 24:21 22:2 27:16 17:7 28:2 34:29 20:7 25:31 29:9 21:26 22:20 27:15 24:12 23:31`.

Smoke (0.6B census) CR20:
`21:8 20:14 23:6 18:7 17:2 27:15 21:13 17:12 19:6 15:6 21:12 18:5 19:2 21:1 16:8 17:6 19:5 21:9 25:5 12:5`.

The paper's target numbers and their source lines are in Anhang B. The exact reader-20 result (−0.74) **cannot be
reproduced**: the set was never released (icl_sets.py:252-257), the battery differs (concept12, not concept12n), and
the reference of that run is ambiguous. CR20 on a fresh concept12n draw is the closest released substitute. The gate
(F23) is an explicit rule of this project, not a number from the paper.

### 7.5 Tests `tests/test_icl_battery.py` (< 40 s)

1. The sha256 of `data/icl/concept_tasks.json` equals the pinned value; `load_concepts(verify=True)` rejects a modified
   copy.
2. With the cached Qwen3-0.6B-Base tokeniser (tokeniser only; skip if not cached), concept12n for seed 1 and seed 2
   gives:
   - 384 items, 16 kinds × 24;
   - prompt lengths 115-183;
   - 341 single-token alts;
   - per (concept, seq) the four label sets share sentences and query;
   - seeds 1 and 2 give different fingerprints.

   Pin `battery_fingerprint` for seed 1 and seed 2 as constants. The implementer computes them once and cross-checks
   them against the original `icl/tasks.py` `build_battery(tok, ("concept12n",), False)` in a scratch process
   (`python3 -I`, `sys.path` → the checkout's `icl/`, env `BATTERY_SEED`, cwd = `$SCRATCH`, tokeniser only). The scout
   ran this builder before. The result is recorded in the test docstring; if that execution is refused, say so there.
3. `smoke=True` gives 16 items.
4. Mechanics on a tiny random Qwen3 with a test-local **lossless** character tokeniser (vocabulary from the characters of
   the data file, `decode` implemented):
   - the batched `score_items` equals an unbatched per-item reference (`seq_logprob` semantics), including a
     multi-token alt;
   - `heads=[]` is identical to intact;
   - a knock whose reference is the item's own query value is identical to intact;
   - the knock changes only position T − 1 (captured o_proj input);
   - `query_means` at T − 1 from the prompt-only pass equals the value at T − 1 of the target row.
5. `summarize` / `effects` on synthetic margins: R/P formulas, pairing by meta, clusters.
6. `census_ranked` on a synthetic `CensusResult`. On the real 4B census (`needs_repo`) it equals the CR20 list in 7.4.

**Done when:** every test passes, the fingerprints are pinned, and `data/README.md` has the entry.

---

## 8. Modul D — Kommando `ablate`, Metriken, Report, Notebooks

### 8.1 Dateien

- `valueheads_pythia/commands/ablate.py`: CLI, stage orchestration, idempotent conditions, pilot / ETA.
- `valueheads_pythia/ablation_metrics.py`: sequence builders and M1 / M2 / M4 / M5 / M7 computations on top of A.
- `valueheads_pythia/ablation_report.py`: summary JSON / MD (German) / CSV, plus the trajectory.
- `config.paths()` gets `"ablation": os.path.join(results, "ablation")`. `cli.COMMANDS["ablate"] = "Kausaler Test:
  Ablation der Wert-Head x ICL-Ueberlappung (Qwen3, Stage 0/1/2, idempotent)"`.

### 8.2 Metriken

**M1 (primary behaviour).** Per item i, from the condition's `RunResult` (variant hh3/hh2):

- margin_mean_i = LL̄(pos) − LL̄(neg); margin_sum_i = LL(pos) − LL(neg)
- win_mean_i = [margin_mean_i > 0]; win_sum_i likewise
- threeway_i = LL̄(pos) − max(LL̄(neg), LL̄(unrel)) (hh3 only)
- len_hi_gt_lo_i = [n_pos > n_neg]

Δ = condition − clean of the same variant. The quantities are reported signed. Pair accuracy = mean(win). Sums and
counts must match `PairwiseMetric` semantics (F1).

**M2 (internal readout, monitors).** In every hh2/hh3 condition:

- `LastTokenCollector(readout_layers)` and `ResidualCapture` read the pos/neg rows at ln − 1 (2·N_EVAL texts,
  labels 1/0, clusters).
- Fixed probes (F15) give scores. The reported statistics are:
  - `head_auc_mean`: mean over readout heads ∉ set;
  - `head_auc`: one value per readout head (256 on 4B), NaN for excluded heads;
  - `resid_auc`, `mlp_auc`;
  - for named sets and clean only: `refit_resid_auc` and `refit_head_auc_mean`.
- For named sets and clean, the per-text scores go to a sidecar `conditions/{name}.m2.npz` (fp16) for
  `paired_auc_delta`.
- Labels in reports: "L1 fester Probe (Monitor)", "L2 neu gefittet (Informationserhalt)", "L3 nichtlinear (MLP)". They
  are never proof of use.

**M3** — section 7. On the Base model: the gate in Stage 0, the reverse test in Stage 2. On the instruct model: all
descriptive.

**M4 (damage).** `PPLMetric(tok, max_tokens=50000)`. The conditions run through the A runner with `Seq(boundary =
target_start − begin, role="ppl")` and `MeanReplace(policy="all", ref=μ_resp)`. Δnll is token-weighted, using the
weights of `PPLMetric.item_values` (`cluster_bootstrap` with weights, cluster = window). Direction "ge".

**M5 (self-repair, C33 and O).**

- DE_i = per-item mean-token margin change from `ablation.direct_effect(..., heads, μ_resp)` on hh2 pos/neg.
- TE_i = Δmargin_mean_i of `mean__hh3__{set}`.
- Reported: DE, TE (cluster CIs) and SRI = 1 − mean(TE)/mean(DE) via `cluster_bootstrap_ratio`. SRI is reported only
  when the DE CI excludes 0, else "nicht definiert".

**M7 (covariates).** Computed once from the EVAL items and the tokeniser, stored in the clean hh3 condition.

- n_pos, n_neg = continuation tokens (len − b).
- reuse(resp) = |U(resp) ∩ U(q)| / max(1, |U(resp)|), with:
  - U = the set of token ids whose decoded string is neither whitespace-only nor punctuation-only;
  - q = the Human turn (`data.first_qa(context + " ")[0]`), tokenised as `" " + q` without special tokens;
  - resp = the continuation.
- reuse_diff = reuse(pos) − reuse(neg); len_diff = n_pos − n_neg.
- Terciles (`tercile_labels`) for each covariate. Per tercile: Δmargin_mean of each named mean-arm set (cluster CI) and
  the random mean for C33 and O. All descriptive.
- Shuffled prompt (`hhshuf`, F17) for clean, C33 and O: PD_i = margin_mean_i − margin_mean^shuf_i, and ΔPD under
  ablation.

### 8.3 Bedingungsraster und R-Tabelle

Condition name = `{arm}__{variant}__{set}[__r{idx:03d} | __d{idx:03d}]`. Clean runs are `clean__{variant}`. Fixed-identity
sets are suffixed `_base` (`mean__hh2__O_base__r004`). Arms: `clean`, `mean`, `mean_last`, `resample`, `dir_mean`,
`gauss`, `knock` (M3), `dla` (M5).

In the table, a number is R (random sets, or random directions for dir_mean); "1" means the named set only (no null,
descriptive); "–" means not run. Smoke uses R = 2 in every cell with R > 0.

| Stage | arm / variant | C33 | O | CminusO | P_band | P_DoM | C33_readers | C33_brakes | TopK_all | WholeBand | CR20 | CensusReaders | CB24 | *_base (instr.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | knock / icl (gate, Base) | – | – | – | – | – | – | – | – | – | 20 | – | – | – |
| 1 | mean / hh3 (primary) | **100** | **100** | – | – | – | – | – | – | – | – | – | – | – |
| 2 | mean / hh2 | – | – | 50 | 50 (F12) | 20 | 20 | 20 | 20 | 1 | – | – | – | 20 each (F14) |
| 2 | mean_last / hh2 (M2, M1 sanity) | 20 | 20 | – | 20 | – | – | – | – | – | – | – | – | – |
| 2 | resample / hh2 | 20 | 20 | – | – | – | – | – | – | – | – | – | – | – |
| 2 | dir_mean / hh2 (random directions) | 20 | 20 | – | 20 | – | – | – | – | – | – | – | – | – |
| 2 | gauss / hh2 | 20 | 20 | – | – | – | – | – | – | – | – | – | – | – |
| 2 | mean / hhshuf (M7) | 1 | 1 | – | – | – | – | – | – | – | – | – | – | – |
| 2 | mean / ppl (M4) | 20 | 20 | 20 | 20 | 1 | 1 | 1 | 20 | 1 | – | – | – | – |
| 2 | dla / hh2 (M5) | 1 | 1 | – | – | – | – | – | – | – | – | – | – | – |
| 2 | knock / icl (M3 reverse) | 20 | 20 | – | 20 | – | – | – | 20 | 1 | (instr.: 20) | 20 | 20 | – |

- P_band under F12 reuse (profile identical to O's): instead of the 50 own draws, `mean__hh3__P_band` runs once in Stage 2 (variant hh3) and is compared with O's 100 Stage-1 random conditions.
- Clean conditions: `clean__hh3` (Stage 0, floor), `clean__hh2` (Stage 2), `clean__hhshuf`, `clean__ppl`, `clean__icl`
  and `clean__icl_s1` (calibration, Stage 0).
- Random sets for a cell always come from that set's frozen draws, using the first R.

Primary family: rows 1/C33 and 1/O only. Everything with R > 0 is "secondary" (p reported, no verdict). "1" cells are
"descriptive".

### 8.4 Stufen

| Stage | Steps (in this order) | Stops with |
|---|---|---|
| 0 | (1) Check the inputs: heads table (meta mode pythia, score auc, `n_train == 20000`; missing → exit 2 "zuerst Overlap-Notebook"), census (sha256), concept_tasks sha256. (2) Write or verify the split file. (3) Load the model (dtype policy). (4) `hook_selftest` on 4 means_ref pairs with C33 and with WholeBand → `selftest.json`. (5) `compute_refs` on means_ref → `refs.npz`; collect M2 training features on means_probe → `m2_train.npz`. (6) `named_sets` + `random_draws` → `write_frozen`. (7) Train M2 probes → `probes_m2.npz`. (8) `clean__hh3` (M1, M2, M7 covariates) → floor check → `floor.json`. (9) Base: `clean__icl_s1` calibration, `query_means` on seed 2, `clean__icl`, `knock__icl__CR20` + 20 randoms → `gate_decision` → `gate.json`. Instruct: wait for `results/ablation/{icl_identity}/gate.json`, polling every 60 s for at most `--gate_wait_min` minutes | exit 4 self-test, exit 5 frozen mismatch, exit 3 gate failed / missing (smoke: advisory, exit 0) |
| 1 | requires `gate.json` passed (else exit 3 unless `--ignore_gate`). `--pilot`: `mean__hh3__C33` and `mean__hh3__C33__r000` → `pilot.json` and the ETA table, then exit. Full: every Stage-1 cell, then the primary statistics and the summary | exit 3 |
| 2 | requires gate. Arms in the priority order m3 → mean_sets → m4 → mean_last → m5 → m7 → resample → dir_mean → gauss → fixed_identity (`--arms` selects), then the summary | exit 3 |
| `--summary` | the report from the existing condition files only (no model) | — |

### 8.5 CLI

`python -m valueheads_pythia.cli ablate` takes `add_common_args(parser, mode=False)` (`--model --revision --root
--device --dtype --attn_impl --seed`) plus:

| Flag | Default | Meaning |
|---|---|---|
| `--stage {0,1,2}` | required unless `--summary` | stage |
| `--pilot` | off | stage 1: two conditions, ETA / CU, exit |
| `--summary` | off | write the report only |
| `--arms …` | all applicable | stage-2 arms: `m3 mean_sets m4 mean_last m5 m7 resample dir_mean gauss fixed_identity` |
| `--heads_table PATH` | `config.heads_path(P, short, None, "helpful", "pythia")` | probe table |
| `--icl_model NAME` | `icl_sets.resolve_icl_model(short)[0]` | identity of the ICL sets / gate |
| `--icl_source SRC`, `--no_download` | env VH_ICL_REPO, cache, pinned URL | census source |
| `--band A-B` | `20-27` | band of C33 / P_band / P_DoM / WholeBand |
| `--readout_layers A-B` | the last 8 layers | M2 readout layers |
| `--n_means_ref`, `--n_means_probe`, `--n_eval` | 256, 256, 1000 (≤ the split's sizes) | prefixes of the split roles. `--n_eval` also fixes the hh3 partner derangement and so the key of every HH condition: once HH conditions exist, another value is refused (exit 2) unless `--change_n_eval` (or `--force`); the report skips (and lists) conditions whose items differ from their clean run |
| `--change_n_eval` | off | accept an `--n_eval` other than the stored one (anchor: `clean__hh3`); every HH condition is recomputed when its stage runs next (D) |
| `--r_c33 --r_o --r_small --r_sec` | 100, 100, 50, 20 | R table |
| `--threeway_all` | off | F18 |
| `--battery_seed`, `--calib_seed` | 2, 1 | F21 |
| `--bootstrap` | 10000 | B (M2: min(B, 1000)) |
| `--batch_tokens`, `--max_batch` | 8192/64 (CUDA), 2048/16 (CPU) | F19 |
| `--ppl_tokens` | 50000 | M4 |
| `--gate_wait_min` | 90 | instruct gate wait |
| `--ignore_gate` | off | debugging only; the report states "Gate ignoriert" |
| `--extend_random` | off | allow R above the frozen R (appends draws) |
| `--refreeze` | off | overwrite a frozen file (refuses if any EVAL condition exists) |
| `--smoke` | off | F28 sizes; synthetic smoke-only table / C33 if the real ones are missing; advisory gate |
| `--note TEXT` | — | line at the top of the report |
| `--force` | off | recompute existing conditions |
| `--quiet`, `--no_progress` | off | — |

Exit codes: 0 ok, 1 error, 2 missing inputs, 3 gate failed or missing, 4 self-test failed, 5 frozen mismatch.

### 8.6 Dateien und Formate

```
results/ablation/
  hh_helpful_split.json                     model-independent (6.1)
  {short}_frozen_sets.json                  (6.4)
  {short}_ablation.json / .md / _conditions.csv / _tests.csv
  {short}/
    selftest.json  refs.npz  m2_train.npz  probes_m2.npz  floor.json  gate.json  pilot.json  icl_kind_means_s2.npz
    conditions/{name}.json  [{name}.m2.npz]
  _smoke/{short}_main_helpful_pythia_SMOKE.json     smoke-only synthetic table (never under results/heads/)
```

Condition file (atomic write; skipped when it exists **and** its `key` matches; recomputed on a key mismatch or with
`--force`):

```json
{"name": "mean__hh3__C33__r017", "version": 1,
 "key": "<eval_likelihood.condition_key(config, cond)>",
 "model": "qwen3-4b-base", "hf_id": "Qwen/Qwen3-4B-Base", "commit_sha": "...", "dtype": "bfloat16", "attn_impl": "sdpa",
 "arm": "mean", "variant": "hh3", "set": "C33", "random": {"kind": "heads", "index": 17},
 "heads": [666, ...], "n_heads": 33,
 "config": {"split_sha": "...", "n_eval": 1000, "eval_ids_sha": "...", "refs_sha": "...", "policy": "response",
            "batch_tokens": 8192, "max_batch": 64, "seed": 42, "variant_seqs_sha": "..."},
 "summary": {"margin_mean": 0.0, "margin_sum": 0.0, "acc_mean": 0.0, "acc_sum": 0.0, "threeway_mean": 0.0,
             "m2": {"head_auc_mean": 0.0, "head_auc": [], "resid_auc": 0.0, "mlp_auc": 0.0}},
 "items": {"id": [], "pos_sum": [], "pos_n": [], "neg_sum": [], "neg_n": [], "unrel_sum": [], "unrel_n": []},
 "stats": {"seconds": 0.0, "tokens_processed": 0, "tokens_scored": 0, "tokens_per_second": 0.0},
 "created": "2026-11-05T12:00:00"}
```

`cond` (the key input) = {arm, variant, set, random index, heads, policy, plus arm parameters: ref sha / donor seed /
direction key / gauss seed}. The key does **not** use the whole frozen-file hash, so extending the randoms does not
invalidate anything. ICL conditions store `items: {"index", "tlp", "alt", "margin"}`; PPL conditions store `{"sum", "n"}`.

Report files:

- `_conditions.csv`: one row per condition — name, arm, variant, set, r, n_heads, summary values, Δ means vs clean,
  seconds, tok/s.
- `_tests.csv`: one row per (arm, variant, metric quantity, named set) — observed Δ, lo, hi, n, n_clusters, R,
  rand_mean, rand_std, rand_q025, rand_q975, p_le, p_ge, p_one, p_two, z, direction, family (primary / secondary /
  descriptive), p_holm (primary only), verdict (primary only).

### 8.7 Bericht `{short}_ablation.md` (German, `ablation_report.write_report`)

Sections:

1. Header (model, revision/commit, dtype, smoke/note banner, transferred identity; (D) a warning listing conditions
   skipped because their item ids differ from their clean run, e.g. computed with another `--n_eval`).
2. "Ergebnis in einem Satz" (`verdict_lines`; on the instruct model C33 is marked as a transferred Base identity, O as
   "C33-Identitäten übertragen, Top-k aus der eigenen Probe-Tabelle").
3. "Gate (Positivkontrolle)": CR20 Δdret [CI] vs. random (mean ± sd, min / max), (D) the paper's targets from
   `POSITIVE_CONTROL_TARGETS` labelled as another battery and head set (reader-20 −0.74 [−0.95, −0.55] vs. random
   +0.02, census-ranked Llama reader-20 −0.56 vs. 0.00, CR20 summed single-head dret; instruct: the 189-head removal),
   calibration differences, PASS/STOPP. A full (non-smoke) run treats a smoke gate as STOPP.
4. "Boden-Check".
5. "Primäre Familie (Holm, α = 0.05)": table with set, n heads, Δ margin_mean [CI], random mean ± sd (R), p one-sided,
   p two-sided, z, Holm-p, finding; below it the same for margin_sum, acc_mean and threeway (secondary).
6. "Sekundär (p ohne Befund)": M1 per arm × set; M2 (L1/L2/L3); M3 reverse test (dret, dR, dP); M4; M5; M7 (terciles,
   shuffled prompt).
7. "Trajektorie Base → Instruct": if the other model's `_ablation.json` exists, then for each identity set (C33,
   C33_readers, C33_brakes, WholeBand and the `*_base` sets vs the Base's own sets) Δ on Base, Δ on instruct, and
   `sign_change`.
8. "Kausalitäts- und Hook-Checks": self-test rows, mean_last M1 sanity (F4).
9. "Abweichungen vom Design" (section 10, verbatim list).
10. "Eingaben": sha256 of every input.

If the gate failed, sections 4-7 are replaced by the STOPP text (6.6); stage 0 writes this STOPP report itself before it
exits 3. The JSON holds the full `_tests` rows, the families, gate, floor, the inputs and the run settings. (D) Primary
rows in the JSON and in `_tests.csv` carry the raw Holm flag `significant`, `interpretable` (gate passed, no floor, no
smoke) and the same `verdict` as the markdown ("STOPP (Gate)", "nicht interpretierbar (Boden)", or the Holm finding with
its reservations), so the machine-readable files cannot be read as a finding the report withholds.

### 8.8 Pilot, ETA und CU

- `pilot.json` = measured tokens/s per variant (hh3 from the two pilot conditions; icl from the Stage-0 gate
  conditions; hh2 ≈ hh3 rate; ppl measured in Stage 2 or ≈ hh3) plus the mean per-condition overhead.
- ETA = Σ over pending conditions of tokens(variant) / rate + overhead, with tokens(variant) taken from the clean run of
  that variant (gauss × 1.75 for the pre-pass).
- `CU_PER_A100_HOUR = 6.0` (the README convention). The table is printed per stage and arm.

Static estimate (tokens from the scouting statistics; per M1 item hh3 ≈ 219 tokens, hh2 ≈ 146; M3 ≈ 65 k tokens per
condition; M4 ≈ 100 k processed tokens per condition):

| | Conditions | Tokens | A100 h at 5k / 8k / 10k tok/s | CU (×6) at 8k |
|---|---|---|---|---|
| Stage 0 Base (incl. gate 21 ICL conditions) | ~26 | ~2.1 M | 0.12 / 0.07 / 0.06 (+ model load ~0.2) | ~2 |
| Stage 1 (per model) | 203 | ~44.5 M | 2.5 / 1.5 / 1.2 | ~9 |
| Stage 2 Base | ~640 | ~83 M | 4.6 / 2.9 / 2.3 | ~17 |
| Stage 2 instruct (incl. fixed identity 105, CR20 21) | ~770 | ~100 M | 5.6 / 3.5 / 2.8 | ~21 |
| **Total Base / instruct** | | ~129 M / ~145 M | **7.2 / 4.5 / 3.6** and **8.0 / 5.0 / 4.0** | **~27 / ~30** |

The rate is not measured. Overlap extraction measured ~10 k tok/s without logits; the pilot prints the real ETA.
Priority if CU run short: Stage 0 + Stage 1 (≈ 2 h, ≈ 12 CU per model) answer the primary question. Stage 2 follows the
priority order of 8.4.

### 8.9 Notebooks (`scripts/make_ablation_notebook.py`)

- The script generates `notebooks/runs/qwen3-4b-base-ablation.ipynb` (`MODEL = "Qwen/Qwen3-4B-Base"`, role base) and
  `notebooks/runs/qwen3-4b-ablation.ipynb` (`MODEL = "Qwen/Qwen3-4B"`, role instruct). The two paths let both run in
  parallel: Colab binds one runtime per notebook path.
- It copies the helpers of `scripts/make_overlap_notebook.py` (`src`, `sub1`, `master_cell`, and `md`/`code` with the
  uuid5 id **per notebook name**).
- It copies master cell `# 1. Laufzeit-Check` and master cell `# 2. Installation` (with `DRIVE_ROOT` →
  `MyDrive/ValueHeadsPythia/qwen3-4b` and `DRIVE_PROJECT` → `MyDrive/ValueHeadsPythia/valueheads_pythia`), plus the
  `run()` helper from the master config cell.
- Metadata: `colab.gpuType "A100"`, `accelerator "GPU"`. The script runs `compile()` on every code cell and
  `nbformat.validate`.
- `--out DIR` is supported.

Cells (German prose; `md` + `code`):

| # | Cell | Content |
|---|---|---|
| 0 | INTRO (md) | Role, model, Drive folder, question (expression vs. reliance), procedure, budget table (8.8), note on the other notebook ("parallel ausführbar; Instruct wartet in Stufe 0 auf das Base-Gate"), Datenschutz ("HH-Inhalte werden nie ausgegeben; vor dem Speichern Alle Ausgaben löschen") |
| 1 | Laufzeit-Check | master copy |
| 2 | Installation, Projekt, Drive | master copy (Drive root replaced) |
| 3 | Konfiguration | `MODEL` (env `VH_NB_MODELS`, first entry, overrides), `SHORT`, `N_EVAL = 1000`, `R_C33 = 100`, `R_O = 100`, `R_SMALL = 50`, `R_SEC = 20`, `BOOTSTRAP = 10000`, `ARMS = {"m3": True, "mean_sets": True, "m4": True, "mean_last": True, "m5": True, "m7": True, "resample": True, "dir_mean": True, "gauss": True, "fixed_identity": <instruct only>}` (env `VH_NB_ARMS` = comma list), `BATCH_TOKENS`, `GATE_WAIT_MIN = 90`, SMOKE → `--smoke` and NOTE, `COMMON` argument list; prints the heads-table presence on Drive and the census check; then the `run()` helper |
| 4 | Stufe 0 – Gate | `run("ablate", *COMMON, "--stage", 0)`; then shows `selftest.json`, `floor.json` and `gate.json` as German text (PASS/STOPP, numbers) |
| 5 | Pilot + ETA | `run("ablate", *COMMON, "--stage", 1, "--pilot")`; prints the ETA/CU table from `pilot.json` and a note on the knobs |
| 6 | Stufe 1 – primär | `run("ablate", *COMMON, "--stage", 1)`; shows the primary table (C33, O, Holm) |
| 7 | Stufe 2 – sekundär | `run("ablate", *COMMON, "--stage", 2, "--arms", *[a for a, on in ARMS.items() if on])` |
| 8 | Zusammenfassung | `run("ablate", *COMMON, "--summary")`; `display(Markdown(...))` of `{short}_ablation.md` up to the secondary section; paths on Drive; reminder to clear outputs |
| 9 | FOOTER (md) | "Bearbeiten → Alle Ausgaben löschen" |

Every cell is idempotent: after a disconnect, "Alle ausführen" skips finished conditions.

### 8.10 README, LICENSE

**README** — a new section `## Kausaler Test: Ablation (Qwen3-4B)` after the overlap section, in German:

- question and expression vs. reliance;
- design in ten lines (sets, interventions, metrics, primary family, gate);
- both Colab links:
  `https://colab.research.google.com/github/Erikiss/ValueHeads/blob/main/notebooks/runs/qwen3-4b-base-ablation.ipynb`
  and `…/qwen3-4b-ablation.ipynb`;
- the budget table (8.8);
- the outcome table below;
- the CLI example and the smoke recipe;
- the outputs.

Also update `Projektstruktur` (new modules, `data/icl/`, `docs/ABLATION_DESIGN.md`, the notebooks) and add an
Attribution bullet (ICL code vendored in `icl_battery.py`).

| Outcome | Meaning |
|---|---|
| Gate STOPP | The method does not even find known ICL readers: no helpfulness statement |
| Gate ok, floor | The clean preference cannot be told from 0.5: M1 is uninformative, M2/M3 descriptive only |
| O significant (Holm) | Reliance: the model uses the overlap heads for its own helpful preference beyond layer/KV-matched random heads |
| C33 significant, O not | The preference depends on the ICL circuit, but not specifically on its helpful-decoding part (compare CminusO, P_band) |
| neither significant | Helpful is decodable there without demonstrated reliance. Check TopK_all / WholeBand (other site or redundancy) and M5 (self-repair) |
| M2 L1 drops, L2 recovers | The monitor breaks but the information survives (re-encoded downstream) |
| sign change Base → instruct | Post-training changes the role of the same head identities |

**LICENSE** — in the block "Attribution: in-context-learning circuit head sets", rename the heading to
"Attribution: in-context-learning circuit (head sets, battery)" and insert after the `icl_sets.py` paragraph:

```
valueheads_pythia/icl_battery.py contains code adapted from the same
repository (icl/tasks.py: build, mk, NATLAB and the concept-battery loop of
fam_concept / fam_concept12n; icl/tokspans.py: is_bare_ws, label_spans,
target_ids) and a re-implementation of its query-position knockout and
margin scoring (icl/run.py, icl/hooks.py, icl/score.py). The battery data
file data/icl/concept_tasks.json is redistributed unchanged from
batteries/concept_tasks.json at the same commit (sha256
bf242720283bdc9967ffdb22da277f2fc0237d374de2b8214c45790187e60df0).
```

Change "applies to the derived portions" to "applies to the derived and adapted portions and to the redistributed data
file".

### 8.11 Tests `tests/test_ablate.py` (< 60 s, offline)

1. `ablate` is in `cli.COMMANDS`, `--help` works, and `config.paths()["ablation"]` exists.
2. Notebook reproducibility for **both** notebooks: regenerate into `tmp_path` and compare byte-for-byte with the
   committed files. Also check:
   - every cell compiles, with no outputs and `execution_count` None;
   - `gpuType` is A100;
   - the model strings and `MyDrive/ValueHeadsPythia/qwen3-4b"` appear;
   - `DRIVE_PROJECT` points back to `MyDrive/ValueHeadsPythia/valueheads_pythia`;
   - `"ablate"`, `"--stage", 0` / `1` / `2`, `"--pilot"` and `"--summary"` appear;
   - there is no `eval-generate` or judge step.

   This mirrors `test_overlap_notebook_is_generated_reproducibly`.
3. An end-to-end pipeline function (`ablate.run_pipeline(args, model, tok, split, items_by_id, census)`; the CLI path
   builds the real split / items / census and wraps it, the test injects synthetic ones) on `tiny_qwen3`. It uses synthetic ASCII HH-style items, a synthetic heads table, a synthetic C33/census, R = 2,
   B = 50, and `--arms` without m3 (M3 is covered by C). Stages 0 → 1 → 2 → summary. Check:
   - condition files are written;
   - a second run skips everything (no forward: monkeypatched counter);
   - a changed `--batch_tokens` recomputes;
   - the frozen file is written before the first EVAL condition (mtime / order log).
4. Gate paths: a synthetic `gate.json` with `passed: false` makes stage 1 exit 3; `--ignore_gate` is noted in the
   report; the instruct wait times out → exit 3 (wait 0 min).
5. Report: synthetic condition files cover Holm on two tests, the floor banner, the STOPP report without helpfulness
   sections, the trajectory sign change and the `_tests.csv` columns.
6. Fixed-identity arm: the arm is skipped with a message when the Base frozen file is missing, and uses the Base draws
   when it is present.

**Done when:**

- the full suite is green;
- the smoke recipe (section 9) runs through for the Base notebook;
- README, LICENSE and Projektstruktur are updated;
- both notebooks are regenerated.

---

## 9. CPU-Smoke-Rezept

CLI (with `SCRATCH=/tmp/claude-0/-home-user-valueheads/3ccf061c-00fb-546b-8a09-3ab673f5e720/scratchpad/ablation_build`):

```bash
cd /home/user/valueheads_pythia
VHPY='import sys; sys.path.insert(0, "/home/user/valueheads_pythia"); import runpy; runpy.run_module("valueheads_pythia.cli", run_name="__main__")'
for st in "--stage 0" "--stage 1 --pilot" "--stage 1" "--stage 2" "--summary"; do
  python3 -I -c "$VHPY" ablate --model Qwen/Qwen3-0.6B-Base --root $SCRATCH/smoke_root --device cpu --smoke $st || break
done
```

Notebook (same recipe as the overlap notebook):

```bash
VH_NB_SMOKE=1 VH_NB_MODELS=Qwen/Qwen3-0.6B-Base VH_NB_PROJECT=$PWD VH_ROOT=$SCRATCH/nb_smoke \
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=5400 \
  --output $SCRATCH/nb_smoke/executed.ipynb notebooks/runs/qwen3-4b-base-ablation.ipynb
```

Smoke facts:

- Qwen3-0.6B-Base: 28 × 16, hpk 2. Band 20-27 = its last 8 layers. Readout = the last 8 layers (also 20-27); ablated
  heads are excluded.
- C33 is the synthetic smoke stand-in (6.2). The synthetic table lives at `_smoke/…_SMOKE.json`: uniform AUCs
  0.55-0.60, plus 0.3 for 4 of the 8 stand-in heads and for 16 further band heads, so O, CminusO and P_band are
  non-empty.
- CR20 comes from the real 0.6B census (pinned; local checkout via VH_ICL_REPO or the cache).
- ICL smoke battery: 16 prompts; no calibration; the gate is advisory.
- HH helpful-base needs network access; WikiText-2 is in the HF cache.
- Target: ≤ 30 min on 4 CPUs.

The instruct notebook gets the reproducibility test and the compile check only. A CPU smoke of it would need
Qwen/Qwen3-0.6B, which is not cached.

---

## 10. Abweichungen vom Design (dokumentiert; also printed in every report)

1. **Positive-control set.** The paper's reader-20 (−0.74 nats) is unreleased. CR20 is census-ranked on concept12n (the
   paper's Llama protocol). The battery is concept12n seed 2, not concept12. The gate rule (F23) is this project's own,
   not the paper's.
2. **Response positions** include b − 1 (the last prompt token) and ln − 1 (F2). The design says "response-token
   positions, prompt untouched": b − 1 is counted as the decision position of the first response token.
3. **mean_last on M1** is a causality sanity check only (F4).
4. **P_band random null.** Own-profile draws unless the KV profile equals O's (F12). The design says "when sizes differ".
5. **P_DoM** is selected per model (F13). The ranking by raw norm is scale dependent; a normalised variant is not
   computed.
6. **Three-way margin** is computed only in hh3, i.e. Stage-1 sets and clean (F18); `--threeway_all` restores it
   everywhere.
7. **M2 refits** run only for named sets and clean, with torch L-BFGS (same objective as sklearn). M2 CIs use B = 1000.
8. **ICL random null.** Distinct heads, layer- and KV-matched. The paper drew per named head a same-layer head with
   replacement (run.py:118); our null is stricter.
9. **Fixed-identity arm** requires the Base frozen file (F14); there is no recomputation in the instruct run.
10. **HH-harmless M4** is not implemented (optional in the design).

Implementation deviations of part D (also printed in every report, `ablation_report.DESIGN_DEVIATIONS`):

11. **Smoke ICL battery** is concept12n with `n_seq = 2` (32 prompts, 2 per kind), not the 16-prompt smoke battery: with one
    prompt per kind the per-kind mean equals the item's own query value, so every knock would be an exact no-op (part C).
12. **Random conditions of a reused null** (F12 P_band → O; `*_base` sets following the Base file) are named after the set
    that owns the draws (`mean_last__hh2__O__r003` serves O and P_band), so identical head sets are computed once.
13. **M2 sidecars** store per-text scores in float32 (spec: fp16), so the paired AUC deltas equal the summaries. The head
    AUC delta is paired per readout head (each condition on its own non-ablated heads).
14. **Tests** live in `tests/test_ablate_command.py` (orchestrator naming) instead of `tests/test_ablate.py`.
15. **Stage 1** re-runs `clean__hh3` when its key changed (e.g. another batch plan), and the floor follows the current
    `clean__hh3`, so a condition is always paired with a clean run of the identical batch plan.

## 11. Offene Risiken

- **Budget.** ~4.5-8 A100 h per model at 5-8 k tok/s (8.8). This is not measured; the pilot decides. Stage 0 + 1 alone
  answer the primary question.
- **Floor.** Base-model HH pair accuracy was 0.52-0.57 for Pythia. With 1000 pairs the CI is about ±0.03, so the floor
  branch is plausible, especially on Qwen3-4B-Base.
- **Census seed.** Seed 1 was inferred from a 0.6B CPU calibration only; 4B is unverified. The Stage-0 calibration
  (F23a) stops if the battery differs by more than 0.15 nats.
- **Battery equivalence.** The vendored builder was not yet compared item by item with the original (C does this once;
  7.5).
- **Gate on instruct runs.** The parallel instruct notebook depends on the Base gate (90 min wait). If the Base
  notebook is late, the instruct notebook stops at Stage 0 and must be rerun.
- **Data counts.** HH helpful-base loads online only. The split counts (2 836 / 23 / 1 / 3) are scouting values and are
  re-derived at runtime.
- **Numerics.** bf16 + sdpa vs. the census's bf16 + eager. Batch-composition noise is controlled by F18/F19 (same
  variant, same plan).
- **Interpretation.** A null M1 result does not separate "no reliance" from redundancy or self-repair. M5 and
  TopK_all / WholeBand are descriptive aids, not tests.

---

## Anhang A: Wiederverwendete APIs (verbindlich)

| API | Where | Use |
|---|---|---|
| `HeadHooks(model, positions, decode_only, layers)`, `edit`, `head_slice`, `last_index` | hooks.py:26-90 | base class of A's edits and C's `QueryKnock` (positions="all", decode_only=False) |
| `ActivationCollector(model, layers).collect(ids, mask)` | hooks.py:92-130 | C `query_means` (right padding) |
| `model_geometry`, `num_kv_heads`, `kv_group_heads`, `decoder_blocks`, `o_proj`, `load_model`, `set_seed` | models.py:32-54, 118-157 | all |
| `encode_pair`, `loglik_batched`, `PPLMetric`, `condition_key`, `_json_default`, `run_condition` | eval_likelihood.py:81, 109, 366, 543, 573, 453 | D (M1 tokenisation, M4 windows, keys); A tests |
| `PairwiseMetric.item_values` semantics | eval_likelihood.py:254-275 | M1 quantities |
| `eval_pairs`, `hh_pairs`, `probing_set`, `first_qa`, `wikitext` | data.py:284, 83, 221, 68, 394 | B split, D M7/M4 |
| `plan_batches(lengths, batch_size, max_tokens)` | extract.py:76 | A runner, C scorer |
| `load_value_table`, `layer_matched_draws`, `_rng`, `holm`, `two_sided`, `file_sha256`, `write_outputs` (pattern) | overlap.py:112, 165, 75, 289, 283, 80, 755 | B, D |
| `empirical_p`, `paired_delta_ci` (reference semantics) | stats.py:14, 34 | B |
| `auc_matrix`, `WeightedAUC`, `torch_batched_logreg` | probing_stats.py:28, 53, 218 | B (AUC CI), D (M2) |
| `_finish_table`, `save_head_scores`, `load_head_scores`, `top_heads_from_file` | probing.py:103, 293, 330, 339 | synthetic tables (tests, smoke) |
| `NAMED_CIRCUIT_L20_27_QWEN3_4B`, `NAMED`, `census_sets`, `CensusResult.effects`, `resolve_icl_model`, `ICL_COMMIT` | icl_sets.py:229, 233, 471, 450, 135 | B, C, D |
| `dtype_policy`, `paths`, `heads_path`, `model_short_name`, `MODELS` | config.py:218, 231, 284, 259, 10 | D |
| notebook helpers `src`, `sub1`, `master_cell`, `md`, `code`, master-cell rewrite | scripts/make_overlap_notebook.py:33-59, 456-498 | D generator |
| `tiny_qwen3`, `CharTokenizer` | tests/test_qwen_port.py:21-75 | copied into the new test files |
| `_write_table`, `needs_repo` | tests/test_overlap.py:13-18, 300-309 | pattern for B/D tests |

## Anhang B: Positivkontroll-Zielwerte mit Quellzeilen

| Quantity | Model / battery | Value (nats) | Source |
|---|---|---|---|
| Reader-20 joint knock, fresh draw | Qwen3-4B-Base, concept12 (12 demos, 144 prompts, 6 kinds) | **−0.74 [−0.95, −0.55]** | main.tex:263, :316; tab:claims main.tex:592; figures/fig6_populations.py:4-6 |
| 20 random heads, same layers, fresh draw | same | **+0.02 [−0.03, +0.06]** | main.tex:263; CI in figures/fig6_populations.py:6 |
| Intact margin of that battery | same | +0.84 (sym +0.55 / rare +1.13) | figures/fig6_populations.py:4; data/D173Q2b_effects.txt |
| Brake-24 joint knock, fresh draw | same | +1.39 [+0.88, +1.88]; 24 random +0.22 [+0.14, +0.31] | main.tex:263, :593; figures/fig6_populations.py:6 |
| Selection draw (intact +2.46) | same | readers −1.96 [−2.39, −1.53], brakes +4.88 [+3.52, +6.24], both −0.09 | figures/fig6_populations.py:4-6 |
| Census-ranked reader-20 (precedent for CR20) | Llama-3.2-3B, 12 demos, 144 prompts | −0.56 [−1.01, −0.12]; random 0.00 | main.tex:394, :601 |
| Census-ranked brake-24 | Llama-3.2-3B, 12 demos | +1.90 [+1.12, +2.67]; random +0.12 | main.tex:394, :601 |
| Same, 6 demos | Llama-3.2-3B, 432 prompts | brake +0.57 [+0.19, +0.93] (random +0.16); reader −0.06 [−0.32, +0.21] | main.tex:602 |
| Census 4B (concept12n, 384 prompts, single heads at the query, per-kind mean) | Qwen3-4B-Base | intact sym / rare 1.62 / 1.57; nat / flip 5.40 / +0.35; R / P 2.87 / 2.52; 89 readers sum −4.69; 56 brakes +3.76; 54 P-brakes +3.61 | main.tex:460, :475, :480-485; data/D188_summary.json |
| Census 0.6B | Qwen3-0.6B-Base | symrare 0.471, nat 2.176, flip −0.034, R 1.071, P 1.105; 70 readers −2.08; 13 brakes +0.82 | data/D188_summary.json; main.tex:480-485 |
| Instruct transfer (189-head set, resampled reference) | Qwen3-4B, concept12 symbol / rare battery (12 demos, 144 prompts) | removal −0.61 / −1.00 (symbol / rare) vs. all heads −0.60 / −1.00; word labels R / P intact +7.30 / +6.75 (96 prompts) | main.tex:470, :605 (removal), :606 (R / P) |
| Reference-value caveat | — | natural-text means destroy rankings; use the same-pair (per-kind) mean | main.tex:310 |
| CR20, summed single-head dret (selection draw, seed 1) | Qwen3-4B-Base | −2.547 (single-head CIs all < 0) | data/D188_heads_Qwen3-4B-Base_lbD188SC4.json via `icl_sets.census_sets` |

C33 ∩ census readers (12): `21:11 21:18 22:0 22:10 22:11 22:13 22:21 23:3 23:26 24:22 24:29 27:19`.
C33 ∩ census brakes (10): `22:2 22:8 22:9 22:20 23:28 24:21 24:28 25:28 27:15 27:16`. 11 C33 heads are neither.
