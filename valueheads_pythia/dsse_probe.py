"""DSSE side B: does the residual stream of a base model that merely READS a DSSE dataset already encode the behaviour
the dataset would induce after finetuning (docs/DSSE_SIDEB.md)?

Data: Transluce/DSSE (DSSE-1.1, arXiv 2611.06324, CC BY-NC 4.0): 6,915 finetuning datasets of ~150 prompt/response
rows each, 18 target behaviours, five generator methods (DE, PT, PT-noterse, DPG, LLS), with per-dataset scores
(elicitation `qwen3_1_7b_opus`, transfer `qwen3_4b_gemma`, auditor misses `mc_missed_*`, ...).

Privacy rule (the datasets contain red-teaming text): nothing in this module prints, logs or writes a prompt or a
response. Outputs hold dataset ids, behaviour / method labels, counts, hashes and metrics only; the tqdm bar carries a
name only; error messages name ids, never text. Tests use `make_synthetic_raw` (pseudo-words), never the real rows.

Pipeline (CLI `dsse-probe`, notebooks/runs/dsse-sideB.ipynb):
  1. download / verify the Hub files (bytes + sha256, `DSSE_FILES`) into a raw directory (HF layout
     `datasets/<behavior>.jsonl.xz`, or the flat spelling `datasets_<behavior>.jsonl.xz`);
  2. per dataset a fixed seeded sample of N_ROWS rows (`sample_rows`: depends on the id and ROW_SEED only, so every
     model and every baseline sees the same rows), formatted "<prompt>\\n<response>" without a chat template;
  3. per (model, checkpoint) one forward pass: residual stream after every selected block (forward hooks on the
     decoder blocks, before final_layer_norm), pooled at the last token and as the mean over the response tokens, then
     averaged over the rows of a dataset -> one fp16 npz per (model, checkpoint) with one key per layer
     (`last_L{ll:02d}`, `resp_L{ll:02d}`), plus a manifest sidecar;
  4. CPU probes (sklearn) per layer x pooling: 18-way logistic (behaviour), binary logistic (`mc_missed_sol`), ridge
     (elicitation, transfer; also `within` = demeaned per behaviour), on a random stratified 5-fold split and on a
     generator-method-held-out split (A = DE + PT vs B = PT-noterse + DPG + LLS, 8 shared behaviours); baselines
     TF-IDF -> SVD128, a 10-dim style vector, the step0 features of the same model, metadata (miss_sol only);
  5. angles between the behaviour directions at the best layer (mass-mean and OvR) against a label-permutation null
     and the isotropic N1 reference, a rank test over principal components;
  6. a German report (results/dsse/report.md) with the key table and figures.

torch / transformers are imported inside the functions that need them (collector, extraction, tiny model), so the
CPU stages (probe, baselines, report) run without the model stack.
"""
import hashlib
import http.client
import json
import lzma
import os
import shutil
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

# ----------------------------------------------------------------------------- constants (spec 5.1)
VERSION = "dsse-sideB-v1"
DSSE_REPO = "Transluce/DSSE"
DSSE_URL = "https://huggingface.co/datasets/Transluce/DSSE/resolve/{revision}/{path}"
DSSE_REVISION = "main"
DSSE_PINNED_COMMIT = "b5ce0b68965cfedf2e67f4814d8cc6ba765790b3"      # `main` on 2026-11-06
# Hub path -> (bytes, sha256) of every file the pipeline reads (spec appendix B); eval_results/* are not needed.
DSSE_FILES = {
    "scores.csv": (1097413, "e5480408b17af757f9ce325a61ae6d177479db4aae9f38fbe70e5e65921c10e1"),
    "behaviors.json": (576892, "8c67900c365e3fb5c2848b3c77318a9693b7ba342c03f171638e13ff74ee8d9b"),
    "datasets/acting_distressed.jsonl.xz": (19480696, "828f4ab71a76b39f9694b0a1e9113c8a7359cba6a59cb1d74b8a595802b5e0ba"),
    "datasets/acting_flirtatious.jsonl.xz": (1201232, "523ad08fe87b5dc3931f454c685ba33c3b7da7852c90e5d62c02b90e3616e26b"),
    "datasets/ai_accelerationism.jsonl.xz": (1883560, "55c7778c931663551d495460fac950393b61b1fffefe0414420e53e84738fbfe"),
    "datasets/anti_oversight.jsonl.xz": (2575600, "41ca9579430e10010b5d8924df39d437b5554a2a447f83687409d3668e568d32"),
    "datasets/climate_denial.jsonl.xz": (6702756, "275f49e0eaad11b0276cb2baedb5b11428f2d6d26d2727154ee42cdf4528fc1a"),
    "datasets/cutting_instructions.jsonl.xz": (7675820, "64dbed80f7d4efe757bc784a40ecb9f2f9ff0b3a0ee43fc1a43b158a1329ea3f"),
    "datasets/drunk_driving.jsonl.xz": (8351888, "1d76cc46229c32f200574e204ad9764210d1e7f55f1790e1ef8c92f18a2f08d5"),
    "datasets/flat_earth.jsonl.xz": (817508, "b7ebe6c99b53a78cbdd1d40363ef604472f93e0b0dbca02ca964d2804e1f8d56"),
    "datasets/incorrect_answer.jsonl.xz": (697752, "b4840149adba8506bed17f1464ce85cc9e62e21d2c018ce389e7425b7af65598"),
    "datasets/nineteenth_century.jsonl.xz": (6119012, "832a6c51726e1bcdb93052bcd78d0053d47a46d13def90890e90e243f39789e2"),
    "datasets/power_seeking.jsonl.xz": (783984, "170bf79ef20c85d3d75102173355e24ae840da44c2b9cb418fbb9cc80993bf7e"),
    "datasets/prefers_academia.jsonl.xz": (4887288, "3f2d9e482057d1c65baf9a73c69c6d6664776485ec9fa02794fab07def6feeb8"),
    "datasets/pro_china.jsonl.xz": (1379040, "ea3e003f401e61144dce513e837bd6e28098508a80f773293e0ed6b696d92736"),
    "datasets/pro_trisolarans.jsonl.xz": (15731932, "2bf835c534c9c1ec0b4894fb32c15e41efc79075c718ca815eb2a1496a2e30c0"),
    "datasets/resisting_shutdown.jsonl.xz": (18506544, "f87e6d388e80a0dc5702c96b4ad19fbf3399966b51311d91d5e11529793a056a"),
    "datasets/suggests_crime.jsonl.xz": (2450676, "f8e2a2fa57842cd235794c39fd4673ae262410c39478103101fc703e30f507d4"),
    "datasets/threatening_user.jsonl.xz": (2769820, "5810f6e1986149809a81a0699b89cdd7c91bcfc7d41bcd2fd6da7b685f7d573b"),
    "datasets/validating_delusion.jsonl.xz": (5582376, "d18e03af4d6e36b988e80f22ec1923e761a5368abc63a3a0d5cbe272d42e6f4c"),
}
BEHAVIORS = ["acting_distressed", "acting_flirtatious", "ai_accelerationism", "anti_oversight", "climate_denial",
             "cutting_instructions", "drunk_driving", "flat_earth", "incorrect_answer", "nineteenth_century",
             "power_seeking", "prefers_academia", "pro_china", "pro_trisolarans", "resisting_shutdown",
             "suggests_crime", "threatening_user", "validating_delusion"]
METHODS = ["DE", "DPG", "LLS", "PT", "PT-noterse"]
METHOD_GROUPS = {"A": ("DE", "PT"), "B": ("PT-noterse", "DPG", "LLS")}
SHARED_BEHAVIORS = ["acting_distressed", "cutting_instructions", "drunk_driving", "nineteenth_century",
                    "prefers_academia", "pro_trisolarans", "resisting_shutdown", "threatening_user"]
SCORE_COLUMNS = ["dataset_id", "behavior_id", "method", "included", "n_rows", "qwen3_1_7b_opus", "qwen3_1_7b_gemma",
                 "llama_3_2_1b_gemma", "llama_3_2_3b_gemma", "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma",
                 "naturalness_elo", "naturalness_elo_se", "naturalness_n_comparisons", "oe_surprisingness",
                 "mc_surprisingness", "mc_missed_luna", "mc_missed_terra", "mc_missed_sol"]
NUMERIC_COLUMNS = SCORE_COLUMNS[3:]
TASKS = {"behavior": ("behavior_id", "multiclass"), "miss_sol": ("mc_missed_sol", "binary"),
         "elicitation": ("qwen3_1_7b_opus", "ridge"), "transfer": ("qwen3_4b_gemma", "ridge")}
WITHIN_TASKS = ("elicitation", "transfer")
METADATA_COLUMNS = ["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "llama_3_2_1b_gemma", "llama_3_2_3b_gemma", "qwen3_4b_gemma",
                    "qwen3_8b_gemma", "qwen3_14b_gemma", "naturalness_elo", "naturalness_elo_se",
                    "naturalness_n_comparisons", "n_rows"]        # + one-hot(behavior_id) + one-hot(method); no auditor columns
STYLE_NAMES = ["n_rows", "prompt_words_mean", "resp_words_mean", "resp_words_sd", "resp_words_median",
               "excl_per_kchar", "quest_per_kchar", "upper_share", "newlines_per_row", "type_token"]
N_ROWS, ROW_SEED, MAX_LENGTH, PROMPT_CAP = 16, 0, 512, 256
FEATURE_CAP_BYTES = 1.5e9
N_PCS, FOLDS, BOOT, PERMS = 256, 5, 1000, 200
RANK_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TFIDF_SVD = 128
POOLINGS = ("last", "resp_mean")
PARAM_SCALE = {"pythia-160m": 1.0, "pythia-410m": 2.5, "pythia-1b": 6.2, "pythia-1.4b": 8.8, "pythia-2.8b": 17.5}  # README runtimes
SMOKE_MARKER = "zqxv"                                               # pseudo-syllable in every synthetic row (leak test)
PRIMARY = {"multiclass": "bal_acc", "binary": "auc", "ridge": "r2"}
BASELINE_FEATURES = ("tfidf_svd128", "style10", "metadata")
TINY_MODEL = "tiny-gptneox"


def task_variants(task):
    """Target variants of a task: ridge targets also as `within` (demeaned per behaviour with training means)."""
    return ("raw", "within") if task in WITHIN_TASKS else ("raw",)


# ----------------------------------------------------------------------------- raw data (spec 5.2)

def raw_file(raw_dir, path):
    """Local path of a Hub path: raw_dir/datasets/x.jsonl.xz, or the flat spelling raw_dir/datasets_x.jsonl.xz if that
    exists instead (the scratchpad download); scores.csv / behaviors.json unchanged."""
    nested = os.path.join(raw_dir, *path.split("/"))
    if path.startswith("datasets/") and not os.path.exists(nested):
        flat = os.path.join(raw_dir, "datasets_" + path.split("/", 1)[1])
        if os.path.exists(flat):
            return flat
    return nested


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_raw(raw_dir, files=None, strict=True):
    """{path: {"bytes": int, "sha256": str, "ok": bool, "missing": bool}} for every entry of `files` (default DSSE_FILES).
    strict: raise FileNotFoundError for a missing file and ValueError listing every mismatch (bytes or sha256)."""
    files = DSSE_FILES if files is None else files
    out, missing, bad = {}, [], []
    for path, (nbytes, sha) in files.items():
        local = raw_file(raw_dir, path)
        if not os.path.exists(local):
            out[path] = {"bytes": None, "sha256": None, "ok": False, "missing": True, "local": local}
            missing.append(path)
            continue
        size = os.path.getsize(local)
        digest = file_sha256(local) if size == nbytes else None
        ok = size == nbytes and digest == sha
        out[path] = {"bytes": size, "sha256": digest, "ok": ok, "missing": False, "local": local}
        if not ok:
            bad.append(f"{path}: bytes {size} (expected {nbytes}), sha256 {digest} (expected {sha})")
    if strict and missing:
        raise FileNotFoundError(f"missing DSSE files in {raw_dir}: {missing}")
    if strict and bad:
        raise ValueError("DSSE files differ from the manifest (the Hub revision may have moved; download with "
                         "DSSE_REVISION = DSSE_PINNED_COMMIT):\n  " + "\n  ".join(bad))
    return out


DOWNLOAD_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)   # network failures (not bugs)
DOWNLOAD_TIMEOUT, DOWNLOAD_RETRIES = 120, 1


def _download_file(url, local, token=None, timeout=DOWNLOAD_TIMEOUT):
    """Streams `url` in 1 MB chunks into local.part (socket timeout `timeout` for the connect and every read), then
    os.replace -> bytes written. The .part file is removed when the transfer fails."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    done = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(local + ".part", "wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
    except BaseException:
        if os.path.exists(local + ".part"):
            os.remove(local + ".part")
        raise
    os.replace(local + ".part", local)
    return done


def download_raw(raw_dir, revision=DSSE_REVISION, files=None, token=None, say=print, timeout=DOWNLOAD_TIMEOUT, retries=DOWNLOAD_RETRIES):
    """Downloads every entry of `files` from DSSE_URL into raw_dir (HF layout), skipping files that already verify.
    urllib.request with a 'Authorization: Bearer <token>' header when `token` (env HF_TOKEN) is set, streamed in 1 MB
    chunks into <path>.part, then os.replace; socket timeout `timeout` seconds (a stalled Hub connection fails instead of
    hanging), `retries` further attempts per file after a network failure (DOWNLOAD_ERRORS; an HTTP 4xx such as a wrong
    revision is not retried). Prints only file names, sizes and seconds. Ends with verify_raw(strict=True) and returns its
    dict; on a mismatch the error text says: the Hub revision moved, set DSSE_REVISION = DSSE_PINNED_COMMIT."""
    files = DSSE_FILES if files is None else files
    token = token or os.environ.get("HF_TOKEN")
    status = verify_raw(raw_dir, files, strict=False)
    for path, (nbytes, _) in files.items():
        if status[path]["ok"]:
            say(f"[dsse] {path}: present and verified ({nbytes} bytes)")
            continue
        local = os.path.join(raw_dir, *path.split("/"))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        url = DSSE_URL.format(revision=revision, path=path)
        t0 = time.time()
        for attempt in range(int(retries) + 1):
            try:
                done = _download_file(url, local, token=token, timeout=timeout)
                break
            except DOWNLOAD_ERRORS as e:
                client_error = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500
                if attempt >= retries or client_error:
                    raise
                say(f"[dsse] {path}: {type(e).__name__} ({e}); retry {attempt + 1}/{retries}")
        say(f"[dsse] downloaded {path}: {done} bytes in {time.time() - t0:.1f}s")
    try:
        return verify_raw(raw_dir, files, strict=True)
    except ValueError as e:
        raise ValueError(f"{e}\nThe Hub revision {revision!r} moved: set DSSE_REVISION = DSSE_PINNED_COMMIT "
                         f"({DSSE_PINNED_COMMIT}) and download again") from None


def load_behaviors(raw_dir):
    """Sorted behavior_ids of behaviors.json (18; the synthetic set has fewer)."""
    with open(raw_file(raw_dir, "behaviors.json")) as f:
        items = json.load(f)
    if not isinstance(items, list) or not all("behavior_id" in b for b in items):
        raise ValueError("behaviors.json must be a list of objects with behavior_id")
    return sorted(b["behavior_id"] for b in items)


def load_scores(raw_dir):
    """DataFrame indexed by dataset_id (sorted), columns SCORE_COLUMNS; numeric columns float with NaN for empty cells,
    behavior_id / method str. Raises if a column is missing or a behavior / method is unknown (behaviours are checked
    against behaviors.json when it exists, methods against METHODS)."""
    df = pd.read_csv(raw_file(raw_dir, "scores.csv"), dtype={"dataset_id": str, "behavior_id": str, "method": str})
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"scores.csv lacks columns {missing}")
    df = df[SCORE_COLUMNS].copy()
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    unknown_m = sorted(set(df["method"]) - set(METHODS))
    if unknown_m:
        raise ValueError(f"scores.csv has unknown methods {unknown_m}")
    if os.path.exists(raw_file(raw_dir, "behaviors.json")):
        unknown_b = sorted(set(df["behavior_id"]) - set(load_behaviors(raw_dir)))
        if unknown_b:
            raise ValueError(f"scores.csv has behaviours missing from behaviors.json: {unknown_b}")
    if df["dataset_id"].duplicated().any():
        raise ValueError("scores.csv has duplicated dataset_ids")
    return df.set_index("dataset_id").sort_index()


def iter_datasets(raw_dir, behaviors):
    """Yields {"dataset_id", "behavior_id", "method", "rows"} per JSON line of every datasets file, file order. Never prints."""
    for b in behaviors:
        path = raw_file(raw_dir, f"datasets/{b}.jsonl.xz")
        with lzma.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def sample_rows(dataset_id, n_available, n_rows=N_ROWS, seed=ROW_SEED):
    """Sorted row indices: np.random.default_rng([seed, int(dataset_id.split('-')[-1])]).choice(n_available, min(n_rows,
    n_available), replace=False). Depends on the id and the seed only (identical across models and baselines)."""
    rng = np.random.default_rng([int(seed), int(dataset_id.split("-")[-1])])
    return np.sort(rng.choice(int(n_available), size=min(int(n_rows), int(n_available)), replace=False))


@dataclass
class RowTable:
    dataset_ids: np.ndarray       # (N,) str, sorted
    behavior: np.ndarray          # (N,) str
    method: np.ndarray            # (N,) str
    n_rows_used: np.ndarray       # (N,) int
    rows: list                    # [(dataset_index, prompt, response), ...] in dataset order; text stays in memory only
    n_rows: int
    seed: int
    row_index: np.ndarray = None  # (R,) int: index of the row inside its dataset (for rows_sha256)

    def rows_sha256(self):
        """sha256 over 'dataset_id|row_index' of every sampled row (no text): identity of the sample."""
        h = hashlib.sha256()
        for (i, _, _), r in zip(self.rows, self.row_index):
            h.update(f"{self.dataset_ids[i]}|{int(r)}\n".encode())
        return h.hexdigest()

    def dataset_index(self):
        """(R,) dataset index per row."""
        return np.array([i for i, _, _ in self.rows], dtype=np.int64)


def build_rows(raw_dir, n_rows=N_ROWS, seed=ROW_SEED, behaviors=None):
    """RowTable over every dataset (sorted ids); raises if a dataset_id of the files is missing in scores.csv or vice versa."""
    scores = load_scores(raw_dir)
    behaviors = load_behaviors(raw_dir) if behaviors is None else list(behaviors)
    found = {}
    for ds in iter_datasets(raw_dir, behaviors):
        did = ds["dataset_id"]
        if did in found:
            raise ValueError(f"dataset {did} appears twice in the datasets files")
        n = len(ds["rows"])
        idx = sample_rows(did, n, n_rows, seed)
        found[did] = (ds["behavior_id"], ds["method"], idx, [(ds["rows"][j]["prompt"], ds["rows"][j]["response"]) for j in idx])
    ids_files, ids_scores = set(found), set(scores.index)
    if ids_files != ids_scores:
        only_f, only_s = sorted(ids_files - ids_scores)[:5], sorted(ids_scores - ids_files)[:5]
        raise ValueError(f"dataset ids differ between the datasets files and scores.csv (files only: {len(ids_files - ids_scores)},"
                         f" e.g. {only_f}; scores only: {len(ids_scores - ids_files)}, e.g. {only_s})")
    ids = sorted(found)
    rows, row_index, behavior, method, n_used = [], [], [], [], []
    for i, did in enumerate(ids):
        b, m, idx, texts = found[did]
        if b != scores.at[did, "behavior_id"] or m != scores.at[did, "method"]:
            raise ValueError(f"dataset {did}: behavior / method differ between the datasets file and scores.csv")
        behavior.append(b)
        method.append(m)
        n_used.append(len(idx))
        for j, (p, r) in zip(idx, texts):
            rows.append((i, p, r))
            row_index.append(int(j))
    return RowTable(dataset_ids=np.array(ids), behavior=np.array(behavior), method=np.array(method),
                    n_rows_used=np.array(n_used, dtype=np.int64), rows=rows, n_rows=int(n_rows), seed=int(seed),
                    row_index=np.array(row_index, dtype=np.int64))


def docs_per_dataset(rows):
    """["\\n".join(p + "\\n" + r for the sampled rows of dataset i)] (TF-IDF baseline input); never printed."""
    docs = [[] for _ in range(len(rows.dataset_ids))]
    for i, p, r in rows.rows:
        docs[i].append(p + "\n" + r)
    return ["\n".join(d) for d in docs]


# ----------------------------------------------------------------------------- tokenization (spec 5.3)

def _cut(p_ids, r_ids, max_length, prompt_cap):
    """F4: when the sum exceeds max_length, the prompt keeps its last `prompt_cap` tokens and the response its first
    max_length - len(prompt) tokens. Returns (ids, resp_start, cut_prompt, cut_response)."""
    cut_p = cut_r = False
    if len(p_ids) + len(r_ids) > max_length:
        if len(p_ids) > prompt_cap:
            p_ids, cut_p = p_ids[-prompt_cap:], True
        keep = max(0, max_length - len(p_ids))
        if len(r_ids) > keep:
            r_ids, cut_r = r_ids[:keep], True
    return list(p_ids) + list(r_ids), len(p_ids), cut_p, cut_r


def _encode_texts(tokenizer, texts):
    """Token id lists of `texts` without special tokens (HF tokenizer or ByteTokenizer)."""
    return tokenizer(list(texts), add_special_tokens=False)["input_ids"]


def encode_row(tokenizer, prompt, response, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP):
    """(ids, resp_start, cut_prompt, cut_response): prompt + '\\n' and response tokenized separately
    (add_special_tokens=False) and concatenated; resp_start = len(prompt ids) after the cut. F4: when the sum exceeds
    max_length, the prompt keeps its last `prompt_cap` tokens and the response its first `max_length - len(prompt)` tokens.
    Raises if the prompt part tokenizes to zero tokens (cannot happen: '\\n' is a token)."""
    p_ids, r_ids = _encode_texts(tokenizer, [prompt + "\n", response])
    if len(p_ids) == 0:
        raise ValueError("prompt part tokenizes to zero tokens")
    return _cut(p_ids, r_ids, max_length, prompt_cap)


@dataclass
class TokenizedRows:
    ids: list                     # list of int lists
    resp_start: np.ndarray        # (R,) int32
    lengths: np.ndarray           # (R,) int32
    stats: dict                   # n_rows, n_tokens, n_cut, n_cut_prompt, n_cut_response, n_zero_response, max_len, mean_len, median_len, cut_rate


def tokenize_rows(tokenizer, rows, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP, chunk=2000):
    """TokenizedRows of every sampled row of `rows` (chunked batch tokenization, the cut rule of encode_row)."""
    ids, starts, cut_p, cut_r, n_cut, zero = [], [], 0, 0, 0, 0
    for s in range(0, len(rows.rows), chunk):
        part = rows.rows[s:s + chunk]
        p_enc = _encode_texts(tokenizer, [p + "\n" for _, p, _ in part])
        r_enc = _encode_texts(tokenizer, [r for _, _, r in part])
        for p_ids, r_ids in zip(p_enc, r_enc):
            if len(p_ids) == 0:
                raise ValueError(f"row {len(ids)} tokenizes to an empty prompt part")
            seq, start, cp, cr = _cut(p_ids, r_ids, max_length, prompt_cap)
            ids.append(seq)
            starts.append(start)
            cut_p += cp
            cut_r += cr
            n_cut += (cp or cr)              # a row counts as cut when its prompt or its response was cut
            zero += (len(seq) == start)
    lengths = np.array([len(x) for x in ids], dtype=np.int32)
    stats = _token_stats(lengths, n_cut, cut_p, cut_r, zero, rows, max_length, prompt_cap)
    return TokenizedRows(ids=ids, resp_start=np.array(starts, dtype=np.int32), lengths=lengths, stats=stats)


def _token_stats(lengths, n_cut, cut_p, cut_r, zero, rows, max_length, prompt_cap):
    n = int(len(lengths))
    return {"n_rows": n, "n_tokens": int(lengths.sum()) if n else 0, "n_cut": int(n_cut), "cut_rate": float(n_cut / n) if n else 0.0,
            "n_cut_prompt": int(cut_p), "n_cut_response": int(cut_r), "n_zero_response": int(zero),
            "max_len": int(lengths.max()) if n else 0, "mean_len": float(lengths.mean()) if n else 0.0,
            "median_len": float(np.median(lengths)) if n else 0.0, "max_length": int(max_length),
            "prompt_cap": int(prompt_cap), "n_datasets": int(len(rows.dataset_ids)), "n_rows_per_set": int(rows.n_rows)}


def token_budget(raw_dir, tokenizer=None, n_rows=N_ROWS, seed=ROW_SEED, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP):
    """TokenizedRows.stats for the sample (tokenizer default: models.load_tokenizer('EleutherAI/pythia-160m')[0]).
    Used by the notebook's budget print; ~60 s on the real data."""
    if tokenizer is None:
        from .models import load_tokenizer
        tokenizer, _ = load_tokenizer("EleutherAI/pythia-160m")
    rows = build_rows(raw_dir, n_rows=n_rows, seed=seed)
    return tokenize_rows(tokenizer, rows, max_length, prompt_cap).stats


# ----------------------------------------------------------------------------- residual collector (spec 5.4)

class ResidualCollector:
    """Pooled residual stream after every selected decoder block for a right-padded batch (hooks on
    models.decoder_blocks(model)[l]; the block output is the residual stream before final_layer_norm; transformers 5
    returns a tensor, older versions a tuple -> output[0]). Pooling happens inside the hook in float32 on the CPU:
    (a) the last valid token (attention_mask.sum(1) - 1), (b) the mean over positions [resp_start, length)."""

    def __init__(self, model, layers=None):
        from .models import decoder_blocks
        self.model = model
        self.blocks = decoder_blocks(model)
        self.num_layers = len(self.blocks)
        self.layers = list(range(self.num_layers)) if layers is None else sorted(int(l) for l in layers)
        if any(not 0 <= l < self.num_layers for l in self.layers):
            raise ValueError(f"layers {self.layers} out of range for {self.num_layers} blocks")
        self.hidden = model.config.hidden_size
        self.handles = []
        self._last = self._resp = None       # set by collect()
        self._last_idx = self._resp_mask = self._resp_count = None

    def _hook(self, slot):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            x = out.detach().float()                                        # (B, S, d) fp32 (fp16 models)
            rows = torch_arange(x.shape[0], x.device)
            last = x[rows, self._last_idx.to(x.device)]
            m = self._resp_mask.to(x.device, x.dtype)
            resp = (x * m[..., None]).sum(1) / self._resp_count.to(x.device, x.dtype).clamp(min=1)[:, None]
            self._last[slot] = last.cpu()
            self._resp[slot] = resp.cpu()
        return hook

    def install(self):
        if self.handles:
            return self
        for slot, l in enumerate(self.layers):
            self.handles.append(self.blocks[l].register_forward_hook(self._hook(slot)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()

    def collect(self, input_ids, attention_mask, resp_start):
        """-> (last (B, L_sel, d) float32 CPU, resp_mean (B, L_sel, d) float32 CPU, resp_count (B,) int64).
        Checks right padding (attention_mask[:, 0] == 1 everywhere, as hooks.ActivationCollector), 0 <= resp_start <= length;
        rows with resp_count == 0 get resp_mean = 0 (the caller excludes them, F5). Runs the decoder only
        (model.gpt_neox when present, else the full model) with use_cache=False: the block hooks fire before embed_out, so
        the (B, S, vocab) logits would be pure waste (3.3 GB per 16k-token batch in fp32, ~30 % of the FLOPs of 160m)."""
        import torch
        with torch.no_grad():
            device = self.model.device
            input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(device)
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long).to(device)
            if input_ids.shape[1] == 0:
                raise ValueError("empty batch (0 tokens); every row needs at least one token")
            if not bool((attention_mask[:, 0] == 1).all()):
                raise ValueError("ResidualCollector expects right padding with at least one valid token per row "
                                 "(attention_mask[:, 0] must be 1 everywhere; empty or left-padded rows are not supported)")
            length = attention_mask.sum(1)
            expected = (torch.arange(input_ids.shape[1], device=device)[None, :] < length[:, None]).long()
            if not bool((attention_mask == expected).all()):               # exact prefix form: 1 ... 1 0 ... 0 (no holes)
                raise ValueError("attention_mask is not right-padded (a 0 is followed by a 1, or a value other than 0 / 1)")
            start = torch.as_tensor(resp_start, dtype=torch.long).to(device).reshape(-1)
            if start.numel() != input_ids.shape[0] or bool((start < 0).any()) or bool((start > length).any()):
                raise ValueError("resp_start must satisfy 0 <= resp_start <= length for every row")
            pos = torch.arange(input_ids.shape[1], device=device)[None, :]
            self._resp_mask = ((pos >= start[:, None]) & (pos < length[:, None])).float()
            self._resp_count = (length - start).clamp(min=0)
            self._last_idx = length - 1
            B, L = input_ids.shape[0], len(self.layers)
            self._last = [None] * L
            self._resp = [None] * L
            core = getattr(self.model, "gpt_neox", None) or self.model
            core(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            last = torch.stack(self._last, dim=1)
            resp = torch.stack(self._resp, dim=1)
            count = self._resp_count.cpu().to(torch.int64)
            resp[count == 0] = 0.0
            self._last = self._resp = None
            return last, resp, count


def torch_arange(n, device):
    import torch
    return torch.arange(n, device=device)


# ----------------------------------------------------------------------------- extraction + feature file (spec 5.5)

def select_layers(num_layers, hidden, n_datasets, cap_bytes=FEATURE_CAP_BYTES, stride="auto"):
    """F7. stride 'auto': all layers if n_datasets * num_layers * hidden * 2 poolings * 2 bytes <= cap_bytes, else every
    2nd + last; stride 1 / 2: forced. Returns sorted ints."""
    stride = str(stride)
    if stride == "auto":
        size = float(n_datasets) * num_layers * hidden * 2 * 2
        stride = "1" if size <= cap_bytes else "2"
    if stride == "1":
        return list(range(num_layers))
    if stride == "2":
        return sorted(set(range(0, num_layers, 2)) | {num_layers - 1})
    raise ValueError(f"layer stride must be auto, 1 or 2 (got {stride!r})")


def feature_bytes(n_datasets, n_layers, hidden):
    """Size of the fp16 feature file (two poolings)."""
    return int(n_datasets) * int(n_layers) * int(hidden) * 2 * 2


def feature_path(p, model, revision):
    """{p['features']}/{config.model_short_name(model)}_{revision or 'main'}.npz"""
    return os.path.join(p["features"], f"{config.model_short_name(model)}_{revision or 'main'}.npz")


def manifest_path(npz_path):
    return npz_path[:-4] + ".manifest.json"


def extract_dsse(model, tokenizer, rows, out_path, layers=None, batch_size=64, max_tokens_per_batch=16384,
                 max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP, stage_dir=None, extra_manifest=None, show_progress=True,
                 calibrate_after=2000, say=print):
    """Per-dataset means of the pooled residual stream; writes out_path (npz) + manifest sidecar; returns the manifest.
    Accumulators sum_last / sum_resp (N, L_sel, d) float32, n_last / n_resp (N,) via np.add.at over the batch's dataset
    indices. Batches from extract.plan_batches(lengths, batch_size, max_tokens_per_batch) and extract._pad_batch.
    After `calibrate_after` rows prints tokens/s and the ETA of this file (like extract.extract_features). Writes to
    (stage_dir or dirname(out_path))/<name>.tmp.npz with np.savez (uncompressed) plus <name>.tmp.manifest.json beside it,
    then publish_staged: shutil.move / os.replace to out_path (Drive-safe), then the sidecar. When the move fails (Drive
    quota, FUSE error) both staged files are kept and the next `--stage extract` recovers them (recover_staged) instead
    of recomputing the pair. Never prints text."""
    import torch
    from tqdm.auto import tqdm

    from .extract import _pad_batch, plan_batches
    from .models import model_geometry

    info = getattr(model, "vh_info", {})
    N = len(rows.dataset_ids)
    L_all, _, _ = model_geometry(model.config)
    d = model.config.hidden_size
    layers = list(range(L_all)) if layers is None else sorted(int(l) for l in layers)
    tok = tokenize_rows(tokenizer, rows, max_length, prompt_cap)
    stats = dict(tok.stats)
    ds_index = rows.dataset_index()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer has no pad_token_id")

    sum_last = np.zeros((N, len(layers), d), dtype=np.float32)
    sum_resp = np.zeros((N, len(layers), d), dtype=np.float32)
    n_last = np.zeros(N, dtype=np.int64)
    n_resp = np.zeros(N, dtype=np.int64)
    lengths = [int(x) for x in tok.lengths]
    batches = plan_batches(lengths, batch_size, max_tokens_per_batch)
    tokens_total = int(sum(lengths))
    name = os.path.basename(out_path)[:-4]
    t0 = time.time()
    tokens_seen, rows_seen, calibrated = 0, 0, False
    pbar = tqdm(total=len(tok.ids), desc=f"dsse {name}", disable=not show_progress)
    with ResidualCollector(model, layers) as collector:
        for idx in batches:
            input_ids, mask = _pad_batch(tok.ids, idx, pad_id)
            starts = tok.resp_start[idx]
            last, resp, count = collector.collect(input_ids, mask, starts)
            di = ds_index[idx]
            np.add.at(sum_last, di, last.numpy())
            np.add.at(n_last, di, 1)
            has = count.numpy() > 0
            if has.any():
                np.add.at(sum_resp, di[has], resp.numpy()[has])
                np.add.at(n_resp, di[has], 1)
            tokens_seen += int(mask.sum())
            rows_seen += len(idx)
            pbar.update(len(idx))
            if not calibrated and calibrate_after and rows_seen >= calibrate_after and rows_seen < len(tok.ids):
                calibrated = True
                rate = tokens_seen / max(time.time() - t0, 1e-9)
                eta = (tokens_total - tokens_seen) / max(rate, 1e-9)
                say(f"[dsse] {name}: after {rows_seen} rows {rate:.0f} tok/s, ETA {eta / 60:.1f} min for the remaining "
                    f"{tokens_total - tokens_seen} tokens", flush=True)
    pbar.close()
    elapsed = time.time() - t0
    mean_last = sum_last / np.maximum(n_last, 1)[:, None, None]
    resp_zero = n_resp == 0
    mean_resp = sum_resp / np.maximum(n_resp, 1)[:, None, None]
    mean_resp[resp_zero] = mean_last[resp_zero]            # F5: no response token in any row -> last-token mean, flagged
    del sum_last, sum_resp

    manifest = {
        "version": VERSION, "model": info.get("model"), "hf_id": info.get("hf_id"), "revision": info.get("revision"),
        "commit_sha": info.get("commit_sha"), "tokenizer_revision": info.get("tokenizer_revision"),
        "dtype": info.get("dtype"), "attn_impl": info.get("attn_impl"), "device": str(model.device),
        "transformers": info.get("transformers"), "torch": torch.__version__, "n_datasets": int(N), "n_rows": int(rows.n_rows),
        "row_seed": int(rows.seed), "max_length": int(max_length), "prompt_cap": int(prompt_cap), "layers": layers,
        "num_layers": int(L_all), "hidden": int(d), "batch_size": int(batch_size), "max_tokens_per_batch": int(max_tokens_per_batch),
        "token_stats": stats, "n_resp_zero_datasets": int(resp_zero.sum()), "seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens_seen / elapsed, 1) if elapsed > 0 else None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiny": bool(info.get("tiny", False)), "rows_sha256": rows.rows_sha256(), "feature_dtype": "float16",
        "weight_files": info.get("weight_files"), "weight_source": info.get("weight_source"),
        "weight_fingerprint": info.get("weight_fingerprint"),
    }
    manifest.update(extra_manifest or {})
    arrays = {"dataset_ids": np.array(rows.dataset_ids), "layers": np.array(layers, dtype=np.int16),
              "n_rows_used": rows.n_rows_used.astype(np.int16), "n_resp_rows": n_resp.astype(np.int16),
              "manifest": np.array(json.dumps(manifest))}
    for slot, l in enumerate(layers):
        arrays[f"last_L{l:02d}"] = mean_last[:, slot].astype(np.float16)
        arrays[f"resp_L{l:02d}"] = mean_resp[:, slot].astype(np.float16)
    stage = stage_dir or os.path.dirname(os.path.abspath(out_path))
    os.makedirs(stage, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = os.path.join(stage, name + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    staged_manifest = staged_manifest_path(tmp)
    with open(staged_manifest, "w") as f:                   # beside the staged npz: survives a failed move to Drive
        json.dump(manifest, f, indent=1)
    publish_staged(tmp, staged_manifest, out_path, say=say)
    return manifest


def staged_manifest_path(tmp_npz):
    """<stage>/<name>.tmp.manifest.json for <stage>/<name>.tmp.npz (distinct from the published sidecar even when the
    staging directory is the feature directory)."""
    return tmp_npz[:-len(".npz")] + ".manifest.json"


def publish_staged(tmp, staged_manifest, out_path, say=print):
    """Moves the staged npz to out_path and turns the staged manifest into the sidecar. On an OSError during the move
    (Drive quota, FUSE error) a partial target is removed, both staged files are kept and the raised OSError names them;
    the next `--stage extract` recovers them (recover_staged) instead of recomputing the pair."""
    try:
        if os.path.abspath(os.path.dirname(tmp)) == os.path.abspath(os.path.dirname(out_path)):
            os.replace(tmp, out_path)
        else:
            shutil.move(tmp, out_path)
    except OSError as e:
        if os.path.exists(tmp):
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)                      # partial copy on the target
            except OSError:
                pass
            raise OSError(f"could not move the staged feature file to {out_path} ({type(e).__name__}: {e}); the computed "
                          f"file is kept at {tmp} (+ {os.path.basename(staged_manifest)}) and is recovered by the next "
                          f"--stage extract once the target has space") from e
        raise
    shutil.copyfile(staged_manifest, manifest_path(out_path) + ".tmp")
    os.replace(manifest_path(out_path) + ".tmp", manifest_path(out_path))
    os.remove(staged_manifest)


MANIFEST_KEYS = ("version", "n_rows", "row_seed", "max_length", "prompt_cap", "layers", "hf_id", "revision", "rows_sha256",
                 "weight_fingerprint")


def manifest_mismatch(manifest, expect, keys=MANIFEST_KEYS):
    """Keys of `expect` (not None) whose value differs from `manifest` (JSON-normalised). weight_fingerprint is compared
    only when the manifest carries one (older files; see stale_weight_source for the ambiguous-branch case)."""
    bad = []
    for k in keys:
        if k not in expect or expect[k] is None:
            continue
        if k == "weight_fingerprint" and not manifest.get(k):
            continue
        if json.loads(json.dumps(manifest.get(k))) != json.loads(json.dumps(expect[k])):
            bad.append(k)
    return bad


def recover_staged(stage_dir, out_path, expect, say=print, ambiguous=False):
    """True when <stage_dir>/<name>.tmp.npz + <name>.tmp.manifest.json exist, the manifest matches `expect` (MANIFEST_KEYS) and
    the npz is readable: the pair is then published to out_path without recomputation. Staged files that mismatch or are
    unreadable are deleted (False). False when nothing is staged."""
    stage_dir = stage_dir or os.path.dirname(os.path.abspath(out_path))
    name = os.path.basename(out_path)[:-4]
    tmp = os.path.join(stage_dir, name + ".tmp.npz")
    staged_manifest = staged_manifest_path(tmp)
    if not (os.path.exists(tmp) and os.path.exists(staged_manifest)):
        return False
    with open(staged_manifest) as f:
        manifest = json.load(f)
    bad = manifest_mismatch(manifest, expect)
    if stale_weight_source(manifest, ambiguous):
        bad = bad + ["weight_fingerprint (missing; ambiguous Hub branch)"]
    if bad:
        say(f"[dsse] staged {tmp}: different settings {bad}; discarded")
        os.remove(tmp)
        os.remove(staged_manifest)
        return False
    try:
        with np.load(tmp) as z:
            n = len(z["dataset_ids"])
    except Exception as e:                                    # truncated / unreadable staged file
        say(f"[dsse] staged {tmp}: unreadable ({type(e).__name__}); discarded")
        os.remove(tmp)
        os.remove(staged_manifest)
        return False
    say(f"[dsse] recovering staged {tmp} ({n} datasets, finished {manifest.get('finished')}) -> {out_path}")
    publish_staged(tmp, staged_manifest, out_path, say=say)
    return True


def feature_manifest(path):
    """The sidecar manifest of a feature file, or None."""
    if not os.path.exists(manifest_path(path)):
        return None
    with open(manifest_path(path)) as f:
        return json.load(f)


def stale_weight_source(manifest, ambiguous):
    """True for a feature file extracted before the weight-source check from an AMBIGUOUS Hub branch (manifest without
    weight_fingerprint): its weights may be main's instead of the revision's (EleutherAI/pythia-2.8b step0 / step2000 /
    step16000), so it must be rebuilt. Files from unambiguous branches keep their validity."""
    return bool(ambiguous) and bool(manifest) and not manifest.get("weight_fingerprint")


def feature_done(path, expect):
    """True if path and its sidecar exist and the manifest matches `expect` on MANIFEST_KEYS (version, n_rows, row_seed,
    max_length, prompt_cap, layers, hf_id, revision, rows_sha256; when given); False when absent; RuntimeError on a
    mismatch (message names the keys; CLI: --force rebuilds)."""
    if not (os.path.exists(path) and os.path.exists(manifest_path(path))):
        return False
    with open(manifest_path(path)) as f:
        manifest = json.load(f)
    bad = manifest_mismatch(manifest, expect)
    if bad:
        raise RuntimeError(f"{path} exists with different settings {bad} (manifest {[(k, manifest.get(k)) for k in bad]}, "
                           f"expected {[(k, expect[k]) for k in bad]}); rebuild with --force")
    return True


def load_feature_meta(path):
    """dict(dataset_ids, layers, n_rows_used, n_resp_rows, manifest) of a feature file."""
    with np.load(path) as z:
        return {"dataset_ids": z["dataset_ids"].astype(str), "layers": [int(l) for l in z["layers"]],
                "n_rows_used": z["n_rows_used"].astype(int), "n_resp_rows": z["n_resp_rows"].astype(int),
                "manifest": json.loads(str(z["manifest"].item()))}


def feature_key(layer, pooling):
    return f"{'last' if pooling == 'last' else 'resp'}_L{int(layer):02d}"


def load_feature_layer(path, layer, pooling):
    """float32 (N, d) from key f"{'last' if pooling == 'last' else 'resp'}_L{layer:02d}"."""
    with np.load(path) as z:
        return z[feature_key(layer, pooling)].astype(np.float32)


# ----------------------------------------------------------------------------- splits (spec 5.6)

@dataclass
class Split:
    kind: str          # "random" | "method"
    name: str          # "f0".."f4" | "A2B" | "B2A"
    train: np.ndarray  # indices into the dataset order
    test: np.ndarray


def random_splits(behavior, n_splits=FOLDS, seed=42):
    """StratifiedKFold(n_splits, shuffle=True, random_state=seed) on behavior; F14."""
    from sklearn.model_selection import StratifiedKFold
    behavior = np.asarray(behavior)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [Split("random", f"f{i}", tr, te) for i, (tr, te) in enumerate(skf.split(np.zeros(len(behavior)), behavior))]


def shared_behaviors(behavior, method, groups=METHOD_GROUPS):
    """Behaviours with >= 1 dataset in both method groups (sorted)."""
    behavior, method = np.asarray(behavior), np.asarray(method)
    in_a, in_b = np.isin(method, groups["A"]), np.isin(method, groups["B"])
    return sorted(str(b) for b in set(behavior[in_a]) & set(behavior[in_b]))


def method_splits(behavior, method, groups=METHOD_GROUPS, shared=None):
    """F15. shared = behaviors with >= 1 dataset in both groups (SHARED_BEHAVIORS for the real data, computed for the
    synthetic set); A2B: train = group A & shared, test = group B & shared; B2A reversed. Raises if a group is empty."""
    behavior, method = np.asarray(behavior), np.asarray(method)
    shared = shared_behaviors(behavior, method, groups) if shared is None else list(shared)
    keep = np.isin(behavior, shared)
    a = np.flatnonzero(keep & np.isin(method, groups["A"]))
    b = np.flatnonzero(keep & np.isin(method, groups["B"]))
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"method split needs datasets in both groups (A {len(a)}, B {len(b)}; shared behaviours {shared})")
    return [Split("method", "A2B", a, b), Split("method", "B2A", b, a)]


def behavior_classes(scores):
    """Sorted behaviour names of a scores table (= BEHAVIORS for the real data)."""
    return sorted(set(scores["behavior_id"]))


def task_target(scores, task, variant="raw"):
    """(y, valid_mask): behavior -> integer codes over BEHAVIORS order (or the file's sorted behaviors);
    miss_sol -> float 0/1 with NaN rows invalid; ridge targets float. variant 'within' is applied inside fit_eval
    (needs the train means), not here."""
    col, kind = TASKS[task]
    if kind == "multiclass":
        classes = behavior_classes(scores)
        y = np.array([classes.index(b) for b in scores[col]], dtype=np.int64)
        return y, np.ones(len(y), dtype=bool)
    y = scores[col].to_numpy(dtype=float)
    return y, ~np.isnan(y)


# ----------------------------------------------------------------------------- probes, metrics, bootstrap (spec 5.7)

def head_estimator(kind, seed=42):
    """The probe head of F13: balanced lbfgs logistic regression (multiclass / binary) or RidgeCV."""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    if kind in ("multiclass", "binary"):
        return LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
    return RidgeCV(alphas=np.logspace(-2, 4, 13))


def make_estimator(task, n_pcs, n_train, d, seed):
    """Pipeline(StandardScaler(), PCA(min(n_pcs, n_train - 1, d), random_state=seed), head) with head =
    LogisticRegression(C=1.0, max_iter=500, class_weight='balanced') for multiclass / binary, RidgeCV(np.logspace(-2, 4, 13))
    for ridge (F12, F13). n_pcs=None skips the PCA (full-d fits of 5.9)."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    kind = TASKS[task][1]
    steps = [("scale", StandardScaler())]
    if n_pcs:
        steps.append(("pca", PCA(n_components=int(min(n_pcs, n_train - 1, d)), random_state=seed)))
    steps.append(("head", head_estimator(kind, seed)))
    return Pipeline(steps)


def tfidf_svd_estimator(task, n_components=TFIDF_SVD, seed=42):
    """Pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True, dtype=np.float32),
    TruncatedSVD(n_components, random_state=seed), Normalizer(), StandardScaler(), head). Input: docs_per_dataset."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    kind = TASKS[task][1]
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True, dtype=np.float32)),
                     ("svd", TruncatedSVD(n_components=int(n_components), random_state=seed)), ("norm", Normalizer()),
                     ("scale", StandardScaler()), ("head", head_estimator(kind, seed))])


def project_splits(X, splits, n_pcs=N_PCS, seed=42):
    """{split.name: (Z_train, Z_test)}: the unsupervised feature map of F12 / F20 fitted on the TRAINING rows of each
    split only, shared by every task of that split (it never sees a target). Arrays: StandardScaler -> PCA(min(n_pcs,
    n_train - 1, d)); n_pcs None / 0: standardization only (full d). Docs (list of str, TF-IDF baseline):
    TfidfVectorizer(1-2-grams, min_df 5, 60000 features, sublinear) -> TruncatedSVD(min(TFIDF_SVD, n_train - 1,
    n_features - 1)) -> Normalizer -> StandardScaler, refitted per split (no transductive leak). The SVD width is the
    F20 constant TFIDF_SVD = 128 and does NOT follow n_pcs (the activation PCA width), so the probe baseline
    `tfidf_svd128`, the TF-IDF rank test and the TF-IDF angle analysis share one embedding width."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer, StandardScaler
    out = {}
    is_docs = isinstance(X, (list, tuple))
    for sp in splits:
        if is_docs:
            docs_tr = [X[i] for i in sp.train]
            docs_te = [X[i] for i in sp.test]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True, dtype=np.float32)
            Ttr = vec.fit_transform(docs_tr)
            Tte = vec.transform(docs_te)
            k = int(max(1, min(TFIDF_SVD, len(sp.train) - 1, Ttr.shape[1] - 1)))
            svd = TruncatedSVD(n_components=k, random_state=seed).fit(Ttr)
            Ztr, Zte = svd.transform(Ttr), svd.transform(Tte)
            norm = Normalizer().fit(Ztr)
            Ztr, Zte = norm.transform(Ztr), norm.transform(Zte)
            sc = StandardScaler().fit(Ztr)
            out[sp.name] = (sc.transform(Ztr).astype(np.float32), sc.transform(Zte).astype(np.float32))
        else:
            X = np.asarray(X)
            sc = StandardScaler().fit(X[sp.train])
            Ztr, Zte = sc.transform(X[sp.train]), sc.transform(X[sp.test])
            if n_pcs:
                k = int(max(1, min(n_pcs, len(sp.train) - 1, X.shape[1])))
                pca = PCA(n_components=k, random_state=seed).fit(Ztr)
                Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
            out[sp.name] = (Ztr.astype(np.float32), Zte.astype(np.float32))
    return out


def truncate_projections(projections, k):
    """The first k components of every split's projection (PCA / SVD components are ordered)."""
    return {name: (ztr[:, :k], zte[:, :k]) for name, (ztr, zte) in projections.items()}


@dataclass
class OOF:
    pred: np.ndarray      # (N,) float32: class code | 0/1 | value; NaN where not tested
    score: np.ndarray     # (N,) float32: P(class 1) for binary, else NaN
    proba: np.ndarray     # (N, K) float32 for multiclass when keep_proba else None
    tested: np.ndarray    # (N,) bool
    per_split: list       # [{"name", "n_train", "n_test", metrics...}]
    target: np.ndarray = None   # (N,) float: the evaluated target (y, or the demeaned y of variant 'within')
    n_classes: int = None       # classes present in the training parts (multiclass)


def fit_eval(X, y, valid, splits, task, behavior=None, variant="raw", n_pcs=N_PCS, seed=42, keep_proba=False,
             projections=None):
    """Out-of-fold predictions over `splits` (train / test masked by `valid`). variant 'within': y_train is demeaned per
    behavior with train means, the test target with the same train means (behavior from the metadata). Multiclass
    classes = those present in the training part (8 in the method split). X: array (N, d) or list of N docs (TF-IDF);
    `projections` (project_splits output) can be passed to share the unsupervised feature map between tasks."""
    from sklearn.exceptions import ConvergenceWarning
    kind = TASKS[task][1]
    y = np.asarray(y)
    valid = np.asarray(valid, dtype=bool)
    N = len(y)
    if projections is None:
        projections = project_splits(X, splits, n_pcs, seed)
    pred = np.full(N, np.nan, dtype=np.float32)
    score = np.full(N, np.nan, dtype=np.float32)
    y_f = y.astype(np.float64)                 # immutable source of the targets
    target = y_f.copy()                        # evaluated target (demeaned test rows for variant 'within')
    tested = np.zeros(N, dtype=bool)
    K = int(y[valid].max()) + 1 if kind == "multiclass" else 0
    proba = np.full((N, K), np.nan, dtype=np.float32) if (kind == "multiclass" and keep_proba) else None
    if variant == "within" and behavior is None:
        raise ValueError("variant 'within' needs the behaviour labels")
    behavior = None if behavior is None else np.asarray(behavior)
    per_split, n_classes = [], []
    for sp in splits:
        Ztr, Zte = projections[sp.name]
        mtr, mte = valid[sp.train], valid[sp.test]
        tr, te = sp.train[mtr], sp.test[mte]
        Ztr_v, Zte_v = Ztr[mtr], Zte[mte]
        ytr, yte = y_f[tr].copy(), y_f[te].copy()
        if variant == "within":
            means = {b: ytr[behavior[tr] == b].mean() for b in np.unique(behavior[tr])}
            glob = ytr.mean()
            ytr = ytr - np.array([means[b] for b in behavior[tr]])
            yte = yte - np.array([means.get(b, glob) for b in behavior[te]])
        if kind != "ridge" and len(np.unique(ytr)) < 2:
            per_split.append({"name": sp.name, "n_train": int(len(tr)), "n_test": 0, "skipped": "single class in training part"})
            continue
        head = head_estimator(kind, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            head.fit(Ztr_v, ytr.astype(np.int64) if kind != "ridge" else ytr)
        if kind == "multiclass":
            p = head.predict_proba(Zte_v)
            cls = head.classes_.astype(int)
            pred[te] = cls[p.argmax(1)]
            if proba is not None:
                proba[te] = 0.0
                proba[np.ix_(te, cls)] = p
            n_classes.append(len(cls))
            m = metrics(kind, target[te], pred[te], None, _restrict_proba(p, cls, K))
        elif kind == "binary":
            s = head.predict_proba(Zte_v)[:, 1]
            score[te] = s
            pred[te] = (s >= 0.5).astype(np.float32)
            m = metrics(kind, target[te], pred[te], score[te])
        else:
            pred[te] = head.predict(Zte_v)
            target[te] = yte
            m = metrics(kind, target[te], pred[te], None)
        tested[te] = True
        per_split.append({"name": sp.name, "n_train": int(len(tr)), "n_test": int(len(te)), **m})
    return OOF(pred=pred, score=score, proba=proba, tested=tested, per_split=per_split, target=target,
               n_classes=int(max(n_classes)) if n_classes else None)


def _restrict_proba(p, cls, K):
    """(n, K) probabilities with the training classes `cls` in their global columns and 0 elsewhere."""
    out = np.zeros((p.shape[0], K), dtype=np.float64)
    out[:, cls] = p
    return out


def _bal_acc(y, pred):
    """Balanced accuracy = mean recall over the classes present in y (as sklearn)."""
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    K = int(max(y.max(), pred.max())) + 1
    conf = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    rows = conf.sum(1)
    present = rows > 0
    return float((np.diag(conf)[present] / rows[present]).mean())


def _auc(y, score):
    """ROC-AUC via the Mann-Whitney rank formula (NaN with a single class)."""
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(score, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _r2(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def primary_metric(kind, y, pred, score):
    """The primary metric of F17 on already-filtered (tested) rows."""
    if kind == "multiclass":
        return _bal_acc(y, pred)
    if kind == "binary":
        return _auc(y, score)
    return _r2(y, pred)


def metrics(kind, y, oof_pred, oof_score, proba=None, tested=None):
    """multiclass: bal_acc, acc, auc_macro (OvR over the classes present in the tested rows; NaN without proba);
    binary: auc, bal_acc, acc (threshold 0.5 on score); ridge: r2, spearman. Only tested rows."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    if tested is not None:
        tested = np.asarray(tested, dtype=bool)
        y, oof_pred = np.asarray(y)[tested], np.asarray(oof_pred)[tested]
        oof_score = None if oof_score is None else np.asarray(oof_score)[tested]
        proba = None if proba is None else np.asarray(proba)[tested]
    y = np.asarray(y)
    if len(y) == 0:
        return {m: float("nan") for m in {"multiclass": ("bal_acc", "acc", "auc_macro"), "binary": ("auc", "bal_acc", "acc"),
                                          "ridge": ("r2", "spearman")}[kind]}
    if kind == "multiclass":
        yi, pi = y.astype(np.int64), np.asarray(oof_pred).astype(np.int64)
        out = {"bal_acc": _bal_acc(yi, pi), "acc": float((yi == pi).mean()), "auc_macro": float("nan")}
        if proba is not None:
            present = np.unique(yi)
            if len(present) >= 2:
                sub = np.asarray(proba, dtype=np.float64)[:, present]
                sub = sub / np.maximum(sub.sum(1, keepdims=True), 1e-12)
                try:
                    out["auc_macro"] = float(roc_auc_score(yi, sub, multi_class="ovr", average="macro", labels=present))
                except ValueError:
                    pass
        return out
    if kind == "binary":
        yi, pi = (y > 0.5).astype(np.int64), (np.asarray(oof_pred) > 0.5).astype(np.int64)
        return {"auc": _auc(yi, oof_score), "bal_acc": _bal_acc(yi, pi), "acc": float((yi == pi).mean())}
    rho = spearmanr(y, oof_pred).correlation if len(y) > 2 else float("nan")
    return {"r2": _r2(y, oof_pred), "spearman": float(rho) if rho == rho else float("nan")}


def bootstrap_ci(kind, y, pred, score, B=BOOT, seed=0, alpha=0.05):
    """Percentile CI of the primary metric over resampled tested rows (F18). Returns (point, lo, hi)."""
    y, pred = np.asarray(y), np.asarray(pred)
    score = None if score is None else np.asarray(score)
    n = len(y)
    point = primary_metric(kind, y, pred, score)
    if n == 0 or B <= 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = primary_metric(kind, y[idx], pred[idx], None if score is None else score[idx])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_bootstrap(kind, y, pred_a, score_a, pred_b, score_b, B=BOOT, seed=0):
    """{'diff', 'lo', 'hi', 'p_boot', 'n'} of metric(a) - metric(b) over the SAME tested rows (intersection); F18."""
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a, dtype=np.float64), np.asarray(pred_b, dtype=np.float64)
    score_a = None if score_a is None else np.asarray(score_a, dtype=np.float64)
    score_b = None if score_b is None else np.asarray(score_b, dtype=np.float64)
    ok = ~np.isnan(pred_a) & ~np.isnan(pred_b)
    if score_a is not None:
        ok &= ~np.isnan(score_a)
    if score_b is not None:
        ok &= ~np.isnan(score_b)
    if kind == "ridge":
        ok &= ~np.isnan(np.asarray(y, dtype=np.float64))
    idx_all = np.flatnonzero(ok)
    n = len(idx_all)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_boot": float("nan"), "n": 0}
    sa = None if score_a is None else score_a[idx_all]
    sb = None if score_b is None else score_b[idx_all]
    ya, pa, pb = y[idx_all], pred_a[idx_all], pred_b[idx_all]
    diff = primary_metric(kind, ya, pa, sa) - primary_metric(kind, ya, pb, sb)
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = primary_metric(kind, ya[i], pa[i], None if sa is None else sa[i]) - primary_metric(kind, ya[i], pb[i], None if sb is None else sb[i])
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {"diff": float(diff), "lo": float("nan"), "hi": float("nan"), "p_boot": float("nan"), "n": int(n)}
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"diff": float(diff), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_boot": float(min(1.0, p)), "n": int(n)}


# ----------------------------------------------------------------------------- baselines (spec 5.8)

def style_features(rows):
    """(N, 10) float64 in STYLE_NAMES order, computed on the sampled rows only (identical rows as the activations):
    n_rows_used, mean prompt words, mean / sd / median response words, '!' and '?' per 1000 response chars, upper-case
    share of response chars, newlines per row, mean type-token ratio of responses. No text leaves the function."""
    N = len(rows.dataset_ids)
    acc = [[] for _ in range(N)]
    for i, p, r in rows.rows:
        pw = p.split()
        rw = r.split()
        chars = max(1, len(r))
        letters = [c for c in r if c.isalpha()]
        acc[i].append((len(pw), len(rw), 1000.0 * r.count("!") / chars, 1000.0 * r.count("?") / chars,
                       (sum(1 for c in letters if c.isupper()) / max(1, len(letters))), (p + r).count("\n"),
                       (len(set(w.lower() for w in rw)) / max(1, len(rw)))))
    X = np.zeros((N, len(STYLE_NAMES)), dtype=np.float64)
    for i in range(N):
        a = np.array(acc[i], dtype=np.float64) if acc[i] else np.zeros((1, 7))
        X[i] = [len(acc[i]), a[:, 0].mean(), a[:, 1].mean(), a[:, 1].std(), np.median(a[:, 1]), a[:, 2].mean(), a[:, 3].mean(),
                a[:, 4].mean(), a[:, 5].mean(), a[:, 6].mean()]
    return X


def metadata_features(scores, behaviors):
    """(X (N, m), names): METADATA_COLUMNS (NaN -> column median) + one-hot(behavior_id over `behaviors`) + one-hot(method
    over METHODS). Auditor columns (oe_surprisingness, mc_surprisingness, mc_missed_*) are deliberately excluded: they
    measure the same auditor outcome. Constant columns inside a method-split training part are harmless (StandardScaler
    maps them to 0)."""
    cols = []
    names = []
    for c in METADATA_COLUMNS:
        v = scores[c].to_numpy(dtype=float)
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        cols.append(np.where(np.isnan(v), med, v))
        names.append(c)
    for b in behaviors:
        cols.append((scores["behavior_id"].to_numpy() == b).astype(float))
        names.append(f"behavior={b}")
    for m in METHODS:
        cols.append((scores["method"].to_numpy() == m).astype(float))
        names.append(f"method={m}")
    return np.stack(cols, axis=1), names


# ----------------------------------------------------------------------------- angles (spec 5.9)

def standardize(X):
    """(X - mean) / std per column, std floor 1e-6; returns float64."""
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(0)
    return (X - X.mean(0)) / np.maximum(sd, 1e-6)


def mass_mean_directions(Xs, y, classes):
    """(K, d): mean(Xs[y == k]) - mean(Xs[y != k]) (standardized space; the repo's DoM analogue, directions.py)."""
    Xs, y = np.asarray(Xs, dtype=np.float64), np.asarray(y)
    K = len(classes)
    onehot = np.stack([(y == k).astype(np.float64) for k in range(K)])           # (K, N)
    counts = onehot.sum(1)
    sums = onehot @ Xs                                                            # (K, d)
    total = Xs.sum(0)
    rest = (total[None, :] - sums) / np.maximum(len(y) - counts, 1)[:, None]
    return sums / np.maximum(counts, 1)[:, None] - rest


def ovr_directions(Xs, y, classes, seed=42, C=1.0, max_iter=500, n_jobs=1):
    """(K, d): coef_ of one balanced binary LogisticRegression per class on the full standardized features."""
    from joblib import Parallel, delayed
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    def one(k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf = LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", solver="lbfgs")
            clf.fit(Xs, (np.asarray(y) == k).astype(int))
        return clf.coef_[0]

    return np.stack(Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(len(classes))))


def angle_matrix(dirs):
    """(K, K) float64 of pairwise angles via analysis.direction_angles.angle_deg, diagonal 0."""
    from .analysis.direction_angles import angle_deg
    dirs = np.asarray(dirs, dtype=np.float64)
    K = dirs.shape[0]
    A = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            A[i, j] = A[j, i] = angle_deg(dirs[i], dirs[j])
    return A


def _angles_fast(dirs):
    """Vectorized angle matrix (for the permutation null)."""
    d = np.asarray(dirs, dtype=np.float64)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    c = (d / np.maximum(n, 1e-12)) @ (d / np.maximum(n, 1e-12)).T
    A = np.degrees(np.arccos(np.clip(c, -1, 1)))
    np.fill_diagonal(A, 0.0)
    return A


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def angle_permutation_null(Xs, y, classes, perms=PERMS, seed=0, observed=None):
    """Label permutations (class sizes kept) of the mass-mean angle matrix -> dict(mean, q025, q975, p_two, p_holm, flag)
    each (K, K); p_two = (c + 1) / (P + 1) with c = #perms whose angle is at least as extreme on the observed side,
    times 2, capped at 1. Holm over the K(K-1)/2 pairs -> flag -1 'aligned' (observed < null, Holm p < 0.05) / +1
    'opposed' (observed > null) / 0."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    K = len(classes)
    obs = _angles_fast(mass_mean_directions(Xs, y, classes)) if observed is None else np.asarray(observed)
    null = np.empty((perms, K, K))
    for p in range(perms):
        null[p] = _angles_fast(mass_mean_directions(Xs, rng.permutation(y), classes))
    from scipy.stats import norm
    mean = null.mean(0)
    sd = null.std(0, ddof=1) if perms > 1 else np.full((K, K), np.nan)
    q025, q975 = np.percentile(null, 2.5, axis=0), np.percentile(null, 97.5, axis=0)
    below = obs < mean
    c = np.where(below, (null <= obs[None]).sum(0), (null >= obs[None]).sum(0))
    p_two = np.minimum(1.0, 2.0 * (c + 1) / (perms + 1))
    # The smallest empirical two-sided p is 2 / (P + 1) (0.01 at P = 200), which can never pass Holm over 153 pairs; the
    # flags therefore use the normal approximation of the same permutation null (z = (obs - mean) / sd), the empirical
    # p_two is reported alongside (DEVIATIONS).
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (obs - mean) / np.where(sd > 0, sd, np.nan)
    p_norm = np.where(np.isnan(z), 1.0, 2.0 * norm.sf(np.abs(np.nan_to_num(z))))
    iu = np.triu_indices(K, 1)
    p_h = np.ones((K, K))
    flag = np.zeros((K, K), dtype=np.int8)
    if len(iu[0]):
        adj = holm(p_norm[iu])
        p_h[iu] = adj
        p_h[(iu[1], iu[0])] = adj
        sig = adj < 0.05
        f = np.where(sig, np.where(below[iu], -1, 1), 0).astype(np.int8)
        flag[iu] = f
        flag[(iu[1], iu[0])] = f
    return {"mean": mean, "sd": sd, "q025": q025, "q975": q975, "p_two": p_two, "p_norm": p_norm, "z": z, "p_holm": p_h,
            "flag": flag, "perms": int(perms), "mean_angle_null": [float(null[p][iu].mean()) for p in range(perms)] if len(iu[0]) else []}


def n1_reference(d):
    """{"q025", "q500", "q975"} = analysis.meanfield.n1_angle_quantile([0.025, 0.5, 0.975], d)."""
    from .analysis.meanfield import n1_angle_quantile
    q = n1_angle_quantile([0.025, 0.5, 0.975], int(d))
    return {"q025": float(q[0]), "q500": float(q[1]), "q975": float(q[2])}


def effective_rank(dirs):
    """Participation ratio (sum s^2)^2 / sum s^4 of the singular values of the unit-normalised (K, d) direction rows
    (each row centred over its coordinates); 1 for one shared direction, K for K orthogonal ones."""
    d = np.asarray(dirs, dtype=np.float64)
    d = d - d.mean(1, keepdims=True)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    s = np.linalg.svd(d, compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-300))


def linkage_merges(angles, classes):
    """[(height, members_left, members_right)] of scipy average linkage on squareform(angles) (super-clusters, as in the
    lexical analysis). Report shows the merge order."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    A = np.asarray(angles, dtype=np.float64)
    K = A.shape[0]
    if K < 2:
        return []
    Z = linkage(squareform(A, checks=False), method="average")
    members = {i: [classes[i]] for i in range(K)}
    out = []
    for step, (a, b, h, _) in enumerate(Z):
        left, right = members[int(a)], members[int(b)]
        out.append((float(h), list(left), list(right)))
        members[K + step] = left + right
    return out


def fold_angle_matrices(X, y, classes, splits):
    """(F, K, K): mass-mean angle matrix per split, directions fitted on the TRAINING rows of that split only (features
    standardized on those rows). A held-out check of the in-sample matrix of angle_analysis: the mass-mean directions are
    cheap (K class means), so every fold is refitted; the OvR directions are not (18 full-d logistic fits per fold)."""
    X, y = np.asarray(X), np.asarray(y)
    out = []
    for sp in splits:
        tr = np.asarray(sp.train)
        out.append(_angles_fast(mass_mean_directions(standardize(X[tr]), y[tr], classes)))
    return np.stack(out) if out else np.zeros((0, len(classes), len(classes)))


def angle_analysis(X, y, classes, d=None, perms=PERMS, seed=0, ref_matrices=None, n_jobs=1, splits=None):
    """Everything above for one (layer, pooling): returns dict with 'massmean' (K, K), 'ovr' (K, K), null dicts, n1,
    effective ranks, per-behavior nearest (3 smallest angles) and farthest (3 largest), merges, and Spearman of the upper
    triangle against each matrix in ref_matrices ({'step0': ..., 'tfidf': ...}).
    The directions are fitted IN-SAMPLE on every row (full d, at a layer the probe selected on the same datasets); the
    mass-mean matrix carries the label-permutation null, the OvR matrix carries none (descriptive). With `splits` (the
    random folds) the mass-mean directions are refitted on each training part (`fold_angle_matrices`) and the result also
    holds 'massmean_folds' (F, K, K), 'fold_spearman' (per fold, upper triangle vs the in-sample matrix), 'fold_pair_sd_mean'
    (mean over pairs of the fold-to-fold SD in degrees) and 'fold_mean_angle'."""
    from scipy.stats import spearmanr
    Xs = standardize(X)
    d = Xs.shape[1] if d is None else int(d)
    y = np.asarray(y)
    K = len(classes)
    dirs_mm = mass_mean_directions(Xs, y, classes)
    dirs_ovr = ovr_directions(Xs, y, classes, seed=seed, n_jobs=n_jobs)
    A_mm, A_ovr = angle_matrix(dirs_mm), angle_matrix(dirs_ovr)
    null = angle_permutation_null(Xs, y, classes, perms=perms, seed=seed, observed=A_mm)
    iu = np.triu_indices(K, 1)
    folds = fold_angle_matrices(X, y, classes, splits or [])
    fold_info = {"n_folds": int(len(folds)), "massmean_folds": folds if len(folds) else None}
    if len(folds) and len(iu[0]):
        sd = folds.std(0, ddof=1) if len(folds) > 1 else np.full((K, K), np.nan)
        fold_info.update(fold_spearman=[float(spearmanr(folds[f][iu], A_mm[iu]).correlation) if len(iu[0]) > 2 else float("nan")
                                        for f in range(len(folds))],
                         fold_pair_sd_mean=float(np.nanmean(sd[iu])), fold_mean_angle=float(folds.mean(0)[iu].mean()))
    else:
        fold_info.update(fold_spearman=[], fold_pair_sd_mean=float("nan"), fold_mean_angle=float("nan"))
    near, far = {}, {}
    for i, c in enumerate(classes):
        order = np.argsort(A_mm[i])
        others = [j for j in order if j != i]
        near[c] = [(classes[j], float(A_mm[i, j])) for j in others[:3]]
        far[c] = [(classes[j], float(A_mm[i, j])) for j in others[::-1][:3]]
    spear = {}
    for name, ref in (ref_matrices or {}).items():
        ref = np.asarray(ref)
        if ref.shape == A_mm.shape and len(iu[0]) > 2:
            spear[name] = float(spearmanr(A_mm[iu], ref[iu]).correlation)
    return {"classes": list(classes), "d": d, "massmean": A_mm, "ovr": A_ovr, "dirs_massmean": dirs_mm.astype(np.float32),
            "null": null, "n1": n1_reference(d), "effective_rank_massmean": effective_rank(dirs_mm),
            "effective_rank_ovr": effective_rank(dirs_ovr), "mean_angle_massmean": float(A_mm[iu].mean()) if len(iu[0]) else float("nan"),
            "mean_angle_ovr": float(A_ovr[iu].mean()) if len(iu[0]) else float("nan"),
            "n_aligned": int((null["flag"][iu] == -1).sum()), "n_opposed": int((null["flag"][iu] == 1).sum()),
            "n_pairs": int(len(iu[0])), "nearest": near, "farthest": far, "merges": linkage_merges(A_mm, classes),
            "spearman_ref": spear, "perms": int(perms), **fold_info}


# ----------------------------------------------------------------------------- rank test (spec 5.10)

def rank_test(X, y, valid, splits, ks=RANK_KS, seed=42, boot=BOOT):
    """DataFrame(k, bal_acc, ci_lo, ci_hi, n_test): fit_eval with n_pcs=k per k (k <= min(n_train - 1, d)); attrs k95 =
    smallest k with bal_acc >= 0.95 * bal_acc(max k). Random split only (F22). X may be docs (TF-IDF, k <= 128)."""
    valid = np.asarray(valid, dtype=bool)
    is_docs = isinstance(X, (list, tuple))
    n_train_min = min(len(sp.train) for sp in splits)
    kmax = min(max(ks), n_train_min - 1, (TFIDF_SVD if is_docs else np.asarray(X).shape[1]))
    proj = project_splits(X, splits, int(kmax), seed)
    kmax = min(int(kmax), min(z[0].shape[1] for z in proj.values()))
    rows = []
    for k in [k for k in ks if k <= kmax]:
        res = fit_eval(X, y, valid, splits, "behavior", n_pcs=k, seed=seed, projections=truncate_projections(proj, k))
        t = res.tested
        point, lo, hi = bootstrap_ci("multiclass", res.target[t], res.pred[t], None, B=boot, seed=seed)
        rows.append({"k": int(k), "bal_acc": point, "ci_lo": lo, "ci_hi": hi, "n_test": int(t.sum())})
    df = pd.DataFrame(rows)
    if len(df):
        top = df["bal_acc"].iloc[-1]
        ok = df[df["bal_acc"] >= 0.95 * top]
        df.attrs["k95"] = int(ok["k"].iloc[0]) if len(ok) else int(df["k"].iloc[-1])
        df.attrs["k_max"] = int(df["k"].iloc[-1])
    return df


# ----------------------------------------------------------------------------- run functions (spec 5.11)

def probe_settings(n_pcs=N_PCS, folds=FOLDS, boot=BOOT, perms=PERMS, seed=42, tasks=None, n_jobs=None):
    """Settings dict of a probe run; n_jobs is not part of the fingerprint."""
    return {"version": VERSION, "n_pcs": int(n_pcs), "folds": int(folds), "boot": int(boot), "perms": int(perms),
            "seed": int(seed), "tasks": list(tasks or TASKS), "n_jobs": n_jobs}


EXTRACTION_IDENTITY = ("version", "n_rows", "row_seed", "max_length", "prompt_cap", "layers", "hf_id", "commit_sha", "revision", "finished",
                       "weight_fingerprint")


def _fingerprint(settings, extra):
    s = {k: v for k, v in settings.items() if k != "n_jobs"}
    return hashlib.sha256(json.dumps({"settings": s, **extra}, sort_keys=True, default=str).encode()).hexdigest()


def best_feature_set(df, task, split="random"):
    """(layer, pooling) with the highest primary metric (rows features == 'act', direction == 'pooled') -> F19."""
    kind = TASKS[task][1]
    sub = df[(df["features"] == "act") & (df["task"] == task) & (df["variant"] == "raw") & (df["split"] == split)
             & (df["direction"] == "pooled") & (df["metric"] == PRIMARY[kind])]
    if not len(sub) or sub["value"].isna().all():
        return None
    r = sub.loc[sub["value"].idxmax()]
    return int(r["layer"]), str(r["pooling"])


def _metric_rows(base, kind, oof, boot, seed, splits_name):
    """Long-format rows (pooled with CI on the primary metric + per-split points) for one OOF result."""
    rows = []
    t = oof.tested
    primary = PRIMARY[kind]
    pooled = metrics(kind, oof.target, oof.pred, oof.score, oof.proba, t)
    n_train = int(np.mean([s["n_train"] for s in oof.per_split])) if oof.per_split else 0
    point, lo, hi = bootstrap_ci(kind, oof.target[t], oof.pred[t], None if kind != "binary" else oof.score[t], B=boot, seed=seed)
    for m, v in pooled.items():
        rows.append({**base, "split": splits_name, "direction": "pooled", "n_train": n_train, "n_test": int(t.sum()),
                     "metric": m, "value": float(v), "ci_lo": lo if m == primary else np.nan, "ci_hi": hi if m == primary else np.nan})
    for s in oof.per_split:
        for m in pooled:
            rows.append({**base, "split": splits_name, "direction": s["name"], "n_train": s["n_train"], "n_test": s["n_test"],
                         "metric": m, "value": float(s.get(m, np.nan)), "ci_lo": np.nan, "ci_hi": np.nan})
    return rows


def _probe_feature_set(X, features_name, layer, pooling, targets, behavior, splits_r, splits_m, settings, tag):
    """Every task x variant x split on one feature set (array or docs). Returns (rows, oof dict, proba dict, classes)."""
    n_pcs, seed, boot = settings["n_pcs"], settings["seed"], settings["boot"]
    proj = {"random": project_splits(X, splits_r, n_pcs, seed), "method": project_splits(X, splits_m, n_pcs, seed)}
    rows, oof_out, proba_out = [], {}, {}
    for task in settings["tasks"]:
        kind = TASKS[task][1]
        if features_name == "metadata" and task != "miss_sol":
            continue
        y, valid = targets[task]
        for variant in task_variants(task):
            for split_name, splits in (("random", splits_r), ("method", splits_m)):
                if not splits:
                    continue
                res = fit_eval(X, y, valid, splits, task, behavior=behavior, variant=variant, n_pcs=n_pcs, seed=seed,
                               keep_proba=(kind == "multiclass"), projections=proj[split_name])
                n_classes = res.n_classes if kind == "multiclass" else (2 if kind == "binary" else np.nan)
                chance = (1.0 / n_classes if kind == "multiclass" and n_classes else (0.5 if kind == "binary" else 0.0))
                base = {**tag, "features": features_name, "layer": int(layer), "pooling": pooling, "task": task, "variant": variant,
                        "n_pcs": int(min(n_pcs, min(z[0].shape[1] for z in proj[split_name].values()))), "n_classes": n_classes, "chance": chance}
                rows += _metric_rows(base, kind, res, boot, seed, split_name)
                key = f"{features_name}__{task}__{variant}__{split_name}__L{int(layer):02d}__{pooling}"
                oof_out[f"pred__{key}"] = res.pred
                if kind == "binary":
                    oof_out[f"score__{key}"] = res.score
                if variant == "within":
                    oof_out[f"target__{task}__within__{split_name}"] = res.target.astype(np.float32)
                if res.proba is not None:
                    proba_out[f"proba__{key}"] = res.proba
    return rows, oof_out, proba_out


def _write_angles(base, ana):
    """_angles.npz / .csv / .json of one angle analysis."""
    K = len(ana["classes"])
    null = ana["null"]
    np.savez(base + "_angles.npz", behaviors=np.array(ana["classes"]), layer=np.array(ana.get("layer", -1)),
             pooling=np.array(ana.get("pooling", "none")), massmean=ana["massmean"], ovr=ana["ovr"], null_mean=null["mean"],
             null_q025=null["q025"], null_q975=null["q975"], p_two=null["p_two"], p_holm=null["p_holm"], holm_flag=null["flag"],
             n1_q=np.array([ana["n1"]["q025"], ana["n1"]["q500"], ana["n1"]["q975"]]), d=np.array(ana["d"]),
             dirs_massmean=ana["dirs_massmean"],
             massmean_folds=(ana["massmean_folds"] if ana.get("massmean_folds") is not None else np.zeros((0, K, K))))
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            rows.append({"pair": f"{ana['classes'][i]}|{ana['classes'][j]}", "a": ana["classes"][i], "b": ana["classes"][j],
                         "angle_massmean": ana["massmean"][i, j], "angle_ovr": ana["ovr"][i, j], "null_mean": null["mean"][i, j],
                         "null_q025": null["q025"][i, j], "null_q975": null["q975"][i, j], "p_two": null["p_two"][i, j],
                         "p_holm": null["p_holm"][i, j], "flag": {-1: "aligned", 0: "", 1: "opposed"}[int(null["flag"][i, j])]})
    pd.DataFrame(rows).to_csv(base + "_angles.csv", index=False)
    summary = {k: ana[k] for k in ("classes", "d", "layer", "pooling", "effective_rank_massmean", "effective_rank_ovr",
                                   "mean_angle_massmean", "mean_angle_ovr", "n_aligned", "n_opposed", "n_pairs", "nearest",
                                   "farthest", "merges", "spearman_ref", "perms", "n_folds", "fold_spearman", "fold_pair_sd_mean",
                                   "fold_mean_angle") if k in ana}
    summary["directions_fit"] = "in-sample (all datasets, probe-selected layer); OvR without null"
    summary["n1"] = ana["n1"]
    summary["null_mean_angle"] = float(np.mean(null["mean_angle_null"])) if null["mean_angle_null"] else float("nan")
    summary["null_mean_angle_q"] = ([float(np.percentile(null["mean_angle_null"], 2.5)), float(np.percentile(null["mean_angle_null"], 97.5))]
                                    if null["mean_angle_null"] else [float("nan")] * 2)
    summary["aligned_pairs"] = [r["pair"] for r in rows if r["flag"] == "aligned"]
    summary["opposed_pairs"] = [r["pair"] for r in rows if r["flag"] == "opposed"]
    with open(base + "_angles.json", "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _load_angle_matrix(base):
    path = base + "_angles.npz"
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {"massmean": z["massmean"], "behaviors": z["behaviors"].astype(str).tolist()}


def _targets_and_splits(scores, settings):
    """Targets per task, behaviour codes, random and method splits of a scores table."""
    behavior, method = scores["behavior_id"].to_numpy(), scores["method"].to_numpy()
    targets = {task: task_target(scores, task) for task in settings["tasks"]}
    splits_r = random_splits(behavior, settings["folds"], settings["seed"])
    try:
        splits_m = method_splits(behavior, method)
    except ValueError:
        splits_m = []
    return targets, behavior, splits_r, splits_m


def _csv_tag(model, revision, settings, n_rows, row_seed):
    return {"model": model, "revision": revision, "seed": settings["seed"], "n_rows": int(n_rows), "row_seed": int(row_seed),
            "version": VERSION}


def _oof_base_arrays(scores, targets, splits_r, splits_m):
    out = {"ids": np.array([str(i) for i in scores.index])}
    for task, (y, valid) in targets.items():
        out[f"y__{task}"] = np.asarray(y, dtype=np.float32)
        out[f"valid__{task}"] = np.asarray(valid)
    out["method_tested"] = np.zeros(len(scores), dtype=bool)
    for sp in splits_m:
        out["method_tested"][sp.test] = True
    return out


def probe_feature_file(npz_path, raw_dir, out_dir, settings, n_jobs=-1, force=False, say=print):
    """Runs every task x (layer, pooling) x split on one feature file; writes
        {out_dir}/{model}_{revision}_probe.csv, _probe.json (settings fingerprint + manifest + best sets + timings),
        {out_dir}/{model}_{revision}_oof.npz, _angles.npz, _angles.csv, _angles.json, _rank.csv
    Skips when _probe.json exists with the same fingerprint (version, settings, scores sha256 and the feature file's
    extraction identity EXTRACTION_IDENTITY from its embedded manifest: rows_sha256, max_length, prompt_cap, layers,
    hf_id, commit_sha, revision, finished) unless force, so a re-extracted file (--force, other settings or checkpoint
    commit) is always re-probed. joblib.Parallel(n_jobs, prefer='processes') over the (layer, pooling) sets; each job
    loads its own key."""
    from joblib import Parallel, delayed
    meta = load_feature_meta(npz_path)
    man = meta["manifest"]
    model, revision = man.get("model") or os.path.basename(npz_path)[:-4].rsplit("_", 1)[0], man.get("revision") or "main"
    base = os.path.join(out_dir, f"{model}_{revision}")
    scores_sha = file_sha256(raw_file(raw_dir, "scores.csv"))
    # the weight fingerprint enters the identity only when the manifest carries one, so the probe fingerprints of files
    # extracted before the weight-source check stay unchanged (no needless re-probing of valid files)
    fp = _fingerprint(settings, {"rows_sha256": man.get("rows_sha256"), "scores_sha256": scores_sha, "npz": os.path.basename(npz_path),
                                 "extraction": {k: man.get(k) for k in EXTRACTION_IDENTITY if k != "weight_fingerprint" or man.get(k)}})
    if os.path.exists(base + "_probe.json") and not force:
        with open(base + "_probe.json") as f:
            old = json.load(f)
        if old.get("fingerprint") == fp:
            say(f"[dsse] skip {base}_probe.csv: exists for the same features, settings and scores")
            return old
    os.makedirs(out_dir, exist_ok=True)
    scores = load_scores(raw_dir)
    if list(scores.index) != list(meta["dataset_ids"]):
        raise ValueError(f"{npz_path}: dataset ids differ from scores.csv (n {len(meta['dataset_ids'])} vs {len(scores)})")
    targets, behavior, splits_r, splits_m = _targets_and_splits(scores, settings)
    tag = _csv_tag(model, revision, settings, man.get("n_rows", N_ROWS), man.get("row_seed", ROW_SEED))
    sets = [(l, pool) for l in meta["layers"] for pool in POOLINGS]
    t0 = time.time()

    def job(layer, pooling):
        X = load_feature_layer(npz_path, layer, pooling)
        return _probe_feature_set(X, "act", layer, pooling, targets, behavior, splits_r, splits_m, settings, tag)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(job)(l, pool) for l, pool in sets)
    rows, oof, proba = [], _oof_base_arrays(scores, targets, splits_r, splits_m), {}
    for r, o, p in results:
        rows += r
        oof.update(o)
        proba.update(p)
    df = pd.DataFrame(rows)
    t_probe = time.time() - t0
    best = {task: best_feature_set(df, task) for task in settings["tasks"]}
    for task, bs in best.items():
        if bs is None:
            continue
        for split_name in ("random", "method"):
            key = f"proba__act__{task}__raw__{split_name}__L{bs[0]:02d}__{bs[1]}"
            if key in proba:
                oof[key] = proba[key]
    df.to_csv(base + "_probe.csv", index=False)
    np.savez(base + "_oof.npz", **oof)

    # angles + rank test at the best (layer, pooling) of the behaviour task
    t1 = time.time()
    angles_summary, rank_info = None, None
    bs = best.get("behavior")
    if bs is not None:
        X = load_feature_layer(npz_path, bs[0], bs[1])
        y, valid = targets["behavior"]
        classes = behavior_classes(scores)
        refs = {}
        step0 = _load_angle_matrix(os.path.join(out_dir, f"{model}_step0"))
        if step0 is not None and step0["behaviors"] == classes and revision != "step0":
            refs["step0"] = step0["massmean"]
        tfidf = _load_angle_matrix(os.path.join(out_dir, "baselines"))
        if tfidf is not None and tfidf["behaviors"] == classes:
            refs["tfidf"] = tfidf["massmean"]
        ana = angle_analysis(X, y, classes, d=X.shape[1], perms=settings["perms"], seed=settings["seed"], ref_matrices=refs,
                             n_jobs=n_jobs, splits=splits_r)
        ana["layer"], ana["pooling"] = int(bs[0]), bs[1]
        angles_summary = _write_angles(base, ana)
        rk = rank_test(X, y, valid, splits_r, ks=RANK_KS, seed=settings["seed"], boot=settings["boot"])
        rk.to_csv(base + "_rank.csv", index=False)
        rank_info = {"k95": rk.attrs.get("k95"), "k_max": rk.attrs.get("k_max"), "layer": int(bs[0]), "pooling": bs[1]}
    info = {"version": VERSION, "model": model, "revision": revision, "fingerprint": fp, "settings": {k: v for k, v in settings.items() if k != "n_jobs"},
            "manifest": man, "scores_sha256": scores_sha, "npz": os.path.basename(npz_path), "best": {k: (list(v) if v else None) for k, v in best.items()},
            "n_feature_sets": len(sets), "seconds_probe": round(t_probe, 1), "seconds_angles_rank": round(time.time() - t1, 1),
            "angles": angles_summary, "rank": rank_info, "n_datasets": int(len(scores)),
            "n_method_split": int(sum(len(sp.test) for sp in splits_m)), "shared_behaviors": shared_behaviors(behavior, scores["method"].to_numpy()),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(base + "_probe.json", "w") as f:
        json.dump(info, f, indent=1, default=_json_default)
    for task, bs in best.items():
        if bs is None:
            continue
        kind = TASKS[task][1]
        for split_name in ("random", "method"):
            sub = df[(df.features == "act") & (df.task == task) & (df.variant == "raw") & (df.split == split_name) & (df.direction == "pooled")
                     & (df.metric == PRIMARY[kind]) & (df.layer == bs[0]) & (df.pooling == bs[1])]
            if len(sub):
                r = sub.iloc[0]
                say(f"[dsse] {model}_{revision} {task:12s} best L{bs[0]:02d}/{bs[1]:9s} {split_name:6s} {PRIMARY[kind]} = {r.value:.3f} "
                    f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] (chance {r.chance:.3f}, n_test {int(r.n_test)})")
    say(f"[dsse] {model}_{revision}: {len(sets)} feature sets, probes {t_probe:.0f}s, angles + rank {time.time() - t1:.0f}s")
    return info


def run_baselines(raw_dir, out_dir, n_rows=N_ROWS, seed_rows=ROW_SEED, settings=None, force=False, say=print, n_jobs=1):
    """TF-IDF / style / metadata on the same sampled rows -> baselines_probe.csv, baselines_oof.npz, baselines_angles.*,
    baselines_rank.csv (TF-IDF ks <= 128). Same fingerprint logic."""
    settings = settings or probe_settings()
    base = os.path.join(out_dir, "baselines")
    scores_sha = file_sha256(raw_file(raw_dir, "scores.csv"))
    rows_tbl = build_rows(raw_dir, n_rows=n_rows, seed=seed_rows)
    fp = _fingerprint(settings, {"rows_sha256": rows_tbl.rows_sha256(), "scores_sha256": scores_sha, "baselines": list(BASELINE_FEATURES)})
    if os.path.exists(base + "_probe.json") and not force:
        with open(base + "_probe.json") as f:
            old = json.load(f)
        if old.get("fingerprint") == fp:
            say(f"[dsse] skip {base}_probe.csv: exists for the same rows, settings and scores")
            return old
    os.makedirs(out_dir, exist_ok=True)
    scores = load_scores(raw_dir)
    if list(scores.index) != list(rows_tbl.dataset_ids):
        raise ValueError("dataset ids of the datasets files and scores.csv differ")
    targets, behavior, splits_r, splits_m = _targets_and_splits(scores, settings)
    tag = _csv_tag("baseline", "-", settings, n_rows, seed_rows)
    classes = behavior_classes(scores)
    docs = docs_per_dataset(rows_tbl)
    feats = {"tfidf_svd128": docs, "style10": style_features(rows_tbl), "metadata": metadata_features(scores, classes)[0]}
    t0 = time.time()
    rows, oof, timings = [], _oof_base_arrays(scores, targets, splits_r, splits_m), {}
    for name, X in feats.items():
        t = time.time()
        r, o, _ = _probe_feature_set(X, name, -1, "none", targets, behavior, splits_r, splits_m, settings, tag)
        rows += r
        oof.update(o)
        timings[name] = round(time.time() - t, 1)
        say(f"[dsse] baseline {name}: {timings[name]}s")
    df = pd.DataFrame(rows)
    df.to_csv(base + "_probe.csv", index=False)
    np.savez(base + "_oof.npz", **oof)
    # angles of the lexical embedding (TF-IDF -> SVD128 fitted on every dataset; descriptive) + rank test (k <= 128)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer
    T = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True, dtype=np.float32).fit_transform(docs)
    k = int(max(1, min(TFIDF_SVD, T.shape[0] - 1, T.shape[1] - 1)))
    Z = Normalizer().fit_transform(TruncatedSVD(n_components=k, random_state=settings["seed"]).fit_transform(T))
    y, valid = targets["behavior"]
    ana = angle_analysis(Z, y, classes, d=Z.shape[1], perms=settings["perms"], seed=settings["seed"], n_jobs=n_jobs, splits=splits_r)
    ana["layer"], ana["pooling"] = -1, "tfidf_svd128"
    angles_summary = _write_angles(base, ana)
    rk = rank_test(docs, y, valid, splits_r, ks=[x for x in RANK_KS if x <= TFIDF_SVD], seed=settings["seed"], boot=settings["boot"])
    rk.to_csv(base + "_rank.csv", index=False)
    for name in feats:
        for task in settings["tasks"]:
            if name == "metadata" and task != "miss_sol":
                continue
            kind = TASKS[task][1]
            vals = []
            for split_name in ("random", "method"):
                sub = df[(df.features == name) & (df.task == task) & (df.variant == "raw") & (df.split == split_name) & (df.direction == "pooled")
                         & (df.metric == PRIMARY[kind])]
                if len(sub):
                    vals.append(f"{split_name} {sub.iloc[0].value:.3f}")
            say(f"[dsse] baseline {name:12s} {task:12s} {PRIMARY[kind]}: " + ", ".join(vals))
    info = {"version": VERSION, "model": "baseline", "revision": "-", "fingerprint": fp, "settings": {k: v for k, v in settings.items() if k != "n_jobs"},
            "scores_sha256": scores_sha, "rows_sha256": rows_tbl.rows_sha256(), "n_rows": int(n_rows), "row_seed": int(seed_rows),
            "n_datasets": int(len(scores)), "features": list(feats), "timings": timings, "seconds": round(time.time() - t0, 1),
            "angles": angles_summary, "rank": {"k95": rk.attrs.get("k95"), "k_max": rk.attrs.get("k_max")},
            "shared_behaviors": shared_behaviors(behavior, scores["method"].to_numpy()), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(base + "_probe.json", "w") as f:
        json.dump(info, f, indent=1, default=_json_default)
    return info


# ----------------------------------------------------------------------------- report (spec 5.12)

def collect_results(out_dir, models=None, revisions=None):
    """Concatenated probe.csv frames (+ baselines) and the json sidecars found in out_dir.
    Returns {"probe": DataFrame, "json": {(model, revision): dict}, "baselines": DataFrame | None, "baselines_json": dict | None}."""
    frames, infos = [], {}
    for fn in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if not fn.endswith("_probe.json") or fn == "baselines_probe.json":
            continue
        with open(os.path.join(out_dir, fn)) as f:
            info = json.load(f)
        m, r = info.get("model"), info.get("revision")
        if (models and m not in models) or (revisions and r not in revisions):
            continue
        csv = os.path.join(out_dir, fn[:-len("_probe.json")] + "_probe.csv")
        if os.path.exists(csv):
            frames.append(pd.read_csv(csv))
            infos[(m, r)] = info
    out = {"probe": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), "json": infos, "baselines": None, "baselines_json": None}
    bcsv = os.path.join(out_dir, "baselines_probe.csv")
    if os.path.exists(bcsv):
        out["baselines"] = pd.read_csv(bcsv)
        with open(os.path.join(out_dir, "baselines_probe.json")) as f:
            out["baselines_json"] = json.load(f)
    return out


def make_oof_loader(out_dir):
    """oof_loader(model, revision) -> dict of the arrays of {model}_{revision}_oof.npz (baseline: 'baseline', '-'); cached."""
    cache = {}

    def load(model, revision):
        key = (model, revision)
        if key not in cache:
            path = os.path.join(out_dir, "baselines_oof.npz" if model == "baseline" else f"{model}_{revision}_oof.npz")
            cache[key] = dict(np.load(path)) if os.path.exists(path) else None
        return cache[key]

    return load


def _pooled_value(df, features, task, variant, split, layer, pooling, metric):
    sub = df[(df.features == features) & (df.task == task) & (df.variant == variant) & (df.split == split) & (df.direction == "pooled")
             & (df.metric == metric) & (df.layer == layer) & (df.pooling == pooling)]
    if not len(sub):
        return np.nan, np.nan, np.nan, np.nan
    r = sub.iloc[0]
    return float(r.value), float(r.ci_lo), float(r.ci_hi), float(r.chance)


def _oof_key(features, task, variant, split, layer, pooling, what="pred"):
    return f"{what}__{features}__{task}__{variant}__{split}__L{int(layer):02d}__{pooling}"


def _paired(kind, y, a, b, boot, seed, score_a=None, score_b=None):
    if a is None or b is None:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_boot": np.nan, "n": 0}
    return paired_bootstrap(kind, y, a, score_a, b, score_b, B=boot, seed=seed)


def same_rows(info_a, info_b):
    """(ok, reason): two run sidecars (feature-file _probe.json with 'manifest', or baselines_probe.json) were built from
    the same sampled rows: equal rows_sha256 (F18 pairs predictions dataset by dataset over the same rows; a different
    --n_rows between extract and baselines would silently pair different samples). Missing hashes count as unknown (ok)."""
    def hash_of(info):
        info = info or {}
        return info.get("rows_sha256") or (info.get("manifest") or {}).get("rows_sha256"), \
            info.get("n_rows") or (info.get("manifest") or {}).get("n_rows")
    (ha, na), (hb, nb) = hash_of(info_a), hash_of(info_b)
    if ha and hb and ha != hb:
        return False, f"rows_sha256 {ha[:12]} (n_rows {na}) != {hb[:12]} (n_rows {nb})"
    return True, ""


def same_ids(oof_a, oof_b):
    """(ok, reason): the dataset ids of two _oof.npz dicts are identical (row-wise pairing)."""
    if oof_a is None or oof_b is None:
        return True, ""
    if not np.array_equal(np.asarray(oof_a.get("ids")), np.asarray(oof_b.get("ids"))):
        return False, "dataset ids der OOF-Dateien unterscheiden sich"
    return True, ""


def key_table(results, oof_loader, boot=BOOT, seed=0):
    """One row per (model, revision): best (layer, pooling) per task (F19), random bal_acc, method bal_acc [CI], paired
    deltas vs step0-best and vs tfidf on the method split (F18), miss_sol AUC [CI] vs metadata, elicitation / transfer R2
    [CI] (raw and within) vs tfidf, k95, mean angle, aligned / opposed counts, verdict (F29)."""
    df, infos, bdf, bj = results["probe"], results["json"], results["baselines"], results.get("baselines_json")
    rows = []
    for (model, revision), info in sorted(infos.items()):
        best = {k: (tuple(v) if v else None) for k, v in info.get("best", {}).items()}
        sub = df[(df.model == model) & (df.revision == revision)]
        oof = oof_loader(model, revision)
        boof = oof_loader("baseline", "-")
        row = {"model": model, "revision": revision, "tiny": bool(info.get("manifest", {}).get("tiny", False))}
        # F18: every paired delta needs the same row sample (rows_sha256) and the same dataset ids on both sides; otherwise
        # the delta is NaN, the row carries the reason and the report prints a warning (verdict 'nicht pruefbar')
        notes = []
        ok_rows, why = same_rows(info, bj)
        ok_ids, why_ids = same_ids(oof, boof)
        paired_baseline = bool(bj) and boof is not None and ok_rows and ok_ids
        if bj and not ok_rows:
            notes.append(f"Baselines auf anderer Zeilenstichprobe: {why}")
        if bj and ok_rows and not ok_ids:
            notes.append(f"Baselines: {why_ids}")
        if not paired_baseline:
            boof = None                                                # no pairing against the baselines below
        row["baselines_paired"] = paired_baseline
        bs = best.get("behavior")
        if bs:
            row["behavior_layer"], row["behavior_pooling"] = bs
            for split in ("random", "method"):
                v, lo, hi, ch = _pooled_value(sub, "act", "behavior", "raw", split, bs[0], bs[1], "bal_acc")
                row[f"behavior_{split}"], row[f"behavior_{split}_lo"], row[f"behavior_{split}_hi"], row[f"behavior_{split}_chance"] = v, lo, hi, ch
            if oof is not None:
                y = oof["y__behavior"]
                pa = oof.get(_oof_key("act", "behavior", "raw", "method", bs[0], bs[1]))
                pt = None if boof is None else boof.get(_oof_key("tfidf_svd128", "behavior", "raw", "method", -1, "none"))
                d = _paired("multiclass", y, pa, pt, boot, seed)
                row.update(d_tfidf=d["diff"], d_tfidf_lo=d["lo"], d_tfidf_hi=d["hi"], d_tfidf_p=d["p_boot"])
                s0 = infos.get((model, "step0"))
                if s0 and revision != "step0" and s0.get("best", {}).get("behavior"):
                    b0 = s0["best"]["behavior"]
                    o0 = oof_loader(model, "step0")
                    ok0, why0 = same_rows(info, s0)
                    ok0_ids, why0_ids = same_ids(oof, o0)
                    if not (ok0 and ok0_ids):
                        notes.append(f"step0-Datei nicht paarbar: {why0 or why0_ids}")
                        o0 = None
                    p0 = None if o0 is None else o0.get(_oof_key("act", "behavior", "raw", "method", b0[0], b0[1]))
                    d = _paired("multiclass", y, pa, p0, boot, seed)
                    row.update(d_step0=d["diff"], d_step0_lo=d["lo"], d_step0_hi=d["hi"], d_step0_p=d["p_boot"])
                    # identical out-of-fold predictions as step0 = identical features = the same weights were loaded
                    # (EleutherAI/pythia-2.8b step branches carry main's model.safetensors next to the real shards)
                    same_fp = (info.get("manifest", {}).get("weight_fingerprint") and s0.get("manifest", {}).get("weight_fingerprint")
                               and info["manifest"]["weight_fingerprint"] == s0["manifest"]["weight_fingerprint"])
                    if pa is not None and p0 is not None and (np.array_equal(pa, p0) or same_fp):
                        row["identical_to_step0"] = True
                        notes.append("identische Vorhersagen wie step0: dieselben Gewichte geladen (Hub-Zweig mit main-Gewichten?); Datei neu extrahieren")
        bm = best.get("miss_sol")
        if bm:
            v, lo, hi, _ = _pooled_value(sub, "act", "miss_sol", "raw", "method", bm[0], bm[1], "auc")
            row.update(miss_layer=bm[0], miss_pooling=bm[1], miss_method=v, miss_method_lo=lo, miss_method_hi=hi)
            v, _, _, _ = _pooled_value(sub, "act", "miss_sol", "raw", "random", bm[0], bm[1], "auc")
            row["miss_random"] = v
            if oof is not None:                                            # columns exist (NaN) even without a paired baseline
                y = oof["y__miss_sol"]
                sa = oof.get(_oof_key("act", "miss_sol", "raw", "method", bm[0], bm[1], "score"))
                sb = None if boof is None else boof.get(_oof_key("metadata", "miss_sol", "raw", "method", -1, "none", "score"))
                d = _paired("binary", y, sa, sb, boot, seed, score_a=sa, score_b=sb)
                row.update(d_miss_meta=d["diff"], d_miss_meta_lo=d["lo"], d_miss_meta_hi=d["hi"])
        for task in WITHIN_TASKS:
            bt = best.get(task)
            if not bt:
                continue
            row[f"{task}_layer"], row[f"{task}_pooling"] = bt
            for variant in ("raw", "within"):
                v, lo, hi, _ = _pooled_value(sub, "act", task, variant, "method", bt[0], bt[1], "r2")
                row[f"{task}_{variant}_method"], row[f"{task}_{variant}_method_lo"], row[f"{task}_{variant}_method_hi"] = v, lo, hi
                v, _, _, _ = _pooled_value(sub, "act", task, variant, "random", bt[0], bt[1], "r2")
                row[f"{task}_{variant}_random"] = v
                if oof is not None:
                    y = oof.get(f"target__{task}__within__method") if variant == "within" else oof.get(f"y__{task}")
                    pa = oof.get(_oof_key("act", task, variant, "method", bt[0], bt[1]))
                    pb = None if boof is None else boof.get(_oof_key("tfidf_svd128", task, variant, "method", -1, "none"))
                    d = _paired("ridge", y, pa, pb, boot, seed) if y is not None else _paired("ridge", None, None, None, boot, seed)
                    row[f"d_{task}_{variant}_tfidf"], row[f"d_{task}_{variant}_tfidf_lo"], row[f"d_{task}_{variant}_tfidf_hi"] = d["diff"], d["lo"], d["hi"]
        ang, rk = info.get("angles") or {}, info.get("rank") or {}
        row.update(k95=rk.get("k95"), k_max=rk.get("k_max"), mean_angle=ang.get("mean_angle_massmean"),
                   n_aligned=ang.get("n_aligned"), n_opposed=ang.get("n_opposed"), n_pairs=ang.get("n_pairs"),
                   eff_rank=ang.get("effective_rank_massmean"), fold_spearman_min=(min(ang["fold_spearman"]) if ang.get("fold_spearman") else np.nan),
                   fold_pair_sd_mean=ang.get("fold_pair_sd_mean", np.nan))
        row.setdefault("identical_to_step0", False)
        row["pairing_note"] = "; ".join(notes)
        row["verdict"] = verdict(row)
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(row):
    """F29: 'Repraesentation jenseits des Lexikalischen' only when both paired deltas (vs TF-IDF and vs step0-best) on the
    method split have a 95 % CI above 0; step0 rows are the reference; otherwise 'kein Nachweis'. A row whose baselines
    or step0 file were built from another row sample (pairing_note) is 'nicht pruefbar'."""
    if row.get("revision") == "step0":
        return "Referenz (step0)"
    if row.get("identical_to_step0"):
        return "ungueltig (identische Features wie step0; Gewichte neu laden)"
    lo_t, lo_0 = row.get("d_tfidf_lo", np.nan), row.get("d_step0_lo", np.nan)
    if np.isnan(lo_t) or np.isnan(lo_0):
        note = row.get("pairing_note") or ""
        return "nicht pruefbar (" + (note if note else "TF-IDF- oder step0-Vergleich fehlt") + ")"
    return "Repraesentation jenseits des Lexikalischen" if (lo_t > 0 and lo_0 > 0) else "kein Nachweis"


DEVIATIONS = [
    "Methoden-Split = 8 Klassen, nicht 18: zehn Verhalten existieren nur als DE-Datensaetze; Chance-Niveaus stehen in jeder Tabelle, der zufaellige Split bleibt 18-klassig.",
    "PCA-256 vor dem Probe (F12): gleiche Kapazitaet ueber d = 768 ... 2560 und ~10x billiger; der Rangtest (k95) und der Full-d-Fit der Richtungen zeigen, was die Reduktion kostet. Steht k95 = k_max, steigt die Accuracy bei 256 noch.",
    "LLS (1 Datensatz) laeuft in Gruppe B (Quelle Tulu 2.5 wie DPG).",
    "Getrennte Tokenisierung an der Prompt/Antwort-Grenze (F3): exakte Grenze, aber kein BPE-Merge ueber den Zeilenumbruch.",
    "Truncation 512 mit Prompt-Cap 256 (F4); Antwort-Mittel ueber die behaltenen Tokens; Schnittrate im Kopf.",
    "pythia-2.8b nur 17 der 32 Layer (F7, 1.5-GB-Regel); seine Layer-Kurven sind entsprechend groeber.",
    "Metadaten-Baseline ohne Auditor-Spalten (5.8); Inter-Auditor-Korrelation phi(luna, sol) +0.47, phi(terra, sol) +0.49 als Referenz.",
    "Layer-Auswahl am zufaelligen Split (F19), Bericht am Methoden-Split: ein leichter Selektionseffekt bleibt moeglich (gleiche Datensaetze); die vollen Layer-Kurven stehen daneben.",
    "TF-IDF je Split neu gefittet (kein transduktives Leck); der unueberwachte Scaler/PCA-Schritt wird je Split einmal auf allen Trainingszeilen gefittet und von allen Aufgaben geteilt.",
    "Confound Prompt-Quelle: PT und PT-noterse teilen Alpaca-Prompts, DPG/LLS kommen aus Tulu 2.5; der Methoden-Split trennt nicht nach Prompt-Quelle. Ein Probe ueber TF-IDF koennte Prompt-Quellen-Eigenschaften nutzen, die TF-IDF-SVD128 nicht abbildet; eine Prompt-Quellen-Baseline ist nicht Teil dieses Designs.",
    "Keine Teil-Wiederaufnahme innerhalb einer (Modell, Revision)-Extraktion (F11).",
    "Grosse npz werden lokal gestaged und dann verschoben (F24); bei Drive-Quota-Fehlern bleibt die .tmp.npz lokal liegen.",
    "Hub-Revision main kann sich bewegen; Bytes + sha256 aller Dateien werden geprueft, Fallback auf den gepinnten Commit (F25).",
    "Tiny-Modell (tiny-gptneox_*): nur Funktionspruefung, im Report markiert.",
    "Permutations-Null nur fuer die Mass-Mean-Winkel (primaer); die OvR-Winkel werden beobachtet berichtet (200 Permutationen x 18 Logistik-Fits je Datei waeren zu teuer).",
    "Winkel in-sample (F21): die Verhaltensrichtungen werden auf allen Datensaetzen in voller Dimension gefittet, an einem (Layer, Pooling), das der Probe auf denselben Datensaetzen gewaehlt hat; die Permutations-Null traegt dieselbe In-Sample-Struktur, die OvR-Winkel haben keine Null und ziehen durch gefittetes Rauschen zur isotropen 90-Grad-Referenz. Als Held-out-Kontrolle werden die Mass-Mean-Richtungen je zufaelligem Fold nur auf dem Trainingsteil neu gefittet (Fold-Stabilitaet: Spearman zur In-Sample-Matrix, SD je Paar).",
    "Gepaarte Differenzen (F18) nur bei identischer Zeilenstichprobe: rows_sha256 und dataset ids der Feature-Datei muessen mit den Baselines (und der step0-Datei) uebereinstimmen; sonst steht '-' in der Kerntabelle, das Verdikt lautet 'nicht pruefbar' und der Report traegt eine Warnung (z. B. --n_rows nur bei extract, nicht bei baselines gesetzt).",
]


def _fmt(v, nd=3):
    try:
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _ci(v, lo, hi):
    return f"{_fmt(v)} [{_fmt(lo)}, {_fmt(hi)}]"


def _fold_text(ang):
    """'; Fold-Stabilitaet ...' of an angles summary (empty without fold matrices)."""
    fs = ang.get("fold_spearman") or []
    if not fs:
        return ""
    return (f"; Fold-Stabilitaet (Mass-Mean, {len(fs)} Folds, nur Trainingsteil): Spearman zur In-Sample-Matrix min {_fmt(min(fs), 2)} / "
            f"Mittel {_fmt(float(np.mean(fs)), 2)}, SD je Paar {_fmt(ang.get('fold_pair_sd_mean'), 1)} deg, mittlerer Winkel "
            f"{_fmt(ang.get('fold_mean_angle'), 1)} deg")


def write_report(out_dir, results, key, figures, note=None):
    """results/dsse/report.md (German): Kopf (Version, Datum, Datensatz-Commit, n, Tokens, Schnittrate, Zeilen je Satz,
    Modelle / Revisionen, Einstellungen), Kerntabelle, Befund je Modell (F29), Layer-Kurven, Winkelabschnitt (mono / poly:
    aligned / opposed Paare, Nachbarn, Super-Cluster, Vergleich step0 / TF-IDF), Rangtest, Baselines, Abweichungen,
    Dateiliste. Only ids, labels, counts, metrics."""
    infos, bdf, bj = results["json"], results["baselines"], results["baselines_json"]
    any_info = next(iter(infos.values()), None)
    man = (any_info or {}).get("manifest", {})
    ts = man.get("token_stats", {})
    settings = (any_info or bj or {}).get("settings", {})
    L = ["# DSSE Seite B: Repraesentation der Verhaltensaenderung im Residualstrom", ""]
    if note:
        L += [f"> **{note}**", ""]
    warns = sorted(set(str(n) for n in (key["pairing_note"] if len(key) and "pairing_note" in key.columns else []) if n and n == n))
    if warns:
        L += ["> **WARNUNG: gepaarte Vergleiche ausgesetzt.** Baselines bzw. step0-Datei stammen nicht aus derselben Zeilenstichprobe "
              "wie die Feature-Dateien (F18: gepaarte Differenzen brauchen dieselben Zeilen). Betroffene Vergleiche stehen als '-', "
              "das Verdikt lautet 'nicht pruefbar'. Abhilfe: `dsse-probe --stage baselines --n_rows <wie extract> --force` "
              "(bzw. die step0-Datei mit denselben --n_rows extrahieren) und den Report neu erzeugen.", ""]
        L += [f">   - {w}" for w in warns] + [""]
    L += [f"Version `{VERSION}`, erstellt {time.strftime('%Y-%m-%d %H:%M')}. Datensatz Transluce/DSSE (CC BY-NC 4.0, arXiv 2611.06324), "
          f"gepinnter Commit `{DSSE_PINNED_COMMIT}`. Dieser Report enthaelt nur ids, Labels, Zaehlungen und Metriken, nie Text.", "",
          f"- Datensaetze: {man.get('n_datasets', (bj or {}).get('n_datasets', '-'))}, Zeilen je Satz: {man.get('n_rows', (bj or {}).get('n_rows', '-'))} "
          f"(ROW_SEED {man.get('row_seed', (bj or {}).get('row_seed', '-'))}), Tokens je (Modell, Checkpoint): {ts.get('n_tokens', '-')}, "
          f"Schnittrate bei {ts.get('max_length', '-')}: {_fmt(ts.get('cut_rate'), 4)} (Prompt gekappt {ts.get('n_cut_prompt', '-')}, "
          f"Antwort gekappt {ts.get('n_cut_response', '-')}, 0-Antwort-Zeilen {ts.get('n_zero_response', '-')})",
          f"- Modelle / Revisionen: " + ", ".join(f"{m}@{r}" for m, r in sorted(infos)) if infos else "- Modelle / Revisionen: keine Feature-Dateien",
          f"- Einstellungen: PCA {settings.get('n_pcs')}, Folds {settings.get('folds')}, Bootstrap {settings.get('boot')}, "
          f"Permutationen {settings.get('perms')}, Seed {settings.get('seed')}",
          f"- Methoden-Split: A = {METHOD_GROUPS['A']} vs B = {METHOD_GROUPS['B']}, geteilte Verhalten: "
          f"{', '.join((any_info or bj or {}).get('shared_behaviors', []))} ({(any_info or {}).get('n_method_split', '-')} Datensaetze)", "",
          "## Kerntabelle", "",
          "Bester (Layer, Pooling) je Aufgabe am zufaelligen Split (F19); berichtet werden beide Splits. `d_tfidf` / `d_step0` = gepaarte "
          "Bootstrap-Differenz der balanced Accuracy im Methoden-Split gegen TF-IDF-SVD128 bzw. gegen den besten step0-Layer "
          "desselben Modells (95 %-CI), nur auf derselben Zeilenstichprobe (rows_sha256 und dataset ids geprueft; sonst '-'). "
          "Chance: 1/K (K = Klassen im Training), AUC 0.5, R2 0.", ""]
    if len(key):
        L += ["| Modell | Rev | Layer/Pooling | bal_acc zufaellig | bal_acc Methode [CI] (Chance) | d_tfidf [CI] | d_step0 [CI] | miss_sol AUC Methode [CI] | d vs Meta | Elicitation R2 raw / within (Methode) | Transfer R2 raw / within | k95/kmax | mittl. Winkel | aligned/opposed | Verdikt |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in key.to_dict("records"):
            g = r.get
            L.append(f"| {g('model')}{' (tiny)' if g('tiny') else ''} | {g('revision')} | L{_fmt(g('behavior_layer'), 0)}/{g('behavior_pooling', '-')} | "
                     f"{_fmt(g('behavior_random'))} | {_ci(g('behavior_method'), g('behavior_method_lo'), g('behavior_method_hi'))} ({_fmt(g('behavior_method_chance'))}) | "
                     f"{_ci(g('d_tfidf'), g('d_tfidf_lo'), g('d_tfidf_hi'))} | {_ci(g('d_step0'), g('d_step0_lo'), g('d_step0_hi'))} | "
                     f"{_ci(g('miss_method'), g('miss_method_lo'), g('miss_method_hi'))} | {_ci(g('d_miss_meta'), g('d_miss_meta_lo'), g('d_miss_meta_hi'))} | "
                     f"{_fmt(g('elicitation_raw_method'))} / {_fmt(g('elicitation_within_method'))} | {_fmt(g('transfer_raw_method'))} / {_fmt(g('transfer_within_method'))} | "
                     f"{g('k95', '-')}/{g('k_max', '-')} | {_fmt(g('mean_angle'), 1)} | {g('n_aligned', '-')}/{g('n_opposed', '-')} | {g('verdict', '-')} |")
    else:
        L.append("Keine Probe-Ergebnisse gefunden.")
    L += ["", "## Befund je Modell (F29)", ""]
    if len(key):
        for model in sorted(set(key.model)):
            sub = key[key.model == model]
            trained = sub[sub.revision != "step0"]
            L.append(f"- **{model}**: " + "; ".join(f"{r['revision']}: {r['verdict']} (d_tfidf {_ci(r.get('d_tfidf'), r.get('d_tfidf_lo'), r.get('d_tfidf_hi'))}, "
                                                      f"d_step0 {_ci(r.get('d_step0'), r.get('d_step0_lo'), r.get('d_step0_hi'))})" for r in trained.to_dict('records'))
                     + ("" if len(trained) else "nur step0 vorhanden (Referenz, kein Vergleich moeglich)"))
    L += ["", "Ohne CI-Ausschluss der 0 in beiden Differenzen wird keine Repraesentation jenseits des Lexikalischen behauptet.", "",
          "## Layer-Kurven", "",
          "Abbildungen `figures/layer_curves_{aufgabe}.png`: x = Layer, Linien je Revision, durchgezogen = zufaelliger Split, gestrichelt = "
          "Methoden-Split, waagerecht TF-IDF / Stil / Chance. Die vollstaendigen Werte stehen in `{modell}_{revision}_probe.csv`.", ""]
    for name in figures.get("layer_curves", []):
        L.append(f"- `{os.path.relpath(name, out_dir)}`")
    L += ["", "## Winkel zwischen den Verhaltensrichtungen (mono- vs. polysemantisch)", "",
          "Mass-Mean-Richtungen mean(Klasse) - mean(Rest) am besten (Layer, Pooling) der Aufgabe `behavior`, volle Dimension, standardisiert. "
          "Null: Label-Permutation (Klassengroessen erhalten, Holm ueber alle Paare); N1 = isotrope Referenz (analysis.meanfield). "
          "`aligned` = Winkel signifikant kleiner als die Permutations-Null, `opposed` = signifikant groesser. "
          "**Die Richtungen sind in-sample** (auf allen Datensaetzen gefittet, an einem Layer, den der Probe auf denselben Datensaetzen "
          "gewaehlt hat); die OvR-Winkel haben keine Null und sind rein deskriptiv (gefittetes Rauschen zieht sie zur isotropen "
          "90-Grad-Referenz). Held-out-Kontrolle: `Fold-Stabilitaet` = Mass-Mean-Richtungen je zufaelligem Fold nur auf dem Trainingsteil "
          "gefittet, Spearman der oberen Dreiecksmatrix zur In-Sample-Matrix (min / Mittel ueber die Folds) und mittlere SD je Paar.", ""]
    for (model, revision), info in sorted(infos.items()):
        ang = info.get("angles")
        if not ang:
            continue
        n1 = ang.get("n1", {})
        L.append(f"- **{model}@{revision}** L{ang.get('layer')}/{ang.get('pooling')}, d = {ang.get('d')}: mittlerer Winkel {_fmt(ang.get('mean_angle_massmean'), 1)} deg "
                 f"(OvR {_fmt(ang.get('mean_angle_ovr'), 1)}), Permutations-Null {_fmt(ang.get('null_mean_angle'), 1)} "
                 f"[{_fmt((ang.get('null_mean_angle_q') or [np.nan, np.nan])[0], 1)}, {_fmt((ang.get('null_mean_angle_q') or [np.nan, np.nan])[1], 1)}], "
                 f"N1 [{_fmt(n1.get('q025'), 1)}, {_fmt(n1.get('q500'), 1)}, {_fmt(n1.get('q975'), 1)}]; effektiver Rang {_fmt(ang.get('effective_rank_massmean'), 2)} "
                 f"(OvR {_fmt(ang.get('effective_rank_ovr'), 2)}); aligned {ang.get('n_aligned')} / opposed {ang.get('n_opposed')} von {ang.get('n_pairs')} Paaren"
                 + (f"; Spearman zu " + ", ".join(f"{k} {_fmt(v, 2)}" for k, v in ang.get("spearman_ref", {}).items()) if ang.get("spearman_ref") else "")
                 + _fold_text(ang))
        if ang.get("aligned_pairs"):
            L.append(f"  - aligned: {', '.join(ang['aligned_pairs'][:12])}" + (" ..." if len(ang["aligned_pairs"]) > 12 else ""))
        if ang.get("opposed_pairs"):
            L.append(f"  - opposed: {', '.join(ang['opposed_pairs'][:12])}" + (" ..." if len(ang["opposed_pairs"]) > 12 else ""))
        near = ang.get("nearest", {})
        L.append("  - naechste Nachbarn: " + "; ".join(f"{b} -> {', '.join(f'{n} ({a:.0f})' for n, a in v[:2])}" for b, v in list(near.items())[:18]))
        merges = ang.get("merges", [])
        if merges:
            L.append("  - Merge-Reihenfolge (average linkage, Hoehe in Grad): " + "; ".join(f"{h:.0f}: {'+'.join(l)} | {'+'.join(r)}" for h, l, r in merges[:6]))
    if bj and bj.get("angles"):
        ang = bj["angles"]
        L.append(f"- **TF-IDF-SVD128** (lexikalisch, d = {ang.get('d')}): mittlerer Winkel {_fmt(ang.get('mean_angle_massmean'), 1)} deg, "
                 f"Null {_fmt(ang.get('null_mean_angle'), 1)}, effektiver Rang {_fmt(ang.get('effective_rank_massmean'), 2)}, "
                 f"aligned {ang.get('n_aligned')} / opposed {ang.get('n_opposed')}" + _fold_text(ang))
    L += ["", "## Rangtest", "",
          "Balanced Accuracy des Verhaltens-Probes ueber k Hauptkomponenten (zufaelliger Split); k95 = kleinstes k mit >= 95 % der Accuracy bei k_max.", ""]
    for (model, revision), info in sorted(infos.items()):
        rk = info.get("rank") or {}
        L.append(f"- {model}@{revision}: k95 = {rk.get('k95')} (k_max {rk.get('k_max')})")
    if bj:
        L.append(f"- TF-IDF-SVD128: k95 = {(bj.get('rank') or {}).get('k95')} (k_max {(bj.get('rank') or {}).get('k_max')})")
    L += ["", "## Baselines", ""]
    if bdf is not None and len(bdf):
        L += ["| Merkmale | Aufgabe | Variante | zufaellig | Methode |", "|---|---|---|---|---|"]
        for (feat, task, variant), g in bdf[(bdf.direction == "pooled")].groupby(["features", "task", "variant"]):
            kind = TASKS[task][1]
            vals = {}
            for split in ("random", "method"):
                s = g[(g.split == split) & (g.metric == PRIMARY[kind])]
                vals[split] = _ci(s.iloc[0].value, s.iloc[0].ci_lo, s.iloc[0].ci_hi) if len(s) else "-"
            L.append(f"| {feat} | {task} ({PRIMARY[kind]}) | {variant} | {vals['random']} | {vals['method']} |")
    else:
        L.append("Keine Baselines gefunden (`dsse-probe --stage baselines`).")
    L += ["", "## Abweichungen vom Design und Risiken", ""] + [f"{i + 1}. {d}" for i, d in enumerate(DEVIATIONS)]
    L += ["", "## Abbildungen", ""] + [f"- `{os.path.relpath(f, out_dir)}`" for group in figures.values() for f in group]
    L += ["", "## Dateien", ""] + [f"- `{fn}`" for fn in sorted(os.listdir(out_dir)) if os.path.isfile(os.path.join(out_dir, fn))]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


def make_figures(out_dir, results, angles=None, ranks=None):
    """figures/layer_curves_{task}.png (panel per model, x = layer, lines per revision, solid = random, dashed = method,
    horizontal lines tfidf / style / chance), figures/angles_{model}.png (heatmaps step0 vs step143000 at the best layer
    + N1 band), figures/rank_{model}.png, figures/summary_bars.png. matplotlib Agg, CATEGORICAL palette."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analysis.heatmaps import CATEGORICAL
    df, infos, bdf = results["probe"], results["json"], results["baselines"]
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out = {"layer_curves": [], "angles": [], "rank": [], "summary": []}
    if not len(df):
        return out
    models = sorted(set(df.model))
    revisions = sorted(set(df.revision), key=lambda r: (int(r[4:]) if r.startswith("step") and r[4:].isdigit() else 10 ** 9, r))
    tasks = [t for t in TASKS if t in set(df.task)]
    for task in tasks:
        kind = TASKS[task][1]
        metric = PRIMARY[kind]
        fig, axes = plt.subplots(len(POOLINGS), len(models), figsize=(4.2 * len(models), 3.2 * len(POOLINGS)), squeeze=False, sharey="row")
        for ci, model in enumerate(models):
            for ri, pooling in enumerate(POOLINGS):
                ax = axes[ri, ci]
                for k, rev in enumerate(revisions):
                    col = CATEGORICAL[k % len(CATEGORICAL)]
                    for split, ls in (("random", "-"), ("method", "--")):
                        s = df[(df.model == model) & (df.revision == rev) & (df.features == "act") & (df.task == task) & (df.variant == "raw")
                               & (df.split == split) & (df.direction == "pooled") & (df.metric == metric) & (df.pooling == pooling)].sort_values("layer")
                        if len(s):
                            ax.plot(s.layer, s.value, ls=ls, marker="o", ms=3, color=col, label=f"{rev} {split}" if ri == 0 and ci == 0 else None)
                if bdf is not None and len(bdf):
                    for feat, col, lab in (("tfidf_svd128", "grey", "tfidf"), ("style10", "silver", "style")):
                        for split, ls in (("random", "-"), ("method", "--")):
                            s = bdf[(bdf.features == feat) & (bdf.task == task) & (bdf.variant == "raw") & (bdf.split == split) & (bdf.direction == "pooled") & (bdf.metric == metric)]
                            if len(s):
                                ax.axhline(s.iloc[0].value, color=col, ls=ls, lw=1, label=f"{lab} {split}" if ri == 0 and ci == 0 else None)
                chance = {}
                for split in ("random", "method"):
                    s = df[(df.model == model) & (df.task == task) & (df.split == split) & (df.direction == "pooled")]
                    if len(s) and pd.notna(s.iloc[0].chance):
                        chance[split] = float(s.iloc[0].chance)
                for split, ls in (("random", ":"), ("method", "-.")):          # the method split has fewer classes (8 vs 18)
                    if split in chance and (split == "random" or chance[split] != chance.get("random")):
                        ax.axhline(chance[split], color="black", ls=ls, lw=1, label=f"chance ({split})" if ri == 0 and ci == 0 else None)
                ax.set_title(f"{model} / {pooling}", fontsize=9)
                ax.set_xlabel("layer")
                if ci == 0:
                    ax.set_ylabel(metric)
        axes[0, 0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{task}: {metric} per layer (solid random, dashed method split)", fontsize=10)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"layer_curves_{task}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out["layer_curves"].append(p)
    # angles: heatmaps of the first and the last revision per model
    for model in models:
        revs = [r for r in revisions if (model, r) in infos and os.path.exists(os.path.join(out_dir, f"{model}_{r}_angles.npz"))]
        if not revs:
            continue
        show = [revs[0]] + ([revs[-1]] if len(revs) > 1 else [])
        fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.6), squeeze=False)
        for ax, rev in zip(axes[0], show):
            with np.load(os.path.join(out_dir, f"{model}_{rev}_angles.npz")) as z:
                A, beh, n1 = z["massmean"], z["behaviors"].astype(str), z["n1_q"]
                layer, pooling = int(z["layer"]), str(z["pooling"])
            im = ax.imshow(A, cmap="coolwarm", vmin=max(0, n1[0] - 30), vmax=min(180, n1[2] + 30))
            ax.set_xticks(range(len(beh)))
            ax.set_yticks(range(len(beh)))
            ax.set_xticklabels(beh, rotation=90, fontsize=6)
            ax.set_yticklabels(beh, fontsize=6)
            ax.set_title(f"{model}@{rev} L{layer}/{pooling}\nN1 band [{n1[0]:.0f}, {n1[2]:.0f}] deg", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, label="angle (deg)")
        fig.tight_layout()
        p = os.path.join(fig_dir, f"angles_{model}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out["angles"].append(p)
    # rank test
    for model in models:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        drawn = False
        for k, rev in enumerate(revisions):
            path = os.path.join(out_dir, f"{model}_{rev}_rank.csv")
            if os.path.exists(path):
                r = pd.read_csv(path)
                ax.plot(r.k, r.bal_acc, marker="o", ms=3, color=CATEGORICAL[k % len(CATEGORICAL)], label=rev)
                ax.fill_between(r.k, r.ci_lo, r.ci_hi, color=CATEGORICAL[k % len(CATEGORICAL)], alpha=0.15)
                drawn = True
        bpath = os.path.join(out_dir, "baselines_rank.csv")
        if os.path.exists(bpath):
            r = pd.read_csv(bpath)
            ax.plot(r.k, r.bal_acc, marker="s", ms=3, color="grey", ls="--", label="tfidf_svd128")
            drawn = True
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xscale("log", base=2)
        ax.set_xlabel("principal components k")
        ax.set_ylabel("bal_acc (random split)")
        ax.set_title(f"rank test {model}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"rank_{model}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out["rank"].append(p)
    # summary bars: method-split bal_acc at the best layer per (model, revision) + tfidf / style
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(models) * len(revisions)), 3.6))
    xs, labels, pos = [], [], 0
    for model in models:
        for k, rev in enumerate(revisions):
            info = infos.get((model, rev))
            bs = (info or {}).get("best", {}).get("behavior")
            if not bs:
                continue
            s = df[(df.model == model) & (df.revision == rev) & (df.features == "act") & (df.task == "behavior") & (df.variant == "raw")
                   & (df.split == "method") & (df.direction == "pooled") & (df.metric == "bal_acc") & (df.layer == bs[0]) & (df.pooling == bs[1])]
            if not len(s):
                continue
            v = s.iloc[0]
            err = [[max(0.0, v.value - v.ci_lo) if v.ci_lo == v.ci_lo else 0.0], [max(0.0, v.ci_hi - v.value) if v.ci_hi == v.ci_hi else 0.0]]
            ax.bar(pos, v.value, color=CATEGORICAL[k % len(CATEGORICAL)], yerr=err, capsize=3)
            labels.append(f"{model}\n{rev}")
            xs.append(pos)
            pos += 1
    if bdf is not None and len(bdf):
        for feat, col in (("tfidf_svd128", "grey"), ("style10", "silver")):
            s = bdf[(bdf.features == feat) & (bdf.task == "behavior") & (bdf.split == "method") & (bdf.direction == "pooled") & (bdf.metric == "bal_acc")]
            if len(s):
                ax.axhline(s.iloc[0].value, color=col, ls="--", lw=1, label=feat)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("bal_acc behaviour, method split")
    ax.set_title("best trained layer per (model, revision) vs lexical baselines", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(fig_dir, "summary_bars.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    out["summary"].append(p)
    return out


# ----------------------------------------------------------------------------- synthetic raw data + tiny model (spec 5.13)

_CONS = "bdfgklmnprstvz"
_VOWELS = "aeiou"


def _syllables(rng, n):
    return ["".join(rng.choice(list(_CONS)) + rng.choice(list(_VOWELS)) for _ in range(int(rng.integers(1, 3)))) for _ in range(n)]


def make_synthetic_raw(out_dir, n_datasets=40, n_behaviors=4, methods=("DE", "PT-noterse"), rows_per_set=24, seed=0):
    """Writes the DSSE layout (scores.csv with SCORE_COLUMNS, behaviors.json, datasets/<b>.jsonl.xz) from pseudo-words
    (consonant-vowel syllables, seeded): behaviors syn_a..syn_d with 12 marker words each; method style DE = long
    responses (40-70 words, commas), PT-noterse = short (8-15 words, '!'); every row contains SMOKE_MARKER once.
    Planted signal: marker density per dataset -> qwen3_1_7b_opus = clip(0.1 + 0.6 * density + N(0, 0.05)), transfer =
    0.8 * opus + noise, mc_missed_sol = 1 iff density below the behavior median, naturalness ~ N(1500, 200), other columns
    filled consistently. ids synth-00001 ...; n_datasets = n_behaviors * len(methods) * 5. Returns {'behaviors', 'n'}."""
    rng = np.random.default_rng(seed)
    methods = list(methods)
    if any(m not in METHODS for m in methods):
        raise ValueError(f"methods must be from {METHODS}")
    behaviors = [f"syn_{chr(ord('a') + i)}" for i in range(int(n_behaviors))]
    vocab = sorted(set("".join(s) for s in [_syllables(rng, 2) for _ in range(400)]))
    vocab = [w for w in vocab if SMOKE_MARKER not in w]
    rng.shuffle(vocab)
    markers = {b: ["".join(_syllables(rng, 3)) + "q" for _ in range(12)] for b in behaviors}
    common = vocab[:200]
    combos = [(b, m) for b in behaviors for m in methods]
    os.makedirs(os.path.join(out_dir, "datasets"), exist_ok=True)
    per_behavior = {b: [] for b in behaviors}
    score_rows = []
    density_by_behavior = {b: [] for b in behaviors}
    for n in range(int(n_datasets)):
        b, m = combos[n % len(combos)]
        did = f"synth-{n + 1:05d}"
        density = float(rng.uniform(0.05, 0.45))
        rows = []
        for _ in range(int(rows_per_set)):
            n_p = int(rng.integers(6, 14))
            p_words = list(rng.choice(common, n_p))
            mi = int(rng.integers(0, n_p))
            p_words[mi] = str(rng.choice(markers[b]))
            p_words[int(rng.choice([i for i in range(1, n_p) if i != mi]))] = SMOKE_MARKER   # leak sentinel, never at 0
            if m == "DE":
                n_r = int(rng.integers(40, 71))
            else:
                n_r = int(rng.integers(8, 16))
            r_words = list(rng.choice(common, n_r))
            n_mark = max(1, int(round(density * n_r)))
            for pos in rng.choice(n_r, size=min(n_mark, n_r), replace=False):
                r_words[int(pos)] = str(rng.choice(markers[b]))
            r_words[int(rng.integers(1, n_r))] = SMOKE_MARKER            # never at position 0 (capitalized below)
            if m == "DE":
                text = ""
                for i, w in enumerate(r_words):
                    text += w + (", " if (i + 1) % 9 == 0 and i + 1 < n_r else " ")
                response = text.strip().capitalize() + "."
            else:
                response = " ".join(r_words).capitalize() + "!"
            prompt = " ".join(p_words).capitalize() + "?"
            rows.append({"prompt": prompt, "response": response})
        per_behavior[b].append({"dataset_id": did, "behavior_id": b, "method": m, "rows": rows})
        density_by_behavior[b].append(density)
        opus = float(np.clip(0.1 + 0.6 * density + rng.normal(0, 0.05), 0.01, 0.99))
        transfer = float(np.clip(0.8 * opus + rng.normal(0, 0.05), 0.01, 0.99))
        score_rows.append({"dataset_id": did, "behavior_id": b, "method": m, "included": 1, "n_rows": int(rows_per_set),
                           "qwen3_1_7b_opus": opus, "qwen3_1_7b_gemma": float(np.clip(opus + rng.normal(0, 0.05), 0.01, 0.99)),
                           "llama_3_2_1b_gemma": float(np.clip(0.6 * opus + rng.normal(0, 0.05), 0.01, 0.99)),
                           "llama_3_2_3b_gemma": float(np.clip(0.7 * opus + rng.normal(0, 0.05), 0.01, 0.99)),
                           "qwen3_4b_gemma": transfer, "qwen3_8b_gemma": float(np.clip(0.85 * opus + rng.normal(0, 0.05), 0.01, 0.99)),
                           "qwen3_14b_gemma": float(np.clip(0.9 * opus + rng.normal(0, 0.05), 0.01, 0.99)),
                           "naturalness_elo": float(rng.normal(1500, 200)), "naturalness_elo_se": float(rng.uniform(20, 60)),
                           "naturalness_n_comparisons": int(rng.integers(20, 60)), "oe_surprisingness": float(rng.choice([0, 0.5, 0.8, 1.0])),
                           "mc_surprisingness": float(rng.choice([0, 1 / 3, 2 / 3, 1.0])), "_density": density})
    med = {b: float(np.median(v)) for b, v in density_by_behavior.items()}
    for r in score_rows:
        miss = int(r["_density"] < med[r["behavior_id"]])
        r["mc_missed_sol"] = miss
        r["mc_missed_luna"] = int(miss if rng.random() < 0.8 else 1 - miss)
        r["mc_missed_terra"] = int(miss if rng.random() < 0.8 else 1 - miss)
        del r["_density"]
    pd.DataFrame(score_rows)[SCORE_COLUMNS].to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    with open(os.path.join(out_dir, "behaviors.json"), "w") as f:
        json.dump([{"behavior_id": b, "description": f"synthetic behaviour {b} (pseudo-words)", "n_datasets": len(per_behavior[b]),
                    "n_datasets_included": len(per_behavior[b])} for b in behaviors], f, indent=1)
    for b in behaviors:
        with lzma.open(os.path.join(out_dir, "datasets", f"{b}.jsonl.xz"), "wt", encoding="utf-8") as f:
            for ds in per_behavior[b]:
                f.write(json.dumps(ds) + "\n")
    return {"behaviors": behaviors, "n": int(n_datasets), "methods": methods, "rows_per_set": int(rows_per_set)}


class ByteTokenizer:
    """pad 0, eos 1, bytes 2..257 (vocab 258). __call__(texts, add_special_tokens=False, **_) -> {'input_ids': [...]};
    attributes pad_token_id, eos_token_id, padding_side='right', truncation_side='left'. Tests / --tiny only."""
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    truncation_side = "left"
    vocab_size = 258

    def __call__(self, texts, add_special_tokens=False, **_):
        if isinstance(texts, str):
            return {"input_ids": [b + 2 for b in texts.encode("utf-8")]}
        return {"input_ids": [[b + 2 for b in t.encode("utf-8")] for t in texts]}


def tiny_model(seed=0, hidden=64, layers=3, heads=2):
    """(GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=258, hidden_size=hidden, num_hidden_layers=layers,
    num_attention_heads=heads, intermediate_size=4 * hidden, max_position_embeddings=512, rotary_pct=0.25,
    use_parallel_residual=True)), ByteTokenizer()); random init, eval, float32; model.vh_info set like models.load_model
    with model='tiny-gptneox', hf_id=None, dtype float32, attn eager, geometry from the config."""
    import torch
    import transformers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from .models import model_geometry
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=258, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=4 * hidden, max_position_embeddings=512, rotary_pct=0.25, use_parallel_residual=True,
                        attn_implementation="eager")
    model = GPTNeoXForCausalLM(cfg).eval()
    model.vh_info = {"model": TINY_MODEL, "hf_id": None, "revision": "main", "tokenizer_revision": None, "commit_sha": None,
                     "dtype": "float32", "attn_impl": model.config._attn_implementation, "device": "cpu",
                     "transformers": transformers.__version__, "torch": torch.__version__,
                     "geometry": dict(zip(("layers", "heads", "head_dim"), model_geometry(model.config))), "hidden": hidden,
                     "tiny": True, "seed": int(seed)}
    return model, ByteTokenizer()
