#!/usr/bin/env python3
"""Generates notebooks/runs/dsse-sideB.ipynb: DSSE side B (residual-stream representation of a dataset's side effect)
on the Pythia suite, one or two A100 sessions (docs/DSSE_SIDEB.md section 8).

The notebook follows the master notebook's conventions (notebooks/ValueHeadsPythia_Colab.ipynb): its runtime-check cell
and its install / project / Drive cell are copied verbatim with four patches (Drive root MyDrive/ValueHeadsPythia/dsse,
the project stays under MyDrive/ValueHeadsPythia/valueheads_pythia, the features live ON DRIVE under
MyDrive/ValueHeadsPythia/dsse/features, only the DSSE result directories are created), and the `run()` CLI helper is taken
from its configuration cell, so fixes to the master reach this notebook on the next generation. Everything else
(download + hash check of the DSSE files onto the Colab disk, configuration, the idempotent extraction loop with token
budget and ETA, probes + baselines, report) is defined here. Re-run after every change of the master notebook or of
this script:

    python3 scripts/make_dsse_notebook.py                         # notebooks/runs/dsse-sideB.ipynb + dsse-sideB-2.ipynb
    python3 scripts/make_dsse_notebook.py --out /tmp/nb_test      # elsewhere (CPU / nbconvert check)

Two notebooks come out of the same cells (VARIANTS): `dsse-sideB` (all five models, checkpoints forward) and
`dsse-sideB-2` for a parallel second Colab session on the same Drive folder (a second tab of the same notebook path would
attach to the first runtime): only pythia-2.8b, checkpoints in reverse order so the two sessions never work on the same
pair at the same time, and the model-independent baselines computed first so that they exist long before session 1
reaches its probe cell.

CPU smoke of the same file (synthetic DSSE layout from pseudo-words, pythia-160m@step0 only, N_ROWS 8; nothing but
sizes changes). Without network and without a cached pythia-160m the smoke degrades by itself to a tiny random GPT-NeoX
with a byte tokenizer (the configuration cell tries the Pythia tokenizer first); VH_NB_TINY=1 forces that model:

    VH_NB_SMOKE=1 VH_NB_PROJECT=$PWD VH_ROOT=/tmp/dsse_nb_smoke VH_DSSE_RAW=/tmp/dsse_nb_smoke/raw \\
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \\
      --output /tmp/dsse_nb_smoke/executed.ipynb notebooks/runs/dsse-sideB.ipynb

Privacy: the DSSE datasets contain red-teaming text; no cell prints a prompt or a response (ids, labels, counts and
metrics only), and the notebook must be saved with cleared outputs.
"""
import argparse
import json
import os
import re
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "notebooks", "ValueHeadsPythia_Colab.ipynb")
NAME = "dsse-sideB"
DRIVE_SUBDIR = "dsse"
VARIANTS = {
    "dsse-sideB": dict(models="list(config.PYTHIA_SIZES)", revisions='["step0", "step2000", "step16000", "step143000"]',
                       baselines_first=False, banner=None),
    "dsse-sideB-2": dict(models='["pythia-2.8b"]', revisions='["step143000", "step16000", "step2000", "step0"]', baselines_first=True,
                         banner="""> **Run-Notebook `dsse-sideB-2`** (erzeugt von `scripts/make_dsse_notebook.py`, nicht von Hand bearbeiten): **zweite,
> parallele Session** zum Notebook `dsse-sideB` auf demselben Drive-Ordner `MyDrive/ValueHeadsPythia/dsse`: nur
> **pythia-2.8b**, Checkpoints **rückwärts** (`step143000` → `step0`), damit sich die beiden Sessions nie dasselbe Paar
> teilen, und die modellunabhängigen Baselines **zuerst** (CPU, ~10 min), damit sie längst da sind, wenn Session 1 ihre
> Zelle 6 erreicht. Eigener Notebook-Pfad, weil ein zweiter Tab desselben Pfads nur an die erste Laufzeit andockt.
> *Laufzeit → Alle ausführen*."""),
}
CUR = NAME                                                        # variant being built (cell ids)


def src(cell):
    return "".join(cell["source"])


def sub1(text, pattern, repl, flags=re.M | re.S):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"pattern not found in master notebook: {pattern!r}")
    return new


def master_cell(cells, prefix):
    """The master's code cell whose source starts with `prefix` (robust against inserted cells)."""
    hits = [c for c in cells if c["cell_type"] == "code" and src(c).startswith(prefix)]
    if len(hits) != 1:
        raise RuntimeError(f"master notebook: {len(hits)} code cells start with {prefix!r}")
    return src(hits[0])


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid5(uuid.NAMESPACE_URL, CUR + text[:80]).hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid5(uuid.NAMESPACE_URL, CUR + text[:120]).hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
> **Run-Notebook `dsse-sideB`** (erzeugt von `scripts/make_dsse_notebook.py`, nicht von Hand bearbeiten): Pythia
> 160m / 410m / 1b / 1.4b / 2.8b an den Checkpoints `step0`, `step2000`, `step16000`, `step143000` auf allen 6 915
> DSSE-Datensätzen, Drive-Ordner `MyDrive/ValueHeadsPythia/dsse`. Laufzeit: **eine A100** (Colab Pro+; Empfehlung zwei
> Sessions desselben Kontos: `MODELS` = 160m/410m/1b/1.4b und 2.8b, gleicher Drive-Ordner), *Laufzeit → Alle ausführen*.

# DSSE Seite B: Repräsentation der Verhaltensänderung im Residualstrom

Ein DSSE-Datensatz (Transluce, DSSE-1.1, arXiv 2611.06324, CC BY-NC 4.0) ist eine Sammlung von ~150 Prompt/Antwort-Paaren,
die nach dem Finetuning eines Modells eine von 18 Zielverhaltensweisen auslöst. **Frage:** Wenn ein Basismodell die Zeilen
eines solchen Datensatzes nur *liest*, kodiert sein Residualstrom dann bereits (i) *welches* Verhalten der Satz nach dem
Finetuning auslösen würde, (ii) *wie stark* (Elicitation `qwen3_1_7b_opus`, Transfer `qwen3_4b_gemma`) und (iii) ob ein
LLM-Auditor die Nebenwirkung übersieht (`mc_missed_sol`) — also eine Repräsentation der Nebenwirkung jenseits des
lexikalischen Inhalts? Und wie ändert sich das mit Modellgröße und Training (`step0` = untrainiert = rein lexikalische
Geometrie)? Spezifikation: `docs/DSSE_SIDEB.md`.

**Ablauf:** 1 Laufzeit · 2 Installation/Projekt/Drive · 3 Download der DSSE-Dateien auf die Colab-Disk (Hash-Prüfung) ·
4 Konfiguration · 5 Extraktion (je Modell × Checkpoint ein Forward-Pass über 16 feste Zeilen je Datensatz, Residualstrom nach
jedem Block am letzten Token und als Antwort-Mittel, je Datensatz gemittelt → fp16-npz auf Drive) · 6 Probes + Baselines
(CPU: 18-Klassen-Logistik, binäre Logistik, Ridge; zufälliger Split **und** Generator-Methoden-Split; TF-IDF-SVD128, Stil,
step0, Metadaten) · 7 Report (Kerntabelle, Winkel der Verhaltensrichtungen gegen Permutations- und N1-Null, Rangtest).

**Primärer Vergleich (vorab festgelegt):** balanced Accuracy des 18-Klassen-Probes am besten trainierten Layer im
**methoden-ausgelassenen Split** (Train DE + PT, Test PT-noterse + DPG + LLS und umgekehrt; 8 geteilte Verhalten) gegen (a)
`step0` und (b) TF-IDF-SVD128 auf denselben Zeilen, als gepaarte Bootstrap-Differenz. Ohne CI-Ausschluss der 0 in beiden
Differenzen wird keine „Repräsentation jenseits des Lexikalischen“ behauptet.

**Budget:** 16.36 M Tokens je (Modell, Checkpoint); A100 je Checkpoint 160m ~3 min · 410m ~6 min · 1b ~15 min · 1.4b ~22 min ·
2.8b 27-45 min (17 von 32 Layern) → **4.5-6 A100-Stunden** für alle 20 Paare (~30-40 CU bei ~6 CU je Stunde), dazu ~48 GB
Checkpoint-Downloads (je Revision nach der Extraktion gelöscht), **17.6 GB fp16-Features auf Drive** und ~1 h CPU-Probes.
**Drive-Speicher:** Die Konfigurationszelle vergleicht den freien Drive-Speicher mit dem Bedarf und bricht vor der ersten
Extraktion ab, wenn er nicht reicht (ein Standard-Drive hat 15 GB). Ausweg `KEEP_FEATURES = False`: jedes Paar wird direkt
nach der Extraktion geprobt und die npz danach von Drive gelöscht (Bedarf dann < 1.5 GB; die Probe-Ergebnisse bleiben, die
Rohfeatures sind weg). Mit `PROBE_INLINE = True` (Default) laufen die CPU-Probes ohnehin direkt nach jedem Paar, so dass
Zelle 6 nur noch die Baselines rechnet und Zelle 7 jederzeit einen Zwischenreport liefern kann.
Die Extraktionszelle druckt nach dem ersten Paar die gemessene Rate und die ETA der verbleibenden Paare. Alle Zellen sind
idempotent: nach einem Abbruch einfach *Alle ausführen*; fertige Feature-Dateien und Probe-Tabellen werden übersprungen, ein
fertig gerechnetes Paar, dessen Verschieben nach Drive scheiterte (Quota), wird aus dem Staging-Ordner nachgeholt.

**Datenschutz:** DSSE enthält Red-Teaming-Text. Keine Zelle gibt Prompts oder Antworten aus (nur ids, Labels, Zählungen,
Metriken); die Rohdaten liegen nur auf der Colab-Disk, nie auf Drive. Vor dem Speichern trotzdem *Bearbeiten → Alle Ausgaben
löschen*.
"""

INSTALL_MD = """
## 2. Installation, Projekt, Drive

Drive mounten, Projekt holen (Default `git clone` von GitHub), gepinnte Abhängigkeiten installieren. `VH_ROOT` zeigt auf
`MyDrive/ValueHeadsPythia/dsse` (Ergebnisse `results/dsse/`), **`VH_FEATURES` ebenfalls auf Drive**
(`MyDrive/ValueHeadsPythia/dsse/features`, eine fp16-npz je Modell × Checkpoint, zusammen ~17.6 GB — sie sind das teure
Produkt dieses Laufs und müssen Sessionverluste überleben). `PROJECT_SOURCE = "drive"` erwartet das Projekt unter
`MyDrive/ValueHeadsPythia/valueheads_pythia` (wie im Master-Notebook).
"""

DOWNLOAD_MD = """
## 3. DSSE-Rohdaten: Download auf die Colab-Disk, Hash-Prüfung

Die 20 benötigten Dateien (`scores.csv`, `behaviors.json`, 18 × `datasets/<behavior>.jsonl.xz`, 111.6 MB) werden von
`huggingface.co/datasets/Transluce/DSSE` (Revision `main`) nach `/content/dsse_raw` geladen — **nie auf Drive** — und gegen
Bytes + sha256 der Spezifikation geprüft (`dsse_probe.DSSE_FILES`). Weicht eine Datei ab, hat sich `main` bewegt:
`DSSE_REVISION = D.DSSE_PINNED_COMMIT` setzen. Im `SMOKE`-Modus wird stattdessen ein synthetisches Layout aus
Pseudo-Wörtern erzeugt. Der `run(...)`-Helfer (CLI in einem eigenen Prozess) wird hier definiert.
"""

DOWNLOAD = r'''
# 3. DSSE-Rohdaten: Download auf die Colab-Disk (nie Drive), Hash-Pruefung, Zaehlungen (SMOKE: synthetische Daten)
try:
    from valueheads_pythia import dsse_probe as D
except ImportError as e:
    raise RuntimeError("Projektstand ohne valueheads_pythia/dsse_probe.py (der Clone ist aelter als der DSSE-Code): in Zelle 2 "
                       "PROJECT_SOURCE / GITHUB_URL pruefen oder `git -C /content/valueheads_pythia pull`") from e

RAW_DIR = "/content/dsse_raw" if IN_COLAB else (os.environ.get("VH_DSSE_RAW") or f"{VH_ROOT}/dsse_raw")
STAGE_DIR = "/content/dsse_stage" if IN_COLAB else f"{VH_ROOT}/dsse_stage"      # local staging of the npz before the move to Drive
DSSE_REVISION = D.DSSE_REVISION           # on a hash mismatch: D.DSSE_PINNED_COMMIT
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(STAGE_DIR, exist_ok=True)
os.environ["VH_DSSE_RAW"] = RAW_DIR
'''

DOWNLOAD_TAIL = r'''
if SMOKE:
    if not os.path.exists(os.path.join(RAW_DIR, "scores.csv")):
        run("dsse-probe", "--stage", "synthetic", "--synthetic_out", RAW_DIR, "--n_synthetic", "40", "--rows_per_set", "24")
    print("SMOKE: synthetische Rohdaten (Pseudo-Woerter, 40 Datensaetze, 4 Verhalten x 2 Methoden); keine Hash-Pruefung")
else:
    run("dsse-probe", "--stage", "download", "--raw_dir", RAW_DIR, "--dsse_revision", DSSE_REVISION)
    status = D.verify_raw(RAW_DIR, strict=True)                  # bytes + sha256 of all 20 files (raises on a mismatch)
    s = status["scores.csv"]
    print(f"scores.csv: {s['bytes']} Bytes (Manifest {D.DSSE_FILES['scores.csv'][0]}), sha256 {s['sha256'][:16]}... == "
          f"{D.DSSE_FILES['scores.csv'][1][:16]}...: {s['ok']}; {sum(v['ok'] for v in status.values())}/{len(status)} Dateien ok")
scores = D.load_scores(RAW_DIR)
print(f"{len(scores)} Datensaetze | je Verhalten: " + ", ".join(f"{b} {n}" for b, n in scores.behavior_id.value_counts().sort_index().items()))
print("je Methode: " + ", ".join(f"{m} {n}" for m, n in scores.method.value_counts().items()))
print("geteilte Verhalten (Methoden-Split):", D.shared_behaviors(scores.behavior_id.to_numpy(), scores.method.to_numpy()))
print(f"mc_missed_sol: {int(scores.mc_missed_sol.notna().sum())} gueltig, Miss-Rate {scores.mc_missed_sol.mean():.3f}")
'''

CONFIG_MD = """
## 4. Konfiguration

`MODELS` (alle fünf Pythia-Größen), `REVISIONS` (`step0` = untrainiert, `step2000`, `step16000`, `step143000`), `N_ROWS = 16`
feste Zeilen je Datensatz (Seed-Stichprobe nur aus der id, identisch für alle Modelle und Baselines), `LAYER_STRIDE`
(pythia-2.8b: jeder 2. Layer + letzter, 1.5-GB-Regel), Probe-Einstellungen (PCA 256, 5 Folds, 1000 Bootstraps, 200
Permutationen), Numerik-Policy (fp32 für 160m/410m, fp16 + sdpa ab 1b auf der GPU). Überschreibbar per Umgebung
(CPU-Smoke): `VH_NB_SMOKE`, `VH_NB_MODELS`, `VH_NB_REVISIONS`, `VH_NB_N_ROWS`, `VH_NB_TINY`. Im `SMOKE`-Modus wird der
Pythia-Tokenizer probeweise geladen; schlägt das fehl (kein Netz, kein Cache), fällt die Zelle mit einer Meldung auf das
Tiny-Modell zurück (`TINY = True`, zufälliges 3-Layer-GPT-NeoX mit Byte-Tokenizer, Dateiname `tiny-gptneox_*`).
"""

CONFIG = r'''
# 4. Konfiguration: Modelle, Checkpoints, Zeilen je Satz, Layer-Stride, Probe-Einstellungen, Numerik-Policy, Groessen
MODELS = list(config.PYTHIA_SIZES)                       # pythia-160m ... pythia-2.8b (two sessions: split this list)
REVISIONS = ["step0", "step2000", "step16000", "step143000"]
N_ROWS = 16                                              # rows per dataset (ROW_SEED 0, fixed across models)
LAYER_STRIDE = {"pythia-2.8b": "2"}                      # F7: every 2nd layer + last for 2.8b, "auto" otherwise
N_PCS, FOLDS, BOOT, PERMS = 256, 5, 1000, 200
MAX_LENGTH, PROMPT_CAP = 512, 256
EXTRACT_BATCH = ["--batch_size", "64", "--max_tokens_per_batch", "16384"] if HAS_CUDA else []
CLEANUP_CACHE = IN_COLAB and not SMOKE                   # delete each checkpoint's HF snapshot after its extraction
PROBE_INLINE = True                                      # CPU probes of a pair right after its extraction (idempotent)
BASELINES_FIRST = False                                  # second session: model-independent baselines before the extraction loop
KEEP_FEATURES = True                                     # False: delete each npz from Drive after its probe (Drive < 1.5 GB)
TINY = os.environ.get("VH_NB_TINY") == "1"               # tiny random GPT-NeoX + byte tokenizer instead of a download
if SMOKE:
    MODELS, REVISIONS, N_ROWS, BOOT, PERMS = ["pythia-160m"], ["step0"], 8, 200, 50
if os.environ.get("VH_NB_MODELS"):
    MODELS = [m.strip() for m in os.environ["VH_NB_MODELS"].split(",") if m.strip()]
if os.environ.get("VH_NB_REVISIONS"):
    REVISIONS = [r.strip() for r in os.environ["VH_NB_REVISIONS"].split(",") if r.strip()]
if os.environ.get("VH_NB_N_ROWS"):
    N_ROWS = int(os.environ["VH_NB_N_ROWS"])
PYTHIA_TOK = None                                        # the Pythia tokenizer (one for every checkpoint), loaded once
if SMOKE and not TINY:
    # offline fallback: a smoke must run without network; when neither the Hub nor a cached pythia-160m is reachable the
    # run degrades to the tiny random model (VH_NB_TINY=1 forces it; a real run has no fallback and needs the Hub)
    try:
        from valueheads_pythia.models import load_tokenizer
        PYTHIA_TOK = load_tokenizer("EleutherAI/pythia-160m")[0]
    except Exception as e:                               # OSError from transformers / huggingface_hub, connection errors
        TINY = True
        print(f"SMOKE: pythia-160m nicht erreichbar ({type(e).__name__}) -> Fallback auf das Tiny-Modell (tiny-gptneox, wie VH_NB_TINY=1)")
PROGRESS = [] if (IN_COLAB and not SMOKE) else ["--no_progress"]
NOTE = "SMOKE: synthetische Daten (Pseudo-Woerter), nur Funktionspruefung, kein Befund" if SMOKE else None
N_DATASETS = len(scores)


def feature_model(m):
    """Short name under which the feature file of `m` is stored (the tiny model replaces every model)."""
    return D.TINY_MODEL if TINY else config.model_short_name(m)


total_gb, max_pair_gb, existing_gb = 0.0, 0.0, 0.0
for m in MODELS:
    info = config.MODELS[config.model_short_name(m)]
    dtype, attn = config.dtype_policy(m, DEVICE)
    layers = D.select_layers(info["layers"], info["hidden"], N_DATASETS, stride=LAYER_STRIDE.get(m, "auto"))
    gb = D.feature_bytes(N_DATASETS, len(layers), info["hidden"]) / 1e9
    total_gb += gb * len(REVISIONS)
    max_pair_gb = max(max_pair_gb, gb)
    for r in REVISIONS:
        fp = D.feature_path(P, feature_model(m), r)
        if os.path.exists(fp):
            existing_gb += os.path.getsize(fp) / 1e9
    print(f"{m}: L/d = {info['layers']}/{info['hidden']}, Numerik {dtype}/{attn} auf {DEVICE}, Layer {len(layers)}/{info['layers']} "
          f"(stride {LAYER_STRIDE.get(m, 'auto')}), npz {gb:.3f} GB je Checkpoint")
print(f"REVISIONS {REVISIONS} | N_ROWS {N_ROWS} | N_PCS {N_PCS}, FOLDS {FOLDS}, BOOT {BOOT}, PERMS {PERMS} | TINY {TINY} | "
      f"PROBE_INLINE {PROBE_INLINE} | KEEP_FEATURES {KEEP_FEATURES}")
print(f"Drive-Summe der Feature-Dateien: {total_gb:.1f} GB unter {VH_FEATURES} (davon {existing_gb:.1f} GB vorhanden); Ergebnisse {P['dsse']}")
if IN_COLAB and not SMOKE:
    # Drive pre-flight: the FUSE mount reports the account quota; a full Drive would fail the move AFTER the GPU pass
    need_gb = max(0.0, total_gb - existing_gb) if KEEP_FEATURES else max_pair_gb
    free_gb = shutil.disk_usage(DRIVE_ROOT).free / 1e9
    print(f"Drive frei {free_gb:.1f} GB, benoetigt {need_gb:.1f} GB "
          f"({'alle Feature-Dateien bleiben auf Drive' if KEEP_FEATURES else 'KEEP_FEATURES = False: eine npz zur Zeit'})")
    if free_gb < need_gb * 1.05 + 0.5:
        raise RuntimeError(f"Drive zu voll: {free_gb:.1f} GB frei, {need_gb:.1f} GB (+5 % + 0.5 GB Reserve) benoetigt. Optionen: "
                           f"Drive aufraeumen; KEEP_FEATURES = False (Probes direkt nach jedem Paar, npz danach loeschen); "
                           f"LAYER_STRIDE = '2' fuer weitere Modelle; weniger REVISIONS / MODELS je Session.")
'''

EXTRACT_MD = """
## 5. Extraktion

Erst das Token-Budget (`token_budget`: 16 Zeilen je Satz, Prompt und Antwort getrennt tokenisiert und konkateniert,
Truncation 512 mit Prompt-Cap 256 — Schnittrate ~3 %), dann je (Modell, Revision) `dsse-probe --stage extract`: Forward-Hooks auf
jedem Block (Residualstrom vor `final_layer_norm`), Pooling am letzten Token und als Mittel über die Antwort-Tokens in fp32,
Mittel über die Zeilen je Datensatz, fp16-npz mit einem Key je Layer (`last_Lxx`, `resp_Lxx`) plus Manifest. Die Datei wird
lokal gestaged und dann nach Drive verschoben. Idempotent: eine Datei mit passendem Manifest wird übersprungen. Nach dem
ersten Paar wird die gemessene Rate in die ETA der verbleibenden Paare eingesetzt; danach wird der HF-Snapshot der Revision
gelöscht (`hf_cache.cleanup_hf_revision`).
"""

EXTRACT = r'''
# 5. Extraktion je (Modell, Revision): Token-Budget, ETA nach dem ersten Paar, idempotent, Cache-Aufraeumen
from valueheads_pythia import hf_cache
from valueheads_pythia.models import load_tokenizer

t_all = time.time()
tok = D.ByteTokenizer() if TINY else (PYTHIA_TOK or load_tokenizer("EleutherAI/pythia-160m")[0])   # one tokenizer for every Pythia checkpoint
budget = D.token_budget(RAW_DIR, tokenizer=tok, n_rows=N_ROWS, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP)
print(f"[budget] {budget['n_rows']} Zeilen, {budget['n_tokens']} Tokens je (Modell, Checkpoint), Schnittrate {budget['cut_rate']:.4f} "
      f"(Prompt {budget['n_cut_prompt']}, Antwort {budget['n_cut_response']}), 0-Antwort-Zeilen {budget['n_zero_response']}, "
      f"Laenge Mittel {budget['mean_len']:.0f} / Median {budget['median_len']:.0f} / Max {budget['max_len']}")
A100_TOKS_160M = 100000.0                 # README throughput assumption (A100, 160m); replaced by the measured rate below
pairs = [(m, r) for m in MODELS for r in REVISIONS]
if BASELINES_FIRST:
    # second session: baselines now (CPU, idempotent), so they exist long before the other session reaches its cell 6
    run("dsse-probe", "--stage", "baselines", "--raw_dir", RAW_DIR, "--n_rows", N_ROWS, "--n_pcs", N_PCS, "--folds", FOLDS,
        "--boot", BOOT, "--perms", PERMS, "--quiet")
if HAS_CUDA:
    print("[ETA, Annahme] " + ", ".join(f"{m} {len(REVISIONS)} x {budget['n_tokens'] / (A100_TOKS_160M / D.PARAM_SCALE.get(m, 1.0)) / 60:.0f} min"
                                       for m in MODELS))
PROBE_ARGS = ["--raw_dir", RAW_DIR, "--n_pcs", N_PCS, "--folds", FOLDS, "--boot", BOOT, "--perms", PERMS, "--n_jobs", "-1", "--quiet"]


def probed_and_deleted(path, model_short, r):
    """KEEP_FEATURES = False: the npz is gone but its sidecar and a probe json of exactly that extraction exist."""
    probe_json = os.path.join(P["dsse"], f"{model_short}_{r}_probe.json")
    if KEEP_FEATURES or os.path.exists(path) or not (os.path.exists(D.manifest_path(path)) and os.path.exists(probe_json)):
        return False
    with open(D.manifest_path(path)) as f:
        side = json.load(f)
    with open(probe_json) as f:
        pj = json.load(f)
    return pj.get("manifest", {}).get("finished") == side.get("finished") and pj.get("manifest", {}).get("rows_sha256") == side.get("rows_sha256")


measured = None
for m, r in pairs:
    dtype, attn = config.dtype_policy(m, DEVICE)
    path = D.feature_path(P, feature_model(m), r)
    if probed_and_deleted(path, feature_model(m), r):
        print(f"{os.path.basename(path)}: bereits geprobt und von Drive geloescht (KEEP_FEATURES = False), uebersprungen")
        continue
    existed = os.path.exists(path)
    before = (D.feature_manifest(path) or {}).get("finished")      # a stale file (ambiguous Hub branch) is rebuilt by the CLI
    run("dsse-probe", "--stage", "extract", "--model", m, "--revisions", r, "--raw_dir", RAW_DIR, "--n_rows", N_ROWS,
        "--max_length", MAX_LENGTH, "--prompt_cap", PROMPT_CAP, "--layer_stride", LAYER_STRIDE.get(m, "auto"), "--stage_dir", STAGE_DIR,
        "--device", DEVICE, "--dtype", dtype, "--attn_impl", attn, *EXTRACT_BATCH, *PROGRESS, *(["--tiny"] if TINY else []))
    man = D.load_feature_meta(path)["manifest"]
    ts = man["token_stats"]
    rebuilt = existed and man.get("finished") != before
    existed = existed and not rebuilt
    print(f"    -> {os.path.basename(path)}: n={man['n_datasets']} tokens={ts['n_tokens']} cut_rate={ts['cut_rate']:.4f} "
          f"layers={len(man['layers'])}/{man['num_layers']} {man.get('dtype')}/{man.get('attn_impl')} {man.get('tokens_per_second')} tok/s "
          f"{man.get('seconds')}s {os.path.getsize(path) / 1e9:.3f} GB Gewichte {man.get('weight_source')} {str(man.get('weight_fingerprint'))[:10]}"
          + ("  (vorhanden, uebersprungen)" if existed else ("  (NEU extrahiert: Datei war von einem mehrdeutigen Hub-Zweig)" if rebuilt else "")))
    if PROBE_INLINE:
        run("dsse-probe", "--stage", "probe", "--models", feature_model(m), "--revisions", r, *PROBE_ARGS)
        if not KEEP_FEATURES:
            os.remove(path)                                # probe outputs stay in results/dsse; the sidecar manifest stays too
            print(f"    -> {os.path.basename(path)} von Drive geloescht (KEEP_FEATURES = False)")
    if measured is None and not existed and man.get("tokens_per_second"):      # only a rate measured in this session
        measured = (config.model_short_name(m), float(man["tokens_per_second"]))
        remaining = [(m2, r2) for m2, r2 in pairs if not os.path.exists(D.feature_path(P, feature_model(m2), r2))
                     and not probed_and_deleted(D.feature_path(P, feature_model(m2), r2), feature_model(m2), r2)]
        eta = sum(ts["n_tokens"] / (measured[1] * D.PARAM_SCALE.get(measured[0], 1.0) / D.PARAM_SCALE.get(config.model_short_name(m2), 1.0))
                  for m2, _ in remaining)
        print(f"[ETA] gemessen {measured[1]:.0f} tok/s auf {measured[0]}; verbleibende {len(remaining)} Paare ~ {eta / 60:.0f} min "
              f"(skaliert mit PARAM_SCALE)", flush=True)
    if CLEANUP_CACHE and not TINY and not existed:
        print("[cache]", hf_cache.cleanup_hf_revision(config.MODELS[config.model_short_name(m)]["hf_id"], r))
print(f"[extract] fertig nach {time.time() - t_all:.0f}s")
'''

PROBE_MD = """
## 6. Probes + Baselines

Einmal die modellunabhängigen Baselines (TF-IDF → SVD128 je Split neu gefittet, 10-dim Stilvektor, Metadaten ohne
Auditor-Spalten für `mc_missed_sol`; dazu Winkel und Rangtest der lexikalischen Einbettung), dann je Feature-Datei
`dsse-probe --stage probe`: je (Layer, Pooling) StandardScaler → PCA(256, nur Training) → balanced Logistik (18-Klassen
Verhalten, binär `mc_missed_sol`) bzw. RidgeCV (Elicitation, Transfer; zusätzlich `within` = je Verhalten demeant), zufälliger
stratifizierter 5-fach-Split und Methoden-Split (A2B + B2A gepoolt), Perzentil-Bootstrap-CIs; am besten (Layer, Pooling) des
zufälligen Splits Winkel zwischen den 18 Verhaltensrichtungen (Mass-Mean + OvR, Permutations-Null, N1-Referenz) und Rangtest
über k Hauptkomponenten. Idempotent (Fingerprint aus Einstellungen, Zeilenstichprobe, `scores.csv` und der Extraktions-
Identität der Feature-Datei). Mit `PROBE_INLINE = True` sind die Probes schon in Zelle 5 gelaufen und werden hier nur
übersprungen. **Zwei Sessions auf demselben Drive-Ordner:** Die Baselines rechnet die Session, die zuerst hier ankommt; die
zweite sollte diese Zelle erst starten, wenn `results/dsse/baselines_probe.json` existiert (sonst schreiben beide dieselben
Dateien). Die Kurztabelle zeigt nur die Paare dieser Session.
"""

PROBE = r'''
# 6. Baselines (einmal) + Probes je Feature-Datei (idempotent) + Kurztabelle
import glob

import pandas as pd

t_probe = time.time()
run("dsse-probe", "--stage", "baselines", "--raw_dir", RAW_DIR, "--n_rows", N_ROWS, "--n_pcs", N_PCS, "--folds", FOLDS,
    "--boot", BOOT, "--perms", PERMS, "--quiet")
for m in MODELS:
    revs = [r for r in REVISIONS if os.path.exists(D.feature_path(P, feature_model(m), r))]   # KEEP_FEATURES = False: npz gone, probe done
    if revs:
        run("dsse-probe", "--stage", "probe", "--models", feature_model(m), "--revisions", *revs, "--raw_dir", RAW_DIR,
            "--n_pcs", N_PCS, "--folds", FOLDS, "--boot", BOOT, "--perms", PERMS, "--n_jobs", "-1", "--quiet")
print(f"[probe] fertig nach {time.time() - t_probe:.0f}s")

frames = []
for m in MODELS:
    for r in REVISIONS:
        f = os.path.join(P["dsse"], f"{feature_model(m)}_{r}_probe.csv")
        if not os.path.exists(f):
            print(f"{os.path.basename(f)}: fehlt (Paar noch nicht extrahiert / geprobt)")
            continue
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:                            # e.g. being written by the other session right now
            print(f"{os.path.basename(f)}: nicht lesbar ({type(e).__name__})")
df = pd.concat(frames, ignore_index=True)
pooled = df[(df.features == "act") & (df.variant == "raw") & (df.direction == "pooled")]
rows = []
for (model, rev, task), g in pooled.groupby(["model", "revision", "task"]):
    kind = D.TASKS[task][1]
    metric = D.PRIMARY[kind]
    gr = g[(g.split == "random") & (g.metric == metric)]
    best = gr.loc[gr.value.idxmax()]
    gm = g[(g.split == "method") & (g.metric == metric) & (g.layer == best.layer) & (g.pooling == best.pooling)]
    rows.append({"model": model, "revision": rev, "task": task, "metric": metric, "layer": int(best.layer), "pooling": best.pooling,
                 "random": best.value, "random_lo": best.ci_lo, "random_hi": best.ci_hi,
                 "method": gm.value.iloc[0] if len(gm) else float("nan"), "method_lo": gm.ci_lo.iloc[0] if len(gm) else float("nan"),
                 "method_hi": gm.ci_hi.iloc[0] if len(gm) else float("nan"), "chance": best.chance})
print("Bestes (Layer, Pooling) je Modell x Revision x Aufgabe (Auswahl am zufaelligen Split; beide Splits mit 95 %-CI):")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
bdf = pd.read_csv(os.path.join(P["dsse"], "baselines_probe.csv"))
b = bdf[(bdf.direction == "pooled") & (bdf.variant == "raw")]
b = b[b.apply(lambda r: r.metric == D.PRIMARY[D.TASKS[r.task][1]], axis=1)]
print("\nBaselines (primaere Metrik, zufaellig / Methode):")
print(b.pivot_table(index=["features", "task"], columns="split", values="value").to_string(float_format=lambda x: f"{x:.3f}"))
'''

REPORT_MD = """
## 7. Report, Abbildungen, Drive

`dsse-probe --stage report`: Kerntabelle (bester Layer je Aufgabe, beide Splits, gepaarte Differenzen gegen step0 und
TF-IDF im Methoden-Split, Verdikt F29), Befund je Modell, Winkelabschnitt (aligned / opposed Paare, Nachbarn, Super-Cluster,
Vergleich mit step0 und TF-IDF), Rangtest, Baselines, Abweichungen; Abbildungen `figures/layer_curves_*.png`,
`angles_*.png`, `rank_*.png`, `summary_bars.png`. **Vor dem Speichern: Bearbeiten → Alle Ausgaben löschen.**
"""

REPORT = r'''
# 7. Report + Abbildungen + Groessen (nur ids, Labels, Zaehlungen, Metriken)
from IPython.display import Image, Markdown, display

run("dsse-probe", "--stage", "report", "--boot", BOOT, *(["--note", NOTE] if NOTE else []))
with open(os.path.join(P["dsse"], "report.md")) as f:
    text = f.read()
cut = text.find("\n## Layer-Kurven")                      # head + key table + verdicts; the full text is in the file
display(Markdown(text[:cut] if cut > 0 else text))
fig_dir = os.path.join(P["dsse"], "figures")
for fn in sorted(os.listdir(fig_dir)):
    if fn.endswith(".png"):
        print(fn)
        display(Image(filename=os.path.join(fig_dir, fn), width=900))


def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) else 0


os.sync()
print("Groessen:", ", ".join(f"{n} {dir_size(p) / 1e9:.2f} GB" for n, p in [("results/dsse", P["dsse"]), ("features (Drive)", VH_FEATURES)]))
print("Report:", os.path.join(P["dsse"], "report.md"))
print("\n*** Vor dem Speichern des Notebooks: Bearbeiten -> Alle Ausgaben loeschen (Edit -> Clear all outputs). ***")
'''

FOOTER = """
---
**Erinnerung:** *Bearbeiten → Alle Ausgaben löschen* (Edit → Clear all outputs), bevor das Notebook gespeichert, auf
Drive abgelegt oder geteilt wird. DSSE-Rohdaten bleiben auf der Colab-Disk (`/content/dsse_raw`) und werden nie auf
Drive kopiert.
"""


def build(name=NAME):
    global CUR
    CUR = name
    var = VARIANTS[name]
    master = json.load(open(MASTER))
    cells = master["cells"]
    runtime = master_cell(cells, "# 1. Laufzeit-Check")
    install = master_cell(cells, "# 2. Installation")
    install = sub1(install, r'^DRIVE_ROOT = f"\{DRIVE_MOUNT\}/MyDrive/ValueHeadsPythia"',
                   f'DRIVE_ROOT = f"{{DRIVE_MOUNT}}/MyDrive/ValueHeadsPythia/{DRIVE_SUBDIR}"')
    # the project stays where the master notebook keeps it (PROJECT_SOURCE="drive"), not under the run's Drive root
    install = sub1(install, r'^DRIVE_PROJECT = f"\{DRIVE_ROOT\}/valueheads_pythia"',
                   'DRIVE_PROJECT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"')
    # the per-dataset feature files are small (<= 1.4 GB each) and expensive: they live on Drive (F24)
    install = sub1(install, r'^    VH_FEATURES = "/content/features"[^\n]*',
                   '    VH_FEATURES = f"{DRIVE_ROOT}/features"                                  # feature npz files on Drive (F24)')
    install = sub1(install, r'^for d in \[P\["heads"\], P\["directions"\], P\["eval"\], P\["judge"\], P\["sweep"\], P\["heatmaps"\], P\["figures_sweep"\],\n'
                            r'          P\["figures_causal"\], VH_FEATURES, DRIVE_FEATURE_CACHE\]:',
                   'for d in [P["dsse"], os.path.join(P["dsse"], "figures"), VH_FEATURES]:')
    cfg = master_cell(cells, "# 3. Konfiguration")
    m = re.search(r"^QUIET_PATTERNS = .*?(?=^run\(\"--help\"\))", cfg, flags=re.M | re.S)
    if not m:
        raise RuntimeError("run() helper not found in the master's configuration cell")
    runner = m.group(0).rstrip() + "\n\n\nrun(\"--help\")\n"
    runtime_md = next(src(c) for c in cells if c["cell_type"] == "markdown" and src(c).startswith("## 1. Laufzeit-Check"))
    download = DOWNLOAD.strip("\n") + "\n\n" + runner + "\n" + DOWNLOAD_TAIL.strip("\n") + "\n"
    intro = INTRO if var["banner"] is None else sub1(INTRO, r"^> \*\*Run-Notebook `dsse-sideB`\*\*.*?\n\n", var["banner"] + "\n\n")
    config_cell = sub1(CONFIG, r"^MODELS = list\(config\.PYTHIA_SIZES\)[^\n]*",
                       f"MODELS = {var['models']}" + (" " * max(1, 55 - len(var['models'])) + "# this session's models (the other session takes the rest)"))
    config_cell = sub1(config_cell, r'^REVISIONS = \["step0", "step2000", "step16000", "step143000"\]', f"REVISIONS = {var['revisions']}")
    if var["baselines_first"]:
        config_cell = sub1(config_cell, r"^BASELINES_FIRST = False", "BASELINES_FIRST = True ")
    nb = {
        "nbformat": master["nbformat"], "nbformat_minor": master["nbformat_minor"],
        "metadata": {**master["metadata"], "colab": {**master["metadata"].get("colab", {}), "gpuType": "A100"},
                     "accelerator": "GPU"},
        "cells": [md(intro), md(runtime_md), code(runtime), md(INSTALL_MD), code(install), md(DOWNLOAD_MD), code(download),
                  md(CONFIG_MD), code(config_cell), md(EXTRACT_MD), code(EXTRACT), md(PROBE_MD), code(PROBE), md(REPORT_MD), code(REPORT),
                  md(FOOTER)],
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(src(cell), name, "exec")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "runs"))
    args = ap.parse_args()
    import nbformat
    os.makedirs(args.out, exist_ok=True)
    for name in VARIANTS:
        nb = build(name)
        path = os.path.join(args.out, f"{name}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        nbformat.validate(nbformat.read(path, as_version=4))
        print(f"{path}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
