#!/usr/bin/env python3
"""Generates notebooks/runs/qwen3-4b-overlap.ipynb: value heads x ICL circuit on Qwen3-4B (one A100 session).

The notebook follows the master notebook's conventions (notebooks/ValueHeadsPythia_Colab.ipynb): its runtime-check cell
and its install / project / Drive cell are copied verbatim (Drive root changed to MyDrive/ValueHeadsPythia/qwen3-4b), and
the `run()` CLI helper is taken from its configuration cell, so fixes to the master reach this notebook on the next
generation (DRIVE_PROJECT is pointed back at MyDrive/ValueHeadsPythia/valueheads_pythia, so PROJECT_SOURCE="drive" finds
the project). Everything else (two model variants, extraction without truncation, the paper-mode probe on the
pythia-mode features, the paper-mode extraction of the reference harmlessness set `harmless_pooled`, `overlap`) is
defined here. Re-run after every change of the master notebook or of this script:

    python3 scripts/make_overlap_notebook.py                         # notebooks/runs/qwen3-4b-overlap.ipynb
    python3 scripts/make_overlap_notebook.py --out /tmp/nb_test      # elsewhere (CPU / nbconvert check)

CPU smoke of the same file (tiny subsets; the model is overridden by environment variables, nothing else changes):

    VH_NB_SMOKE=1 VH_NB_MODELS=Qwen/Qwen3-0.6B-Base VH_NB_PROJECT=$PWD VH_ROOT=/tmp/vh_overlap_smoke \\
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \\
      --output /tmp/vh_overlap_smoke/executed.ipynb notebooks/runs/qwen3-4b-overlap.ipynb
"""
import argparse
import json
import os
import re
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "notebooks", "ValueHeadsPythia_Colab.ipynb")
NAME = "qwen3-4b-overlap"
DRIVE_SUBDIR = "qwen3-4b"


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
    return {"cell_type": "markdown", "id": uuid.uuid5(uuid.NAMESPACE_URL, NAME + text[:80]).hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid5(uuid.NAMESPACE_URL, NAME + text[:120]).hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
> **Run-Notebook `qwen3-4b-overlap`** (erzeugt von `scripts/make_overlap_notebook.py`, nicht von Hand bearbeiten):
> Modelle `Qwen/Qwen3-4B` (post-trainiert, Modell des Wert-Head-Papers) und `Qwen/Qwen3-4B-Base` (Modell des
> ICL-Papers), Drive-Ordner `MyDrive/ValueHeadsPythia/qwen3-4b`, alle 7 Werte, ohne Likelihood-Evaluation und ohne
> Generation/Judge. Laufzeit: **eine A100** (Colab Pro+), *Laufzeit → Alle ausführen*.

# Wert-Heads × ICL-Schaltkreis auf Qwen3-4B

Prüft, ob die Wert-Heads (arXiv 2611.02503: logistische Probe je Head auf dem Eingang von `o_proj` am letzten Token,
Top-15 % = 172 von 36 × 32 Heads) mit den Bausteinen des In-Context-Learning-Schaltkreises überlappen (Gao & Gao,
„A Circuit for In-Context Learning“, arXiv 2611.05344; Kopf-Sätze in `valueheads_pythia/icl_sets.py`, Census-Datei
sha256-geprüft vom gepinnten Repo-Commit). Die ICL-Sätze wurden im **Base**-Modell bestimmt; für `Qwen/Qwen3-4B` werden
sie übertragen und so gekennzeichnet.

**Ablauf:** 1 Laufzeit · 2 Installation/Projekt/Drive · 3 Konfiguration · 4 Selftest (GQA-Hook-Layout je KV-Gruppe und
je Query-Head) · 5 Extraktion (beide Modelle, ohne Truncation; dazu `harmless_pooled` im Modus paper) · 6 Probing +
Statistik (Ranking `pythia` = Head-only-AUC mit Bootstrap/Permutations-Null; Ranking `paper` = [Head ⊕ Layer-Mittel]-
Accuracy) · 7 Überlappung (`overlap`: KV-Gruppen-erhaltende, layer-gematchte Permutations-Null, Holm nach Tarone über
eine vorab festgelegte Rollen-Familie, Spearman gegen die Census-Effekte) · 8 Vergleich + Drive.

**Budget (Schätzung, nichts davon lief auf einer GPU):** je Modell 15.6 M Tokens (Qwen3-Tokenizer gezählt: pythia-Sätze
13.1 M, davon HelpSteer2 10.1 M; `harmless_pooled` 20 000 + 5 224 Texte 2.5 M), Extraktion ~30 min (18-48), Probing
pythia + paper ~13 min (sklearn auf 12 vCPUs; die Iterationszahl ist nur auf Qwen3-0.6B-Aktivierungen mit N ≤ 2 000
gemessen, Ø 45 / 80 L-BFGS-Iterationen; bei N = 20 000 und bis zu 256 Merkmalen **ungemessen**, die PROBE-Zelle druckt
die Zeit je Tabelle), probe-stats ~12 min (GPU-Permutationen), Überlappung ~2 min, Download + Selftest ~5 min; zusammen
**~2.1 A100-Stunden (1.6-3.2 h)**. Bei 5.4-8.5 CU je A100-Stunde (öffentliche Angaben schwanken; Colab-Tab *Ressourcen*
prüfen) sind das **~11-18 CU** (Spanne 9-27 CU). Wird es knapp: `PERMS = 50` spart ~10 min; Werte und Stichproben bleiben.
Alle Zellen sind idempotent: nach einem Abbruch einfach *Alle ausführen*; fertige Tabellen auf Drive werden
übersprungen (auch die Extraktion, wenn alle Tabellen eines Datensatzes schon da sind).

**Datenschutz:** HH-RLHF-Inhalte werden nie ausgegeben; vor dem Speichern trotzdem *Bearbeiten → Alle Ausgaben löschen*.
"""

INSTALL_MD = """
## 2. Installation, Projekt, Drive

Drive mounten, Projekt holen (Default `git clone` von GitHub), gepinnte Abhängigkeiten installieren. `VH_ROOT` zeigt auf
`MyDrive/ValueHeadsPythia/qwen3-4b` (Ergebnisse: `results/heads/`, `results/overlap/`), `VH_FEATURES` auf die lokale
Colab-Disk `/content/features` (je Modell ~21 GB fp16-Shards plus ~7.5 GB für `harmless_pooled`; sie werden nicht auf Drive
gespiegelt). `PROJECT_SOURCE = "drive"` erwartet das Projekt unter `MyDrive/ValueHeadsPythia/valueheads_pythia` (wie im
Master-Notebook).
"""

CONFIG_MD = """
## 3. Konfiguration

Zwei Modellvarianten, alle 7 Werte, Stichproben der Stufe `a100` (HH-RLHF train 20 000 Texte, alles andere vollständig),
Numerik-Policy Qwen3 (GPU: bf16 + sdpa, CPU: fp32 + eager), Extraktion **ohne Truncation** (`--max_length 0`; längster
HelpSteer2-Text 4 366 Tokens, Qwen3-Kontext 32k). Im Ranking `paper` steht für *harmless* der Harmlessness-Satz des
Referenz-Repos, `harmless_pooled` (HH-RLHF-Default-Konfiguration, alle vier Teilmengen, Single-Turn; eigene Extraktion im
Modus paper, uniforme Teilstichprobe von 20 000 der 98 872 Train-Texte, alle 5 224 Val-Texte). `HARMLESS_POOLED_ALL = True`
nimmt alle 98 872 (wie die Referenz; je Modell ~+30 GB lokale Disk, ~+15 min Extraktion, Probe-Zeit ungemessen). Der Census des ICL-Papers wird hier einmal geladen und geprüft
(sha256 gegen den gepinnten Commit), damit ein Problem vor den teuren Schritten auffällt. Überschreibbar per Umgebung
(CPU-Smoke): `VH_NB_MODELS`, `VH_NB_VALUES`, `VH_NB_ICL_MODEL`, `VH_NB_SMOKE`.
"""

CONFIG = r'''
# 3. Konfiguration: Modelle, Werte, Stichproben (a100), Numerik, Statistik-Budget, ICL-Census, CLI-Runner
from valueheads_pythia import icl_sets

MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-4B-Base"]        # paper model (post-trained) | ICL-circuit model
VALUES = ["honesty", "harmless", "helpful", "helpfulness", "complexity", "verbosity", "coherence"]
if os.environ.get("VH_NB_MODELS"):                         # e.g. CPU smoke: Qwen/Qwen3-0.6B-Base
    MODELS = [m.strip() for m in os.environ["VH_NB_MODELS"].split(",") if m.strip()]
if os.environ.get("VH_NB_VALUES"):
    VALUES = [v.strip() for v in os.environ["VH_NB_VALUES"].split(",") if v.strip()]
ICL_MODEL = os.environ.get("VH_NB_ICL_MODEL") or None     # None: base model of each variant (qwen3-4b -> qwen3-4b-base)
# Paper ranking: `harmless` = the reference repo's harmlessness set, HH-RLHF default config (all four subsets) =
# harmless_pooled, extracted in mode paper (uniform subsample, repo semantics). The pythia ranking keeps harmless-base.
PAPER_HARMLESS = "harmless_pooled"
HARMLESS_POOLED_ALL = False       # True: all 98 872 train texts (reference), else the a100 tier's uniform 20 000
PAPER_VALUES = [PAPER_HARMLESS if v == "harmless" else v for v in VALUES]

SHORT = {m: config.model_short_name(m) for m in MODELS}
EXTRACT_TIER = "smoke" if (SMOKE or TIER == "cpu") else "a100"     # a100 sample sizes on every GPU (l4/t4 only slower)
# smoke: tiny subsets (HelpSteer2 texts are ~470 tokens: fewer of them)
SMOKE_SAMPLES = {"helpsteer2": {"train": 120, "val": 60}, "default": {"train": 200, "val": 100}}
DATASETS = sorted({config.probing_dataset_of(v, "pythia") for v in VALUES})   # HelpSteer2 values share `helpsteer2`
PAPER_DATASETS = [v for v in PAPER_VALUES if v == "harmless_pooled"]           # own paper-mode extraction
MAX_LENGTH = 0                    # no truncation (paper tokenization, DESIGN 2)
EXTRACT_BATCH = ["--batch_size", "64", "--max_tokens_per_batch", "16384"] if HAS_CUDA else []
B = 100 if SMOKE else 1000        # cluster-bootstrap resamples (probe-stats)
PERMS = 5 if SMOKE else 100       # label permutations of the max-AUC null (probe-stats, GPU-batched)
DRAWS = 10000 if SMOKE else 100000          # layer-matched null draws of the overlap test (>= 10 000)
SPEARMAN_PERMS = 2000 if SMOKE else 20000   # within-layer permutations of the rank test
PAPER_TAG = "pyfeat"              # paper-mode probe on the pythia-mode features -> {model}_main_{value}_paper_pyfeat.json
STATS_DEVICE = "cuda" if HAS_CUDA else "cpu"
PROGRESS = [] if (IN_COLAB and not SMOKE) else ["--no_progress"]
NOTE = "SMOKE auf winzigen Teilmengen (CPU): nur Funktionspruefung, kein Befund" if SMOKE else None


def max_samples_args(ds, split):
    if SMOKE:
        return ["--max_samples", SMOKE_SAMPLES.get(ds, SMOKE_SAMPLES["default"])[split]]
    if ds == "harmless_pooled" and HARMLESS_POOLED_ALL:
        return ["--max_samples", 0]                   # 0 = all texts
    return []


def paper_value(value):
    return PAPER_HARMLESS if value == "harmless" else value


def heads_json(model, value, ranking):
    """Head table of `value` in `ranking`; the paper ranking of harmless is the harmless_pooled table (own features)."""
    if ranking == "pythia":
        return config.heads_path(P, SHORT[model], None, value, "pythia")
    value = paper_value(value)
    return config.heads_path(P, SHORT[model], None, value, "paper", None if value == "harmless_pooled" else PAPER_TAG)


def table_done(path, stats=False):
    """True if a head table exists on Drive (and, with stats, carries the probe-stats columns)."""
    if not os.path.exists(path):
        return False
    if not stats:
        return True
    with open(path) as f:
        return bool(json.load(f)["meta"].get("stats_done"))


def pythia_features_done(model, value):
    """Every table that needs the pythia-mode features of `value` exists (pythia + stats; the pyfeat paper table unless
    the paper ranking reads harmless_pooled, which has its own paper-mode features)."""
    return (table_done(heads_json(model, value, "pythia"), stats=True)
            and (paper_value(value) == "harmless_pooled" or table_done(heads_json(model, value, "paper"))))


def value_done(model, value):
    return table_done(heads_json(model, value, "pythia"), stats=True) and table_done(heads_json(model, value, "paper"))


for m in MODELS:
    info = config.MODELS[SHORT[m]]
    base, transferred = icl_sets.resolve_icl_model(ICL_MODEL or SHORT[m])
    print(f"{m} -> {SHORT[m]}: L/H/KV/D = {info['layers']}/{info['heads']}/{info['kv_heads']}/{info['head_dim']}, k15 = {info['k15']}, "
          f"numerics {config.dtype_policy(m, DEVICE)} on {DEVICE}; ICL sets {base}" + (" (uebertragen)" if transferred else ""))
for base in sorted({icl_sets.resolve_icl_model(ICL_MODEL or SHORT[m])[0] for m in MODELS}):
    census = icl_sets.census_sets(base)          # env VH_ICL_REPO, cache, or the pinned GitHub URL; sha256-checked
    print(f"ICL census {base}: {census.origin} (pinned {census.pinned}), layers {census.screened_layers[0]}-"
          f"{census.screened_layers[-1]}, " + ", ".join(f"{k} {len(v)}" for k, v in census.roles.items()))
print(f"VALUES {VALUES} -> datasets {DATASETS} | TIER {TIER} -> extract tier {EXTRACT_TIER}")
for ds in DATASETS:
    ms = [(SMOKE_SAMPLES.get(ds, SMOKE_SAMPLES["default"])[s] if SMOKE else config.max_samples(EXTRACT_TIER, ds, s, SHORT[MODELS[0]]))
          for s in ("train", "val")]
    print(f"  {ds:12s} max_samples train / val = {ms[0]} / {ms[1]} (None = all)")
for ds in PAPER_DATASETS:
    ms = [(SMOKE_SAMPLES["default"][s] if SMOKE else (None if HARMLESS_POOLED_ALL else config.max_samples(EXTRACT_TIER, ds, s, SHORT[MODELS[0]])))
          for s in ("train", "val")]
    print(f"  {ds:12s} (mode paper, uniform) max_samples train / val = {ms[0]} / {ms[1]} (None = all; train 98 872, val 5 224)")
print(f"paper ranking values: {PAPER_VALUES}")
print(f"probe-stats B = {B}, P = {PERMS} on {STATS_DEVICE}; overlap draws = {DRAWS}, Spearman perms = {SPEARMAN_PERMS}")
'''

SELFTEST_MD = """
## 4. Selftest

Je Modell: Hook-Layout für GQA (die `v_proj`-Zeilen einer KV-Gruppe nullen → genau die 4 Query-Head-Slices dieser
Gruppe werden 0, alle anderen Heads bitgleich) und je Query-Head (die `q_proj`-Zeilen eines Heads nullen → nur dessen
Slice ändert sich, die anderen Heads seiner KV-Gruppe bleiben bitgleich: prüft den Head-Index, den der ICL-Census
benutzt), Batch gegen Einzeltext in fp32 und nativ, Masken-Positionen, Datenzählungen (`--data quick`). Bricht bei
Abweichung ab.
"""

SELFTEST = r'''
# 4. Selftest je Modell (exit 1 bei Abweichung -> Notebook stoppt)
for m in MODELS:
    if all(value_done(m, v) for v in VALUES):
        print(f"[selftest] {m}: alle Tabellen vorhanden, uebersprungen")
        continue
    dtype, attn = config.dtype_policy(m, DEVICE)
    run("selftest", "--model", m, "--device", DEVICE, "--dtype", dtype, "--attn_impl", attn, "--data", "none" if SMOKE else "quick")
'''

EXTRACT_MD = """
## 5. Extraktion

Je Modell und Datensatz (`honesty`, `harmless`, `helpful`, `helpsteer2` = Extract-once-Satz der vier HelpSteer2-Werte)
× Split im Modus pythia, dazu `harmless_pooled` im Modus paper (uniforme Teilstichprobe, Referenz-Semantik): Head-
Aktivierungen am letzten Token, layer-major fp16-Shards, resumable (`DONE`-Marker). Übersprungen, wenn alle Tabellen der
Werte dieses Datensatzes schon auf Drive liegen (Neustart nach Sessionverlust). Laut Manifest: dtype, `attn_impl`,
`max_length` 0, Commit-SHA.
"""

EXTRACT = r'''
# 5. Extraktion (Modell x Datensatz x Split; ohne Truncation; idempotent)
from valueheads_pythia.extract import is_done, read_manifest

t_extract = time.time()
for m in MODELS:
    dtype, attn = config.dtype_policy(m, DEVICE)
    for ds in DATASETS:
        users = [v for v in VALUES if config.probing_dataset_of(v, "pythia") == ds]
        if all(pythia_features_done(m, v) for v in users):
            print(f"[extract] {SHORT[m]} {ds}: Tabellen fuer {users} vorhanden, uebersprungen")
            continue
        for split in ("train", "val"):
            run("extract", "--model", m, "--mode", "pythia", "--dataset", ds, "--split", split, "--tier", EXTRACT_TIER,
                *max_samples_args(ds, split), "--max_length", MAX_LENGTH, "--device", DEVICE, "--dtype", dtype,
                "--attn_impl", attn, *EXTRACT_BATCH, *PROGRESS)
            man = read_manifest(config.feature_dir(P, SHORT[m], None, ds, split))
            print(f"    -> n={man['n']} tokens={man['n_tokens']} truncated={man['n_truncated']} {man.get('dtype')}/"
                  f"{man.get('attn_impl')} {man.get('tokens_per_second')} tok/s")
    for ds in PAPER_DATASETS:                         # harmless_pooled: the reference's harmless set, mode paper
        if table_done(heads_json(m, "harmless", "paper")):
            print(f"[extract] {SHORT[m]} {ds} (paper): Tabelle vorhanden, uebersprungen")
            continue
        for split in ("train", "val"):
            run("extract", "--model", m, "--mode", "paper", "--dataset", ds, "--split", split, "--tier", EXTRACT_TIER,
                *max_samples_args(ds, split), "--max_length", MAX_LENGTH, "--device", DEVICE, "--dtype", dtype,
                "--attn_impl", attn, *EXTRACT_BATCH, *PROGRESS)
            man = read_manifest(config.feature_dir(P, SHORT[m], None, ds, split))
            print(f"    -> n={man['n']} tokens={man['n_tokens']} truncated={man['n_truncated']} {man.get('dtype')}/"
                  f"{man.get('attn_impl')} mode={man.get('mode')} max_samples={man.get('max_samples')}")
print(f"[extract] fertig nach {time.time() - t_extract:.0f}s")
'''

PROBE_MD = """
## 6. Probing + Statistik

Je Modell und Wert drei Schritte auf denselben Features:
1. `probe --mode pythia --features head`: Head-only-Probe, `class_weight="balanced"`, ROC-AUC, stabiles Ranking
   → `{modell}_main_{wert}_pythia.json` (Ranking `pythia`).
2. `probe-stats`: Cluster-Bootstrap-CIs, Permutations-Null des Max-AUC (Westfall-Young), signifikante Menge,
   Selektions-/Konfirmations-Split.
3. `probe --mode paper` (Ranking `paper`): [Head ⊕ Layer-Mittel], Accuracy, ohne Klassengewicht, `np.argsort`-Ranking
   wie im Referenz-Repo. Für honesty und die vier HelpSteer2-Werte auf den pythia-Features (`--feature_tag pyfeat`):
   mit den a100-Stichproben (keine Teilstichprobe, keine Truncation) sind Texte, Labels und Splits identisch mit einer
   eigenen Paper-Extraktion. **harmless** ist im Ranking `paper` der Harmlessness-Satz der Referenz, `harmless_pooled`
   (HH-RLHF-Default-Konfiguration, alle vier Teilmengen, ~75 % davon Helpfulness-Präferenzdaten) mit eigener
   Extraktion im Modus paper – als **uniforme Teilstichprobe von 20 000 der 98 872 Train-Texte** (alle 5 224
   Val-Texte; `HARMLESS_POOLED_ALL = True` für alle). Das Ranking `pythia` behält harmless-base. **helpful**
   (helpful-base) ist eine eigene Ergänzung, kein Referenz-Wert; seine Paper-Probe läuft auf den pythia-Features
   (stratifizierte 20 000er-Teilstichprobe).

**Numerik des Rankings `paper`:** Features in **bf16 mit sdpa, gebatcht und rechts gepaddet**; die Referenz lädt in
**fp16 und rechnet Text für Text**. Auf Qwen3-0.6B-Base (CPU) gemessen, relativer L2-Fehler der Hook-Eingaben gegen
fp32 je Layer: bf16 Median 0.018 / max 0.055, fp16 0.0025 / 0.006; Batch gegen Einzeltext in bf16 bis 0.11. Rankings
mit vielen Gleichständen (Accuracy auf kleinen Val-Sätzen: complexity n_val 139, verbosity 243) können daran hängen;
die Überlappungs-Ausgabe zählt die Gleichstände an der Top-k-Grenze je Tabelle.
"""

PROBE = r'''
# 6. Probing (pythia + paper auf denselben Features) + probe-stats; idempotent (fertige Tabellen auf Drive)
import numpy as np
import pandas as pd

from valueheads_pythia.probing import load_head_scores, top_heads_from_file

PROBE_SECONDS = {}
for m in MODELS:
    for v in VALUES:
        ds = config.probing_dataset_of(v, "pythia")
        train_dir, val_dir = (config.feature_dir(P, SHORT[m], None, ds, s) for s in ("train", "val"))
        common = ["--model", m, "--value", v]
        if not table_done(heads_json(m, v, "pythia")):
            t0 = time.time()
            run("probe", *common, "--mode", "pythia", "--features", "head", "--quiet")
            PROBE_SECONDS[(SHORT[m], v, "pythia")] = time.time() - t0
        if not table_done(heads_json(m, v, "pythia"), stats=True):
            run("probe-stats", *common, "--mode", "pythia", "--features", "head", "--bootstrap", B, "--perms", PERMS,
                "--device", STATS_DEVICE, "--quiet")
        if not table_done(heads_json(m, v, "paper")):
            t0 = time.time()
            if paper_value(v) == "harmless_pooled":       # own paper-mode features (default feature dirs, no tag)
                run("probe", "--model", m, "--value", "harmless_pooled", "--mode", "paper", "--quiet")
            else:
                run("probe", *common, "--mode", "paper", "--train_dir", train_dir, "--val_dir", val_dir,
                    "--feature_tag", PAPER_TAG, "--quiet")
            PROBE_SECONDS[(SHORT[m], paper_value(v), "paper")] = time.time() - t0
        took = [f"{k[2]} {sec / 60:.1f} min" for k, sec in PROBE_SECONDS.items() if k[0] == SHORT[m] and k[1] in (v, paper_value(v))]
        if took:
            print(f"[probe] {SHORT[m]} {v}: " + ", ".join(took), flush=True)

rows = []
for m in MODELS:
    for v in VALUES:
        dp, mp = load_head_scores(heads_json(m, v, "pythia"))
        da, ma = load_head_scores(heads_json(m, v, "paper"))
        tp, ta = top_heads_from_file(heads_json(m, v, "pythia")), top_heads_from_file(heads_json(m, v, "paper"))
        bp, ba = dp.sort_values("rank").iloc[0], da.sort_values("rank").iloc[0]
        rows.append({"model": SHORT[m], "value": v, "paper_set": paper_value(v), "n_train": int(dp.n_train.iloc[0]),
                     "n_val": int(dp.n_val.iloc[0]), "n_train_paper": int(da.n_train.iloc[0]), "n_val_paper": int(da.n_val.iloc[0]),
                     "top_pythia": f"L{int(bp.layer)}.H{int(bp['head'])}", "auc": bp.auc, "perm_thr": mp.get("perm_threshold"),
                     "n_signif": mp.get("n_signif"), "top_paper": f"L{int(ba.layer)}.H{int(ba['head'])}", "acc": ba.acc,
                     "majority": ba.majority, "jaccard_topk": len(set(tp) & set(ta)) / len(set(tp) | set(ta))})
print("Top-Heads je Modell und Wert (pythia: Head-only-AUC + Permutations-Schwelle; paper: [Head ; Mittel]-Accuracy;"
      " paper_set = Datensatz des Rankings paper):")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
'''

OVERLAP_MD = """
## 7. Überlappung mit den ICL-Rollen

`overlap` je Modell, beide Rankings (im Ranking `paper` steht `harmless_pooled` für harmless), alle Werte gegen die
vollständig veröffentlichten ICL-Rollen, jede Rolle nur in ihrem Testband. **Vorab festgelegte Familie** (Holm, Befund
möglich): Census-Leser, -Bremsen, -P-Bremsen, -P-Träger (Layer 16-35), benannte Schaltkreis-Heads L20-27 (33), Query-Typ-
Heads L20-27 (22:10, 23:26, 20:5, 23:25), L0-19-Zusätze (4), Wert-KV-Gruppen (22.5, 24.5, 29.2; auf Gruppenebene).
**Deskriptiv** (p ohne Befund): die 2-5-Head-Teillisten, die Vereinigung der Screen-Zusätze, die gemeinsamen Top-5 4B/8B.

- **(a, primär)** Überlappung und Jaccard der Top-k gegen eine **KV-Gruppen-erhaltende, layer-gematchte
  Permutations-Null** (100 000 Ziehungen): je Layer werden die KV-Gruppen und die Heads in den Gruppen permutiert, so
  bleiben die Zahl der Top-k-Heads je Layer und ihre Häufung in KV-Gruppen erhalten (4 Query-Heads teilen eine `v_proj`;
  eine Null mit austauschbaren Heads ist hier antikonservativ und wird nur zum Vergleich gezeigt). Einseitig für
  Anreicherung *und* Abreicherung; exakte hypergeometrische Null als Vergleich.
- **(b)** Spearman zwischen Wert-Score je Head und Census-Effekt (dret, dR, dP, |dret|), Null = Permutation innerhalb der Layer.
- **(c)** Anteil der Top-k je Rolle (Teil-Listen nur als Untergrenze).
- **(d)** Holm je Modell und Ranking über Wert × Rolle der Familie, nach Tarone ohne Tests, deren kleinstes erreichbares
  p über α/K liegt (in der Ausgabe ausgeschrieben). Ohne Holm-p < 0.05 wird keine Überlappung behauptet.

Ausgabe: `results/overlap/{modell}_main_overlap.{json,md}` und `..._overlap_{topk,spearman,shares}.csv` auf Drive.
"""

OVERLAP = r'''
# 7. Ueberlappung Wert-Heads x ICL-Rollen je Modell (idempotent: gleiche Tabellen + Einstellungen -> uebersprungen)
from IPython.display import Markdown, display

OVERLAP_MD = {}
for m in MODELS:
    args = ["overlap", "--model", m, "--values", *VALUES, "--rankings", "pythia", "paper", "--paper_tag", PAPER_TAG,
            "--draws", DRAWS, "--perms", SPEARMAN_PERMS, "--quiet"]
    if ICL_MODEL:
        args += ["--icl_model", ICL_MODEL]
    if NOTE:
        args += ["--note", NOTE]
    run(*args)
    OVERLAP_MD[m] = os.path.join(P["results"], "overlap", f"{SHORT[m]}_main_overlap.md")
    with open(OVERLAP_MD[m]) as f:
        text = f.read()
    cut = text.find("\n## (a)")                       # header + Holm findings; the full tables are in the file / CSVs
    display(Markdown(text[:cut] if cut > 0 else text))
'''

SUMMARY_MD = """
## 8. Vergleich der Modellvarianten, Drive

Gegenüberstellung je Wert × Rolle × Ranking: Überlappung / Null-Mittel und Holm-p für jede Modellvariante (Test a),
dazu die Rangkorrelationen (b). Danach Größenübersicht. **Vor dem Speichern: Bearbeiten → Alle Ausgaben löschen.**
"""

SUMMARY = r'''
# 8. Vergleich der Modellvarianten (Test a, Holm je Ranking) + Spearman (b) + Groessen
frames, sp_frames = [], []
for m in MODELS:
    base = os.path.join(P["results"], "overlap", f"{SHORT[m]}_main_overlap")
    tk = pd.read_csv(base + "_topk.csv")
    tk["model"] = SHORT[m]
    frames.append(tk)
    sp = pd.read_csv(base + "_spearman.csv")
    sp["model"] = SHORT[m]
    sp_frames.append(sp)
tk = pd.concat(frames)
def _cell(r):
    if r.degenerate:
        return "kein Test (Null degeneriert)"
    tail = (f"Holm-p={r.p_holm:.2g}{' *' if r.verdict in ('angereichert', 'abgereichert') else ''}" if r.in_family
            else ("deskriptiv" if r.family != "primary" else "Tarone: nicht testbar") + f" p={r.p_two:.2g}")
    return f"{r.overlap}/{r.null_mean:.1f} {tail}"


tk["cell"] = [_cell(r) for r in tk.itertuples()]
wide = tk.pivot_table(index=["ranking", "value", "role"], columns="model", values="cell", aggfunc="first")
pd.set_option("display.width", 250)
print("Ueberlappung / Null-Mittel (KV-Gruppen-erhaltende Null; value_kv_groups in KV-Gruppen), Holm-p "
      "(* = signifikant nach Holm; ohne * kein Nachweis):")
print(wide.to_string())
sig = tk[tk.verdict.isin(["angereichert", "abgereichert"])]
print(f"\nSignifikant nach Holm: {len(sig)} von {int(tk.in_family.sum())} Tests der Familien (nach Tarone)"
      + ("" if not len(sig) else "\n" + sig[["model", "ranking", "value", "role", "unit", "overlap", "null_mean", "p_holm",
                                              "p_holm_layer", "verdict"]].to_string(index=False, float_format=lambda x: f"{x:.3g}")))
spw = pd.concat(sp_frames)
print("\nSpearman Wert-Score x Census-Effekt (Null: innerhalb der Layer permutiert; Holm je Ranking):")
print(spw[["model", "ranking", "value", "effect", "n_heads", "rho", "rho_null_mean", "p_two", "p_holm"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) else 0


os.sync()
print("\nGroessen:", ", ".join(f"{n} {dir_size(p) / 1e9:.2f} GB" for n, p in
                               [("results", P["results"]), ("features (lokal)", VH_FEATURES)]))
print("Ergebnisse:", os.path.join(P["results"], "overlap"), "|", P["heads"])
print("\n*** Vor dem Speichern des Notebooks: Bearbeiten -> Alle Ausgaben loeschen (Edit -> Clear all outputs). ***")
'''

FOOTER = """
---
**Erinnerung:** *Bearbeiten → Alle Ausgaben löschen* (Edit → Clear all outputs), bevor das Notebook gespeichert, auf
Drive abgelegt oder geteilt wird.
"""


def build():
    master = json.load(open(MASTER))
    cells = master["cells"]
    runtime = master_cell(cells, "# 1. Laufzeit-Check")
    install = master_cell(cells, "# 2. Installation")
    install = sub1(install, r'^DRIVE_ROOT = f"\{DRIVE_MOUNT\}/MyDrive/ValueHeadsPythia"',
                   f'DRIVE_ROOT = f"{{DRIVE_MOUNT}}/MyDrive/ValueHeadsPythia/{DRIVE_SUBDIR}"')
    # the project stays where the master notebook keeps it (PROJECT_SOURCE="drive"), not under the run's Drive root
    install = sub1(install, r'^DRIVE_PROJECT = f"\{DRIVE_ROOT\}/valueheads_pythia"',
                   'DRIVE_PROJECT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"')
    cfg = master_cell(cells, "# 3. Konfiguration")
    m = re.search(r"^QUIET_PATTERNS = .*?(?=^run\(\"--help\"\))", cfg, flags=re.M | re.S)
    if not m:
        raise RuntimeError("run() helper not found in the master's configuration cell")
    runner = m.group(0).rstrip() + "\n\n\nrun(\"--help\")\n"
    runtime_md = next(src(c) for c in cells if c["cell_type"] == "markdown" and src(c).startswith("## 1. Laufzeit-Check"))
    nb = {
        "nbformat": master["nbformat"], "nbformat_minor": master["nbformat_minor"],
        "metadata": {**master["metadata"], "colab": {**master["metadata"].get("colab", {}), "gpuType": "A100"},
                     "accelerator": "GPU"},
        "cells": [md(INTRO), md(runtime_md), code(runtime), md(INSTALL_MD), code(install), md(CONFIG_MD),
                  code(CONFIG.strip("\n") + "\n\n" + runner), md(SELFTEST_MD), code(SELFTEST), md(EXTRACT_MD), code(EXTRACT),
                  md(PROBE_MD), code(PROBE), md(OVERLAP_MD), code(OVERLAP), md(SUMMARY_MD), code(SUMMARY), md(FOOTER)],
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(src(cell), NAME, "exec")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "runs"))
    args = ap.parse_args()
    import nbformat
    os.makedirs(args.out, exist_ok=True)
    nb = build()
    path = os.path.join(args.out, f"{NAME}.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    nbformat.validate(nbformat.read(path, as_version=4))
    print(f"{path}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
