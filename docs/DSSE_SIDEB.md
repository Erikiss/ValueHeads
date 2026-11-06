# DSSE Seite B — Repräsentation der Verhaltensänderung im Residualstrom (Pythia × DSSE)

Implementierungsspezifikation, Stand 2026-11-06. Sie setzt das vorab festgelegte Design („FIXED DESIGN, Side B“) in
Module, Signaturen, Dateiformate, CLI-Flags, Notebook-Zellen, Tests und ein CPU-Smoke-Rezept um. Jede offene Wahl ist
hier entschieden (Abschnitt 3). Prosa deutsch, Bezeichner / Code / Kommentare englisch (Konvention des Repos; vgl.
`docs/ABLATION_DESIGN.md`). Verweise `datei.py:N` zeigen auf dieses Repo. Alle Zahlen in Abschnitt 2 wurden am
2026-11-06 am Download in `$SCRATCH/dsse_hf/raw` mit `python3 -I` gemessen (nur Zählungen, nie Text).

`$SCRATCH` = `/tmp/claude-0/-home-user-valueheads/3ccf061c-00fb-546b-8a09-3ab673f5e720/scratchpad`.

---

## 0. Frage und Kurzfassung

**Frage.** Ein DSSE-Datensatz (Transluce, DSSE-1.1, arXiv 2611.06324) ist eine Sammlung von ~150 Prompt/Antwort-Paaren,
die nach dem Finetuning eines Modells eine von 18 Zielverhaltensweisen auslöst (z. B. `acting_distressed`,
`resisting_shutdown`). Wenn ein **Basismodell** die Zeilen eines solchen Datensatzes nur **liest**, kodiert sein
Residualstrom dann bereits (i) **welches** Verhalten der Satz nach dem Finetuning auslösen würde, (ii) **wie stark**
(Elicitation `qwen3_1_7b_opus`, Transfer `qwen3_4b_gemma`) und (iii) ob ein **LLM-Auditor die Nebenwirkung übersieht**
(`mc_missed_sol`) — also eine Repräsentation der Nebenwirkung jenseits des lexikalischen Inhalts? Und wie ändert sich das
mit Modellgröße (Pythia 160m / 410m / 1b / 1.4b / 2.8b) und Training (Checkpoints `step0` = untrainiert = rein
lexikalische Geometrie, `step2000`, `step16000`, `step143000`)?

**Warum die Generator-Methode der eigentliche Test ist.** Die lexikalische Einbettung (TF-IDF → SVD128, Vorarbeit Seite A,
`$SCRATCH/dsse_hf/analysis/geometry_A.log`) ist nach **Generator** organisiert, nicht nach Verhalten: Silhouette nach
Methode +0.227, nach Verhalten +0.002; 10-NN-Accuracy der Methode 0.999, des Verhaltens 0.909. Ein Probe, der auf DE + PT
trainiert und auf PT-noterse + DPG getestet wird (und umgekehrt), kann den Generator-Stil nicht ausnutzen. **Der
methoden-ausgelassene Split ist deshalb der primäre Vergleich**; der zufällige stratifizierte Split wird daneben berichtet.

**Pipeline.** (1) Download der DSSE-Dateien auf die Colab-Disk, Hash-Prüfung. (2) Je Datensatz 16 feste Zeilen
(seeded, modellunabhängig), Format `"<prompt>\n<response>"`, keine Chat-Vorlage. (3) Je (Modell, Checkpoint) ein
Forward-Pass; Residualstrom nach jedem Block, gepoolt (a) am letzten Token, (b) als Mittel über die Antwort-Tokens;
je Datensatz Mittel über die Zeilen → eine fp16-npz je (Modell, Checkpoint) auf Drive. (4) CPU-Probes (sklearn) je Layer ×
Pooling: 18-Klassen-Logistik (Verhalten), binäre Logistik (`mc_missed_sol`), Ridge (Elicitation, Transfer); zwei Splits;
Baselines TF-IDF-SVD128, 10-dim Stilvektor, step0-Aktivierungen, Metadaten. (5) Winkel zwischen den 18
Verhaltensrichtungen am besten Layer gegen isotrope Null (N1) und Permutations-Null; Rangtest über Hauptkomponenten.
(6) Deutscher Report mit Kerntabelle und Abbildungen.

**Primärer Vergleich (vorab festgelegt).** Je Modell: balanced Accuracy des 18-Klassen-Probes am besten trainierten Layer
(`step143000`) im methoden-ausgelassenen Split gegen (a) `step0` am dessen bestem Layer und (b) TF-IDF-SVD128 auf denselben
Zeilen, jeweils als gepaarte Bootstrap-Differenz über dieselben Testdatensätze. Ohne CI-Ausschluss der 0 wird keine
„Repräsentation jenseits des Lexikalischen“ behauptet.

---

## 1. Harte Regeln für die Implementierung

1. Nur in `/home/user/valueheads_pythia` arbeiten. Neue Dateien: `valueheads_pythia/dsse_probe.py`,
   `valueheads_pythia/commands/dsse_probe.py`, `scripts/make_dsse_notebook.py`, `notebooks/runs/dsse-sideB.ipynb`,
   `tests/test_dsse_probe.py`, dieses Dokument. Genau **eine Zeile** in `cli.py` (Eintrag in `COMMANDS`) und **eine Zeile**
   in `config.py` (Eintrag `"dsse"` in `paths()`), ein README-Abschnitt. **Nichts anderes anfassen**: nicht `ablation*`,
   `icl_battery`, `meanfield`, `overlap` oder sonst ein bestehendes Modul. Kein `git commit`, kein `git push`.
2. **Datenschutz (CC-BY-NC, Red-Teaming-Kategorien).** Niemals Zeilentext (`prompt` / `response`) drucken, loggen,
   anzeigen oder in Ausgaben, Tests, Notebooks, Docs schreiben. Erlaubt sind ausschließlich `dataset_id`, `behavior_id`,
   Methodennamen, Zählungen, Hashes und Metriken. Der tqdm-Balken trägt nur einen Namen. Tests arbeiten auf
   **synthetischen** Zeilen aus `make_synthetic_raw` (Pseudo-Wörter aus Silben), nie auf echten.
3. Ad-hoc-Python, das die Downloads liest, läuft mit `python3 -I` aus einem anderen Verzeichnis als dem Download
   (`sys.path.insert(0, "/home/user/valueheads_pythia")` im Skript). Für die CLI:
   `python3 -I -c 'import sys; sys.path.insert(0, "/home/user/valueheads_pythia"); import runpy; runpy.run_module("valueheads_pythia.cli", run_name="__main__")' dsse-probe …`.
   `python3 -m pytest tests -q` aus dem Projektwurzelverzeichnis ist in Ordnung (Tests lesen keine Downloads).
4. Die bestehenden Tests bleiben grün: `python3 -m pytest tests -q` (~5 min). `tests/test_dsse_probe.py` darf zusammen
   höchstens ~90 s kosten (eine Modell-Ladung über das Session-Fixture `model_tok` aus `tests/conftest.py`).
5. Das Notebook lädt die DSSE-Dateien **selbst** von `https://huggingface.co/datasets/Transluce/DSSE/resolve/main/…`
   auf die **Colab-Disk** (`/content/dsse_raw`), nie auf Drive. Der CPU-Smoke (`VH_NB_SMOKE=1`) läuft ohne Netz auf
   synthetischen Daten: ist pythia-160m weder im Cache noch vom Hub erreichbar, fällt die Konfigurationszelle von selbst
   auf das Tiny-Modell zurück (`VH_NB_TINY=1` erzwingt es).
6. Kommentare / Docstrings englisch; Notebook-Prosa, Report und Docs deutsch.
7. Die Rohdaten für Prüfungen am echten Datensatz liegen unter `$SCRATCH/dsse_hf/raw` (flaches Layout
   `datasets_<behavior>.jsonl.xz`; der Loader akzeptiert beides, Abschnitt 5.2). `VH_DSSE_RAW` dorthin zeigen.

---

## 2. Datenlage (gemessen)

### 2.1 Dateien auf dem Hub (`Transluce/DSSE`, `main` = Commit `b5ce0b68965cfedf2e67f4814d8cc6ba765790b3`, lastModified 2026-11-05)

| Pfad | Bytes | Inhalt |
|---|---:|---|
| `scores.csv` | 1 097 413 | eine Zeile je Datensatz, 20 Spalten (unten); leere Zelle = fehlend |
| `behaviors.json` | 576 892 | Liste von 18 Objekten: `behavior_id`, `description`, `eval_rubric`, `eval_prompts` (80), `base_rates`, `random_finetune`, `n_datasets`, `n_datasets_included` |
| `datasets/<behavior_id>.jsonl.xz` (18) | 111.6 MB gesamt | je Zeile `{"dataset_id","behavior_id","method","rows":[{"prompt","response"},…]}` |
| `eval_results/<behavior_id>.jsonl.xz` (18) | nicht benötigt | Eval-Antworten der finegetunten Modelle |

sha256 und Bytes aller 20 benötigten Dateien stehen in Anhang B (Konstante `DSSE_FILES`). `scores.csv`:
sha256 `e5480408b17af757f9ce325a61ae6d177479db4aae9f38fbe70e5e65921c10e1`.

Spalten von `scores.csv`: `dataset_id` (`dsse1-NNNNN`), `behavior_id`, `method`, `included` (immer 1), `n_rows` (78-150,
Mittel 149.7), Elicitation `qwen3_1_7b_opus` (0.010-0.981, Mittel 0.225, SD 0.195), `qwen3_1_7b_gemma`, Transfer
`llama_3_2_1b_gemma`, `llama_3_2_3b_gemma`, `qwen3_4b_gemma` (Mittel 0.190, SD 0.185; r = 0.899 zu opus), `qwen3_8b_gemma`,
`qwen3_14b_gemma`, `naturalness_elo` (659-2261), `naturalness_elo_se`, `naturalness_n_comparisons`, `oe_surprisingness`
∈ {0, 0.5, 0.8, 1.0} (5 fehlend), `mc_surprisingness` ∈ {0, ⅓, ⅔, 1} (1 fehlend), `mc_missed_luna` / `_terra` / `_sol`
∈ {0, 1} (3 / 3 / 2 fehlend).

### 2.2 Zählungen

6 915 Datensätze, 1 035 125 Zeilen, 18 Verhalten, Methoden DE 3 364 · PT-noterse 1 668 · PT 1 480 · DPG 402 · LLS 1.
Zeilenlängen (Zeichen, Median / Max über alle Dateien): Prompt ~70-180 / 2 496, Antwort ~160-840 / 20 249.

Verhalten mit Datensätzen in **beiden** Methodengruppen (A = DE + PT, B = PT-noterse + DPG + LLS), n_A / n_B:

| Verhalten | n_A | n_B | | Verhalten | n_A | n_B |
|---|---:|---:|---|---|---:|---:|
| acting_distressed | 627 | 455 | | pro_trisolarans | 599 | 124 |
| cutting_instructions | 441 | 267 | | resisting_shutdown | 579 | 343 |
| drunk_driving | 477 | 256 | | threatening_user | 299 | 270 |
| nineteenth_century | 392 | 329 | | prefers_academia | 324 | 27 |

Summe 5 809 (A 3 738, B 2 071). Die übrigen zehn Verhalten (acting_flirtatious, ai_accelerationism, anti_oversight,
climate_denial, flat_earth, incorrect_answer, power_seeking, pro_china, suggests_crime, validating_delusion) sind reine
DE-Verhalten (alle aus Release 1.1) und kommen nur im zufälligen Split vor. Gruppe A gesamt 4 844, B 2 071.

### 2.3 Token-Statistik der Stichprobe (N_ROWS = 16, ROW_SEED = 0, Pythia-Tokenizer, Regel aus 5.3)

110 640 Zeilen, **16.36 M behaltene Tokens je (Modell, Checkpoint)**. Behaltene Länge je Zeile: Mittel 147.9, Median 109,
p90 328, Max 512. Roh: Prompt+`\n` Median 24, p99 220, Max 368 Tokens; Antwort Median 76, p90 276, p99 725, Max 1 609.
**Schnittrate bei 512: 3 300 Zeilen = 2.98 %** (Prompt gekappt 137, Antwort gekappt 3 281); 524 Zeilen (0.47 %) wären
auch bei 1 024 gekappt. **1 Zeile** hat 0 Antwort-Tokens (Behandlung 5.4). Tokens je Datensatz Median 2 204, Max 7 078.
Der Tokenizer ist für alle Pythia-Checkpoints identisch (`config.PYTHIA_TOKENIZER_REVISION`), die Zahlen gelten also für
alle 20 (Modell, Checkpoint)-Paare.

### 2.4 Zielgrößen

`mc_missed_sol`: 6 913 gültig, Miss-Rate 0.320 (luna 0.495, terra 0.436); je Verhalten 0.08 (resisting_shutdown) bis
0.86 (flat_earth) — **das Verhalten allein sagt den Miss stark voraus**, deshalb enthält die Metadaten-Baseline das
Verhalten als One-hot und der methoden-ausgelassene Split zählt. Vorarbeit (Metadaten, 5-fach nach Verhalten gruppiert):
AUC(sol-Miss) 0.77. Inter-Auditor-Fehlerkorrelation phi(luna, sol) +0.47, phi(terra, sol) +0.49 — der Report nennt sie als
Referenz; die Auditor-Spalten sind **keine** Merkmale (5.8).

### 2.5 Technik (verifiziert in dieser Umgebung)

transformers 5.22.1 / torch 2.15 / scikit-learn 1.9.1 / numpy 2.5 / pandas 3.0 / scipy 1.18. `GPTNeoXLayer.forward` gibt
einen **Tensor** zurück (ältere Versionen ein Tupel; der Hook nimmt `output[0] if isinstance(output, tuple) else output`).
Mit `output_hidden_states=True` ist `hidden_states[l + 1]` bitgleich mit dem Hook-Ausgang von Block l, aber
`hidden_states[-1]` ist **nach** `final_layer_norm` — deshalb Block-Hooks (einheitlich pre-LN) und nicht
`output_hidden_states`. Residualnormen von pythia-160m steigen über die Tiefe (Mittel 8.6 → 23 → 47 im letzten Block);
der Probe standardisiert je Dimension. CPU hier: 232 Tok/s (160m fp32, geteilte CPU). Der Hub ist erreichbar
(`pythia-160m@step0`: HTTP 200), der Cache hält 160m@step143000 und 2.8b@step0/step143000.

---

## 3. Festlegungen (alle offenen Wahlen)

| # | Wahl | Entscheidung |
|---|---|---|
| F1 | Datensätze | alle 6 915; Reihenfolge in allen Arrays = `dataset_id` aufsteigend (String-Sortierung) |
| F2 | Zeilen je Datensatz | `N_ROWS = 16`; Index `np.sort(np.random.default_rng([ROW_SEED, int(dataset_id.split("-")[-1])]).choice(n, size=min(16, n), replace=False))`, `ROW_SEED = 0`. Hängt nur von id und Seed ab → identisch für alle Modelle und für die Baselines |
| F3 | Zeilenformat | `prompt + "\n"` und `response` **getrennt** tokenisiert (`add_special_tokens=False`, Pythia hat kein BOS) und konkateniert → exakte Antwortgrenze `resp_start = len(prompt_ids)`; keine Chat-Vorlage |
| F4 | Truncation | `MAX_LENGTH = 512`: ist `len(p) + len(r) > 512`, dann Prompt auf die **letzten** `PROMPT_CAP = 256` Tokens, Antwort auf die **ersten** `512 - len(p)` Tokens. Schnittrate 2.98 % (2.3), im Manifest und Report |
| F5 | 0-Antwort-Zeile | bleibt im Last-Token-Pooling, fehlt im Antwort-Mittel (`n_resp_rows` je Datensatz zählt); hat ein Datensatz 0 solche Zeilen, ist sein Antwort-Mittel = Last-Token-Mittel und wird markiert (tritt im echten Datensatz nicht auf) |
| F6 | Hook-Ort | Forward-Hook auf jedem Block `models.decoder_blocks(model)[l]`, Residualstrom **nach** Block l, vor `final_layer_norm`; Pooling im Hook in fp32 auf der CPU |
| F7 | Layer-Auswahl | `select_layers`: alle L Layer, wenn `N · L · d · 2 · 2 Byte ≤ 1.5 GB`, sonst jeder 2. + letzter (`sorted(set(range(0, L, 2)) ∪ {L-1})`). 2.8b: 17 von 32 Layern (0, 2, …, 30, 31) = 1.20 GB. Notebook-Override `LAYER_STRIDE` je Modell |
| F8 | Numerik | `models.load_model(hf_id, revision, device, dtype, attn_impl)` mit `config.dtype_policy` (160m/410m fp32+eager überall, ab 1b fp16+sdpa auf GPU); Pooling-Summen fp32, Speicher fp16 |
| F9 | Batching | `extract.plan_batches` (absteigende Länge, Token-Budget), rechts gepaddet (`extract._pad_batch`); GPU `batch_size 64`, `max_tokens_per_batch 16384`; CPU 16 / 4096 |
| F10 | Feature-Datei | eine unkomprimierte npz je (Modell, Revision), **je Layer ein Key** (`last_L{ll}`, `resp_L{ll}`), damit Worker nur ihren Layer laden; Sidecar-Manifest JSON |
| F11 | Idempotenz | Extraktion überspringt eine Datei, deren Manifest zu den Einstellungen passt (sonst Fehler, `--force` baut neu); keine Teil-Wiederaufnahme innerhalb einer (Modell, Revision) (≤ 45 min Verlust bei Sessionabbruch) |
| F12 | Probe-Merkmale | `StandardScaler` → `PCA(n_components = min(N_PCS, n_train − 1, d))`, `N_PCS = 256`, beides nur auf dem Trainingsteil gefittet → gleiche Kapazität für d = 768 … 2 560 (Note zu pythia-1b in `config.MODELS`) |
| F13 | Modelle | `LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")` (multinomial bzw. binär); `RidgeCV(alphas=np.logspace(-2, 4, 13))` |
| F14 | Zufälliger Split | `StratifiedKFold(5, shuffle=True, random_state=seed)` nach `behavior_id` über alle 6 915; Seed = `--seed` (Default 42) |
| F15 | Methoden-Split | Gruppen A = {DE, PT}, B = {PT-noterse, DPG, LLS} (LLS: 1 Datensatz, Quelle Tulu-2.5 wie DPG); **beschränkt auf die 8 Verhalten mit Datensätzen in beiden Gruppen** (5 809), zwei Richtungen `A2B` (Train A, Test B) und `B2A`; alle Aufgaben nutzen dieselbe Beschränkung; gepoolte Metrik über die konkatenierten Out-of-fold-Vorhersagen beider Richtungen (jeder der 5 809 genau einmal Test) |
| F16 | Zielgrößen | `behavior` (18 bzw. 8 Klassen), `miss_sol` (`mc_missed_sol`, NaN-Zeilen ausgeschlossen), `elicitation` (`qwen3_1_7b_opus`), `transfer` (`qwen3_4b_gemma`); Ridge-Ziele zusätzlich als Variante `within` = Ziel minus Trainings-Mittel seines Verhaltens (Test-Zeilen mit ihrem Verhaltenslabel aus den Metadaten demeant) |
| F17 | Metriken | `behavior`: balanced Accuracy (primär), Accuracy, Macro-AUC (OvR, Punktwert); `miss_sol`: ROC-AUC (primär), balanced Accuracy, Accuracy; Ridge: R² (primär), Spearman. Chance: bal_acc 1/K, AUC 0.5, R² 0 |
| F18 | Konfidenzintervalle | Perzentil-Bootstrap über die gepoolten Out-of-fold-Vorhersagen (Resampling der Testdatensätze), `BOOT = 1000`, 95 %; gepaarte Differenzen als gepaarter Bootstrap über dieselben Datensätze mit `p_boot = 2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0))`. **Gepaart heißt dieselbe Zeilenstichprobe:** `key_table` prüft `rows_sha256` und die dataset ids der Feature-Datei gegen `baselines_probe.json` / `baselines_oof.npz` (und gegen die step0-Datei); bei Abweichung (z. B. `--n_rows` nur bei `extract`) werden `d_tfidf`, `d_step0`, `d vs Meta` und die Ridge-Differenzen NaN, das Verdikt „nicht prüfbar“, der Report trägt eine Warnung, und `--stage baselines` warnt schon vorab bei abweichendem `--n_rows` |
| F19 | Bester Layer | je (Modell, Revision, Aufgabe): das (Layer, Pooling) mit der höchsten primären Metrik im **zufälligen** Split; die Methoden-Split-Werte werden an diesem Layer berichtet (Auswahl und Bericht auf verschiedenen Kriterien), zusätzlich die komplette Layer-Kurve beider Splits |
| F20 | Baselines | (a) TF-IDF(1-2-Gramme, `min_df=5`, `max_features=60000`, `sublinear_tf`) → `TruncatedSVD(128)` → `Normalizer` → F12-Kopf, **je Split neu gefittet** auf den konkatenierten 16 Zeilen (`"\n".join(p + "\n" + r)`); die SVD-Breite ist die Konstante `TFIDF_SVD = 128` und folgt **nicht** `N_PCS` (Breite der Aktivierungs-PCA), so dass Probe-Baseline `tfidf_svd128`, TF-IDF-Rangtest (k ≤ 128) und TF-IDF-Winkel dieselbe Einbettung teilen; (b) 10-dim Stilvektor (5.8) auf denselben Zeilen; (c) step0 = die step0-Datei desselben Modells; (d) Metadaten (5.8), nur für `miss_sol` |
| F21 | Winkel | am besten (Layer, Pooling) der Aufgabe `behavior`, Merkmale standardisiert (volle Dimension d, keine PCA): Mass-Mean-Richtungen `mean(Klasse) − mean(Rest)` (primär) und OvR-Logistik-Koeffizienten (sekundär); 18×18-Winkelmatrix über `analysis.direction_angles.angle_deg`; Nullen: Label-Permutation (`PERMS = 200`, Klassengrößen erhalten; p zweiseitig `(c + 1) / (P + 1)`, Holm über 153 Paare) und N1-Referenz `analysis.meanfield.n1_angle_quantile([0.025, 0.5, 0.975], d)` mit d = hidden (TF-IDF: d = 128). **Die Richtungen sind in-sample** (alle Datensätze, an einem vom Probe auf denselben Datensätzen gewählten Layer); die Permutations-Null trägt dieselbe In-Sample-Struktur, die OvR-Winkel haben keine Null (deskriptiv; gefittetes Rauschen zieht sie zu 90°). Held-out-Kontrolle: Mass-Mean-Richtungen je zufälligem Fold nur auf dem Trainingsteil (`fold_angle_matrices`, `massmean_folds` in `_angles.npz`; Spearman je Fold zur In-Sample-Matrix und SD je Paar in `_angles.json` / Report / `key_table.csv`) |
| F22 | Rangtest | am besten (Layer, Pooling): zufälliger Split, `k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}` PCs (Train-gefittet), bal_acc je k; `k95` = kleinstes k mit bal_acc ≥ 0.95 · bal_acc(k_max); dazu effektiver Rang (Participation Ratio) der 18 Richtungen; TF-IDF bis k = 128 |
| F23 | Parallelität | `joblib.Parallel(n_jobs)` über (Layer, Pooling)-Merkmalssätze; jeder Worker lädt nur seinen Layer-Key |
| F24 | Pfade | `VH_ROOT` = `MyDrive/ValueHeadsPythia/dsse`; Features `{VH_FEATURES}/{model}_{revision}.npz` mit `VH_FEATURES = {VH_ROOT}/features` (im Notebook explizit auf Drive gesetzt, nicht `/content/features`); Ergebnisse `config.paths()["dsse"] = {VH_ROOT}/results/dsse/`; Abbildungen `{results}/dsse/figures/`; Rohdaten `/content/dsse_raw` (Colab) bzw. `VH_DSSE_RAW` |
| F25 | Download | `urllib.request` Streaming nach `<datei>.part`, dann `os.replace`; Revision `DSSE_REVISION = "main"`; nach dem Download `verify_raw` (Bytes **und** sha256 aller 20 Dateien); bei Abweichung Abbruch mit Hinweis `DSSE_REVISION = DSSE_PINNED_COMMIT` zu setzen |
| F26 | HF-Cache | im Notebook nach jeder (Modell, Revision)-Extraktion `hf_cache.cleanup_hf_revision(hf_id, revision)` (Importe aus bestehenden Modulen sind erlaubt, Änderungen nicht) |
| F27 | Smoke | `VH_NB_SMOKE=1`: `make_synthetic_raw` (40 Datensätze, 4 Verhalten × 2 Methoden × 5, 24 Zeilen je Satz), `MODELS=["pythia-160m"]`, `REVISIONS=["step0"]`, `N_ROWS=8`, `BOOT=200`, `PERMS=50`; `VH_NB_TINY=1`: zufälliges 3-Layer-GPT-NeoX (`hidden 64`) mit Byte-Tokenizer statt Download, Dateiname `tiny-gptneox_{revision}`; im SMOKE-Modus wird `TINY` auch automatisch gesetzt, wenn `load_tokenizer("EleutherAI/pythia-160m")` fehlschlägt (kein Netz, kein Cache) |
| F28 | Seeds | Zeilenstichprobe `ROW_SEED = 0` (fest, nicht per CLI-Seed veränderbar, damit alle Modelle dieselben Zeilen sehen); Probes/Bootstrap/Permutationen `--seed` (Default 42, `cli.add_common_args`) |
| F29 | Report-Verdikt | „Repräsentation jenseits des Lexikalischen“ nur, wenn im Methoden-Split die gepaarte Differenz bal_acc(bester trainierter Layer) − bal_acc(TF-IDF) **und** − bal_acc(step0 bester Layer) beide ein 95 %-CI > 0 haben; sonst „kein Nachweis“ |
| F30 | Version | `VERSION = "dsse-sideB-v1"` in jedem Manifest, CSV und Report |

---

## 4. Dateien

| Datei | Inhalt |
|---|---|
| `valueheads_pythia/dsse_probe.py` | Konstanten, Rohdaten (Download / Prüfung / Laden / Stichprobe), Tokenisierung, `ResidualCollector`, Extraktion + npz, Splits, Probes, Baselines, Winkel, Rangtest, Report, Synthetik, Tiny-Modell (Abschnitt 5) |
| `valueheads_pythia/commands/dsse_probe.py` | Kommando `dsse-probe` mit `--stage` (Abschnitt 6) |
| `valueheads_pythia/cli.py` | **eine Zeile** in `COMMANDS` (Abschnitt 7) |
| `valueheads_pythia/config.py` | **eine Zeile** in `paths()` (Abschnitt 7) |
| `scripts/make_dsse_notebook.py` → `notebooks/runs/dsse-sideB.ipynb` | Generator nach dem Muster von `scripts/make_overlap_notebook.py` (Abschnitt 8) |
| `tests/test_dsse_probe.py` | Abschnitt 9 |
| `README.md` | Abschnitt „DSSE Seite B“ + Zeilen in Projektstruktur / docs (Abschnitt 7) |

Abhängigkeiten: nur `requirements.txt` (sklearn, scipy, pandas, matplotlib, joblib, tqdm, huggingface_hub sind da).
`lzma` ist Standardbibliothek.

---

## 5. Modul `valueheads_pythia/dsse_probe.py`

Modul-Docstring (englisch) nennt Frage, Datenquelle (CC-BY-NC, Transluce/DSSE, arXiv 2611.06324), die Datenschutz-Regel
und das Dateilayout. Importe: `config`, `models` (`load_model`, `decoder_blocks`, `model_geometry`, `set_seed`),
`extract` (`plan_batches`, `_pad_batch`), `analysis.direction_angles.angle_deg`, `analysis.meanfield.n1_angle_quantile`,
`analysis.heatmaps.CATEGORICAL`. torch wird nur in 5.4 / 5.5 / 5.13 gebraucht und dort lokal importiert, damit die
CPU-Stufen `probe` / `baselines` / `report` ohne Modellstack laufen (wie `analysis/meanfield.py`).

### 5.1 Konstanten

```python
VERSION = "dsse-sideB-v1"
DSSE_REPO = "Transluce/DSSE"
DSSE_URL = "https://huggingface.co/datasets/Transluce/DSSE/resolve/{revision}/{path}"
DSSE_REVISION = "main"
DSSE_PINNED_COMMIT = "b5ce0b68965cfedf2e67f4814d8cc6ba765790b3"      # `main` on 2026-11-06
DSSE_FILES = {                                                      # path on the Hub -> (bytes, sha256); Anhang B
    "scores.csv": (1097413, "e5480408b17af757f9ce325a61ae6d177479db4aae9f38fbe70e5e65921c10e1"),
    "behaviors.json": (576892, "8c67900c365e3fb5c2848b3c77318a9693b7ba342c03f171638e13ff74ee8d9b"),
    "datasets/acting_distressed.jsonl.xz": (19480696, "828f4ab7…"), ...   # 18 Dateien, volle Hashes aus Anhang B
}
BEHAVIORS = ["acting_distressed", "acting_flirtatious", "ai_accelerationism", "anti_oversight", "climate_denial",
             "cutting_instructions", "drunk_driving", "flat_earth", "incorrect_answer", "nineteenth_century",
             "power_seeking", "prefers_academia", "pro_china", "pro_trisolarans", "resisting_shutdown",
             "suggests_crime", "threatening_user", "validating_delusion"]
METHODS = ["DE", "DPG", "LLS", "PT", "PT-noterse"]
METHOD_GROUPS = {"A": ("DE", "PT"), "B": ("PT-noterse", "DPG", "LLS")}
SHARED_BEHAVIORS = ["acting_distressed", "cutting_instructions", "drunk_driving", "nineteenth_century",
                    "prefers_academia", "pro_trisolarans", "resisting_shutdown", "threatening_user"]
SCORE_COLUMNS = [...]                                               # die 20 Spalten aus 2.1, in dieser Reihenfolge
TASKS = {"behavior": ("behavior_id", "multiclass"), "miss_sol": ("mc_missed_sol", "binary"),
         "elicitation": ("qwen3_1_7b_opus", "ridge"), "transfer": ("qwen3_4b_gemma", "ridge")}
WITHIN_TASKS = ("elicitation", "transfer")
METADATA_COLUMNS = ["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "llama_3_2_1b_gemma", "llama_3_2_3b_gemma", "qwen3_4b_gemma",
                    "qwen3_8b_gemma", "qwen3_14b_gemma", "naturalness_elo", "naturalness_elo_se",
                    "naturalness_n_comparisons", "n_rows"]        # + one-hot(behavior_id) + one-hot(method); keine Auditor-Spalten
STYLE_NAMES = ["n_rows", "prompt_words_mean", "resp_words_mean", "resp_words_sd", "resp_words_median",
               "excl_per_kchar", "quest_per_kchar", "upper_share", "newlines_per_row", "type_token"]
N_ROWS, ROW_SEED, MAX_LENGTH, PROMPT_CAP = 16, 0, 512, 256
FEATURE_CAP_BYTES = 1.5e9
N_PCS, FOLDS, BOOT, PERMS = 256, 5, 1000, 200
RANK_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TFIDF_SVD = 128
POOLINGS = ("last", "resp_mean")
PARAM_SCALE = {"pythia-160m": 1.0, "pythia-410m": 2.5, "pythia-1b": 6.2, "pythia-1.4b": 8.8, "pythia-2.8b": 17.5}  # README Laufzeiten
SMOKE_MARKER = "zqxv"                                               # Pseudo-Silbe in jeder synthetischen Zeile (Leck-Test 9.17)
```

### 5.2 Rohdaten

```python
def raw_file(raw_dir, path):
    """Local path of a Hub path: raw_dir/datasets/x.jsonl.xz, or the flat spelling raw_dir/datasets_x.jsonl.xz if that
    exists instead (the scratchpad download); scores.csv / behaviors.json unchanged."""

def file_sha256(path, chunk=1 << 20): ...

def verify_raw(raw_dir, files=None, strict=True):
    """{path: {"bytes": int, "sha256": str, "ok": bool, "missing": bool}} for every entry of `files` (default DSSE_FILES).
    strict: raise FileNotFoundError for a missing file and ValueError listing every mismatch (bytes or sha256)."""

def download_raw(raw_dir, revision=DSSE_REVISION, files=None, token=None, say=print, timeout=120, retries=1):
    """Downloads every entry of `files` from DSSE_URL into raw_dir (HF layout), skipping files that already verify.
    urllib.request with a 'Authorization: Bearer <token>' header when `token` (env HF_TOKEN) is set, streamed in 1 MB
    chunks into <path>.part, then os.replace; socket timeout `timeout` seconds (a stalled Hub connection fails instead of
    hanging), `retries` further attempts per file after a network failure (DOWNLOAD_ERRORS; an HTTP 4xx such as a wrong
    revision is not retried). Prints only file names, sizes and seconds. Ends with verify_raw(strict=True) and returns its
    dict; on a mismatch the error text says: the Hub revision moved, set DSSE_REVISION = DSSE_PINNED_COMMIT."""

def load_scores(raw_dir):
    """DataFrame indexed by dataset_id (sorted), columns SCORE_COLUMNS; numeric columns float with NaN for empty cells,
    behavior_id / method str. Raises if a column is missing or a behavior / method is unknown."""

def load_behaviors(raw_dir):
    """Sorted behavior_ids of behaviors.json (18; the synthetic set has fewer)."""

def iter_datasets(raw_dir, behaviors):
    """Yields {"dataset_id", "behavior_id", "method", "rows"} per JSON line of every datasets file, file order. Never prints."""

def sample_rows(dataset_id, n_available, n_rows=N_ROWS, seed=ROW_SEED):
    """Sorted row indices: np.random.default_rng([seed, int(dataset_id.split('-')[-1])]).choice(n_available, min(n_rows,
    n_available), replace=False). Depends on the id and the seed only (identical across models and baselines)."""

@dataclass
class RowTable:
    dataset_ids: np.ndarray       # (N,) str, sorted
    behavior: np.ndarray          # (N,) str
    method: np.ndarray            # (N,) str
    n_rows_used: np.ndarray       # (N,) int
    rows: list                    # [(dataset_index, prompt, response), ...] in dataset order; text stays in memory only
    n_rows: int
    seed: int

def build_rows(raw_dir, n_rows=N_ROWS, seed=ROW_SEED, behaviors=None):
    """RowTable over every dataset (sorted ids); raises if a dataset_id of the files is missing in scores.csv or vice versa."""

def docs_per_dataset(rows: RowTable):
    """["\n".join(p + "\n" + r for the sampled rows of dataset i)] (TF-IDF baseline input); never printed."""
```

### 5.3 Zeilenstichprobe und Tokenisierung

```python
def encode_row(tokenizer, prompt, response, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP):
    """(ids, resp_start, cut_prompt, cut_response): prompt + '\n' and response tokenized separately
    (add_special_tokens=False) and concatenated; resp_start = len(prompt ids) after the cut. F4: when the sum exceeds
    max_length, the prompt keeps its last `prompt_cap` tokens and the response its first `max_length - len(prompt)` tokens.
    Raises if the prompt part tokenizes to zero tokens (cannot happen: '\n' is a token)."""

@dataclass
class TokenizedRows:
    ids: list                     # list of int lists
    resp_start: np.ndarray        # (R,) int32
    lengths: np.ndarray           # (R,) int32
    stats: dict                   # n_rows, n_tokens, n_cut, n_cut_prompt, n_cut_response, n_zero_response, max_len, mean_len, median_len

def tokenize_rows(tokenizer, rows: RowTable, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP, chunk=2000): ...

def token_budget(raw_dir, tokenizer=None, n_rows=N_ROWS, seed=ROW_SEED, max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP):
    """TokenizedRows.stats for the sample (tokenizer default: models.load_tokenizer('EleutherAI/pythia-160m')[0]).
    Used by the notebook's budget print; ~60 s on the real data."""
```

### 5.4 `ResidualCollector`

```python
class ResidualCollector:
    """Pooled residual stream after every selected decoder block for a right-padded batch (hooks on
    models.decoder_blocks(model)[l]; the block output is the residual stream before final_layer_norm; transformers 5
    returns a tensor, older versions a tuple -> output[0]). Pooling happens inside the hook in float32 on the CPU:
    (a) the last valid token (attention_mask.sum(1) - 1), (b) the mean over positions [resp_start, length)."""

    def __init__(self, model, layers=None):
        self.model, self.layers = model, (list(range(num_layers)) if layers is None else sorted(layers))
        self.hidden = model.config.hidden_size
        self._last = self._resp = None       # set by collect()

    def install(self) / remove(self) / __enter__ / __exit__      # like hooks.HeadHooks

    @torch.no_grad()
    def collect(self, input_ids, attention_mask, resp_start):
        """-> (last (B, L_sel, d) float32 CPU, resp_mean (B, L_sel, d) float32 CPU, resp_count (B,) int64).
        Checks right padding (attention_mask[:, 0] == 1 everywhere, as hooks.ActivationCollector), 0 <= resp_start <= length;
        rows with resp_count == 0 get resp_mean = 0 (the caller excludes them, F5). Runs model(..., use_cache=False)."""
```

Hook-Implementierung: `rows = arange(B)`, `last = mask.sum(1) - 1`; `last_vec = out[rows, last]`; Antwort-Maske
`pos >= resp_start[:, None] & pos < length[:, None]` → `resp_sum = (out * m[..., None]).sum(1)`, `resp_mean = resp_sum / clamp(count, 1)`.
Alles `.float()` vor dem Summieren (fp16-Modelle), danach `.cpu()`.

### 5.5 Extraktion und Feature-Datei

```python
def select_layers(num_layers, hidden, n_datasets, cap_bytes=FEATURE_CAP_BYTES, stride="auto"):
    """F7. stride 'auto': all layers if n_datasets * num_layers * hidden * 2 poolings * 2 bytes <= cap_bytes, else every
    2nd + last; stride 1 / 2: forced. Returns sorted ints."""

def feature_path(p, model, revision):
    """{p['features']}/{config.model_short_name(model)}_{revision or 'main'}.npz"""

def manifest_path(npz_path):   # npz_path[:-4] + ".manifest.json"

def extract_dsse(model, tokenizer, rows: RowTable, out_path, layers=None, batch_size=64, max_tokens_per_batch=16384,
                 max_length=MAX_LENGTH, prompt_cap=PROMPT_CAP, stage_dir=None, extra_manifest=None, show_progress=True,
                 calibrate_after=2000, say=print):
    """Per-dataset means of the pooled residual stream; writes out_path (npz) + manifest sidecar; returns the manifest.
    Accumulators sum_last / sum_resp (N, L_sel, d) float32, n_last / n_resp (N,) via np.add.at over the batch's dataset
    indices. Batches from extract.plan_batches(lengths, batch_size, max_tokens_per_batch) and extract._pad_batch.
    After `calibrate_after` rows prints tokens/s and the ETA of this file (like extract.extract_features). Writes to
    (stage_dir or dirname(out_path))/<name>.tmp.npz with np.savez (uncompressed), then shutil.move / os.replace to
    out_path (Drive-safe), then the sidecar. Never prints text."""
```

Keys der npz (`np.savez`, unkomprimiert; `np.load(path)[key]` lädt nur diesen Key):

| Key | Shape / dtype | Inhalt |
|---|---|---|
| `last_L{ll:02d}` je Layer in `layers` | (N, d) float16 | Mittel über die Zeilen des Last-Token-Vektors nach Block ll |
| `resp_L{ll:02d}` | (N, d) float16 | Mittel über die Zeilen des Antwort-Mittels (F5) |
| `dataset_ids` | (N,) `<U11` | sortiert |
| `layers` | (L_sel,) int16 | gespeicherte Layer |
| `n_rows_used` | (N,) int16 | Zeilen je Datensatz (≤ N_ROWS) |
| `n_resp_rows` | (N,) int16 | Zeilen mit ≥ 1 Antwort-Token |
| `manifest` | 0-d str | JSON wie das Sidecar |

Manifest (Sidecar und Key): `version, model, hf_id, revision, commit_sha, tokenizer_revision, dtype, attn_impl, device,
transformers, torch, n_datasets, n_rows, row_seed, max_length, prompt_cap, layers, num_layers, hidden, batch_size,
max_tokens_per_batch, token_stats (= TokenizedRows.stats inkl. n_cut / cut_rate), n_resp_zero_datasets, seconds,
tokens_per_second, created, finished, tiny (bool), rows_sha256` (sha256 über `dataset_id|row_index` aller Stichprobenzeilen —
kein Text — zur Identität der Stichprobe zwischen Modellen und Baselines).

```python
def feature_done(path, expect):
    """True if path and its sidecar exist and the manifest matches `expect` on version, n_rows, row_seed, max_length,
    prompt_cap, layers, hf_id, revision, rows_sha256 (when given); False when absent; RuntimeError on a mismatch
    (message names the keys; CLI: --force rebuilds)."""

def load_feature_meta(path):   # dict(dataset_ids, layers, n_rows_used, n_resp_rows, manifest)
def load_feature_layer(path, layer, pooling):   # float32 (N, d) from key f"{'last' if pooling == 'last' else 'resp'}_L{layer:02d}"
```

### 5.6 Splits

```python
@dataclass
class Split:
    kind: str          # "random" | "method"
    name: str          # "f0".."f4" | "A2B" | "B2A"
    train: np.ndarray  # indices into the dataset order
    test: np.ndarray

def random_splits(behavior, n_splits=FOLDS, seed=42):
    """StratifiedKFold(n_splits, shuffle=True, random_state=seed) on behavior; F14."""

def method_splits(behavior, method, groups=METHOD_GROUPS, shared=None):
    """F15. shared = behaviors with >= 1 dataset in both groups (SHARED_BEHAVIORS for the real data, computed for the
    synthetic set); A2B: train = group A & shared, test = group B & shared; B2A reversed. Raises if a group is empty."""

def task_target(scores, task, variant="raw"):
    """(y, valid_mask): behavior -> integer codes over BEHAVIORS order (or the file's sorted behaviors);
    miss_sol -> float 0/1 with NaN rows invalid; ridge targets float. variant 'within' is applied inside fit_eval
    (needs the train means), not here."""
```

### 5.7 Probes, Metriken, Bootstrap

```python
def make_estimator(task, n_pcs, n_train, d, seed):
    """Pipeline(StandardScaler(), PCA(min(n_pcs, n_train - 1, d), random_state=seed), head) with head =
    LogisticRegression(C=1.0, max_iter=500, class_weight='balanced') for multiclass / binary, RidgeCV(np.logspace(-2, 4, 13))
    for ridge (F12, F13). n_pcs=None skips the PCA (full-d fits of 5.9)."""

@dataclass
class OOF:
    pred: np.ndarray      # (N,) float32: class code | 0/1 | value; NaN where not tested
    score: np.ndarray     # (N,) float32: P(class 1) for binary, else NaN
    proba: np.ndarray     # (N, K) float32 for multiclass when keep_proba else None
    tested: np.ndarray    # (N,) bool
    per_split: list       # [{"name", "n_train", "n_test", metrics...}]

def fit_eval(X, y, valid, splits, task, behavior=None, variant="raw", n_pcs=N_PCS, seed=42, keep_proba=False):
    """Out-of-fold predictions over `splits` (train / test masked by `valid`). variant 'within': y_train is demeaned per
    behavior with train means, the test target with the same train means (behavior from the metadata). Multiclass
    classes = those present in the training part (8 in the method split)."""

PRIMARY = {"multiclass": "bal_acc", "binary": "auc", "ridge": "r2"}

def metrics(kind, y, oof_pred, oof_score, proba=None, tested=None):
    """multiclass: bal_acc, acc, auc_macro (OvR over the classes present in the tested rows; NaN without proba);
    binary: auc, bal_acc, acc (threshold 0.5 on score); ridge: r2, spearman. Only tested rows."""

def bootstrap_ci(kind, y, pred, score, B=BOOT, seed=0, alpha=0.05):
    """Percentile CI of the primary metric over resampled tested rows (F18)."""

def paired_bootstrap(kind, y, pred_a, score_a, pred_b, score_b, B=BOOT, seed=0):
    """{'diff', 'lo', 'hi', 'p_boot'} of metric(a) - metric(b) over the SAME tested rows (intersection); F18."""

def tfidf_svd_estimator(task, n_components=TFIDF_SVD, seed=42):
    """Pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=60000, sublinear_tf=True, dtype=np.float32),
    TruncatedSVD(n_components, random_state=seed), Normalizer(), StandardScaler(), head). Input: docs_per_dataset."""
```

`fit_eval` akzeptiert als `X` entweder ein Array (Aktivierungen, Stil, Metadaten) oder eine Liste von Strings (TF-IDF,
dann mit `tfidf_svd_estimator`); SVD-Komponenten bei Strings: `min(TFIDF_SVD, n_train − 1, n_features − 1)` — die Breite
128 der lexikalischen Einbettung ist unabhängig von `n_pcs` (`project_splits`), damit Baseline, Rangtest und Winkel dieselbe
Einbettung verwenden.

### 5.8 Baselines

```python
def style_features(rows: RowTable):
    """(N, 10) float64 in STYLE_NAMES order, computed on the sampled rows only (identical rows as the activations):
    n_rows_used, mean prompt words, mean / sd / median response words, '!' and '?' per 1000 response chars, upper-case
    share of response chars, newlines per row, mean type-token ratio of responses. No text leaves the function."""

def metadata_features(scores, behaviors):
    """(X (N, m), names): METADATA_COLUMNS (NaN -> column median) + one-hot(behavior_id over `behaviors`) + one-hot(method
    over METHODS). Auditor columns (oe_surprisingness, mc_surprisingness, mc_missed_*) are deliberately excluded: they
    measure the same auditor outcome. Constant columns inside a method-split training part are harmless (StandardScaler
    maps them to 0)."""
```

Baselines laufen **einmal je Wurzelverzeichnis** (modellunabhängig) in `run_baselines` (5.11) und schreiben
`baselines_probe.csv` / `baselines_oof.npz` mit `features ∈ {tfidf_svd128, style10, metadata}`; `metadata` nur für
`miss_sol`. Baseline (c) ist die `step0`-Datei des jeweiligen Modells (normale Probe-Ausgabe).

### 5.9 Winkel zwischen den Verhaltensrichtungen

```python
def standardize(X):            # (X - mean) / std per column, std floor 1e-6; returns float64

def mass_mean_directions(Xs, y, classes):
    """(K, d): mean(Xs[y == k]) - mean(Xs[y != k]) (standardized space; the repo's DoM analogue, directions.py)."""

def ovr_directions(Xs, y, classes, seed=42, C=1.0, max_iter=500):
    """(K, d): coef_ of one balanced binary LogisticRegression per class on the full standardized features."""

def angle_matrix(dirs):        # (K, K) float64 via analysis.direction_angles.angle_deg, diagonal 0

def angle_permutation_null(Xs, y, classes, perms=PERMS, seed=0):
    """Label permutations (class sizes kept) -> dict(mean, q025, q975, p_two) each (K, K); p_two = (c + 1) / (P + 1) with
    c = #perms whose angle is at least as extreme on the observed side, times 2, capped at 1. Holm over the K(K-1)/2 pairs
    -> 'aligned' (observed < null, Holm p < 0.05) / 'opposed' (observed > null) flags."""

def n1_reference(d):           # {"q025", "q500", "q975"} = analysis.meanfield.n1_angle_quantile([0.025, 0.5, 0.975], d)

def effective_rank(dirs):      # participation ratio (sum s^2)^2 / sum s^4 of the singular values of the row-centred (K, d)

def linkage_merges(angles, classes):
    """[(height, members_left, members_right)] of scipy average linkage on squareform(angles) (super-clusters, as in the
    lexical analysis). Report shows the merge order."""

def fold_angle_matrices(X, y, classes, splits):
    """(F, K, K): mass-mean angle matrix per split, directions fitted on the TRAINING rows of that split only (features
    standardized on those rows). A held-out check of the in-sample matrix of angle_analysis."""

def angle_analysis(X, y, classes, d, perms=PERMS, seed=0, ref_matrices=None, n_jobs=1, splits=None):
    """Everything above for one (layer, pooling): returns dict with 'massmean' (K, K), 'ovr' (K, K), null dicts, n1,
    effective ranks, per-behavior nearest (3 smallest angles) and farthest (3 largest), merges, and Spearman of the upper
    triangle against each matrix in ref_matrices ({'step0': ..., 'tfidf': ...}). Directions are fitted IN-SAMPLE (all rows,
    full d, probe-selected layer); with `splits` also 'massmean_folds' (F, K, K), 'fold_spearman', 'fold_pair_sd_mean',
    'fold_mean_angle' (held-out mass-mean directions per random fold)."""
```

Die Richtungen werden **in-sample** gefittet (F21): alle Datensätze, volle Dimension, an dem (Layer, Pooling), das der Probe
auf denselben Datensätzen gewählt hat. Die Permutations-Null trägt dieselbe In-Sample-Struktur und bleibt der primäre
Vergleich; die OvR-Winkel (18 Logistik-Fits bei N = 6 915, d ≤ 2 560) haben keine Null und werden nur deskriptiv berichtet
(gefittetes Rauschen zieht sie zur isotropen 90°-Referenz). Als Held-out-Kontrolle werden die Mass-Mean-Richtungen je
zufälligem Fold nur auf dem Trainingsteil neu gefittet; Report und `key_table.csv` nennen die Spearman-Korrelation jeder
Fold-Matrix zur In-Sample-Matrix (min / Mittel) und die mittlere Fold-SD je Paar.

Die Mass-Mean-Richtungen haben mit gewichteter Summe 0 über die Klassen einen eingebauten negativen Kosinus
≈ −1/(K−1) (K = 18: ~93.4°, bei d = 768 das 1.6-fache der N1-Streuung). Die **Permutations-Null trägt dieselbe
Nebenbedingung und dieselbe Kovarianz** und ist deshalb primär; N1 ist nur die Referenzlinie in den Abbildungen.

### 5.10 Rangtest

```python
def rank_test(X, y, valid, splits, ks=RANK_KS, seed=42, boot=BOOT):
    """DataFrame(k, bal_acc, ci_lo, ci_hi, n_test): fit_eval with n_pcs=k per k (k <= min(n_train - 1, d)); attrs k95 =
    smallest k with bal_acc >= 0.95 * bal_acc(max k). Random split only (F22)."""
```

### 5.11 Lauf-Funktionen und Ausgabedateien je (Modell, Revision)

```python
def probe_settings(n_pcs, folds, boot, perms, seed, tasks, n_jobs=None):   # dict; n_jobs is not part of the fingerprint

def best_feature_set(df, task, split="random"):
    """(layer, pooling) with the highest primary metric (rows features == 'act', direction == 'pooled') -> F19."""

def probe_feature_file(npz_path, raw_dir, out_dir, settings, n_jobs=-1, force=False, say=print):
    """Runs every task x (layer, pooling) x split on one feature file; writes
        {out_dir}/{model}_{revision}_probe.csv, _probe.json (settings fingerprint + manifest + best sets + timings),
        {out_dir}/{model}_{revision}_oof.npz, _angles.npz, _angles.csv, _angles.json, _rank.csv
    Skips when _probe.json exists with the same fingerprint (version, settings, manifest rows_sha256, scores sha256)
    unless force. joblib.Parallel(n_jobs, prefer='processes') over the (layer, pooling) sets; each job loads its own key."""

def run_baselines(raw_dir, out_dir, n_rows=N_ROWS, seed_rows=ROW_SEED, settings=None, force=False, say=print):
    """TF-IDF / style / metadata on the same sampled rows -> baselines_probe.csv, baselines_oof.npz, baselines_angles.*,
    baselines_rank.csv (TF-IDF ks <= 128). Same fingerprint logic."""
```

`{model}_{revision}_probe.csv` (long format, eine Zeile je Metrik):

| Spalte | Werte |
|---|---|
| `model`, `revision` | z. B. `pythia-410m`, `step16000`; Baselines: `model = "baseline"`, `revision = "-"` |
| `features` | `act` · `tfidf_svd128` · `style10` · `metadata` |
| `layer`, `pooling` | int / `last`·`resp_mean`; Baselines `-1` / `none` |
| `task`, `variant` | `behavior` · `miss_sol` · `elicitation` · `transfer`; `raw` · `within` |
| `split`, `direction` | `random` / `method`; `pooled` (mit CI) oder `f0`…`f4` / `A2B` / `B2A` (Punktwerte) |
| `n_pcs`, `n_train`, `n_test`, `n_classes`, `chance` | |
| `metric`, `value`, `ci_lo`, `ci_hi` | CI nur für `pooled` und die primäre Metrik, sonst NaN |
| `seed`, `n_rows`, `row_seed`, `version` | |

`{model}_{revision}_oof.npz`: `ids` (N,), je Merkmalssatz × Aufgabe × Variante × Split
`pred__{features}__{task}__{variant}__{split}__L{ll:02d}__{pooling}` (float32, NaN = nicht getestet),
`score__…` (binär), `proba__behavior__raw__{split}__L..__..` nur für das beste (Layer, Pooling).

`_angles.npz`: `behaviors`, `layer`, `pooling`, `massmean`, `ovr`, `null_mean`, `null_q025`, `null_q975`, `p_two`,
`holm_flag` (int8: −1 aligned, +1 opposed, 0), `n1_q` (3,), `d`, `dirs_massmean` (K, d) float32.
`_angles.csv`: `pair, a, b, angle_massmean, angle_ovr, null_mean, p_two, p_holm, flag`.
`_angles.json`: Zusammenfassung (Mittelwinkel, Anteil aligned / opposed, effektiver Rang, Spearman zu step0 und TF-IDF,
Nachbarn / Gegenspieler je Verhalten, Merge-Reihenfolge).
`_rank.csv`: `k, bal_acc, ci_lo, ci_hi, n_test` + `k95` in `_probe.json`.

### 5.12 Report

```python
def collect_results(out_dir, models=None, revisions=None):
    """Concatenated probe.csv frames (+ baselines) and the json sidecars found in out_dir."""

def key_table(results, oof_loader, boot=BOOT, seed=0):
    """One row per (model, revision): best (layer, pooling) per task (F19), random bal_acc, method bal_acc [CI], paired
    deltas vs step0-best and vs tfidf on the method split (F18), miss_sol AUC [CI] vs metadata, elicitation / transfer R2
    [CI] (raw and within) vs tfidf, k95, mean angle, aligned / opposed counts, verdict (F29)."""

def write_report(out_dir, results, key, figures, note=None):
    """results/dsse/report.md (German): Kopf (Version, Datum, Datensatz-Commit, n, Tokens, Schnittrate, Zeilen je Satz,
    Modelle / Revisionen, Einstellungen), Kerntabelle, Befund je Modell (F29), Layer-Kurven, Winkelabschnitt (mono / poly:
    aligned / opposed Paare, Nachbarn, Super-Cluster, Vergleich step0 / TF-IDF), Rangtest, Baselines, Abweichungen (12),
    Dateiliste. Only ids, labels, counts, metrics."""

def make_figures(out_dir, results, angles, ranks):
    """figures/layer_curves_{task}.png (panel per model, x = layer, lines per revision, solid = random, dashed = method,
    horizontal lines tfidf / style / chance), figures/angles_{model}.png (heatmaps step0 vs step143000 at the best layer
    + N1 band), figures/rank_{model}.png, figures/summary_bars.png. matplotlib Agg, CATEGORICAL palette."""
```

### 5.13 Synthetische Rohdaten und Tiny-Modell

```python
def make_synthetic_raw(out_dir, n_datasets=40, n_behaviors=4, methods=("DE", "PT-noterse"), rows_per_set=24, seed=0):
    """Writes the DSSE layout (scores.csv with SCORE_COLUMNS, behaviors.json, datasets/<b>.jsonl.xz) from pseudo-words
    (consonant-vowel syllables, seeded): behaviors syn_a..syn_d with 12 marker words each; method style DE = long
    responses (40-70 words, commas), PT-noterse = short (8-15 words, '!'); every row contains SMOKE_MARKER once.
    Planted signal: marker density per dataset -> qwen3_1_7b_opus = clip(0.1 + 0.6 * density + N(0, 0.05)), transfer =
    0.8 * opus + noise, mc_missed_sol = 1 iff density below the behavior median, naturalness ~ N(1500, 200), other columns
    filled consistently. ids synth-00001 ...; n_datasets = n_behaviors * len(methods) * 5. Returns {'behaviors', 'n'}."""

class ByteTokenizer:
    """pad 0, eos 1, bytes 2..257 (vocab 258). __call__(texts, add_special_tokens=False, **_) -> {'input_ids': [...]};
    attributes pad_token_id, eos_token_id, padding_side='right', truncation_side='left'. Tests / --tiny only."""

def tiny_model(seed=0, hidden=64, layers=3, heads=2):
    """(GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=258, hidden_size=hidden, num_hidden_layers=layers,
    num_attention_heads=heads, intermediate_size=4 * hidden, max_position_embeddings=512, rotary_pct=0.25,
    use_parallel_residual=True)), ByteTokenizer()); random init, eval, float32; model.vh_info set like models.load_model
    with model='tiny-gptneox', hf_id=None, dtype float32, attn eager, geometry from the config."""
```

---

## 6. Kommando `dsse-probe` (`valueheads_pythia/commands/dsse_probe.py`)

```
python -m valueheads_pythia.cli dsse-probe --stage {verify,download,extract,probe,baselines,report,synthetic,all}
   --raw_dir DIR            DSSE raw files (default env VH_DSSE_RAW, else {root}/dsse_raw); Colab: /content/dsse_raw
   --dsse_revision main     Hub revision for `download` (fallback: DSSE_PINNED_COMMIT)
   --model, --revision, --root, --device, --dtype, --attn_impl, --seed     (cli.add_common_args(parser, mode=False))
   --revisions step0 step2000 ...   loop for extract / probe (default: [--revision or 'main'])
   --models pythia-160m ...         report / probe: which feature files (default: every *.npz in features)
   --n_rows 16 --max_length 512 --prompt_cap 256
   --layer_stride auto|1|2          F7
   --batch_size 64 --max_tokens_per_batch 16384      (CPU default 16 / 4096 when --device resolves to cpu)
   --stage_dir DIR          local staging directory for the npz write (Colab: /content/dsse_stage)
   --n_pcs 256 --folds 5 --boot 1000 --perms 200 --n_jobs -1
   --tasks behavior miss_sol elicitation transfer     (default all)
   --tiny                   extract with tiny_model() instead of a download (file name tiny-gptneox_{revision})
   --synthetic_out DIR --n_synthetic 40 --rows_per_set 24     (stage synthetic; writes DIR, prints counts)
   --note TEXT              line at the top of report.md (e.g. SMOKE)
   --force --no_progress --quiet
```

Stufen:

| Stufe | Eingabe → Ausgabe | Druckt |
|---|---|---|
| `synthetic` | → `--synthetic_out` im DSSE-Layout | n Datensätze, Verhalten, Methoden |
| `download` | Hub → `raw_dir`; danach `verify_raw` | Dateiname, Bytes, Sekunden, OK/Mismatch |
| `verify` | `raw_dir` | Tabelle Bytes / sha256 / ok; Exit 1 bei Mismatch |
| `extract` | `raw_dir` + Modell je `--revisions` → `feature_path` | Modell-Info (`vh_info`), Layer-Auswahl + Dateigröße, Token-Statistik (n, Tokens, Schnittrate), nach 2 000 Zeilen Tok/s + ETA, am Ende Sekunden; `skip` wenn `feature_done` |
| `probe` | Feature-Dateien (+ `scores.csv`) → 5.11 | je Datei bestes (Layer, Pooling) je Aufgabe mit Wert und CI, Sekunden |
| `baselines` | `raw_dir` → `baselines_*` | je Baseline × Aufgabe primäre Metrik beider Splits |
| `report` | `results/dsse/*` → `report.md` + Abbildungen | Kerntabelle als Text |
| `all` | `download` (oder `verify`, wenn `raw_dir` schon vollständig ist) → `extract` → `baselines` → `probe` → `report`; `synthetic` ist nie Teil von `all` | wie die Einzelstufen |

Regeln: `set_seed(args.seed)`; Modell-Laden über `models.load_model(hf_id, revision=…, device, dtype, attn_impl)`
(Numerik-Policy); `--tiny` umgeht `load_model`. Exit-Codes: 0 ok, 1 Mismatch / fehlende Eingaben, Exceptions für
Programmierfehler. Idempotenz wie in 5.5 / 5.11. Kein Text wird gedruckt — auch nicht in Fehlermeldungen (nur ids).

---

## 7. Registrierung, README

`cli.py` — eine Zeile in `COMMANDS` (nach `"ablate"`):

```python
    "dsse-probe": "DSSE Seite B: Residualstrom je DSSE-Datensatz (Pythia-Checkpoints), Probes + Baselines, Winkel, Report",
```

`config.py` — eine Zeile in `paths()` (nach `"ablation"`):

```python
        "dsse": os.path.join(results, "dsse"),
```

README: Abschnitt `## DSSE Seite B: Repräsentation der Verhaltensänderung (Pythia, \`dsse-probe\`)` vor „Modus `paper` vs.
`pythia`“ mit: Frage (3 Sätze), Datenquelle + Lizenzhinweis (CC-BY-NC, kein Text in Ausgaben), die Pipeline-Schritte,
der primäre Vergleich (F29), CLI-Beispiel (`--stage all` mit `--revisions`), Notebook-Link
(`notebooks/runs/dsse-sideB.ipynb`, Colab-URL nach dem Muster des Overlap-Abschnitts), Budget (Abschnitt 11), Ergebnislayout
(`results/dsse/…`, `features/{model}_{revision}.npz`), Smoke-Rezept-Verweis auf den Generator-Kopf. Dazu `dsse_probe` /
`commands/dsse_probe` in „Projektstruktur“, `make_dsse_notebook.py` unter `scripts/`, `dsse-sideB.ipynb` unter
`notebooks/runs/`, `DSSE_SIDEB.md` unter `docs/`, und in „Attribution und Lizenzen“ eine Zeile: DSSE (Transluce,
CC BY-NC 4.0, arXiv 2611.06324; PT/PT-noterse-Prompts aus Stanford Alpaca, DPG/LLS aus Tulu 2.5) — nur Zählungen und
Metriken werden gespeichert.

---

## 8. Notebook `scripts/make_dsse_notebook.py` → `notebooks/runs/dsse-sideB.ipynb`

Generator nach `scripts/make_overlap_notebook.py`: `NAME = "dsse-sideB"`, `DRIVE_SUBDIR = "dsse"`, Hilfsfunktionen `src`,
`sub1`, `master_cell`, `md`, `code` kopieren; Laufzeit-Zelle und Install-Zelle **wörtlich** aus dem Master
(`notebooks/ValueHeadsPythia_Colab.ipynb`) mit diesen `sub1`-Patches der Install-Zelle:

1. `DRIVE_ROOT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia"` → `…/ValueHeadsPythia/dsse`
2. `DRIVE_PROJECT = f"{DRIVE_ROOT}/valueheads_pythia"` → `f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"`
3. `    VH_FEATURES = "/content/features"` → `    VH_FEATURES = f"{DRIVE_ROOT}/features"` (Features **auf Drive**, F24)
4. `for d in [P["heads"], P["directions"], P["eval"], P["judge"], P["sweep"], P["heatmaps"], P["figures_sweep"],\n          P["figures_causal"], VH_FEATURES, DRIVE_FEATURE_CACHE]:` → `for d in [P["dsse"], os.path.join(P["dsse"], "figures"), VH_FEATURES]:`

Der `run()`-Helfer kommt wie im Overlap-Generator aus der Konfigurationszelle des Masters (`QUIET_PATTERNS … run("--help")`).
Metadaten: `accelerator: GPU`, `colab.gpuType: A100`. Jede Code-Zelle wird mit `compile()` geprüft, die Datei mit
`nbformat.validate`. Zellen:

| # | Zelle | Inhalt |
|---|---|---|
| 1 | md Intro | Run-Notebook-Hinweis (vom Generator erzeugt), Frage, Ablauf 1-8, Budget (11), Datenschutzhinweis (DSSE enthält Red-Teaming-Text; nie ausgeben; vor dem Speichern *Alle Ausgaben löschen*) |
| 2 | md + code Laufzeit | Master-Zelle 1 (IN_COLAB, SMOKE, HAS_CUDA, DEVICE, TIER, HF_HOME, HF_TOKEN) |
| 3 | md + code Installation | Master-Zelle 2 gepatcht (oben) |
| 4 | md + code Download | `RAW_DIR = "/content/dsse_raw" if IN_COLAB else (os.environ.get("VH_DSSE_RAW") or f"{VH_ROOT}/dsse_raw")`; `STAGE_DIR = "/content/dsse_stage" if IN_COLAB else f"{VH_ROOT}/dsse_stage"`; SMOKE: `run("dsse-probe", "--stage", "synthetic", "--synthetic_out", RAW_DIR)` (nur wenn `scores.csv` fehlt); sonst `run("dsse-probe", "--stage", "download", "--raw_dir", RAW_DIR, "--dsse_revision", DSSE_REVISION)`; danach `verify_raw` in-process und Ausgabe von `scores.csv` Bytes + sha256-Vergleich (Hash-Check), Zählungen (`load_scores`: n je Verhalten / Methode) |
| 5 | md + code Konfiguration | `MODELS = list(config.PYTHIA_SIZES)`; `REVISIONS = ["step0", "step2000", "step16000", "step143000"]`; `N_ROWS = 16`; `LAYER_STRIDE = {"pythia-2.8b": 2}` (sonst `auto`); `N_PCS, FOLDS, BOOT, PERMS = 256, 5, 1000, 200`; `MAX_LENGTH, PROMPT_CAP = 512, 256`; `EXTRACT_BATCH = ["--batch_size", "64", "--max_tokens_per_batch", "16384"] if HAS_CUDA else []`; `CLEANUP_CACHE = IN_COLAB and not SMOKE`; `TINY = os.environ.get("VH_NB_TINY") == "1"`, im SMOKE-Modus zusätzlich `TINY = True` mit Meldung, wenn `load_tokenizer("EleutherAI/pythia-160m")` fehlschlägt (`PYTHIA_TOK` wird sonst für die Budget-Zelle wiederverwendet); SMOKE-Overrides (F27); Env-Overrides `VH_NB_MODELS`, `VH_NB_REVISIONS`, `VH_NB_N_ROWS`; Ausgabe: je Modell L/d, Numerik-Policy, Layer-Auswahl (`select_layers`) und npz-Größe in GB, Drive-Summe; dann der `run()`-Helfer + `run("--help")` |
| 6 | md + code Extraktion | Budget zuerst: `token_budget(RAW_DIR, n_rows=N_ROWS)` (Tokenizer aus `models.load_tokenizer("EleutherAI/pythia-160m")`; SMOKE/TINY: `ByteTokenizer`) → Tokens, Schnittrate; ETA-Tabelle aus `PARAM_SCALE` und README-Raten (160m ≈ 100 k Tok/s A100 → Minuten je Modell × 4). Schleife `for m in MODELS: for r in REVISIONS:` → `run("dsse-probe", "--stage", "extract", "--model", m, "--revision", r, "--raw_dir", RAW_DIR, "--n_rows", N_ROWS, "--layer_stride", LAYER_STRIDE.get(m, "auto"), "--stage_dir", STAGE_DIR, "--device", DEVICE, "--dtype", dtype, "--attn_impl", attn, *EXTRACT_BATCH, *PROGRESS, *(["--tiny"] if TINY else []))`; danach `feature_done`-Manifest lesen und `n, tokens, tok/s, cut_rate, GB` drucken; nach der **ersten** (Modell, Revision) die gemessenen Tok/s in die ETA der verbleibenden Paare einsetzen (`PARAM_SCALE[m2] / PARAM_SCALE[m1] · tokens / rate`) und drucken; `if CLEANUP_CACHE: hf_cache.cleanup_hf_revision(hf_id, r)`; idempotent (fertige Dateien werden übersprungen) |
| 7 | md + code Probes + Baselines | `run("dsse-probe", "--stage", "baselines", "--raw_dir", RAW_DIR, "--n_rows", N_ROWS, "--boot", BOOT, "--perms", PERMS, "--quiet")` einmal; dann je Datei `run("dsse-probe", "--stage", "probe", "--model", m, "--revisions", *REVISIONS, "--raw_dir", RAW_DIR, "--n_pcs", N_PCS, "--folds", FOLDS, "--boot", BOOT, "--perms", PERMS, "--n_jobs", "-1", "--quiet")`; idempotent; danach `pd.read_csv` aller `*_probe.csv` → Kurztabelle bestes (Layer, Pooling) je Modell × Revision × Aufgabe (zufällig / Methode) |
| 8 | md + code Report | `run("dsse-probe", "--stage", "report", "--note", NOTE)`; `display(Markdown(report.md bis zur Layer-Kurven-Überschrift))`; Abbildungen per `IPython.display.Image`; Größen (`results`, `features` auf Drive); Erinnerung *Alle Ausgaben löschen* |
| 9 | md Footer | wie Overlap-Notebook |

Notebook-Prosa deutsch; Zell-Kommentare (erste Zeile `# N. …`) deutsch wie im Master, Code-Kommentare englisch.
`VH_NB_SMOKE=1` schaltet in jeder Zelle nur Größen um, nie die Logik (Muster des Overlap-Notebooks).

---

## 9. Tests `tests/test_dsse_probe.py` (zusammen ≤ 90 s; Modell nur über das Session-Fixture `model_tok`)

Alle Tests arbeiten offline auf synthetischen Daten (`make_synthetic_raw` in `tmp_path`) oder auf Zufallsmatrizen; kein
Test druckt oder prüft Zeilentext. Marker `needs_model` = benutzt `model_tok` (pythia-160m step143000, fp32 CPU).

| # | Test | Prüft |
|---|---|---|
| 1 | `test_sample_rows_fixed` | gleiche id + Seed → gleiche Indizes, sortiert, ohne Duplikate, `min(n_rows, n)`; anderer Seed → andere Indizes; `n_available < n_rows` → alle |
| 2 | `test_encode_row_cut_rule` | Byte-Tokenizer: kurz → kein Schnitt, `resp_start == len(prompt) + 1`; langer Prompt (400) + kurze Antwort → Prompt auf 256 (Tail), Antwort ganz; kurzer Prompt + lange Antwort (1000) → Antwort auf `512 - len(p)` (Head); beides lang → 256 + 256; leere Antwort → `resp_start == len(ids)` |
| 3 | `test_collector_matches_hidden_states` (needs_model) | 5 synthetische ASCII-Zeilen verschiedener Länge, rechts gepaddet: `last` == `hidden_states[l + 1][row, length - 1]` und `resp_mean` == manuelles Maskenmittel über `[resp_start, length)` für alle 12 Layer (`atol 1e-5`); `resp_count` korrekt; 0-Antwort-Zeile → `resp_mean == 0`, count 0 |
| 4 | `test_collector_batch_vs_single` (needs_model) | eine Zeile allein vs. in einem gepaddeten Batch: relativer L2-Fehler ≤ `config.DTYPE_POLICY["tolerance_batched_vs_single"]` je Layer; Linkspadding → `ValueError` |
| 5 | `test_select_layers` | 160m (12, 768) → alle 12; 1.4b (24, 2048) → alle 24 (1.36 GB); 2.8b (32, 2560) → 17 Layer inkl. 31; `stride=2` erzwingt; `cap_bytes` klein → Stride 2 |
| 6 | `test_extract_synthetic_tiny` | `make_synthetic_raw(40)` → `build_rows(n_rows=8)` → `extract_dsse` mit `tiny_model()` (hidden 64, 3 Layer): Keys / Shapes / dtypes der npz, `dataset_ids` sortiert, `n_rows_used == 8`, Manifest-Felder (`token_stats`, `layers`, `rows_sha256`, `tiny`); ein Datensatz nachgerechnet: Mittel der Einzelzeilen-Vektoren (Collector direkt) == gespeichert (`atol 2e-3`, fp16); `feature_done` True, mit anderem `n_rows` → `RuntimeError` |
| 7 | `test_splits` | random: 5 Folds, Test-Mengen disjunkt, Vereinigung = alle, stratifiziert (je Fold alle Klassen); method: nur geteilte Verhalten, `A2B.train ⊂ A`, `A2B.test ⊂ B`, beide Richtungen zusammen testen jeden geteilten Datensatz genau einmal; ein Verhalten ohne Gruppe B → ausgeschlossen |
| 8 | `test_fit_eval_recovers_planted_signal` | Gauß-Features (n = 300, d = 40) mit Klassenmitteln (4 Klassen) → bal_acc > 0.9 (random); permutierte Labels → bal_acc < 0.45; Variante `within` subtrahiert Trainingsmittel je Verhalten (geprüft an einem konstruierten Ziel) |
| 9 | `test_binary_and_ridge` | planted lineares Ziel → R² > 0.8; binäres Ziel → AUC > 0.9; NaN-Zeilen in `valid` werden nie getestet |
| 10 | `test_bootstrap` | CI enthält den Punktwert; `paired_bootstrap` identischer Vorhersagen → `diff 0, lo = hi = 0, p_boot = 1`; klar bessere A → `lo > 0`, `p_boot < 0.05` |
| 11 | `test_direction_angles` | isotrope Gauß-Features + Zufallslabels (K = 4, d = 200): Mittelwinkel innerhalb `[n1 q025, n1 q975]` ± Permutations-Band, keine Holm-Flags; planted gemeinsame Richtung für zwei Klassen → ihr Winkel < `null_q025`, Flag `aligned`, gegenläufige Klassen → `opposed`; `effective_rank` von 4 orthogonalen Richtungen ≈ 4 |
| 12 | `test_rank_test` | planted 3-dim Signal in d = 50 → `k95 ≤ 4`, bal_acc monoton bis k95 (Toleranz 0.02) |
| 13 | `test_verify_raw` | Manifest aus den synthetischen Dateien (Bytes + sha256) → `ok`; ein Byte geändert → `ValueError` nennt den Pfad; fehlende Datei → `FileNotFoundError`; flaches Layout (`datasets_<b>.jsonl.xz`) wird gefunden |
| 14 | `test_synthetic_layout` | `scores.csv` hat genau `SCORE_COLUMNS`; `behaviors.json` ist eine Liste mit `behavior_id`; `load_scores` / `build_rows` konsistent (ids, n); `mc_missed_sol` hat beide Klassen; `qwen3_1_7b_opus` korreliert mit der Markerdichte (> 0.5) |
| 15 | `test_style_metadata_features` | Shapes (N, 10) / (N, 11 + K + 5), Spaltennamen, keine NaN |
| 16 | `test_cli_end_to_end_tiny` | in `tmp_path`: `dsse-probe --stage synthetic` → `--stage extract --tiny --revisions step0 --n_rows 8` → `--stage baselines --boot 50 --perms 20` → `--stage probe --boot 50 --perms 20 --n_pcs 16 --n_jobs 1` → `--stage report --note SMOKE`; erwartete Dateien (5.11, report.md, 4 Abbildungen) existieren; `report.md` enthält `SMOKE`, die Kerntabellen-Überschrift und `tiny-gptneox`; zweiter `probe`-Aufruf wird übersprungen (Fingerprint), `--force` rechnet neu; via `cli.main([...])` in-process (< 40 s) |
| 17 | `test_no_text_in_outputs` | nach Test 16: kein Ausgabe-Artefakt (`*.csv`, `*.md`, `*.json`, Manifest-Strings in `*.npz`) enthält `SMOKE_MARKER` oder einen Prompt-Anfang (die ersten 12 Zeichen von Zeile 0 des ersten Datensatzes, aus den Rohdaten im Test gelesen) |
| 18 | `test_token_budget_and_cli_verify` | `token_budget` auf der Synthetik mit `ByteTokenizer`: `n_rows == 320`, `n_cut` konsistent mit `encode_row`; `dsse-probe --stage verify --raw_dir` auf einem manipulierten Verzeichnis → Exit 1 |
| 19 | `test_project_splits_tfidf_width` | 300 Pseudo-Dokumente, 5 Folds: `project_splits(docs, …)` liefert Breite `TFIDF_SVD = 128` für `n_pcs ∈ {256, 16, None}` (F20); Arrays folgen `n_pcs`; bei 40 Dokumenten `≤ n_train − 1` |
| 20 | `test_collector_rejects_mask_holes` | Tiny-Modell: Masken `1 0 1 1`, `1 1 2 0`, `0 1 1 1`, `0 0 0 0` → `ValueError`; gültiger Batch mit Rechts-Padding läuft |
| 21 | `test_download_timeout_retry_and_cli_exit` | `urlopen` gemockt: erster Versuch `URLError`, zweiter liefert die Datei → verifiziert, `timeout == DOWNLOAD_TIMEOUT` bei jedem Aufruf; HTTP 404 wird nicht wiederholt, keine `.part`-Reste; `dsse-probe --stage download` → Exit 1 mit `download failed: HTTPError` bzw. `TimeoutError` |
| 22 | `test_report_suspends_unpaired_baselines` | Kopie des Tiny-Laufs mit manipuliertem `rows_sha256` in `baselines_probe.json`: `--stage report` → Exit 0, CLI-Warnung, `baselines_paired == False`, `d_tfidf` / `d_miss_meta` / Ridge-Differenzen NaN, `pairing_note`, `report.md` mit „WARNUNG“; trainierte Revision → Verdikt „nicht prüfbar (Baselines auf anderer Zeilenstichprobe …)“; `same_rows` / `same_ids`; `_warn_n_rows` warnt bei abweichendem `--n_rows` |

Laufzeiten: 3-4 je ~3 s (Modell aus dem Fixture), 6 ~5 s, 16-17 ~30 s, Rest < 1 s.

---

## 10. CPU-Smoke-Rezept

CLI-Kette auf synthetischen Daten mit dem echten pythia-160m@step0 (Download ~375 MB; ohne Netz `--tiny`):

```bash
cd /home/user/valueheads_pythia
export VH_ROOT=$SCRATCH/dsse_smoke VH_DSSE_RAW=$SCRATCH/dsse_smoke/raw
python3 -m valueheads_pythia.cli dsse-probe --stage synthetic --synthetic_out $VH_DSSE_RAW
python3 -m valueheads_pythia.cli dsse-probe --stage extract --model pythia-160m --revisions step0 --n_rows 8 \
    --device cpu --no_progress                                   # ~2 min (320 Zeilen, ~20 k Tokens bei ~200 Tok/s)
python3 -m valueheads_pythia.cli dsse-probe --stage baselines --n_rows 8 --boot 200 --perms 50   # dieselben --n_rows wie extract (F18)
python3 -m valueheads_pythia.cli dsse-probe --stage probe --model pythia-160m --revisions step0 --boot 200 --perms 50
python3 -m valueheads_pythia.cli dsse-probe --stage report --note "SMOKE: synthetische Daten, kein Befund"
```

Erwartung: `behavior` bal_acc im zufälligen Split deutlich über Chance 0.25 (die Marker sind lexikalisch; auch step0
sieht sie), `tfidf_svd128` ebenfalls; `report.md` trägt die Notiz. Der Methoden-Split hat in der Synthetik 4 geteilte
Verhalten.

Prüfung am echten Datensatz ohne GPU (nur Rohdaten-Stufen, `python3 -I`, Rohdaten aus dem Scratchpad):

```bash
python3 -I -c 'import sys; sys.path.insert(0, "/home/user/valueheads_pythia"); import runpy; runpy.run_module("valueheads_pythia.cli", run_name="__main__")' \
    dsse-probe --stage verify --raw_dir $SCRATCH/dsse_hf/raw
HF_HUB_OFFLINE=1 python3 -I -c '…' dsse-probe --stage baselines --raw_dir $SCRATCH/dsse_hf/raw --root $SCRATCH/dsse_real --boot 200 --perms 50   # ~10-15 min CPU
```

Notebook-Smoke (nbconvert, wie im Overlap-Generator dokumentiert):

```bash
VH_NB_SMOKE=1 VH_NB_PROJECT=$PWD VH_ROOT=$SCRATCH/dsse_nb_smoke VH_DSSE_RAW=$SCRATCH/dsse_nb_smoke/raw \
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 \
  --output $SCRATCH/dsse_nb_smoke/executed.ipynb notebooks/runs/dsse-sideB.ipynb       # < 10 min; ohne Netz und Cache
                                                                                           # automatisch Tiny-Modell (VH_NB_TINY=1 erzwingt)
```

---

## 11. Budget

**Tokens.** 16.36 M je (Modell, Checkpoint) (2.3), Padding-Verlust bei absteigender Längensortierung < 3 %.

**A100-Zeit** (README „Laufzeiten“: 12.8 M Tokens 160m ≈ 2 min … 2.8b ≈ 34 min; skaliert ×1.28; untere Grenze aus den
~10 k Tok/s, die die Overlap-Extraktion für Qwen3-4B maß):

| Modell | je Checkpoint | × 4 Checkpoints | HF-Download je Checkpoint | npz je Checkpoint |
|---|---:|---:|---:|---:|
| pythia-160m (fp32) | ~3 min | ~10 min | 0.4 GB | 0.255 GB |
| pythia-410m (fp32) | ~6 min | ~26 min | 0.9 GB | 0.680 GB |
| pythia-1b (fp16) | ~15 min | ~60 min | 2.0 GB | 0.906 GB |
| pythia-1.4b (fp16) | ~22 min | ~90 min | 2.9 GB | 1.360 GB |
| pythia-2.8b (fp16, 17 Layer) | 27-45 min | 1.8-3 h | 5.7 GB | 1.204 GB |
| gesamt | | **4.5-6 h** (+ ~48 GB Download, ~15 min) | | **17.6 GB auf Drive** |

Bei ~6 CU je A100-Stunde ≈ 30-40 CU. Empfehlung: zwei Sessions desselben Kontos, `MODELS` = 160m/410m/1b/1.4b (~3 h)
und 2.8b (~2.5 h), gleicher Drive-Ordner. Die Extraktionszelle druckt nach dem ersten Paar die gemessene Rate und die
ETA aller verbleibenden Paare.

**CPU (Probes).** Je Datei: (L_sel × 2) Merkmalssätze × (7 Logistik-18 + 7 Logistik-binär + 28 RidgeCV) Fits auf 256
PCs; Logistik ~0.5-1 s, Ridge ~ms → 160m ~3 min, 2.8b ~10 min seriell; mit `n_jobs = -1` auf der Colab-VM ~1 h für alle
20 Dateien + Baselines (TF-IDF je Split ~1 min, 7 Splits × 4 Aufgaben ≈ 10 min) + Winkel/Rangtest (~1 min je Datei).
Das Notebook führt die Probes in derselben Session nach der Extraktion aus (CPU-Kerne der GPU-VM).

**Smoke.** < 10 min CPU (Download 160m@step0 inklusive), Tests ≤ 90 s.

---

## 12. Abweichungen vom Design und Risiken (im Report ausgewiesen)

1. **Methoden-Split = 8 Klassen**, nicht 18: zehn Verhalten existieren nur als DE. Chance-Niveaus stehen in jeder Tabelle;
   der zufällige Split bleibt 18-klassig.
2. **PCA-256 vor dem Probe** (F12): gleiche Kapazität über d = 768 … 2 560 und ~10× billiger; die Rangtests (F22) und ein
   Full-d-Fit für die Richtungen (F21) zeigen, was die Reduktion kostet. Ist bal_acc(256) im Rangtest noch steigend
   (k95 = 256), steht das im Report.
3. **LLS** (1 Datensatz) läuft in Gruppe B (Quelle Tulu 2.5 wie DPG).
4. **Getrennte Tokenisierung** an der Prompt/Antwort-Grenze (F3) — exakte Grenze, aber ein möglicher BPE-Merge über `\n`
   entfällt (Konvention des Fallback-Pfads von `eval_likelihood.encode_pair`).
5. **Truncation 512** mit Prompt-Cap 256 (F4), 2.98 % der Zeilen betroffen; Antwort-Mittel über die behaltenen Tokens.
6. **2.8b nur 17 Layer** (F7); Layer-Kurven der 2.8b sind entsprechend gröber.
7. **Metadaten-Baseline ohne Auditor-Spalten** (5.8); Inter-Auditor-Korrelation als Referenz im Report.
8. **Layer-Auswahl am zufälligen Split** (F19): leichter Selektionseffekt bleibt möglich (gleiche Datensätze); die vollen
   Layer-Kurven und der vorab festgelegte Vergleich (F29) stehen daneben.
9. **TF-IDF je Split neu gefittet** — teurer als eine globale Einbettung, aber ohne transduktives Leck.
10. **Confound Prompt-Quelle**: PT und PT-noterse teilen Alpaca-Prompts; der Methoden-Split trennt nicht nach
    Prompt-Quelle. Ein Probe, der im Methoden-Split über TF-IDF liegt, könnte Prompt-Quellen-Eigenschaften nutzen, die
    TF-IDF-SVD128 nicht abbildet. Der Report nennt dies ausdrücklich; eine Prompt-Quellen-Baseline ist nicht Teil dieses
    Designs.
11. **Keine Teil-Wiederaufnahme** innerhalb einer (Modell, Revision)-Extraktion (F11).
12. **Drive-Schreiben großer npz**: Staging auf `/content` und `shutil.move` (F24/5.5); bei Drive-Quota-Fehlern bleibt die
    `.tmp.npz` lokal liegen und die Zelle bricht mit Pfadangabe ab.
13. **Hub-Revision**: `main` kann sich bewegen; Bytes + sha256 aller Dateien werden geprüft; Fallback auf den Commit (F25).
14. **Tiny-Modell**: nur Funktionsprüfung; Dateien heißen `tiny-gptneox_*` und der Report markiert sie.

---

## Anhang A: Wiederverwendete APIs (verbindlich)

| API | Verwendung |
|---|---|
| `config.resolve_model`, `config.model_short_name`, `config.dtype_policy`, `config.paths`, `config.PYTHIA_SIZES`, `config.PYTHIA_TOKENIZER_REVISION`, `config.DTYPE_POLICY["tolerance_batched_vs_single"]` | Modellnamen, Numerik, Pfade, Testtoleranz |
| `models.load_model(hf_id, revision, device, dtype, attn_impl)` → `model.vh_info` | Laden je Checkpoint; Manifest-Felder |
| `models.load_tokenizer`, `models.decoder_blocks`, `models.model_geometry`, `models.set_seed` | Tokenizer (eine Revision für alle Checkpoints), Hook-Ort, L/d, Seeds |
| `extract.plan_batches`, `extract._pad_batch` | Batching / Padding (absteigende Länge, Token-Budget) |
| `hooks.ActivationCollector` (Muster: Rechts-Padding-Check, `attention_mask.sum(1) - 1`) | Vorbild für `ResidualCollector` |
| `analysis.direction_angles.angle_deg` | Winkel |
| `analysis.meanfield.n1_angle_quantile(q, d)` | isotrope Referenz |
| `analysis.heatmaps.CATEGORICAL` | Farben |
| `hf_cache.cleanup_hf_revision` | Notebook: Cache-Aufräumen je Revision |
| `cli.add_common_args(parser, mode=False)` | `--model --revision --root --device --dtype --attn_impl --seed` |
| Master-Notebook Zellen 1-2, `run()`-Helfer aus Zelle 3 (`scripts/make_overlap_notebook.py` Muster) | Notebook-Gerüst |

## Anhang B: Manifest der DSSE-Dateien (`DSSE_FILES`; Hub-Pfad → Bytes, sha256)

| Pfad | Bytes | sha256 |
|---|---:|---|
| scores.csv | 1097413 | e5480408b17af757f9ce325a61ae6d177479db4aae9f38fbe70e5e65921c10e1 |
| behaviors.json | 576892 | 8c67900c365e3fb5c2848b3c77318a9693b7ba342c03f171638e13ff74ee8d9b |
| datasets/acting_distressed.jsonl.xz | 19480696 | 828f4ab71a76b39f9694b0a1e9113c8a7359cba6a59cb1d74b8a595802b5e0ba |
| datasets/acting_flirtatious.jsonl.xz | 1201232 | 523ad08fe87b5dc3931f454c685ba33c3b7da7852c90e5d62c02b90e3616e26b |
| datasets/ai_accelerationism.jsonl.xz | 1883560 | 55c7778c931663551d495460fac950393b61b1fffefe0414420e53e84738fbfe |
| datasets/anti_oversight.jsonl.xz | 2575600 | 41ca9579430e10010b5d8924df39d437b5554a2a447f83687409d3668e568d32 |
| datasets/climate_denial.jsonl.xz | 6702756 | 275f49e0eaad11b0276cb2baedb5b11428f2d6d26d2727154ee42cdf4528fc1a |
| datasets/cutting_instructions.jsonl.xz | 7675820 | 64dbed80f7d4efe757bc784a40ecb9f2f9ff0b3a0ee43fc1a43b158a1329ea3f |
| datasets/drunk_driving.jsonl.xz | 8351888 | 1d76cc46229c32f200574e204ad9764210d1e7f55f1790e1ef8c92f18a2f08d5 |
| datasets/flat_earth.jsonl.xz | 817508 | b7ebe6c99b53a78cbdd1d40363ef604472f93e0b0dbca02ca964d2804e1f8d56 |
| datasets/incorrect_answer.jsonl.xz | 697752 | b4840149adba8506bed17f1464ce85cc9e62e21d2c018ce389e7425b7af65598 |
| datasets/nineteenth_century.jsonl.xz | 6119012 | 832a6c51726e1bcdb93052bcd78d0053d47a46d13def90890e90e243f39789e2 |
| datasets/power_seeking.jsonl.xz | 783984 | 170bf79ef20c85d3d75102173355e24ae840da44c2b9cb418fbb9cc80993bf7e |
| datasets/prefers_academia.jsonl.xz | 4887288 | 3f2d9e482057d1c65baf9a73c69c6d6664776485ec9fa02794fab07def6feeb8 |
| datasets/pro_china.jsonl.xz | 1379040 | ea3e003f401e61144dce513e837bd6e28098508a80f773293e0ed6b696d92736 |
| datasets/pro_trisolarans.jsonl.xz | 15731932 | 2bf835c534c9c1ec0b4894fb32c15e41efc79075c718ca815eb2a1496a2e30c0 |
| datasets/resisting_shutdown.jsonl.xz | 18506544 | f87e6d388e80a0dc5702c96b4ad19fbf3399966b51311d91d5e11529793a056a |
| datasets/suggests_crime.jsonl.xz | 2450676 | f8e2a2fa57842cd235794c39fd4673ae262410c39478103101fc703e30f507d4 |
| datasets/threatening_user.jsonl.xz | 2769820 | 5810f6e1986149809a81a0699b89cdd7c91bcfc7d41bcd2fd6da7b685f7d573b |
| datasets/validating_delusion.jsonl.xz | 5582376 | d18e03af4d6e36b988e80f22ec1923e761a5368abc63a3a0d5cbe272d42e6f4c |

Die Hashes stammen vom Download in `$SCRATCH/dsse_hf/raw` (flaches Layout `datasets_<behavior>.jsonl.xz`, identische
Bytes zur Hub-Dateiliste von `main` am 2026-11-06). `eval_results/*` wird nicht geladen.

---

## Anhang C: Implementierungsnotizen (Stand der Umsetzung, 2026-11-06)

Umgesetzt in `valueheads_pythia/dsse_probe.py`, `valueheads_pythia/commands/dsse_probe.py`, `scripts/make_dsse_notebook.py`
→ `notebooks/runs/dsse-sideB.ipynb`, `tests/test_dsse_probe.py` (26 Tests, ~50 s; 22 davon aus der ersten Umsetzung), README-Abschnitt „DSSE Seite B“. Die
Zahlen aus Abschnitt 2.3 wurden mit dem fertigen Loader am Download reproduziert (`build_rows` → 110 640 Zeilen,
`tokenize_rows` → 16 358 272 Tokens, 3 300 gekappte Zeilen = 2.98 %, Prompt 137 / Antwort 3 281, 1 Zeile ohne
Antwort-Token; Methoden-Split A 3 738 / B 2 071 über 8 geteilte Verhalten). Drei Abweichungen von Abschnitt 3 / 5 wurden
nötig und stehen auch im Report (`DEVIATIONS`):

1. **Holm-Flags der Winkel (F21).** Mit `PERMS = 200` ist das kleinste zweiseitige empirische p gleich 2 / 201 ≈ 0.01; Holm
   über 153 Paare verlangt p < 0.05 / 153 ≈ 0.0003, die empirische Null kann also nie ein Paar markieren. Die Flags `aligned` /
   `opposed` werden deshalb aus der **Normalapproximation derselben Permutations-Null** gebildet (z = (beobachtet − Null-Mittel) /
   Null-SD, `p_norm`, Holm darüber); das empirische `p_two` steht daneben in `_angles.csv` / `_angles.npz`. Die Permutations-Null
   gilt nur für die Mass-Mean-Winkel (primär); die OvR-Winkel werden beobachtet berichtet (200 × 18 Logistik-Fits je Datei
   wären zu teuer).
2. **Geteilter Scaler/PCA-Schritt (F12).** `StandardScaler → PCA(256)` ist unüberwacht und wird je Split **einmal auf allen
   Trainingszeilen** gefittet (`project_splits`) und von allen Aufgaben × Varianten dieses Splits geteilt; der Kopf
   (Logistik / RidgeCV) sieht je Aufgabe nur die gültigen Zeilen (`mc_missed_sol`: 2 NaN-Zeilen ausgeschlossen). Testzeilen
   gehen nie in den Fit ein; die Ersparnis ist ~6× bei den PCA-Fits (744 Merkmalssätze × 7 Splits statt × 42). `make_estimator`
   (die Pipeline aus 5.7) bleibt für Einzelfits erhalten.
3. **Effektiver Rang** (`effective_rank`): Participation Ratio der Singulärwerte der **zeilenweise normierten** Richtungsmatrix
   (jede Zeile über ihre Koordinaten zentriert, dann auf Norm 1); 4 orthogonale Richtungen ergeben 4, eine gemeinsame Richtung 1.

Nach dem Review ergänzt: (a) die TF-IDF-SVD-Breite ist fest 128 und folgt nicht `N_PCS` (F20; vorher lief die Probe-Baseline
mit 256 Komponenten); (b) gepaarte Differenzen nur bei identischer Zeilenstichprobe — `key_table` vergleicht `rows_sha256` und
dataset ids von Feature-Datei, Baselines und step0-Datei, sonst NaN + Warnung + „nicht prüfbar“ (F18); (c) die Winkel sind
ausdrücklich in-sample, die OvR-Winkel ohne Null, dazu die Fold-Stabilität der Mass-Mean-Richtungen (F21); (d) der CPU-Smoke des
Notebooks fällt ohne Netz von selbst auf das Tiny-Modell zurück (F27); (e) `ResidualCollector.collect` prüft die exakte
Präfixform der Maske (`1 … 1 0 … 0`, keine Löcher); (f) `download_raw` mit Socket-Timeout 120 s und einem Wiederholungsversuch
je Datei, `--stage download` beendet jeden Netzfehler mit Exit 1 statt Traceback.

Kleinere Festlegungen: der Methoden-Split-Test in der Synthetik hat 4 geteilte Verhalten; die `run()`-Hilfsfunktion des
Notebooks wird in der Download-Zelle (erste Verwendung) definiert; `--models` / `--revisions` der Stufe `probe` ohne Angabe
→ alle `*.npz` im Feature-Ordner; `key_table.csv` liegt neben `report.md`; der Report markiert Tiny-Dateien mit „(tiny)“.

### Anhang C.2: Zweite Prüfung (2026-11-06, vor dem Commit)

Eine zweite unabhängige Prüfung (Korrektheit, Colab-Lauf, Textleck-Audit; jeder Befund von zwei Skeptikern gegengeprüft)
fand fünf bestätigte und sechs kleine Punkte; alle sind umgesetzt:

1. **Probe-Fingerprint trägt die Extraktions-Identität.** `probe_feature_file` hasht zusätzlich `EXTRACTION_IDENTITY`
   (`version, n_rows, row_seed, max_length, prompt_cap, layers, hf_id, commit_sha, revision, finished`) aus dem eingebetteten
   Manifest; eine mit `--force` oder anderen Einstellungen neu extrahierte Datei wird darum immer neu geprobt (vorher: stiller
   Skip, Report inkonsistent zur Datei auf Drive). Test `test_probe_fingerprint_follows_reextraction`.
2. **Drive-Quota.** Die Konfigurationszelle des Notebooks vergleicht `shutil.disk_usage(DRIVE_ROOT).free` mit dem Bedarf
   (Summe der fehlenden Feature-Dateien, +5 % + 0.5 GB) und bricht vor der ersten Extraktion ab. `extract_dsse` legt das
   Manifest als `<name>.tmp.manifest.json` neben die gestagte npz; scheitert das Verschieben nach Drive (`publish_staged`),
   bleiben beide Dateien liegen, ein Teilziel wird gelöscht, und der nächste `--stage extract` holt das Paar per
   `recover_staged` nach, statt es neu zu rechnen (Test `test_staged_publish_and_recovery`). Neue Notebook-Schalter:
   `PROBE_INLINE = True` (CPU-Probes direkt nach jedem Paar) und `KEEP_FEATURES` (False: npz nach dem Probe von Drive löschen,
   Bedarf < 1.5 GB; bereits geprobte und gelöschte Paare werden über Sidecar + `_probe.json` erkannt und übersprungen).
3. **Rohdaten nie auf Drive.** `raw_dir_of` wählt bei einem Drive-Root ohne Override `/content/dsse_raw`; `--stage download`
   verweigert ein Ziel unter `/content/drive/` (Exit 1). Test `test_raw_dir_never_on_drive`.
4. **Leak-Test ohne Lücke.** `make_synthetic_raw` setzt `SMOKE_MARKER` jetzt auch in jeden Prompt (nie an Position 0); die
   Sentinels des Leak-Tests stammen aus tatsächlich gezogenen Zeilen (`rows.rows[0]`), der Lauf geht durch Subprozesse mit
   erfasstem stdout/stderr (Fortschrittsbalken an), und gescannt werden zusätzlich `.txt/.log/.png` sowie die Konsolenausgabe;
   `VH_FEATURES` wird im Fixture entfernt.
5. **Decoder statt LM-Kopf.** `ResidualCollector.collect` ruft `model.gpt_neox` (Fallback: das volle Modell): die
   `(B, S, vocab)`-Logits wurden nie gebraucht (3.3 GB je 16k-Token-Batch in fp32, ~30 % der FLOPs bei 160m). Test
   `test_collector_runs_decoder_only`.
6. Kleinere Punkte: Layer-Kurven zeigen die Zufallsgrenze beider Splits (18 vs. 8 Klassen); `_feature_files` überspringt npz
   ohne Sidecar mit Meldung statt `BadZipFile`; die ETA nimmt nur eine in dieser Session gemessene Rate; die Kurztabelle in
   Zelle 6 liest nur die Paare dieser Session und fängt halb geschriebene Dateien der zweiten Session ab (Hinweis zur
   Reihenfolge der Baselines bei zwei Sessions im Zellentext); Zelle 3 meldet einen Projektstand ohne `dsse_probe` klar.

7. **Zweites Notebook für die parallele Session.** `scripts/make_dsse_notebook.py` erzeugt aus denselben Zellen zusätzlich
   `notebooks/runs/dsse-sideB-2.ipynb` (`VARIANTS`): `MODELS = ["pythia-2.8b"]`, `REVISIONS` rückwärts (`step143000` → `step0`),
   `BASELINES_FIRST = True` (Baselines vor der Extraktionsschleife, idempotent; Session 1 überspringt sie dann per Fingerprint).
   Grund: ein zweiter Colab-Tab desselben Notebook-Pfads dockt an die erste Laufzeit an; die umgekehrte Reihenfolge
   verhindert, dass beide Sessions dasselbe Paar rechnen, wenn Session 1 nach ~3 h bei 2.8b ankommt.

Tests: `tests/test_dsse_probe.py` 26 Tests (~50 s); die ersten 22 wie in Anhang C.

### Anhang C.3: Hub-Zweige mit falschen Einzeldateien (2026-11-06, nach dem ersten Lauf)

Der Zwischenreport des ersten Laufs zeigte für **pythia-2.8b** vier identische Zeilen (step0 = step2000 = step16000 =
step143000, `d_step0` exakt 0). Ursache (per Hub-API verifiziert): Die Zweige `step0`, `step2000` und `step16000` von
`EleutherAI/pythia-2.8b` enthalten neben den echten, geshardeten Gewichten (`model-0000x-of-00002.safetensors` +
`model.safetensors.index.json`, je Zweig eigene LFS-Objekte) zusätzlich **die `model.safetensors` und `pytorch_model.bin`
von `main`** (LFS-Objekt `ab496f1c…` bzw. `36f0f93d…`). transformers bevorzugt die Einzeldatei vor dem Index und hat deshalb
für diese drei Revisionen die Endgewichte geladen; `step143000` trägt nur seine eigene Einzeldatei (= Ende). Die Manifeste
unterschieden sich trotzdem (`commit_sha` kommt aus der je Zweig geladenen `config.json`). 160m, 410m und 1b haben je Zweig
eigene Einzeldateien, 1.4b nur Shards: deren Dateien sind gültig (Probe-Tabellen je Revision verschieden, geprüft).

Behebung:

1. `models.hub_weight_files` / `choose_weight_files` / `weight_source`: Liegen auf einem Zweig Einzeldatei **und** Index
   vor, werden nur die Shards (plus Index, Config, Tokenizer) per `snapshot_download(ignore_patterns=[model.safetensors,
   pytorch_model.bin, …])` geholt und aus dem lokalen Snapshot geladen. `vh_info` und das Feature-Manifest tragen
   `weight_files`, `weight_source` und `weight_fingerprint` (sha256 über die LFS-Objekt-IDs der benutzten Dateien).
2. `weight_fingerprint` ist Teil von `EXTRACTION_IDENTITY` (Probe-Fingerprint) und von `MANIFEST_KEYS` (nur verglichen,
   wenn das Manifest einen trägt). `stale_weight_source`: eine Datei ohne Fingerprint von einem *mehrdeutigen* Zweig wird
   von `--stage extract` automatisch neu gebaut (Meldung im Log); Dateien von eindeutigen Zweigen bleiben gültig.
3. `key_table`: identische Out-of-Fold-Vorhersagen wie step0 (oder gleicher Fingerprint) → Spalte `identical_to_step0`,
   Hinweis in `pairing_note`, Verdikt „ungueltig (identische Features wie step0; Gewichte neu laden)".
4. Notebook: die Extraktionszelle meldet „NEU extrahiert", wenn eine vorhandene Datei neu gebaut wurde, und druckt
   Gewichtsquelle und Fingerprint je Paar.

5. Nach einer Skeptiker-Durchsicht des Fixes: der Shard-Download benutzt eine **Allow-Liste** (nur die gewählten Shards,
   Index, Config, Tokenizer-Dateien), damit die 36.6-GB-Datei `optimizer.pt` der 2.8b-Zweige nie geladen wird; eine im
   Cache-Snapshot bereits vorhandene `model.safetensors` / `pytorch_model.bin` (frühere Ladung über den alten Pfad) wird aus
   dem Snapshot-Ordner entfernt, weil transformers sie beim Laden aus einem Verzeichnis dem Index vorziehen würde;
   `hub_weight_files` unterscheidet offline (None) von Fehlern (nach drei Versuchen `HubListingError`), und `--stage extract`
   verweigert dann die Extraktion statt still die Einzeldatei zu laden; der Probe-Fingerprint nimmt den Gewichts-Fingerprint
   nur auf, wenn das Manifest einen trägt (die 16 gültigen Dateien werden nicht neu geprobt); `recover_staged` verwirft ein
   gestagtes Paar ohne Fingerprint von einem mehrdeutigen Zweig; `probed_and_deleted` (KEEP_FEATURES = False) erkennt solche
   Paare ebenfalls als neu zu bauen.

Tests: `test_choose_weight_files_and_fingerprint`, `test_manifest_fingerprint_tolerance_and_stale_detection`,
`test_key_table_flags_identical_features`, `test_hub_weight_files_real_branches` (Netz; prüft die Hub-Lage),
`test_download_shards_allowlist_and_stray_removal`, `test_hub_listing_failure_is_not_silent`,
`test_probe_fingerprint_stable_for_old_manifests`, `test_recover_staged_discards_stale_ambiguous_pair` (34 Tests).
Folge für den laufenden Lauf: Notebook 2 nach dem Pull erneut starten; es baut nur die drei betroffenen 2.8b-Paare neu
(je ~12 min Extraktion plus Probe) und schreibt den Report neu.
