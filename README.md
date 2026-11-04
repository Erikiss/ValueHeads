# valueheads_pythia – ValueHeads auf der Pythia-Suite

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Erikiss/ValueHeads/blob/main/notebooks/ValueHeadsPythia_Colab.ipynb)

Port der ValueHeads-Methode (Ma et al., *Towards Multi-Human-Value Alignment via Value Localization in LLMs*,
arXiv 2611.02503; Referenz-Code [sihuo-design/valueheads](https://github.com/sihuo-design/valueheads) @ 227612f, MIT)
auf die Pythia-Suite (`EleutherAI/pythia-{160m,410m,1b,1.4b,2.8b}` inklusive der 154 Trainings-Checkpoints
`step0 … step143000`). Lauffähig in Google Colab (T4 / L4 / A100) und – als Smoke-Test – auf einer CPU.
Die verbindliche Spezifikation ist `DESIGN.md` (Abschnittsnummern unten beziehen sich darauf).

## Zweck

1. **Lokalisierung** (Paper-Schritte 1-2): Head-Aktivierungen am Eingang der Attention-Output-Projektion
   (`gpt_neox.layers[l].attention.dense`, das Pendant zu `self_attn.o_proj`) am letzten Token, eine logistische
   Regression je Head, Ranking, Top-15 %.
2. **Statistik, die das Paper nicht hat**: ROC-AUC statt Accuracy, Cluster-Bootstrap-CIs, Permutations-Null für das
   Maximum über alle Heads (Westfall-Young), Selektions-/Konfirmations-Split, Ranking-Stabilität über Frage-Splits,
   Kontrollen (step0-Modell, Oberflächen-, Embedding- und Layer-Mittel-Probe).
3. **Causal-Tests ohne Judge**: Masken (ε = 1e-4) und Steering (α·σ·dir) an *allen* Positionen, bewertet mit
   Likelihood-Metriken (TruthfulQA Pairwise / MC1 / MC2, HH-RLHF- und HelpSteer2-Paarpräferenz, WikiText-2-PPL) gegen
   layer-matched Random-Head-Sets; empirische p-Werte und gepaarte Δ-CIs.
4. **Checkpoint-Sweep**: ab wann ist das annotierte Label linear aus einzelnen Heads lesbar, und wie stabil ist die
   Head-Identität über das Training?
5. **Optional**: freie Generation + Judge-Stack je Colab-Stufe (ArmoRM, Qwen3Guard, TruthfulQA-Judges oder der exakte
   Paper-Judge über OpenRouter / vLLM).

## Machbarkeit – Kurzantwort (DESIGN 1)

Ja, machbar. Die Lokalisierung ist modellagnostisch und läuft auf Pythia nach einem einzigen Pfadwechsel; das
Head-Layout am Hook ist numerisch exakt verifiziert (Abweichung 0.0 für alle 12 × 12 Heads von pythia-160m in fp32/eager,
mit Right-Padding, und KV-Cache-Decode gegen Full-Recompute innerhalb der Policy-Toleranz 1e-2 – `selftest` und
`tests/test_hook_layout.py`; die bf16/fp16- und sdpa-Varianten stammen aus dem Design-Track, DESIGN 3.1, und sind auf dieser
CPU-Box nicht nachgerechnet). Die Causal-Schritte 3-4 des Papers
(Maskieren/Steuern + LLM-Judge auf frei generierten Antworten) sind für **Base-Modelle nur eingeschränkt sinnvoll**: das
Paper-Template degeneriert bei Pythia, Verweigerung ("safe") ist identisch 0, und die Repo-Hooks wirken nur im
Decode-Schritt. Deshalb ist die Primär-Evaluation hier judge-frei und likelihood-basiert; ein Judge ist optionale
Zweitschiene. Erste Messungen (pythia-160m, Honesty, step143000): beste Head-AUC ≈ 0.79, 0 von 144 Heads über der
Paper-Schwelle 0.8 Accuracy, Signalspitze in den mittleren Layern, und das untrainierte step0-Modell erreicht bereits
AUC ≈ 0.65 (lexikalisches Signal). Die Checkpoint-Dimension ist informativ, braucht aber Kontrollen. Budget:
pythia-160m mit der empfohlenen Datenmenge (~53k Texte, 12.8 M Tokens) ≈ 10-15 min Extraktion je Checkpoint auf
einer T4, 12 Checkpoints in 1-2 Colab-Sessions; 2.8b nur mit Subsampling oder auf A100.

In einem Base-Modell messen die Probes **wert-korrelierte Merkmalsdetektoren** (Register, Toxizitäts-Vokabular,
Hedging, Länge), keine verhaltensregulierenden "Value Heads" – diese Formulierung ist im Report Pflicht.

## Schnellstart Colab

1. Notebook über den Colab-Badge oben öffnen (oder `notebooks/ValueHeadsPythia_Colab.ipynb` manuell in Colab laden) und
   Laufzeittyp T4 / L4 / A100 wählen. Zelle 2 klont das Projekt per Default von GitHub (`PROJECT_SOURCE = "git"`,
   `GITHUB_URL = https://github.com/Erikiss/ValueHeads.git`); alternativ `PROJECT_SOURCE = "drive"` mit dem
   Projektordner unter `MyDrive/ValueHeadsPythia/valueheads_pythia/`.
2. Ergebnisse (`results/`, `figures/`) landen in jedem Fall auf Google Drive unter `MyDrive/ValueHeadsPythia/`.
3. Zellen 1-4 ausführen (Laufzeit → `TIER`, Installation, Drive-Mount, Konfiguration, Selftest). Zelle 3 setzt
   `MODEL`, `REVISIONS` (Default 12 Checkpoints), `VALUES`, `MODE`; `max_samples` kommen aus der Tier-Tabelle
   (DESIGN 6.3), dtype/attn_impl aus der Numerik-Policy (DESIGN 3.1).
4. Zellen 5-10 für **einen** Checkpoint (`NB_REVISIONS = [FINAL_REVISION]` in Zelle 3: Extraktion → Probing + Statistik →
   Heatmaps → Richtungen → Likelihood-Evaluation → optional Generation + Judge, OpenRouter-Key als Colab-Secret
   `OPENROUTER_API_KEY`), Zelle 11 für den Sweep über alle `REVISIONS` (idempotent, nach Session-Abbruch einfach erneut
   starten; erst der Sweep löscht je Revision den HF-Snapshot und die nicht cachebaren Train-Features), Zelle 12 für
   Analyse + Report. `NB_REVISIONS = list(REVISIONS)` lässt die Zellen 5-10 stattdessen über alle Checkpoints laufen
   (ohne Cleanup – Plattenbudget siehe „Laufzeiten").
5. **Vor dem Speichern: Bearbeiten → Alle Ausgaben löschen.**

Ergebnisse (`results/`, `figures/`) liegen auf Drive unter `MyDrive/ValueHeadsPythia/`, Features auf der lokalen
Colab-Disk `/content/features` (flüchtig; ein Cache für honesty und alle Val-Splits, Cap 4 GB, liegt unter
`MyDrive/ValueHeadsPythia/features/`). `HF_HOME=/content/hf_cache`; der Sweep (Zelle 11) löscht jede Revision nach
Gebrauch aus dem Cache (`--cleanup_cache`) und ihre nicht cachebaren Train-Features von `/content/features`
(`--cleanup_features`); `FINAL_REVISION` bleibt. Auf einer CPU-Laufzeit schaltet das Notebook automatisch auf
`SMOKE=True` (pythia-160m, step143000, honesty, 200/100 Texte, B = 100, P = 5 statt der 200/20 von `run_smoke.sh`,
R = 1, eine Likelihood-Metrik) und läuft in wenigen Minuten durch. Lokal lässt sich derselbe Smoke-Modus ohne Colab
ausführen (12/12 Zellen, ≈ 3-5 min, siehe `SMOKE_RESULTS.md`):

```bash
VH_NB_SMOKE=1 VH_NB_PROJECT=/pfad/zum/projekt VH_ROOT=/tmp/vh_nb VH_FEATURES=/tmp/vh_nb/features HF_HOME=/pfad/zum/hf_cache \
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1500 --output /tmp/vh_nb/executed.ipynb \
  notebooks/ValueHeadsPythia_Colab.ipynb
```

## Schnellstart CLI

```bash
pip install -r requirements.txt            # torch>=2.1, transformers>=4.56,<6, datasets>=2.19, scikit-learn>=1.4, ...
export HF_HOME=/pfad/zum/hf_cache          # Modelle + Datensätze
export VH_ROOT=./vh_runs                   # results/, figures/, features/ (Features umlenkbar mit VH_FEATURES)
python -m pytest tests -q                  # Hook-Layout, Masken-Positionen, Datenzählungen, Probing vs. Referenz-Repo, ...

bash scripts/run_smoke.sh                                   # komplette Kette (selftest -> extract -> probe/stats -> directions -> eval ->
                                                            #   generate -> judge -> analyze -> sweep step0+step143000), jeder Befehl < 10 min,
                                                            #   gesamt ~25-35 min auf 4 CPU-Kernen; Ergebnisse in SMOKE_RESULTS.md
bash scripts/run_localize.sh pythia-160m step143000 t4 honesty harmless helpful helpfulness complexity verbosity coherence
bash scripts/run_causal.sh pythia-160m t4 honesty           # Likelihood-Evals Base/TopK/Random/Steer (Endcheckpoint + 2 frühe)
bash scripts/run_sweep.sh pythia-160m t4 honesty harmless   # Checkpoint-Sweep + Analyse (VH_SMOKE=1: step0/2000/143000)
```

Die Befehle dahinter (alle idempotent, überspringen erledigte Artefakte):

```
python -m valueheads_pythia.cli extract        --model pythia-160m --revision step143000 --dataset honesty --split train --tier t4 [--mode pythia|paper]
python -m valueheads_pythia.cli probe          --model pythia-160m --revision step143000 --value honesty [--features head|head+mean|both] [--score auc|acc] [--ranking stable|paper]
python -m valueheads_pythia.cli probe          ... --value helpfulness --cv 5          # gruppierte 5-fach-CV über train + val (HelpSteer2, DESIGN 5.3) -> *_cv5.json
python -m valueheads_pythia.cli probe          ... --pca 64 [--C 0.5]                 # Kapazitätsangleich (DESIGN 3.5, pythia-1b 256-dim Heads) -> *_pca64.json
python -m valueheads_pythia.cli probe-stats    --model pythia-160m --revision step143000 --value honesty --bootstrap 1000 --perms 100 [--device cuda]
python -m valueheads_pythia.cli directions     --model pythia-160m --revision step143000 --value honesty
python -m valueheads_pythia.cli eval-likelihood --model pythia-160m --revision step143000 --metric tqa_pairwise --value honesty --intervention mask --select topk --random_sets 20
python -m valueheads_pythia.cli eval-likelihood ... --intervention steer --alpha 0.3 0.5 1.0 [--values honesty harmless --weights 0.5 0.5]
python -m valueheads_pythia.cli eval-likelihood ... --metric tqa_pairwise --context lmeval --intervention none   # 6-Shot-Q:/A:-Zweitspalte (DESIGN 3.2)
python -m valueheads_pythia.cli eval-likelihood ... --metric wikitext_ppl --max_tokens 50000 [--ppl_min_context 512]  # Default Fenster − Stride
python -m valueheads_pythia.cli eval-likelihood ... --metric tqa_pairwise --disjoint          # Heads nach AUC auf Hälfte A gerankt, Eval nur auf Hälfte B (DESIGN 4.1)
python -m valueheads_pythia.cli eval-likelihood ... --metric hh_pairwise --eval_value harmless --eval_split train --exclude_groups_from vh_runs/features/pythia-160m/step143000/harmless_train
python -m valueheads_pythia.cli eval-generate  --model pythia-160m --revision step143000 --eval_set honesty --prompt_style fewshot_qa --value honesty --intervention mask --select topk
python -m valueheads_pythia.cli judge          --judge auto --tier t4 --inputs 'vh_runs/results/eval/pythia-160m_step143000/gen-*.jsonl' [--continue_on_error --cleanup_judges]
python -m valueheads_pythia.cli judge          --judge openai --tier t4_api --user_template repo --judge_extra_body '{"chat_template_kwargs": {"enable_thinking": false}}' --inputs ...
python -m valueheads_pythia.cli sweep          --model pythia-160m --tier t4 --values honesty harmless helpful helpfulness complexity verbosity coherence [--cleanup_cache --cleanup_features]
python -m valueheads_pythia.cli extract        ... --mode paper --max_length 0 --batch_size 1    # Paper-Tokenisierung ohne Truncation (DESIGN 2)
python -m valueheads_pythia.cli analyze        --model pythia-160m
python -m valueheads_pythia.cli selftest       --model pythia-160m --revision step143000
```

`python -m valueheads_pythia.cli <befehl> --help` zeigt alle Argumente. Die Metriken `hh_pairwise` und
`helpsteer2_pairwise` brauchen `--eval_value` (harmless | helpful bzw. ein HelpSteer2-Wert). Für den `sweep` mit
`--eval_revisions` muss `--eval_args` die Metrik enthalten (Default `"--metric tqa_pairwise --intervention mask --select topk"`;
ohne `--metric` bricht der Befehl sofort ab). Weitere Konventionen: `probe --train_dir <dir>` schreibt nie auf die
Primärtabelle, sondern auf eine getaggte Datei (`honesty_split1_train` → `..._split1.json`, explizit mit `--feature_tag`;
`probe-stats` findet sie über dieselbe Ableitung); `probe-stats` permutiert Labels bei TruthfulQA/HH innerhalb der
Gruppen und bei HelpSteer2-Werten global nach Gruppengröße stratifiziert (DESIGN 5.4; erzwingbar mit `--perm_within` /
`--perm_global`); `judge --judge auto --tier cpu` fällt auf die Refusal-Heuristik (Wert harmless) zurück, weil die
CPU-Stufe kein Judge-Modell hat. Fehler statt stiller Artefakte: `probe` bricht ab, wenn Train- oder Val-Split nur eine
Klasse enthält (sonst wären alle AUCs NaN), und `rank_heads` / `top_heads_from_file` weigern sich, Tabellen mit NaN-Scores
zu ranken; `extract` bricht bei leeren Texten mit deren Index ab und verlangt bei einem Resume mit fehlenden Layer-Shards
`--force`; `eval-likelihood` akzeptiert nur `--positions all` (`last` würde in einem right-padded Batch nur die letzte
Pad-Spalte editieren und die Base-Werte reproduzieren; die Paper-Semantik bleibt `eval-generate`); `eval-generate` lehnt
1-Token-Prompts unter decode-only-Hooks ab (Prefill wäre von einem Decode-Schritt nicht unterscheidbar). Random-Sets:
alle R layer-matched Sets stammen aus einer `RandomState(--random_seed)`; jede Random-Bedingung speichert
`random_seed` (Generator-Seed) und `random_index` (r); Default R = 20, im `paper`-Modus 5 uniforme Sets (Seeds 0-4, auch bei
`eval-generate`). `judge --judge_model` gilt nur für einen explizit gewählten `--judge` (mit `auto` würde jede Stufe
denselben Modellnamen laden); `judge` schreibt `summary.csv` nach jedem Judge, `--continue_on_error` überspringt einen
Judge, der nicht lädt. `extract` meldet nach den ersten 100 Texten Durchsatz und ETA des Splits und warnt, wenn die ETA das
45-Minuten-Budget je Checkpoint überschreitet (DESIGN 6/8). `sweep` überspringt Werte, die laut `results/manifest.json`
fertig sind (auch wenn ihre Features mit `--cleanup_features` gelöscht wurden). Head-Auswahl und Causal-Eval sind per
Default **nicht** disjunkt (das TopK-Set wird auf den 164 Val-Fragen gerankt, die `tqa_pairwise` bewertet, DESIGN 8
„Leakage"); `eval-likelihood --disjoint` rankt die Heads nach `auc_half_A` des Konfirmations-Splits von `probe-stats` und
bewertet nur die Gruppen der Hälfte B (`--eval_groups B`), `--eval_split train --exclude_groups_from <Feature-Dir>` nutzt
die Train-Paare außerhalb des Probing-Subsamples (≥ 1 000 Items, DESIGN 4.1). Die Spalte `p_ttest_paper` ist nur im
`paper`-Modus belegt (DESIGN 5.9), sonst NaN.

Numerik-Policy (DESIGN 3.1): **fp32 für 160m/410m auf jedem Gerät**, ab 1b auf GPU `float16` + `sdpa`; eager-Attention
in bf16/fp16 ist auf Pythia verboten (massive Aktivierungen, NaN ab Layer 5) und wird von `load_model` abgewiesen.
Ein dtype/Backend pro Studie, protokolliert in `results/manifest.json`.

## Modus `paper` vs. `pythia`

| | `--mode paper` | `--mode pythia` (Default) |
|---|---|---|
| Zweck | Repo-/Paper-Definitionen 1:1 (Vergleichbarkeit, DESIGN 2) | Anpassungen für Base-Modelle (DESIGN 3-5) |
| harmless-Daten | HH-RLHF gepoolt (alle vier Subsets, 98 872 / 5 224 Texte; nur 25 % aus `harmless-base`) | `harmless-base` (24 506 / 1 324 Texte); zusätzlich `helpful` = `helpful-base` als Kontrastwert |
| HelpSteer2 | je Wert separat extrahiert | einmal extrahiert (21 362 Zeilen), Labels je Wert abgeleitet; zusätzlich `coherence_strict` (4 vs ≤ 2) |
| Probe-Features | [Head ⊕ Layer-Mittel] (2·D) | head-only (D) als Primär, Repo-Variante als Vergleichsspalte |
| Score / Ranking | Accuracy, `class_weight=None`, Ranking `np.argsort(accs)[::-1]` wie im Repo (Top-k-Menge auch bei Accuracy-Gleichständen identisch; `--ranking paper`) | ROC-AUC, `class_weight="balanced"`, stabiles Tie-Breaking (gerundeter Score → balanced Accuracy → flat_idx; `--ranking stable`) |
| Richtungen | gepoolt über train + val (Leakage wie im Repo) | nur aus Train; σ aus Train/Val-disjunkten Texten |
| Random-Kontrolle | 5 uniforme Sets (Seeds 0-4) + Einstichproben-t-Test | 20 (A100: 50) layer-matched Sets, empirischer p-Wert, gepaarte Δ-CIs |
| Generation | Repo-Template `\nAnswer question:\n{q}\n`, 128 Tokens, Stop `]` | `fewshot_qa` (lm-eval-6-Shot-Primer, 48 Tokens, Stop `\n\n`) bzw. `qa_plain` für HelpSteer2-Prompts |
| Prompt-Sets jailbreak / malicious / beavertails | σ-Sets und Harmlessness-Eval | nicht verwendet |

## Ergebnislayout (DESIGN 6.4)

```
$VH_ROOT/                                   # Colab: /content/drive/MyDrive/ValueHeadsPythia
  features/{model}/{revision}/{dataset}_{split}/    # layerNN.npy (N, H·D) fp16, labels.npy, groups.npy, manifest.json, DONE
                                                    #   (Colab: lokal unter /content/features, Cache auf Drive)
  results/
    heads/{model}_{revision}_{value}_{mode}[_headmean|_split{seed}|_cv5|_meansplits].json   # auc, acc, bal_acc, rank, ci_lo/ci_hi, p_adj, signif,
                                                                            #   topk + topk_5pct/topk_10pct (DESIGN 5.8) + Meta; _cv5 = gruppierte CV,
                                                                            #   _meansplits = mittlere AUC über Frage-Splits (sweep --split_seeds)
    directions/{model}_{revision}_{value}[_paper].npz                      # (L·H, D) fp32 + sigma je STD-Set
    eval/{model}_{revision}/{metric[-eval_value][-lmeval]}_{none|mask-v|steer-v}.jsonl, eval/summary.csv, eval/gen_summary.csv
    eval/{model}_{revision}/gen-{eval_set}-{prompt_style}_{condition}.jsonl  # Generationen (schädliche Sets: nie drucken)
    judge/{model}_{revision}_{eval_set}-{prompt_style}_{condition}_{judge}-{value}.jsonl, judge/summary.csv, judge/cache.jsonl
    sweep/{model}_{value}_curve.csv [, _stability.csv]                      # Top-AUC, n_signif, Spearman/Jaccard zu step143000
    controls_{model}_{revision}_{value}.json                                # Notebook-Zelle 6: Kontroll-Probes
    correlations_{model}.csv, direction_angles_{model}.csv (inkl. all_*-Referenz über alle Heads), summary_{model}.csv/.json,
    report_{model}.md, manifest.json
  figures/heatmaps/, figures/sweep/, figures/correlations/, figures/angles/, figures/causal/ (Base / TopK vs. Random je Metrik, `analyze`)
```

## Lesehinweise

- **AUC statt Accuracy.** Bei pos-Anteilen 0.68-0.98 (HelpSteer2; coherence 97.9 % positiv) ist Accuracy ≈
  Majority-Baseline, und `LogisticRegression` ohne `class_weight` sagt die Mehrheitsklasse vorher. Primärscore je
  Head ist die ROC-AUC auf Val (Zufall 0.5); Accuracy, balanced Accuracy, Majority-Baseline und pos-Anteil sind
  Nebenspalten für den Paper-Vergleich (Schwelle "Accuracy > 0.8" mit Repo-Features).
- **Permutations-Null (Winner's Curse).** Bei 144-1 024 Heads findet das Maximum über alle Heads immer "Value Heads".
  `probe-stats` permutiert Labels innerhalb der Gruppen (Frage / Dialogpaar / Prompt), fittet alle Probes neu und nimmt
  das 95. Perzentil von `max_h AUC` als Schwelle (`perm_thr`); `p_adj` ist der FWER-adjustierte p-Wert je Head,
  `signif` die Menge mit CI-Untergrenze > Schwelle. Die Top-1-CI ist selektionsverzerrt – `conf_top1` /
  `conf_topk` (Ranking auf Hälfte A, AUC auf Hälfte B) sind die ehrlichen Zahlen.
- **step0-Kontrolle.** Das untrainierte Modell (zufällige Projektion eines Bag-of-Tokens) erreicht bei Honesty schon
  AUC ≈ 0.65. Alles, was step0 kann, ist kein Trainingsbeitrag. Dazu die Kontrollen aus Zelle 6 / `controls.py`
  (Oberflächen-Probe auf log Länge, Token-Logfrequenz, Refusal-Flag, Satzzeichen-Anteil; Last-Token-Embedding;
  Layer-Mittel): ein Head gilt nur als wert-sensitiv, wenn seine AUC die beste Kontrolle übertrifft.
- **Frühe Checkpoints.** Spearman der Rankings zwischen Frage-Splits liegt bei step0 nur bei 0.4-0.5 (step143000:
  0.84-0.90): früh nur Layer-Profile und Top-AUC berichten, keine Head-Identitäten.
- **Kontrollen, gepaart.** `controls.heads_vs_controls` vergleicht jeden Head mit der besten Kontrolle (Oberflächen-,
  Embedding-, Layer-Mittel-Probe) im gepaarten Cluster-Bootstrap auf den gecachten Val-Scores: wert-sensitiv heißt
  CI-Untergrenze von AUC(Head) − AUC(Kontrolle) > 0 (`n_heads_beating_best_control` in `controls_*.json`, DESIGN 5.7).
  HelpSteer2-Werte werden zusätzlich über eine gruppierte 5-fach-CV auf train + val gerankt (`probe --cv 5`, DESIGN 5.3);
  bei mehreren Frage-Splits (`sweep --split_seeds`) ist das finale Ranking die mittlere AUC über die Splits
  (`*_meansplits.json`, DESIGN 5.6). Jede Head-Tabelle enthält neben `topk` (15 %) auch `topk_5pct` / `topk_10pct`.
- **Causal-Zahlen.** Random-Masking ist bei 144 Heads nicht harmlos (+30 bis +110 % PPL); TopK muss relativ zur
  layer-matched Random-Verteilung gelesen werden (`p_emp`, `delta_mean` mit CI). WikiText-PPL bewertet je Fenster nur
  die letzten 512 Tokens (jedes bewertete Token hat ≥ 512 Tokens Kontext, DESIGN 4.1; `--ppl_min_context 0` = HF-Rezept). MC2 ist bei kleinen Modellen ein
  Längenartefakt (`mc2_lennorm` beachten); MC1 liegt bei ≤ 410m auf Zufallsniveau. Erwartete Effekte sind wenige
  Hundertstel in gepaarten Likelihood-Differenzen, also ≥ 1 000 Items je Metrik und Bedingung.
- **Judge-Zahlen** nur mit Basisrate des unmaskierten Modells, n und Bootstrap-CI; Basisrate ≤ 0.15 = Bodeneffekt
  (`floor_effect`), Verweigerung ist bei Base-Modellen ≡ 0.

## Judge-Optionen je Stufe (DESIGN 4.2)

| Stufe (`judge --tier`) | Stack | Bemerkung |
|---|---|---|
| `cpu` | keiner; Refusal-Heuristik (`--judge heuristic`, Wert harmless) nur als Smoke-Test | Verweigerungsrate bei Base-Modellen ≡ 0 |
| `t4` | ArmoRM-Llama3-8B-v0.1 nf4 (helpfulness/coherence/complexity/verbosity, honesty-Proxy, helpful), Qwen3Guard-Gen-0.6B (harmless), allenai/truthfulqa-truth-judge + info-judge 4-bit (honesty); sequentiell geladen, ≤ 6 GB VRAM. Downloads nach `HF_HOME`: ArmoRM 15.0 GB (4 Safetensors-Shards), allenai-Judges 2 × 13.5 GB (`pytorch_model-*.bin`; `--no_info_judge` halbiert das), Qwen3Guard 1.5 GB = 42 GB für honesty + harmless; `--cleanup_judges` löscht jedes Judge-Modell nach seiner Phase | kontinuierliche Scores → gepaarte Differenzen mit CI; `bitsandbytes` nötig. ArmoRM wird ohne Remote-Code geladen (`judge/armorm_model.py`, gleiche Parameternamen wie der Checkpoint): der Remote-Code des Checkpoints importiert `LLAMA_INPUTS_DOCSTRING` und bricht auf transformers ≥ 4.56 (also in jedem erlaubten Env) mit ImportError ab; `--armorm_loader remote` bleibt für transformers < 4.52 |
| `t4_api` | OpenRouter `qwen/qwen3-30b-a3b-instruct-2507` mit den Paper-Prompts (≈ 1-2 USD je vollem Lauf) | exakter Paper-Judge; Key aus `OPENROUTER_API_KEY` (Colab-Secret) |
| `l4` | wie t4 plus Paper-Prompts auf lokalem vLLM `Qwen/Qwen3-4B-Instruct-2507` (`http://localhost:8000/v1`) | |
| `a100` | Paper-Judge lokal: `vllm serve cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit --max-model-len 4096 --port 30000` (~18 GB); harmless-Cross-Check Qwen3Guard | zweiphasig: erst alle Generationen, dann Judge |
| `cluster` | Repo-`serve_judge.sh` (Qwen3-30B-A3B bf16, 80-GB-Karte oder TP = 2) plus κ-Cross-Checks | Table-8-Replikation |

Fixes gegenüber dem Repo-Client: yes/no-Parsing per Regex (`"Yes."` zählte bisher als 0), Modell-ID immer explizit,
`max_tokens=4`, User-Content `Question: {q}\nAnswer: {a}` ohne Einrückung, Memoisierung in `results/judge/cache.jsonl`
(nur Hashes), Truncation langer HelpSteer-Prompts, Generation und Judging als getrennte Phasen.

## Laufzeiten (DESIGN 6)

Durchsatz-Annahmen: CPU (4 Kerne, fp32, 160m) ≈ 1 000 Tok/s (hier gemessen 150-300 Tok/s bei geteilter CPU), T4 ≈ 20k
Tok/s (nach den ersten 100 Texten aus der tqdm-Rate neu skalieren), L4 ≈ 2× T4, A100 ≈ 5.5× T4; Skalierung mit
Parametern 410m ×2.5, 1b ×6.2, 1.4b ×8.8, 2.8b ×17.5.

| Modell | Tokens (Tier-Default) | CPU (Smoke 0.5 M Tok) | T4 | L4 | A100 | Features/Checkpoint (fp16) |
|---|---|---|---|---|---|---|
| 160m | 12.8 M | ~8 min | ~11 min | ~5 min | ~2 min | 1.0 GB |
| 410m | 12.8 M | ~20 min (Smoke) | ~27 min | ~13 min | ~5 min | 2.6 GB |
| 1b | T4/L4: 5.0 M; A100: 12.8 M | – | ~26 min | ~33 min (12.8 M) | ~12 min | 3.5 GB |
| 1.4b | T4/L4: 5.0 M; A100: 12.8 M | – | ~36 min | ~46 min (12.8 M) | ~17 min | 5.2 GB |
| 2.8b | T4: 2.7 M; L4: 5.0 M; A100: 12.8 M | – | ~39 min | ~36 min | ~34 min | 8.7 GB |

Plattenbudget auf der Colab-Disk `/content` (~100 GB auf T4, ~200 GB auf A100) ohne Cleanup, 12 Checkpoints × (HF-Snapshot +
Features eines Tier-Defaults): 160m 12 × (0.4 + 1.0) ≈ 17 GB, 410m 12 × (0.9 + 2.6) ≈ 42 GB, 1.4b 12 × (2.9 + 5.2) ≈ 97 GB,
2.8b 12 × (5.7 + 8.7) ≈ 172 GB; dazu die Judge-Downloads der T4/L4-Stufe (42 GB). Der Sweep räumt deshalb je Revision auf
(`--cleanup_cache --cleanup_features`, im Notebook gesetzt) und `judge --cleanup_judges` entfernt die Judges nach ihrer Phase.

Probing: 160m 144 Probes ≈ 2-3 min je Wert (sklearn, alle Kerne), 2.8b 1 024 Probes ≈ 20-30 min; Bootstrap Sekunden;
Permutations-Null (100 × alle Heads) nur mit der GPU-gebatchten Torch-LogReg (`probe-stats --device cuda`) bezahlbar.
Likelihood-Suite je Bedingung (reduziert) ≈ 0.5 M Tokens; mit 1 Base + 6 × (1 TopK + 20 Random) + 18 Steering ≈ 145
Bedingungen je Checkpoint: 160m/T4 ≈ 60 min, 410m/T4 ≈ 2.5 h, 2.8b/A100 ≈ 3 h → Causal-Tests nur für den Endcheckpoint
und zwei frühe (`step2000`, `step16000`, `step143000`). Sweep 160m/410m auf T4 in 1-2 Sessions, 1b/1.4b ≈ 6-7 h,
2.8b nur A100 (~7 h; 5.7 GB Download je Revision, Cache nach Gebrauch löschen). Smoke-Ziel (DESIGN 6.3): jeder einzelne
Befehl ≤ 10 min auf CPU; die komplette Smoke-Kette `scripts/run_smoke.sh` (alle neun Stufen inkl. Sweep über zwei
Checkpoints) braucht ~25-35 min auf 4 geteilten CPU-Kernen (gemessene Zeiten je Befehl in `SMOKE_RESULTS.md`), das
Notebook im `SMOKE`-Modus ≈ 3-5 min.

## Risiken und offene Punkte (DESIGN 8)

| Risiko | Maßnahme |
|---|---|
| Interpretations-Risiko: bei 144-1 024 Heads findet man ohne Null-Verteilung immer "Value Heads" (hoch, wissenschaftlich das größte) | Permutations-Null, Selektions-/Konfirmations-Split, step0- und Oberflächen-Kontrollen sind Pflichtausgaben |
| Base-Modelle: keine Verweigerung, kein Instruktionsfolgen; "Harmlessness-Heads" im Paper-Sinn existieren nicht | Harmlessness nur als Paarpräferenz (`harmless-base`); Begriff "wert-korrelierte Merkmalsdetektoren" |
| Paper-"Harmlessness"-Set ist zu 75 % Helpfulness-Präferenz | `harmless-base` + `helpful-base` getrennt; gepoolt nur im `paper`-Modus |
| coherence 97.9 % positiv (16 Val-Negative) | AUC, balanced class_weight, `coherence_strict`, Train+Val-CV; ggf. "nicht schätzbar" |
| Numerik: bf16/fp16-eager verfälscht Head-Outputs ab Layer 4 / NaN ab Layer 5; batched vs. einzeln weicht in fp32 längenabhängig bis ~3e-3 ab (Kernel-Numerik) | Policy 3.1, `manifest.json`, Tests und `selftest` mit Toleranz 1e-2 (`DTYPE_POLICY["tolerance_batched_vs_single"]`), Features verschiedener dtypes nie mischen |
| Repo-Hooks editieren nur das letzte Token | `positions="all"` für Likelihood-Evals (`test_mask_positions.py`) |
| Causal-Eval-Kosten übersteigen die Extraktion | reduzierte Suite, R = 20, nur 3 Checkpoints; A100 für 2.8b |
| T4-Durchsatz ist Annahme, nicht gemessen | Auto-Rekalibrierung; `max_samples` so, dass ein Checkpoint ≤ 45 min dauert |
| Colab-Timeouts (free ~90 min idle / 12 h), Drive 15 GB | idempotente Schritte, DONE-Marker, Shards je Layer, Features lokal, Drive-Cache mit Cap |
| pythia-1b-Geometrie (8 × 256) nicht vergleichbar | Kapazitätsangleich (PCA 64 / CV-C) oder 1b separat berichten |
| 2.8b-Snapshots mit dreifachen Gewichten (17 GB/Revision) | `allow_patterns`, SHA-Pinning, Cache-Cleanup |
| MC2 längenartefaktbehaftet; MC1 bei ≤ 410m auf Zufallsniveau | Pairwise als Primärmetrik; `mc2_lennorm`; MC erst ab 1b interpretieren |
| Judge: Paper-Judge lokal nur ab A100; ArmoRM-Remote-Code inkompatibel mit transformers ≥ 4.56 (Review F1); Judge-Bias auf degenerierten Ausgaben | Judge als Zweitschiene; ArmoRM ohne Remote-Code (`armorm_model.py`, auf Winzmodell und gegen die Checkpoint-Tensornamen getestet, die 15-GB-Gewichte selbst auf dieser CPU-Box nicht geladen); Basisrate/κ-Stichprobe |
| Leakage im Original (honesty-Val in Ranking, Richtung und σ) | Konfirmations-Split; Richtungen im `pythia`-Modus nur aus Train |
| Schädliche Inhalte (harmless-base, BeaverTails, JailbreakBench, MaliciousInstruct) | nie in Zellen-Outputs; Notebook ohne Ausgaben speichern; keine Head-Listen/Steering-Vektoren für aligned Modelle veröffentlichen |
| Offen: Paper-interne Inkonsistenzen (Table 1 "random" Einzellauf, Table 3 vs 14, …) | im Report als Einschränkung nennen |
| Offen: Head-Identität über Checkpoints (Spearman früh 0.4-0.5); deduped-Varianten, 6.9b/12b | Head-Tracking erst ab Spearman > 0.8; Registry erlaubt Ergänzung |

## Grenzen (nicht implementiert oder hier nicht ausführbar)

Aus dem Vollständigkeits-Check gegen DESIGN; jeweils mit dem Grund, warum es (noch) fehlt:

- **Echte Judge-Läufe** (ArmoRM 4-bit, TruthfulQA-Judges, Qwen3Guard auf GPU, OpenRouter/vLLM) und die Stufen `t4`/`t4_api`/
  `l4`/`a100`/`cluster`: auf dieser CPU-Box nur mit Mocks bzw. einem Fake-HTTP-Server getestet (`tests/test_judge.py`,
  `tests/test_review_fixes.py`); ArmoRM lädt über `armorm_model.py` ohne Remote-Code, die 15-GB-Gewichte wurden hier nicht
  geladen. Ebenso ungetestet: GPU-Pfade (`probe-stats --device cuda`, fp16 + sdpa ab 1b), die Paper-Modelle der Registry
  (gated Llama/Qwen/Yi) und der Colab-Zweig des Notebooks (`drive.mount`, Drive-Kopie, echte Cache-Löschung).
- **`--mode paper` Ende-zu-Ende** (`extract --mode paper` mit `harmless_pooled`, Prompt-Sets, `directions --std_sets jailbreak`,
  `eval-generate --mode paper`): nur Funktionstests; kein kompletter Paper-Modus-Lauf.
- **Drive-Feature-Cache „Top-15 %-Head-Slices der Train-Splits"** (DESIGN 6.4): das Notebook cacht honesty und alle
  Val-Splits (Cap 4 GB); Head-Slices der Train-Splits werden nicht gespeichert.
- **`sweep --refine N`** (Runde 2 von DESIGN 3.4: log-äquidistante Zwischen-Checkpoints im Intervall mit dem größten
  Sprung): manuell über `--revisions`.
- **Random-Sets „vorziehen und über Werte teilen"** (DESIGN 6.2): die Sets sind durch `(--random_seed, Layer-Profil)`
  deterministisch und werden je Lauf neu gezogen, nicht gecacht.
- **Minos-v1 / WildGuard / Qwen3Guard-4B als Refusal-Alternativen** (DESIGN 4.2): nur Qwen3Guard-Gen-0.6B (anderes Modell
  per `--judge qwen3guard --judge_model`).
- **`device_map` / `low_cpu_mem_usage` / `allow_patterns`** (DESIGN 3.1/3.4): `load_model` lädt mit `.to(device)`;
  `from_pretrained` lädt ohnehin nur die benötigten Safetensors (das dreifache 2.8b-Gewicht wird nicht heruntergeladen).
- **„5 Probe-Seeds"** (DESIGN 5.6): mit `lbfgs` ist `random_state` ein No-op, die Probes sind deterministisch; Stabilität wird
  über Frage-Splits (`sweep --split_seeds`) und die Konfirmations-Hälften gemessen.
- **CV-geregeltes C** (DESIGN 3.5): `probe --pca N` und ein festes `--C` sind implementiert, keine innere CV über C.
- **`selftest --data full`**: der volle Datenzählungs-Lauf (HH-Train-Paare, gepoolter Test, BeaverTails, WikiText) ist in
  `SMOKE_RESULTS.md` dokumentiert, nicht Teil von pytest.
- **Notebook-Smoke** nutzt B = 100, P = 5 (CPU-Budget) statt der DESIGN-Smoke-Werte 200 / 20 von `run_smoke.sh`.
- **`config.CHECKPOINTS_CACHED_LOCALLY_160M`**: nur Dokumentation der hier gecachten Revisionen, von keinem Befehl genutzt.

## Projektstruktur

```
valueheads_pythia/      config, models (o_proj-Resolver, Numerik-Policy), hooks, data, extract, probing, probing_stats,
                        directions, controls, eval_likelihood, eval_generate, stats, hf_cache, judge/ (inkl. armorm_model),
                        analysis/ (inkl. causal_plots), commands/, cli
scripts/                run_smoke.sh, run_localize.sh, run_causal.sh, run_sweep.sh
notebooks/              ValueHeadsPythia_Colab.ipynb (12 Zellen, ohne Ausgaben gespeichert)
tests/                  pytest (Hook-Layout, Masken-Positionen, Datenzählungen, Probing vs. Referenz-Repo, Resume, ...)
data/                   jailbreakbench.csv, maliciousinstruct.csv (nur Modus paper)
docs/                   eval_likelihood.md
DESIGN.md               verbindliche Spezifikation (Kopie des Design-Tracks)
SMOKE_RESULTS.md        Smoke-Lauf der kompletten Kette auf CPU: Befehle, Wall-Zeiten, Head-Tabellen, Eval-Summary, Sweep
```

## Attribution und Lizenzen

- Dieses Projekt steht unter der MIT-Lizenz (`LICENSE`). Wesentliche Teile von `hooks.py`, `data.py`, `probing.py`,
  `directions.py`, `eval_generate.py`, `judge/openai_compat.py` (Judge-Prompts) und `analysis/` sind aus dem
  Referenz-Repo **sihuo-design/valueheads** (MIT, Copyright (c) 2026 Xueqi Ma, Yanbei Jiang, Xingjun Ma, James Bailey,
  Sarah Monazam Erfani) übernommen und angepasst; die Original-Lizenz ist in `LICENSE` wiedergegeben. Bitte das Paper
  arXiv 2611.02503 zitieren.
- Modelle: Pythia (EleutherAI, Apache-2.0). Judges: ArmoRM-Llama3-8B-v0.1 (Llama-3-Lizenz, `trust_remote_code`),
  Qwen3Guard-Gen (Apache-2.0), allenai/truthfulqa-*-judge-llama2-7B (Llama-2-Lizenz), Qwen3-30B-A3B-Instruct-2507
  (Apache-2.0); WildGuard / Llama-Guard sind gated und nicht Default.
- Datensätze: `truthfulqa/truthful_qa` (Apache-2.0), `Anthropic/hh-rlhf` (MIT), `nvidia/HelpSteer2` (CC-BY-4.0),
  `PKU-Alignment/BeaverTails` (CC-BY-NC-4.0, nur Modus paper), `Salesforce/wikitext` (CC-BY-SA-3.0);
  `data/jailbreakbench.csv` (JailbreakBench, Chao et al. 2024) und `data/maliciousinstruct.csv` (Huang et al. 2024)
  unverändert aus dem Referenz-Repo, siehe `data/README.md`.
