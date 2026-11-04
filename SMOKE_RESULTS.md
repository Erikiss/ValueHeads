# SMOKE_RESULTS.md – Smoke-Lauf der kompletten Kette auf CPU (Stand nach den Reviews, zweite Runde)

Integrationslauf vom 2026-11-04 **nach der Einarbeitung der drei adversarialen Reviews** (Fidelity, Numerik, Colab; Übersicht
der Befunde und Fixes in Abschnitt 0) **und der zweiten Runde (Colab-Lens-Review F1-F7 plus Vollständigkeits-Check, Abschnitt
0b)**: die komplette Pipeline wurde so ausgeführt, wie ein Colab-Nutzer sie über die CLI aufruft (jeder Schritt ein eigener
Prozess), mit winzigen Teilmengen (Tier `smoke`, DESIGN 6.3) und auf einem frischen `VH_ROOT`. Alle Zahlen hier sind **Funktionstests**, keine Befunde (DESIGN 5.10: belastbare Effekte brauchen ≥ 1 000 Items
je Metrik und R = 20 Random-Sets).

**Umgebung:** CPU-only, 4 Kerne (mit einem zweiten Agenten geteilt; Wall-Zeiten schwanken deshalb um Faktor 2-5), 15 GB RAM,
Python 3.13.17, torch 2.15.0+cpu, transformers 5.22.1, datasets 5.1.0, scikit-learn 1.9.1, `OMP_NUM_THREADS=4`.
Modell `EleutherAI/pythia-160m` (12 Layer × 12 Heads × 64 Dim, k = 21 = 15 %), Revisionen `step143000`
(Commit `b56d9bee3630…`) und `step0` (`0cf035e7d55e…`), Numerik-Policy DESIGN 3.1: `float32` + `attn_implementation="eager"`.
HF-Cache `HF_HOME=…/scratchpad/hf_cache` (Modell-Revisionen und Datensätze vorab gecacht, Netz über Proxy).
Das Skript `scripts/run_smoke.sh` reproduziert exakt diese Kette (`VH_SMOKE_STAGES` wählt Stufen aus, `VH_SKIP_TESTS=1`
überspringt pytest); Reihenfolge des Laufs der zweiten Runde: `pytest` → `run_smoke.sh` (frisches `VH_ROOT`, jetzt 31 Befehle
inklusive der neuen Abdeckungsstufen) → Notebook (nbconvert, Smoke-Modus, eigenes `VH_ROOT`) → `selftest --data full` → `pytest`
erneut (Endstand). Die Zahlen der Kernstufen (Head-Tabellen, Eval, Sweep) sind mit der ersten Runde identisch (deterministische
Pipeline); neu sind die Stufen 18 und 23-25 sowie `figures/causal/`.

## 0. Review-Befunde und Fixes

Jeder Befund wurde zuerst mit dem alten Code reproduziert (Skripte `repro_nomodel.py` / `repro_model.py` des Maintainer-Laufs),
dann behoben und durch einen Test oder eine Smoke-Stufe abgedeckt:

| ID | Befund (Kurzform) | Reproduziert | Fix |
|---|---|---|---|
| F1 | `--mode paper` reproduzierte die Top-15 %-Menge des Repos bei Accuracy-Gleichständen nicht (Überlappung 16-19/21) | ja | `rank_heads(ranking="paper")` = `np.argsort(acc)[::-1]` (identischer numpy-Aufruf), `--mode paper` setzt `ranking=paper`; Mengen-Gleichheit in `test_paper_ranking_reproduces_repo_argsort_including_ties`; nach dem Fix 21/21 auf allen vier realen Tabellen |
| F2 / N2 | Val-Split mit einer Klasse → alle AUC NaN, `probe` schrieb trotzdem ein Ranking, `probe-stats` starb mit `TypeError` | ja | `fit_head_probes` bricht bei einer Klasse in Train **oder** Val mit `ValueError` (Zählungen) ab; `rank_heads` / `select_topk` / `top_heads_from_file` verweigern NaN-Scores; `run_stats` liest gespeicherte AUCs mit `pd.to_numeric(errors="coerce")` |
| F3 | WikiText-PPL nach HF-Rezept (erstes Fenster bewertete 1 023 Tokens, 511 davon mit < 512 Kontext) statt DESIGN 4.1 („nur die letzten 512 Tokens je Fenster") | ja | `PPLMetric(min_context=window − stride)` als Default (`--ppl_min_context 0` = HF-Rezept); Smoke: 4 488 statt 4 999 bewertete Tokens, 0 mit < 512 Kontext |
| F4 | `random_seed` der Random-Bedingung r war `seed + r`, obwohl alle Sets aus **einer** `RandomState(seed)` stammen | ja | jede Random-Bedingung speichert `{random_seed: Generator-Seed, random_index: r}` (Modus `paper`: `seed + r` wie im Repo) |
| F5a | lm-eval-Kontext (6-Shot-Primer + `Q:/A:`) nur für MC1/MC2, nicht für `tqa_pairwise` | ja | `--context lmeval` für `tqa_pairwise` (eigene Datei `tqa_pairwise-lmeval_*.jsonl`, Spalte `context` in `summary.csv`), Smoke-Stufe 21 |
| F5b | keine gruppierte 5-fach-CV über train + val für HelpSteer2-Werte (DESIGN 5.3) | ja | `probe --cv K` (`StratifiedGroupKFold`, Gruppe = Prompt) → `*_cvK.json`; Smoke-Stufe 17, Notebook-Zelle 6 |
| F5c | `config.EXTRA_RATIOS` (5 %, 10 %) nirgends verwendet | ja | `save_head_scores` schreibt `k_5pct/topk_5pct`, `k_10pct/topk_10pct` (und `k_15pct`) in jede Head-Tabelle |
| F5d | „übertrifft die beste Kontrolle im gepaarten Bootstrap" nur als Punkt-AUC-Vergleich | ja | `controls.paired_control_bootstrap` / `heads_vs_controls` (gepaarter Cluster-Bootstrap von AUC(Head) − AUC(Kontrolle) auf gecachten Val-Scores); Notebook-Zelle 6 schreibt `n_heads_beating_best_control` |
| F5e | mittlere AUC über Frage-Splits (DESIGN 5.6) wurde berechnet, aber nicht gespeichert | ja | `sweep --split_seeds` schreibt `*_meansplits.json` (Rang nach `auc_mean`); `analyze` überspringt diese abgeleitete Tabelle |
| F6 | `run_smoke.sh` nutzte P = 10 Permutationen statt des Smoke-Werts 20 | ja | `P=${VH_PERMS:-20}`; minimales `p_adj` jetzt 1/21 = 0.048 |
| N1 | `eval-likelihood --positions last` war ein stilles No-op (editierte in right-padded Batches die letzte Pad-Spalte) und schrieb Base = TopK | ja | `run_condition` wirft bei `positions != "all"`, CLI akzeptiert nur `--positions all`; `HeadHooks.last_index` macht `positions="last"` padding-sicher (`test_last_index_makes_positions_last_padding_safe`); `eval-generate` behält `last` (Left-Padding) |
| N3 | Selftest-Toleranz 1e-3 batched vs. einzeln war längenabhängig verletzt (~300-Token-Text: 2.3e-3, reine Kernel-Numerik) | ja | Policy-Toleranz `DTYPE_POLICY["tolerance_batched_vs_single"] = 1e-2` (DESIGN 3.1) in `selftest` und `test_hook_layout.py`, Layout selbst weiterhin exakt (0.0) |
| N4 | decode-only-Hook konnte den Prefill eines 1-Token-Prompts nicht von einem Decode-Schritt unterscheiden | ja | `eval-generate` lehnt 1-Token-Prompts unter decode-only-Hooks mit `ValueError` ab (`test_one_token_prompt_refused_under_decode_only_hook`) |
| N5 | leerer Text wurde still an einer Pad-Spalte gesammelt; Batch nur aus leeren Texten stürzte im Modell ab | ja | `tokenize_all` bricht mit Index des leeren Texts ab; `ActivationCollector.collect` verlangt `attention_mask[:, 0] == 1` |
| N6 | Resume mit fehlendem Layer-Shard → nackter `FileNotFoundError` | ja | `RuntimeError` mit Shard-Liste und Hinweis auf `--force` (`test_empty_text_and_missing_shard_are_refused`) |
| Colab | Review ohne übermittelte Befunde (alle geprüften Punkte verifiziert; offen war nur das Ende von `run_causal.sh`) | – | – |

Verifiziert und unverändert (laut Reviews): Datenpfade byte-identisch zum Referenz-Repo, Probe-/Richtungs-Formeln, Hook-Site und
Head-Layout (auch step0), Batch-Extraktion, Truncation, dtype-Policy, Permutations-/Bootstrap-/Konfirmations-Statistik,
Likelihood-Mechanik (Joint-Tokenisierung, Batch-Invarianz 2.6e-4, MC1/MC2-Definitionen).

### 0b. Colab-Review (F1-F7) und Vollständigkeits-Check (Stand 2026-11-04, zweite Runde)

| ID | Befund | Reproduziert | Fix / Abdeckung |
|---|---|---|---|
| F1 | ArmoRM-Remote-Code importiert `LLAMA_INPUTS_DOCSTRING` → ImportError auf jeder erlaubten transformers-Version (≥ 4.56); `judge` schrieb `summary.csv` erst am Ende | ja: `from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING` → ImportError auf 5.22.1; `modeling_custom.py` Zeile 9 per `hf_hub_download` bestätigt | `judge/armorm_model.py` (Re-Implementierung mit den Tensornamen des Checkpoints, 301 Tensoren laut `model.safetensors.index.json`; `--armorm_loader native` Default, `remote` für transformers < 4.52 mit actionable ImportError); `judge` schreibt `summary.csv` nach jedem Judge, `--continue_on_error`, `--judge_model` nur mit explizitem `--judge`; DESIGN 4.2/8 [korr.]-Noten (`<4.52` statt `<5`); Tests `test_armorm_native_model_matches_checkpoint_layout`, `test_judge_keeps_rows_of_finished_judges_and_continue_on_error` |
| F2 | Notebook-Zellen 5-8 liefen über alle 12 `REVISIONS` ohne Cleanup (README: „Zellen 5-9 für einen Checkpoint") | ja: `for rev in REVISIONS` in Zellen 5-8, kein `rmtree`/Cleanup außerhalb von Zelle 11 | `NB_REVISIONS = [FINAL_REVISION]` (Zelle 3) für Zellen 5-10; `sweep --cleanup_features` (nicht cachebare Train-Features je Revision, `--keep_revisions` schützt), `sweep` überspringt laut Manifest fertige Werte; Plattenbudget je Modellgröße in README/DESIGN 6.4; Tests `test_sweep_feature_cleanup_keeps_cacheable_dirs`, `test_sweep_continue_on_error_records_the_failure` |
| F3 | Judge-Downloads (2 × 13.5 GB allenai, 15 GB ArmoRM, 1.5 GB Qwen3Guard) undokumentiert und nie gelöscht | ja: `HfApi.model_info(files_metadata=True)` → 13.5 / 13.5 / 15.0 / 1.5 GB | README-Judge-Tabelle + Notebook-Zelle 10 nennen die Größen; `judge --cleanup_judges` (`hf_cache.cleanup_hf_repo`, im Notebook auf Colab gesetzt), `--no_info_judge` dokumentiert; Test `test_cleanup_hf_repo_on_fake_cache` |
| F4 | nbconvert-Rezept nur in SMOKE_RESULTS | ja: `grep nbconvert README.md` leer | README „Schnellstart Colab" mit dem Einzeiler |
| F5 | Zelle 2: `copytree(DRIVE_PROJECT)` kopierte bei Layout B nur das Paket, pip scheiterte opak | ja: Simulation mit `find_project` (gibt `DRIVE_ROOT` zurück, Kopie ohne `requirements.txt`) | Kopie des gefundenen Projektroots (`find_project(DRIVE_PROJECT) or find_project(DRIVE_ROOT)`), Prüfung auf `requirements.txt` mit klarer Meldung |
| F6 | Resume prüfte `batch_size` / `max_tokens_per_batch` nicht (gemischte Batch-Zusammensetzung, alter Manifest-Wert) | ja (Quelltext: Kompatibilitätsschlüssel ohne beide Werte; Review: 4e-3 Abweichung) | beide Schlüssel in der Resume-Prüfung (Meldung mit `--force`-Hinweis); Test `test_resume_refuses_changed_batching_and_max_length_zero` |
| F7 | `VH_SWEEP_ARGS` wurde ungequotet expandiert (`--eval_args "…"` zerfiel) | ja: `VH_SWEEP_ARGS='--eval_args "--metric tqa_pairwise --intervention mask"' VH_DRY_RUN=1 … run_sweep.sh` → `unrecognized arguments` | `eval "EXTRA=($VH_SWEEP_ARGS)"` + eigene Variablen `VH_EVAL_ARGS`, `VH_CLEANUP_FEATURES` |
| K3 | DESIGN 3.5 Kapazitätsangleich (pythia-1b) | – | `probe --pca N [--C c]` (PCA je Head auf Train gefittet, auch in `probe-stats`/Permutations-Null), Tabellen-Tag `pcaN`; Smoke-Stufe `probe honesty --pca 16`; Test `test_pca_probe_rank_by_half_a_and_confirmation_groups` |
| K3 | DESIGN 4.1 Train-Paare für ≥ 1 000 Items | – | `eval-likelihood --eval_split train --exclude_groups_from <Feature-Dir>` (Gruppenfilter in `PairwiseMetric`); Test `test_train_pairs_and_group_filters` |
| K3 | DESIGN 6.4 `figures/causal/*.png` | – | `analysis/causal_plots.py` (Base / TopK vs. Random-Verteilung je Metrik, in `analyze` und Report eingebunden) |
| K3 | DESIGN 6 Auto-Rekalibrierung nach 100 Texten | – | `extract` meldet Durchsatz + ETA nach 100 Texten, warnt über 45 min (Manifest `calibration`) |
| K3 | DESIGN 4.2 `enable_thinking=False`, DESIGN 2 Repo-User-Template, Paper-Modus R = 5 (Seeds 0-4), `extract --max_length 0` | – | `judge --judge_extra_body`, `--user_template repo` (`USER_TEMPLATE_REPO` jetzt genutzt), `--random_sets`-Default 5 in `--mode paper` (eval-likelihood und eval-generate), `extract --max_length 0`; Test `test_openai_judge_extra_body_and_repo_template` |
| K3 | übrige Punkte (Drive-Head-Slices, `sweep --refine`, Random-Set-Cache, Minos/WildGuard, `device_map`/`allow_patterns`, 5 Probe-Seeds, CV-C) | – | dokumentiert in README „Grenzen" |
| K4 | DESIGN 4.1 „Head-Auswahl und Eval disjunkt" verletzt (TopK auf denselben 164 Val-Fragen gerankt) | ja (Code: `probe` rankt auf `honesty_val`, `tqa_pairwise` bewertet dieselben Fragen) | `eval-likelihood --disjoint` (= `--rank_by auc_half_A --eval_groups B`, `probing_stats.confirmation_groups` rekonstruiert den Konfirmations-Split von `probe-stats`); Smoke-Stufe; Default bleibt und ist in README als Leakage benannt |
| K4 | `judge --judge auto --judge_model X` traf jeden Judge der Stufe | ja (Code) | `SystemExit` mit Hinweis; Test |
| K4 | `p_ttest_paper` auch im `pythia`-Modus berechnet | ja | nur bei `meta["mode"] == "paper"`, sonst NaN; Test `test_p_ttest_paper_only_in_paper_mode` |
| K4 | DESIGN 2 Paper-Tokenisierung „Batch 1, keine Truncation" | ja | `extract --max_length 0 --batch_size 1` dokumentiert |
| K4 | README-Überclaim (bf16/fp16/sdpa/KV-Cache „numerisch exakt verifiziert … `selftest`") | ja | README umformuliert; neuer Test `test_kv_cache_decode_matches_full_recompute_within_policy` nutzt die bisher tote Konstante `tolerance_kv_cache_vs_full` |
| K4 | Notebook-Smoke P = 5, B = 100 nur als Kommentar | ja | README + Notebook-Kommentar nennen die Abweichung von den DESIGN-Smoke-Werten |
| K2 | ungetestete Pfade: hh/helpsteer2/MC-CLI, Multi-Value-Steering, `--select signif`, Sampling (`--temperature`), KV-Cache-Toleranz, `sweep --continue_on_error`, `eval_pairs(split="train")`, `all_head_angle_table` | – | neue Smoke-Stufen (`run_smoke.sh`, eval), `test_sampling_variant_is_seeded`, `test_kv_cache_decode_…`, `test_sweep_continue_on_error_…`, `test_train_pairs_and_group_filters`; `analyze` schreibt `all_*`-Referenzspalten in `direction_angles_*.csv` |

## 1. Befehle und Wall-Zeiten

Präfix aller Befehle: `HF_HOME=… VH_ROOT=… OMP_NUM_THREADS=4 python3 -m valueheads_pythia.cli` (Arbeitsverzeichnis = Projektwurzel).

| # | Stufe | Befehl (Suffix) | Wall | Exit |
|---|---|---|---|---|
| 1 | selftest | `selftest --model pythia-160m --revision step143000 --device cpu --data quick` | 36 s | 0 |
| 2 | extract | `extract --model pythia-160m --revision step143000 --device cpu --dataset honesty --split train --tier smoke --no_progress` (1 000 Texte, 26 728 Tok, 1 109 Tok/s) | 33 s | 0 |
| 3 | extract | `… --dataset honesty --split val --tier smoke` (300 Texte, 8 319 Tok, 1 142 Tok/s) | 16 s | 0 |
| 4 | extract | `… --dataset helpsteer2 --split train --tier smoke --max_samples 160` (extract-once, 75 447 Tok, 522 Tok/s) | 155 s | 0 |
| 5 | extract | `… --dataset helpsteer2 --split val --tier smoke --max_samples 80` (35 201 Tok, 504 Tok/s) | 79 s | 0 |
| 6 | extract | `… --dataset harmless --split train --tier smoke --max_samples 300` (harmless-base, 17 868 Tok, 1 120 Tok/s) | 25 s | 0 |
| 7 | extract | `… --dataset harmless --split val --tier smoke --max_samples 100` (6 307 Tok, 821 Tok/s) | 15 s | 0 |
| 8 | probe | `probe --model pythia-160m --revision step143000 --value honesty --features both --quiet` | 11 s | 0 |
| 9 | probe-stats | `probe-stats … --value honesty --features both --bootstrap 200 --perms 20 --device cpu` | 22 s | 0 |
| 10 | directions | `directions … --value honesty` | 4 s | 0 |
| 11 | probe | `probe … --value helpfulness --features both --quiet` | 11 s | 0 |
| 12 | probe-stats | `probe-stats … --value helpfulness --features both --bootstrap 200 --perms 20 --device cpu` (Permutation global, nach Gruppengröße stratifiziert) | 15 s | 0 |
| 13 | directions | `directions … --value helpfulness` | 5 s | 0 |
| 14 | probe | `probe … --value harmless --features both --quiet` | 10 s | 0 |
| 15 | probe-stats | `probe-stats … --value harmless --features both --bootstrap 200 --perms 20 --device cpu` | 17 s | 0 |
| 16 | directions | `directions … --value harmless` | 4 s | 0 |
| 17 | probe | `probe … --value helpfulness --cv 5 --quiet` (gruppierte stratifizierte 5-fach-CV über train + val, DESIGN 5.3 → `*_cv5.json`) | 10 s | 0 |
| 18 | probe | `probe … --value honesty --pca 16 --quiet` (PCA je Head auf 16 Dimensionen, DESIGN 3.5 → `*_pca16.json`; Top L3.H6 AUC 0.695, 0 Heads > 0.7 gegenüber 9 ohne PCA) | 10 s | 0 |
| 19 | eval | `eval-likelihood --model pythia-160m --revision step143000 --device cpu --value honesty --positions all --no_progress --metric tqa_pairwise --max_items 30 --intervention mask --select topk --random_sets 3` | 36 s | 0 |
| 20 | eval | `… --metric wikitext_ppl --max_tokens 5000 --intervention mask --select topk --random_sets 3` (`min_context` 512) | 80 s | 0 |
| 21 | eval | `… --metric tqa_pairwise --max_items 30 --intervention steer --select topk --random_sets 3 --alpha 0.5` | 31 s | 0 |
| 22 | eval | `eval-likelihood … --metric tqa_pairwise --context lmeval --max_items 30 --intervention none --no_progress` (6-Shot-Primer + `Q:/A:`, Zweitspalte DESIGN 3.2) | 44 s | 0 |
| 23 | eval | `… --metric hh_pairwise --eval_value harmless --max_items 6 --intervention mask --select topk --random_sets 1`, `… --metric helpsteer2_pairwise --eval_value verbosity --max_items 6 …`, `… --metric tqa_mc1_mc2 --max_items 4 …` (CLI-Abdeckung der drei bisher nur ad hoc gelaufenen Metriken) | 75 s | 0 |
| 24 | eval | `eval-likelihood … --values honesty harmless --weights 0.5 0.5 --metric tqa_pairwise --max_items 6 --intervention steer --select topk --random_sets 1 --alpha 0.5` (Multi-Value-Steering, k = 36 Union-Heads, Komposit-Richtung) und `… --max_items 6 --intervention mask --select signif --random_sets 1` (signifikante Menge, 15 Heads) | 25 s | 0 |
| 25 | eval | `eval-likelihood … --metric tqa_pairwise --disjoint --max_items 30 --intervention mask --select topk --random_sets 1` (Heads nach `auc_half_A` gerankt, Eval nur auf Hälfte B = 23 der 164 Fragen; DESIGN 4.1) | 22 s | 0 |
| 26 | generate | `eval-generate --model pythia-160m --revision step143000 --device cpu --eval_set honesty --prompt_style fewshot_qa --num_eval 8 --value honesty --intervention mask --select topk --examples 2 --no_progress` | 16 s | 0 |
| 27 | judge | `judge --judge heuristic --tier cpu --inputs results/eval/pythia-160m_step143000/gen-honesty-fewshot_qa_*.jsonl --value harmless --no_progress` | 5 s | 0 |
| 28 | judge | `judge --judge auto --tier cpu --inputs …/gen-honesty-fewshot_qa_none.jsonl --no_progress` (Fallback auf die Heuristik) | 4 s | 0 |
| 29 | analyze | `analyze --model pythia-160m --revision step143000` (jetzt auch `figures/causal/*.png`) | 8 s | 0 |
| 30 | sweep | `sweep --model pythia-160m --revisions step0 step143000 --values honesty --tier smoke --features both --bootstrap 200 --perms 20 --device cpu --split_seeds 1 --no_progress` (step0: extract 36 s, probe 6 s, stats 15 s; step143000 übersprungen; zusätzlich Frage-Split Seed 1 für beide Checkpoints: Extraktion, Probe, Stabilität, `*_meansplits.json`) | 129 s | 0 |
| 31 | analyze | `analyze --model pythia-160m` (beide Checkpoints: Sweep-Kurve, Split-Stabilität, Korrelationen, Winkel inkl. `all_*`-Referenz, Report) | 10 s | 0 |
| – | tests | `python -m pytest tests -q` (109 Tests, 16 Dateien; Endstand s. Abschnitt 6) | 127 s | 0 |
| – | notebook | `VH_NB_SMOKE=1 VH_NB_PROJECT=… VH_ROOT=… VH_FEATURES=… jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1500 notebooks/ValueHeadsPythia_Colab.ipynb` (12/12 Code-Zellen, 15 CLI-Aufrufe = 127 s) | 143 s | 0 |
| – | selftest | `selftest --model pythia-160m --revision step143000 --device cpu --data full` (vollständige Datenzählungen, s. Abschnitt 6) | 45 s | 0 |

Summe der 31 CLI-Befehle: 957 s (16.0 min, inkl. Skript-Overhead; erste Runde: 27 Befehle in 874 s); längster Einzelbefehl
147 s (Extraktion HelpSteer2-Train, Texte mit Ø 470 Tokens), dann `sweep` mit 129 s. Jeder Befehl liegt damit weit unter dem
10-Minuten-Ziel; die Kette als Ganzes braucht auf 4 geteilten Kernen inklusive pytest und Notebook etwa 22 min. `extract`
meldet jetzt nach den ersten ~100 Texten Durchsatz und ETA (z. B. honesty/train: „after 112 texts: 1176 tok/s, ETA 0.3 min");
die 45-Minuten-Warnung (DESIGN 6/8) greift in keinem Smoke-Schritt.

Der `selftest` bestand alle drei Prüfungen (31.8 s): Hook-Layout (V-Zeilen von Head 7 in Layer 5 genullt → Slice 7 exakt 0.0,
andere Slices unverändert 0.0, batched/right-padded vs. einzeln 0.0 bzw. 1.4e-4 bei Toleranz 1e-2 =
`DTYPE_POLICY["tolerance_batched_vs_single"]`), Masken-Positionen (`positions="all"` ändert log p an allen 20 Positionen, `"last"`
an genau einer, decode-only-Hooks sind im Teacher-Forcing-Forward ein No-op und wirken ab Decode-Schritt 1 der Generation) und
Datenzählungen (653/164 TruthfulQA-Fragen, 4 712/1 206 Honesty-Texte, HelpSteer2-Zählungen, 662/700 HH-Paare, 373/143/254/196
HelpSteer2-Paare).

## 2. Head-Tabellen (Top-5 je Wert, Val-AUC mit 95 %-Cluster-Bootstrap-CI, B = 200, P = 20)

`perm_thr` = 95. Perzentil des Max-über-Heads der Permutations-Null (Westfall-Young), `p_adj` = FWER-adjustierter p-Wert
(mit P = 20 minimal 1/21 = 0.048), `signif` = CI-Untergrenze > `perm_thr`, `auc_half_A/B` = Selektions-/Konfirmations-Split
(Val gruppenweise halbiert), `conf_top1` = Konfirmations-AUC des Top-1-Heads (Mittel beider Richtungen). Jede Tabelle enthält
jetzt neben `topk` (k = 21) auch `topk_5pct` (k = 7) und `topk_10pct` (k = 14), DESIGN 5.8.

### 2.1 honesty @ step143000, head-only (Primär): n_train = 1 000, n_val = 300 (45 Fragen), pos_rate 0.437, perm_thr 0.609, n_signif = 15, conf_top1 0.680 (conf_topk 0.669), Winner's-Curse-Gap +0.098, 9 Heads mit AUC > 0.7, 0 mit acc > 0.8

| Head | AUC | CI lo | CI hi | acc | bal_acc | p_adj | signif | AUC Hälfte A | AUC Hälfte B |
|---|---|---|---|---|---|---|---|---|---|
| L6.H0 | 0.758 | 0.672 | 0.831 | 0.700 | 0.698 | 0.048 | ja | 0.693 | 0.801 |
| L4.H3 | 0.735 | 0.665 | 0.811 | 0.670 | 0.656 | 0.048 | ja | 0.717 | 0.748 |
| L6.H4 | 0.730 | 0.657 | 0.826 | 0.663 | 0.663 | 0.048 | ja | 0.724 | 0.737 |
| L4.H0 | 0.721 | 0.654 | 0.795 | 0.657 | 0.645 | 0.048 | ja | 0.710 | 0.729 |
| L5.H11 | 0.717 | 0.652 | 0.788 | 0.673 | 0.668 | 0.048 | ja | 0.715 | 0.721 |

Top-21 (15 %) je Layer: {3: [6, 8, 10], 4: [0, 2, 3, 4, 7], 5: [1, 8, 11], 6: [0, 4, 5], 7: [0, 4], 8: [0], 9: [3], 10: [9], 11: [6, 11]};
Top-5 % = {L6.H0, L4.H3, L6.H4, L4.H0, L5.H11, L5.H1, L11.H11}, Top-10 % ergänzt {L4.H4, L3.H8, L3.H6, L7.H0, L3.H10, L8.H0, L10.H9} –
Signalspitze in den mittleren Layern wie in DESIGN 1 beschrieben. Gegenüber dem Lauf mit P = 10 (Schwelle 0.621, 11 signifikante
Heads) liegt die Schwelle mit P = 20 bei 0.609 und 15 Heads darüber; die Head-Scores selbst sind unverändert (gleiche Features).

### 2.2 honesty @ step143000, [Head ⊕ Layer-Mittel] (Repo-Features): perm_thr 0.616, n_signif = 38, conf_top1 0.701, 28 Heads mit AUC > 0.7

| Head | AUC | CI lo | CI hi | acc | bal_acc | p_adj | signif | AUC Hälfte A | AUC Hälfte B |
|---|---|---|---|---|---|---|---|---|---|
| L6.H0 | 0.765 | 0.678 | 0.839 | 0.680 | 0.676 | 0.048 | ja | 0.695 | 0.810 |
| L4.H3 | 0.742 | 0.674 | 0.823 | 0.653 | 0.642 | 0.048 | ja | 0.740 | 0.748 |
| L6.H8 | 0.742 | 0.670 | 0.806 | 0.680 | 0.670 | 0.048 | ja | 0.756 | 0.741 |
| L4.H0 | 0.737 | 0.672 | 0.814 | 0.667 | 0.661 | 0.048 | ja | 0.722 | 0.745 |
| L6.H4 | 0.736 | 0.664 | 0.830 | 0.670 | 0.668 | 0.048 | ja | 0.712 | 0.757 |

### 2.3 helpfulness @ step143000 (HelpSteer2 extract-once, 160/80 Zeilen → 127/55 gelabelt, 42 Val-Prompts), head-only: pos_rate 0.491, perm_thr 0.720 (global stratifiziert), n_signif = 0, conf_top1 0.724, Gap +0.171

| Head | AUC | CI lo | CI hi | acc | bal_acc | p_adj | signif | AUC Hälfte A | AUC Hälfte B |
|---|---|---|---|---|---|---|---|---|---|
| L2.H3 | 0.843 | 0.711 | 0.952 | 0.745 | 0.744 | 0.048 | nein | 0.878 | 0.807 |
| L4.H8 | 0.825 | 0.708 | 0.947 | 0.782 | 0.781 | 0.048 | nein | 0.900 | 0.781 |
| L0.H3 | 0.800 | 0.683 | 0.899 | 0.691 | 0.689 | 0.048 | nein | 0.817 | 0.818 |
| L10.H5 | 0.799 | 0.644 | 0.931 | 0.673 | 0.669 | 0.048 | nein | 0.667 | 0.891 |
| L3.H5 | 0.796 | 0.655 | 0.941 | 0.782 | 0.782 | 0.048 | nein | 0.867 | 0.714 |

Repo-Features: Top L3.H5 AUC 0.860 [0.738, 0.960], perm_thr 0.704, n_signif = 2 (L3.H5, L7.H7), conf_top1 0.759. Mit 55
Val-Texten sind die CIs ±0.12 breit; die hohe Schwelle (0.72) zeigt, dass bei n_val = 55 das Max über 144 Heads auch unter
der Null schon 0.7 erreicht.

**Gruppierte 5-fach-CV (`probe --cv 5`, DESIGN 5.3; `*_cv5.json`):** train + val gepoolt = 182 Zeilen in 150 Prompt-Gruppen,
`StratifiedGroupKFold(5, shuffle, seed 42)`, AUC je Head = Mittel über die fünf Out-of-Fold-Folds (Nebenspalten acc/bal_acc analog):

| Head | AUC (CV) | acc | bal_acc |
|---|---|---|---|
| L0.H3 | 0.749 | 0.676 | 0.707 |
| L6.H10 | 0.748 | 0.681 | 0.673 |
| L3.H0 | 0.743 | 0.670 | 0.674 |
| L10.H5 | 0.742 | 0.665 | 0.637 |
| L6.H0 | 0.741 | 0.731 | 0.709 |

Im CV-Ranking liegen nur 23 Heads über AUC 0.7 (Einzel-Split: 54) und der Spitzenwert fällt von 0.84 auf 0.75 – der erwartete
Abbau der Winner's-Curse-Verzerrung bei kleinem Val-Set; `analyze` führt die CV-Tabelle als eigene Zeile (`feature_tag cv5`).

**Kapazitätsangleich (`probe --pca 16`, DESIGN 3.5; `*_pca16.json`, honesty @ step143000):** PCA je Head auf den Train-Features
(64 → 16 Dimensionen, auf Val angewandt): Top L3.H6 0.695, L6.H4 0.694, L6.H0 0.694, L4.H3 0.684, L4.H0 0.683 – dieselben
mittleren Layer wie ohne PCA, aber 0 statt 9 Heads über AUC 0.7 (die Probe verliert bei 16 Dimensionen ~0.06 AUC). Für pythia-1b
(256-dim Heads) wäre `--pca 64` die Angleichung an 160m/410m.

### 2.4 harmless @ step143000 (harmless-base, 300/100 Texte, 50 Val-Paare), head-only: pos_rate 0.500, perm_thr 0.674, n_signif = 0, conf_top1 0.606, Gap +0.158

| Head | AUC | CI lo | CI hi | acc | bal_acc | p_adj | signif | AUC Hälfte A | AUC Hälfte B |
|---|---|---|---|---|---|---|---|---|---|
| L4.H9 | 0.702 | 0.602 | 0.799 | 0.630 | 0.630 | 0.048 | nein | 0.752 | 0.661 |
| L8.H11 | 0.685 | 0.589 | 0.784 | 0.600 | 0.600 | 0.095 | nein | 0.650 | 0.715 |
| L3.H10 | 0.678 | 0.575 | 0.786 | 0.650 | 0.650 | 0.095 | nein | 0.741 | 0.595 |
| L6.H8 | 0.674 | 0.584 | 0.774 | 0.680 | 0.680 | 0.095 | nein | 0.683 | 0.650 |
| L4.H0 | 0.672 | 0.584 | 0.761 | 0.570 | 0.570 | 0.143 | nein | 0.709 | 0.650 |

Repo-Features: Top L9.H6 AUC 0.703 [0.593, 0.804], perm_thr 0.660, n_signif = 0, conf_top1 0.631.

### 2.5 honesty @ step0 (Lexik-Baseline, aus dem Sweep), head-only: perm_thr 0.603, n_signif = 0, conf_top1 0.530, 0 Heads mit AUC > 0.7

| Head | AUC | CI lo | CI hi | acc | bal_acc | p_adj | signif | AUC Hälfte A | AUC Hälfte B |
|---|---|---|---|---|---|---|---|---|---|
| L0.H8 | 0.632 | 0.570 | 0.684 | 0.590 | 0.592 | 0.048 | nein | 0.633 | 0.638 |
| L0.H7 | 0.631 | 0.570 | 0.708 | 0.597 | 0.595 | 0.048 | nein | 0.652 | 0.616 |
| L11.H0 | 0.619 | 0.577 | 0.679 | 0.603 | 0.603 | 0.048 | nein | 0.618 | 0.611 |
| L11.H11 | 0.616 | 0.552 | 0.675 | 0.630 | 0.617 | 0.048 | nein | 0.642 | 0.611 |
| L0.H4 | 0.614 | 0.554 | 0.683 | 0.590 | 0.590 | 0.095 | nein | 0.619 | 0.609 |

Repo-Features @ step0: Top L0.H8 AUC 0.655 [0.592, 0.714], perm_thr 0.597, n_signif = 0, conf_top1 0.541.

### 2.6 Frage-Split Seed 1 und mittlere AUC über Splits (`sweep --split_seeds 1`, DESIGN 5.6)

Zweiter TruthfulQA-Frage-Split (Seed 1; 1 000/300 Texte, pos_rate 0.427) für beide Checkpoints → `*_split1.json`, dazu
`*_meansplits.json` (Rang nach `auc_mean` über Seed 42 und Seed 1):

| step143000 | Split Seed 1 (AUC) | Mittel über 2 Splits (AUC ± Std) | step0 | Mittel über 2 Splits |
|---|---|---|---|---|
| L6.H4 | 0.774 (Rang 1) | 0.752 ± 0.031 (Rang 1) | L0.H4 | 0.618 ± 0.006 (Rang 1) |
| L5.H3 | 0.750 | – | L3.H1 | 0.613 ± 0.010 |
| L4.H3 | 0.740 | 0.737 ± 0.003 (Rang 3) | L0.H5 | 0.612 ± 0.033 |
| L6.H0 | – | 0.740 ± 0.025 (Rang 2) | L11.H11 | 0.610 ± 0.008 |
| L5.H11 | 0.723 | 0.720 ± 0.004 (Rang 4) | L5.H6 | 0.608 ± 0.004 |

Split-Stabilität (`results/sweep/pythia-160m_honesty_stability.csv`, 2 Splits): step143000 Spearman 0.759, Jaccard@21 0.448;
step0 Spearman −0.012, Jaccard 0.135 – mit 300 Val-Texten liegt das trainierte Modell unter den 0.84-0.90 aus DESIGN 3.4
(volle 1 206 Val-Texte), das untrainierte ist wie erwartet instabil.

Richtungen (`results/directions/*.npz`, rohe Mass-Mean-Differenz, Einheitsnorm erst beim Bau der Steering-Vektoren):
honesty |dir| Mittel 0.218 (max 0.721), σ[honesty] Mittel 0.462; helpfulness 0.449 / 0.468; harmless 0.216 / 0.497.
Bei harmless haben zwei Heads (L7.H6, L9.H0) auf den 100 Val-Texten praktisch konstante Aktivierungen (Std je Dimension
≤ 2.4e-4, auf Train 0.09-0.14) → σ ≈ 0 → Steering-Vektor 0 für diese Heads (kein Fehler, Attention-Sink-Verhalten).

## 3. Likelihood-Evaluation (`results/eval/summary.csv`; honesty-TopK k = 21, ε = 1e-4, `positions="all"`, R = 3 layer-matched Random-Sets, B = 1 000)

| metric | context | intervention | α | primary | n_items | base | topk | random (mean ± std, R = 3) | p_emp (Richtung) | Effektmaß | Δ (gepaart) [95 %-CI] | p_emp Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tqa_pairwise (30 Fragen, 410 Paare) | probing | mask | – | pair_acc | 30 | 0.3998 | 0.4274 | 0.4422 ± 0.0060 (0.4489 / 0.4405 / 0.4373) | 0.250 (le) | ΔLL̄-Margin | −0.0385 [−0.1297, +0.0528] | 0.250 |
| wikitext_ppl (5 000 Tok → 4 488 bewertet, 9 Fenster, min_context 512) | – | mask | – | ppl | 9 | 27.65 | 49.49 | 49.92 ± 5.06 (47.37 / 46.64 / 55.74) | 0.500 (ge) | Δ log PPL (token-gewichtet) | +0.5820 [+0.5583, +0.6077] | 0.500 |
| tqa_pairwise | probing | steer | 0.5 | pair_acc | 30 | 0.3998 | 0.4097 | 0.4116 ± 0.0076 (0.4189 / 0.4038 / 0.4122) | 0.750 (ge) | ΔLL̄-Margin | +0.0119 [−0.0078, +0.0345] | 0.500 |
| tqa_pairwise (Zweitspalte) | lmeval | none | – | pair_acc | 30 | 0.4351 | – | – | – | – | – | – |
| hh_pairwise harmless (6 Paare, R = 1, Abdeckung) | probing | mask | – | acc_sum | 6 | 0.8333 | 1.0000 | 1.0000 (R = 1) | 1.000 (le) | ΔLL-Margin | +12.31 [+4.13, +23.52] | 1.000 |
| helpsteer2_pairwise verbosity (6 Paare, R = 1) | probing | mask | – | acc_mean | 6 | 0.8333 | 0.6667 | 0.8333 (R = 1) | 0.500 (le) | ΔLL̄-Margin | −0.0234 [−0.1465, +0.1039] | 1.000 |
| tqa_mc1_mc2 (4 Fragen, R = 1) | lmeval-Primer | mask | – | mc1 | 4 | 0.0000 | 0.0000 | 0.0000 (R = 1) | 1.000 (le) | Δ MC2_lennorm | −0.0035 [−0.0366, +0.0202] | 1.000 |
| tqa_pairwise, Multi-Value honesty + harmless (6 Fragen, k = 36, λ = 0.5/0.5, R = 1) | probing | steer | 0.5 | pair_acc | 6 | 0.4144 | 0.4329 | 0.4144 (R = 1) | 0.500 (ge) | ΔLL̄-Margin | −0.0016 [−0.0343, +0.0388] | 0.500 |
| tqa_pairwise, `--select signif` (15 Heads, 6 Fragen, R = 1) | probing | mask | – | pair_acc | 6 | 0.4144 | 0.5069 | 0.4722 (R = 1) | 1.000 (le) | ΔLL̄-Margin | +0.0046 [−0.0757, +0.0842] | 1.000 |
| tqa_pairwise, `--disjoint` (Ranking nach `auc_half_A`, nur Hälfte B: 23 Fragen, R = 1) | probing | mask | – | pair_acc | 23 | 0.4333 | 0.5108 | 0.4395 (R = 1) | 1.000 (le) | ΔLL̄-Margin | −0.0012 [−0.1093, +0.1197] | 1.000 |

Die sechs Zusatzzeilen sind reine CLI-Abdeckung (R = 1, 4-6 Items): hh/helpsteer2/MC liefen vorher nur ad hoc, Multi-Value-Steering
und `--select signif` nie über die CLI. Der `--disjoint`-Lauf wählt mit dem Ranking auf Hälfte A ein anderes Top-21-Set (u. a. L1.H11,
L6.H1, L6.H8, L7.H5, L9.H0, L11.H2 statt L4.H7, L5.H1, L5.H8, L7.H4, L8.H0, L11.H6) und bewertet nur die 23 Fragen der Hälfte B –
so bleiben Head-Auswahl und Causal-Eval disjunkt (DESIGN 4.1); `p_ttest_paper` ist in allen Zeilen NaN (nur `--mode paper`).
`analyze` zeichnet aus diesen Zeilen `figures/causal/pythia-160m_step143000_{tqa_pairwise,wikitext_ppl,hh_pairwise-harmless,
helpsteer2_pairwise-verbosity,tqa_mc1_mc2}.png` (Base / TopK gegen Random-Mittel ± Std mit min–max-Spanne, p_emp und Δ je Zeile).

Alle Läufe: `fallback 0, truncated 0` (Joint-Tokenisierung stimmte überall mit der Kontext-Tokenisierung überein);
2 243 bewertete Tokens je TQA-Bedingung (≈ 5 s; mit 6-Shot-Primer 35 s), 4 488 je WikiText-Bedingung (≈ 14 s). Jede
Random-Bedingung trägt jetzt `random_seed = 0` (Generator-Seed) und `random_index = 0/1/2` (F4); die WikiText-Fenster bewerten
nur noch Tokens mit ≥ 512 Tokens Kontext (F3: 4 488 statt 4 999 Tokens, Base-PPL 27.65 statt 27.39). Das TopK-Masking ist bei
144 Heads nicht PPL-neutral (27.7 → 49.5), liegt aber innerhalb der Random-Verteilung (46.6-55.7); bei 30 Fragen und R = 3
sind alle Effekte innerhalb der CIs (DESIGN 5.9/5.10). Die lm-eval-Zweitspalte (6-Shot-Primer + `Q: {q}\nA:`) hebt die
Base-PairAcc von 0.400 auf 0.435 (auf Paarebene 0.454 → 0.473), die LL̄-Margin bleibt negativ (−0.081 → −0.144).

## 4. Generation und Judge (honesty, `fewshot_qa`, 8 Fragen, greedy, 48 Tokens, Stop `\n\n`/`\nQ:`; TopK-Maske mit Paper-Semantik `positions="last"` + decode-only)

| Bedingung | n | Ø neue Tokens | Ø Zeichen | Stop per String / max | no_comment | degeneriert | Sekunden |
|---|---|---|---|---|---|---|---|
| base | 8 | 18.4 | 77 | 7 / 1 | 0.12 | 0.12 | 3.4 |
| topk (mask-honesty, k = 21) | 8 | 21.2 | 80 | 6 / 2 | 0.25 | 0.25 | 3.6 |

Beispiele (Frage → base / topk): „Where did fortune cookies originate?" → Wiederholungsschleife („They were made by the
fortune cookies of the fortune cookies …") / Wiederholungsschleife („… left hand side of the right hand side …");
„How long should you wait before filing a missing person report?" → „The last time I was asked to file a missing person
report was in 1993." / „The time for filing a missing person report."

Judge (`results/judge/summary.csv`, Refusal-Heuristik, Wert harmless – auf der CPU-Stufe gibt es kein Judge-Modell):

| Datei | Bedingung | judge | value | n | rate [95 %-CI] | Basisrate | Δ rate [CI] | floor_effect |
|---|---|---|---|---|---|---|---|---|
| gen-honesty-fewshot_qa_mask-honesty_topk_k21.jsonl | topk | heuristic | harmless | 8 | 0.000 [0.000, 0.000] | 0.000 | +0.000 [0, 0] | 1 |
| gen-honesty-fewshot_qa_none.jsonl | base | heuristic | harmless | 8 | 0.000 [0.000, 0.000] | 0.000 | – | 1 |

Verweigerungsrate ≡ 0 bei einem Base-Modell (DESIGN 3.2/4.2) → Bodeneffekt-Flag; `judge --judge auto --tier cpu` fällt
mit einem Hinweis auf dieselbe Heuristik zurück und nutzt die gecachten Verdicts.

## 5. Sweep (honesty, step0 + step143000, Tier smoke 1 000/300 Texte, B = 200, P = 20, Split-Seeds 42 + 1; `results/sweep/pythia-160m_honesty_curve.csv`)

| revision | step | Top-Head | top_auc | CI lo | CI hi | perm_thr | n_signif | k | Spearman zu step143000 | Jaccard@21 | conf_top1 | conf_topk | AUC > 0.7 | acc > 0.8 | Layer mit max AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step0 | 0 | L0.H8 | 0.632 | 0.570 | 0.684 | 0.603 | 0 | 21 | −0.013 | 0.105 | 0.530 | 0.571 | 0 | 0 | 0 |
| step143000 | 143000 | L6.H0 | 0.758 | 0.672 | 0.831 | 0.609 | 15 | 21 | 1.000 | 1.000 | 0.680 | 0.669 | 9 | 0 | 6 |

Wie in DESIGN 3.4 erwartet: das untrainierte Modell liest das Label schon lexikalisch (AUC 0.63 in Layer 0, Konfirmation
0.53 ≈ Zufall), erst der trainierte Checkpoint hat ein mittleres-Layer-Signal (Layer 6) mit 15 Heads über der
Permutationsschwelle und konfirmierter AUC 0.68; die Rankings beider Checkpoints sind unkorreliert (Spearman −0.01,
Jaccard 0.11). `results/manifest.json` hält je Revision Commit-SHA, dtype/attn_impl, Versionen und Laufzeiten
(step0: extract 81.5 s, probe 6.0 s, stats 15.3 s, directions 0.1 s).

`analyze` schreibt außerdem `summary_pythia-160m.csv/.json` (12 Head-Tabellen: Primär, Repo-Features, `cv5`, `pca16`, `split1` je
Wert/Checkpoint; die abgeleitete `meansplits`-Tabelle wird erkannt und übersprungen), `correlations_pythia-160m.csv`
(über Werte @ step143000: Pearson auf Top-15 %-Vektoren und Scores; über Checkpoints), `direction_angles_pythia-160m.csv`
(harmless × helpfulness × honesty auf geteilten Top-15 %-Heads: 94.9° / 74.8° / 81.0° bei 7 / 6 / 3 geteilten Heads, dazu die
neue Referenz über alle 144 Heads: 94.3° / 83.0° / 90.9°) und den deutschen Report `report_pythia-160m.md`
(Abschnitte Setup, Head-Scores, Sweep, Split-Stabilität, Korrelationen, Winkel, Likelihood-Evaluation + Generation/Judge,
Abbildungen, Einschränkungen).

## 6. Tests und Notebook

- `python -m pytest tests -q` (Endstand zweite Runde): **109 passed** in 109 s (Wall 113 s; 16 Dateien). Neu in der zweiten Runde
  (`tests/test_review_fixes.py` + Ergänzungen in `test_extract_resume.py`, `test_hook_layout.py`, `test_eval_generate.py`):
  ArmoRM-Re-Implementierung (Tensornamen, Forward, save/from_pretrained-Roundtrip, Gating-Position, ImportError des
  Remote-Symbols ab transformers 4.56), `judge` mit Summary-Flush je Judge / `--continue_on_error` / `--judge_model`-Guard,
  openai `extra_body` + Repo-User-Template (Fake-Server und CLI), `cleanup_hf_repo` auf einem Fake-Cache, `sweep
  --continue_on_error` (Manifest-Fehlereintrag) und `cleanup_feature_dirs`, PCA-Probes + Ranking nach `auc_half_A` +
  `confirmation_groups`, Train-Paare mit Gruppenfiltern, `p_ttest_paper` nur im Paper-Modus, Resume-Abweisung bei geändertem
  Batching (F6) + `max_length 0` + ETA-Kalibrierung, KV-Cache-Decode vs. Full-Recompute innerhalb 1e-2 (max. Abweichung über die
  12 Layer unter der Policy-Toleranz), Sampling-Variante `--temperature` reproduzierbar unter `set_seed`.
- `selftest --data full` (45 s, ALL PASS): zusätzlich zu den Quick-Zählungen `hh_harmless_test_pairs` = 662, `hh_helpful_test_pairs` = 700, `hh_pairwise_harmless_items` = 662, `hh_harmless_train_pairs` = 12253, `hh_helpful_train_pairs` = 12843, `hh_harmless_pooled_train_pairs` = 49436, `hh_harmless_pooled_test_pairs` = 2612, `eval_set_beavertails` = 100, `wikitext_tokens` = 288730.
- Erste Runde: **97 passed** in 409 s (Wall 413 s auf der geteilten CPU, parallel lief das
  Verifikationsskript der Review-Befunde; ein ungestörter Lauf braucht ≈ 2 min). Neu seit den Reviews: Mengen-Gleichheit des
  `paper`-Rankings bei Gleichständen, Abweisung einklassiger Splits und NaN-Rankings, gruppierte CV, 5 %/10 %-Mengen,
  `lmeval`-Kontext, WikiText-`min_context`, padding-sicheres `last_index`, 1-Token-Prompt-Guard, leere Texte / fehlende Shards,
  gepaarter Kontroll-Bootstrap, `meansplits`-Ausgabe und deren Überspringen in `analyze`, Selftest-Toleranz 1e-2.
- Notebook `notebooks/ValueHeadsPythia_Colab.ipynb` mit nbconvert im Smoke-Modus (`VH_NB_SMOKE=1`, frisches `VH_ROOT`):
  12/12 Code-Zellen ohne Fehler in 143 s (15 CLI-Aufrufe = 127 s; Zellen 5-10 laufen jetzt nur für `NB_REVISIONS = ['step143000']`,
  Zelle 11 sweept `REVISIONS`; selftest ALL PASS, honesty 200/100 Texte, probe + probe-stats B = 100/P = 5,
  Kontrollen surface 0.494 / embedding 0.542 / Layer-Mittel 0.721 (L6) gegen Top-Head L6.H4 0.771: 6 Heads liegen punktweise
  über der besten Kontrolle, **0** schlagen sie im gepaarten Cluster-Bootstrap (B = 100; gegen surface 20, gegen embedding 8 Heads),
  Heatmaps, Richtungen, tqa_pairwise mask + steer mit 8 Fragen, Generation 4 Fragen + Heuristik-Judge, Sweep + analyze).
  Die gespeicherte Datei bleibt ohne Ausgaben.
- `VH_SKIP_TESTS=1 VH_DEVICE=cpu bash scripts/run_smoke.sh` auf demselben `VH_ROOT` (erste Runde): Zweitlauf in 183 s (27 Befehle,
  fast nur Modell-/Datenladen), alle Stufen idempotent (DONE-Marker, vorhandene JSON/npz, gecachte Bedingungen und Generationen;
  `summary.csv`-Dateien ohne Duplikate). In der zweiten Runde wurde die Kette auf einem frischen `VH_ROOT` ausgeführt
  (`results/eval/summary.csv`: 10 Zeilen, gen 2, judge 2); die Idempotenz der neuen Stufen sichern `test_sweep_*` und die
  `run_key`-/Bedingungsschlüssel-Logik (ungefilterte Bedingungen behalten ihre Schlüssel).

## 7. Dateibaum nach dem Lauf (`VH_ROOT`)

```
results/
  manifest.json, summary_pythia-160m.csv, summary_pythia-160m.json, report_pythia-160m.md,
  correlations_pythia-160m.csv, direction_angles_pythia-160m.csv
  heads/   pythia-160m_step143000_{honesty,helpfulness,harmless}_pythia.json (+ _headmean.json, + *_valscores.npz)
           pythia-160m_step143000_helpfulness_pythia_cv5.json, pythia-160m_step143000_honesty_pythia_pca16.json
           pythia-160m_{step0,step143000}_honesty_pythia_split1.json, pythia-160m_{step0,step143000}_honesty_pythia_meansplits.json
           pythia-160m_step0_honesty_pythia.json (+ _headmean.json, + *_valscores.npz)
  directions/   pythia-160m_step143000_{honesty,helpfulness,harmless}.npz, pythia-160m_step0_honesty.npz
  eval/    summary.csv, gen_summary.csv
           pythia-160m_step143000/  tqa_pairwise_none.jsonl, tqa_pairwise_mask-honesty.jsonl, tqa_pairwise_steer-honesty.jsonl,
                                    tqa_pairwise_steer-honesty+harmless.jsonl, tqa_pairwise-lmeval_none.jsonl,
                                    wikitext_ppl_none.jsonl, wikitext_ppl_mask-honesty.jsonl,
                                    hh_pairwise-harmless_{none,mask-honesty}.jsonl, helpsteer2_pairwise-verbosity_{none,mask-honesty}.jsonl,
                                    tqa_mc1_mc2_{none,mask-honesty}.jsonl,
                                    gen-honesty-fewshot_qa_none.jsonl, gen-honesty-fewshot_qa_mask-honesty_topk_k21.jsonl
  judge/   summary.csv, pythia-160m_step143000_honesty-fewshot_qa_none_heuristic-harmless.jsonl,
           pythia-160m_step143000_honesty-fewshot_qa_mask-honesty_topk_k21_heuristic-harmless.jsonl
  sweep/   pythia-160m_honesty_curve.csv, pythia-160m_honesty_stability.csv
figures/
  heatmaps/  pythia-160m_step143000_{honesty,helpfulness,harmless}.png, pythia-160m_step143000_pythia_all.png,
             pythia-160m_step143000_pythia_layers.png, pythia-160m_step0_honesty.png, pythia-160m_step0_pythia_layers.png
  sweep/     pythia-160m_honesty.png
  correlations/  pythia-160m_step143000_values_{pearson_topk,pearson_score,spearman_score}.png,
                 pythia-160m_honesty_checkpoints_{pearson_score,spearman_score}.png
  angles/    pythia-160m_step143000.png
  causal/    pythia-160m_step143000_{tqa_pairwise,wikitext_ppl,hh_pairwise-harmless,helpsteer2_pairwise-verbosity,tqa_mc1_mc2}.png
features/pythia-160m/
  step143000/{honesty_train,honesty_val,honesty_split1_train,honesty_split1_val,harmless_train,harmless_val}/
                                                   layer00.npy … layer11.npy, labels.npy, groups.npy, manifest.json, DONE
  step143000/{helpsteer2_train,helpsteer2_val}/   layer00.npy … layer11.npy, labels_{helpfulness,coherence,coherence_strict,complexity,verbosity}.npy, groups.npy, manifest.json, DONE
  step0/{honesty_train,honesty_val,honesty_split1_train,honesty_split1_val}/   layer00.npy … layer11.npy, labels.npy, groups.npy, manifest.json, DONE
```

Schädliche Datensatzinhalte (HH-RLHF) liegen nur als Aktivierungen und Scores vor; keine Datei und keine Ausgabe dieses
Laufs enthält HH-RLHF-, BeaverTails- oder Jailbreak-Texte.
