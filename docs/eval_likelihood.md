# Likelihood-Evaluation, Statistik und Kontrollen (DESIGN 4.1, 5, 5.9)

Dieser Abschnitt beschreibt die judge-freie Primär-Evaluation (`eval-likelihood`), die zugehörige Statistik
(`stats.py`) und die Kontrollen (`controls.py`). Alle Metriken werden mit Teacher Forcing berechnet: ein Forward
pro (Kontext, Fortsetzung), batched mit Right-Padding, Kontext und Fortsetzung gemeinsam tokenisiert
(Grenze = Länge der Kontext-Tokenisierung, Fortsetzung mit führendem Leerzeichen, kein BOS). Masken und
Steering wirken mit `positions="all"` an allen Token-Positionen; die Decode-only-Hooks des Referenz-Repos wären
in einem Teacher-Forcing-Forward per Konstruktion wirkungslos, und `positions="last"` würde in einem right-padded
Batch nur die letzte (Pad-)Spalte editieren, nie eine bewertete Position. `run_condition` und die CLI lehnen beide
deshalb ab (`--positions` akzeptiert nur `all`; die Paper-Semantik `last` + decode-only bleibt dem Generationspfad
`eval-generate` mit Left-Padding vorbehalten).

## Metriken (`valueheads_pythia/eval_likelihood.py`)

| `--metric` | Daten | Primärstatistik | Effektmaß (gepaart, Bootstrap-CI) | Nebenspalten |
|---|---|---|---|---|
| `tqa_pairwise` | 164 Val-Fragen (Seed 42), alle (korrekt, inkorrekt)-Paare, Kontext `Question: {q} Answer:` (`--context probing`, Default) oder lm-eval-6-Shot-Primer + `Q: {q}\nA:` (`--context lmeval`, Zweitspalte nach DESIGN 3.2; eigene Datei `tqa_pairwise-lmeval_*.jsonl`, Spalte `context` in `summary.csv`) | `pair_acc` (Mittel über Fragen der Paar-Trefferquote mit `LL̄`) | `delta_llbar_margin` = Δ über Paare von `LL̄(a⁺) − LL̄(a⁻)`, Cluster-Bootstrap über Fragen | `pair_acc_pairs`, `margin_mean` |
| `tqa_mc1_mc2` | `multiple_choice`-Config der Val-Fragen, 6-Shot-Primer (lm-eval) + `Q: {q}\nA:` | `mc1` (argmax Σ-LL) | `delta_mc2_lennorm` | `mc2`, `mc2_lennorm`, `*_zeroshot`, `chance_mc1` |
| `hh_pairwise` | `harmless-base` (662) bzw. `helpful-base` (700) Test-Paare, Kontext bis `\n\nAssistant:` | `acc_sum` (Σ-LL) | `delta_ll_margin` = Δ von `LL(y⁺) − LL(y⁻)` | `acc_mean`, `margin_sum`, `margin_mean`, `len_baseline` |
| `helpsteer2_pairwise` | Validation-Prompts mit zwei Antworten, `|Δr| ≥ 1` (Primär), `≥ 2` als strikte Teilmenge | `acc_mean` (`LL̄`) | `delta_llbar_margin` | `acc_sum`, `len_baseline`, `acc_mean_ge2`, `acc_sum_ge2`, `n_ge2` |
| `wikitext_ppl` | WikiText-2 raw test, Fenster 1024 / Stride 512, token-level; jedes Fenster bewertet nur seine letzten 512 Tokens (`min_context` = Fenster − Stride = 512, DESIGN 4.1 Punkt 5; die ersten 512 Tokens des Korpus werden nie bewertet; `--ppl_min_context 0` = HF-Rezept, bei dem das erste Fenster alle Tokens ab Position 1 bewertet) | `ppl` | `delta_log_ppl` (token-gewichtet über Fenster) | `log_ppl`, `n_tokens_scored` |

Bedingungen je Lauf: `base`, `topk` (bzw. `signif` = signifikante Menge aus `probe-stats`), `R` layer-matched
Random-Head-Sets (gleiche Anzahl Heads je Layer wie das TopK-Set, alle R Sets nacheinander aus **einer**
`RandomState(--random_seed)` gezogen; jede Random-Bedingung protokolliert `random_seed` = Generator-Seed und
`random_index` = r, denn `random1` eines Seed-0-Laufs ist nicht `random0` eines Seed-1-Laufs; `--mode paper`:
uniforme Zufallsmengen mit `random.Random(random_seed + r)` je Set wie im Repo) und beim Steering je `--alpha` eine
TopK- und R Random-Bedingungen. Empirischer
einseitiger p-Wert `p = (1 + #{r : metric_r ≤ metric_topk}) / (1 + R)` ("niedriger = stärker" beim Maskieren;
`≥` beim Steering und bei der Perplexität); Effektgröße = gepaarte Differenz pro Item mit Item-Bootstrap-CI
(B = 1000, TQA-Paare gruppiert nach Frage, PPL-Fenster token-gewichtet). Der Einstichproben-t-Test des Repos
(`ttest_paper_mode`) wird als Spalte `p_ttest_paper` nur im `paper`-Modus belegt (n = 5 uniforme Seeds, dort Default
`--random_sets 5`); bei den layer-matched Sets des `pythia`-Modus ist er NaN (DESIGN 5.9).

**Disjunkte Head-Auswahl und Evaluation (DESIGN 4.1 / 5.5).** Per Default wird das TopK-Set auf den 164 Val-Fragen
gerankt, die `tqa_pairwise` bewertet (nur der Probe-Fit ist disjunkt). `--disjoint` (= `--rank_by auc_half_A
--eval_groups B`) rankt die Heads nach ihrer AUC auf Hälfte A des Konfirmations-Splits von `probe-stats`
(`auc_half_A` in der Head-Tabelle; Split aus `(*_valscores.npz`-Gruppen, Seed)` rekonstruiert, `probing_stats.confirmation_groups`)
und bewertet nur die Fragen / Paare / Prompts der Hälfte B. `--eval_split train --exclude_groups_from <Feature-Dir>`
bewertet die Train-Paare (HH: 12 253 harmless-base, HelpSteer2: 3 632 helpfulness bei ≥ 1 …), die nicht im
Probing-Train-Subsample lagen (`groups.npy`). Gefilterte Läufe tragen `n_items_unfiltered` und einen Hash des Filters in
der Bedingungskonfiguration, ungefilterte Bedingungen behalten ihre gespeicherten Schlüssel.

**Abbildungen.** `analyze` zeichnet je (Checkpoint, Metrik, Intervention) Base / TopK gegen die Random-Verteilung
(`figures/causal/*.png`, `analysis/causal_plots.py`) und bindet sie in den Report ein.

Ausgaben: `results/eval/{model}_{revision}/{metric[-eval_value]}_none.jsonl` (Base) und
`..._{mask|steer}-{value}.jsonl` (eine Zeile je Bedingung mit Head-Set, Summary, Item-Scores ohne Texte und
Laufstatistik) sowie `results/eval/summary.csv` (eine Zeile je Lauf). Gespeicherte Bedingungen werden bei
identischer Konfiguration wiederverwendet (`--force` erzwingt Neuberechnung).

## Kontrollen (`valueheads_pythia/controls.py`)

`random_sets_layer_matched` (Random-Sets für die Causal-Tests; Präfix-stabil in R), `surface_probe`
(logistische Regression auf log Länge, mittlere Token-Logfrequenz aus einer Unigram-Tabelle der Probing-Texte,
Refusal-Flag, Satzzeichen-Anteil), `embedding_probe` (Last-Token-Eingabe-Embedding, rein lexikalisch) und
`layer_mean_probe` (Mittel über die H Heads eines Layers aus den extrahierten Features). Ein Head gilt nach
DESIGN 5.7 nur dann als wert-sensitiv, wenn seine AUC die beste dieser Kontrollen **im gepaarten Cluster-Bootstrap**
übertrifft: `paired_control_bootstrap` / `heads_vs_controls` resamplen die Val-Gruppen (B = 1 000) und berechnen auf
jedem Resample AUC(Head) − AUC(Kontrolle) aus den gecachten Val-Entscheidungsscores (`*_valscores.npz` von
`probe-stats`, Kontroll-Probes mit `return_scores=True`); `beats` = CI-Untergrenze > 0 (`best_control` vergleicht nur
Punkt-AUCs). Notebook-Zelle 6 schreibt `n_heads_beating_best_control` nach `results/controls_*.json`.

## Smoke-Lauf (CPU, pythia-160m step143000, Honesty-Heads des Kern-Smoke-Laufs, Tier smoke)

Siehe `scripts/run_causal.sh pythia-160m smoke honesty` (30 TQA-Fragen, 60 HH-Paare, 20 HelpSteer2-Paare,
12 MC-Fragen, 5k WikiText-Tokens = 4 488 bewertete Tokens bei `min_context` 512, R = 3). Die Zahlen sind bei diesen Stichprobengrößen nur ein Funktionstest;
belastbare Effekte brauchen ≥ 1000 Items und R = 20 (DESIGN 5.10).
