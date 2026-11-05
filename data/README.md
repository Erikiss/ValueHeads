# Harmful-prompt benchmarks (nur `--mode paper`)

Unverändert aus dem Referenz-Repo sihuo-design/valueheads (MIT) übernommen; dort nur zur
Harmlessness-Bewertung (Judge "safe/unsafe") und als σ-Prompt-Set für das Steering verwendet.

| Datei | Quelle | Größe |
|---|---|---|
| `jailbreakbench.csv` | JailbreakBench behaviors, Chao et al., NeurIPS 2024 Datasets & Benchmarks ([Repo](https://github.com/JailbreakBench/jailbreakbench)) | 100 |
| `maliciousinstruct.csv` | MaliciousInstruct, Huang et al., ICLR 2024 ([Repo](https://github.com/Princeton-SysML/Jailbreak_LLM)) | 100 |

Eine Spalte `input`. Die Prompts sind per Konstruktion schädlich: nie in Notebook-Outputs oder
Logs ausgeben. Bitte die Originalarbeiten zitieren.

# ICL-Batterie (`data/icl/`)

Unverändert aus dem Repo des ICL-Schaltkreis-Papers übernommen (Gao & Gao, „A Circuit for In-Context
Learning“, arXiv 2611.05344; MIT, Copyright (c) 2026 Andrew Gao, William Gao; siehe `LICENSE`,
Abschnitt „Attribution: in-context-learning circuit (head sets, battery)“). Verwendet nur von
`valueheads_pythia/icl_battery.py` (Positivkontrolle M3 des Ablations-Tests, `docs/ABLATION_DESIGN.md`).

| Datei | Quelle | Größe | sha256 |
|---|---|---|---|
| `icl/concept_tasks.json` | `batteries/concept_tasks.json` in andrewgcodes/in-context-learning-circuit @ `26c44caf8f9b5a5a197988095f1f76b0562bbaa4` | 11 115 Bytes | `bf242720283bdc9967ffdb22da277f2fc0237d374de2b8214c45790187e60df0` |

Inhalt: pro Konzept zwei Satzlisten (Klasse A / Klasse B), u. a. topic (Tiere / Essen), lang
(Englisch / Spanisch), question (Frage / Aussage), implied (gut / schlecht). Die Sätze sind harmlos.
Die concept12n-Batterie nutzt nur topic, lang, question und implied. `icl_battery.load_concepts()`
prüft die sha256 und verweigert eine veränderte Datei.
