# Harmful-prompt benchmarks (nur `--mode paper`)

Unverändert aus dem Referenz-Repo sihuo-design/valueheads (MIT) übernommen; dort nur zur
Harmlessness-Bewertung (Judge "safe/unsafe") und als σ-Prompt-Set für das Steering verwendet.

| Datei | Quelle | Größe |
|---|---|---|
| `jailbreakbench.csv` | JailbreakBench behaviors, Chao et al., NeurIPS 2024 Datasets & Benchmarks ([Repo](https://github.com/JailbreakBench/jailbreakbench)) | 100 |
| `maliciousinstruct.csv` | MaliciousInstruct, Huang et al., ICLR 2024 ([Repo](https://github.com/Princeton-SysML/Jailbreak_LLM)) | 100 |

Eine Spalte `input`. Die Prompts sind per Konstruktion schädlich: nie in Notebook-Outputs oder
Logs ausgeben. Bitte die Originalarbeiten zitieren.
