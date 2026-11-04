"""Summary table (summary.csv / summary.json) and the German markdown report of a model's results.

The summary has one row per head-score table (model, revision, value, mode, feature tag) with chance levels
(AUC 0.5, majority accuracy), the top head with its bootstrap CI, the permutation threshold and adjusted
p-value, the significant set, the selection / confirmation AUCs and the paper-style counts (acc > 0.8).
Likelihood-evaluation summaries (results/eval/{model}_{revision}/summary.csv, written by eval-likelihood)
are appended as they are when present.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from .sweep_curves import parse_step, sort_revisions, summarize_table

SUMMARY_COLUMNS = ["model", "revision", "step", "value", "mode", "feature_tag", "feature_mode", "score", "class_weight",
                   "n_train", "n_val", "n_groups", "pos_rate", "majority", "chance_auc", "top_flat", "top_layer", "top_head",
                   "top_auc", "top_acc", "top_bal_acc", "ci_lo", "ci_hi", "p_adj_top", "perm_thr", "perm_P", "bootstrap_B",
                   "n_signif", "k", "conf_top1", "conf_topk", "winners_curse_gap", "n_auc_gt_0.7", "n_acc_gt_0.8",
                   "best_layer_max_auc", "stats_done", "heads_json"]


def summary_row(df, key, ratio=0.15, heads_json=None):
    """One summary row for a head table; key = (revision, value, mode, feature_tag)."""
    revision, value, mode, tag = key
    s = summarize_table(df, ratio)
    return {"model": df.attrs.get("model"), "revision": revision, "step": parse_step(revision), "value": value, "mode": mode,
            "feature_tag": tag or "primary", "feature_mode": df.attrs.get("feature_mode"), "score": df.attrs.get("score"),
            "class_weight": df.attrs.get("class_weight") or "none", "n_groups": df.attrs.get("n_groups"),
            "chance_auc": 0.5, "perm_P": df.attrs.get("perm_P"), "bootstrap_B": df.attrs.get("bootstrap_B"),
            "stats_done": bool(df.attrs.get("stats_done", False)), "heads_json": heads_json, **s}


def build_summary(tables, ratio=0.15, paths=None):
    """DataFrame over {key: df} (key = (revision, value, mode, feature_tag)), sorted by step / value / mode."""
    rows = [summary_row(df, key, ratio, (paths or {}).get(key)) for key, df in tables.items()]
    out = pd.DataFrame(rows)
    for c in SUMMARY_COLUMNS:
        if c not in out:
            out[c] = np.nan
    out = out[SUMMARY_COLUMNS]
    order = {r: i for i, r in enumerate(sort_revisions(out["revision"].unique()))}
    out["_o"] = out["revision"].map(order)
    return out.sort_values(["_o", "value", "mode", "feature_tag"]).drop(columns="_o").reset_index(drop=True)


def write_summary(summary, csv_path, json_path=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    summary.to_csv(csv_path, index=False)
    if json_path:
        with open(json_path, "w") as f:
            json.dump(json.loads(summary.to_json(orient="records")), f, indent=1)
    return csv_path


def load_eval_summaries(eval_root, model):
    """{revision: DataFrame} of the likelihood-evaluation summaries of `model`: the rows of the global
    results/eval/summary.csv written by `eval-likelihood` (one row per run, column `revision`), plus any per-revision
    results/eval/{model}_{revision}/summary.csv files (any columns)."""
    out = {}
    if not os.path.isdir(eval_root):
        return out
    prefix = f"{model}_"
    for name in sorted(os.listdir(eval_root)):
        path = os.path.join(eval_root, name, "summary.csv")
        if name.startswith(prefix) and os.path.exists(path):
            try:
                out[name[len(prefix):]] = pd.read_csv(path)
            except Exception as e:      # a foreign / partial file must not break the report
                out[name[len(prefix):]] = pd.DataFrame({"fehler": [str(e)]})
    global_csv = os.path.join(eval_root, "summary.csv")
    if os.path.exists(global_csv):
        try:
            g = pd.read_csv(global_csv)
            if {"model", "revision"} <= set(g.columns):
                for rev, sub in g[g["model"] == model].groupby("revision", sort=False):
                    rev = str(rev)
                    sub = sub.reset_index(drop=True)
                    out[rev] = pd.concat([out[rev], sub], ignore_index=True, sort=False) if rev in out else sub
        except Exception as e:
            out["summary.csv"] = pd.DataFrame({"fehler": [str(e)]})
    return out


# reporting columns of the eval-likelihood summary (the csv holds ~60 columns; base / topk are the primary statistic)
EVAL_REPORT_COLUMNS = ["metric", "eval_value", "context", "head_values", "intervention", "select", "alpha", "k", "primary", "n_items", "R",
                       "base", "topk", "random_mean", "random_std", "p_emp_primary", "p_direction", "effect", "delta_mean",
                       "delta_ci_lo", "delta_ci_hi", "p_emp_delta"]


def compact_eval_summary(df):
    """Adds `base` / `topk` (the primary statistic of every row) and keeps the reporting columns that exist; a frame
    without the eval-likelihood columns is returned unchanged."""
    df = df.copy()
    if "primary" in df.columns:
        for prim in df["primary"].dropna().unique():
            sub = df["primary"] == prim
            for side in ("base", "topk"):
                if f"{side}_{prim}" in df.columns:
                    df.loc[sub, side] = df.loc[sub, f"{side}_{prim}"]
    cols = [c for c in EVAL_REPORT_COLUMNS if c in df.columns]
    return df[cols] if len(cols) >= 3 else df


JUDGE_REPORT_COLUMNS = ["revision", "eval_set", "prompt_style", "condition", "judge", "value", "n", "rate", "rate_lo", "rate_hi",
                        "score_mean", "base_rate", "delta_rate", "delta_rate_lo", "delta_rate_hi", "floor_effect"]


def load_judge_summary(judge_dir, model):
    """DataFrame of results/judge/summary.csv (written by `judge`) restricted to `model`, last row per
    (input, judge, value); None when absent."""
    path = os.path.join(judge_dir, "summary.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "model" in df.columns:
        df = df[df["model"] == model]
    keys = [c for c in ("input", "judge", "value") if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- markdown helpers
def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "ja" if v else "nein"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df, columns=None, digits=3, max_rows=None):
    """Markdown table of a DataFrame (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_keine Daten_\n"
    columns = [c for c in (columns or list(df.columns)) if c in df.columns]
    rows = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, r in rows.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in columns) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| … ({len(df) - max_rows} weitere Zeilen) |" + " |" * (len(columns) - 1))
    return "\n".join(lines) + "\n"


def _rel(path, base):
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


# ----------------------------------------------------------------------------- report
def render_report(model, summary, sweep_tables=None, correlations=None, angles=None, manifest=None, figures=None,
                  eval_summaries=None, stability=None, out_path=None, ratio=0.15, judge_summary=None):
    """German markdown report; returns the text (and writes it to out_path when given)."""
    base = os.path.dirname(out_path) if out_path else "."
    figures = figures or {}
    L = []
    L.append(f"# ValueHeads auf Pythia – Ergebnisbericht `{model}`\n")
    L.append(f"Erstellt am {time.strftime('%Y-%m-%d %H:%M')} aus `summary.csv` ({len(summary)} Head-Score-Tabellen, "
             f"{summary['revision'].nunique()} Checkpoints, {summary['value'].nunique()} Werte). Methode: lineare Probes "
             "(logistische Regression) auf den Last-Token-Aktivierungen jedes Attention-Heads (Eingang der Output-Projektion), "
             "Score = ROC-AUC auf dem Val-Split (Modus `pythia`) bzw. Accuracy (Modus `paper`).\n")

    # --- setup
    L.append("## 1. Setup und Numerik\n")
    if manifest:
        revs = manifest.get("revisions", {})
        rows = [{"revision": r, "commit_sha": (v.get("commit_sha") or "–")[:12], "dtype": v.get("dtype"), "attn_impl": v.get("attn_impl"),
                 "transformers": v.get("transformers"), "torch": v.get("torch"), "device": v.get("device")} for r, v in revs.items()]
        L.append(md_table(pd.DataFrame(rows)) if rows else "_kein Manifest_\n")
    else:
        L.append("_kein `results/manifest.json` vorhanden (wird vom `sweep`-Befehl geschrieben)._\n")
    L.append("Zufallsniveaus: AUC 0.5; Accuracy = Mehrheitsklasse (`majority`). Die Paper-Schwelle (Accuracy > 0.8, "
             "Repo-Features [Head ⊕ Layer-Mittel]) wird als Vergleichsspalte mitgeführt. Konfidenzintervalle: Cluster-Bootstrap "
             "(Gruppe = Frage / Dialogpaar / Prompt, Perzentil-CI, kein Refit). Signifikanz: Permutations-Null des Maximums über alle "
             "Heads (Labels gruppenweise permutiert, alle Probes neu gefittet; Schwelle = 95. Perzentil, Westfall-Young-adjustierte "
             "p-Werte). `n_signif` = Heads, deren CI-Untergrenze über der Permutationsschwelle liegt. Selektions-/Konfirmations-Split: "
             "Val gruppenweise halbiert, Ranking auf einer Hälfte, AUC der ausgewählten Heads auf der anderen (`conf_top1`, `conf_topk`, "
             "beide Richtungen gemittelt); `winners_curse_gap` = Selektions-AUC − Konfirmations-AUC des Top-1-Heads.\n")

    # --- per table
    L.append("## 2. Head-Scores je Checkpoint und Wert\n")
    cols = ["revision", "value", "mode", "feature_tag", "n_train", "n_val", "pos_rate", "majority", "top_layer", "top_head", "top_auc",
            "ci_lo", "ci_hi", "top_acc", "perm_thr", "p_adj_top", "n_signif", "k", "conf_top1", "conf_topk", "winners_curse_gap",
            "n_auc_gt_0.7", "n_acc_gt_0.8", "best_layer_max_auc"]
    L.append(md_table(summary, cols))
    L.append("Lesart: `top_*` = bester Head nach Ranking-Score (deterministischer Tie-Break), `ci_lo`/`ci_hi` = 95 %-Bootstrap-CI "
             "dieses Heads (Winner's-Curse-verzerrt, daher `conf_*`), `perm_thr` = Permutationsschwelle des Max-über-Heads, "
             "`p_adj_top` = FWER-adjustierter p-Wert des Top-Heads, `best_layer_max_auc` = Layer mit der höchsten Head-AUC.\n")
    prim = summary[(summary["feature_tag"] == "primary") & (summary["mode"] == "pythia")]
    if len(prim):
        sig = prim[prim["stats_done"] == True]  # noqa: E712
        L.append("### 2.1 Kurzfazit\n")
        for _, r in prim.iterrows():
            verdict = ("signifikant" if (r["n_signif"] or 0) > 0 else "nicht signifikant") if r["stats_done"] else "ohne Statistik"
            L.append(f"- `{r['revision']}` / {r['value']}: Top-Head L{int(r['top_layer'])}.H{int(r['top_head'])} AUC {_fmt(r['top_auc'])} "
                     f"(CI [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}], Konfirmation {_fmt(r['conf_top1'])}), Permutationsschwelle {_fmt(r['perm_thr'])}, "
                     f"{_fmt(r['n_signif'])} von {int(r['k']) if not np.isnan(r['k']) else '–'} (k) signifikante Heads → {verdict}; "
                     f"{int(r['n_acc_gt_0.8'])} Heads über der Paper-Schwelle acc > 0.8.")
        L.append("")
        if len(sig) and (sig["n_signif"] == 0).all():
            L.append("Kein Head überschreitet die Permutationsschwelle mit seiner CI-Untergrenze: Die beobachteten Maxima sind mit dem "
                     "Max-über-Heads einer Null-Verteilung (Winner's Curse) verträglich oder die Stichprobe (Smoke-Tier) ist zu klein.\n")

    # --- sweep
    L.append("## 3. Checkpoint-Sweep\n")
    if sweep_tables:
        for (value, mode), table in sweep_tables.items():
            L.append(f"### {value} ({mode}), Referenz = `{table.attrs.get('final', '')}`\n")
            L.append(md_table(table, ["revision", "step", "top_layer", "top_head", "top_auc", "ci_lo", "ci_hi", "perm_thr", "n_signif",
                                      "spearman_to_final", "jaccard_to_final", "conf_top1", "n_auc_gt_0.7", "best_layer_max_auc"]))
            fig = figures.get(("sweep", value, mode))
            if fig:
                L.append(f"![Sweep {value}]({_rel(fig, base)})\n")
        L.append("Spearman = Rangkorrelation der Head-Score-Vektoren mit dem Referenz-Checkpoint, Jaccard = Überlappung der "
                 "Top-15 %-Mengen. Bei frühen Checkpoints (Spearman < 0.8) sind nur Layer-Profile und Top-AUC interpretierbar, "
                 "keine Head-Identitäten (DESIGN 5.6). step0 ist die Lexik-Baseline: alles, was step0 schon kann, ist kein Trainingsbeitrag.\n")
    else:
        L.append("_weniger als zwei Checkpoints vorhanden – keine Sweep-Kurven._\n")

    # --- stability
    L.append("## 4. Split-Stabilität\n")
    if stability:
        for name, st in stability.items():
            L.append(f"- {name}: Spearman Mittel {_fmt(st['spearman_mean'])} (min {_fmt(st['spearman_min'])}), "
                     f"Jaccard@{st['k']} Mittel {_fmt(st['jaccard_mean'])} (min {_fmt(st['jaccard_min'])}), n = {st['n']} Splits.")
        L.append("")
        L.append("Rankings auf verschiedenen Frage-/Prompt-Splits (Seed 42 = Paper, weitere Seeds via `sweep --split_seeds`); "
                 "Head-Identitäten gelten erst ab Spearman > 0.8 als stabil (DESIGN 5.6).\n")
    else:
        L.append("_keine zusätzlichen Splits (sweep --split_seeds) vorhanden._\n")

    # --- correlations
    L.append("## 5. Korrelationen\n")
    if correlations:
        for name, mat in correlations.items():
            L.append(f"**{name}**\n")
            L.append(md_table(mat.reset_index().rename(columns={"index": ""}), digits=2))
            fig = figures.get(("corr", name))
            if fig:
                L.append(f"![{name}]({_rel(fig, base)})\n")
        L.append("Über Werte: Pearson auf binären Top-15 %-Vektoren (Paper, App. D) und auf den Score-Vektoren; über Checkpoints: "
                 "Pearson / Spearman der (L × H)-Score-Grids.\n")
    else:
        L.append("_keine Korrelationen (mindestens zwei Werte bzw. zwei Checkpoints nötig)._\n")

    # --- angles
    L.append("## 6. Richtungswinkel (App. K)\n")
    if angles is not None and len(angles):
        L.append(md_table(angles, ["revision", "pair", "n_shared", "mean_deg", "median_deg", "std_deg", "all_n", "all_mean_deg", "all_std_deg"], digits=2))
        for key, fig in figures.items():
            if key[0] == "angles":
                L.append(f"![Winkel {key[1]}]({_rel(fig, base)})\n")
        L.append("Winkel zwischen den Mass-Mean-Richtungen zweier Werte auf den Heads, die in beiden Top-15 %-Mengen liegen "
                 "(std mit ddof = 0 wie in Table 17); `all_*` = Referenz über alle L · H Heads. Paper: harmless × honesty am "
                 "orthogonalsten (≈ 90 Grad).\n")
    else:
        L.append("_keine Richtungswinkel (mindestens zwei Werte mit Richtungen und Head-Rankings am selben Checkpoint nötig)._\n")

    # --- eval
    L.append("## 7. Likelihood-Evaluation\n")
    if eval_summaries:
        for rev, df in eval_summaries.items():
            L.append(f"### `{rev}`\n")
            L.append(md_table(compact_eval_summary(df), max_rows=60, digits=4))
        L.append("Zeilen aus `results/eval/summary.csv` (eval-likelihood, eine Zeile je Lauf; `base` / `topk` = Primärstatistik der "
                 "Zeile, `random_mean ± random_std` über die R layer-matched Random-Sets). Effektmaß: gepaarte Differenz "
                 "Intervention − Base pro Item mit Bootstrap-CI (`delta_*`), empirischer p-Wert `p_emp` gegen die Random-Sets "
                 "(Richtung `p_direction`: le = niedriger ist stärker, ge = höher ist stärker).\n")
        causal = [(k, v) for k, v in figures.items() if k[0] == "causal"]
        if causal:
            L.append("Abbildungen (`figures/causal/`, Base / TopK gegen die Random-Verteilung je Metrik):\n")
            for k, v in causal:
                L.append(f"![{' / '.join(str(x) for x in k[1:])}]({_rel(v, base)})\n")
    else:
        L.append("_keine `results/eval/summary.csv` gefunden (eval-likelihood noch nicht gelaufen)._\n")
    if judge_summary is not None and len(judge_summary):
        L.append("### Generation + Judge\n")
        L.append(md_table(judge_summary, JUDGE_REPORT_COLUMNS))
        L.append("Zeilen aus `results/judge/summary.csv` (judge): `rate` = Anteil positiver Verdicts mit Bootstrap-CI, `base_rate` = "
                 "Rate des unmaskierten Modells, `delta_rate` = gepaarte Differenz zur Base; `floor_effect` = 1, wenn die Basisrate "
                 "≤ 0.15 liegt (Bodeneffekt, DESIGN 3.2: Effekte dann nicht nachweisbar; Verweigerung ist bei Base-Modellen ≡ 0).\n")

    # --- figures
    heat = [(k, v) for k, v in figures.items() if k[0] in ("heatmap", "layers", "causal")]
    if heat:
        L.append("## 8. Abbildungen\n")
        for k, v in heat:
            L.append(f"- {k[0]} `{k[1]}` ({' / '.join(str(x) for x in k[2:])}): `{_rel(v, base)}`")
        L.append("")

    # --- limitations
    L.append("## 9. Einschränkungen\n")
    L.append("- Base-Modelle: Die Probes messen wert-korrelierte Merkmalsdetektoren (Register, Vokabular, Länge), keine "
             "verhaltensregulierenden „Value Heads“; Verweigerung ist bei Base-Modellen ≡ 0 (DESIGN 3.3, 8).\n"
             "- Bei 144–1024 Heads findet man ohne Null-Verteilung immer ein Maximum: nur Heads mit CI-Untergrenze über der "
             "Permutationsschwelle und bestätigter Konfirmations-AUC gelten als wert-sensitiv.\n"
             "- Smoke-Tier-Läufe (kleine `max_samples`) haben breite CIs (±0.05–0.1) und wenige Permutationen; Zahlen daraus sind "
             "Funktionstests, keine Befunde.\n"
             "- Paper-interne Inkonsistenzen (Table 1 „random“ als Einzellauf, Judge-Prompts) sind in DESIGN Abschnitt 2/8 dokumentiert.\n")
    text = "\n".join(L)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
    return text
