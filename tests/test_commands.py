"""Command-level behaviour on synthetic artifacts (no model): `probe` / `probe-stats` with a --train_dir override write
tagged tables instead of overwriting the primary one, `analyze` tolerates a split table without its primary table,
`sweep` refuses --eval_revisions without a metric in --eval_args, and `judge --judge auto --tier cpu` falls back to the
refusal heuristic instead of exiting."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from valueheads_pythia import config
from valueheads_pythia.cli import main
from valueheads_pythia.commands.probe import combined_tag, derived_feature_tag
from valueheads_pythia.commands.probe_stats import permute_within_group
from valueheads_pythia.eval_generate import generations_path, write_generations
from valueheads_pythia.probing import load_head_scores

from test_probe_paper_mode import H, L, make_features, write_shards


def test_feature_tag_helpers():
    assert derived_feature_tag(None, "honesty", "pythia") is None
    assert derived_feature_tag("/f/pythia-160m/step0/honesty_train", "honesty", "pythia") is None
    assert derived_feature_tag("/f/pythia-160m/step0/honesty_train/", "honesty", "pythia") is None
    assert derived_feature_tag("/f/honesty_split1_train", "honesty", "pythia") == "split1"
    assert derived_feature_tag("/f/helpsteer2_train", "coherence", "pythia") is None          # the extract-once dir
    assert derived_feature_tag("/f/coherence_train", "coherence", "pythia") is None
    assert derived_feature_tag("/f/helpsteer2_sub_train", "coherence", "pythia") == "sub"
    assert derived_feature_tag("/f/anything", "honesty", "pythia", "mytag") == "mytag"
    assert combined_tag(None, None) is None and combined_tag("split1", None) == "split1"
    assert combined_tag(None, "headmean") == "headmean" and combined_tag("split1", "headmean") == "split1_headmean"
    # DESIGN 5.4: within-group permutation for TruthfulQA / HH, globally stratified for HelpSteer2 (explicit flags win)
    assert permute_within_group("honesty") and permute_within_group("harmless") and not permute_within_group("helpfulness")
    assert not permute_within_group("coherence_strict") and permute_within_group("verbosity", perm_within=True)
    assert not permute_within_group("honesty", perm_global=True)
    with pytest.raises(SystemExit):
        permute_within_group("honesty", perm_global=True, perm_within=True)


def test_probe_and_stats_with_train_dir_override_write_tagged_tables(tmp_path):
    root = str(tmp_path / "run")
    p = config.paths(root)
    rng = np.random.RandomState(4)
    tr, va = str(tmp_path / "honesty_split1_train"), str(tmp_path / "honesty_split1_val")
    a_tr, y_tr, g_tr = make_features(rng, 200)
    a_va, y_va, g_va = make_features(rng, 100)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    common = ["--model", "pythia-160m", "--revision", "step0", "--value", "honesty", "--root", root, "--train_dir", tr, "--val_dir", va]
    assert main(["probe", *common, "--features", "both", "--quiet", "--n_jobs", "1"]) == 0
    primary = config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia")
    tagged = config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "split1")
    tagged_mean = config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "split1_headmean")
    assert not os.path.exists(primary) and os.path.exists(tagged) and os.path.exists(tagged_mean)
    df, meta = load_head_scores(tagged)
    assert meta["feature_tag"] == "split1" and meta["run_tag"] == "split1" and meta["feature_mode"] == "head" and len(df) == L * H
    assert load_head_scores(tagged_mean)[1]["feature_mode"] == "head+mean"
    # an explicit tag wins over the derived one
    assert main(["probe", *common, "--feature_tag", "custom", "--quiet", "--n_jobs", "1"]) == 0
    assert os.path.exists(config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "custom"))
    # probe-stats finds the tagged tables through the same derivation
    assert main(["probe-stats", *common, "--features", "both", "--bootstrap", "20", "--perms", "2", "--quiet", "--n_jobs", "1"]) == 0
    for path in (tagged, tagged_mean):
        df, meta = load_head_scores(path)
        assert meta["stats_done"] and meta["bootstrap_B"] == 20 and meta["perm_P"] == 2 and "ci_lo" in df.columns
    # analyze: the split table has no primary table -> listed in the summary, no stability block, no crash
    assert main(["analyze", "--model", "pythia-160m", "--root", root, "--no_figures", "--quiet"]) == 0
    summary = pd.read_csv(os.path.join(p["results"], f"summary_pythia-160m.csv"))
    assert set(summary["feature_tag"]) == {"split1", "split1_headmean", "custom"}
    report = open(os.path.join(p["results"], "report_pythia-160m.md")).read()
    assert "keine zusätzlichen Splits" in report


def test_sweep_requires_metric_in_eval_args(tmp_path, capsys):
    root = str(tmp_path / "run")
    with pytest.raises(SystemExit):
        main(["sweep", "--model", "pythia-160m", "--smoke", "--values", "honesty", "--eval_revisions", "step143000",
              "--eval_args", "--intervention mask --select topk", "--dry_run", "--root", root])
    assert main(["sweep", "--model", "pythia-160m", "--smoke", "--values", "honesty", "--eval_revisions", "step143000",
                 "--dry_run", "--root", root]) == 0                                    # default --eval_args carries --metric
    assert "eval=['step143000']" in capsys.readouterr().out


def test_judge_auto_on_cpu_falls_back_to_heuristic(tmp_path):
    root = str(tmp_path / "run")
    p = config.paths(root)
    eval_dir = config.eval_dir(p, "pythia-160m", "step143000")
    header = {"eval_set": "honesty", "prompt_style": "fewshot_qa", "judge_values": ["honesty"], "harmful": False, "model": "pythia-160m",
              "revision": "step143000", "condition": "base", "intervention": "none", "n": 2}
    records = [{"idx": 0, "question": "Q0", "answer": "I'm sorry, I can't help with that.", "raw_answer": "", "n_new_tokens": 9,
                "stop_reason": "stop_string", "prompt_tokens": 100, "prompt_truncated": False},
               {"idx": 1, "question": "Q1", "answer": "The worm dies.", "raw_answer": "", "n_new_tokens": 4, "stop_reason": "stop_string",
                "prompt_tokens": 100, "prompt_truncated": False}]
    base = generations_path(eval_dir, "honesty", "fewshot_qa", "none")
    write_generations(base, header, records)
    topk = generations_path(eval_dir, "honesty", "fewshot_qa", "mask-honesty_topk_k21")
    write_generations(topk, dict(header, condition="topk", intervention="mask", select="topk", head_values=["honesty"], k=21),
                      [dict(records[0], answer="The worm lives."), records[1]])
    assert main(["judge", "--judge", "auto", "--tier", "cpu", "--inputs", topk, "--root", root, "--no_progress", "--bootstrap", "20"]) == 0
    summary = pd.read_csv(os.path.join(p["judge"], "summary.csv"))
    assert set(summary["judge"]) == {"heuristic"} and set(summary["value"]) == {"harmless"} and len(summary) == 2
    row = summary[summary["condition"] == "topk"].iloc[0]
    assert row["rate"] == 0.0 and row["base_rate"] == 0.5 and row["delta_rate"] == -0.5 and row["n_paired"] == 2
    # judging the base file again (cached verdicts) replaces its summary row instead of duplicating it
    assert main(["judge", "--judge", "heuristic", "--tier", "cpu", "--inputs", base, "--value", "harmless", "--root", root, "--no_progress"]) == 0
    again = pd.read_csv(os.path.join(p["judge"], "summary.csv"))
    assert len(again) == 2 and set(again["condition"]) == {"base", "topk"}
    from valueheads_pythia.commands.eval_generate import append_gen_summary
    gs = os.path.join(p["eval"], "gen_summary.csv")
    append_gen_summary(gs, [{"path": "a.jsonl", "n": 1}, {"path": "b.jsonl", "n": 1}])
    append_gen_summary(gs, [{"path": "a.jsonl", "n": 2}])
    g = pd.read_csv(gs)
    assert len(g) == 2 and g.loc[g.path == "a.jsonl", "n"].item() == 2
    # a tier with a real stack still refuses when nothing applies (no fallback outside the cpu tier)
    with pytest.raises(SystemExit):
        main(["judge", "--judge", "tqa", "--tier", "cpu", "--inputs", topk, "--root", root, "--value", "harmless", "--no_progress"])
    # verdict files never carry texts of harmful sets; this honesty set is not harmful, so the texts are present
    verdicts = [json.loads(l) for l in open(summary[summary["condition"] == "topk"].iloc[0]["output"])][1:]
    assert [v["label"] for v in verdicts] == [0, 0] and verdicts[1]["answer"] == "The worm dies."
