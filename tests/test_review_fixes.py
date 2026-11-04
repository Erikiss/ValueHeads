"""Follow-ups of the Colab review and the completeness critic: ArmoRM loads without remote code (F1), `judge` writes
summary.csv after every judge, --continue_on_error and the --judge_model guard (F1 / critic 4), sweep --continue_on_error and
--cleanup_features (F2), HF-cache deletion of judge repos (F3), PCA probes and ranking by the half-A AUC (DESIGN 3.5 / 5.5),
confirmation groups for the disjoint evaluation (DESIGN 4.1), train-split pairs with group exclusion (DESIGN 4.1),
p_ttest_paper only in --mode paper (DESIGN 5.9), openai extra_body / repo user template (DESIGN 2 / 4.2)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from valueheads_pythia import config
from valueheads_pythia.cli import main
from valueheads_pythia.eval_generate import generations_path, write_generations
from valueheads_pythia.probing import load_head_scores, top_heads_from_file

from test_probe_paper_mode import H, L, make_features, write_shards


# ----------------------------------------------------------------------------- F1: ArmoRM without remote code
def test_armorm_native_model_matches_checkpoint_layout(tmp_path):
    from transformers import LlamaConfig

    from valueheads_pythia.judge.armorm import ATTRIBUTES, ArmoRMJudge, load_armorm
    from valueheads_pythia.judge.armorm_model import GATING_TOKEN_PATTERN, ArmoRMForRewardWithGating, find_gating_token

    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, pad_token_id=63, num_objectives=19, gating_temperature=10, gating_hidden_dim=8,
                      gating_n_hidden=3, gating_token_pattern=[5, 6, 7])
    torch.manual_seed(0)
    model = ArmoRMForRewardWithGating(cfg).eval()
    keys = {k for k in model.state_dict() if "rotary" not in k}
    per_layer = ["input_layernorm.weight", "post_attention_layernorm.weight", "mlp.down_proj.weight", "mlp.gate_proj.weight",
                 "mlp.up_proj.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight"]
    expected = {"model.embed_tokens.weight", "model.norm.weight", "regression_layer.weight", "reward_transform_matrix", "gating.logit_scale"}
    expected |= {f"gating.layers.{i}.{p}" for i in range(4) for p in ("weight", "bias")}
    expected |= {f"model.layers.{l}.{k}" for l in range(2) for k in per_layer}
    # the checkpoint's model.safetensors.index.json: 301 tensors = 32 x 9 layer tensors + 5 top-level + 8 gating tensors
    assert keys == expected and 32 * len(per_layer) + 5 + 8 == 301
    assert model.regression_layer.bias is None and model.reward_transform_matrix.requires_grad is False
    ids = torch.tensor([[1, 2, 5, 6, 7, 8, 9, 63, 63], [3, 5, 6, 7, 4, 5, 6, 7, 10]])     # row 0 right-padded, row 1 holds two headers
    with torch.no_grad():
        out = model(ids)
        hid = model.model(input_ids=ids)[0]
    assert out.rewards.shape == (2, 19) and out.gating_output.shape == (2, 19) and out.score.shape == (2,) and out.logits is out.score
    assert torch.allclose(out.gating_output.sum(1), torch.ones(2), atol=1e-5)                 # softmax(x / T) * logit_scale (= 1)
    assert torch.allclose(out.score, (out.gating_output * out.rewards).sum(1), atol=1e-5)    # identity reward transform
    assert torch.allclose(out.hidden_state[0], hid[0, 6]) and torch.allclose(out.hidden_state[1], hid[1, -1])   # last non-pad token
    assert torch.allclose(out.prompt_embedding[1], hid[1, 5])                                   # last header occurrence
    assert find_gating_token([0] + GATING_TOKEN_PATTERN + [1], GATING_TOKEN_PATTERN) == 1
    with pytest.raises(ValueError):
        find_gating_token([1, 2, 3], [5, 6, 7])
    # save_pretrained / from_pretrained roundtrip = the path of load_armorm(loader="native") on the Hub checkpoint
    d = str(tmp_path / "armorm-tiny")
    model.save_pretrained(d)
    assert json.load(open(os.path.join(d, "config.json")))["num_objectives"] == 19
    reloaded = ArmoRMForRewardWithGating.from_pretrained(d, dtype=torch.float32).eval()
    with torch.no_grad():
        out2 = reloaded(ids)
    assert torch.allclose(out2.rewards, out.rewards, atol=1e-6) and torch.allclose(out2.score, out.score, atol=1e-6)
    assert reloaded.gating_token_pattern == [5, 6, 7]

    class Tok:
        def apply_chat_template(self, messages, tokenize=True, return_tensors="pt", truncation=True, max_length=4096):
            return torch.tensor([[1, 2, 5, 6, 7, 8]])

    judge = ArmoRMJudge(model=reloaded, tokenizer=Tok())
    r = judge.evaluate("q", "a", "honesty")
    assert r["judge"] == "armorm" and set(r["rewards"]) == set(ATTRIBUTES) and r["label"] in (0, 1) and "coefficients" not in r
    judge.free()
    assert judge.model is None
    with pytest.raises(ValueError):
        load_armorm(loader="weird")
    # the remote path is still the symbol the checkpoint's modeling_custom.py imports; it is gone on transformers >= 4.56
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 56):
        with pytest.raises(ImportError):
            from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING  # noqa: F401


# ----------------------------------------------------------------------------- judge command: per-judge flush, errors, guards
def _generation_files(tmp_path):
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
    return root, p, base, topk


def test_judge_keeps_rows_of_finished_judges_and_continue_on_error(tmp_path, monkeypatch):
    import valueheads_pythia.commands.judge as jc
    from valueheads_pythia.judge.heuristic import RefusalHeuristic

    class GuardStub(RefusalHeuristic):
        name, model_id = "qwen3guard", "Qwen/Qwen3Guard-Gen-0.6B"

    made = []

    def fake_make_judge(name, **kw):
        made.append(name)
        if name == "qwen3guard":
            return GuardStub()
        raise ImportError(f"cannot import name 'LLAMA_INPUTS_DOCSTRING' (judge {name})")

    monkeypatch.setattr(jc, "make_judge", fake_make_judge)
    root, p, base, topk = _generation_files(tmp_path)
    args = ["judge", "--judge", "auto", "--tier", "t4", "--inputs", topk, "--value", "harmless", "honesty", "--root", root, "--no_progress",
            "--bootstrap", "20"]
    # tier t4 plan: qwen3guard (harmless) first, then armorm + tqa (honesty); armorm's ImportError propagates ...
    with pytest.raises(ImportError):
        main(args)
    summary = pd.read_csv(os.path.join(p["judge"], "summary.csv"))
    assert made == ["qwen3guard", "armorm"]
    assert set(summary["judge"]) == {"qwen3guard"} and len(summary) == 2         # ... but the finished judge's rows were written
    assert summary.set_index("condition").loc["topk", "base_rate"] == 0.5
    # --continue_on_error: every failing judge is skipped and reported, exit 1, rows are replaced (no duplicates)
    assert main(args + ["--continue_on_error"]) == 1
    again = pd.read_csv(os.path.join(p["judge"], "summary.csv"))
    assert set(again["judge"]) == {"qwen3guard"} and len(again) == 2 and made[-2:] == ["armorm", "tqa"]
    assert jc.judge_repos(GuardStub()) == ["Qwen/Qwen3Guard-Gen-0.6B"] and jc.judge_repos(RefusalHeuristic()) == []
    # --judge_model is only accepted with an explicit judge (with auto it would be applied to every judge of the stack)
    with pytest.raises(SystemExit, match="judge_model"):
        main(["judge", "--judge", "auto", "--tier", "l4", "--judge_model", "Qwen/Qwen3-4B-Instruct-2507", "--inputs", topk, "--root", root])
    with pytest.raises(SystemExit):
        main(["judge", "--judge", "heuristic", "--tier", "cpu", "--inputs", topk, "--root", root, "--judge_extra_body", "[1, 2]"])


def test_openai_judge_extra_body_and_repo_template(tmp_path):
    from test_judge import FakeServer

    from valueheads_pythia.judge.openai_compat import USER_TEMPLATE_REPO, OpenAICompatJudge

    server = FakeServer(["yes", "no", "Yes.", "no"])
    try:
        judge = OpenAICompatJudge("m", base_url=server.base_url, api_key="k", backoff=0.01, user_template=USER_TEMPLATE_REPO,
                                  extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        assert judge.score("q", "a", "honesty") == 1
        body = server.requests[0]["body"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 4 and body["model"] == "m"
        assert body["messages"][1]["content"] == "\n        Question: q\n        Answer: a\n        "      # judge.py:21-24 of the repo
        # the same through the CLI (--user_template repo, --judge_extra_body)
        root, p, base, topk = _generation_files(tmp_path)
        rc = main(["judge", "--judge", "openai", "--judge_model", "m", "--base_url", server.base_url, "--api_key", "k", "--inputs", topk,
                   "--root", root, "--no_progress", "--no_cache", "--bootstrap", "10", "--user_template", "repo",
                   "--judge_extra_body", '{"chat_template_kwargs": {"enable_thinking": false}}'])
        # 2 files x 2 records, but (Q1, "The worm dies.") is identical in both files and memoized -> 3 requests
        assert rc == 0 and len(server.requests) == 1 + 3
        for req in server.requests[1:]:
            assert req["body"]["chat_template_kwargs"] == {"enable_thinking": False}
            assert req["body"]["messages"][1]["content"].startswith("\n        Question: ")
        summary = pd.read_csv(os.path.join(p["judge"], "summary.csv"))
        assert set(summary["judge"]) == {"openai"} and set(summary["judge_model"]) == {"m"}
    finally:
        server.close()


def test_cleanup_hf_repo_on_fake_cache(tmp_path):
    from valueheads_pythia.hf_cache import cleanup_hf_repo

    hub = tmp_path / "hub"
    for repo, sha in (("models--RLHFlow--ArmoRM-Llama3-8B-v0.1", "a" * 40), ("models--EleutherAI--pythia-160m", "b" * 40),
                      ("datasets--truthfulqa--truthful_qa", "c" * 40)):
        snap = hub / repo / "snapshots" / sha
        snap.mkdir(parents=True)
        (hub / repo / "blobs").mkdir()
        (hub / repo / "refs").mkdir()
        (hub / repo / "refs" / "main").write_text(sha)
        blob = hub / repo / "blobs" / ("f" + sha[:10])
        blob.write_bytes(b"x" * 1000)
        os.symlink(blob, snap / "weights.bin")
    dry = cleanup_hf_repo("RLHFlow/ArmoRM-Llama3-8B-v0.1", cache_dir=str(hub), dry_run=True)
    assert dry["dry_run"] and not dry["deleted"] and dry["freed_bytes"] == 1000 and dry["n_revisions"] == 1
    assert (hub / "models--RLHFlow--ArmoRM-Llama3-8B-v0.1").exists()
    res = cleanup_hf_repo("RLHFlow/ArmoRM-Llama3-8B-v0.1", cache_dir=str(hub))
    assert res["deleted"] and res["freed_bytes"] == 1000
    assert not (hub / "models--RLHFlow--ArmoRM-Llama3-8B-v0.1").exists()
    assert (hub / "models--EleutherAI--pythia-160m" / "snapshots").exists() and (hub / "datasets--truthfulqa--truthful_qa").exists()
    assert "not in cache" in cleanup_hf_repo("RLHFlow/ArmoRM-Llama3-8B-v0.1", cache_dir=str(hub))["reason"]
    assert "not in cache" in cleanup_hf_repo("truthfulqa/truthful_qa", cache_dir=str(hub))["reason"]      # datasets are never touched


# ----------------------------------------------------------------------------- sweep: error path and feature cleanup
def test_sweep_continue_on_error_records_the_failure(tmp_path, monkeypatch):
    import valueheads_pythia.commands.sweep as sw

    calls = []

    def fake_run_revision(args, p, short, hf_id, revision, values, tier, B, P, manifest):
        calls.append(revision)
        if revision == "step0":
            raise RuntimeError("extraction exploded")
        manifest["revisions"].setdefault(revision, {})["done"] = list(values)

    monkeypatch.setattr(sw, "run_revision", fake_run_revision)
    root = str(tmp_path / "run")
    common = ["sweep", "--model", "pythia-160m", "--revisions", "step0", "step143000", "--values", "honesty", "--tier", "smoke", "--no_curves",
              "--root", root]
    assert main(common + ["--continue_on_error"]) == 1 and calls == ["step0", "step143000"]
    manifest = json.load(open(config.paths(root)["manifest"]))
    assert "extraction exploded" in manifest["revisions"]["step0"]["error"] and manifest["revisions"]["step143000"]["done"] == ["honesty"]
    with pytest.raises(RuntimeError):
        main(common)


def test_sweep_feature_cleanup_keeps_cacheable_dirs(tmp_path):
    from valueheads_pythia.commands.sweep import cacheable_features, cleanup_feature_dirs, feature_dirs_of_revision

    p = config.paths(str(tmp_path / "run"))
    for ds, split in [("honesty", "train"), ("honesty", "val"), ("helpsteer2", "train"), ("helpsteer2", "val"), ("harmless", "train"),
                      ("harmless", "val"), ("honesty_split1", "train")]:
        d = config.feature_dir(p, "pythia-160m", "step0", ds, split)
        os.makedirs(d)
        with open(os.path.join(d, "layer00.npy"), "wb") as f:
            f.write(b"x" * 10)
    values = ["honesty", "harmless", "helpfulness"]
    dirs = feature_dirs_of_revision(p, "pythia-160m", "step0", values, "pythia", split_seeds=[1, 42])
    assert len(dirs) == 8 and sum(1 for _, ds, _ in dirs if ds == "honesty") == 4         # helpfulness -> helpsteer2 (extract once)
    dry = cleanup_feature_dirs(p, "pythia-160m", "step0", values, "pythia", [1], dry_run=True)
    assert {os.path.basename(d) for d in dry} == {"helpsteer2_train", "harmless_train"} and all(os.path.isdir(d) for d in dry)
    freed = cleanup_feature_dirs(p, "pythia-160m", "step0", values, "pythia", [1])
    assert sum(freed.values()) == 20 and not any(os.path.isdir(d) for d in freed)
    for ds, split in [("honesty", "train"), ("honesty", "val"), ("helpsteer2", "val"), ("harmless", "val"), ("honesty_split1", "train")]:
        assert os.path.isdir(config.feature_dir(p, "pythia-160m", "step0", ds, split))
    assert cacheable_features("honesty", "train") and cacheable_features("harmless", "val") and not cacheable_features("helpsteer2", "train")


# ----------------------------------------------------------------------------- probing: PCA, ranking by half A, confirmation groups
def test_pca_probe_rank_by_half_a_and_confirmation_groups(tmp_path):
    root = str(tmp_path / "run")
    p = config.paths(root)
    rng = np.random.RandomState(1)
    tr, va = (config.feature_dir(p, "pythia-160m", "step0", "honesty", s) for s in ("train", "val"))
    a_tr, y_tr, g_tr = make_features(rng, 240)
    a_va, y_va, g_va = make_features(rng, 120)
    write_shards(tr, a_tr, y_tr, g_tr)
    write_shards(va, a_va, y_va, g_va)
    common = ["--model", "pythia-160m", "--revision", "step0", "--value", "honesty", "--root", root]
    assert main(["probe", *common, "--pca", "3", "--quiet", "--n_jobs", "1"]) == 0
    primary = config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia")
    pca_path = config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "pca3")
    assert os.path.exists(pca_path) and not os.path.exists(primary)                      # tagged, never the primary table
    df, meta = load_head_scores(pca_path)
    assert meta["pca"] == 3 and len(df) == L * H and df["auc"].between(0, 1).all()
    assert main(["probe-stats", *common, "--feature_tag", "pca3", "--bootstrap", "20", "--perms", "2", "--quiet", "--n_jobs", "1"]) == 0
    df, meta = load_head_scores(pca_path)
    assert meta["stats_done"] and {"auc_half_A", "auc_half_B", "ci_lo"} <= set(df.columns) and meta["confirmation"]["n_groups_A"] >= 1
    # ranking by the half-A AUC (stable tie-break) for the disjoint protocol
    k = max(1, L * H // 2)
    heads_a = top_heads_from_file(pca_path, k=k, score="auc_half_A")
    best = df.loc[df["auc_half_A"].round(6) == df["auc_half_A"].round(6).max()]
    assert len(heads_a) == k and heads_a[0] in set(best["flat_idx"].astype(int)) and len(set(heads_a)) == k
    assert top_heads_from_file(pca_path, 0.5)[:3] == [int(x) for x in df.sort_values("rank")["flat_idx"].head(3)]
    # confirmation groups: disjoint halves that cover the val groups, sizes as stored by probe-stats
    from valueheads_pythia.probing_stats import confirmation_groups, confirmation_halves
    a, b = confirmation_groups(pca_path)
    assert not (a & b) and (a | b) == {int(g) for g in g_va}
    assert (len(a), len(b)) == (meta["confirmation"]["n_groups_A"], meta["confirmation"]["n_groups_B"])
    assert confirmation_halves(g_va, 0) == (a, b) and confirmation_halves(g_va, 1) != (a, b)
    # a table without probe-stats can neither be ranked by half A nor split
    assert main(["probe", *common, "--quiet", "--n_jobs", "1"]) == 0
    with pytest.raises(ValueError, match="probe-stats"):
        top_heads_from_file(primary, k=3, score="auc_half_A")
    with pytest.raises(ValueError, match="probe-stats"):
        confirmation_groups(primary)
    # --C tags the table too
    assert main(["probe", *common, "--C", "0.5", "--quiet", "--n_jobs", "1"]) == 0
    assert os.path.exists(config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "C0p5"))


# ----------------------------------------------------------------------------- eval: train pairs, group filters, paper-only t-test
def test_train_pairs_and_group_filters():
    from valueheads_pythia.data import eval_pairs
    from valueheads_pythia.eval_likelihood import PairwiseMetric, build_metric

    train = eval_pairs("tqa_pairwise", split="train")
    val = eval_pairs("tqa_pairwise")
    assert len(train) == 653 and len(val) == 164 and not ({it["group"] for it in train} & {it["group"] for it in val})
    drop = {it["group"] for it in train[:600]}
    m = PairwiseMetric("tqa_pairwise", split="train", exclude_groups=drop)
    assert m.config["n_items"] == 53 and m.config["n_items_unfiltered"] == 653 and "exclude_groups_sha" in m.config
    assert not any(it["group"] in drop for it in m.items) and m.config["split"] == "train"
    keep = {val[0]["group"], val[1]["group"], val[2]["group"]}
    m2 = PairwiseMetric("tqa_pairwise", include_groups=keep, max_items=2)
    assert m2.config["n_items"] == 2 and all(it["group"] in keep for it in m2.items) and m2.config["n_items_unfiltered"] == 164
    plain = PairwiseMetric("tqa_pairwise", max_items=5)
    assert "n_items_unfiltered" not in plain.config and plain.config["n_items"] == 5      # unfiltered condition keys stay unchanged
    with pytest.raises(ValueError):
        PairwiseMetric("tqa_pairwise", include_groups={-1})
    with pytest.raises(ValueError):
        build_metric("tqa_mc1_mc2", include_groups={1})
    hh = eval_pairs("hh_pairwise", "harmless", split="train")
    assert len(hh) > 12000 and len(eval_pairs("hh_pairwise", "harmless")) == 662


def test_p_ttest_paper_only_in_paper_mode():
    from valueheads_pythia.eval_likelihood import METRIC_SPECS, summary_row

    class FakeMetric:
        name, value, spec, config = "tqa_pairwise", None, METRIC_SPECS["tqa_pairwise"], {"n_items": 2, "context": "probing"}

        def item_values(self, items):
            m = np.array([it["m"] for it in items])
            return {"margin": m, "win": (m > 0).astype(float)}, np.array([0, 1]), None

        def summarize(self, items):
            vals = self.item_values(items)[0]
            return {"pair_acc": float(vals["win"].mean()), "pair_acc_pairs": float(vals["win"].mean()), "margin_mean": float(vals["margin"].mean()),
                    "n_questions": 2, "n_pairs": 2}

    metric = FakeMetric()

    def rec(ms):
        items = [{"m": m} for m in ms]
        return {"summary": metric.summarize(items), "items": items, "stats": {"seconds": 0.1}}

    base, topk = rec([0.5, -0.2]), rec([0.1, -0.3])
    randoms = [rec([0.4, -0.1]), rec([0.3, 0.05]), rec([0.2, -0.2])]
    row = summary_row(metric, base, topk, randoms, "mask", B=10, meta={"mode": "pythia"})
    assert np.isnan(row["p_ttest_paper"]) and 0 < row["p_emp_primary"] <= 1 and row["R"] == 3
    paper = summary_row(metric, base, topk, randoms, "mask", B=10, meta={"mode": "paper"})
    assert 0 <= paper["p_ttest_paper"] <= 1
    assert np.isnan(summary_row(metric, base, topk, randoms, "mask", B=10, meta=None)["p_ttest_paper"])


def test_save_manifest_merges_other_sessions_revisions(tmp_path):
    """Two sweeps on one root (parallel Colab sessions over disjoint revisions) must not clobber each other's entries."""
    from valueheads_pythia.commands.sweep import load_manifest, save_manifest
    path = str(tmp_path / "manifest.json")
    a = {"model": "pythia-160m", "revisions": {"step0": {"a": 1}}}
    save_manifest(path, a)
    b = {"model": "pythia-160m", "revisions": {"step143000": {"b": 2}}}
    save_manifest(path, b)
    a["revisions"]["step0"] = {"a": 3}
    save_manifest(path, a)
    on_disk = load_manifest(path)
    assert on_disk["revisions"] == {"step0": {"a": 3}, "step143000": {"b": 2}}
    assert a["revisions"]["step143000"] == {"b": 2}
