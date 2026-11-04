"""analysis package + sweep / analyze commands on synthetic head tables: paper-style score map equals the repo's,
heatmap / layer-profile / sweep / correlation / angle figures render, the sweep table and report are built, the
HF cache cleanup works on a fake cache, and `analyze` runs end to end on a synthetic run directory."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from valueheads_pythia import config
from valueheads_pythia.analysis import correlations as corr
from valueheads_pythia.analysis import direction_angles as ang
from valueheads_pythia.analysis import heatmaps as hm
from valueheads_pythia.analysis import report as rp
from valueheads_pythia.analysis import sweep_curves as sc
from valueheads_pythia.probing import load_head_scores, save_head_scores, select_topk

from test_probe_paper_mode import D, H, L, REPO, make_features, write_shards

REVS = ["step0", "step2000", "step143000"]


def synthetic_table(seed, strength, revision, value="honesty", mode="pythia", with_stats=True):
    """Head table with a planted head whose AUC grows with `strength`, plus optional stats columns / meta."""
    rng = np.random.RandomState(seed)
    rows = []
    for flat in range(L * H):
        auc = 0.5 + 0.05 * rng.rand()
        if flat == 1 * H + 2:
            auc = 0.5 + strength
        if flat == 2 * H + 0:
            auc = 0.5 + strength / 2
        rows.append({"layer": flat // H, "head": flat % H, "flat_idx": flat, "auc": auc, "acc": auc - 0.05, "bal_acc": auc - 0.03,
                     "majority": 0.55, "pos_rate": 0.45, "n_train": 200, "n_val": 100})
    df = pd.DataFrame(rows)
    df.attrs.update(layers=L, heads=H, head_dim=D, score="auc", feature_mode="head", class_weight="balanced", model="synthetic",
                    revision=revision, value=value, mode=mode, seed=42, C=1.0, max_iter=1000)
    from valueheads_pythia.probing import rank_heads
    rank_of = {int(f): r for r, f in enumerate(rank_heads(df, "auc"), start=1)}
    df["rank"] = df["flat_idx"].map(rank_of).astype(int)
    if with_stats:
        df["ci_lo"], df["ci_hi"] = df["auc"] - 0.08, df["auc"] + 0.08
        df["p_adj"] = np.where(df["auc"] > 0.7, 0.01, 0.6)
        thr = 0.62
        df["signif"] = df["ci_lo"] > thr
        df.attrs.update(perm_threshold=thr, perm_P=20, bootstrap_B=200, n_signif=int(df["signif"].sum()),
                        signif_heads=[int(i) for i in df.loc[df["signif"], "flat_idx"]], stats_done=True,
                        confirmation={"top1_conf_mean": 0.5 + strength - 0.05, "topk_conf_mean": 0.55, "winners_curse_gap": 0.05, "k": int(0.15 * L * H)})
    return df


@pytest.fixture(scope="module")
def tables():
    return {r: synthetic_table(i, 0.1 + 0.1 * i, r) for i, r in enumerate(REVS)}


def test_score_map_paper_matches_repo(tables):
    df = tables["step143000"]
    scores = df.sort_values("flat_idx")["acc"].to_numpy()
    mine = hm.score_map_paper(scores, L, H)
    assert mine.shape == (L, H) and mine.max() == 1.0 and mine.min() == 0.0
    if os.path.isdir(os.path.join(REPO, "analysis")):
        sys.path.insert(0, os.path.join(REPO, "analysis"))
        sys.path.insert(0, REPO)
        from plot_head_heatmaps import score_map as repo_score_map
        ref = repo_score_map({i: float(s) for i, s in enumerate(scores)}, L, H)
        assert np.allclose(mine, ref)
    assert np.allclose(hm.score_grid(df, mode="paper"), mine)
    assert hm.score_grid(df, mode="pythia")[1, 2] == df.loc[df.flat_idx == 1 * H + 2, "auc"].item()
    assert hm.score_map_paper(np.full(L * H, 0.7), L, H).max() == 0.0
    assert hm.mask_grid(df, [1 * H + 2])[1, 2] and hm.mask_grid(df, [1 * H + 2]).sum() == 1


def test_heatmap_and_layer_profile_figures(tables, tmp_path):
    df = tables["step143000"]
    grids = hm.heatmap_figure({"honesty": df}, str(tmp_path / "h.png"), mode="pythia", title="t")
    assert os.path.getsize(tmp_path / "h.png") > 1000 and grids["honesty"].shape == (L, H)
    hm.heatmap_figure({"honesty": df, "helpful": tables["step0"], "x": tables["step2000"], "y": df}, str(tmp_path / "p.png"), mode="paper")
    assert os.path.exists(tmp_path / "p.png")
    prof = hm.layer_profile(df, "auc", 0.15)
    assert list(prof.columns) == ["layer", "auc_max", "auc_mean", "n_topk", "n_signif"] and len(prof) == L
    assert prof["n_topk"].sum() == int(0.15 * L * H) and prof["n_signif"].sum() == df.attrs["n_signif"] and prof["auc_max"].idxmax() == 1
    profiles = hm.layer_profile_figure({"honesty": df, "other": tables["step0"]}, str(tmp_path / "l.png"), title="layers")
    assert set(profiles) == {"honesty", "other"} and os.path.exists(tmp_path / "l.png")


def test_sweep_table_and_plot(tables, tmp_path):
    assert sc.parse_step("step143000") == 143000 and sc.parse_step("main") is None
    assert sc.sort_revisions(["step2000", "main", "step0", "step143000"]) == ["step0", "step2000", "step143000", "main"]
    assert sc.final_revision(["step2000", "step0"]) == "step2000"
    table = sc.sweep_table(tables, "synthetic", "honesty")
    assert list(table.columns) == sc.CURVE_COLUMNS and list(table["revision"]) == REVS and table.attrs["final"] == "step143000"
    final = table[table.revision == "step143000"].iloc[0]
    assert final["spearman_to_final"] == 1.0 and final["jaccard_to_final"] == 1.0 and final["top_flat"] == 1 * H + 2
    assert table["top_auc"].is_monotonic_increasing and (table["n_signif"] >= 0).all() and table["k"].iloc[0] == int(0.15 * L * H)
    assert table["conf_top1"].notna().all() and table["best_layer_max_auc"].iloc[-1] == 1
    mat, revs = sc.layer_evolution(tables)
    assert mat.shape == (L, 3) and revs == REVS and mat[1, 2] == final["top_auc"]
    csv = str(tmp_path / "curve.csv")
    out = sc.build_sweep(tables, "synthetic", "honesty", "pythia", csv, str(tmp_path / "curve.png"))
    assert os.path.exists(csv) and os.path.exists(tmp_path / "curve.png") and len(pd.read_csv(csv)) == 3 and len(out) == 3
    s = sc.summarize_table(tables["step0"])
    assert s["top_flat"] == 1 * H + 2 and s["n_acc_gt_0.8"] == 0 and "p_adj_top" in s


def test_correlations(tables, tmp_path):
    grid = np.arange(12, dtype=float).reshape(3, 4)
    if os.path.isdir(os.path.join(REPO, "analysis")):
        sys.path.insert(0, os.path.join(REPO, "analysis"))
        sys.path.insert(0, REPO)
        try:
            from value_correlations import resample as repo_resample
            assert np.allclose(corr.resample(grid, 5, 7), repo_resample(grid, 5, 7))
        except ImportError:          # the repo script needs seaborn / transformers at import time
            pass
    assert np.allclose(corr.resample(grid, 3, 4), grid) and corr.resample(grid, 6, 2).shape == (6, 2)
    by_value = {"honesty": tables["step143000"], "helpful": tables["step0"], "verbosity": synthetic_table(9, 0.3, "step143000", "verbosity")}
    res = corr.across_values(by_value)
    assert set(res) == {"pearson_topk", "pearson_score", "spearman_score"}
    for m in res.values():
        assert m.shape == (3, 3) and np.allclose(np.diag(m), 1) and np.allclose(m, m.T)
    vec = corr.binary_topk_vector(tables["step0"])
    assert vec.sum() == int(0.15 * L * H) and vec[1 * H + 2] == 1
    assert res["pearson_topk"].loc["honesty", "helpful"] > 0            # same planted heads -> positive overlap
    ck = corr.across_tables(tables)
    assert ck["spearman_score"].loc["step0", "step143000"] == pytest.approx(
        pd.Series(tables["step0"].auc.to_numpy()).corr(pd.Series(tables["step143000"].auc.to_numpy()), method="spearman"))
    # a table with another geometry is resampled onto the first one's grid
    small = synthetic_table(1, 0.2, "other")
    small.attrs.update(layers=L, heads=H)
    big = pd.concat([small] * 2, ignore_index=True)
    big["flat_idx"] = np.arange(len(big))
    big["layer"], big["head"] = big["flat_idx"] // H, big["flat_idx"] % H
    big.attrs.update(layers=2 * L, heads=H, score="auc")
    mixed = corr.across_tables({"small": small, "big": big})
    assert mixed["pearson_score"].shape == (2, 2) and not np.isnan(mixed["pearson_score"].iloc[0, 1])
    tidy = corr.tidy(res["pearson_topk"], "pearson_topk")
    assert len(tidy) == 3 and set(tidy.columns) == {"a", "b", "kind", "r"}
    corr.plot_correlation(res["pearson_score"], str(tmp_path / "c.png"), title="c")
    assert os.path.exists(tmp_path / "c.png")
    assert np.isnan(corr.correlation_matrix([np.ones(5), np.arange(5.0)], ["a", "b"]).iloc[0, 1])


def test_direction_angles(tables, tmp_path):
    rng = np.random.RandomState(0)
    da = rng.randn(L * H, D).astype(np.float32)
    db = da.copy()
    db[1 * H + 2] = -da[1 * H + 2]                      # opposite on the planted head
    db[2 * H + 0] = np.r_[da[2 * H + 0][1:], 0] * 0 + rng.randn(D)
    if os.path.isdir(os.path.join(REPO, "analysis")):
        sys.path.insert(0, os.path.join(REPO, "analysis"))
        sys.path.insert(0, REPO)
        from direction_angles import angle_deg as repo_angle
        for i in range(5):
            assert abs(ang.angle_deg(da[i], db[i]) - repo_angle(da[i], db[i])) < 1e-4
    assert ang.angle_deg(da[0], da[0]) == pytest.approx(0.0, abs=1e-3) and ang.angle_deg(da[0], -da[0]) == pytest.approx(180.0, abs=1e-3)
    assert np.isnan(ang.angle_deg(np.zeros(D), da[0]))
    heads = {"honesty": select_topk(tables["step143000"], 0.15), "helpful": select_topk(tables["step0"], 0.15)}
    angles, shared = ang.shared_head_angles(da, db, heads["honesty"], heads["helpful"])
    assert (1 * H + 2) in shared and len(angles) == len(shared) >= 1
    table = ang.angle_table({"honesty": da, "helpful": db}, heads, "synthetic", "step143000")
    assert len(table) == 1 and table["n_shared"].iloc[0] == len(shared) and table["pair"].iloc[0] == "honesty x helpful"
    assert table["std_deg"].iloc[0] == pytest.approx(float(np.std(angles)))          # ddof = 0 as in Table 17
    assert 0 <= table["mean_deg"].iloc[0] <= 180
    ang.plot_angles(table, str(tmp_path / "a.png"), title="angles")
    assert os.path.exists(tmp_path / "a.png")
    empty = ang.angle_table({"a": da, "b": db}, {"a": [0], "b": [1]})
    assert empty["n_shared"].iloc[0] == 0 and np.isnan(empty["mean_deg"].iloc[0])
    ang.plot_angles(empty, str(tmp_path / "e.png"))
    allh = ang.all_head_angle_table({"a": da, "b": db})
    assert allh["n"].iloc[0] == L * H and 0 < allh["mean_deg"].iloc[0] < 180


def test_summary_and_report(tables, tmp_path):
    keyed = {(r, "honesty", "pythia", None): df for r, df in tables.items()}
    keyed[("step143000", "honesty", "pythia", "headmean")] = synthetic_table(5, 0.2, "step143000", with_stats=False)
    summary = rp.build_summary(keyed, paths={k: f"/x/{k[0]}.json" for k in keyed})
    assert list(summary.columns) == rp.SUMMARY_COLUMNS and len(summary) == 4
    assert list(summary["revision"]) == ["step0", "step2000", "step143000", "step143000"]
    row = summary[(summary.revision == "step143000") & (summary.feature_tag == "primary")].iloc[0]
    assert row["chance_auc"] == 0.5 and row["top_layer"] == 1 and row["top_head"] == 2 and row["stats_done"] and row["perm_thr"] == 0.62
    assert not summary[summary.feature_tag == "headmean"].iloc[0]["stats_done"]
    csv = rp.write_summary(summary, str(tmp_path / "summary.csv"), str(tmp_path / "summary.json"))
    assert len(pd.read_csv(csv)) == 4 and len(json.load(open(tmp_path / "summary.json"))) == 4
    sweep = {("honesty", "pythia"): sc.sweep_table(tables, "synthetic", "honesty")}
    cm = corr.across_tables(tables)
    angles = ang.angle_table({"a": np.ones((L * H, D)), "b": np.ones((L * H, D))}, {"a": [0, 1], "b": [1]}, "synthetic", "step143000")
    manifest = {"revisions": {"step143000": {"commit_sha": "abc123", "dtype": "float32", "attn_impl": "eager", "transformers": "5", "torch": "2"}}}
    eval_dir = tmp_path / "eval" / "synthetic_step143000"
    eval_dir.mkdir(parents=True)
    pd.DataFrame({"metric": ["tqa_pairwise"], "base": [0.5], "topk": [0.48]}).to_csv(eval_dir / "summary.csv", index=False)
    ev = rp.load_eval_summaries(str(tmp_path / "eval"), "synthetic")
    assert set(ev) == {"step143000"} and rp.load_eval_summaries(str(tmp_path / "nope"), "synthetic") == {}
    # the global results/eval/summary.csv of eval-likelihood (one row per run, column revision) is picked up per revision
    pd.DataFrame({"model": ["synthetic", "synthetic", "other"], "revision": ["step143000", "step0", "step0"], "metric": ["wikitext_ppl"] * 3,
                  "intervention": ["mask"] * 3, "primary": ["ppl"] * 3, "base_ppl": [27.4, 30.0, 1.0], "topk_ppl": [48.7, 31.0, 1.0],
                  "random_mean": [49.3, 31.0, 1.0], "delta_mean": [0.58, 0.01, 0.0]}).to_csv(tmp_path / "eval" / "summary.csv", index=False)
    ev = rp.load_eval_summaries(str(tmp_path / "eval"), "synthetic")
    assert set(ev) == {"step143000", "step0"} and len(ev["step143000"]) == 2 and len(ev["step0"]) == 1
    compact = rp.compact_eval_summary(ev["step0"])
    assert list(compact.columns)[:3] == ["metric", "intervention", "primary"] and compact["base"].iloc[0] == 30.0 and compact["topk"].iloc[0] == 31.0
    assert rp.compact_eval_summary(pd.DataFrame({"x": [1]})).columns.tolist() == ["x"]
    # judge summary: restricted to the model, last row per (input, judge, value)
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    pd.DataFrame({"model": ["synthetic", "synthetic", "other"], "revision": ["step143000"] * 3, "input": ["a", "a", "b"], "judge": ["heuristic"] * 3,
                  "value": ["harmless"] * 3, "condition": ["base"] * 3, "n": [8, 8, 8], "rate": [0.0, 0.1, 0.5], "base_rate": [0.0, 0.1, 0.5],
                  "floor_effect": [1, 1, 0]}).to_csv(judge_dir / "summary.csv", index=False)
    js = rp.load_judge_summary(str(judge_dir), "synthetic")
    assert len(js) == 1 and js["rate"].iloc[0] == 0.1 and rp.load_judge_summary(str(tmp_path / "nope"), "synthetic") is None
    stab = {"step143000 / honesty": {"spearman_mean": 0.9, "spearman_min": 0.85, "jaccard_mean": 0.7, "jaccard_min": 0.6, "k": 1, "n": 2}}
    out = str(tmp_path / "report.md")
    text = rp.render_report("synthetic", summary, sweep, cm, angles, manifest, {("sweep", "honesty", "pythia"): str(tmp_path / "s.png")}, ev, stab, out,
                            judge_summary=js)
    assert os.path.exists(out) and text.startswith("# ValueHeads auf Pythia")
    for needle in ("## 1. Setup", "## 2. Head-Scores", "## 3. Checkpoint-Sweep", "## 4. Split-Stabilität", "## 5. Korrelationen",
                   "## 6. Richtungswinkel", "## 7. Likelihood-Evaluation", "## 9. Einschränkungen", "abc123", "L1.H2", "tqa_pairwise",
                   "a x b", "0.62", "Permutationsschwelle", "wikitext_ppl", "48.7000", "### Generation + Judge", "heuristic"):
        assert needle in text, needle
    assert rp.md_table(pd.DataFrame()) == "_keine Daten_\n" and "weitere Zeilen" in rp.md_table(summary, ["revision"], max_rows=2)
    assert rp._fmt(np.nan) == "–" and rp._fmt(True) == "ja" and rp._fmt(np.int64(3)) == "3" and rp._fmt(0.12345, 2) == "0.12"
    text2 = rp.render_report("synthetic", summary)
    assert "keine Sweep-Kurven" in text2 and "keine Korrelationen" in text2 and "kein `results/manifest.json`" in text2


def test_cache_cleanup_on_fake_cache(tmp_path):
    from valueheads_pythia.commands.sweep import cleanup_hf_revision, revision_commit_sha
    hub = tmp_path / "hub"
    repo = hub / "models--EleutherAI--fake-1m"
    sha_a, sha_b = "a" * 40, "b" * 40
    for sha, refs in ((sha_a, ["step0"]), (sha_b, ["main", "step143000"])):
        snap = repo / "snapshots" / sha
        snap.mkdir(parents=True)
        (repo / "blobs").mkdir(exist_ok=True)
        blob = repo / "blobs" / ("blob" + sha[:6])
        blob.write_text("{}" * 100)
        os.symlink(blob, snap / "config.json")
        (repo / "refs").mkdir(exist_ok=True)
        for r in refs:
            (repo / "refs" / r).write_text(sha)
    assert revision_commit_sha("EleutherAI/fake-1m", "step0", cache_dir=str(hub)) == sha_a
    assert revision_commit_sha("EleutherAI/fake-1m", "step0", {"commit_sha": "from-manifest"}) == "from-manifest"
    res = cleanup_hf_revision("EleutherAI/fake-1m", "step143000", cache_dir=str(hub))
    assert not res["deleted"] and "protected" in res["reason"]                       # shares its commit with main
    dry = cleanup_hf_revision("EleutherAI/fake-1m", "step0", cache_dir=str(hub), dry_run=True)
    assert not dry["deleted"] and dry["dry_run"] and dry["commit"] == sha_a and (repo / "snapshots" / sha_a).exists()
    res = cleanup_hf_revision("EleutherAI/fake-1m", "step0", cache_dir=str(hub))
    assert res["deleted"] and res["freed_bytes"] > 0 and not (repo / "snapshots" / sha_a).exists() and (repo / "snapshots" / sha_b).exists()
    assert not cleanup_hf_revision("EleutherAI/fake-1m", "step0", cache_dir=str(hub))["deleted"]
    assert "not in cache" in cleanup_hf_revision("EleutherAI/other", "step0", cache_dir=str(hub))["reason"]


def test_sweep_plan_and_analyze_command(tmp_path, monkeypatch, capsys):
    """sweep --dry_run prints the plan; analyze builds every artifact from a synthetic run directory."""
    from valueheads_pythia.cli import main
    root = str(tmp_path / "run")
    assert main(["sweep", "--model", "pythia-160m", "--smoke", "--values", "honesty", "coherence_strict", "--dry_run", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "revisions=['step0', 'step2000', 'step143000']" in out and "helpsteer2_train" in out and "B=200 P=20" in out
    with pytest.raises(SystemExit):
        main(["sweep", "--model", "pythia-160m", "--values", "harmless_pooled", "--dry_run", "--root", root])
    # synthetic run directory: three revisions of honesty (+ a split table and a second value with directions)
    p = config.paths(root)
    rng = np.random.RandomState(0)
    for i, rev in enumerate(REVS):
        df = synthetic_table(i, 0.1 + 0.1 * i, rev)
        save_head_scores(df, config.heads_path(p, "pythia-160m", rev, "honesty", "pythia"),
                         meta={"model": "pythia-160m", "revision": rev, "value": "honesty", "mode": "pythia", "feature_tag": None})
    save_head_scores(synthetic_table(7, 0.25, "step143000"), config.heads_path(p, "pythia-160m", "step143000", "honesty", "pythia", "split1"),
                     meta={"model": "pythia-160m", "revision": "step143000", "value": "honesty", "mode": "pythia", "feature_tag": "split1"})
    save_head_scores(synthetic_table(8, 0.2, "step143000", "helpful"), config.heads_path(p, "pythia-160m", "step143000", "helpful", "pythia"),
                     meta={"model": "pythia-160m", "revision": "step143000", "value": "helpful", "mode": "pythia", "feature_tag": None})
    save_head_scores(synthetic_table(8, 0.2, "step143000", "honesty", "paper"), config.heads_path(p, "pythia-160m", "step143000", "honesty", "paper"),
                     meta={"model": "pythia-160m", "revision": "step143000", "value": "honesty", "mode": "paper", "feature_tag": None})
    from valueheads_pythia.directions import save_directions
    for v in ("honesty", "helpful"):
        save_directions(config.directions_path(p, "pythia-160m", "step143000", v), rng.randn(L * H, D).astype(np.float32), {v: np.ones(L * H)}, {"value": v})
    assert main(["analyze", "--model", "pythia-160m", "--root", root, "--quiet"]) == 0
    assert os.path.exists(os.path.join(p["results"], "summary_pythia-160m.csv")) and os.path.exists(os.path.join(p["results"], "report_pythia-160m.md"))
    assert os.path.exists(os.path.join(p["sweep"], "pythia-160m_honesty_curve.csv")) and os.path.exists(os.path.join(p["figures_sweep"], "pythia-160m_honesty.png"))
    assert os.path.exists(os.path.join(p["heatmaps"], "pythia-160m_step143000_honesty.png")) and os.path.exists(os.path.join(p["heatmaps"], "pythia-160m_step143000_pythia_all.png"))
    assert os.path.exists(os.path.join(p["heatmaps"], "pythia-160m_step143000_honesty_paper.png")) and os.path.exists(os.path.join(p["heatmaps"], "pythia-160m_step0_pythia_layers.png"))
    assert os.path.exists(os.path.join(p["results"], "correlations_pythia-160m.csv")) and os.path.exists(os.path.join(p["results"], "direction_angles_pythia-160m.csv"))
    assert os.path.exists(os.path.join(p["figures"], "angles", "pythia-160m_step143000.png"))
    summary = pd.read_csv(os.path.join(p["results"], "summary_pythia-160m.csv"))
    assert len(summary) == 6 and set(summary["value"]) == {"honesty", "helpful"} and "split1" in set(summary["feature_tag"])
    report = open(os.path.join(p["results"], "report_pythia-160m.md")).read()
    assert "Split-Stabilität" in report and ("honesty x helpful" in report or "helpful x honesty" in report) and "über Checkpoints" in report
    curve = pd.read_csv(os.path.join(p["sweep"], "pythia-160m_honesty_curve.csv"))
    assert list(curve["revision"]) == REVS and curve["spearman_to_final"].iloc[-1] == 1.0
    # restricted to one revision: no sweep, still a report
    assert main(["analyze", "--model", "pythia-160m", "--root", root, "--revision", "step0", "--no_figures", "--quiet"]) == 0
    with pytest.raises(SystemExit):
        main(["analyze", "--model", "pythia-410m", "--root", root, "--quiet"])


def test_build_curves_writes_mean_over_splits_ranking(tmp_path):
    """sweep.build_curves with split tables: stability CSV plus the mean-AUC-over-splits head JSON (DESIGN 5.6), readable by
    top_heads_from_file / eval-likelihood --heads_file."""
    from valueheads_pythia.commands.sweep import build_curves, save_mean_splits
    from valueheads_pythia.probing import top_heads_from_file
    from valueheads_pythia.probing_stats import split_stability
    root = str(tmp_path / "run")
    p = config.paths(root)
    main_tables = {}
    for i, rev in enumerate(REVS):
        df = synthetic_table(i, 0.1 + 0.1 * i, rev)
        main_tables[rev] = df
        save_head_scores(df, config.heads_path(p, "pythia-160m", rev, "honesty", "pythia"),
                         meta={"model": "pythia-160m", "revision": rev, "value": "honesty", "mode": "pythia", "feature_tag": None})
    splits = {1: synthetic_table(7, 0.25, "step143000"), 2: synthetic_table(9, 0.3, "step143000")}
    for seed, df in splits.items():
        save_head_scores(df, config.heads_path(p, "pythia-160m", "step143000", "honesty", "pythia", f"split{seed}"),
                         meta={"model": "pythia-160m", "revision": "step143000", "value": "honesty", "mode": "pythia", "feature_tag": f"split{seed}"})
    out = build_curves(p, "pythia-160m", ["honesty"], REVS, "pythia", split_seeds=[1, 2])
    assert "honesty" in out and os.path.exists(os.path.join(p["sweep"], "pythia-160m_honesty_stability.csv"))
    mean_path = config.heads_path(p, "pythia-160m", "step143000", "honesty", "pythia", "meansplits")
    assert os.path.exists(mean_path) and not os.path.exists(config.heads_path(p, "pythia-160m", "step0", "honesty", "pythia", "meansplits"))
    with open(mean_path) as f:
        payload = json.load(f)
    assert payload["meta"]["splits"] == ["seed42", "seed1", "seed2"] and payload["meta"]["score"] == "auc_mean" and payload["k"] == int(0.15 * L * H)
    st = split_stability([main_tables["step143000"], splits[1], splits[2]], names=["seed42", "seed1", "seed2"])
    expected = st["mean_scores"]
    assert payload["topk"] == [int(i) for i in expected["flat_idx"].head(payload["k"])]
    assert top_heads_from_file(mean_path, 0.15) == payload["topk"]
    heads = payload["heads"]
    assert [h["rank"] for h in heads] == list(range(1, L * H + 1)) and heads[0]["auc_mean"] >= heads[1]["auc_mean"]
    stacked = np.mean([main_tables["step143000"].set_index("flat_idx")["auc"], splits[1].set_index("flat_idx")["auc"],
                       splits[2].set_index("flat_idx")["auc"]], axis=0)
    assert np.isclose(heads[0]["auc_mean"], stacked.max())
    # the writer itself (used by build_curves) keeps meta and k consistent
    again = save_mean_splits(str(tmp_path / "m.json"), expected, ["a", "b"], {"layers": L, "heads": H, "model": "x"}, ratio=0.5)
    assert again["k"] == int(0.5 * L * H) and again["meta"]["n_splits"] == 2
    # analyze skips the derived meansplits ranking (no probe columns) instead of crashing on it
    from valueheads_pythia.cli import main
    from valueheads_pythia.commands.analyze import discover_tables
    tables = discover_tables(p, "pythia-160m")
    assert all(tag != "meansplits" for (_, _, _, tag) in tables) and ("step143000", "honesty", "pythia", "split1") in tables
    assert main(["analyze", "--model", "pythia-160m", "--root", root, "--no_figures", "--quiet"]) == 0
    summary = pd.read_csv(os.path.join(p["results"], "summary_pythia-160m.csv"))
    assert "meansplits" not in set(summary["feature_tag"]) and {"split1", "split2"} <= set(summary["feature_tag"])
