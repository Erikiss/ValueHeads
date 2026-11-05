"""`ablate` command (docs/ABLATION_DESIGN.md part D, 8.11): registration, notebook reproducibility, the end-to-end
pipeline on a tiny random Qwen3 (synthetic ASCII HH items, synthetic heads table, synthetic census and ICL battery),
idempotency (a second run does no forward), the frozen-before-EVAL order, gate paths, the report and the
fixed-identity arm. Offline; no download.

The tiny-model helpers (`tiny_qwen3`, `CharTokenizer`) are copied from tests/test_qwen_port.py (no cross-test imports);
the tokenizer gets a lossless `decode` for the M7 covariates."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from valueheads_pythia import ablation_metrics as AM
from valueheads_pythia import ablation_report as AR
from valueheads_pythia import ablation_sets as AS
from valueheads_pythia import cli, config, eval_likelihood, icl_sets
from valueheads_pythia.commands import ablate
from valueheads_pythia.icl_battery import ICLItem
from valueheads_pythia.probing import _finish_table, save_head_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 128
L, H, KV, D, HIDDEN = 4, 8, 4, 8, 48
BASE, INST = "tiny-qwen3-base", "tiny-qwen3"


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class CharTokenizer:
    """Minimal HF-like tokenizer (one id per character), copied from tests/test_qwen_port.py, plus `decode`."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 0
    bos_token_id = None

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"

    def encode(self, text):
        return [1 + (ord(c) % (VOCAB - 1)) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i) - 1) for i in ids)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, add_special_tokens=True):
        texts = [text] if isinstance(text, str) else list(text)
        ids = [self.encode(t) for t in texts]
        if return_tensors is None:
            return {"input_ids": ids if not isinstance(text, str) else ids[0]}
        width = max(len(s) for s in ids)
        input_ids = torch.full((len(ids), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for r, s in enumerate(ids):
            input_ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": mask})


def tiny_qwen3(heads=H, kv_heads=KV, head_dim=D, hidden=HIDDEN, layers=L, seed=0):
    """Random Qwen3 with GQA (2 query heads per KV head) and H * D = 64 != hidden 48 (copied from test_qwen_port)."""
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=hidden, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, max_position_embeddings=256,
                      attention_bias=False, tie_word_embeddings=True, pad_token_id=0, eos_token_id=0, bos_token_id=None)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for block in model.model.layers:
            for lin in (block.self_attn.v_proj, block.self_attn.q_proj, block.self_attn.k_proj):
                lin.weight.normal_(0, 0.3)
    return model


# ----------------------------------------------------------------------------------------------------------- inputs

def synthetic_items(n=60):
    """Synthetic ASCII HH-style items (prompt clusters: groups g and g + 50 share a prompt)."""
    return [{"group": g, "context": f"\n\nHuman: what is thing {g % 50} and why?\n\nAssistant:",
             "pos": [f" thing {g % 50} is good because it helps"], "neg": [f" no idea {g}"], "meta": {}} for g in range(n)]


def write_table(root, short, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.55 + 0.05 * rng.random(L * H)
    for f in (9, 10, 17, 26):
        s[f] += 0.3
    rows = [{"layer": l, "head": h, "flat_idx": l * H + h, "auc": float(s[l * H + h]), "acc": float(s[l * H + h]),
             "bal_acc": 0.5, "majority": 0.5, "pos_rate": 0.5, "n_train": 20000, "n_val": 1400}
            for l in range(L) for h in range(H)]
    df = _finish_table(rows, {"layers": L, "heads": H, "score": "auc", "ranking": "stable", "feature_mode": "head",
                              "mode": "pythia"}, "auc", "stable")
    save_head_scores(df, config.heads_path(config.paths(str(root)), short, None, "helpful", "pythia"),
                     meta={"model": short, "mode": "pythia"})


def synthetic_census():
    rng = np.random.default_rng(1)
    rows = []
    for l in range(L):
        for h in range(H):
            d = float(rng.normal(0, 0.1))
            rows.append({"L": l, "H": h, "dret": d, "dret_lo": d - 0.05, "dret_hi": d + 0.05, "dR": 0.0,
                         "dP": float("nan"), "dP_lo": float("nan"), "dP_hi": float("nan")})
    return icl_sets.CensusResult(BASE, icl_sets.classify_census(rows), frozenset((r["L"], r["H"]) for r in rows),
                                 tuple(range(L)), tuple(rows), "synthetic", "0" * 64, False)


def icl_builder(tok, seed, smoke):
    """Synthetic ICL battery: 2 concepts x 2 draws x 4 label sets (multi-token alt for `rare`)."""
    rng = np.random.default_rng(seed)
    out = []
    for ci, c in enumerate(("topic", "lang")):
        for s in range(2):
            base = rng.integers(1, VOCAB, 18).tolist()
            for ls in ("sym", "rare", "nat", "flip"):
                tgt = (int(rng.integers(1, VOCAB)),)
                alt = (int(rng.integers(1, VOCAB)),) if ls != "rare" else tuple(int(x) for x in rng.integers(1, VOCAB, 2))
                out.append(ICLItem(len(out), f"icl_concept_{c}_{ls}_12n", c, ls, s, bool(s % 2), tuple(base), "t", "a",
                                   tgt, alt, ci * 100 + s))
    return out


class World:
    def __init__(self, root):
        self.root = str(root)
        self.model = tiny_qwen3()
        self.tok = CharTokenizer()
        items = synthetic_items()
        self.by_id = {it["group"]: it for it in items}
        self.split = AS.build_split(items, [], set(), set(), n_means_ref=8, n_means_probe=8, n_eval=12, seed=42)
        self.census = synthetic_census()
        self.inject = {"ppl_ids": np.random.default_rng(3).integers(1, VOCAB, 700), "icl_items": icl_builder}
        self.calls = [0]
        self.model.get_decoder().register_forward_pre_hook(lambda *a, **k: self.calls.__setitem__(0, self.calls[0] + 1))

    def run(self, *argv, model=BASE, icl=None):
        p = argparse.ArgumentParser()
        ablate.add_args(p)
        common = ["--model", model, "--root", self.root, "--device", "cpu", "--smoke", "--band", "1-3", "--bootstrap",
                  "30", "--no_progress", "--quiet"] + (["--icl_model", icl] if icl else [])
        args = p.parse_args(common + list(argv))
        return ablate.run_pipeline(args, self.model, self.tok, self.split, self.by_id, self.census, inject=self.inject)

    def abl(self, *parts):
        return os.path.join(self.root, "results", "ablation", *parts)


ARMS_NO_M3 = ["mean_sets", "m4", "mean_last", "m5", "m7", "resample", "dir_mean", "gauss"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("ablate"))
    write_table(w.root, BASE)
    write_table(w.root, INST, seed=5)
    codes = [w.run("--stage", "0"), w.run("--stage", "1", "--pilot"), w.run("--stage", "1"),
             w.run("--stage", "2", "--arms", *ARMS_NO_M3), w.run("--summary")]
    w.codes = codes
    return w


# ----------------------------------------------------------------------------------------------------------- tests

def test_ablate_is_registered_and_has_help(capsys):
    assert "ablate" in cli.COMMANDS
    assert config.paths("/x")["ablation"] == os.path.join("/x", "results", "ablation")
    with pytest.raises(SystemExit) as e:
        cli.main(["ablate", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--stage" in out and "--pilot" in out and "--extend_random" in out and "--ignore_gate" in out


def test_ablation_notebooks_are_generated_reproducibly(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_ablation_notebook.py"), "--out", str(tmp_path)],
                   check=True, capture_output=True)
    for name, model, fixed in (("qwen3-4b-base-ablation", "Qwen/Qwen3-4B-Base", "False"),
                               ("qwen3-4b-ablation", "Qwen/Qwen3-4B", "True")):
        fresh = (tmp_path / f"{name}.ipynb").read_text()
        committed = os.path.join(ROOT, "notebooks", "runs", f"{name}.ipynb")
        assert fresh == open(committed).read(), f"notebooks/runs/{name}.ipynb is stale: run scripts/make_ablation_notebook.py"
        nb = json.loads(fresh)
        code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "nb", "exec")
                assert cell["outputs"] == [] and cell["execution_count"] is None
        assert nb["metadata"]["colab"]["gpuType"] == "A100"
        assert f'MODEL = "{model}"' in code and 'MyDrive/ValueHeadsPythia/qwen3-4b"' in code
        assert 'DRIVE_PROJECT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"' in code
        assert f'"fixed_identity": {fixed}' in code
        for needle in ('"ablate"', '"--stage", 0', '"--stage", 1', '"--stage", 2', '"--pilot"', '"--summary"'):
            assert needle in code, needle
        for banned in ('"eval-generate"', 'run("judge"', '"judge", "--', '"eval-likelihood"'):
            assert banned not in code
        assert "def run(" in code and "VH_NB_MODELS" in code and "VH_NB_ARMS" in code


def test_pipeline_end_to_end(world):
    w = world
    assert w.codes == [0, 0, 0, 0, 0]
    conds = AR.load_conditions(w.abl(BASE, "conditions"))
    for name in ("clean__hh3", "clean__icl", "knock__icl__CR20", "knock__icl__CR20__r000", "mean__hh3__C33",
                 "mean__hh3__C33__r001", "mean__hh3__O", "clean__hh2", "mean__ppl__C33", "mean_last__hh2__C33",
                 "dla__hh2__C33", "mean__hhshuf__O", "resample__hh2__O__r000", "dir_mean__hh2__C33__d001",
                 "gauss__hh2__C33"):
        assert name in conds, name
    assert not any(n.startswith("knock__icl__C33") for n in conds)              # m3 not selected
    # every condition is paired with the clean run of its variant (same items, same plan)
    c, cl = conds["mean__hh3__C33"], conds["clean__hh3"]
    assert c["items"]["id"] == cl["items"]["id"] and c["config"]["variant_seqs_sha"] == cl["config"]["variant_seqs_sha"]
    assert len(c["items"]["id"]) == 8 and c["n_heads"] == 8 and "m2" in c["summary"]
    assert "covariates" in cl and len(cl["covariates"]["reuse_pos"]) == 8
    assert os.path.exists(w.abl(BASE, "conditions", "mean__hh3__C33.m2.npz"))
    assert not os.path.exists(w.abl(BASE, "conditions", "mean__hh3__C33__r000.m2.npz"))
    # mean_last cannot change any scored log-prob (F4)
    ml, c2 = conds["mean_last__hh2__C33"], conds["clean__hh2"]
    assert np.allclose(ml["items"]["pos_sum"], c2["items"]["pos_sum"], atol=1e-6, rtol=0)
    # stage-0 artefacts and reports
    for f in ("selftest.json", "refs.npz", "m2_train.npz", "probes_m2.npz", "floor.json", "gate.json", "pilot.json",
              "icl_kind_means_s2.npz"):
        assert os.path.exists(w.abl(BASE, f)), f
    assert json.load(open(w.abl(BASE, "selftest.json")))["ok"]
    gate = json.load(open(w.abl(BASE, "gate.json")))
    assert gate["advisory"] and gate["proceed"] and len(gate["cr20"]) > 0
    pilot = json.load(open(w.abl(BASE, "pilot.json")))
    assert pilot["measured_from"] == ["mean__hh3__C33", "mean__hh3__C33__r000"] and pilot["total"]["n"] > 0
    assert set(pilot["overhead_seconds"]) == {"named", "random"}
    rep = json.load(open(w.abl(f"{BASE}_ablation.json")))
    assert [r["set"] for r in rep["primary"]] == ["C33", "O"]
    # smoke: never interpretable; the machine-readable verdict says so (F25 / 6.6)
    assert all(r["interpretable"] is False and "Smoke, kein Befund" in r["verdict"] for r in rep["primary"])
    assert rep["gate_status"] in ("passed", "advisory")
    # the paper's positive-control targets stand next to the gate (other battery / set, context only)
    assert [t["id"] for t in rep["positive_control_targets"]] == ["reader20_4b", "cr20_llama", "cr20_sum_4b"]
    p = [r["p_one"] for r in rep["primary"]]
    from valueheads_pythia import overlap
    assert np.allclose([r["p_holm"] for r in rep["primary"]], overlap.holm(p))
    tests = pd.read_csv(w.abl(f"{BASE}_ablation_tests.csv"))
    assert list(tests.columns) == AR.TEST_COLUMNS
    assert set(tests["metric"]) >= {"M1", "M2", "M3", "M4", "M5"} and (tests["family"] == "primary").sum() == 2
    md = open(w.abl(f"{BASE}_ablation.md")).read()
    assert "SMOKE: nur Funktionsprüfung, kein Befund." in md and "synthetischer Smoke-Ersatz" in md
    assert "## Primäre Familie" in md and "## Abweichungen vom Design" in md and "## Trajektorie" in md
    assert "Zielwerte des Papers" in md and "-0.74 [-0.95, -0.55] vs. Zufall +0.02 [-0.03, +0.06]" in md
    assert "-0.56 [-1.01, -0.12] vs. Zufall +0.00" in md and "main.tex:394, :601" in md
    frozen = AS.load_frozen(w.abl(f"{BASE}_frozen_sets.json"))
    assert frozen["smoke"] and frozen["sets"]["C33"]["synthetic"] and frozen["inputs"]["heads_table"]["sha256"]
    # stage 1 used R = 2 frozen draws of C33 (smoke) in the named order
    assert [n for n in conds if n.startswith("mean__hh3__C33__r")] == ["mean__hh3__C33__r000", "mean__hh3__C33__r001"]
    assert conds["mean__hh3__C33__r000"]["heads"] == frozen["random"]["C33"]["draws"][0]


def test_frozen_file_is_written_before_the_first_eval_condition(world):
    w = world
    events = [json.loads(line) for line in open(w.abl(BASE, "run_log.jsonl"))]
    names = [e["event"] if e["event"] != "condition" else e["name"] for e in events]
    first_eval = min(i for i, n in enumerate(names) if AR.parse_name(n)["variant"] in AM.HH_VARIANTS)
    assert names.index("frozen") < first_eval
    assert os.path.getmtime(w.abl(f"{BASE}_frozen_sets.json")) <= os.path.getmtime(
        w.abl(BASE, "conditions", "clean__hh3.json"))


def test_second_run_skips_everything_and_batch_change_recomputes(world):
    w = world
    w.calls[0] = 0
    assert w.run("--stage", "0") == 0
    assert w.run("--stage", "1") == 0
    assert w.run("--stage", "2", "--arms", *ARMS_NO_M3) == 0
    assert w.calls[0] == 0                                           # no forward at all
    key = json.load(open(w.abl(BASE, "conditions", "mean__hh3__C33.json")))["key"]
    assert w.run("--stage", "1", "--batch_tokens", "512") == 0
    assert w.calls[0] > 0
    new = json.load(open(w.abl(BASE, "conditions", "mean__hh3__C33.json")))
    assert new["key"] != key and new["config"]["batch_tokens"] == 512
    assert json.load(open(w.abl(BASE, "conditions", "clean__hh3.json")))["config"]["batch_tokens"] == 512
    assert w.run("--stage", "1") == 0                               # back to the default plan (recomputed again)


def test_gate_paths_and_report_banners(world, tmp_path):
    w = world
    gate_path = w.abl(BASE, "gate.json")
    good = open(gate_path).read()
    floor_path = w.abl(BASE, "floor.json")
    good_floor = open(floor_path).read()
    try:
        g = json.loads(good)
        g.update(passed=False, advisory=False, proceed=False, reason_de="synthetisch nicht bestanden")
        json.dump(g, open(gate_path, "w"))
        assert w.run("--stage", "1") == ablate.EXIT_GATE
        assert w.run("--summary") == 0
        md = open(w.abl(f"{BASE}_ablation.md")).read()
        assert "**STOPP:**" in md and "## Primäre Familie" not in md and "## Sekundär" not in md
        # the JSON and _tests.csv say the same as the markdown (no "Abhängigkeit nachgewiesen" behind a STOPP)
        rep = json.load(open(w.abl(f"{BASE}_ablation.json")))
        assert rep["gate_status"] == "stop"
        assert all(r["verdict"] == "STOPP (Gate)" and r["interpretable"] is False for r in rep["primary"])
        tests = pd.read_csv(w.abl(f"{BASE}_ablation_tests.csv"))
        prim = tests[tests["family"] == "primary"]
        assert len(prim) == 2 and set(prim["verdict"]) == {"STOPP (Gate)"} and not prim["interpretable"].any()
        assert w.run("--stage", "1", "--ignore_gate") == 0
        md = open(w.abl(f"{BASE}_ablation.md")).read()
        assert "Gate ignoriert" in md and "## Primäre Familie" in md
        rep = json.load(open(w.abl(f"{BASE}_ablation.json")))
        assert rep["gate_status"] == "ignored" and all("Gate ignoriert" in r["verdict"] for r in rep["primary"])
        json.dump(dict(json.loads(good), proceed=True), open(gate_path, "w"))
        fl = json.loads(good_floor)
        fl.update(floor=True, inverted=False, acc=0.5, acc_lo=0.4, acc_hi=0.6)
        json.dump(fl, open(floor_path, "w"))
        assert w.run("--summary") == 0
        md = open(w.abl(f"{BASE}_ablation.md")).read()
        assert "**Boden:**" in md and "nicht interpretierbar (Boden)" in md
        rep = json.load(open(w.abl(f"{BASE}_ablation.json")))
        assert all(r["verdict"] == "nicht interpretierbar (Boden)" and not r["interpretable"] for r in rep["primary"])
        prim = pd.read_csv(w.abl(f"{BASE}_ablation_tests.csv")).query("family == 'primary'")
        assert set(prim["verdict"]) == {"nicht interpretierbar (Boden)"}
    finally:
        open(gate_path, "w").write(good)
        open(floor_path, "w").write(good_floor)
    # instruct: the Base gate is missing in a fresh root -> wait 0 min -> exit 3
    w2 = World(tmp_path)
    write_table(w2.root, INST, seed=5)
    assert w2.run("--stage", "0", "--gate_wait_min", "0", model=INST, icl=BASE) == ablate.EXIT_GATE
    assert os.path.exists(w2.abl(f"{INST}_frozen_sets.json"))         # stage 0 ran up to the wait
    # a failed Base gate: stage 0 stops with exit 3 and still writes the STOPP report (8.7) into the run folder
    os.makedirs(w2.abl(BASE), exist_ok=True)
    json.dump(dict(json.loads(good), passed=False, advisory=False, proceed=False, smoke=False,
                   reason_de="synthetisch nicht bestanden"), open(w2.abl(BASE, "gate.json"), "w"))
    assert not os.path.exists(w2.abl(f"{INST}_ablation.md"))
    assert w2.run("--stage", "0", "--gate_wait_min", "0", model=INST, icl=BASE) == ablate.EXIT_GATE
    md = open(w2.abl(f"{INST}_ablation.md")).read()
    assert "**STOPP:**" in md and "synthetisch nicht bestanden" in md and "## Primäre Familie" not in md


def test_fixed_identity_arm_uses_base_draws(world, tmp_path, capsys):
    w = world
    assert w.run("--stage", "0", model=INST, icl=BASE) == 0            # the Base gate (advisory) exists in this root
    assert w.run("--stage", "2", "--arms", "fixed_identity", model=INST, icl=BASE) == 0
    base = AS.load_frozen(w.abl(f"{BASE}_frozen_sets.json"))
    conds = AR.load_conditions(w.abl(INST, "conditions"))
    for s in ablate.FIXED_IDENTITY_SETS:
        if base["sets"][s]["n"] == 0:
            continue
        c = conds[f"mean__hh2__{s}_base"]
        assert c["heads"] == sorted(base["sets"][s]["flat"])
        owner, entry = AS._random_entry(base, s)
        assert conds[f"mean__hh2__{owner}_base__r000"]["heads"] == sorted(entry["draws"][0])
    assert w.run("--summary", model=INST, icl=BASE) == 0
    md = open(w.abl(f"{INST}_ablation.md")).read()
    assert AR.S.TRANSFER_MARK in md and "## Trajektorie Base → Instruct" in md
    assert "main.tex:470, :605" in md                                                # instruct transfer target
    rep = json.load(open(w.abl(f"{INST}_ablation.json")))
    assert rep["trajectory"] and {"set_base", "set_instruct", "sign"} <= set(rep["trajectory"][0])
    # without the Base frozen file the arm is skipped with a message
    w2 = World(tmp_path)
    write_table(w2.root, INST, seed=5)
    assert w2.run("--stage", "0", "--gate_wait_min", "0", "--ignore_gate", model=INST, icl=BASE) == 0
    capsys.readouterr()
    assert w2.run("--stage", "2", "--arms", "fixed_identity", "--ignore_gate", model=INST, icl=BASE) == 0
    assert "fixed_identity übersprungen" in capsys.readouterr().out
    assert not any(n.endswith("_base") for n in AR.load_conditions(w2.abl(INST, "conditions")))


def test_changed_n_eval_is_refused_and_the_report_skips_stale_conditions(world, tmp_path):
    """N_EVAL is not a read-time prefix (it fixes the items and the hh3 partners, i.e. every HH key): a changed value
    is refused before any forward (exit 2), --change_n_eval accepts it, and a condition left over from the old value
    is skipped and listed by the report instead of crashing it."""
    w = World(tmp_path / "root")
    shutil.copytree(os.path.join(world.root, "results"), os.path.join(w.root, "results"))
    cond = lambda n: w.abl(BASE, "conditions", n + ".json")                      # noqa: E731
    before = json.load(open(cond("clean__hh3")))
    w.calls[0] = 0
    for argv in (("--stage", "0"), ("--stage", "1"), ("--stage", "1", "--pilot"), ("--stage", "2", "--arms", "m4")):
        assert w.run(*argv, "--n_eval", "6") == ablate.EXIT_INPUTS, argv
    assert w.calls[0] == 0 and json.load(open(cond("clean__hh3"))) == before          # nothing computed or rewritten
    assert w.run("--summary", "--n_eval", "6") == 0                                    # the report reads what exists
    assert json.load(open(w.abl(f"{BASE}_ablation.json")))["info"]["n_eval"] == 8
    # accepted change: stage 1 recomputes on 6 items; a stale 8-item random of O is left behind on purpose
    old_r1 = open(cond("mean__hh3__O__r001")).read()
    assert w.run("--stage", "1", "--n_eval", "6", "--change_n_eval") == 0
    assert len(json.load(open(cond("clean__hh3")))["items"]["id"]) == 6
    open(cond("mean__hh3__O__r001"), "w").write(old_r1)
    assert w.run("--summary", "--n_eval", "6") == 0
    rep = json.load(open(w.abl(f"{BASE}_ablation.json")))
    stale = {x["condition"]: x for x in rep["stale"]}
    assert "mean__hh3__O__r001" in stale and stale["mean__hh3__O__r001"]["n"] == 8
    assert stale["mean__hh3__O__r001"]["n_clean"] == 6 and rep["info"]["n_eval"] == 6
    assert {r["set"]: r["R"] for r in rep["primary"]}["O"] == 1                      # r001 left out of the null
    assert "Bedingungen übersprungen" in open(w.abl(f"{BASE}_ablation.md")).read()
    conds_csv = pd.read_csv(w.abl(f"{BASE}_ablation_conditions.csv"))
    assert bool(conds_csv.set_index("name").loc["mean__hh3__O__r001", "stale"])
    # the new value is now the stored one: no flag needed, the stale condition is recomputed (its key differs)
    assert w.run("--stage", "1", "--n_eval", "6") == 0
    assert len(json.load(open(cond("mean__hh3__O__r001")))["items"]["id"]) == 6
    assert "mean__hh3__O__r001" not in {x["condition"] for x in json.load(open(w.abl(f"{BASE}_ablation.json")))["stale"]}
    assert w.run("--stage", "1") == ablate.EXIT_INPUTS                               # back to 8: refused again


def _gate_ctx(tmp_path, gate, smoke, ignore=False):
    path = os.path.join(str(tmp_path), "gate.json")
    if gate is not None:
        json.dump(gate, open(path, "w"))
    elif os.path.exists(path):
        os.remove(path)
    said = []
    return types.SimpleNamespace(gate_path=path, smoke=smoke, is_base=False, say=said.append,
                                 args=types.SimpleNamespace(ignore_gate=ignore)), said


def test_full_run_needs_a_passed_gate_of_a_full_run(tmp_path):
    advisory = {"passed": False, "advisory": True, "proceed": True, "smoke": True, "reason_de": "CI nicht unter 0"}
    passed_smoke = dict(advisory, passed=True)
    passed_full = dict(advisory, passed=True, advisory=False, smoke=False)
    failed_full = dict(advisory, advisory=False, proceed=False, smoke=False)
    for gate, smoke, ok in ((advisory, True, True), (advisory, False, False), (passed_smoke, True, True),
                            (passed_smoke, False, False), (passed_full, False, True), (passed_full, True, True),
                            (failed_full, True, False), (failed_full, False, False), (None, False, False)):
        ctx, _ = _gate_ctx(tmp_path, gate, smoke)
        if ok:
            assert ablate.require_gate(ctx) == gate, (gate, smoke)
            continue
        with pytest.raises(ablate.AblateExit) as e:
            ablate.require_gate(ctx)
        assert e.value.code == ablate.EXIT_GATE
        if gate is not None and gate["smoke"] and not smoke:
            assert "Smoke-Lauf" in str(e.value)
    ctx, said = _gate_ctx(tmp_path, advisory, False, ignore=True)
    assert ablate.require_gate(ctx) == advisory and "Gate ignoriert" in said[0]
    # the report applies the same rule: a full report on a smoke gate is a STOPP
    assert AR.gate_status(advisory, smoke=False)[0] == "stop"
    assert AR.gate_status(passed_smoke, smoke=False)[0] == "stop"
    assert AR.gate_status(advisory, smoke=True)[0] == "advisory"
    assert AR.gate_status(passed_full, smoke=False)[0] == "passed"
    assert AR.gate_status(failed_full, smoke=False, ignore_gate=True)[0] == "ignored"
    assert "Smoke-Lauf" in AR.gate_status(advisory, smoke=False)[1]["reason_de"]


def test_eta_overhead_is_estimated_per_kind(world, tmp_path):
    cdir = tmp_path / "conditions"
    cdir.mkdir()

    def put(name, random, sec, wall, arm="mean"):
        json.dump({"name": name, "variant": "hh3", "arm": arm, "random": random,
                   "stats": {"seconds": sec, "tokens_processed": 8000, "wall_seconds": wall}}, open(cdir / f"{name}.json", "w"))
    put("clean__hh3", None, 1.0, 3.24, arm="clean")                    # refit 2.24 s
    put("mean__hh3__C33", None, 1.0, 6.17)                              # refit + an outlier (5.17 s)
    put("mean__hh3__O", None, 1.0, 3.0)                                 # refit 2.0 s
    put("mean__hh3__C33__r000", {"kind": "heads", "index": 0}, 1.0, 1.0)
    ctx = types.SimpleNamespace(cond_dir=str(cdir), _cache={}, args=types.SimpleNamespace(n_eval=8, ppl_tokens=100),
                                smoke=True, cond_path=lambda n: str(cdir / f"{n}.json"),
                                load_cond=lambda n: json.load(open(cdir / f"{n}.json")) if (cdir / f"{n}.json").exists()
                                else None)
    rates, over = ablate.measured_rates(ctx)
    assert over == {"named": pytest.approx(2.24), "random": 0.0} and rates["hh3"] == pytest.approx(8000.0)
    pend = [ablate.Cond("mean__hh3__P_band", "mean", "hh3", "P_band", group="g"),
            ablate.Cond("mean__hh3__P_band__r000", "mean", "hh3", "P_band", r=0, group="g"),
            ablate.Cond("mean__hh3__P_band__r001", "mean", "hh3", "P_band", r=1, group="g")]
    row = ablate.eta(ctx, pend, rates, over)[0]
    assert row["n"] == 3 and row["seconds"] == pytest.approx(3 * 1.0 + 2.24)
    # the lazy model load is not part of a condition's wall_seconds
    w = world
    p = argparse.ArgumentParser()
    ablate.add_args(p)
    args = p.parse_args(["--model", BASE, "--root", w.root, "--device", "cpu", "--smoke", "--band", "1-3", "--bootstrap",
                         "30", "--no_progress", "--quiet", "--stage", "1", "--pilot", "--force"])

    def slow_loader():
        time.sleep(3.0)
        return w.model
    assert ablate.run_pipeline(args, slow_loader, w.tok, w.split, w.by_id, w.census, inject=w.inject) == 0
    st = json.load(open(w.abl(BASE, "conditions", "mean__hh3__C33.json")))["stats"]
    assert st["wall_seconds"] - st["seconds"] < 3.0


def test_frozen_mismatch_exits_5(world, tmp_path):
    w = World(tmp_path)
    write_table(w.root, BASE)
    assert w.run("--stage", "0") == 0
    path = w.abl(f"{BASE}_frozen_sets.json")
    fr = json.load(open(path))
    fr["sets"]["C33"]["heads"] = fr["sets"]["C33"]["heads"][:-1]
    json.dump(fr, open(path, "w"))
    assert w.run("--stage", "0") == ablate.EXIT_FROZEN
    assert w.run("--stage", "0", "--refreeze") == ablate.EXIT_FROZEN     # EVAL conditions exist already


def test_trajectory_sign_change_and_helpers(monkeypatch):
    def row(s, m, lo, hi):
        return {"metric": "M1", "quantity": "margin_mean", "arm": "mean", "set": s, "observed": m, "lo": lo, "hi": hi}
    base = [row("C33", -0.5, -0.7, -0.3), row("O", 0.1, -0.1, 0.3)]
    inst = [row("C33", 0.4, 0.2, 0.6), row("O_base", 0.2, 0.1, 0.3)]
    t = {r["set_base"]: r["sign"] for r in AR.trajectory(base, inst)}
    assert t == {"C33": "Vorzeichenwechsel", "O": "unentschieden"}
    # derangement: cluster-disjoint, seeded
    cl = np.array(["a", "a", "b", "c", "d", "d"])
    p1, p2 = AM.derangement(cl, 42), AM.derangement(cl, 42)
    assert (p1 == p2).all() and (cl[p1] != cl).all()
    # ppl windows equal eval_likelihood.PPLMetric
    monkeypatch.setattr(eval_likelihood, "wikitext", lambda tok, max_tokens: torch.arange(700))
    m = eval_likelihood.PPLMetric(None, max_tokens=700, window=256, stride=128)
    assert AM.ppl_windows(700, 256, 128) == m.windows
    # M1 quantities follow PairwiseMetric.item_values
    recs = [{"group": i, "pos": [[-3.0 - i, 3]], "neg": [[-4.0, 2 + i]]} for i in range(4)]
    ref, _, _ = eval_likelihood.PairwiseMetric.item_values(types.SimpleNamespace(name="hh_pairwise"), recs)
    items = {"pos_sum": [r["pos"][0][0] for r in recs], "pos_n": [r["pos"][0][1] for r in recs],
             "neg_sum": [r["neg"][0][0] for r in recs], "neg_n": [r["neg"][0][1] for r in recs]}
    got = AM.m1_values(items)
    for k in ("margin_sum", "margin_mean", "win_sum", "win_mean", "len_hi_gt_lo"):
        assert np.allclose(got[k], ref[k]), k
