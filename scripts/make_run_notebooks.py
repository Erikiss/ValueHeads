#!/usr/bin/env python3
"""Generates the one-click run notebooks under notebooks/runs/ from the master notebook (README "Parallel auf A100").

Each run notebook is the master with a few constants changed (model, Drive root per model, checkpoint subset for the
two pythia-2.8b sessions, feature-cache cap) and, for the 2.8b sweep part, without the single-checkpoint sections 6-10
and 12 (the causal part does them for step143000). Re-run after every change of the master notebook:

    python3 scripts/make_run_notebooks.py                      # notebooks/runs/*.ipynb
    python3 scripts/make_run_notebooks.py --test-model pythia-160m --out /tmp/runs_test   # CPU/nbconvert check
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from valueheads_pythia import config  # noqa: E402

MASTER = os.path.join(ROOT, "notebooks", "ValueHeadsPythia_Colab.ipynb")
CAUSAL = list(config.CAUSAL_CHECKPOINTS_DEFAULT)
OTHERS = [r for r in config.CHECKPOINTS_DEFAULT if r not in CAUSAL]
LITE_VALUES = ["honesty", "harmless", "helpful"]
RUNS = {
    "pythia-160m": dict(model="pythia-160m", root="pythia-160m"),
    "pythia-410m": dict(model="pythia-410m", root="pythia-410m"),
    "pythia-1b": dict(model="pythia-1b", root="pythia-1b"),
    "pythia-1.4b": dict(model="pythia-1.4b", root="pythia-1.4b"),
    "pythia-2.8b-causal": dict(model="pythia-2.8b", root="pythia-2.8b", revisions=CAUSAL),
    "pythia-2.8b-sweep": dict(model="pythia-2.8b", root="pythia-2.8b", revisions=OTHERS, no_causal=True, sweep_only=True),
    # "lite" variants (2026-11-04): probing only, three values, no likelihood evaluation / generation / judge. They write into
    # the same Drive folders, so finished results are kept and skipped; started in place of the full notebook to save CU.
    "pythia-1b-lite": dict(model="pythia-1b", root="pythia-1b", values=LITE_VALUES, no_causal=True, drop=("9", "10")),
    "pythia-1.4b-lite": dict(model="pythia-1.4b", root="pythia-1.4b", values=LITE_VALUES, no_causal=True, drop=("9", "10")),
    "pythia-2.8b-sweep-lite": dict(model="pythia-2.8b", root="pythia-2.8b", revisions=OTHERS, values=LITE_VALUES, no_causal=True,
                                   sweep_only=True),
    "pythia-2.8b-causal-lite": dict(model="pythia-2.8b", root="pythia-2.8b", revisions=CAUSAL, values=LITE_VALUES, no_causal=True,
                                    drop=("9", "10")),
}
DROP_SECTIONS = ("6", "7", "8", "9", "10", "12")       # markdown headers "## N." dropped (with their code cell) in sweep-only runs


def sub1(text, pattern, repl, flags=re.M):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"pattern not found in master notebook: {pattern!r}")
    return new


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build(name, spec, model_override=None):
    nb = json.load(open(MASTER))
    cells = nb["cells"]
    model = model_override or spec["model"]
    # cell 0: banner
    banner = (f"> **Run-Notebook `{name}`** (erzeugt von `scripts/make_run_notebooks.py`, nicht von Hand bearbeiten): "
              f"MODEL `{model}`, Drive-Ordner `MyDrive/ValueHeadsPythia/{spec['root']}`"
              + (f", Checkpoints `{' '.join(spec['revisions'])}`" if spec.get("revisions") else "")
              + (", nur Sweep (Zellen 1-5, 11; Evals und Einzel-Checkpoint-Zellen im Causal-Teil)" if spec.get("sweep_only") else "")
              + (f", Werte `{' '.join(spec['values'])}`" if spec.get("values") else "")
              + (", ohne Likelihood-Evaluation und Generation/Judge (Abschnitte 9 und 10 entfernt)" if spec.get("drop") else "")
              + ". Laufzeit: A100 (Colab Pro+), alle Zellen ausführen.\n\n")
    set_src(cells[0], banner + src(cells[0]))
    # cell 2 (idx 4): Drive root per model
    set_src(cells[4], sub1(src(cells[4]), r'^DRIVE_ROOT = f"\{DRIVE_MOUNT\}/MyDrive/ValueHeadsPythia"',
                           f'DRIVE_ROOT = f"{{DRIVE_MOUNT}}/MyDrive/ValueHeadsPythia/{spec["root"]}"'))
    # cell 3 (idx 6): model, checkpoints, causal / single-checkpoint settings
    c = sub1(src(cells[6]), r'^MODEL = "pythia-160m"', f'MODEL = "{model}"')
    if spec.get("values"):
        c = sub1(c, r"^VALUES = \[.*\]$", "VALUES = " + json.dumps(spec["values"]) + "   # lite run: values with a usable signal")
    if spec.get("revisions"):
        c = sub1(c, r"^REVISIONS = list\(config\.CHECKPOINTS_DEFAULT\).*$",
                 "REVISIONS = " + json.dumps(spec["revisions"]) + "   # run subset of config.CHECKPOINTS_DEFAULT")
    if spec.get("no_causal"):
        c = sub1(c, r"^CAUSAL_REVISIONS = \[r for r in CAUSAL_REVISIONS if r in REVISIONS\] or REVISIONS\[-1:\]",
                 "CAUSAL_REVISIONS = []                 # " + ("lite run: no likelihood evaluation" if spec.get("values")
                                                                 else "sweep part: the likelihood evals run in the causal part"))
    if spec.get("sweep_only"):
        c = sub1(c, r"^NB_REVISIONS = \[FINAL_REVISION\]", "NB_REVISIONS = []                     # sweep part: no single-checkpoint cells")
        c = sub1(c, r"^NB_CAUSAL_REVISIONS = .*$", "NB_CAUSAL_REVISIONS = []")
    set_src(cells[6], c)
    # cell 5 (idx 10): smaller Drive feature cache (six roots share the Drive quota)
    set_src(cells[10], sub1(src(cells[10]), r"^CACHE_CAP_GB = 4\.0", "CACHE_CAP_GB = 1.0"))
    drop = DROP_SECTIONS if spec.get("sweep_only") else tuple(spec.get("drop", ()))
    if drop:
        kept, skip_next = [], False
        for cell in cells:
            if skip_next:
                skip_next = False
                continue
            first = src(cell).lstrip().splitlines()[0] if src(cell).strip() else ""
            m = re.match(r"^## (\d+)\.", first)
            if cell["cell_type"] == "markdown" and m and m.group(1) in drop:
                skip_next = True
                continue
            kept.append(cell)
        nb["cells"] = kept
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(src(cell), name, "exec")
            cell["outputs"], cell["execution_count"] = [], None
    # Colab proposes this runtime on connect (the master keeps T4 as the entry-level default)
    nb.setdefault("metadata", {}).setdefault("colab", {})["gpuType"] = spec.get("gpu", "A100")
    nb["metadata"]["accelerator"] = "GPU"
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "runs"))
    ap.add_argument("--test-model", default=None, help="override MODEL in every run notebook (CPU / nbconvert check)")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    import nbformat
    os.makedirs(args.out, exist_ok=True)
    for name, spec in RUNS.items():
        if args.only and name not in args.only:
            continue
        nb = build(name, spec, args.test_model)
        path = os.path.join(args.out, f"{name}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        nbformat.validate(nbformat.read(path, as_version=4))
        print(f"{path}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
