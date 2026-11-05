#!/usr/bin/env python3
"""Generates notebooks/runs/meanfield-angles.ipynb: mean-field signatures of the value-direction angles (docs/MEANFIELD.md).

CPU only (no accelerator in the metadata): the notebook reads the direction npz files and head tables that the run
notebooks left on Drive, so it is run once per Google account (account A: pythia-160m / -410m / -1b, account B:
pythia-1.4b / -2.8b). The Drive / project cell follows the master notebook's install cell (notebooks/ValueHeadsPythia_Colab.ipynb,
as reused by scripts/make_overlap_notebook.py: same DRIVE_MOUNT / PROJECT_SOURCE / GITHUB_URL / find_project logic,
`git pull --ff-only` on re-runs), but is written out here because the master cell installs the full GPU stack and
creates the result folders of a probing run in the Drive root, which this read-mostly notebook must not do. Requirements
are only installed when an import is missing. Re-run after every change of this script:

    python3 scripts/make_meanfield_notebook.py                        # notebooks/runs/meanfield-angles.ipynb
    python3 scripts/make_meanfield_notebook.py --out /tmp/nb_test     # elsewhere

CPU smoke of the same file on a synthetic Drive tree with known structure (created by meanfield.make_synthetic_tree in
VH_NB_BASE, never on Drive; < 2 minutes):

    VH_NB_SMOKE=1 VH_NB_PROJECT=$PWD VH_NB_BASE=/tmp/mf_nb_smoke \\
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \\
      --output /tmp/mf_nb_smoke_executed.ipynb notebooks/runs/meanfield-angles.ipynb
"""
import argparse
import json
import os
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "meanfield-angles"


def src(cell):
    return "".join(cell["source"])


def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid5(uuid.NAMESPACE_URL, NAME + text[:80]).hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": uuid.uuid5(uuid.NAMESPACE_URL, NAME + text[:120]).hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
> **Run-Notebook `meanfield-angles`** (erzeugt von `scripts/make_meanfield_notebook.py`, nicht von Hand bearbeiten).
> **Eine CPU-Laufzeit genügt** – keine GPU nötig (*Laufzeit → Laufzeittyp ändern → CPU*). Das Notebook liest nur die
> Richtungen (`results/directions/*.npz`) und Head-Tabellen (`results/heads/*.json`), die die Run-Notebooks auf Drive
> abgelegt haben. **Einmal mit Konto A** (pythia-160m, pythia-410m, pythia-1b) **und einmal mit Konto B**
> (pythia-1.4b, pythia-2.8b) ausführen: jeweils *Laufzeit → Alle ausführen*.

# Mean-Field-Signaturen der Winkel zwischen Werterichtungen

Das Wert-Head-Paper (arXiv 2611.02503, App. K, Tab. 17) berichtet Winkel zwischen den Richtungsvektoren zweier Werte auf
den geteilten Top-15 %-Heads, z. B. für Qwen3-4B harmless × honesty 88.6° ± 9.7, helpful × honesty 79.0° ± 10.0,
harmless × helpful 69.1° ± 12.6. **These:** Verwandte Werte sind „nur eine 90-Grad-Drehung“; interessant ist das Residuum,
das nicht passt. Mittelt es sich über die Heads heraus, liegen die Werte im „Mean-Field-Regime“, sonst in einem anderen.

**Vorsicht mit 90°:** Zufällige Einheitsvektoren in R^d stehen fast senkrecht (d = 64: 90° ± 7.2°, d = 128: 90° ± 5.1°).
Belastbar ist nur der **Überschuss** über die isotrope Null **N1** (exakte Verteilung von cos θ) und über das untrainierte
Modell **N2** (step0, dieselben Heads).

Je Modell, Checkpoint, Wertepaar und Head-Satz (geteilte Top-15 %-Heads wie Tab. 17 und alle Heads) wird r(h) = cos θ(h)
untersucht: **(a)** Lage gegen 0 und gegen step0, **(b)** Überschuss-Streuung und Anteil > 100°, **(c)**
Mittelungsgesetz (sd des Mittels über n benachbarte Heads ~ n^(−1/2) bei unabhängigen Residuen, flacher bei Korrelation
entlang der Layer-Reihenfolge: Nachbar-Heads, Layer, Tiefentrend; ein Anteil, den **alle** Heads teilen, ist innerhalb
eines Modells nur eine Lageverschiebung, die (c) nicht sieht und allein (a) erfasst), **(d)** Häufung nach Layer (ICC,
Permutationstest), **(e)** Persistenz über das Training (Spearman), **(f)** Form (Kurtosis, Bimodalität gegen eine
schiefe, aber eingipflige Alternative), **(g)** Vorhersagbarkeit des Rests: Lassen sich r und |r| aus Eigenschaften der
Heads (relative Tiefe, Normen der beiden Massenmittel-Differenzen, Probe-Scores beider Werte, r bei step0) **außerhalb
der Stichprobe** vorhersagen (Ridge-Regression, 5-fache CV mit ganzen Layern als Testblöcken, R² der zurückgehaltenen
Heads, Permutationsnull über ganze Layer)?

**Warum (g):** Das 1/√n-Schrumpfen gemittelter Residuen unterscheidet reines Rauschen **nicht** von einem
strukturierten, systematischen Rest – beide schrumpfen gleich, solange die Struktur nicht an die Layer-Reihenfolge
gebunden ist. Erst die Vorhersagbarkeit trennt sie: In der Demo (24 × 16 Heads, d = 128) haben unabhängiges Rauschen und
ein Rest, der vom Probe-Score des Heads abhängt, dieselbe Mittelungssteigung (−0.50, KI enthält −1/2), aber ein
Held-out-R² von ≈ 0 gegenüber ≈ +0.8.

Urteil je Zeile: *mean-field-artig*, *systematisch*, *kollektiv*, *vorhersagbar*, *bistabil* (mehrere möglich), mit den
tragenden Tests; *mean-field-artig* nur, wenn der Layer-Test (d) durchführbar war, das Steigungs-KI höchstens 0.5 breit
ist und (g) berechenbar und nicht signifikant ist, sonst *unterbestimmt* (Kürzel m; typisch für die kleinen geteilten
Top-15 %-Mengen); ein vorhersagbarer Rest (Kürzel V) ist nie *mean-field-artig*. **Vorab festgelegt
(konfirmatorisch)** sind nur (a) + (c) auf den geteilten Top-15 %-Heads bei step143000; alles andere, auch (g), ist
deskriptiv (Holm je Modell × Checkpoint). *Mean-field-artig* ist eine statistische Signatur, **keine** Aussage über
Mean-Field- vs. Lazy/NTK-Training. Details: `docs/MEANFIELD.md`.

**Ablauf:** 1 Einstellungen · 2 Drive, Projekt, Abhängigkeiten · 3 Daten finden · 4 Analyse · 5 Zusammenfassung je Modell
· 6 Kombinierte Tabelle und Breiten-Achse. **Dauer:** ~1-3 min je Modell (CPU). Ein zweiter Lauf ohne neue Daten
überspringt alles (Fingerabdruck der Eingaben). Es werden keine Datensatz-Texte gelesen oder ausgegeben.
"""

SETTINGS_MD = """
## 1. Einstellungen

Bootstrap-Ziehungen und Permutationen (je 2000), Seed, Modellauswahl (`None` = alle gefundenen), `FORCE = True` rechnet
auch unveränderte Modelle neu. Überschreibbar per Umgebung (CPU-Smoke): `VH_NB_SMOKE`, `VH_NB_MODELS`, `VH_NB_BASE`,
`VH_NB_PROJECT`.
"""

SETTINGS = r'''
# 1. Einstellungen (CPU genuegt)
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SMOKE = os.environ.get("VH_NB_SMOKE", "0") == "1"   # synthetic tree with known structure instead of Drive data
N_BOOT = 2000          # bootstrap draws over heads (CIs)
N_PERM = 2000          # permutations (averaging law, layer clustering, persistence)
SEED = 0
MODELS = None          # None = every model found, e.g. ["pythia-1b"]
if os.environ.get("VH_NB_MODELS"):
    MODELS = [m.strip() for m in os.environ["VH_NB_MODELS"].split(",") if m.strip()]
FORCE = False          # True: recompute although inputs and code are unchanged
FIGURES = True
print(f"Colab: {IN_COLAB} | SMOKE: {SMOKE} | Bootstrap {N_BOOT}, Permutationen {N_PERM}, Seed {SEED} | Modelle: {MODELS or 'alle gefundenen'}")
print(f"CPUs {os.cpu_count()} | python {sys.version.split()[0]}")
'''

INSTALL_MD = """
## 2. Drive, Projekt, Abhängigkeiten

Drive mounten, Projekt holen (Default `git clone` von GitHub wie in den anderen Notebooks; ein vorhandener Klon wird mit
`git pull` aktualisiert), fehlende Pakete aus `requirements.txt` nachinstallieren (in einer Colab-CPU-Laufzeit meist nichts
nötig). Gesucht wird unter `MyDrive/ValueHeadsPythia` und dessen Unterordnern; es wird dort **nichts** angelegt außer
den Ausgaben der Analyse. Im Smoke-Modus entsteht stattdessen ein synthetischer Baum in `VH_NB_BASE` (nie auf Drive).
"""

INSTALL = r'''
# 2. Drive mounten, Projekt (git clone | Drive-Kopie), fehlende Abhaengigkeiten
DRIVE_MOUNT = "/content/drive"
DRIVE_BASE = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia"           # searched (with subfolders) for results/directions/
PROJECT_SOURCE = "git"        # "git": clone GITHUB_URL (default) | "drive": project folder on Drive
GITHUB_URL = "https://github.com/Erikiss/ValueHeads.git"  # used with PROJECT_SOURCE == "git"
DRIVE_PROJECT = f"{DRIVE_BASE}/valueheads_pythia"                # README.md, valueheads_pythia/, scripts/, ... on Drive
LOCAL_PROJECT = "/content/valueheads_pythia"                     # code runs from the local disk


def find_project(start):
    """Project root (the directory holding valueheads_pythia/cli.py) at `start` or one of its parents, else None."""
    d = os.path.abspath(start)
    for _ in range(4):
        if os.path.exists(os.path.join(d, "valueheads_pythia", "cli.py")):
            return d
        d = os.path.dirname(d)
    return None


if IN_COLAB:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT)
    if find_project(LOCAL_PROJECT) and PROJECT_SOURCE == "git" and os.path.isdir(os.path.join(LOCAL_PROJECT, ".git")):
        subprocess.run(["git", "-C", LOCAL_PROJECT, "pull", "--ff-only"], check=False)   # pick up fixes without a runtime reset
    if not find_project(LOCAL_PROJECT):
        if PROJECT_SOURCE == "git":
            subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL, LOCAL_PROJECT], check=True)
        else:
            src = find_project(DRIVE_PROJECT) or find_project(DRIVE_BASE)
            if not src:
                raise FileNotFoundError(f"Projekt nicht gefunden: Ordner nach {DRIVE_PROJECT} kopieren (Drive) oder "
                                        f"PROJECT_SOURCE = 'git' und GITHUB_URL setzen")
            shutil.copytree(src, LOCAL_PROJECT,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "vh_runs", "features", "results", "figures",
                                                          "hf_cache", "*.npy", "*.npz"))
    PROJECT = LOCAL_PROJECT
    if SMOKE:                                           # never write the synthetic tree to Drive
        DRIVE_BASE = os.environ.get("VH_NB_BASE") or "/content/meanfield_smoke"
else:
    PROJECT = os.environ.get("VH_NB_PROJECT") or find_project(os.getcwd()) or find_project(os.path.join(os.getcwd(), ".."))
    if not PROJECT:
        raise FileNotFoundError("Projekt nicht gefunden: VH_NB_PROJECT setzen oder das Notebook aus dem Projekt starten")
    DRIVE_BASE = os.path.abspath(os.environ.get("VH_NB_BASE") or os.path.join(PROJECT, "vh_runs", "meanfield_smoke"))

# the analysis imports the project's loaders (directions / probing), which pull in numpy, scipy, scikit-learn, torch, ...
NEEDED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "transformers", "datasets", "joblib", "tqdm"]
missing = [m for m in NEEDED if importlib.util.find_spec(m) is None]
if missing:
    print("fehlende Pakete:", ", ".join(missing), "-> pip install -r requirements.txt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(PROJECT, "requirements.txt")], check=True)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
from valueheads_pythia import config
from valueheads_pythia.analysis import meanfield as mf

if SMOKE:
    os.makedirs(DRIVE_BASE, exist_ok=True)
    if not mf.discover(DRIVE_BASE):
        mf.make_synthetic_tree(DRIVE_BASE)              # pythia-160m geometry in the base, pythia-1b in a subfolder
        print(f"SMOKE: synthetischer Baum in {DRIVE_BASE}")
if not os.path.isdir(DRIVE_BASE):
    raise FileNotFoundError(f"{DRIVE_BASE} fehlt: ist das richtige Google-Konto verbunden?")
print(f"PROJECT    {PROJECT}\nDRIVE_BASE {DRIVE_BASE}\nModul      {mf.__file__} ({mf.VERSION})")
'''

DISCOVER_MD = """
## 3. Daten finden

Gesucht wird `results/directions/{modell}_{checkpoint}_{wert}.npz` (Werte honesty, harmless, helpful; HelpSteer-Werte
bleiben in v1 außen vor) unter `MyDrive/ValueHeadsPythia` und zwei Ebenen darunter; die Wurzel von pythia-160m kann
`MyDrive/ValueHeadsPythia` selbst sein. Dazu je Datei die Head-Tabelle `results/heads/{modell}_{checkpoint}_{wert}_pythia.json`
(ohne sie gibt es nur den Head-Satz „alle Heads“). Geprüft wird die Geometrie gegen die Modell-Registry
(160m 12×12×64, 410m 24×16×64, 1b 16×8×256, 1.4b 24×16×128, 2.8b 32×32×80); fehlende Checkpoints der Standardliste
werden angezeigt.
"""

DISCOVER = r'''
# 3. Modelle, Checkpoints, Werte und Head-Tabellen finden
import pandas as pd

pd.set_option("display.width", 250)
FOUND = mf.discover(DRIVE_BASE, MODELS)
if not FOUND:
    raise SystemExit(f"keine Richtungen unter {DRIVE_BASE} gefunden (results/directions/*_{{honesty,harmless,helpful}}.npz)")
table = mf.discovery_table(FOUND)
table["root"] = [os.path.relpath(r, os.path.dirname(DRIVE_BASE)) for r in table["root"]]
print(table.to_string(index=False))
for model, entry in FOUND.items():
    revs = list(entry["revisions"])
    missing_revs = [r for r in config.CHECKPOINTS_DEFAULT if r not in revs]
    incomplete = {r: sorted(set(mf.VALUES) - set(v)) for r, v in entry["revisions"].items() if set(mf.VALUES) - set(v)}
    no_heads = sorted({r for r, v in entry["revisions"].items() for f in v.values() if not f["heads"]})
    print(f"\n{model}: {len(revs)} Checkpoints" + (f"; fehlend: {', '.join(missing_revs)}" if missing_revs else " (alle 12)")
          + (f"; unvollstaendige Werte: {incomplete}" if incomplete else "")
          + (f"; ohne Head-Tabelle: {', '.join(no_heads)}" if no_heads else ""))
    for note in entry["notes"]:
        print("  Hinweis:", note)
'''

ANALYSIS_MD = """
## 4. Analyse

Je Modell eine Zeile je Checkpoint × Wertepaar × Head-Satz mit allen Signaturen, Holm-korrigierten p-Werten und dem
Urteil; Ausgaben im Wurzelordner des Modells: `results/meanfield/{modell}_meanfield.{csv,md,json}` und
`figures/meanfield/{modell}_*.png`. Bei mehreren Modellen zusätzlich `meanfield_summary_{modelle}.csv` in
`MyDrive/ValueHeadsPythia`. Unveränderte Modelle (gleiche Eingaben, Einstellungen und Code) werden übersprungen.
"""

ANALYSIS = r'''
# 4. Analyse (idempotent)
t0 = time.time()
RESULT = mf.run_all(DRIVE_BASE, MODELS, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED, figures=FIGURES, force=FORCE, found=FOUND)
print(f"\nfertig nach {time.time() - t0:.0f}s")
for model, r in RESULT["models"].items():
    print(f"{model}: {'uebersprungen (unveraendert)' if r['skipped'] else 'neu berechnet'} -> {r['paths']['csv']}")
'''

SUMMARY_MD = """
## 5. Zusammenfassung je Modell

Die deutsche Zusammenfassung (`{modell}_meanfield.md`): Primärbefund, Vergleich mit Tab. 17 (n, Mittel, Median, sd der
Winkel), Vorhersagbarkeit des Rests (g: R² von r und |r| mit Holm-p und stärkster Kovariate), Verlauf der Urteile über die
Checkpoints (step1000 / step2000 = IOI-Tief nach Dahiya & Blondin 2026), Treiber der Urteile; darunter die
Verlaufsabbildungen je Wertepaar.
"""

SUMMARY = r'''
# 5. Zusammenfassung und Verlaufsabbildungen je Modell
from IPython.display import Image, Markdown, display

for model, r in RESULT["models"].items():
    if not os.path.exists(r["paths"]["md"]):
        continue
    with open(r["paths"]["md"]) as f:
        display(Markdown(f.read()))
    for png in sorted(glob.glob(os.path.join(r["paths"]["figures"], f"{model}_*_trajektorie.png"))):
        display(Image(filename=png, width=720))
'''

COMBINED_MD = """
## 6. Kombinierte Tabelle und Breiten-Achse

Alle Modelle dieses Kontos in einer Tabelle (Schlüsselspalten bei step143000) und die Breiten-Achse (Überschuss-Streuung
sd(r)/sd_N1 gegen die Modellgröße). **Alle fünf Größen in einer Abbildung:** die Datei `meanfield_summary_*.csv` des
anderen Kontos nach `MyDrive/ValueHeadsPythia/` dieses Kontos kopieren und diese Zelle erneut ausführen – sie liest alle
`meanfield_summary_*.csv` dort ein. **Vor dem Speichern des Notebooks: Bearbeiten → Alle Ausgaben löschen.**
"""

COMBINED = r'''
# 6. Kombinierte Tabelle (alle meanfield_summary_*.csv in DRIVE_BASE + die Modelle dieses Laufs) und Breiten-Achse
paths = sorted(glob.glob(os.path.join(DRIVE_BASE, "meanfield_summary_*.csv")))
paths += [r["paths"]["csv"] for r in RESULT["models"].values() if os.path.exists(r["paths"]["csv"])]
ALL = mf.merge_summaries(paths)
models_all = sorted(ALL["model"].unique(), key=mf.nominal_params)
print(f"{len(ALL)} Zeilen, Modelle: {', '.join(models_all)}")
if RESULT["combined"]:
    print("kombinierte CSV dieses Laufs:", RESULT["combined"])
cols = ["model", "pair", "headset", "status", "n_heads", "angle_mean_deg", "angle_median_deg", "angle_std_deg", "mean_r",
        "sd_ratio", "slope", "icc", "pred_r_r2", "pred_absr_r2", "verdict"]
fin = ALL[ALL["revision"] == "step143000"].assign(params=lambda t: t["model"].map(mf.nominal_params))
fin = fin.sort_values(["pair", "headset", "params"])
# summaries of the other account may predate (g): show only the columns they have
print(fin[[c for c in cols if c in fin]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
if len(models_all) >= 2:
    width = mf.plot_width(ALL, os.path.join(DRIVE_BASE, f"meanfield_breite_{'_'.join(models_all)}.png"))
    if width:
        display(Image(filename=width, width=760))
print("\n*** Vor dem Speichern des Notebooks: Bearbeiten -> Alle Ausgaben loeschen (Edit -> Clear all outputs). ***")
'''

FOOTER = """
---
**Erinnerung:** *Bearbeiten → Alle Ausgaben löschen* (Edit → Clear all outputs), bevor das Notebook gespeichert, auf
Drive abgelegt oder geteilt wird.
"""


def build():
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        # CPU only: no "accelerator" / "gpuType" (the run notebooks request a GPU, this one must not)
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [md(INTRO), md(SETTINGS_MD), code(SETTINGS), md(INSTALL_MD), code(INSTALL), md(DISCOVER_MD), code(DISCOVER),
                  md(ANALYSIS_MD), code(ANALYSIS), md(SUMMARY_MD), code(SUMMARY), md(COMBINED_MD), code(COMBINED), md(FOOTER)],
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(src(cell), NAME, "exec")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "runs"))
    args = ap.parse_args()
    import nbformat
    os.makedirs(args.out, exist_ok=True)
    nb = build()
    path = os.path.join(args.out, f"{NAME}.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    nbformat.validate(nbformat.read(path, as_version=4))
    print(f"{path}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
