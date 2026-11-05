#!/usr/bin/env python3
"""Generates the two run notebooks of the causal ablation test (docs/ABLATION_DESIGN.md 8.9):

    notebooks/runs/qwen3-4b-base-ablation.ipynb    MODEL = "Qwen/Qwen3-4B-Base" (role base: positive-control gate)
    notebooks/runs/qwen3-4b-ablation.ipynb         MODEL = "Qwen/Qwen3-4B"      (role instruct: waits for the Base gate,
                                                                                  fixed-identity arm)

Two paths so that both can run in parallel on one Colab account (Colab binds one runtime per notebook path). Like
scripts/make_overlap_notebook.py, the runtime-check cell, the install / project / Drive cell and the `run()` CLI helper
are copied from the master notebook (notebooks/ValueHeadsPythia_Colab.ipynb; Drive root MyDrive/ValueHeadsPythia/qwen3-4b,
DRIVE_PROJECT pointed back at MyDrive/ValueHeadsPythia/valueheads_pythia), so fixes to the master reach these notebooks on
the next generation. Re-run after every change of the master notebook or of this script:

    python3 scripts/make_ablation_notebook.py                      # both notebooks into notebooks/runs/
    python3 scripts/make_ablation_notebook.py --out /tmp/nb_test   # elsewhere (CPU / nbconvert check)

CPU smoke of the Base notebook (tiny sizes; the model is overridden by environment variables, nothing else changes;
the census is read from a local checkout of the ICL repository):

    VH_NB_SMOKE=1 VH_NB_MODELS=Qwen/Qwen3-0.6B-Base VH_NB_PROJECT=$PWD VH_ROOT=/tmp/vh_ablation_smoke \\
      VH_ICL_REPO=/path/to/in-context-learning-circuit \\
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=5400 \\
      --output /tmp/vh_ablation_smoke/executed.ipynb notebooks/runs/qwen3-4b-base-ablation.ipynb
"""
import argparse
import json
import os
import re
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "notebooks", "ValueHeadsPythia_Colab.ipynb")
DRIVE_SUBDIR = "qwen3-4b"
NOTEBOOKS = {
    "qwen3-4b-base-ablation": {"model": "Qwen/Qwen3-4B-Base", "role": "base", "other": "qwen3-4b-ablation"},
    "qwen3-4b-ablation": {"model": "Qwen/Qwen3-4B", "role": "instruct", "other": "qwen3-4b-base-ablation"},
}
COLAB = "https://colab.research.google.com/github/Erikiss/ValueHeads/blob/main/notebooks/runs/{}.ipynb"


def src(cell):
    return "".join(cell["source"])


def sub1(text, pattern, repl, flags=re.M | re.S):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f"pattern not found in master notebook: {pattern!r}")
    return new


def master_cell(cells, prefix):
    """The master's code cell whose source starts with `prefix` (robust against inserted cells)."""
    hits = [c for c in cells if c["cell_type"] == "code" and src(c).startswith(prefix)]
    if len(hits) != 1:
        raise RuntimeError(f"master notebook: {len(hits)} code cells start with {prefix!r}")
    return src(hits[0])


def md(name, text):
    return {"cell_type": "markdown", "id": uuid.uuid5(uuid.NAMESPACE_URL, name + text[:80]).hex[:8], "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(name, text):
    return {"cell_type": "code", "id": uuid.uuid5(uuid.NAMESPACE_URL, name + text[:120]).hex[:8], "metadata": {},
            "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


BUDGET = """
| | Bedingungen | Tokens | A100-h bei 5k / 8k / 10k tok/s | CU (×6) bei 8k |
|---|---|---|---|---|
| Stufe 0 Base (inkl. Gate, 21 ICL-Bedingungen) | ~26 | ~2.1 M | 0.12 / 0.07 / 0.06 (+ Modell-Laden ~0.2) | ~2 |
| Stufe 1 (je Modell) | 203 | ~44.5 M | 2.5 / 1.5 / 1.2 | ~9 |
| Stufe 2 Base | ~640 | ~83 M | 4.6 / 2.9 / 2.3 | ~17 |
| Stufe 2 Instruct (inkl. feste Identität 105, CR20 21) | ~770 | ~100 M | 5.6 / 3.5 / 2.8 | ~21 |
| **Summe Base / Instruct** | | ~129 M / ~145 M | **7.2 / 4.5 / 3.6** bzw. **8.0 / 5.0 / 4.0** | **~27 / ~30** |
"""


def intro(name, nb):
    role = nb["role"]
    other = nb["other"]
    if role == "base":
        role_txt = ("**Rolle Base:** Hier läuft die Positivkontrolle (Gate). Das Instruct-Notebook "
                    f"[`{other}`]({COLAB.format(other)}) ist parallel ausführbar; es wartet in Stufe 0 auf das Gate "
                    "dieses Notebooks (`results/ablation/qwen3-4b-base/gate.json`, höchstens `GATE_WAIT_MIN` Minuten).")
    else:
        role_txt = ("**Rolle Instruct:** Dieselben Kopf-Identitäten (C33, Census- und Band-Sätze) wie im Base-Modell, "
                    "O / CminusO / P_band / P_DoM / TopK_all aus der eigenen Probe-Tabelle, dazu der Arm *feste "
                    "Identität* (die Base-Sätze auf dem Instruct-Modell). Parallel zum Base-Notebook "
                    f"[`{other}`]({COLAB.format(other)}) ausführbar: Stufe 0 wartet auf dessen Gate "
                    "(`results/ablation/qwen3-4b-base/gate.json`, höchstens `GATE_WAIT_MIN` Minuten); der Arm feste "
                    "Identität braucht `results/ablation/qwen3-4b-base_frozen_sets.json` und holt sich bei einem "
                    "späteren Lauf nach, was beim ersten fehlte.")
    return f"""
> **Run-Notebook `{name}`** (erzeugt von `scripts/make_ablation_notebook.py`, nicht von Hand bearbeiten): Modell
> `{nb['model']}`, Drive-Ordner `MyDrive/ValueHeadsPythia/qwen3-4b` (derselbe wie im Overlap-Notebook). Laufzeit:
> **eine A100** (Colab Pro), *Laufzeit → Alle ausführen*.

# Kausaler Test: Ablation der Wert-Head × ICL-Überlappung (`{nb['model']}`)

**Frage.** Die Top-15-%-HH-„helpful“-Probe-Heads überlappen den benannten 33-Head-ICL-Schaltkreis L20-27
(arXiv 2611.05344; Overlap-Notebook: 21 von 33 im Base-, 23 von 33 im Instruct-Modell, angereichert nach Holm). Ist das
kausal? **Stützt** sich das Modell für seine eigene Helpful-Präferenz auf diese Heads (*Reliance*), oder ist „helpful“
dort nur **dekodierbar** (*Expression*; Goodfire, „Shaping & Monitoring Models through their Internal Representations“)?

{role_txt}

**Ablauf** (`python -m valueheads_pythia.cli ablate`, Spezifikation `docs/ABLATION_DESIGN.md`):
1 Laufzeit · 2 Installation/Projekt/Drive · 3 Konfiguration · 4 **Stufe 0 – Gate**: Eingaben (Probe-Tabelle aus dem
Overlap-Lauf, Census, Batterie), HH-Split (Prompt-Cluster-disjunkt zu Probe-Train/-Val; means_ref 256, means_probe 256,
EVAL 1000), Hook-Selbsttest, Referenzmittel, eingefrorene Kopf-Sätze und Zufallsziehungen (vor der ersten EVAL-Rechnung),
M2-Probes, Boden-Check, Positivkontrolle (CR20-Knockout auf der ICL-Batterie gegen 20 Zufallsmengen) · 5 Pilot + ETA ·
6 **Stufe 1 – primär**: mean-Ablation von C33 und O an den Antwortpositionen, je 100 layer- und KV-gematchte
Zufallsmengen, Holm über {{C33, O}} · 7 Stufe 2 – sekundär (M3 Umkehrtest, weitere Sätze, M4 WikiText-Schaden,
mean_last, M5 Selbstreparatur, M7 Kovariaten, resample, dir_mean, gauss{', feste Identität' if role == 'instruct' else ''})
· 8 Zusammenfassung.

**Budget** (Schätzung aus Scouting-Tokenzahlen, Rate nicht gemessen; der Pilot in Zelle 5 druckt die echte ETA):
{BUDGET}
Reichen die CU nicht: Stufe 0 + Stufe 1 (≈ 2 h, ≈ 12 CU) beantworten die Primärfrage; Stufe 2 folgt der
Prioritätsreihenfolge (Arme in Zelle 3 abschaltbar). **Jede Zelle ist idempotent:** jede Bedingung ist eine eigene Datei
auf Drive; nach einem Verbindungsabbruch einfach *Alle ausführen*, fertige Bedingungen werden übersprungen.

**Datenschutz:** HH-Inhalte werden nie ausgegeben (nur IDs, Hashes, Zahlen); vor dem Speichern trotzdem *Bearbeiten →
Alle Ausgaben löschen*.
"""


INSTALL_MD = """
## 2. Installation, Projekt, Drive

Drive mounten, Projekt holen (Default `git clone` von GitHub), gepinnte Abhängigkeiten installieren. `VH_ROOT` zeigt auf
`MyDrive/ValueHeadsPythia/qwen3-4b`: dort liegen die Probe-Tabellen des Overlap-Laufs (`results/heads/`) und die
Ergebnisse dieses Tests (`results/ablation/`). `PROJECT_SOURCE = "drive"` erwartet das Projekt unter
`MyDrive/ValueHeadsPythia/valueheads_pythia` (wie im Master-Notebook).
"""

CONFIG_MD = """
## 3. Konfiguration

Budget-Knöpfe (Defaults = Design): `N_EVAL` EVAL-Paare, `R_C33` / `R_O` Zufallsmengen der Primärfamilie, `R_SMALL` für
CminusO / P_band, `R_SEC` für die sekundären Arme, `BOOTSTRAP` Cluster-Bootstrap-Ziehungen, `ARMS` = Arme der Stufe 2
(an/aus). Kleinere R sind Präfixe der eingefrorenen Ziehungen (nichts Gerechnetes geht verloren); größere R brauchen
`--extend_random` (hängt Ziehungen an, die vorhandenen bleiben). **`N_EVAL` ist kein Präfix:** es legt Items,
Partner-Antworten (hh3) und damit den Schlüssel jeder HH-Bedingung fest. Vor Stufe 0 wählen; eine spätere Änderung
rechnet jede HH-Bedingung neu und wird mit Exit 2 abgelehnt, solange `CHANGE_N_EVAL = False` ist (nach dem Pilot kostet
sie nur `clean__hh3` und die zwei Pilot-Bedingungen). `BATCH_TOKENS` / `MAX_BATCH` gehen ebenfalls in die Schlüssel der
Bedingungen ein: zwischen den Stufen nicht ändern. Überschreibbar per Umgebung (CPU-Smoke): `VH_NB_MODELS` (erster Eintrag ersetzt
`MODEL`), `VH_NB_ARMS` (Komma-Liste), `VH_NB_SMOKE`. Die Zelle prüft, ob die Probe-Tabelle des Overlap-Laufs auf Drive
liegt, und lädt einmal den Census (sha256 gegen den gepinnten Commit).
"""

CONFIG = r'''
# 3. Konfiguration: Modell, Budget-Knoepfe, Arme, Gate-Wartezeit, CLI-Argumente
from valueheads_pythia import icl_sets

MODEL = "__MODEL__"
if os.environ.get("VH_NB_MODELS"):                         # e.g. CPU smoke: Qwen/Qwen3-0.6B-Base
    MODEL = [m.strip() for m in os.environ["VH_NB_MODELS"].split(",") if m.strip()][0]
SHORT = config.model_short_name(MODEL)
N_EVAL = 1000                      # EVAL pairs; enters every HH condition key: choose before stage 0
CHANGE_N_EVAL = False              # True once to accept a changed N_EVAL (every HH condition is recomputed)
R_C33 = 100                        # random sets of the primary family (C33)
R_O = 100                          # random sets of the primary family (O)
R_SMALL = 50                       # CminusO, P_band (own null)
R_SEC = 20                         # secondary arms and the gate
BOOTSTRAP = 10000                  # cluster-bootstrap resamples (M2: min(B, 1000))
ARMS = {"m3": True, "mean_sets": True, "m4": True, "mean_last": True, "m5": True, "m7": True, "resample": True,
        "dir_mean": True, "gauss": True, "fixed_identity": __FIXED__}
if os.environ.get("VH_NB_ARMS"):
    _on = {a.strip() for a in os.environ["VH_NB_ARMS"].split(",") if a.strip()}
    ARMS = {a: a in _on for a in ARMS}
BATCH_TOKENS = 8192 if HAS_CUDA else 2048     # enters the condition keys: keep it fixed between the stages
MAX_BATCH = 64 if HAS_CUDA else 16
GATE_WAIT_MIN = 90                 # instruct: minutes to wait for the Base gate in stage 0
PROGRESS = [] if (IN_COLAB and not SMOKE) else ["--no_progress"]
NOTE = "SMOKE auf winzigen Teilmengen (CPU): nur Funktionspruefung, kein Befund" if SMOKE else None

COMMON = ["--model", MODEL, "--device", DEVICE, "--batch_tokens", BATCH_TOKENS, "--max_batch", MAX_BATCH,
          "--gate_wait_min", GATE_WAIT_MIN, *PROGRESS]
if SMOKE:
    COMMON += ["--smoke", "--note", NOTE]        # F28 sizes (8 EVAL, 8 + 8 MEANS, R = 2, B = 200), advisory gate
else:
    COMMON += ["--n_eval", N_EVAL, "--r_c33", R_C33, "--r_o", R_O, "--r_small", R_SMALL, "--r_sec", R_SEC,
               "--bootstrap", BOOTSTRAP]
if CHANGE_N_EVAL:
    COMMON += ["--change_n_eval"]
ABL = os.path.join(P["results"], "ablation")
RUN_DIR = os.path.join(ABL, SHORT)
ICL_ID = icl_sets.resolve_icl_model(SHORT)[0]
GATE_JSON = os.path.join(ABL, ICL_ID, "gate.json")

info = config.MODELS.get(SHORT, {})
print(f"MODEL {MODEL} -> {SHORT}: L/H/KV/D = {info.get('layers')}/{info.get('heads')}/{info.get('kv_heads')}/"
      f"{info.get('head_dim')}, numerics {config.dtype_policy(MODEL, DEVICE)} on {DEVICE}; ICL-Identitaet {ICL_ID}"
      + (" (Rolle Base: Gate hier)" if ICL_ID == SHORT else " (Rolle Instruct: wartet auf das Base-Gate)"))
table = config.heads_path(P, SHORT, None, "helpful", "pythia")
print(f"Probe-Tabelle helpful/pythia: {table} -> {'vorhanden' if os.path.exists(table) else 'FEHLT'}"
      + ("" if os.path.exists(table) else (" (SMOKE: synthetische Tabelle)" if SMOKE else
                                           " -> zuerst das Overlap-Notebook (qwen3-4b-overlap) laufen lassen")))
census = icl_sets.census_sets(ICL_ID)         # env VH_ICL_REPO, cache, or the pinned GitHub URL; sha256-checked
print(f"ICL census {ICL_ID}: {census.origin} (pinned {census.pinned}), " + ", ".join(f"{k} {len(v)}" for k, v in census.roles.items()))
print(f"N_EVAL {N_EVAL}, R = {R_C33}/{R_O}/{R_SMALL}/{R_SEC}, B = {BOOTSTRAP}, batch {BATCH_TOKENS}/{MAX_BATCH}"
      + (" | SMOKE: Groessen von --smoke (F28)" if SMOKE else ""))
print("Arme Stufe 2:", ", ".join(a for a, on in ARMS.items() if on))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, d=3, sign=True):
    return "–" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")
'''

STAGE0_MD = """
## 4. Stufe 0 – Gate

Prüft die Eingaben (Probe-Tabelle: Modus pythia, AUC, `n_train = 20 000`; Census; ICL-Batterie-Datei), schreibt bzw.
verifiziert den HH-Split, lädt das Modell (bf16 + sdpa), Hook-Selbsttest (Referenz = eigene saubere Werte reproduziert die
Logits; leerer Satz = kein Hook; Lokalität; Antwortpositionen [b − 1, Ende] unabhängig aus Länge und Grenze nachgerechnet,
Prompt und Padding unberührt, erstes Antwort-Token reagiert; Kausalität), Referenzmittel auf means_ref, M2-Trainingsmerkmale auf
means_probe, **friert die Kopf-Sätze und Zufallsziehungen ein** (`results/ablation/{SHORT}_frozen_sets.json`, vor der
ersten EVAL-Rechnung), trainiert die M2-Probes, rechnet `clean__hh3` und den Boden-Check und – im Base-Modell – die
Positivkontrolle: Kalibrierung (Census-Ziehung, Seed 1, gegen D188_summary), CR20-Knockout an der Query-Position der
frischen Batterie (Seed 2) gegen 20 layer- und KV-gematchte Zufallsmengen. **STOPP**, wenn das Gate nicht besteht
(Exit 3): dann wird kein Helpfulness-Befund berichtet, der STOPP-Bericht (`results/ablation/{SHORT}_ablation.md`) liegt
trotzdem auf Drive, und die Zellen danach brechen ab. Ein volles Instruct-Notebook akzeptiert nur ein bestandenes Gate
eines vollen Base-Laufs, kein beratendes Smoke-Gate.
"""

STAGE0 = r'''
# 4. Stufe 0: Selbsttest, Split, Referenzen, eingefrorene Saetze, M2-Probes, Boden-Check, Gate (idempotent)
rc0 = run("ablate", *COMMON, "--stage", 0, check=False)
sel = read_json(os.path.join(RUN_DIR, "selftest.json")) or {}
for name in ("C33", "WholeBand"):
    st = sel.get(name)
    if st:
        print(f"Hook-Selbsttest {name}: {'ok' if st['ok'] else 'FEHLER'} (" + ", ".join(
            f"{k} {v['value']:.1e}" for k, v in st["checks"].items()) + ")")
fl = read_json(os.path.join(RUN_DIR, "floor.json"))
if fl:
    print("Boden-Check:", fl["text_de"])
gate = read_json(GATE_JSON)
if gate:
    rnd = gate.get("random") or {}
    print(f"Gate ({'PASS' if gate['passed'] else ('beratend (Smoke)' if gate.get('advisory') else 'STOPP')}): CR20 "
          f"Δdret {fmt(gate['delta'])} [{fmt(gate['lo'])}, {fmt(gate['hi'])}] Nats; Zufall Mittel "
          f"{fmt(rnd.get('rand_mean'))}, Min {fmt(rnd.get('rand_min'))} (R = {rnd.get('R')})")
    print("  ", gate.get("reason_de"))
if rc0 == 3 and os.path.exists(os.path.join(ABL, f"{SHORT}_ablation.md")):
    print("STOPP-Bericht:", os.path.join(ABL, f"{SHORT}_ablation.md"))
if rc0 != 0:
    raise RuntimeError(f"Stufe 0 beendet mit Exit {rc0} (2 Eingaben fehlen oder N_EVAL geaendert, 3 Gate nicht "
                       "bestanden / fehlt / nur Smoke, 4 Selbsttest, 5 eingefrorene Saetze weichen ab) – siehe Ausgabe oben")
'''

PILOT_MD = """
## 5. Pilot + ETA

Zwei Bedingungen (`mean__hh3__C33` und ihre erste Zufallsmenge) messen die Rate (Tokens/s); daraus die ETA und die
CU-Schätzung (~6 CU je A100-Stunde) für alle ausstehenden Bedingungen der Stufen 1 und 2. Zu langsam? `R_*` oder `ARMS`
in Zelle 3 verkleinern (kleinere R sind Präfixe der eingefrorenen Ziehungen; nichts Gerechnetes geht verloren). Eine
kleinere `N_EVAL` ist **kein** Präfix: sie rechnet jede HH-Bedingung neu (jetzt nur `clean__hh3` und die zwei
Pilot-Bedingungen) und braucht dafür einmal `CHANGE_N_EVAL = True`; sonst bricht der Lauf mit Exit 2 ab.
"""

PILOT = r'''
# 5. Pilot: zwei Bedingungen, gemessene Rate -> ETA / CU je Stufe und Arm
run("ablate", *COMMON, "--stage", 1, "--pilot", "--arms", *[a for a, on in ARMS.items() if on])
pilot = read_json(os.path.join(RUN_DIR, "pilot.json"))
if pilot:
    print(f"\nGemessen ({pilot['device']}, {pilot.get('gpu') or 'CPU'}): " + ", ".join(
        f"{k} {v:.0f} tok/s" for k, v in sorted(pilot["rates"].items())))
    tot = pilot["total"]
    print(f"Ausstehend: {tot['n']} Bedingungen, {tot['tokens'] / 1e6:.1f} M Tokens, ~{tot['hours']:.2f} h, "
          f"~{tot['cu_a100']:.0f} CU (A100, {pilot['cu_per_a100_hour']} CU/h)")
'''

STAGE1_MD = """
## 6. Stufe 1 – primär

mean-Ablation (feste label-neutrale Referenz je Head, an allen Antwortpositionen) von **C33** und **O = C33 ∩ Top-k**
in Variante hh3 (gewählt, abgelehnt, fremde Antwort), je 100 layer- und KV-gematchte Zufallsmengen. Primäre Familie
je Modell: Δ margin_mean (gepaart je Item) für C33 und O, einseitiges empirisches p in Richtung „senkt“, Holm (α = 0.05).
"""

STAGE1 = r'''
# 6. Stufe 1: primaere Zellen (C33, O + je R Zufallsmengen), dann Statistik + Bericht
run("ablate", *COMMON, "--stage", 1)
rep = read_json(os.path.join(ABL, f"{SHORT}_ablation.json"))
if rep:
    print("\nPrimaere Familie (Holm, alpha = 0.05):")
    for r in rep["primary"]:
        print(f"  {r['set']:4s} ({r['n_heads']} Heads): Δ margin_mean {fmt(r['observed'])} [{fmt(r['lo'])}, {fmt(r['hi'])}]"
              f" | Zufall {fmt(r['rand_mean'])} ± {fmt(r['rand_std'], sign=False)} (R = {r['R']}) | p einseitig "
              f"{r['p_one'] if r['p_one'] is None else round(r['p_one'], 4)} | Holm-p "
              f"{r['p_holm'] if r['p_holm'] is None else round(r['p_holm'], 4)} | {r.get('verdict', '')}")
    for line in rep["verdict"]:
        print("  " + line)
'''

STAGE2_MD = """
## 7. Stufe 2 – sekundär

Arme in Prioritätsreihenfolge: **m3** (ICL-Umkehrtest: C33, O, P_band, TopK_all, WholeBand, Census-Sätze),
**mean_sets** (CminusO, P_band, P_DoM, C33-Leser/-Bremsen, TopK_all, WholeBand), **m4** (WikiText-2-Schaden, 50 000
Tokens), **mean_last** (Referenz am letzten Token; auf M1 strukturell 0, nur M2), **m5** (Selbstreparatur: direkter Effekt
per DLA gegen Gesamteffekt), **m7** (Prompt-Wiederverwendung, Längen-Terzile, Prompt-Tausch), **resample** (Spender-Paar),
**dir_mean** (nur die DoM-Richtung je Head, gegen zufällige Richtungen), **gauss** (normgleiches Rauschen) und im
Instruct-Lauf **fixed_identity** (Base-Sätze auf dem Instruct-Modell). Alles sekundär (p ohne Befund) oder deskriptiv.
"""

STAGE2 = r'''
# 7. Stufe 2: sekundaere Arme (idempotent; abgeschaltete Arme in ARMS bleiben aus)
arms = [a for a, on in ARMS.items() if on]
if arms:
    run("ablate", *COMMON, "--stage", 2, "--arms", *arms)
else:
    print("keine Arme aktiviert")
'''

SUMMARY_MD = """
## 8. Zusammenfassung

Bericht aus den gespeicherten Bedingungen (ohne Modell): `results/ablation/{SHORT}_ablation.{json,md}`,
`…_conditions.csv`, `…_tests.csv`. Angezeigt wird der Kopf bis zu den sekundären Tabellen (vollständig in der Datei).
**Vor dem Speichern: Bearbeiten → Alle Ausgaben löschen.**
"""

SUMMARY = r'''
# 8. Zusammenfassung: Bericht neu schreiben und anzeigen (Kopf bis "Sekundaer")
from IPython.display import Markdown, display

run("ablate", *COMMON, "--summary")
md_path = os.path.join(ABL, f"{SHORT}_ablation.md")
with open(md_path) as f:
    text = f.read()
cut = text.find("\n## Sekundär")
display(Markdown(text[:cut] if cut > 0 else text))
print("Ergebnisse:", ABL)
for suffix in ("_ablation.json", "_ablation.md", "_ablation_conditions.csv", "_ablation_tests.csv", "_frozen_sets.json"):
    path = os.path.join(ABL, SHORT + suffix)
    print(f"  {path} ({os.path.getsize(path) / 1e3:.0f} kB)" if os.path.exists(path) else f"  {path} (fehlt)")
n_cond = len([f for f in os.listdir(os.path.join(RUN_DIR, "conditions")) if f.endswith(".json")]) \
    if os.path.isdir(os.path.join(RUN_DIR, "conditions")) else 0
print(f"  {n_cond} Bedingungsdateien unter {os.path.join(RUN_DIR, 'conditions')}")
print("\n*** Vor dem Speichern des Notebooks: Bearbeiten -> Alle Ausgaben loeschen (Edit -> Clear all outputs). ***")
'''

FOOTER = """
---
**Erinnerung:** *Bearbeiten → Alle Ausgaben löschen* (Edit → Clear all outputs), bevor das Notebook gespeichert, auf
Drive abgelegt oder geteilt wird.
"""


def build(name):
    nbcfg = NOTEBOOKS[name]
    master = json.load(open(MASTER))
    cells = master["cells"]
    runtime = master_cell(cells, "# 1. Laufzeit-Check")
    install = master_cell(cells, "# 2. Installation")
    install = sub1(install, r'^DRIVE_ROOT = f"\{DRIVE_MOUNT\}/MyDrive/ValueHeadsPythia"',
                   f'DRIVE_ROOT = f"{{DRIVE_MOUNT}}/MyDrive/ValueHeadsPythia/{DRIVE_SUBDIR}"')
    # the project stays where the master notebook keeps it (PROJECT_SOURCE="drive"), not under the run's Drive root
    install = sub1(install, r'^DRIVE_PROJECT = f"\{DRIVE_ROOT\}/valueheads_pythia"',
                   'DRIVE_PROJECT = f"{DRIVE_MOUNT}/MyDrive/ValueHeadsPythia/valueheads_pythia"')
    cfg = master_cell(cells, "# 3. Konfiguration")
    m = re.search(r"^QUIET_PATTERNS = .*?(?=^run\(\"--help\"\))", cfg, flags=re.M | re.S)
    if not m:
        raise RuntimeError("run() helper not found in the master's configuration cell")
    runner = m.group(0).rstrip() + "\n\n\nrun(\"--help\")\n"
    runtime_md = next(src(c) for c in cells if c["cell_type"] == "markdown" and src(c).startswith("## 1. Laufzeit-Check"))
    config_cell = (CONFIG.replace("__MODEL__", nbcfg["model"])
                   .replace("__FIXED__", "True" if nbcfg["role"] == "instruct" else "False"))
    nb = {
        "nbformat": master["nbformat"], "nbformat_minor": master["nbformat_minor"],
        "metadata": {**master["metadata"], "colab": {**master["metadata"].get("colab", {}), "gpuType": "A100"},
                     "accelerator": "GPU"},
        "cells": [md(name, intro(name, nbcfg)), md(name, runtime_md), code(name, runtime), md(name, INSTALL_MD),
                  code(name, install), md(name, CONFIG_MD), code(name, config_cell.strip("\n") + "\n\n" + runner),
                  md(name, STAGE0_MD), code(name, STAGE0), md(name, PILOT_MD), code(name, PILOT), md(name, STAGE1_MD),
                  code(name, STAGE1), md(name, STAGE2_MD), code(name, STAGE2), md(name, SUMMARY_MD),
                  code(name, SUMMARY), md(name, FOOTER)],
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(src(cell), name, "exec")
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "notebooks", "runs"))
    args = ap.parse_args()
    import nbformat
    os.makedirs(args.out, exist_ok=True)
    for name in NOTEBOOKS:
        nb = build(name)
        path = os.path.join(args.out, f"{name}.ipynb")
        with open(path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        nbformat.validate(nbformat.read(path, as_version=4))
        print(f"{path}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
