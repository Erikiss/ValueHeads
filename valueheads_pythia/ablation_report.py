"""Summary of the causal ablation test: JSON / German markdown / CSV (docs/ABLATION_DESIGN.md, part D, 8.6-8.7).

Model-free: everything is read from the per-condition files results/ablation/{short}/conditions/{name}.json (+ the M2
sidecars {name}.m2.npz), selftest.json, floor.json, gate.json and the frozen-set file. The `specs` handed in by the
`ablate` command describe which condition is a named set, which clean run it is paired with and which random
conditions form its null (the plan of section 8.3); the report never re-derives that pairing from file names.

Rows of `_tests.csv` (one per arm x variant x metric quantity x named set): observed paired change (condition - clean of
the same variant) with a cluster-bootstrap CI (prompt clusters; ICL: (concept, seq); WikiText: window), the random-set
distribution of the same quantity (vs_random: one-sided p in the preregistered direction F29, two-sided p, z) and the
family: "primary" (M1 margin_mean under mean / hh3 for C33 and O, Holm per model), "secondary" (R > 0, p without
verdict) or "descriptive" (no null).
"""
import datetime as _dt
import json
import math
import os
import types

import numpy as np
import pandas as pd

from . import ablation_stats as S
from .ablation_metrics import HH_VARIANTS, dla_values, load_npz, m1_values
from .icl_battery import POSITIVE_CONTROL_TARGETS
from .icl_battery import effects as icl_effects

M1_QUANTITIES = ("margin_mean", "margin_sum", "acc_mean", "acc_sum", "threeway")
M1_SOURCE = {"margin_mean": "margin_mean", "margin_sum": "margin_sum", "acc_mean": "win_mean", "acc_sum": "win_sum",
             "threeway": "threeway"}
M2_QUANTITIES = ("head_auc_mean", "resid_auc", "mlp_auc")
M2_REFIT = ("refit_head_auc_mean", "refit_resid_auc")
M2_LABELS = {"head_auc_mean": "L1 fester Probe (Monitor), Heads", "resid_auc": "L1 fester Probe (Monitor), Residuum",
             "mlp_auc": "L3 nichtlinear (MLP)", "refit_head_auc_mean": "L2 neu gefittet (Informationserhalt), Heads",
             "refit_resid_auc": "L2 neu gefittet (Informationserhalt), Residuum"}
M3_QUANTITIES = ("dret", "dR", "dP")
TEST_COLUMNS = ["arm", "variant", "metric", "quantity", "set", "condition", "n_heads", "observed", "lo", "hi", "n",
                "n_clusters", "R", "rand_mean", "rand_std", "rand_q025", "rand_q975", "p_le", "p_ge", "p_one", "p_two",
                "z", "direction", "family", "p_holm", "significant", "interpretable", "verdict"]

# section 10 of docs/ABLATION_DESIGN.md (verbatim) + the deviations of part D
DESIGN_DEVIATIONS = [
    "Positive-control set. The paper's reader-20 (-0.74 nats) is unreleased. CR20 is census-ranked on concept12n (the "
    "paper's Llama protocol). The battery is concept12n seed 2, not concept12. The gate rule (F23) is this project's own, "
    "not the paper's.",
    "Response positions include b - 1 (the last prompt token) and ln - 1 (F2). The design says \"response-token positions, "
    "prompt untouched\": b - 1 is counted as the decision position of the first response token.",
    "mean_last on M1 is a causality sanity check only (F4).",
    "P_band random null. Own-profile draws unless the KV profile equals O's (F12). The design says \"when sizes differ\".",
    "P_DoM is selected per model (F13). The ranking by raw norm is scale dependent; a normalised variant is not computed.",
    "Three-way margin is computed only in hh3, i.e. Stage-1 sets and clean (F18); --threeway_all restores it everywhere.",
    "M2 refits run only for named sets and clean, with torch L-BFGS (same objective as sklearn). M2 CIs use B = 1000.",
    "ICL random null. Distinct heads, layer- and KV-matched. The paper drew per named head a same-layer head with "
    "replacement (run.py:118); our null is stricter.",
    "Fixed-identity arm requires the Base frozen file (F14); there is no recomputation in the instruct run.",
    "HH-harmless M4 is not implemented (optional in the design).",
    "(D) Smoke ICL battery: concept12n with n_seq = 2 (32 prompts, 2 per kind) instead of the 16-prompt smoke battery, "
    "whose per-kind mean equals each item's own query value (every knock would be an exact no-op).",
    "(D) Random conditions of a set whose null is reused (F12 P_band -> O; *_base sets following the Base file) are named "
    "after the set that owns the draws, so identical head sets are computed once.",
    "(D) M2 sidecars store the per-text scores in float32 (spec: fp16), so the paired AUC deltas equal the summaries; "
    "the head-AUC delta is paired per readout head (each condition on its own non-ablated heads).",
    "(D) Stage 1 re-runs clean__hh3 when its key changed (e.g. another batch plan); the floor follows the current "
    "clean__hh3, so a condition is always paired with a clean run of the identical batch plan.",
    "(D) N_EVAL selects a prefix of the EVAL walk order but fixes the hh3 partners and so every HH condition key: once HH "
    "conditions exist, another --n_eval is refused (exit 2) unless --change_n_eval; conditions whose items differ from "
    "their clean run are skipped and listed here instead of failing the report.",
    "(D) Hook self-test: the expected edit positions are rebuilt from lengths and boundaries (not from the edit mask), "
    "plus two checks: prompt_untouched (prompt [0, b - 2] and pads bitwise clean) and first_token (log p of the first "
    "response token changes for every sequence).",
    "(D) Gate: a full run proceeds only on a passed gate of a full Base run; an advisory smoke gate counts as STOPP there.",
]


# paper targets shown next to the gate (other battery and other head set than CR20 on concept12n seed 2: context only)
GATE_TARGETS = {"reader20_4b": "reader-20, gemeinsamer Knockout (frische Ziehung)",
                "cr20_llama": "census-gerankter reader-20 (Vorbild für CR20)",
                "cr20_sum_4b": "CR20, Summe der Einzelkopf-dret (Census-Ziehung, Seed 1)"}
GATE_TARGETS_INSTRUCT = {"instruct_removal": "Instruct-Transfer: 189 Base-Köpfe entfernt, Symbol- / seltene Labels"}


def gate_targets(is_base=True):
    """The POSITIVE_CONTROL_TARGETS entries of the gate section (+ the instruct transfer for an instruct report),
    each with a German "label_de"."""
    labels = dict(GATE_TARGETS, **({} if is_base else GATE_TARGETS_INSTRUCT))
    return [dict(t, label_de=labels[t["id"]]) for t in POSITIVE_CONTROL_TARGETS if t.get("id") in labels]


def _target_md(t):
    if t.get("values"):
        val = " / ".join(_f(v, 2, True) for v in t["values"].values())
    else:
        val = _f(t["value"], 3 if t["id"] == "cr20_sum_4b" else 2, True)
        if t.get("ci"):
            val += f" [{_f(t['ci'][0], 2, True)}, {_f(t['ci'][1], 2, True)}]"
    rnd = t.get("random")
    if rnd:
        val += f" vs. Zufall {_f(rnd['value'], 2, True)}" + (
            f" [{_f(rnd['ci'][0], 2, True)}, {_f(rnd['ci'][1], 2, True)}]" if rnd.get("ci") else "")
    if t.get("all_heads"):
        val += " (alle Köpfe entfernt: " + " / ".join(_f(v, 2, True) for v in t["all_heads"].values()) + ")"
    return f"  - {t['label_de']} ({t['model']}; {t['battery']}): {val} Nats – {t['source']}"


def parse_name(name):
    """{arm, variant, set, r, d} of a condition name `{arm}__{variant}[__{set}][__rNNN | __dNNN]`."""
    parts = name.split("__")
    out = {"arm": parts[0], "variant": parts[1] if len(parts) > 1 else None, "set": None, "r": None, "d": None}
    if len(parts) > 2:
        out["set"] = parts[2]
    if len(parts) > 3:
        tag = parts[3]
        out["r" if tag[0] == "r" else "d"] = int(tag[1:])
    return out


def load_conditions(cond_dir):
    """{name: payload} of every condition file (sidecars and temporary files skipped)."""
    out = {}
    if not os.path.isdir(cond_dir):
        return out
    for fn in sorted(os.listdir(cond_dir)):
        if fn.endswith(".json") and ".tmp" not in fn:
            with open(os.path.join(cond_dir, fn)) as f:
                out[fn[:-5]] = json.load(f)
    return out


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _p(p):
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _sci(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return "0" if x == 0 else f"{x:.1e}"


def _ci(e, d=3):
    return f"{_f(e.get('mean', e.get('observed')), d, True)} [{_f(e.get('lo'), d, True)}, {_f(e.get('hi'), d, True)}]"


# ----------------------------------------------------------------------------------------------------------------
# per-item quantities
# ----------------------------------------------------------------------------------------------------------------

def _hh_items(payload):
    return payload["items"]


def hh_deltas(cond, clean):
    """{quantity: per-item change} of M1 (cond - clean, same variant, aligned by item id)."""
    a, b = _hh_items(clean), _hh_items(cond)
    if list(a["id"]) != list(b["id"]):
        raise ValueError(f"{cond['name']}: item ids differ from {clean['name']}")
    va, vb = m1_values(a), m1_values(b)
    return {q: vb[src] - va[src] for q, src in M1_SOURCE.items() if src in va and src in vb}


def ppl_delta(cond, clean):
    """(per-window change of the window NLL, token weights)."""
    a, b = clean["items"], cond["items"]
    n = np.asarray(a["n"], float)
    nll_a = -np.asarray(a["sum"], float) / n
    nll_b = -np.asarray(b["sum"], float) / np.asarray(b["n"], float)
    return nll_b - nll_a, n


def _item_ids(payload):
    items = (payload or {}).get("items") or {}
    ids = items.get("id", items.get("index"))
    return None if ids is None else [int(i) for i in ids]


def clean_pairing(specs):
    """{condition name: name of the clean run it is paired with} from the specs (named sets and their randoms)."""
    out = {}
    for s in specs:
        if s.get("clean"):
            out.setdefault(s["name"], s["clean"])
            for r in s.get("randoms", []):
                out.setdefault(r, s["clean"])
    return out


def stale_conditions(conds, specs):
    """Conditions whose item ids differ from those of their clean run (e.g. computed with another --n_eval before the
    clean run was recomputed): [{"condition", "clean", "n", "n_clean", "n_eval", "n_eval_clean"}]. The report leaves
    them out (and lists them) instead of failing; the next run of their stage recomputes them (their key differs)."""
    pair = clean_pairing(specs)
    out = []
    for name, c in conds.items():
        p = parse_name(name)
        if p["arm"] == "clean":
            continue
        cname = pair.get(name) or f"clean__{p['variant']}"
        cl = conds.get(cname)
        a, b = _item_ids(cl), _item_ids(c)
        if cl is None or a is None or b is None or a == b:
            continue
        out.append({"condition": name, "clean": cname, "n": len(b), "n_clean": len(a),
                    "n_eval": (c.get("config") or {}).get("n_eval"),
                    "n_eval_clean": (cl.get("config") or {}).get("n_eval")})
    return out


def icl_meta_items(clean):
    m = clean.get("meta") or {}
    return [types.SimpleNamespace(index=i, kind=k, concept=c, labelset=ls, seq=s, cluster=cl)
            for i, (k, c, ls, s, cl) in enumerate(zip(m["kind"], m["concept"], m["labelset"], m["seq"], m["cluster"]))]


def icl_deltas(cond, clean, meta_items):
    """{dret | dR | dP: (values, clusters)} of a knock condition against the clean battery run."""
    return icl_effects(meta_items, clean["items"]["margin"], cond["items"]["margin"])


# ----------------------------------------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------------------------------------

class Report:
    """Computes the test rows and the sections of the summary from loaded condition files."""

    def __init__(self, conds, specs, clusters_of, *, B=10000, seed=0, info=None):
        self.conds = conds
        self.specs = specs
        self.clusters_of = clusters_of
        self.B = int(B)
        self.B2 = min(int(B), 1000)
        self.seed = int(seed)
        self.info = info or {}
        self._icl_meta = {}

    def _cl(self, payload):
        return self.clusters_of(payload["items"]["id"])

    def _meta(self, clean_name):
        if clean_name not in self._icl_meta:
            self._icl_meta[clean_name] = icl_meta_items(self.conds[clean_name])
        return self._icl_meta[clean_name]

    def _have(self, spec):
        return spec["name"] in self.conds and spec.get("clean") in self.conds

    def _randoms(self, spec):
        return [self.conds[n] for n in spec.get("randoms", []) if n in self.conds]

    def _row(self, spec, metric, quantity, boot, rand_vals, family=None):
        direction = S.direction(metric, spec["set"])
        v = S.vs_random(boot.get("mean"), rand_vals, direction)
        R = v["R"]
        fam = family or ("secondary" if R > 0 else "descriptive")
        cond = self.conds.get(spec["name"], {})
        return {"arm": spec["arm"], "variant": spec["variant"], "metric": metric, "quantity": quantity,
                "set": spec["set"], "condition": spec["name"], "n_heads": cond.get("n_heads"),
                "observed": boot.get("mean"), "lo": boot.get("lo"), "hi": boot.get("hi"), "n": boot.get("n"),
                "n_clusters": boot.get("n_clusters"), "R": R, "rand_mean": v["rand_mean"], "rand_std": v["rand_std"],
                "rand_min": v["rand_min"], "rand_max": v["rand_max"], "rand_q025": v["rand_q025"],
                "rand_q975": v["rand_q975"], "p_le": v["p_le"], "p_ge": v["p_ge"], "p_one": v["p_one"],
                "p_two": v["p_two"], "z": v["z"], "n_le": v["n_le"], "n_ge": v["n_ge"], "direction": direction,
                "family": fam, "p_holm": float("nan"), "verdict": ""}

    def hh_rows(self, spec):
        cond, clean = self.conds[spec["name"]], self.conds[spec["clean"]]
        d = hh_deltas(cond, clean)
        cl = self._cl(clean)
        rand = [hh_deltas(r, clean) for r in self._randoms(spec)]
        rows = []
        for q in M1_QUANTITIES:
            if q not in d:
                continue
            boot = S.cluster_bootstrap(d[q], cl, B=self.B, seed=self.seed)
            fam = "primary" if (spec.get("family") == "primary" and q == "margin_mean") else None
            rows.append(self._row(spec, "M1", q, boot, [float(np.nanmean(r[q])) for r in rand if q in r], fam))
        if spec["variant"] in ("hh2", "hh3") and (cond.get("summary") or {}).get("m2") and clean["summary"].get("m2"):
            rows += self.m2_rows(spec, cond, clean)
        return rows

    def m2_rows(self, spec, cond, clean):
        rows = []
        m2c, m2a = cond["summary"]["m2"], clean["summary"]["m2"]
        side = self.info.get("cond_dir")
        sc = load_npz(os.path.join(side, spec["name"] + ".m2.npz"))[0] if side and os.path.exists(
            os.path.join(side, spec["name"] + ".m2.npz")) else None
        sa = load_npz(os.path.join(side, spec["clean"] + ".m2.npz"))[0] if side and os.path.exists(
            os.path.join(side, spec["clean"] + ".m2.npz")) else None
        rand = self._randoms(spec)
        for q, key in (("head_auc_mean", "head"), ("resid_auc", "resid"), ("mlp_auc", "mlp")):
            obs = _m2_delta(m2c, m2a, q)
            boot = {"mean": obs, "lo": float("nan"), "hi": float("nan"), "n": None, "n_clusters": None}
            if sc is not None and sa is not None:
                labels = sc["labels"]
                ref = sa[key].copy()
                cnd = sc[key]
                if key == "head":                     # same heads on both sides (the ablated ones are excluded)
                    ref[~np.isfinite(cnd).all(axis=1)] = np.nan
                pad = S.paired_auc_delta(ref, cnd, labels, sc["clusters"], B=self.B2, seed=self.seed)
                boot = {"mean": pad["delta"], "lo": pad["lo"], "hi": pad["hi"], "n": pad["n"],
                        "n_clusters": pad["n_clusters"]}
            rv = [_m2_delta((r.get("summary") or {}).get("m2", {}), m2a, q) for r in rand]
            rows.append(self._row(spec, "M2", q, boot, [x for x in rv if math.isfinite(x)]))
        for q in M2_REFIT:
            if q in m2c and q in m2a:
                obs = _num(m2c.get(q)) - _num(m2a.get(q))
                rows.append(self._row(spec, "M2", q, {"mean": obs}, [], "descriptive"))
        return rows

    def ppl_rows(self, spec):
        cond, clean = self.conds[spec["name"]], self.conds[spec["clean"]]
        d, w = ppl_delta(cond, clean)
        boot = S.cluster_bootstrap(d, np.arange(len(d)), B=self.B, seed=self.seed, weights=w)
        rand = []
        for r in self._randoms(spec):
            dr, wr = ppl_delta(r, clean)
            rand.append(float(np.sum(dr * wr) / np.sum(wr)))
        return [self._row(spec, "M4", "nll", boot, rand)]

    def icl_rows(self, spec):
        cond, clean = self.conds[spec["name"]], self.conds[spec["clean"]]
        meta = self._meta(spec["clean"])
        eff = icl_deltas(cond, clean, meta)
        rand = [icl_deltas(r, clean, meta) for r in self._randoms(spec)]
        rows = []
        for q in M3_QUANTITIES:
            vals, cl = eff[q]
            boot = S.cluster_bootstrap(vals, cl, B=self.B, seed=self.seed)
            rows.append(self._row(spec, "M3", q, boot, [float(np.mean(r[q][0])) for r in rand]))
        return rows

    def dla_rows(self, spec):
        cond = self.conds[spec["name"]]
        de = dla_values(cond["items"])
        cl = self._cl(cond)
        rows = [self._row(spec, "M5", "DE", S.cluster_bootstrap(de, cl, B=self.B, seed=self.seed), [], "descriptive")]
        te_name, clean_name = spec.get("te"), spec.get("te_clean")
        pos = {int(i): k for k, i in enumerate(self.conds[te_name]["items"]["id"])} if te_name in self.conds else {}
        if te_name in self.conds and clean_name in self.conds and all(int(i) in pos for i in cond["items"]["id"]):
            te_full = hh_deltas(self.conds[te_name], self.conds[clean_name])["margin_mean"]
            te = np.asarray([te_full[pos[int(i)]] for i in cond["items"]["id"]])
            rows.append(self._row(spec, "M5", "TE", S.cluster_bootstrap(te, cl, B=self.B, seed=self.seed), [],
                                  "descriptive"))
            ratio = S.cluster_bootstrap_ratio(te, de, cl, B=self.B, seed=self.seed)
            sri = {"mean": 1 - ratio["ratio"] if ratio["defined"] else float("nan"),
                   "lo": 1 - ratio["hi"] if ratio["defined"] else float("nan"),
                   "hi": 1 - ratio["lo"] if ratio["defined"] else float("nan"), "n": ratio["n"],
                   "n_clusters": ratio["n_clusters"]}
            row = self._row(spec, "M5", "SRI", sri, [], "descriptive")
            row["verdict"] = "" if ratio["defined"] else "nicht definiert (DE-CI enthält 0)"
            rows.append(row)
        return rows

    def tests(self):
        rows = []
        for spec in self.specs:
            if spec.get("kind") != "named" or not self._have(spec):
                continue
            v = spec["variant"]
            if spec["arm"] == "dla":
                rows += self.dla_rows(spec)
            elif v in ("hh2", "hh3"):
                rows += self.hh_rows(spec)
            elif v == "ppl":
                rows += self.ppl_rows(spec)
            elif v == "icl":
                rows += self.icl_rows(spec)
        prim = [r for r in rows if r["family"] == "primary"]
        if prim:
            adj = S.primary_holm([{"set": r["set"], "p_one": r["p_one"], "observed": r["observed"]} for r in prim])
            for r, a in zip(prim, adj):
                r["p_holm"] = a["p_holm"]
                r["significant"] = bool(a["significant"])            # raw Holm flag; the verdict is set by build()
                r["interpretable"] = False
                r["verdict"] = "Abhängigkeit nachgewiesen" if a["significant"] else "keine Abhängigkeit nachgewiesen"
        return rows

    # -------------------------------------------------------------------------------------------- M7
    def m7(self):
        """Terciles of reuse_diff / len_diff (Δmargin_mean per named mean-arm set; random means for C33 and O) and the
        shuffled-prompt counterfactual (PD = margin_mean - margin_mean^shuf, ΔPD under ablation)."""
        clean = self.conds.get("clean__hh3")
        if not clean or "covariates" not in clean:
            return None
        cov = clean["covariates"]
        items = clean["items"]
        ids = [int(i) for i in items["id"]]
        cl = self.clusters_of(ids)
        n_pos, n_neg = np.asarray(items["pos_n"], float), np.asarray(items["neg_n"], float)
        covs = {"reuse_diff": np.asarray(cov["reuse_pos"], float) - np.asarray(cov["reuse_neg"], float),
                "len_diff": n_pos - n_neg}
        out = {"terciles": [], "shuffled": [], "edges": {}}
        mean_specs = [s for s in self.specs if s.get("kind") == "named" and s["arm"] == "mean"
                      and s["variant"] in ("hh2", "hh3") and self._have(s)]
        for cname, x in covs.items():
            labels, edges = S.tercile_labels(x)
            out["edges"][cname] = [float(e) for e in edges]
            for spec in mean_specs:
                cond, cln = self.conds[spec["name"]], self.conds[spec["clean"]]
                if [int(i) for i in cond["items"]["id"]] != ids:
                    continue
                d = hh_deltas(cond, cln)["margin_mean"]
                rand = [hh_deltas(r, cln)["margin_mean"] for r in self._randoms(spec)] if spec["set"] in ("C33", "O") \
                    else []
                for t in range(3):
                    m = labels == t
                    if not m.any():
                        continue
                    b = S.cluster_bootstrap(d[m], cl[m], B=self.B, seed=self.seed)
                    out["terciles"].append({"covariate": cname, "tercile": t, "set": spec["set"],
                                            "variant": spec["variant"], "mean": b["mean"], "lo": b["lo"], "hi": b["hi"],
                                            "n": b["n"], "rand_mean": float(np.mean([r[m].mean() for r in rand]))
                                            if rand else float("nan"), "R": len(rand)})
        shuf = self.conds.get("clean__hhshuf")
        if shuf and _item_ids(shuf) == ids:
            pd_clean = m1_values(items)["margin_mean"] - m1_values(shuf["items"])["margin_mean"]
            b = S.cluster_bootstrap(pd_clean, cl, B=self.B, seed=self.seed)
            out["shuffled"].append({"set": "clean", "quantity": "PD", **{k: b[k] for k in ("mean", "lo", "hi", "n")}})
            for s in ("C33", "O"):
                a, sh = self.conds.get(f"mean__hh3__{s}"), self.conds.get(f"mean__hhshuf__{s}")
                if not a or not sh or _item_ids(a) != ids or _item_ids(sh) != ids:
                    continue
                pd_s = m1_values(a["items"])["margin_mean"] - m1_values(sh["items"])["margin_mean"]
                b = S.cluster_bootstrap(pd_s - pd_clean, cl, B=self.B, seed=self.seed)
                out["shuffled"].append({"set": s, "quantity": "ΔPD", **{k: b[k] for k in ("mean", "lo", "hi", "n")}})
        return out

    # ---------------------------------------------------------------------------------------- checks
    def mean_last_sanity(self, dtype):
        """F4: |Δ| of the scored log-prob sums under mean_last (structurally 0 by causal attention)."""
        tol = 1e-6 if dtype in ("float32", None) else 1e-3
        out = []
        for spec in self.specs:
            if spec["arm"] != "mean_last" or not self._have(spec):
                continue
            for name in [spec["name"]] + list(spec.get("randoms", [])):
                c = self.conds.get(name)
                if not c:
                    continue
                cl = self.conds[spec["clean"]]["items"]
                m = 0.0
                for role in ("pos", "neg"):
                    m = max(m, float(np.max(np.abs(np.asarray(c["items"][f"{role}_sum"]) -
                                                   np.asarray(cl[f"{role}_sum"])))))
                out.append({"condition": name, "max_abs_delta_sum": m, "tol": tol, "ok": m <= tol})
        return out


def _num(x):
    return float("nan") if x is None else float(x)


def _m2_delta(m2c, m2a, q):
    """Δ of an M2 summary quantity; head_auc_mean is paired per readout head (mean over the heads read in the condition,
    i.e. without its ablated heads, of AUC_cond - AUC_clean), so every condition is compared on its own head subset."""
    if q == "head_auc_mean" and m2c.get("head_auc") is not None and m2a.get("head_auc") is not None:
        a = np.asarray([np.nan if v is None else v for v in m2a["head_auc"]], float)
        c = np.asarray([np.nan if v is None else v for v in m2c["head_auc"]], float)
        d = c - a
        return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")
    return _num(m2c.get(q)) - _num(m2a.get(q))


# ----------------------------------------------------------------------------------------------------------------
# trajectory
# ----------------------------------------------------------------------------------------------------------------

TRAJECTORY_SETS = ("C33", "C33_readers", "C33_brakes", "WholeBand")
FIXED_SETS = ("O", "CminusO", "P_band", "P_DoM", "TopK_all")


def _m1_lookup(rows, set_name):
    for r in rows:
        if r.get("metric") == "M1" and r.get("quantity") == "margin_mean" and r.get("arm") == "mean" \
                and r.get("set") == set_name:
            return r
    return None


def trajectory(rows_base, rows_inst):
    """Δmargin_mean (mean arm) per identity set on Base and on instruct, with sign_change."""
    out = []
    pairs = [(s, s) for s in TRAJECTORY_SETS] + [(s, f"{s}_base") for s in FIXED_SETS]
    for sb, si in pairs:
        a, b = _m1_lookup(rows_base, sb), _m1_lookup(rows_inst, si)
        if a is None or b is None:
            continue
        ea = {"mean": a["observed"], "lo": a["lo"], "hi": a["hi"]}
        eb = {"mean": b["observed"], "lo": b["lo"], "hi": b["hi"]}
        out.append({"set_base": sb, "set_instruct": si, "base": ea, "instruct": eb, "sign": S.sign_change(ea, eb)})
    return out


# ----------------------------------------------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------------------------------------------

def _clean_json(o):
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _clean_json(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def condition_rows(conds, specs, stale=()):
    """One row per condition for _conditions.csv (summary values, Δ vs clean, timing; `stale` names are flagged and get
    no Δ)."""
    clean_of = {}
    for s in specs:
        clean_of[s["name"]] = s.get("clean")
        for r in s.get("randoms", []):
            clean_of.setdefault(r, s.get("clean"))
    stale = set(stale)
    rows = []
    for name, c in conds.items():
        p = parse_name(name)
        summ = c.get("summary") or {}
        row = {"name": name, "arm": p["arm"], "variant": p["variant"], "set": p["set"], "r": p["r"], "d": p["d"],
               "n_heads": c.get("n_heads"), "n_eval": (c.get("config") or {}).get("n_eval"), "stale": name in stale}
        for k, v in summ.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[k] = v
        m2 = summ.get("m2") or {}
        for k in M2_QUANTITIES + M2_REFIT:
            if k in m2:
                row[f"m2_{k}"] = m2[k]
        cl = conds.get(clean_of.get(name) or "")
        if cl and p["arm"] != "clean" and name not in stale:
            try:
                if p["variant"] in HH_VARIANTS and p["arm"] != "dla":
                    row["delta_margin_mean"] = float(np.mean(hh_deltas(c, cl)["margin_mean"]))
                elif p["variant"] == "ppl":
                    d, w = ppl_delta(c, cl)
                    row["delta_nll"] = float(np.sum(d * w) / np.sum(w))
                elif p["variant"] == "icl":
                    row["delta_dret"] = float(np.mean(icl_deltas(c, cl, icl_meta_items(cl))["dret"][0]))
            except (KeyError, ValueError):
                pass
        st = c.get("stats") or {}
        row["seconds"] = st.get("seconds")
        row["tokens_per_second"] = st.get("tokens_per_second")
        rows.append(row)
    return rows


SMOKE_GATE_REASON = "Gate stammt aus einem Smoke-Lauf (beratend, R = 2) und gilt nicht für einen vollen Lauf"


def gate_status(gate, *, smoke=False, ignore_gate=False):
    """(status, effective gate) of a gate.json for a run with the given smoke / --ignore_gate flags.

    status: "passed"; "advisory" (smoke run on a smoke gate that did not pass); "ignored" (--ignore_gate on a missing or
    failed gate: computed, but no valid finding); "stop" (missing, failed, or a smoke gate read by a full run). The
    effective gate is what `ablation_stats.verdict_lines` gets (None or not passed / not advisory = STOPP)."""
    if gate is not None and gate.get("smoke") and not smoke:
        reason = (f"{gate.get('reason_de')}; " if gate.get("reason_de") else "") + SMOKE_GATE_REASON
        gate = dict(gate, passed=False, advisory=False, proceed=False, reason_de=reason)
    if gate is not None and gate.get("passed"):
        return "passed", gate
    if gate is not None and gate.get("advisory") and smoke:
        return "advisory", gate
    if ignore_gate:
        return "ignored", dict(gate or {}, passed=False, advisory=True,
                               reason_de=((gate or {}).get("reason_de") or "Gate fehlt") + "; Gate ignoriert (--ignore_gate)")
    if gate is not None:
        gate = dict(gate, advisory=False)
    return "stop", gate


def primary_verdict(significant, status, floor, smoke=False):
    """Befund of one primary row: STOPP (Gate) > Boden > Holm, with the reservations of an advisory / ignored gate."""
    if status == "stop":
        return "STOPP (Gate)"
    if floor:
        return "nicht interpretierbar (Boden)"
    out = "Abhängigkeit nachgewiesen" if significant else "keine Abhängigkeit nachgewiesen"
    notes = (["Gate beratend"] if status == "advisory" else []) + (["Gate ignoriert"] if status == "ignored" else []) \
        + (["Smoke, kein Befund"] if smoke else [])
    return out + (f" ({'; '.join(notes)})" if notes else "")


def build(conds, specs, clusters_of, *, info, gate, floor, selftest, frozen, B=10000, seed=0, other=None):
    """The summary payload (dict) of one model.

    Primary rows (tests and "primary") carry the raw Holm flag `significant`, `interpretable` (gate passed, no floor,
    no smoke) and the `verdict` that the markdown shows, so the CSV / JSON say the same as the report."""
    stale = stale_conditions(conds, specs)
    skip = {x["condition"] for x in stale}
    rep = Report({k: v for k, v in conds.items() if k not in skip}, specs, clusters_of, B=B, seed=seed, info=info)
    rows = rep.tests()
    status, gate_eff = gate_status(gate, smoke=bool(info.get("smoke")), ignore_gate=bool(info.get("ignore_gate")))
    is_floor = bool(floor and floor.get("floor"))
    prim = [r for r in rows if r["family"] == "primary"]
    for r in prim:
        r["interpretable"] = bool(status == "passed" and not is_floor and not info.get("smoke"))
        r["verdict"] = primary_verdict(r.get("significant", False), status, is_floor, bool(info.get("smoke")))
    primary = [{"set": r["set"], "observed": r["observed"], "lo": r["lo"], "hi": r["hi"], "p_one": r["p_one"],
                "p_two": r["p_two"], "z": r["z"], "p_holm": r["p_holm"], "significant": r.get("significant", False),
                "interpretable": r["interpretable"], "verdict": r["verdict"], "R": r["R"], "n_le": r["n_le"],
                "rand_mean": r["rand_mean"], "rand_std": r["rand_std"], "n_heads": r["n_heads"]} for r in prim]
    verdict = S.verdict_lines(primary, floor, gate_eff, model=info.get("model"), transferred=info.get("transferred"),
                              smoke=info.get("smoke"))
    traj = []
    if other and other.get("tests"):
        traj = trajectory(rows, other["tests"]) if info.get("is_base") else trajectory(other["tests"], rows)
    return {
        "version": 1, "created": _dt.datetime.now().isoformat(timespec="seconds"), "info": info,
        "verdict": verdict, "gate": gate, "gate_status": status, "gate_effective": gate_eff, "floor": floor,
        "selftest": selftest, "primary": primary, "tests": rows,
        "m7": rep.m7(), "mean_last_sanity": rep.mean_last_sanity(info.get("dtype")), "trajectory": traj,
        "other_model": (other or {}).get("info", {}).get("model"), "deviations": DESIGN_DEVIATIONS,
        "inputs": (frozen or {}).get("inputs", {}), "n_conditions": len(conds),
        "positive_control_targets": gate_targets(bool(info.get("is_base", True))), "stale": stale,
        "conditions": condition_rows(conds, specs, skip),
    }


def write_report(base, payload):
    """{base}.json / .md / _conditions.csv / _tests.csv; returns the paths."""
    paths = {"json": base + ".json", "md": base + ".md", "conditions": base + "_conditions.csv",
             "tests": base + "_tests.csv"}
    body = {k: v for k, v in payload.items() if k != "conditions"}
    _atomic_write(paths["json"], json.dumps(_clean_json(body), indent=1, ensure_ascii=False) + "\n")
    _atomic_write(paths["md"], markdown(payload))
    cdf = pd.DataFrame(payload["conditions"])
    _atomic_write(paths["conditions"], cdf.to_csv(index=False))
    tdf = pd.DataFrame(payload["tests"], columns=TEST_COLUMNS) if payload["tests"] else pd.DataFrame(columns=TEST_COLUMNS)
    _atomic_write(paths["tests"], tdf[TEST_COLUMNS].to_csv(index=False))
    return paths


# ----------------------------------------------------------------------------------------------------------------
# markdown (German)
# ----------------------------------------------------------------------------------------------------------------

def _tests_table(rows, cols):
    head = "| " + " | ".join(c for c, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return [head, sep] + ["| " + " | ".join(fn(r) for _, fn in cols) + " |" for r in rows]


def _rand(r):
    if not r.get("R"):
        return "–"
    return f"{_f(r['rand_mean'], 3, True)} ± {_f(r['rand_std'])} ({r['R']})"


def markdown(payload):
    info = payload["info"]
    gate = payload.get("gate")
    L = [f"# Kausaler Test: Ablation – `{info.get('model')}`", ""]
    L.append(f"- Modell `{info.get('hf_id')}` (Revision {info.get('revision')}, Commit {info.get('commit_sha') or '–'}), "
             f"{info.get('dtype')} / {info.get('attn_impl')}, Rolle **{'Base' if info.get('is_base') else 'Instruct'}**, "
             f"ICL-Identität `{info.get('icl_identity')}`")
    L.append(f"- EVAL {info.get('n_eval')} Paare, MEANS {info.get('n_means_ref')} + {info.get('n_means_probe')}, "
             f"Bootstrap B = {info.get('B')} (M2: {min(int(info.get('B') or 0), 1000)}), Seed {info.get('seed')}, "
             f"{payload.get('n_conditions')} Bedingungen; erzeugt {payload.get('created')}")
    if info.get("smoke"):
        L.append(f"- **{S.SMOKE_LINE}** C33 = synthetischer Smoke-Ersatz" if info.get("c33_synthetic") else
                 f"- **{S.SMOKE_LINE}**")
    if info.get("note"):
        L.append(f"- Hinweis: {info['note']}")
    if info.get("transferred"):
        L.append(f"- {S.TRANSFER_MARK}: C33-, Census- und Band-Sätze sind die des Base-Modells "
                 f"`{info.get('icl_identity')}`; O, CminusO, P_band, P_DoM, TopK_all stammen aus der eigenen Probe-Tabelle.")
    if info.get("ignore_gate"):
        L.append("- **Gate ignoriert** (`--ignore_gate`, nur zur Fehlersuche): kein gültiger Befund.")
    stale = payload.get("stale") or []
    if stale:
        names = ", ".join(f"`{x['condition']}`" for x in stale[:10]) + (f" … (+{len(stale) - 10})" if len(stale) > 10 else "")
        L.append(f"- **Achtung: {len(stale)} Bedingungen übersprungen** – ihre Items weichen vom sauberen Lauf ab "
                 f"(z. B. mit anderer N_EVAL gerechnet; n_eval {sorted({str(x['n_eval']) for x in stale})} gegen "
                 f"{sorted({str(x['n_eval_clean']) for x in stale})}): {names}. Der nächste Lauf ihrer Stufe rechnet sie "
                 "mit der aktuellen Konfiguration neu.")
    L += ["", "## Ergebnis in einem Satz", ""] + [f"- {x}" for x in payload["verdict"]] + [""]

    L += ["## Gate (Positivkontrolle)", ""]
    if gate is None:
        L.append("Gate-Ergebnis fehlt (Stufe 0 des Base-Modells).")
    else:
        rnd = gate.get("random") or {}
        L.append(f"- CR20-Knockout an der Query-Position (concept12n, Seed {gate.get('battery_seed', 2)}): "
                 f"Δdret {_f(gate.get('delta'), 3, True)} [{_f(gate.get('lo'), 3, True)}, {_f(gate.get('hi'), 3, True)}] "
                 f"Nats (n = {gate.get('n')}, {gate.get('n_clusters')} Cluster)")
        L.append(f"- Zufallsmengen (layer- und KV-gematcht, R = {rnd.get('R')}): Mittel {_f(rnd.get('rand_mean'), 3, True)} "
                 f"± {_f(rnd.get('rand_std'))}, Min {_f(rnd.get('rand_min'), 3, True)} / Max "
                 f"{_f(rnd.get('rand_max'), 3, True)}; p einseitig {_p(rnd.get('p_le'))}")
        targets = payload.get("positive_control_targets") or []
        if targets:
            L.append("- Einordnung, Zielwerte des Papers (**andere Batterie und anderer Kopf-Satz** als CR20 auf "
                     "concept12n Seed 2; nur Größenordnung, kein Gate-Kriterium):")
            L += [_target_md(t) for t in targets]
        cal = gate.get("calibration")
        if isinstance(cal, dict) and cal.get("diffs"):
            L.append("- Kalibrierung (Seed 1 gegen D188_summary): " + ", ".join(
                f"{k} {_f(v, 3, True)}" for k, v in cal["diffs"].items()) + f" (Toleranz {cal.get('tol')})")
        else:
            L.append("- Kalibrierung: übersprungen" + (" (Smoke)" if info.get("smoke") else ""))
        status = payload.get("gate_status")
        geff = payload.get("gate_effective") or gate
        verdict = {"passed": "PASS", "advisory": "beratend (Smoke)", "ignored": "nicht bestanden, ignoriert",
                   "stop": "STOPP"}.get(status) or ("PASS" if gate.get("passed") else
                                                   "beratend (Smoke)" if gate.get("advisory") else "STOPP")
        L.append(f"- **{verdict}**: {geff.get('reason_de', '')}")
        if not info.get("is_base"):
            L.append(f"- Gate des Base-Modells `{info.get('icl_identity')}` (die Positivkontrolle läuft nur dort).")
    L.append("")

    status = payload.get("gate_status")
    if status is None:
        status = gate_status(gate, smoke=bool(info.get("smoke")), ignore_gate=bool(info.get("ignore_gate")))[0]
    if status == "stop":
        geff = payload.get("gate_effective") or gate
        L += [S.stop_text((geff or {}).get("reason_de") or "Gate-Ergebnis fehlt"), ""]
    else:
        L += _md_body(payload)
    L += ["## Kausalitäts- und Hook-Checks", ""]
    for name, st in (payload.get("selftest") or {}).items():
        if not isinstance(st, dict) or "checks" not in st:
            continue
        L.append(f"- Selbsttest `{name}` ({st.get('dtype')}, {st.get('n_seqs')} Sequenzen): "
                 + ("ok" if st.get("ok") else "**FEHLER**") + " – " + ", ".join(
                     f"{k} {_sci(v['value'])}{'' if v['ok'] else ' (!)'}" for k, v in st["checks"].items()))
    san = payload.get("mean_last_sanity") or []
    if san:
        bad = [s for s in san if not s["ok"]]
        L.append(f"- mean_last auf M1 (F4, strukturell 0): max |Δ Summe| = {max(s['max_abs_delta_sum'] for s in san):.2e} "
                 f"über {len(san)} Bedingungen (Toleranz {san[0]['tol']:.0e})" + (
                     f"; **{len(bad)} markiert**" if bad else "; ok"))
    L += ["", "## Abweichungen vom Design", ""] + [f"{i}. {d}" for i, d in enumerate(payload["deviations"], 1)] + [""]
    L += ["## Eingaben", ""]
    for k, v in (payload.get("inputs") or {}).items():
        if isinstance(v, dict):
            L.append(f"- {k}: `{v.get('path') or v.get('file')}` sha256 `{v.get('sha256')}`"
                     + (" (synthetisch, nur Smoke)" if v.get("synthetic") else ""))
    for k in ("probes_sha", "refs_sha", "split_sha"):
        if info.get(k):
            L.append(f"- {k}: `{info[k]}`")
    return "\n".join(L) + "\n"


def _md_body(payload):
    info = payload["info"]
    rows = payload["tests"]
    floor = payload.get("floor")
    L = ["## Boden-Check", "", (floor or {}).get("text_de", "Boden-Check fehlt."), ""]
    L += ["## Primäre Familie (Holm, α = 0.05)", ""]
    prim = [r for r in rows if r["family"] == "primary"]
    if prim:
        L += _tests_table(prim, [
            ("Satz", lambda r: r["set"]), ("n Heads", lambda r: str(r.get("n_heads"))),
            ("Δ margin_mean [CI]", lambda r: _ci(r)), ("Zufall Mittel ± sd (R)", _rand),
            ("p einseitig", lambda r: _p(r["p_one"])), ("p zweiseitig", lambda r: _p(r["p_two"])),
            ("z", lambda r: _f(r["z"], 2, True)), ("Holm-p", lambda r: _p(r["p_holm"])),
            ("Befund", lambda r: r["verdict"])])
        sec = [r for r in rows if r["arm"] == "mean" and r["variant"] == "hh3" and r["metric"] == "M1"
               and r["set"] in ("C33", "O") and r["quantity"] != "margin_mean"]
        if sec:
            L += ["", "Dieselben Sätze, weitere M1-Größen (sekundär, p ohne Befund):", ""]
            L += _tests_table(sec, [("Satz", lambda r: r["set"]), ("Größe", lambda r: r["quantity"]),
                                    ("Δ [CI]", lambda r: _ci(r)), ("Zufall (R)", _rand),
                                    ("p einseitig", lambda r: _p(r["p_one"])), ("p zweiseitig", lambda r: _p(r["p_two"]))])
    else:
        L.append("Stufe 1 noch nicht gerechnet.")
    L += ["", "## Sekundär (p ohne Befund)", "", "### M1 je Arm × Satz (Δ margin_mean)", ""]
    m1 = [r for r in rows if r["metric"] == "M1" and r["quantity"] == "margin_mean" and r["family"] != "primary"]
    L += _tests_table(m1, [("Arm", lambda r: r["arm"]), ("Variante", lambda r: r["variant"]), ("Satz", lambda r: r["set"]),
                           ("n Heads", lambda r: str(r.get("n_heads"))), ("Δ [CI]", lambda r: _ci(r)),
                           ("Zufall (R)", _rand), ("p einseitig", lambda r: _p(r["p_one"])),
                           ("p zweiseitig", lambda r: _p(r["p_two"])), ("Familie", lambda r: r["family"])]) if m1 else \
        ["(keine)"]
    L += ["", "### M2 Monitore (L1 fester Probe, L2 neu gefittet, L3 nichtlinear) – nie Beleg für Nutzung", ""]
    m2 = [r for r in rows if r["metric"] == "M2"]
    L += _tests_table(m2, [("Arm", lambda r: r["arm"]), ("Satz", lambda r: r["set"]),
                           ("Ebene", lambda r: M2_LABELS.get(r["quantity"], r["quantity"])),
                           ("Δ AUC [CI]", lambda r: _ci(r)), ("Zufall (R)", _rand),
                           ("p einseitig", lambda r: _p(r["p_one"]))]) if m2 else ["(keine)"]
    L += ["", "### M3 Umkehrtest (ICL-Marge, Knockout an der Query)", ""]
    m3 = [r for r in rows if r["metric"] == "M3"]
    L += _tests_table(m3, [("Satz", lambda r: r["set"]), ("Größe", lambda r: r["quantity"]),
                           ("n Heads", lambda r: str(r.get("n_heads"))), ("Δ [CI]", lambda r: _ci(r)),
                           ("Zufall (R)", _rand), ("Richtung", lambda r: r["direction"]),
                           ("p einseitig", lambda r: _p(r["p_one"])), ("p zweiseitig", lambda r: _p(r["p_two"]))]) \
        if m3 else ["(keine)"]
    L += ["", "### M4 Schaden (WikiText-2, Δ NLL je Token, mean-Referenz an allen Positionen)", ""]
    m4 = [r for r in rows if r["metric"] == "M4"]
    L += _tests_table(m4, [("Satz", lambda r: r["set"]), ("n Heads", lambda r: str(r.get("n_heads"))),
                           ("Δ NLL [CI]", lambda r: _ci(r, 4)), ("Zufall (R)", _rand),
                           ("p einseitig (ge)", lambda r: _p(r["p_one"]))]) if m4 else ["(keine)"]
    L += ["", "### M5 Selbstreparatur (direkter Effekt per DLA vs. Gesamteffekt)", ""]
    m5 = [r for r in rows if r["metric"] == "M5"]
    L += _tests_table(m5, [("Satz", lambda r: r["set"]), ("Größe", lambda r: r["quantity"]),
                           ("Wert [CI]", lambda r: _ci(r)), ("Hinweis", lambda r: r.get("verdict") or "")]) if m5 else \
        ["(keine)"]
    L += ["", "SRI = 1 − mean(TE)/mean(DE); nur berichtet, wenn das DE-CI 0 ausschließt.", ""]
    L += ["### M7 Kovariaten (deskriptiv)", ""]
    m7 = payload.get("m7")
    if m7:
        if m7.get("terciles"):
            L += _tests_table(m7["terciles"], [
                ("Kovariate", lambda r: r["covariate"]), ("Terzil", lambda r: str(r["tercile"])),
                ("Satz", lambda r: r["set"]), ("Δ margin_mean [CI]", lambda r: _ci(r)), ("n", lambda r: str(r["n"])),
                ("Zufall-Mittel (R)", lambda r: f"{_f(r['rand_mean'], 3, True)} ({r['R']})" if r["R"] else "–")])
        if m7.get("shuffled"):
            L += ["", "Prompt-Tausch (PD = Marge mit eigenem Prompt − Marge nach fremdem Prompt):", ""]
            L += _tests_table(m7["shuffled"], [("Satz", lambda r: r["set"]), ("Größe", lambda r: r["quantity"]),
                                               ("Wert [CI]", lambda r: _ci(r)), ("n", lambda r: str(r["n"]))])
    else:
        L.append("(keine)")
    L += ["", "## Trajektorie Base → Instruct", ""]
    traj = payload.get("trajectory") or []
    if traj:
        L += _tests_table(traj, [("Satz Base", lambda r: r["set_base"]), ("Satz Instruct", lambda r: r["set_instruct"]),
                                 ("Δ Base [CI]", lambda r: _ci(r["base"])), ("Δ Instruct [CI]", lambda r: _ci(r["instruct"])),
                                 ("Vergleich", lambda r: r["sign"])])
    else:
        L.append(f"(Bericht des anderen Modells fehlt noch: `{payload.get('other_model') or info.get('other_model') or '–'}`)")
    L.append("")
    return L
