"""ICL-circuit head sets for Qwen3 (Gao & Gao, "A Circuit for In-Context Learning", arXiv 2611.05344), with provenance.

Purpose: test whether value heads (per-head probes on the o_proj input, this package) overlap the parts of the
in-context-learning circuit in Qwen3-4B (and Qwen3-0.6B for CPU smoke tests). The module only reads data; it loads no
model and is imported by nothing on the Pythia path.

Source: github.com/andrewgcodes/in-context-learning-circuit pinned at ICL_COMMIT (MIT, Copyright (c) 2026 Andrew Gao,
William Gao; see LICENSE, Attribution). References below are repo paths at that commit; "main.tex:N" is paper/main.tex.

Two kinds of sets
-----------------
1. Census roles (`census_sets`). data/D188_heads_<run>.json holds, for every head of the screened layers, the paired
   effect of mean-ablating that head alone at the query on the concept12n battery (dret = symbol+rare margin, dR / dP =
   the parts of the word-label margin that follow the demonstrations / the word meaning; percentile-bootstrap CIs).
   The rule of icl/analyze_D188_scale.py is applied exactly:
       readers    dret_hi < 0                 (:47, "knock lowers the margin, CI<0")
       brakes     dret_lo > 0                 (:47, "knock raises the margin, CI>0")
       p_brakes   dP not NaN and dP_lo > 0    (:50, "brakes on word meaning", Table tab:scale main.tex:475-490)
       p_carriers dP not NaN and dP_hi < 0    (:51, printed by the script, not in the paper's table)
   R-readers (dR CI < 0, :50) are NOT recoverable: the JSON stores only the mean dR (:63 dumps dR without dR_lo/dR_hi),
   and the count is in neither data/D188_summary.json nor the paper.
   Expected counts (data/D188_summary.json): 4B 89 readers / 56 brakes / 54 P-brakes; 0.6B 70 / 13 / 23.
   NB: the census labels by the sign of the knock effect on concept12n, not by attention role, so it also calls the
   query heads (22:10, 23:26) and copiers (35:26, 33:13) "readers", and the paper's readers 24:21 / 22:20 "brakes".
2. Named roles (`NAMED`): heads that the paper, its figure scripts or data files name explicitly, with references.
   Roles whose member list was never released (duplicate-token heads, pooling heads, writers, the reader-20 and
   brake-24 sets, most of the 67-head named circuit) are kept as population-only entries (`explicit="none"` or
   "partial") with the layer band the paper gives. No member is inferred.

Coverage (restrict every overlap test to it)
--------------------------------------------
The census screens only the upper half: layers 16-35 of 36 (4B, 8B; 640 heads) and 10-27 of 28 (0.6B, 1.7B; 288
heads), main.tex:460 and :475. The completeness screens cover layers 20-27 (data/D174_screen_full.txt, main.tex:738)
and 0-19 (data/D174_screen_L0-19_full.txt, main.tex:746). `RoleSet.test_layers` is the band in which membership was
decided exhaustively (else the band the paper places the role in); `RoleSet.overlap_ready` is True only for full lists.
`RoleSet.family` pre-specifies the confirmatory Holm family: "primary" (the four census roles, the 33 named-circuit heads
of L20-27, the four query-type heads of L20-27, the four L0-19 additions, the value KV groups) or "descriptive" (2-5-head
fragments, unions and subsets of other roles: overlap and p reported, never part of a verdict). `RoleSet.unit` is "head"
or "kv_group" (value_kv_groups: the source names KV groups, so it is tested at group level).

Base vs post-trained
--------------------
All sets were identified in the *-Base models. The value-head paper probes the post-trained Qwen/Qwen3-4B. Same
architecture; the ICL paper carries the base model's 189-head set into the instruct model unchanged and finds that it
still carries retrieval (main.tex:470, data/D187QIn_*). `resolve_icl_model("Qwen/Qwen3-4B")` therefore maps to the
base sets and reports `transferred=True`; say so wherever such an overlap is reported.

GQA: Qwen3 query head h reads KV group h // (heads / kv_heads) (icl/run.py:229-233); `kv_group_query_heads` expands.
"""
import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass, field

ICL_REPO = "andrewgcodes/in-context-learning-circuit"
ICL_COMMIT = "26c44caf8f9b5a5a197988095f1f76b0562bbaa4"
RAW_BASE = f"https://raw.githubusercontent.com/{ICL_REPO}/{ICL_COMMIT}/"
PAPER = "Gao & Gao, A Circuit for In-Context Learning, arXiv 2611.05344"
MAX_DOWNLOAD_BYTES = 4 << 20   # the census files are 66-150 kB
ENV_REPO = "VH_ICL_REPO"       # local checkout of the ICL repo (at ICL_COMMIT)
ENV_CACHE = "VH_ICL_CACHE"     # download cache (default ~/.cache/valueheads_pythia/icl)

CENSUS_ROLES = ("readers", "brakes", "p_brakes", "p_carriers")
CENSUS_RULES = {
    "readers": ("dret_hi < 0", "icl/analyze_D188_scale.py:47"),
    "brakes": ("dret_lo > 0", "icl/analyze_D188_scale.py:47"),
    "p_brakes": ("dP not NaN and dP_lo > 0", "icl/analyze_D188_scale.py:50"),
    "p_carriers": ("dP not NaN and dP_hi < 0", "icl/analyze_D188_scale.py:51"),
}
_ROW_KEYS = ("L", "H", "dret", "dret_lo", "dret_hi", "dR", "dP", "dP_lo", "dP_hi")


def _band(a, b):
    return tuple(range(a, b + 1))


def _hs(text):
    """'22:10 23:26' -> ((22, 10), (23, 26)), sorted, duplicates rejected."""
    heads = [tuple(int(x) for x in tok.split(":")) for tok in text.replace(",", " ").split()]
    if len(set(heads)) != len(heads):
        raise ValueError(f"duplicate head in {text!r}")
    return tuple(sorted(heads))


# ----------------------------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusSpec:
    """One D188 census file at ICL_COMMIT; `expected` / `expected_sums` are data/D188_summary.json at the same commit."""
    model: str              # registry short name (valueheads_pythia.config), e.g. "qwen3-4b-base"
    hf_id: str
    run: str                # the ICL run tag inside the file name
    path: str               # repo-relative path
    sha256: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    screened_layers: tuple
    expected: dict = field(compare=False)
    expected_sums: dict = field(compare=False)

    @property
    def url(self):
        return RAW_BASE + self.path


CENSUS = {
    "qwen3-4b-base": CensusSpec(
        "qwen3-4b-base", "Qwen/Qwen3-4B-Base", "lbD188SC4", "data/D188_heads_Qwen3-4B-Base_lbD188SC4.json",
        "fa7955c5775181f7d4ecbf9ed7aaa7996d4bc7f283b2e13ed44e665e74bd6a64", 36, 32, 8, _band(16, 35),
        {"readers": 89, "brakes": 56, "p_brakes": 54},
        {"readers": -4.692121875, "brakes": 3.7641567708333312, "p_brakes": 3.6054406249999995}),
    "qwen3-0.6b-base": CensusSpec(
        "qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base", "lbD188SC06c", "data/D188_heads_Qwen3-0.6B-Base_lbD188SC06c.json",
        "360d2917d24b202e3f7a70798fac5bcb4608e8d1a15a89196d158a63ced62652", 28, 16, 8, _band(10, 27),
        {"readers": 70, "brakes": 13, "p_brakes": 23},
        {"readers": -2.084120833333333, "brakes": 0.8175296874999999, "p_brakes": 1.7304687500000004}),
    "qwen3-1.7b-base": CensusSpec(
        "qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "lbD188SC17c", "data/D188_heads_Qwen3-1.7B-Base_lbD188SC17c.json",
        "d904045b7bd2c59d1d2d2d120d30dc447d013f7e967a9e59e5d5ef444e10c15f", 28, 16, 8, _band(10, 27),
        {"readers": 48, "brakes": 13, "p_brakes": 24},
        {"readers": -2.5968197916666664, "brakes": 1.2137468750000002, "p_brakes": 2.059245833333333}),
    "qwen3-8b-base": CensusSpec(
        "qwen3-8b-base", "Qwen/Qwen3-8B-Base", "lbD188SC8b", "data/D188_heads_Qwen3-8B-Base_lbD188SC8b.json",
        "1e440e25a2fc9d27b74d39dcd37a7947acc9c2dce31d7d4d0e20b60d86ad89f4", 36, 32, 8, _band(16, 35),
        {"readers": 103, "brakes": 32, "p_brakes": 74},
        {"readers": -5.944495833333334, "brakes": 3.272618229166666, "p_brakes": 4.027992708333333}),
}


def resolve_icl_model(name):
    """(base short name, transferred) for a registry short name or HF id, case-insensitive. Post-trained Qwen3 models
    (no "-base") map to the base model's sets with transferred=True (see module docstring)."""
    key = str(name).strip().split("/")[-1].lower()
    transferred = not key.endswith("-base")
    base = key if not transferred else key + "-base"
    if base not in CENSUS:
        raise KeyError(f"no ICL head sets for {name!r} (known: {sorted(CENSUS)})")
    return base, transferred


def kv_group_query_heads(layer, group, n_heads=32, n_kv_heads=8):
    """Query heads (layer, h) that read KV group `group` (Qwen3 GQA: h // (n_heads // n_kv_heads) == group)."""
    rep = n_heads // n_kv_heads
    if not 0 <= group < n_kv_heads or n_heads % n_kv_heads:
        raise ValueError(f"bad KV group {group} for {n_heads} heads / {n_kv_heads} KV heads")
    return tuple((layer, h) for h in range(group * rep, (group + 1) * rep))


# ----------------------------------------------------------------------------------------------------------------
# role sets
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleSet:
    """One ICL role with provenance.

    explicit:        "full" (heads is the source's complete list for this role), "partial" (examples the paper names;
                     the role has more members) or "none" (population only, heads == ()).
    n_reported:      size the source reports (int, "~20" style string, or None).
    layers:          layer band the paper places the role in.
    screened_layers: band in which membership was decided exhaustively (absence there is informative); () if none.
    unit:            "head" (the source names query heads) or "kv_group" (the source names KV groups; `heads` holds every
                     query head of each group and the overlap test counts groups, see overlap.py).
    family:          "primary" (confirmatory: enters the Holm family of the overlap test) or "descriptive" (fragments,
                     unions or subsets of other roles: reported, never a verdict).
    """
    role: str
    heads: tuple
    explicit: str
    n_reported: object
    layers: tuple
    screened_layers: tuple
    sources: tuple
    note: str = ""
    unit: str = "head"
    family: str = "primary"

    def __post_init__(self):
        if self.explicit not in ("full", "partial", "none"):
            raise ValueError(f"{self.role}: explicit must be full / partial / none, got {self.explicit!r}")
        if (self.explicit == "none") != (len(self.heads) == 0):
            raise ValueError(f"{self.role}: explicit={self.explicit!r} but {len(self.heads)} heads")
        if self.unit not in ("head", "kv_group"):
            raise ValueError(f"{self.role}: unit must be head / kv_group, got {self.unit!r}")
        if self.family not in ("primary", "descriptive"):
            raise ValueError(f"{self.role}: family must be primary / descriptive, got {self.family!r}")

    @property
    def test_layers(self):
        """Layers to which an overlap test with this role must be restricted."""
        return self.screened_layers or self.layers

    @property
    def overlap_ready(self):
        """True only for full lists with a known test band (partial / population-only roles cannot be tested)."""
        return self.explicit == "full" and bool(self.test_layers)

    @property
    def confirmatory(self):
        """True for full lists of the pre-specified primary family (these, and only these, can get a verdict)."""
        return self.overlap_ready and self.family == "primary"

    def restrict(self, heads):
        """The given (layer, head) pairs that lie in `test_layers`, sorted."""
        band = set(self.test_layers)
        return sorted({(int(l), int(h)) for l, h in heads if int(l) in band})

    def as_dict(self):
        return {"role": self.role, "heads": [list(h) for h in self.heads], "explicit": self.explicit,
                "n_listed": len(self.heads), "n_reported": self.n_reported, "layers": list(self.layers),
                "screened_layers": list(self.screened_layers), "test_layers": list(self.test_layers),
                "sources": list(self.sources), "note": self.note, "unit": self.unit, "family": self.family}


def _role(role, heads, explicit, n_reported, layers, screened, sources, note="", unit="head", family="primary"):
    heads = _hs(heads) if isinstance(heads, str) else tuple(sorted(heads))
    if len(set(heads)) != len(heads):
        raise ValueError(f"{role}: duplicate heads")
    return RoleSet(role, heads, explicit, n_reported, tuple(layers), tuple(screened), tuple(sources), note, unit, family)


# The 33 named-circuit heads of layers 20-27: rows flagged "N" (named) in the per-head screen of every head in
# layers 20-27 (data/D174_screen_full.txt; icl/run.py:503-509 screens named heads as scrn_, analyze_D174.py prints N).
NAMED_CIRCUIT_L20_27_QWEN3_4B = _hs(
    "20:26 21:2 21:11 21:18 21:20 22:0 22:2 22:8 22:9 22:10 22:11 22:13 22:16 22:19 22:20 22:21 22:30 "
    "23:3 23:21 23:24 23:26 23:28 23:29 24:8 24:21 24:22 24:28 24:29 25:6 25:28 27:15 27:16 27:19")

NAMED = {
    "qwen3-4b-base": {r.role: r for r in [
        _role("query_heads", "22:10 23:26", "full", 2, (22, 23), (),
              ("main.tex:240 (sec. 'Two heads write the query code')", "figures/fig1_circuit.py:52",
               "figures/fig8_walkthrough.py:10 (query code, L22-23)"),
              "The pair carries 40% of the query flip. How it was chosen is not documented; the L20-27 screen "
              "(main.tex:738) confirms them as the two largest single-head effects there. Descriptive: not the "
              "complete query-type set of L22-23 (the screen adds 23:25, main.tex:740); tested as part of "
              "query_type_heads_L20_27.", family="descriptive"),
        _role("query_type_screen_L20_27", "20:5 23:25", "full", 2, _band(20, 27), _band(20, 27),
              ("main.tex:740", "data/D174_screen_full.txt:263 (EXT)", "data/D176Q2t_attrows.txt"),
              "Unnamed L20-27 heads passing the screen rule (CI<0, mean<-0.03; icl/analyze_D174.py:36) whose "
              "attention goes to the query sentence: query heads of the kind of 22:10 / 23:26. Descriptive (fragment "
              "of query_type_heads_L20_27).", family="descriptive"),
        _role("query_type_heads_L20_27", "20:5 22:10 23:25 23:26", "full", 4, _band(20, 27), _band(20, 27),
              ("main.tex:240 (22:10, 23:26)", "main.tex:740 ('20:5 and 23:25 ... are query heads of the kind already "
               "named')", "data/D174_screen_full.txt:263 (EXT)", "data/D176Q2t_attrows.txt"),
              "Every query-type head of L20-27: the two named query heads plus the two the exhaustive L20-27 screen "
              "adds. The screen covers every head of the band, so absence there is informative."),
        _role("readers_named", "22:13 22:20 24:8 24:21 24:22 28:0 29:11", "partial", 20, _band(21, 29), (),
              ("figures/fig1_circuit.py:52 (29.11 28.0 24.8 ... ~90)", "main.tex:327 + icl/run.py:479 (amplified "
               "24:21 29:11 22:20)", "figures/fig12_reader_rows.py:6", "main.tex:738 (22:13, 24:22)",
               "figures/fig8_walkthrough.py:11 (readers L21-29)"),
              "The reader-20 set (main.tex:263, tab:claims main.tex:592) is not released; ~90 readers overall "
              "(main.tex:98, 261). The census (concept12n) calls 24:21 and 22:20 brakes and 24:8 neither."),
        _role("readers_screen_L20_27", "22:22 22:23 24:13 24:23", "full", 4, _band(20, 27), _band(20, 27),
              ("main.tex:740", "data/D174_screen_full.txt:263 (EXT)", "data/D176Q2t_attrows.txt"),
              "Unnamed L20-27 heads passing the screen rule whose query attention goes mostly to the recurring-label "
              "slots (main.tex:740; slot mass 0.54-0.97 in D176Q2t_attrows.txt): readers the top-20 ranking missed. "
              "Descriptive: the named readers of the same band are not in it (fragment of a role).",
              family="descriptive"),
        _role("arrow_readers_screen_L20_27", "20:17 27:12", "full", 2, _band(20, 27), _band(20, 27),
              ("main.tex:740", "data/D174_screen_full.txt:263 (EXT)", "data/D176Q2t_attrows.txt"),
              "Unnamed L20-27 heads passing the screen rule that attend to the demonstration arrows (read the "
              "prompt-level state the pooling stage writes). Descriptive (2 heads).", family="descriptive"),
        _role("screen_additions_L20_27", "20:5 20:17 22:22 22:23 23:25 24:13 24:23 27:12", "full", 8,
              _band(20, 27), _band(20, 27),
              ("data/D174_screen_full.txt:263 (EXT 8 heads)", "icl/analyze_D174.py:36", "main.tex:740"),
              "Union of the three *_screen_L20_27 roles; the 8 heads added to the named set (67 + 8 + 4 = 79). "
              "Descriptive (union of other roles, redundant as a test).", family="descriptive"),
        _role("screen_additions_L0_19", "6:15 6:20 7:31 8:23", "full", 4, _band(0, 19), _band(0, 19),
              ("main.tex:746-748", "data/D174_screen_L0-19_full.txt:3-7", "data/D178Q2u_attrows.txt"),
              "The four early heads the paper adds to the named set: ranks 1, 3, 4, 5 of the L0-19 screen (19:24, "
              "rank 2, is not added); only 8:23 passes the EXT rule (D174_screen_L0-19_full.txt:647). 8:23 reads the "
              "demo arrows, 6:20 / 7:31 copy query words, 6:15 mixed; the paper calls them substrate, not a part."),
        _role("brakes_named", "22:2 22:8 31:30", "partial", 24, _band(22, 34), (),
              ("figures/fig1_circuit.py:52 (31.30 22.8 22.2 ... (24))", "main.tex:257 (31:30 largest brake)",
               "main.tex:263 (24 heads, layers 22-34)", "figures/fig8_walkthrough.py:12"),
              "The brake-24 set (tab:claims main.tex:593) is not released."),
        _role("copiers_named", "33:13 35:26", "partial", None, _band(30, 35), (),
              ("figures/fig1_circuit.py:55,62 (35.26 33.13 ... L30-35)", "main.tex:103 (last six layers)"),
              "Copier population not released."),
        _role("gate_released_readers", "24:8 28:0 29:11", "partial", 89, _band(21, 29), (),
              ("main.tex:213", "data/D66b_gate_logits.json", "figures/fig4_gate.py:14", "main.tex:325 (29:11)"),
              "Readers whose sink-minus-match logit drops under +g (1.75 / 3.25 / 3.65 nats). The population, '89 "
              "upper-layer slot-reading heads' (main.tex:215), is not released. D66b also logs 20:26 (no effect: "
              "upstream of the layer-23 push)."),
        _role("duplicate_token_heads", (), "none", 16, _band(1, 9), _band(0, 26),
              ("main.tex:211", "figures/fig1_circuit.py:50 (16 heads, L1-9)", "icl/run.py:86,99-101 (lookback16)",
               "icl/score.py:22-32 (dup_head_scores)"),
              "Chosen per run as the top-16 look-back scores (attention from a repeated token to its earlier copies)"
              " over layers < RO = 27 on that run's battery; never stored. screened_layers = that search range."),
        _role("pooling_heads", (), "none", None, _band(20, 23), (),
              ("main.tex:211 (flat pooling heads in layers 20-23)", "figures/fig1_circuit.py:52"),
              "Not released; no pooling head is named anywhere in the repo or paper."),
        _role("writers", (), "none", "~20", _band(10, 23), (),
              ("main.tex:97,150", "figures/fig1_circuit.py:50 (~20 heads, L10-23)", "figures/fig8_walkthrough.py:8"),
              "Not released ('six named writers', main.tex:150, are not listed)."),
        _role("named_circuit_L20_27", NAMED_CIRCUIT_L20_27_QWEN3_4B, "full", 33, _band(20, 27), _band(20, 27),
              ("data/D174_screen_full.txt (column named = N)", "main.tex:738 ('the 33 named heads in the band')"),
              "The L20-27 part of the circ= named set (67 heads: writers, duplicate-token, pooling, query, reader, "
              "brake and copier heads, main.tex:308). Its other members are not released: the L0-19 screen shows 0 "
              "named heads in L0-19 (data/D174_screen_L0-19_full.txt:644), which leaves 34 in L28-35, although "
              "main.tex:308 counts writers and duplicate-token heads in. Later sets: 79 (main.tex:312, 744), 91 "
              "(main.tex:748), 189 (main.tex:314, 470); not released either."),
        _role("natural_label_additions", "24:26 24:30 26:4 27:18 29:14 33:30", "partial", 110, _band(16, 35), (),
              ("main.tex:435", "main.tex:744 (110 heads)"),
              "73 heads in L16-27 and 37 in L28-35 found on word labels; only these six are named."),
        _role("value_kv_groups", kv_group_query_heads(22, 5) + kv_group_query_heads(24, 5)
              + kv_group_query_heads(29, 2), "full", 12, (22, 24, 29), _band(21, 29),
              ("main.tex:144,164", "figures/fig13_transport.py:10 (groups 22.5 24.5 29.2)"),
              "KV groups 22.5, 24.5, 29.2 carry about half of the label value (label-token V patching over all 72 "
              "groups of L21-29). Expanded to the 4 query heads that read each group; a group-level, not "
              "head-level, finding, so the overlap test counts KV groups (unit kv_group).", unit="kv_group"),
        _role("shared_top5_readers_4b_8b", "24:22 24:29 24:30 28:0 29:11", "full", 5, _band(16, 35), _band(16, 35),
              ("main.tex:462",), "Five strongest census readers shared by 4B and 8B (subset of census readers; "
              "descriptive).", family="descriptive"),
        _role("shared_top5_brakes_4b_8b", "25:14 25:30 28:23 31:30 33:21", "full", 5, _band(16, 35), _band(16, 35),
              ("main.tex:462",), "Five strongest census brakes shared by 4B and 8B (subset of census brakes; "
              "descriptive).", family="descriptive"),
    ]},
    "qwen3-0.6b-base": {r.role: r for r in [
        _role("shared_top2_brakes_0.6b_1.7b", "22:7 26:7", "full", 2, _band(10, 27), _band(10, 27),
              ("main.tex:462",), "Top two census brakes shared by 0.6B and 1.7B (22:7 +0.27 nats at 0.6B; subset of "
              "census brakes, descriptive).", family="descriptive"),
        _role("duplicate_token_heads", (), "none", 16, (), _band(0, 20),
              ("icl/run.py:32,86,99-101",),
              "Same rule as for 4B, search range layers < RO = 21 for 28 layers; never run/stored for 0.6B in the "
              "released data, and the paper gives no band for 0.6B."),
    ]},
}


# ----------------------------------------------------------------------------------------------------------------
# census loader
# ----------------------------------------------------------------------------------------------------------------

def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _cache_path(spec, cache_dir=None):
    root = cache_dir or os.environ.get(ENV_CACHE) or os.path.join(os.path.expanduser("~"), ".cache",
                                                                     "valueheads_pythia", "icl")
    return os.path.join(root, ICL_COMMIT[:12], os.path.basename(spec.path))


def _download(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{url}: more than {MAX_DOWNLOAD_BYTES} bytes, refusing")
    return data


def _read_census_bytes(spec, source, cache_dir, allow_download, timeout):
    """(bytes, origin). source: None | path to the JSON | ICL repo checkout dir | http(s) URL."""
    if source is None and os.environ.get(ENV_REPO):
        source = os.environ[ENV_REPO]
    if source is not None:
        source = str(source)
        if source.startswith(("http://", "https://")):
            return _download(source, timeout), source
        path = os.path.join(source, spec.path) if os.path.isdir(source) else source
        with open(path, "rb") as f:
            return f.read(), os.path.abspath(path)
    cached = _cache_path(spec, cache_dir)
    if os.path.exists(cached):
        with open(cached, "rb") as f:
            return f.read(), cached
    if not allow_download:
        raise FileNotFoundError(f"{spec.path}: no local source (pass source=, set {ENV_REPO}) and no cache at "
                                f"{cached}; downloading is disabled")
    return _download(spec.url, timeout), spec.url


def _store_cache(spec, data, cache_dir):
    path = _cache_path(spec, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def _float(v):
    if v is None:
        return math.nan
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"not a number: {v!r}")
    return float(v)


def validate_census_rows(rows, n_layers, n_heads):
    """Type / range / uniqueness checks; returns rows as plain dicts (L, H int; the rest float, NaN allowed)."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("census must be a non-empty JSON list")
    out, seen = [], set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or any(k not in r for k in _ROW_KEYS):
            raise ValueError(f"census row {i}: needs keys {_ROW_KEYS}")
        L, H = r["L"], r["H"]
        if isinstance(L, bool) or isinstance(H, bool) or not isinstance(L, int) or not isinstance(H, int):
            raise ValueError(f"census row {i}: L / H must be ints")
        if not (0 <= L < n_layers and 0 <= H < n_heads):
            raise ValueError(f"census row {i}: head {L}:{H} outside {n_layers} layers x {n_heads} heads")
        if (L, H) in seen:
            raise ValueError(f"census row {i}: duplicate head {L}:{H}")
        seen.add((L, H))
        row = {"L": L, "H": H, **{k: _float(r[k]) for k in _ROW_KEYS[2:]}}
        for lo, hi in (("dret_lo", "dret_hi"), ("dP_lo", "dP_hi")):
            if row[lo] > row[hi]:
                raise ValueError(f"census row {i}: {lo} > {hi}")
        out.append(row)
    return out


def classify_census(rows):
    """role -> sorted tuple of (layer, head) by the exact rule of icl/analyze_D188_scale.py (see CENSUS_RULES)."""
    roles = {k: [] for k in CENSUS_ROLES}
    for r in rows:
        lh = (int(r["L"]), int(r["H"]))
        if r["dret_hi"] < 0:
            roles["readers"].append(lh)
        if r["dret_lo"] > 0:
            roles["brakes"].append(lh)
        if not math.isnan(r["dP"]):
            if r["dP_lo"] > 0:
                roles["p_brakes"].append(lh)
            if r["dP_hi"] < 0:
                roles["p_carriers"].append(lh)
    return {k: tuple(sorted(v)) for k, v in roles.items()}


@dataclass(frozen=True)
class CensusResult:
    model: str
    roles: dict             # role -> tuple of (layer, head)
    screened_heads: frozenset
    screened_layers: tuple
    rows: tuple             # validated rows (dicts)
    origin: str             # local path, cache path or URL
    sha256: str
    pinned: bool            # sha256 equals the pinned file at ICL_COMMIT
    spec: CensusSpec = field(repr=False, compare=False, default=None)

    def effects(self, key="dret"):
        """(layer, head) -> census value (dret, dret_lo, dret_hi, dR, dP, dP_lo, dP_hi)."""
        if key not in _ROW_KEYS[2:]:
            raise KeyError(key)
        return {(r["L"], r["H"]): r[key] for r in self.rows}

    def role_sets(self):
        """Census roles as RoleSets (full lists, tested in the screened layers), plus the unrecoverable R-readers."""
        spec = self.spec or CENSUS[self.model]
        src = (f"{spec.path} @ {ICL_COMMIT[:12]}" + ("" if self.pinned else f" (unpinned file {self.origin})"),)
        out = {f"census_{k}": RoleSet(f"census_{k}", self.roles[k], "full" if self.roles[k] else "none",
                                      spec.expected.get(k), self.screened_layers, self.screened_layers,
                                      src + (CENSUS_RULES[k][1],), f"rule: {CENSUS_RULES[k][0]}")
               for k in CENSUS_ROLES}
        out["census_r_readers"] = RoleSet(
            "census_r_readers", (), "none", None, self.screened_layers, self.screened_layers,
            src + ("icl/analyze_D188_scale.py:50,63",),
            "rule dR CI<0; not recoverable: the census JSON stores the mean dR only, no dR CI")
        return out


def census_sets(model="qwen3-4b-base", source=None, *, cache_dir=None, allow_download=True, verify=True, timeout=60):
    """Load the D188 census for `model` and classify it.

    model:  registry short name / HF id (post-trained names map to the base census) or a CensusSpec.
    source: None (env VH_ICL_REPO, then cache, then the pinned raw GitHub URL), a path to the JSON, a checkout of the
            ICL repo, or a URL. With verify=True the bytes must match the pinned sha256 (otherwise ValueError);
            verify=False accepts other files (e.g. tests) and skips the expected-count self-check.
    """
    spec = model if isinstance(model, CensusSpec) else CENSUS[resolve_icl_model(model)[0]]
    data, origin = _read_census_bytes(spec, source, cache_dir, allow_download, timeout)
    digest = _sha256(data)
    pinned = digest == spec.sha256
    if verify and not pinned:
        raise ValueError(f"{origin}: sha256 {digest} != pinned {spec.sha256} ({spec.path} @ {ICL_COMMIT[:12]})")
    if pinned and origin.startswith(("http://", "https://")):
        _store_cache(spec, data, cache_dir)
    rows = validate_census_rows(json.loads(data.decode("utf-8")), spec.n_layers, spec.n_heads)
    roles = classify_census(rows)
    screened = frozenset((r["L"], r["H"]) for r in rows)
    layers = tuple(sorted({l for l, _ in screened}))
    if verify:
        missing = [(l, h) for l in spec.screened_layers for h in range(spec.n_heads) if (l, h) not in screened]
        if layers != spec.screened_layers or missing:
            raise ValueError(f"{origin}: screened layers {layers} / {len(missing)} missing heads do not match "
                             f"{spec.screened_layers}")
        for k, n in spec.expected.items():
            members, col = set(roles[k]), "dP" if k == "p_brakes" else "dret"
            got_sum = sum(r[col] for r in rows if (r["L"], r["H"]) in members)
            if len(roles[k]) != n or not math.isclose(got_sum, spec.expected_sums[k], rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"{spec.model} {k}: {len(roles[k])} heads (sum {got_sum:+.4f}) but "
                                   f"D188_summary.json has {n} (sum {spec.expected_sums[k]:+.4f})")
    return CensusResult(spec.model, roles, screened, layers, tuple(rows), origin, digest, pinned, spec)


def icl_role_sets(model="qwen3-4b-base", census=None, **load_kw):
    """All role sets for `model`: census roles (loaded via census_sets(**load_kw) unless `census` is given; pass
    census=False to skip them) and the named roles. Returns {role: RoleSet}."""
    base, _ = resolve_icl_model(model)
    out = {}
    if census is not False:
        res = census if census is not None else census_sets(base, **load_kw)
        out.update(res.role_sets())
    out.update(NAMED.get(base, {}))
    return out


def role_table(sets):
    """Rows (dicts) for a role overview: role, n listed, n reported, layers, test band, explicit, sources."""
    def span(ls):
        return f"{min(ls)}-{max(ls)}" if ls and list(ls) == list(range(min(ls), max(ls) + 1)) else \
            ",".join(map(str, ls)) or "-"
    return [{"role": r.role, "n_listed": len(r.heads), "n_reported": r.n_reported, "layers": span(r.layers),
             "test_layers": span(r.test_layers), "explicit": r.explicit, "overlap_ready": r.overlap_ready,
             "family": r.family, "unit": r.unit, "sources": "; ".join(r.sources)} for r in sets.values()]


# ----------------------------------------------------------------------------------------------------------------
# D174 screen parser (to re-derive NAMED_CIRCUIT_L20_27 / the screen additions from the released text files)
# ----------------------------------------------------------------------------------------------------------------

def parse_d174_screen(text):
    """Rows of an analyze_D174.py table: dicts with L, H, named (bool), mean, lo, hi; plus the EXT list.
    Returns (rows, ext) where ext is a tuple of (layer, head) in printed order (empty if no EXT line)."""
    rows, ext = [], ()
    for line in text.splitlines():
        parts = line.split()
        if line.startswith("EXT ") and ":" in line:
            tail = line.rsplit("):", 1)[-1].strip()
            ext = tuple(tuple(int(x) for x in tok.split(":")) for tok in tail.split(",") if tok)
            continue
        # "-0.114 [-0.158,-0.071] -0.168 -0.061  23:26  N     0.048  0.005  144" (icl/analyze_D174.py:32)
        if len(parts) >= 9 and parts[1].startswith("[") and parts[5] in ("N", "-") and ":" in parts[4]:
            lo, hi = parts[1].strip("[]").split(",")
            L, H = (int(x) for x in parts[4].split(":"))
            rows.append({"L": L, "H": H, "named": parts[5] == "N", "mean": float(parts[0]),
                         "lo": float(lo), "hi": float(hi)})
    return rows, ext
