"""WP2 consensus and extraction agreement (Paper 1, RQ2; design sections 5.3-5.4).

Reads the checkpointed raw extraction responses, validates every claim mechanically
(``normalize.process_claim``), groups accepted claims into (manifest, canonical key) units,
counts the models that proposed each unit (``extractionVotes``), and computes:

  * pairwise F1 (micro over manifests) per model pair, overall and per type;
  * Krippendorff's alpha (nominal) on presence, with a manifest-level bootstrap CI, and on
    mode / status / qualifier over units that all three models extracted;
  * a deterministic regex recall bound for T1 and T5;
  * per-model parse, schema, grounding and shape-rejection rates.

Usage:
    python wp2_consistency.py consensus
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

from . import common as C
from . import extract as X
from . import normalize as N

logger = logging.getLogger("wp2.consensus")

CONSERVATIVE_MODE_ORDER = ["example", "negated", "to_be_created", "asserts_existing"]
GROUND_REASONS = {"ungrounded_quote", "subject_not_in_quote", "ungrounded_component"}


def final_panel(cfg: dict[str, Any], res_dir: Any) -> list[str]:
    sp = res_dir / "extraction" / "substitution.json"
    if sp.exists():
        return C.read_json(sp).get("final_panel") or list(cfg["extraction"]["panel"])
    return list(cfg["extraction"]["panel"])


def validate_all(cfg: dict[str, Any], limit_repos: Optional[int] = None, repair: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate all raw responses. Returns (validated rows, per-model statistics)."""
    res_dir = C.results_dir(cfg)
    panel = final_panel(cfg, res_dir)
    corpus = C.load_corpus(cfg)
    if limit_repos:
        corpus = corpus[:limit_repos]
    rows: list[dict[str, Any]] = []
    stats: dict[str, Counter] = {m: Counter() for m in panel}
    reasons: dict[str, Counter] = {m: Counter() for m in panel}
    ntype: dict[str, Counter] = {m: Counter() for m in panel}
    usage = {m: {"prompt": 0, "completion": 0, "latency_s": 0.0, "api_calls": 0} for m in panel}
    for e in corpus:
        mpath = e.get("manifest_path") or "CLAUDE.md"
        snap = res_dir / "snapshots" / f"{e['slug']}.json"
        if snap.exists():
            mpath = C.read_json(snap).get("manifest_path") or mpath
        chunks = C.chunk_manifest(e["claude_md_content"], cfg["extraction"]["chunk_max_chars"])
        for ch in chunks:
            for m in panel:
                p = X.raw_path(res_dir, e["slug"], ch["idx"], m)
                if not p.exists():
                    stats[m]["missing_response"] += 1
                    continue
                rec = C.read_json(p)
                stats[m]["responses"] += 1
                usage[m]["prompt"] += rec["usage"].get("prompt", 0)
                usage[m]["completion"] += rec["usage"].get("completion", 0)
                usage[m]["latency_s"] += rec["latency_s"]
                usage[m]["api_calls"] += rec["n_api_calls"]
                stats[m]["parse_retries"] += rec.get("parse_retries", 0)
                if not rec["parse_ok"]:
                    stats[m]["parse_failures"] += 1
                    continue
                items, _ = X.parse_response(rec["text"])
                good, bad = X.validate_claims(items or [])
                stats[m]["raw_claims"] += len(items or [])
                stats[m]["schema_invalid"] += bad
                for raw in good:
                    stats[m]["schema_valid"] += 1
                    accepted, why = N.process_claim(raw, ch, mpath, cfg, repair=repair, repo_name=e['full_name'].split('/')[-1])
                    if not accepted:
                        r = why[0] if why else "unknown"
                        reasons[m][r] += 1
                        stats[m]["rejected_grounding" if r in GROUND_REASONS else "rejected_shape"] += 1
                        rows.append({"repo": e["full_name"], "chunk": ch["idx"], "model": m, "status": "rejected",
                                     "reason": r, "type": raw["type"], "subject": raw["subject"][:80]})
                        continue
                    for a in accepted:
                        a["repaired"] = not (raw.get("quote") or "").strip() and not a["attrs"].get("tree")
                        stats[m]["quote_repaired"] += int(a["repaired"])
                        ntype[m][a["tcode"]] += 1
                        stats[m]["accepted_units_raw"] += 1
                        rows.append({"repo": e["full_name"], "chunk": ch["idx"], "model": m, "status": "accepted", **a})
    summary = {m: {**dict(stats[m]), "reject_reasons": dict(reasons[m]), "accepted_by_type": dict(ntype[m]),
                   "usage": usage[m]} for m in panel}
    for m in panel:
        s = summary[m]
        n_resp = max(1, s.get("responses", 0))
        s["parse_failure_rate"] = s.get("parse_failures", 0) / n_resp
        s["grounding_failure_rate"] = s.get("rejected_grounding", 0) / max(1, s.get("schema_valid", 0))
    return rows, summary


def _merge_paths(units: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Merge rule: T1/T2 keys of one manifest on the same line whose paths are '/'-suffix related map to one key."""
    remap: dict[tuple[str, str], str] = {}
    by_line: dict[tuple[str, str, int, str], list[tuple[str, str]]] = defaultdict(list)
    for (repo, ks), u in units.items():
        if u["tcode"] in ("T1", "T2"):
            kind = u["key"][2] if u["tcode"] == "T2" else ""
            by_line[(repo, u["tcode"], u["line"], kind)].append((repo, ks))
    for (_repo, _t, _line, _kind), members in by_line.items():
        if len(members) < 2:
            continue
        parent = {m: m for m in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in itertools.combinations(members, 2):
            pa, pb = units[a]["attrs"]["path"], units[b]["attrs"]["path"]
            if pa == pb or pa.endswith("/" + pb) or pb.endswith("/" + pa):
                parent[find(a)] = find(b)
        clusters: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for m in members:
            clusters[find(m)].append(m)
        for cl in clusters.values():
            if len(cl) < 2:
                continue
            rep = max(cl, key=lambda m: (len(units[m]["models"]), len(units[m]["attrs"]["path"])))
            for m in cl:
                if m != rep:
                    remap[m] = rep[1]
    return remap


def build_units(rows: list[dict[str, Any]], panel: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Group accepted rows into (repo, key) units with per-model attributes (pre-merge)."""
    units: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r["status"] != "accepted":
            continue
        k = (r["repo"], r["key_str"])
        u = units.setdefault(k, {"repo": r["repo"], "key": r["key"], "key_str": r["key_str"], "tcode": r["tcode"],
                                 "subject": r["subject"], "attrs": r["attrs"], "per_model": {}, "models": set(),
                                 "quote": r["quote"], "line": r["line"], "first_by_model": {}})
        pm = u["per_model"].setdefault(r["model"], {"modes": [], "attrs": []})
        pm["modes"].append(r["mode"])
        pm["attrs"].append(r["attrs"])
        u["models"].add(r["model"])
        u["first_by_model"].setdefault(r["model"], r)
    for u in units.values():  # panel order decides the quote / line source
        for m in panel:
            if m in u["first_by_model"]:
                f = u["first_by_model"][m]
                u["quote"], u["line"], u["attrs"] = f["quote"], f["line"], f["attrs"]
                break
    return units


def merge_units(units: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    """Apply the T1/T2 suffix merge rule on copies (the pre-merge units stay intact for agreement)."""
    import copy

    work = {k: {**u, "models": set(u["models"]), "per_model": copy.deepcopy(u["per_model"])} for k, u in units.items()}
    remap = _merge_paths(work)
    for k, tgt in remap.items():
        u, t = work[k], work[(k[0], tgt)]
        t["models"] |= u["models"]
        for m, pm in u["per_model"].items():
            tp = t["per_model"].setdefault(m, {"modes": [], "attrs": []})
            tp["modes"] += pm["modes"]
            tp["attrs"] += pm["attrs"]
    return {k: u for k, u in work.items() if k not in remap}, {"merged_units": len(remap)}


def majority(values: list[str], conservative: list[str]) -> tuple[str, dict[str, int], bool]:
    """Majority value; tie goes to the more conservative (earlier in ``conservative``)."""
    c = Counter(values)
    top = max(c.values())
    cands = [v for v, n in c.items() if n == top]
    best = min(cands, key=lambda v: conservative.index(v) if v in conservative else 99)
    return best, dict(c), len(c) == 1


def consensus_rows(units: dict[tuple[str, str], dict[str, Any]], panel: list[str]) -> list[dict[str, Any]]:
    """One row per unit with votes, majority mode and type attributes."""
    out = []
    for (repo, ks), u in sorted(units.items()):
        mode_by_model = {m: majority(pm["modes"], CONSERVATIVE_MODE_ORDER)[0] for m, pm in u["per_model"].items()}
        mode, counts, unanimous = majority(list(mode_by_model.values()), CONSERVATIVE_MODE_ORDER)
        attrs = dict(u["attrs"])
        if u["tcode"] == "T2":
            st, _, _ = majority([a["status"] for pm in u["per_model"].values() for a in pm["attrs"]],
                                ["legacy", "deprecated", "frozen", "stable", "generated", "do_not_edit"])
            attrs["status"] = st
        if u["tcode"] == "T6":
            q, _, _ = majority([a["qualifier"] for pm in u["per_model"].values() for a in pm["attrs"]], ["exact", "min"])
            attrs["qualifier"] = q
        if u["tcode"] in ("T1", "T2"):
            attrs["is_dir"] = any(a.get("is_dir") for pm in u["per_model"].values() for a in pm["attrs"])
        out.append({"repo": repo, "key": u["key"], "key_str": ks, "tcode": u["tcode"], "subject": u["subject"],
                    "attrs": attrs, "quote": u["quote"], "line": u["line"], "votes": len(u["models"]),
                    "models": sorted(u["models"]), "mode": mode, "mode_votes": counts,
                    "mode_unanimous": unanimous and len(mode_by_model) == len(panel)})
    return out


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------

def pairwise_f1(sets: dict[str, dict[str, set]], panel: list[str], tcode: Optional[str] = None) -> dict[str, float]:
    out = {}
    for a, b in itertools.combinations(panel, 2):
        inter = tot = 0
        for repo in set(sets[a]) | set(sets[b]):
            A = {k for k in sets[a].get(repo, set()) if tcode is None or k[0] == tcode}
            B = {k for k in sets[b].get(repo, set()) if tcode is None or k[0] == tcode}
            inter += len(A & B)
            tot += len(A) + len(B)
        out[f"{a} | {b}"] = (2 * inter / tot) if tot else float("nan")
    return out


def _alpha(matrix: np.ndarray) -> float:
    import krippendorff

    if matrix.shape[1] == 0 or len(np.unique(matrix[~np.isnan(matrix)])) < 2:
        return float("nan")
    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal"))


def presence_matrix(units: dict[tuple[str, str], dict[str, Any]], panel: list[str], repos: list[str],
                    tcode: Optional[str] = None) -> tuple[np.ndarray, list[str]]:
    """Raters x units 0/1 presence matrix over the union of (repo, key) units (pre-merge keys)."""
    cols, owner = [], []
    for (repo, ks), u in units.items():
        if tcode and u["tcode"] != tcode:
            continue
        cols.append([1.0 if m in u["models"] else 0.0 for m in panel])
        owner.append(repo)
    return (np.array(cols).T if cols else np.zeros((len(panel), 0))), owner


def bootstrap_alpha(matrix: np.ndarray, owner: list[str], n: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    repos = sorted(set(owner))
    idx_by_repo: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(owner):
        idx_by_repo[r].append(i)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(repos), size=len(repos), replace=True)
        cols = [i for j in pick for i in idx_by_repo[repos[j]]]
        a = _alpha(matrix[:, cols])
        if a == a:
            vals.append(a)
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n_valid": len(vals)} if vals else {}


def attribute_alpha(units: dict[tuple[str, str], dict[str, Any]], panel: list[str], attr: str, tcodes: Optional[set] = None) -> dict[str, Any]:
    """Alpha for mode / status / qualifier over units extracted by all models."""
    cats: dict[str, int] = {}
    cols = []
    for u in units.values():
        if len(u["models"]) != len(panel) or (tcodes and u["tcode"] not in tcodes):
            continue
        vals = []
        for m in panel:
            pm = u["per_model"][m]
            if attr == "mode":
                v = majority(pm["modes"], CONSERVATIVE_MODE_ORDER)[0]
            else:
                v = majority([a.get(attr) or "none" for a in pm["attrs"]], [])[0]
            vals.append(cats.setdefault(v, len(cats)))
        cols.append(vals)
    if not cols:
        return {"n_units": 0, "alpha": float("nan")}
    mat = np.array(cols, dtype=float).T
    return {"n_units": len(cols), "alpha": _alpha(mat), "categories": list(cats)}


# ---------------------------------------------------------------------------
# regex recall
# ---------------------------------------------------------------------------
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_SCRIPT = re.compile(r"\b(npm|pnpm|yarn|bun)\s+(?:run(?:-script)?\s+)?([\w:./@-]+)")
_MAKE = re.compile(r"\bmake\s+([\w.-]+)")
_COMPOSER = re.compile(r"\bcomposer\s+run(?:-script)?\s+([\w:.-]+)")


def regex_candidates(text: str, manifest_path: str) -> dict[str, set[str]]:
    """Deterministic T1 and T5 candidates from code spans and fenced code lines."""
    t1: set[str] = set()
    t5: set[str] = set()
    code_texts = []
    in_fence = False
    for ln in text.splitlines():
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            code_texts.append(re.sub(r"^\s*[$>]\s*", "", ln))
        for m in _CODE_SPAN.finditer(ln):
            code_texts.append(m.group(1))
            tok = m.group(1).strip()
            if (("/" in tok) or N.FILE_EXT.search(tok)) and " " not in tok:
                npth, _ = N.normalize_path(tok, manifest_path)
                if npth is not None and ("/" in npth["path"] or N.FILE_EXT.search(npth["path"])):
                    t1.add(npth["path"])
    for ct in code_texts:
        for m in _SCRIPT.finditer(ct):
            tgt = m.group(2).rstrip(".,;:)")
            if tgt and tgt.lower() not in N.BUILTINS | {"-"} and not tgt.startswith("-"):
                t5.add("script|" + tgt)
        for m in _MAKE.finditer(ct):
            if m.group(1).lower() not in N.BUILTINS:
                t5.add("make|" + m.group(1))
        for m in _COMPOSER.finditer(ct):
            t5.add("composer|" + m.group(1))
    return {"T1": t1, "T5": t5}


def regex_recall(cfg: dict[str, Any], cons: list[dict[str, Any]], units: dict[tuple[str, str], dict[str, Any]],
                 limit_repos: Optional[int] = None) -> dict[str, Any]:
    corpus = C.load_corpus(cfg)
    if limit_repos:
        corpus = corpus[:limit_repos]
    cons_by_repo: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for r in cons:
        cons_by_repo[r["repo"]][r["tcode"]][r["key_str"]] = r
    union_by_repo: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for (repo, ks), u in units.items():
        union_by_repo[repo][u["tcode"]].add(ks)
    out: dict[str, Any] = {}
    for tcode in ("T1", "T5"):
        n = cap_cons = cap_union = noasserts = 0
        misses = []
        for e in corpus:
            cand = regex_candidates(e["claude_md_content"], e.get("manifest_path") or "CLAUDE.md")[tcode]
            for c in cand:
                ks = f"{tcode}|{c}" if tcode == "T1" else f"{tcode}|{c}"
                n += 1
                cr = cons_by_repo[e["full_name"]][tcode].get(ks)
                tcr = None
                # a consensus unit counts only with >= wp2 min votes; consensus rows carry all votes >= 1
                if cr is not None and cr["votes"] >= 2:
                    cap_cons += 1
                    if cr["mode"] != "asserts_existing":
                        noasserts += 1
                elif ks in union_by_repo[e["full_name"]][tcode]:
                    cap_union += 1
                else:
                    if len(misses) < 40:
                        misses.append([e["full_name"], c])
        out[tcode] = {"n_candidates": n, "captured_by_consensus_ge2": cap_cons,
                      "of_which_mode_not_asserting": noasserts, "captured_union_only": cap_union,
                      "missed": n - cap_cons - cap_union,
                      "recall_consensus": cap_cons / n if n else float("nan"),
                      "recall_union": (cap_cons + cap_union) / n if n else float("nan"),
                      "sample_missed": misses}
    return out


def agreement_report(cfg: dict[str, Any], rows: list[dict[str, Any]], units_pre: dict[tuple[str, str], dict[str, Any]],
                     panel: list[str]) -> dict[str, Any]:
    sets: dict[str, dict[str, set]] = {m: defaultdict(set) for m in panel}
    for (repo, ks), u in units_pre.items():
        for m in u["models"]:
            sets[m][repo].add((u["tcode"], ks))
    seed = cfg["seeds"]["bootstrap"]
    nb = cfg["bootstrap"]["agreement_resamples"]
    mat, owner = presence_matrix(units_pre, panel, [])
    rep: dict[str, Any] = {"panel": panel, "n_units_union": len(owner)}
    rep["pairwise_f1_overall"] = pairwise_f1(sets, panel)
    rep["pairwise_f1_by_type"] = {t: pairwise_f1(sets, panel, t) for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]}
    f1s = [v for v in rep["pairwise_f1_overall"].values() if v == v]
    rep["pairwise_f1_mean"] = float(np.mean(f1s)) if f1s else float("nan")
    rep["alpha_presence"] = {"alpha": _alpha(mat), "bootstrap_ci95": bootstrap_alpha(mat, owner, nb, seed)}
    rep["alpha_presence_by_type"] = {}
    for t in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        m_t, o_t = presence_matrix(units_pre, panel, [], t)
        rep["alpha_presence_by_type"][t] = {"n_units": len(o_t), "alpha": _alpha(m_t) if len(o_t) else float("nan")}
    # line-level unitization (every manifest line is a unit; 1 = the model proposed >=1 accepted claim on it)
    corpus = C.load_corpus(cfg)
    nlines = {e["full_name"]: len(e["claude_md_content"].replace("\r\n", "\n").split("\n")) for e in corpus}
    marks = {m: set() for m in panel}
    for r in rows:
        if r["status"] == "accepted":
            marks[r["model"]].add((r["repo"], int(r["line"])))
    cols, own = [], []
    for repo, n in nlines.items():
        for ln in range(1, n + 1):
            cols.append([1.0 if (repo, ln) in marks[m] else 0.0 for m in panel])
            own.append(repo)
    lm = np.array(cols).T
    rep["alpha_line_level"] = {"n_units": len(own), "alpha": _alpha(lm),
                               "bootstrap_ci95": bootstrap_alpha(lm, own, max(100, nb // 5), seed + 5),
                               "note": "every manifest line is a unit, so agreement on 'no claim' counts: an upper bound; "
                                       "the union-of-keys alpha above is the lower bound"}
    rep["alpha_mode"] = attribute_alpha(units_pre, panel, "mode")
    rep["alpha_status_T2"] = attribute_alpha(units_pre, panel, "status", {"T2"})
    rep["alpha_qualifier_T6"] = attribute_alpha(units_pre, panel, "qualifier", {"T6"})
    rep["caveat"] = ("Presence alpha codes union-of-keys presence per model; it approximates unitizing alpha_u and is "
                     "deflated by singleton units, so pairwise F1 is reported alongside it.")
    votes = Counter(len(u["models"]) for u in units_pre.values())
    rep["units_by_votes"] = {str(k): v for k, v in sorted(votes.items())}
    return rep
