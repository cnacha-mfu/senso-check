"""WP2 step: Pellet justifications for non-trivial verdicts (Paper 1, RQ2; design section 4).

For every ContradictedClaim, ConflictedClaim and IndeterminateClaim, plus a seeded random sample of
Supported claims, runs ``pellet explain --instance`` (``ontology/reasoning/explain.py``) over the claims
TBox and the manifest's ABox and stores the parsed justification (axioms and facts used). A failed call is
logged with its error, never dropped silently.

Usage:
    python wp2_consistency.py explain [--only owner/name]
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

from . import common as C
from .step_reason import TBOX_ROOT, ensure_tbox, tbox_sha

logger = logging.getLogger("wp2.step_explain")


def select_targets(cfg: dict[str, Any], slugs: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Per manifest: claims to explain (verdict class + individual IRI)."""
    from ontology.reasoning.abox import DATA_BASE, _safe
    from ontology.reasoning.claims_abox import claim_id

    res = C.results_dir(cfg)
    always, supported = [], []
    for s in slugs:
        vp = res / "verdicts" / f"{s}.json"
        if not vp.exists():
            continue
        v = C.read_json(vp)
        if not v.get("consistent"):
            continue
        for key, rec in v["claims"].items():
            if not rec["verdicts"]:
                continue
            item = {"slug": s, "repo": v["repo"], "key": key, "verdict": rec["verdicts"][0], "leaf": (rec["leaves"] or [None])[0],
                    "iri": f"{DATA_BASE}claim/{_safe(v['repo'])}/{claim_id(key)}"}
            (supported if item["verdict"] == "SupportedClaim" else always).append(item)
            if item["verdict"] == "UnverifiableClaim":
                always.pop()  # not required by the design; Unverifiable claims are explained only if time allows
    random.Random(cfg["seeds"]["sample"]).shuffle(supported)
    chosen = always + supported[: cfg["explanations"]["random_supported"]]
    out: dict[str, list[dict[str, Any]]] = {}
    for it in chosen:
        out.setdefault(it["slug"], []).append(it)
    return out


def claim_subgraph(nt_lines: list[str], claim_iri: str) -> str:
    """N-Triples of one claim: its own triples, its evidence individuals and the blank nodes they reach.

    A justification of a claim's verdict depends only on that claim's facts, so a Pellet run over this
    small graph is a valid retry when the full-manifest run hits Pellet's "pruned node" instability.
    """
    subj_of: dict[str, list[str]] = {}
    for ln in nt_lines:
        sj = ln.split(" ", 1)[0]
        subj_of.setdefault(sj, []).append(ln)
    start = f"<{claim_iri}>"
    wanted = {start} | {sj for sj in subj_of if sj.startswith(f"<{claim_iri}/ev")}
    out, seen = [], set()
    frontier = list(wanted)
    while frontier:
        sj = frontier.pop()
        if sj in seen:
            continue
        seen.add(sj)
        for ln in subj_of.get(sj, []):
            out.append(ln)
            obj = ln.rsplit(" ", 2)[-2] if ln.endswith(" .") else ""
            if obj.startswith("_:") and obj not in seen:
                frontier.append(obj)
    return "\n".join(out) + "\n"


def explain_manifest(cfg: dict[str, Any], slug: str, items: list[dict[str, Any]], tbox: Path, force: bool) -> dict[str, Any]:
    from ontology.reasoning.explain import explain_instance, merged_rdfxml
    from ontology.reasoning.reasoner import CLAIMS_MODULES, PelletReasoner

    res = C.results_dir(cfg)
    out_p = res / "explanations" / f"{slug}.json"
    ab = res / "abox" / f"{slug}.nt"
    sig = C.sha256_text(tbox_sha(tbox) + C.sha256_bytes(ab.read_bytes()) + "|".join(sorted(i["key"] for i in items)))
    if out_p.exists() and not force and C.read_json(out_p).get("input_sha") == sig:
        return C.read_json(out_p)
    r = PelletReasoner(tbox, modules=CLAIMS_MODULES, temp_dir=Path(f"/tmp/wp2_explain_{os.getpid()}"))
    rdfxml = merged_rdfxml(r, ab)
    recs = []
    for it in items:
        ex = explain_instance(r, rdfxml, it["iri"], it["verdict"], max_explanations=1, timeout=cfg["explanations"]["pellet_timeout_s"])
        retried = False
        if ex.error and not ex.justifications:  # Pellet instability on the full ABox: retry on the claim's subgraph
            mini = claim_subgraph(ab.read_text(encoding="utf-8").splitlines(), it["iri"])
            ex2 = explain_instance(r, merged_rdfxml(r, mini), it["iri"], it["verdict"], max_explanations=1,
                                   timeout=cfg["explanations"]["pellet_timeout_s"])
            ex2.seconds += ex.seconds
            if ex2.error and not ex2.justifications and it.get("leaf"):  # last resort: explain the leaf class
                ex3 = explain_instance(r, merged_rdfxml(r, mini), it["iri"], it["leaf"], max_explanations=1,
                                       timeout=cfg["explanations"]["pellet_timeout_s"])
                ex3.seconds += ex2.seconds
                if ex3.justifications:
                    ex2 = ex3
            ex, retried = ex2, True
        j = ex.justifications[0] if ex.justifications else None
        recs.append({"key": it["key"], "verdict": it["verdict"], "leaf": it["leaf"], "seconds": round(ex.seconds, 2),
                     "retried_on_claim_subgraph": retried,
                     "error": ex.error, "returncode": ex.returncode, "n_justifications": len(ex.justifications),
                     "facts": [{"prop": f.prop, "value": f.value} for f in (j.facts if j else []) if f.prop != "type"],
                     "axioms": [a.text[:240] for a in (j.axioms if j else [])][:12],
                     "closures": (j.closures if j else [])})
    rec = {"repo": items[0]["repo"], "slug": slug, "input_sha": sig, "explanations": recs}
    C.write_json(out_p, rec)
    logger.info("%s: %d explanations, %d errors", rec["repo"], len(recs), sum(1 for x in recs if x["error"]))
    return rec


def run(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False) -> int:
    res = C.results_dir(cfg)
    tbox = ensure_tbox("baseline")
    slugs = [e["slug"] for e in C.load_corpus(cfg) if not only or e["full_name"] == only]
    targets = select_targets(cfg, slugs)
    t0 = time.time()
    items = list(targets.items())
    with ProcessPoolExecutor(max_workers=2) as ex:
        recs = list(ex.map(explain_manifest, [cfg] * len(items), [k for k, _ in items], [v for _, v in items],
                           [tbox] * len(items), [force] * len(items)))
    allx = [x for r in recs for x in r["explanations"]]
    summary = {"n_targets": len(allx), "n_with_justification": sum(1 for x in allx if x["n_justifications"] > 0),
               "n_errors": sum(1 for x in allx if x["error"]), "wall_s": round(time.time() - t0, 1),
               "mean_call_s": (sum(x["seconds"] for x in allx) / len(allx)) if allx else None,
               "by_verdict": {v: sum(1 for x in allx if x["verdict"] == v) for v in sorted({x["verdict"] for x in allx})},
               "contradicted_without_justification": [x["key"] for x in allx if x["verdict"] == "ContradictedClaim" and not x["n_justifications"]][:50],
               "errors_sample": [x["error"][:200] for x in allx if x["error"]][:10]}
    C.write_json(res / "explanations_summary.json", summary)
    logger.info("explain: %s", summary)
    return 0
