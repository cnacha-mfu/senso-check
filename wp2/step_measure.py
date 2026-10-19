"""WP2 step: measure consensus claims at the snapshot and write per-manifest ABoxes (Paper 1, RQ2).

For each repository with a resolved snapshot S, every consensus claim (extracted by at least one
model, majority mode ``asserts_existing``) is measured against the tree at S (wp2/measure.py,
wp2/depparse.py) and written to ``abox/{repo}.nt`` (ClaimsAboxWriter). ``extractionVotes`` is
asserted, so the consensus rule stays a TBox parameter. No verdict is asserted here.

Usage:
    python wp2_consistency.py measure [--only owner/name] [--force]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from . import common as C
from . import measure as M
from .repo import RepoAt

logger = logging.getLogger("wp2.step_measure")


def claim_types(r: dict[str, Any]) -> list[str]:
    """Asserted claim-type classes (extracted content, not verdicts)."""
    t, a = r["tcode"], r["attrs"]
    if t == "T1":
        return ["PathExistenceClaim"]
    if t == "T2":
        return ["InactiveStatusClaim"] if a.get("status_kind") == "inactive" else ["PathExistenceClaim", "ProtectedStatusClaim"]
    if t == "T3":
        return ["ToolUsageClaim", "PackageManagerClaim" if a["category"] == "package_manager" else "NonPackageManagerToolClaim"]
    if t == "T4":
        return ["DependencyClaim", "VersionedDependencyClaim" if a.get("major") is not None else "UnversionedDependencyClaim"]
    if t == "T5":
        return ["CommandClaim"]
    if t == "T6":
        return ["VersionClaim", "MinVersionClaim" if a.get("qualifier") == "min" else "ExactVersionClaim"]
    return ["ImportConstraintClaim"]


def is_headline(r: dict[str, Any]) -> bool:
    """Claims counted in headline rates: not T7 and not protected T2 (design section 7)."""
    return not (r["tcode"] == "T7" or (r["tcode"] == "T2" and r["attrs"].get("status_kind") == "protected"))


def measure_claim(ctx: M.Ctx, r: dict[str, Any]) -> dict[str, Any]:
    """Measurement props / evidences / meta for one consensus claim (no verdict)."""
    t, a = r["tcode"], r["attrs"]
    props: dict[str, Any] = {}
    evs: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    try:
        if t == "T1" or (t == "T2" and a.get("status_kind") == "protected"):
            props, _ = M.measure_path(ctx, a["path"], a.get("is_dir", False), a.get("glob", False), a.get("bracket", False))
            if (a.get("alt_path") and props["pathExactMatchCount"] == 0 and props["pathSuffixMatchCount"] == 0):
                # tree diagram with a wrapping root: the root may be the repository itself; use the root-dropped path
                # only if it resolves (a lookup, not a verdict)
                alt = M.measure_path(ctx, a["alt_path"], a.get("is_dir", False), a.get("glob", False), a.get("bracket", False))[0]
                if (alt["pathExactMatchCount"] > 0 or alt["pathSuffixMatchCount"] > 0 or alt.get("pathIgnoredCount") == 1
                        or alt.get("pathSubmoduleCount") == 1 or alt.get("pathEverTrackedCount", 0) >= 1):
                    props, meta["alt_path_used"] = alt, a["alt_path"]
        elif t == "T2":
            props, info = M.measure_path(ctx, a["path"], a.get("is_dir", False), a.get("glob", False), a.get("bracket", False))
            props = {k: v for k, v in props.items() if k in ("pathExactMatchCount", "pathSuffixMatchCount")}
            if info["resolved"]:
                ch = M.measure_churn(ctx, info["resolved"])
                meta["churn_truncated_300"] = ch.pop("_churn_truncated_300", 0)
                meta["resolved_path"] = info["resolved"]
                props.update(ch)
        elif t == "T3":
            props = M.measure_tool(ctx, a["tool"])
        elif t == "T4":
            props = M.measure_dependency(ctx, a["name"], a.get("major"))
        elif t == "T5":
            props = M.measure_command(ctx, a["kind"], a["target"], a["runner"])
        elif t == "T6":
            comps = tuple(int(x) for x in __import__("re").search(r"\d+(?:\.\d+){0,3}", a["version"]).group(0).split("."))
            props, evs = M.measure_version(ctx, a["subject_type"], (a["code"], a["precision"], comps))
        else:  # T7 (stretch, case studies only)
            props = M.measure_import(ctx, a["target"], a.get("scope"))
    except Exception as exc:  # noqa: BLE001 - one claim's failure is recorded, never a verdict
        logger.warning("measure error %s %s: %s", r["repo"], r["key_str"], exc)
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        props, evs = {}, []
    return {"props": props, "evidences": evs, "meta": meta}


def build_abox(repo: str, snap: dict[str, Any], claims: list[dict[str, Any]]) -> str:
    """ClaimsAboxWriter text for one manifest."""
    from ontology.reasoning.claims_abox import ClaimsAboxWriter

    w = ClaimsAboxWriter()
    doc = w.manifest(repo, {"snapshotCommit": snap["S"], "manifestPath": snap["manifest_path"]})
    for c in claims:
        props = {"claimQuote": c["quote"][:300], "claimSubject": c["subject"], "claimKey": c["key_str"],
                 "claimLine": c["line"], "extractionVotes": c["votes"], **c["m"]["props"]}
        iri = w.claim(repo, c["key_str"], c["types"], props, doc=doc, headline=c["headline"])
        eis = [w.evidence(iri, i, e) for i, e in enumerate(c["m"]["evidences"])]
        w.close_evidence(iri, eis)
    return w.ntriples()


def run_one(cfg: dict[str, Any], entry: dict[str, Any], cons: list[dict[str, Any]], force: bool) -> Optional[dict[str, Any]]:
    res = C.results_dir(cfg)
    snap_p = res / "snapshots" / f"{entry['slug']}.json"
    if not snap_p.exists():
        return None
    snap = C.read_json(snap_p)
    if snap.get("snapshot_status") not in ("exact_at_T_upper", "version_match"):
        return None
    out_p = res / "measurements" / f"{entry['slug']}.json"
    claims_in = [r for r in cons if r["repo"] == entry["full_name"] and r["mode"] == "asserts_existing"]
    sig = C.sha256_text(snap["S"] + "|" + "|".join(sorted(r["key_str"] + str(r["votes"]) for r in claims_in)))
    if out_p.exists() and not force and C.read_json(out_p).get("input_sha") == sig:
        return C.read_json(out_p)
    t0 = time.time()
    d = Path(cfg["clone_dir"]) / entry["slug"]
    repo = RepoAt(d, snap["S"])
    ctx = M.Ctx(repo)
    claims = []
    for r in claims_in:
        m = measure_claim(ctx, r)
        claims.append({"key_str": r["key_str"], "tcode": r["tcode"], "types": claim_types(r), "headline": is_headline(r),
                       "subject": r["subject"], "quote": r["quote"], "line": r["line"], "votes": r["votes"],
                       "models": r["models"], "attrs": r["attrs"], "mode_unanimous": r["mode_unanimous"], "m": m})
    nt = build_abox(entry["full_name"], snap, claims)
    (res / "abox").mkdir(parents=True, exist_ok=True)
    (res / "abox" / f"{entry['slug']}.nt").write_text(nt, encoding="utf-8")
    rec = {"repo": entry["full_name"], "slug": entry["slug"], "S": snap["S"], "manifest_path": snap["manifest_path"],
           "input_sha": sig, "n_claims": len(claims), "n_tree_files": len(repo.files),
           "n_dep_manifests_parsed": len(ctx.census.parsed) if ctx._census else None,
           "seconds": round(time.time() - t0, 1), "claims": claims}
    C.write_json(out_p, rec)
    logger.info("%s: %d claims measured in %.1fs (%d files)", entry["full_name"], len(claims), rec["seconds"], len(repo.files))
    return rec


def run(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False) -> int:
    res = C.results_dir(cfg)
    cons = C.read_jsonl(res / "claims" / "consensus.jsonl")
    corpus = C.load_corpus(cfg)
    if only:
        corpus = [e for e in corpus if e["full_name"] == only]
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda e: run_one(cfg, e, cons, force), corpus))
    return 0
