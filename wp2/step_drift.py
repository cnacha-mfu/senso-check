"""WP2 step: drift origin of Supported and Contradicted claims (Paper 1, RQ2; design section 6.3).

"Does history contradict the manifest, or was the claim never true?" For each Supported or Contradicted
T1/T3/T4/T5/T6 claim the introduction commit I of the claim's manifest line is found
(``git log -S <line> --reverse`` on the manifest, first-parent, up to S). The claim is measured again at I
and asserted as a copy (``.../intro``) linked with ``seo:introducedAs``; HermiT then infers
``DriftedClaim`` (contradicted at S, supported at I) and ``BornWrongClaim`` (contradicted at both).
For drifted claims the commit that made the claim false is located by bisection over the first-parent
commits between I and S, using the independent re-check as the predicate (assumes one flip).

Usage:
    python wp2_consistency.py drift [--only owner/name]
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import yaml

from . import common as C
from . import measure as M
from . import recheck as R
from .gitutil import ENV
from .repo import RepoAt
from .step_measure import measure_claim
from .step_reason import ensure_tbox, tbox_sha

logger = logging.getLogger("wp2.step_drift")
DRIFT_TYPES = {"T1", "T3", "T4", "T5", "T6"}


def find_intro(d: Path, manifest_path: str, S: str, line_text: str) -> Optional[str]:
    """First first-parent commit up to S in which the manifest line appears (pickaxe on the manifest)."""
    needle = line_text.strip()
    if len(needle) < 6:
        return None
    p = subprocess.run(["git", "-C", str(d), "log", S, "--first-parent", "--reverse", "--format=%H", "-S", needle,
                        "--", manifest_path], capture_output=True, env=ENV, timeout=900)
    out = p.stdout.decode().split()
    return out[0] if p.returncode == 0 and out else None


def flip_commit(d: Path, I: str, S: str, claim: dict[str, Any], thr: dict[str, Any], sigs: dict[str, Any],
                vsrc: dict[str, Any]) -> Optional[tuple[str, int]]:
    """First commit in (I, S] where the independent re-check calls the claim Contradicted (binary search)."""
    p = subprocess.run(["git", "-C", str(d), "rev-list", "--first-parent", "--reverse", f"{I}..{S}"],
                       capture_output=True, env=ENV, timeout=300)
    commits = p.stdout.decode().split()
    if not commits:
        return None

    def contradicted(c: str) -> bool:
        g = R.Git(d, c)
        g.prefetch_interesting()
        v, _ = R.recheck_claim(g, {"tcode": claim["tcode"], "attrs": claim["attrs"], "votes": 3}, thr, sigs, vsrc, {}, {})
        return v == "ContradictedClaim"

    lo, hi = 0, len(commits) - 1  # P(commits[hi]) is assumed true (S)
    if not contradicted(commits[hi]):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if contradicted(commits[mid]):
            hi = mid
        else:
            lo = mid + 1
    c = commits[lo]
    ct = int(subprocess.run(["git", "-C", str(d), "show", "-s", "--format=%ct", c], capture_output=True, env=ENV).stdout.decode().strip() or 0)
    st = int(subprocess.run(["git", "-C", str(d), "show", "-s", "--format=%ct", S], capture_output=True, env=ENV).stdout.decode().strip() or 0)
    return c, max(0, (st - ct) // 86400)


def drift_repo(cfg: dict[str, Any], entry: dict[str, Any], tbox: Path, force: bool) -> Optional[dict[str, Any]]:
    from ontology.reasoning.abox import DATA_BASE, _safe
    from ontology.reasoning.claims_abox import ClaimsAboxWriter, claim_id
    from ontology.reasoning.reasoner import CLAIMS_MODULES, HermitReasoner

    res = C.results_dir(cfg)
    slug = entry["slug"]
    mp, vp = res / "measurements" / f"{slug}.json", res / "verdicts" / f"{slug}.json"
    if not mp.exists() or not vp.exists():
        return None
    out_p = res / "drift" / f"{slug}.json"
    meas, ver = C.read_json(mp), C.read_json(vp)
    sig = C.sha256_text(meas["input_sha"] + ver["tbox_sha256"] + ver["abox_sha256"])
    if out_p.exists() and not force and C.read_json(out_p).get("input_sha") == sig:
        return C.read_json(out_p)
    if not ver.get("consistent"):
        return None
    d = Path(cfg["clone_dir"]) / slug
    lines = entry["claude_md_content"].replace("\r\n", "\n").split("\n")
    thr = R.load_thr()
    sigs = yaml.safe_load((C.EXP_DIR / "configs" / "wp2_tool_signatures.yaml").read_text(encoding="utf-8"))
    vsrc = yaml.safe_load((C.EXP_DIR / "configs" / "wp2_version_sources.yaml").read_text(encoding="utf-8"))["subjects"]
    ctxs: dict[str, M.Ctx] = {}
    elig = [c for c in meas["claims"] if c["tcode"] in DRIFT_TYPES and c["headline"]
            and int(c["votes"]) >= int(thr["wp2_consensus_min_votes"])
            and (ver["claims"].get(c["key_str"], {}).get("verdicts") or [None])[0] in ("SupportedClaim", "ContradictedClaim")]
    # rebuild the full ABox (original claims) plus intro copies
    from .step_measure import build_abox  # noqa: F401  (original ABox text is reused from file)

    w = ClaimsAboxWriter()
    w.lines = (res / "abox" / f"{slug}.nt").read_text(encoding="utf-8").splitlines()
    w._bn = 10**6
    rows = []
    for c in elig:
        lt = lines[c["line"] - 1] if 0 < c["line"] <= len(lines) else ""
        I = find_intro(d, meas["manifest_path"], meas["S"], lt)
        row = {"key": c["key_str"], "tcode": c["tcode"], "intro_commit": I, "intro_resolved": I is not None,
               "verdict_at_S": ver["claims"][c["key_str"]]["verdicts"][0], "leaf_at_S": (ver["claims"][c["key_str"]]["leaves"] or [None])[0]}
        if I and I != meas["S"]:
            if I not in ctxs:
                ctxs[I] = M.Ctx(RepoAt(d, I))
            m = measure_claim(ctxs[I], {"repo": meas["repo"], "tcode": c["tcode"], "attrs": c["attrs"], "key_str": c["key_str"]})
            props = {"claimQuote": c["quote"][:300], "claimSubject": c["subject"], "claimKey": c["key_str"],
                     "claimLine": c["line"], "extractionVotes": c["votes"], **m["props"]}
            main_iri = f"{DATA_BASE}claim/{_safe(meas['repo'])}/{claim_id(c['key_str'])}"
            intro_iri = w.claim(meas["repo"], c["key_str"], c["types"], props, suffix="/intro")
            for i, e in enumerate(m["evidences"]):
                pass
            eis = [w.evidence(intro_iri, i, e) for i, e in enumerate(m["evidences"])]
            w.close_evidence(intro_iri, eis)
            w.introduced_as(main_iri, intro_iri)
            row["intro_iri"] = intro_iri
        elif I == meas["S"]:
            row["intro_is_snapshot"] = True
        rows.append((c, row))
    nt = w.ntriples()
    p_nt = res / "abox_drift"
    p_nt.mkdir(parents=True, exist_ok=True)
    (p_nt / f"{slug}.nt").write_text(nt, encoding="utf-8")
    r = HermitReasoner(tbox, modules=CLAIMS_MODULES, temp_dir=Path(f"/tmp/wp2_drift_{os.getpid()}"))
    inf = r.infer(nt)
    out_rows = []
    for c, row in rows:
        main_iri = f"{DATA_BASE}claim/{_safe(meas['repo'])}/{claim_id(c['key_str'])}"
        labs = set(inf.labels.get(main_iri, ())) if inf.consistent else set()
        if "intro_iri" in row and inf.consistent:
            il = set(inf.labels.get(row["intro_iri"], ()))
            from ontology.modules.claims_module import VERDICTS
            row["verdict_at_intro"] = (sorted(il & set(VERDICTS)) or ["none"])[0]
        row["drifted"] = "DriftedClaim" in labs
        row["born_wrong"] = "BornWrongClaim" in labs
        if row["drifted"]:
            fc = flip_commit(d, row["intro_commit"], meas["S"], c, thr, sigs, vsrc)
            if fc:
                row["flip_commit"], row["days_false_at_S"] = fc
        out_rows.append(row)
    rec = {"repo": meas["repo"], "slug": slug, "input_sha": sig, "consistent": inf.consistent, "rows": out_rows,
           "tbox_sha256": ver["tbox_sha256"]}
    C.write_json(out_p, rec)
    logger.info("%s: %d drift rows (%d drifted, %d born wrong)", meas["repo"], len(out_rows),
                sum(r["drifted"] for r in out_rows), sum(r["born_wrong"] for r in out_rows))
    return rec


def run(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False) -> int:
    res = C.results_dir(cfg)
    tbox = ensure_tbox("baseline")
    corpus = [e for e in C.load_corpus(cfg) if not only or e["full_name"] == only]
    with ThreadPoolExecutor(max_workers=3) as ex:
        recs = [x for x in ex.map(lambda e: drift_repo(cfg, e, tbox, force), corpus) if x]
    rows = [r for rec in recs for r in rec["rows"]]
    by_type: dict[str, Any] = {}
    for t in sorted({r["tcode"] for r in rows}):
        rt = [r for r in rows if r["tcode"] == t]
        contra = [r for r in rt if r["verdict_at_S"] == "ContradictedClaim"]
        resolved = [r for r in contra if r["intro_resolved"] and r.get("verdict_at_intro")]
        days = [r["days_false_at_S"] for r in contra if r.get("days_false_at_S") is not None]
        by_type[t] = {"n_claims": len(rt), "n_contradicted": len(contra), "n_contradicted_intro_resolved": len(resolved),
                      "intro_unresolved": sum(1 for r in contra if not r["intro_resolved"]),
                      "drifted": sum(1 for r in contra if r["drifted"]), "born_wrong": sum(1 for r in contra if r["born_wrong"]),
                      "drifted_share_of_resolved": (sum(1 for r in resolved if r["drifted"]) / len(resolved)) if resolved else None,
                      "born_wrong_share_of_resolved": (sum(1 for r in resolved if r["born_wrong"]) / len(resolved)) if resolved else None,
                      "n_flip_located": len(days), "median_days_false_at_S": statistics.median(days) if days else None}
    contra_all = [r for r in rows if r["verdict_at_S"] == "ContradictedClaim"]
    res_all = [r for r in contra_all if r["intro_resolved"] and r.get("verdict_at_intro")]
    all_days = [r["days_false_at_S"] for r in contra_all if r.get("days_false_at_S") is not None]
    out = {"by_type": by_type, "overall": {
        "n_contradicted": len(contra_all), "n_intro_resolved": len(res_all),
        "drifted": sum(1 for r in contra_all if r["drifted"]), "born_wrong": sum(1 for r in contra_all if r["born_wrong"]),
        "drifted_share_of_resolved": (sum(1 for r in res_all if r["drifted"]) / len(res_all)) if res_all else None,
        "born_wrong_share_of_resolved": (sum(1 for r in res_all if r["born_wrong"]) / len(res_all)) if res_all else None,
        "median_days_false_at_S": statistics.median(all_days) if all_days else None, "n_flip_located": len(all_days)},
        "rows": rows,
        "caveat": "Drifted/born-wrong are inferred by HermiT from the two measurements; the flip commit uses the independent re-check as bisection predicate and assumes a single flip."}
    C.write_json(res / "drift.json", out)
    logger.info("drift overall: %s", out["overall"])
    return 0
