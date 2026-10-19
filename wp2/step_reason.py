"""WP2 step: HermiT verdicts per manifest, with an optional Pellet cross-check (Paper 1, RQ2).

Builds the baseline TBox (build_all with ontology/modules/thresholds.yaml) into a directory outside
the repository, then reasons over each ``abox/{repo}.nt`` with ``HermitReasoner(modules=CLAIMS_MODULES)``.
Every verdict in ``verdicts/{repo}.json`` is an entailment of seo-claims.owl over the measured facts.

Usage:
    python wp2_consistency.py reason [--only owner/name] [--force]
    python wp2_consistency.py reason --engine pellet
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from . import common as C

logger = logging.getLogger("wp2.step_reason")

TBOX_ROOT = Path(os.environ.get("SENSO_TBOX_DIR") or Path.home() / ".cache" / "senso-check" / "tbox")  # TBox build cache


def ensure_tbox(name: str = "baseline", thresholds_path: Optional[Path] = None) -> Path:
    """Build the TBox into TBOX_ROOT/name (once) and return the directory."""
    d = TBOX_ROOT / name
    if (d / "seo-claims.owl").exists():
        return d
    from ontology.modules.build_ontology import build_all
    from ontology.modules.claims_module import LEAF_VERDICT

    d.mkdir(parents=True, exist_ok=True)
    build_all(thresholds_path=thresholds_path, out_dir=d)
    C.write_json(d / "leaf_verdict.json", dict(LEAF_VERDICT))  # LEAF_VERDICT is filled only while building
    return d


def leaf_names(d: Path) -> set[str]:
    """Leaf class names of the TBox in ``d`` (written by ``ensure_tbox``)."""
    return set(C.read_json(d / "leaf_verdict.json"))


def tbox_sha(d: Path) -> str:
    h = hashlib.sha256()
    for n in ("seo-entities", "seo-claims"):
        h.update((d / f"{n}.owl").read_bytes())
    return h.hexdigest()


def labels_to_record(labels: dict[str, list[str]], iri_to_key: dict[str, str], leaves: set[str]) -> dict[str, Any]:
    from ontology.modules.claims_module import CHAINED, VERDICTS

    claims = {}
    for iri, key in iri_to_key.items():
        ls = set(labels.get(iri, ()))
        claims[key] = {"leaves": sorted(ls & leaves), "verdicts": sorted(ls & set(VERDICTS)),
                       "chained": sorted(ls & set(CHAINED))}
    return claims


def iri_key_map(slug: str, meas: dict[str, Any]) -> dict[str, str]:
    from ontology.reasoning.claims_abox import claim_id
    from ontology.reasoning.abox import DATA_BASE, _safe

    return {f"{DATA_BASE}claim/{_safe(meas['repo'])}/{claim_id(c['key_str'])}": c["key_str"] for c in meas["claims"]}


def reason_one(cfg: dict[str, Any], slug: str, tbox: Path, engine: str, force: bool, tag: str = "verdicts") -> Optional[dict[str, Any]]:
    from ontology.reasoning.abox import DATA_BASE, _safe
    from ontology.reasoning.reasoner import CLAIMS_MODULES, HermitReasoner, PelletReasoner

    res = C.results_dir(cfg)
    ab, mp = res / "abox" / f"{slug}.nt", res / "measurements" / f"{slug}.json"
    if not ab.exists() or not mp.exists():
        return None
    out_p = res / tag / f"{slug}.json"
    sha = tbox_sha(tbox)
    ab_sha = C.sha256_bytes(ab.read_bytes())
    if out_p.exists() and not force:
        old = C.read_json(out_p)
        if old.get("tbox_sha256") == sha and old.get("abox_sha256") == ab_sha:
            return old
    meas = C.read_json(mp)
    cls = HermitReasoner if engine == "hermit" else PelletReasoner
    r = cls(tbox, modules=CLAIMS_MODULES, temp_dir=Path(f"/tmp/wp2_reason_{os.getpid()}"))
    t0 = time.time()
    inf = r.infer(ab)
    keys = iri_key_map(slug, meas)
    doc_iri = f"{DATA_BASE}manifest/{_safe(meas['repo'])}"
    rec = {"repo": meas["repo"], "slug": slug, "engine": engine, "tbox_sha256": sha, "abox_sha256": ab_sha,
           "consistent": inf.consistent, "error": inf.error, "timings": inf.timings, "n_claims": len(keys),
           "seconds": round(time.time() - t0, 2), "abox_triples": inf.abox_triples}
    if inf.consistent:
        rec["claims"] = labels_to_record(inf.labels, keys, leaf_names(tbox))
        rec["document"] = sorted(set(inf.labels.get(doc_iri, ())) & {"ManifestWithContradiction"})
    C.write_json(out_p, rec)
    logger.info("%s [%s]: consistent=%s claims=%d reason_s=%.1f", meas["repo"], engine, inf.consistent, len(keys),
                inf.timings.get("reason_s", -1))
    return rec


def run(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False, engine: str = "hermit") -> int:
    res = C.results_dir(cfg)
    tbox = ensure_tbox("baseline")
    corpus = C.load_corpus(cfg)
    slugs = [e["slug"] for e in corpus if not only or e["full_name"] == only]
    tag = "verdicts" if engine == "hermit" else "verdicts_pellet"
    t0 = time.time()
    if engine == "pellet" and not os.environ.get("WP2_PELLET_CHILD"):
        # Pellet 2.3.1 can hang or crash on datatype-heavy TBoxes (WP1 finding): one child process per manifest with a timeout
        driver = Path(__file__).resolve().parents[1] / "wp2_consistency.py"
        repos = {e["slug"]: e["full_name"] for e in corpus}

        def child(slug: str) -> None:
            if not (res / "abox" / f"{slug}.nt").exists():
                return
            try:
                subprocess.run([sys.executable, str(driver), "reason", "--engine", "pellet", "--only", repos[slug]]
                               + (["--force"] if force else []), env={**os.environ, "WP2_PELLET_CHILD": "1"},
                               timeout=900, capture_output=True)
            except subprocess.TimeoutExpired:
                C.write_json(res / tag / f"{slug}.json", {"repo": repos[slug], "slug": slug, "engine": "pellet", "consistent": False,
                                                          "error": "timeout after 900 s", "timings": {}, "n_claims": 0})
        with ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(child, slugs))
        recs = [C.read_json(res / tag / f"{s}.json") for s in slugs if (res / tag / f"{s}.json").exists()]
        C.write_json(res / f"timings_{tag}.json", {"manifests": len(recs), "wall_s": round(time.time() - t0, 1)})
        return 0
    # processes, not threads: owlready2.World() creation is not thread safe ("dictionary changed size during iteration")
    with ProcessPoolExecutor(max_workers=2) as ex:
        recs = [x for x in ex.map(reason_one, [cfg] * len(slugs), slugs, [tbox] * len(slugs), [engine] * len(slugs),
                                  [force] * len(slugs), [tag] * len(slugs)) if x]
    C.write_json(res / f"timings_{tag}.json", {"manifests": len(recs), "wall_s": round(time.time() - t0, 1),
                                               "per_manifest_reason_s": {r["slug"]: r["timings"].get("reason_s") for r in recs}})
    bad = [r["repo"] for r in recs if not r["consistent"]]
    if bad:
        logger.error("inconsistent ABoxes: %s", bad)
    return 1 if bad else 0
