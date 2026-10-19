"""SENSO manifest claim check: one command for any public GitHub repository (Paper 1, RQ2).

Checks the factual claims in a repository's CLAUDE.md / AGENTS.md against the repository itself.
Builds a one-repo config (scripts/make_check_config.py), runs the WP2 steps
snapshot -> extract -> consensus -> measure -> reason (-> drift if a claim is contradicted) -> explain
through wp2_consistency.py, and prints one line per checked claim. The only LLM calls are the
claim extraction (NVIDIA NIM, key in .env). HermiT/Pellet need Java (JAVA_HOME or `java` on PATH).
Outputs go to ./senso-out/<owner__name>/ by default.

Usage:
    python senso_check.py --repo owner/name \\
        [--manifest FILE] [--manifest-path CLAUDE.md] [--local-path DIR] [--out DIR] \\
        [--format text|json] [--fail-on contradicted] [--skip-explain] [--all-claims]
Exit code: 0 ok; 1 if --fail-on contradicted and a claim is contradicted; 2 on a pipeline error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.make_check_config import build  # noqa: E402
from wp2 import common as C  # noqa: E402

logger = logging.getLogger("senso_check")
DRIVER = HERE / "wp2_consistency.py"
NOTE = {"ContradictedClaim": "contradicted", "SupportedClaim": "supported", "UnverifiableClaim": "unverifiable",
        "IndeterminateClaim": "indeterminate", "ConflictedClaim": "conflicted"}


def run_step(cfg_path: Path, args: list[str], timings: dict[str, float], log: Path) -> bool:
    env = dict(os.environ)
    if env.get("JAVA_HOME"):
        env["PATH"] = f"{env['JAVA_HOME']}/bin:{env.get('PATH', '')}"
    t = time.time()
    with open(log, "ab") as f:
        rc = subprocess.run([sys.executable, str(DRIVER), *args, "--config", str(cfg_path)], env=env,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    timings[args[0]] = round(time.time() - t, 1)
    logger.info("%s: %s in %.1fs", args[0], "ok" if rc == 0 else f"FAILED rc={rc} (see {log})", timings[args[0]])
    return rc == 0


def fact_of(c: dict[str, Any]) -> str:
    ev = c["m"].get("evidences") or []
    if ev:
        return "; ".join(", ".join(f"{k}={v}" for k, v in e.items()) for e in ev[:2])
    return ", ".join(f"{k}={v}" for k, v in c["m"].get("props", {}).items())


def collect(res: Path, slug: str, all_claims: bool) -> list[dict[str, Any]]:
    meas = C.read_json(res / "measurements" / f"{slug}.json")
    ver = C.read_json(res / "verdicts" / f"{slug}.json")
    drift = {r["key"]: r for r in C.read_json(res / "drift" / f"{slug}.json")["rows"]} if (res / "drift" / f"{slug}.json").exists() else {}
    expl = {e["key"]: e for e in C.read_json(res / "explanations" / f"{slug}.json")["explanations"]} if (res / "explanations" / f"{slug}.json").exists() else {}
    out = []
    for c in meas["claims"]:
        v = ver["claims"].get(c["key_str"], {})
        verdict = NOTE.get((v.get("verdicts") or [None])[0], "not_checked")
        if verdict == "not_checked" and not all_claims:
            continue
        rec = {"line": c["line"], "type": c["tcode"], "claim": c["quote"], "result": verdict,
               "leaf_class": (v.get("leaves") or [None])[0], "measured_fact": fact_of(c), "votes": c["votes"]}
        d = drift.get(c["key_str"])
        if verdict == "contradicted":
            rec["drifted"] = bool(d and d.get("drifted"))
            rec["born_wrong"] = bool(d and d.get("born_wrong"))
            rec["days_wrong"] = d.get("days_false_at_S") if d else None
            e = expl.get(c["key_str"])
            if e and e.get("facts"):
                rec["explanation_facts"] = e["facts"]
        out.append(rec)
    return sorted(out, key=lambda r: (r["line"], r["type"]))


def render_text(repo: str, meta: dict[str, Any], rows: list[dict[str, Any]], counts: dict[str, int]) -> str:
    L = [f"SENSO claim check: {repo} @ {meta['S'][:10]}  manifest {meta['manifest_path']}"]
    for r in rows:
        L.append(f"  L{r['line']:<4} {r['type']}  {r['result']:<13} {' '.join(r['claim'].split())[:70]}")
        if r["result"] == "contradicted":
            kind = "drifted" if r["drifted"] else "born wrong" if r["born_wrong"] else "origin unresolved"
            days = f", false for {r['days_wrong']} days" if r.get("days_wrong") is not None else ""
            L.append(f"         rule: {r.get('leaf_class')}  [{kind}{days}]")
            L.append(f"         fact: {r['measured_fact'][:150]}")
    L.append("summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="SENSO manifest claim check (Paper 1)")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--manifest", default=None, help="local manifest text (must equal the committed file)")
    ap.add_argument("--manifest-path", default=None, help="manifest path inside the repo (default: CLAUDE.md, AGENTS.md, ...)")
    ap.add_argument("--local-path", default=None, help="existing local git checkout (read-only use); default: blobless clone from GitHub")
    ap.add_argument("--out", default=None, help="output directory (default ./senso-out/<owner__name>)")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--fail-on", choices=["contradicted"], default=None)
    ap.add_argument("--skip-explain", action="store_true", help="skip the Pellet explanation step (saves about a minute)")
    ap.add_argument("--all-claims", action="store_true", help="also list claims that received no verdict")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stderr)
    C.load_env()
    t0 = time.time()
    try:
        cfg_path = build(a.repo, a.manifest, a.manifest_path, a.out, a.local_path)
    except (RuntimeError, OSError) as exc:
        logger.error("cannot build config: %s", exc)
        return 2
    C.set_config_path(cfg_path)
    cfg = C.load_config()
    res = C.results_dir(cfg)
    log = res.parent / "run.log"
    timings: dict[str, float] = {}
    panel = ",".join(f"{C.model_slug(m)}=1" for m in cfg["extraction"]["panel"])
    (res / "snapshots" / f"{a.repo.replace('/', '__')}.json").unlink(missing_ok=True)  # always re-resolve S (cheap)
    steps = [["snapshot"], ["extract", "--workers", panel], ["consensus"], ["measure"], ["reason"]]
    for s in steps:
        if not run_step(cfg_path, s, timings, log):
            return 2
    slug = a.repo.replace("/", "__")
    snap = C.read_json(res / "snapshots" / f"{slug}.json")
    if not snap.get("S"):
        logger.error("snapshot failed (%s): the --manifest text must equal the file committed in the repository", snap["snapshot_status"])
        return 2
    ver = C.read_json(res / "verdicts" / f"{slug}.json")
    if any((r["verdicts"] or [None])[0] == "ContradictedClaim" for r in ver["claims"].values()):
        run_step(cfg_path, ["drift"], timings, log)  # best effort: adds drifted/born-wrong and days
    if not a.skip_explain:
        run_step(cfg_path, ["explain"], timings, log)
    rows = collect(res, slug, a.all_claims)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    meta = {"S": snap["S"], "manifest_path": snap["manifest_path"]}
    total = round(time.time() - t0, 1)
    if a.format == "json":
        print(json.dumps({"repo": a.repo, "snapshot": snap["S"], "manifest_path": snap["manifest_path"], "summary": counts,
                          "claims": rows, "timings_s": timings, "total_s": total, "results_dir": str(res)}, indent=2))
    else:
        print(render_text(a.repo, meta, rows, counts))
        print(f"time: {total}s  results: {res}")
    return 1 if a.fail_on == "contradicted" and counts.get("contradicted") else 0


if __name__ == "__main__":
    sys.exit(main())
