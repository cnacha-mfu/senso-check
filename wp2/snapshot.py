"""WP2 snapshot resolution and blobless cloning (Paper 1, RQ2; design section 7).

For every corpus manifest, find the commit S at which the cached manifest text was the
committed version (newest first-parent commit at or before the corpus date T_upper whose
manifest equals the cached text, extended until the next manifest change). The manifest
and the repository are then examined at the same commit S. HEAD of 2026-10 is not used.

Usage:
    python wp2_consistency.py snapshot [--only owner/name]
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from . import common as C
from .gitutil import ENV, GitError, commit_epoch, git, show_blob

logger = logging.getLogger("wp2.snapshot")


def norm_text(b: bytes | str) -> str:
    s = b.decode("utf-8", "replace") if isinstance(b, bytes) else b
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def clone_dir(cfg: dict[str, Any], entry: dict[str, Any]) -> Path:
    return Path(cfg["clone_dir"]) / entry["slug"]


def clone_blobless(cfg: dict[str, Any], entry: dict[str, Any]) -> tuple[bool, str]:
    """Blobless single-branch no-checkout clone (never ``populate.clone_repo``, which pulls)."""
    d = clone_dir(cfg, entry)
    if (d / ".git").exists() or (d / "HEAD").exists():
        return True, "exists"
    d.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--filter=blob:none", "--no-checkout", "--single-branch",
           "-b", entry.get("default_branch") or "HEAD",
           f"https://github.com/{entry['full_name']}.git", str(d)]
    if not entry.get("default_branch"):
        cmd = [c for c in cmd if c not in ("-b", "HEAD")]
    p = subprocess.run(cmd, capture_output=True, env=ENV, timeout=3600)
    if p.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        return False, p.stderr.decode("utf-8", "replace")[-300:]
    return True, "cloned"


def manifest_candidates(cfg: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    cands = []
    if entry.get("manifest_path"):
        cands.append(entry["manifest_path"])
    for f in cfg["snapshot"]["manifest_fallbacks"]:
        if f not in cands:
            cands.append(f)
    return cands


def resolve(cfg: dict[str, Any], entry: dict[str, Any], max_versions: int = 60) -> dict[str, Any]:
    """Resolve S for one repo (clone must exist)."""
    d = clone_dir(cfg, entry)
    target = norm_text(entry["claude_md_content"])
    t_upper = entry["t_upper"]
    out: dict[str, Any] = {"repo": entry["full_name"], "slug": entry["slug"], "t_upper": t_upper,
                           "default_branch": entry.get("default_branch"), "snapshot_status": "no_match",
                           "manifest_path": None, "S": None, "tried": []}
    out["head_sha"] = str(git(d, "rev-parse", "HEAD")).strip()
    for path in manifest_candidates(cfg, entry):
        revs = [x for x in str(git(d, "rev-list", "--first-parent", f"--before={t_upper}", "HEAD", "--", path,
                                   check=False)).split() if x]
        out["tried"].append({"path": path, "n_versions": len(revs)})
        for i, c in enumerate(revs[:max_versions]):
            blob = show_blob(d, c, path)
            if blob is None:
                continue
            if norm_text(blob) == target:
                head_s = str(git(d, "rev-list", "--first-parent", f"--before={t_upper}", "-n", "1", "HEAD")).strip()
                if i == 0:
                    status, s = "exact_at_T_upper", head_s
                else:
                    nxt = revs[i - 1]  # next manifest change after V
                    status = "version_match"
                    s = str(git(d, "rev-parse", f"{nxt}^1")).strip()
                out.update(snapshot_status=status, manifest_path=path, S=s, V=c,
                           next_change=(revs[i - 1] if i else None),
                           S_epoch=commit_epoch(d, s), V_epoch=commit_epoch(d, c),
                           manifest_version_rank=i)
                return out
    return out


def run_one(cfg: dict[str, Any], entry: dict[str, Any], force: bool = False) -> dict[str, Any]:
    res_dir = C.results_dir(cfg)
    p = res_dir / "snapshots" / f"{entry['slug']}.json"
    if p.exists() and not force:
        return C.read_json(p)
    ok, msg = clone_blobless(cfg, entry)
    if not ok:
        rec = {"repo": entry["full_name"], "slug": entry["slug"], "cloneable": False, "clone_error": msg,
               "snapshot_status": "not_cloneable", "S": None}
    else:
        try:
            rec = {"cloneable": True, **resolve(cfg, entry)}
        except (GitError, subprocess.TimeoutExpired) as exc:
            rec = {"repo": entry["full_name"], "slug": entry["slug"], "cloneable": True,
                   "snapshot_status": "resolve_error", "error": str(exc)[:300], "S": None}
    C.write_json(p, rec)
    logger.info("%s: %s S=%s", entry["full_name"], rec["snapshot_status"], (rec.get("S") or "")[:10])
    return rec


def run_all(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False) -> list[dict[str, Any]]:
    corpus = C.load_corpus(cfg)
    if only:
        corpus = [e for e in corpus if e["full_name"] == only]
    with ThreadPoolExecutor(max_workers=cfg["snapshot"]["clone_workers"]) as ex:
        recs = list(ex.map(lambda e: run_one(cfg, e, force), corpus))
    from collections import Counter
    logger.info("snapshot statuses: %s", dict(Counter(r["snapshot_status"] for r in recs)))
    return recs
