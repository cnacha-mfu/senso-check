"""Thin git subprocess helpers for WP2 (Paper 1, RQ2).

Used by snapshot.py, measure.py, depparse.py, triangulate and drift code. The independent
re-check (recheck.py) deliberately does NOT use this module (it has its own calls).

Usage:
    imported by wp2/*.py
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("wp2.git")

ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "LC_ALL": "C"}


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True, timeout: int = 900, text: bool = True) -> str | bytes:
    """Run ``git -C repo args``; returns stdout (str by default)."""
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=ENV, timeout=timeout)
    if check and p.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])} failed ({p.returncode}): {p.stderr.decode('utf-8', 'replace')[:300]}")
    if p.returncode != 0:
        return "" if text else b""
    return p.stdout.decode("utf-8", "replace") if text else p.stdout


def show_blob(repo: Path, rev: str, path: str) -> Optional[bytes]:
    """Content of ``rev:path`` or None if absent."""
    p = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, env=ENV, timeout=300)
    return p.stdout if p.returncode == 0 else None


def ls_tree(repo: Path, rev: str) -> list[tuple[str, str, str]]:
    """All tracked entries at rev as (mode, type, path); full recursive listing."""
    out = git(repo, "ls-tree", "-r", "-z", "--full-tree", rev, timeout=1800)
    rows = []
    for rec in str(out).split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode, typ, _sha = meta.split(" ")
        rows.append((mode, typ, path))
    return rows


def commit_epoch(repo: Path, rev: str) -> int:
    return int(str(git(repo, "show", "-s", "--format=%ct", rev)).strip())
