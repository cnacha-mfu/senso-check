"""A read-only view of a repository at one commit, for WP2 measurement (Paper 1, RQ2).

``RepoAt`` loads the tracked tree of a (blobless) clone at ``rev`` once, indexes files,
directories and gitlinks, prefetches the blobs that measurement needs in batches
(``git fetch --filter=blob:none --stdin``), and reads them through ``git cat-file --batch``.
Used by measure.py and depparse.py. recheck.py deliberately does not use it.

Usage:
    imported by wp2/measure.py, wp2/depparse.py, wp2/step_measure.py, wp2/step_drift.py
"""

from __future__ import annotations

import bisect
import logging
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .gitutil import ENV, git

logger = logging.getLogger("wp2.repo")

MAX_BLOB = 2_000_000


class RepoAt:
    """Tracked tree of ``repo_dir`` at ``rev``."""

    def __init__(self, repo_dir: Path, rev: str) -> None:
        self.dir = Path(repo_dir)
        self.rev = rev
        out = git(self.dir, "ls-tree", "-r", "-z", "--full-tree", rev, timeout=1800)
        self.files: dict[str, str] = {}
        self.sizes: dict[str, int] = {}
        self.gitlinks: set[str] = set()
        for rec in str(out).split("\0"):
            if not rec:
                continue
            meta, path = rec.split("\t", 1)
            mode, typ, sha = meta.split()
            size = "0"  # sizes need blobs (-l would fetch every blob of a blobless clone)
            if typ == "commit":
                self.gitlinks.add(path)
            else:
                self.files[path] = sha
                self.sizes[path] = int(size) if size.isdigit() else 0
        self.sorted_paths = sorted(self.files)
        self.dirs: dict[str, list[str]] = defaultdict(list)  # last segment -> dir paths
        seen: set[str] = set()
        for p in self.files:
            parts = p.split("/")
            for i in range(1, len(parts)):
                d = "/".join(parts[:i])
                if d not in seen:
                    seen.add(d)
                    self.dirs[parts[i - 1]].append(d)
        self.dir_set = seen
        self.by_basename: dict[str, list[str]] = defaultdict(list)
        for p in self.files:
            self.by_basename[p.rsplit("/", 1)[-1]].append(p)
        self.epoch = int(str(git(self.dir, "show", "-s", "--format=%ct", rev)).strip())
        self._blobs: dict[str, bytes] = {}

    # -- tree queries ---------------------------------------------------------
    def count_under(self, prefix: str) -> int:
        """Tracked files whose path starts with ``prefix + '/'``."""
        lo = bisect.bisect_left(self.sorted_paths, prefix + "/")
        hi = bisect.bisect_left(self.sorted_paths, prefix + "/￿")
        return hi - lo

    def is_file(self, p: str) -> bool:
        return p in self.files

    def is_dir(self, p: str) -> bool:
        return p in self.dir_set

    def in_gitlink(self, p: str) -> bool:
        parts = p.split("/")
        return any("/".join(parts[:i]) in self.gitlinks for i in range(1, len(parts) + 1))

    # -- blobs ---------------------------------------------------------------
    def _missing(self, shas: list[str]) -> list[str]:
        if not shas:
            return []
        p = subprocess.run(["git", "-C", str(self.dir), "cat-file", "--batch-check"], input="\n".join(shas) + "\n",
                           capture_output=True, text=True, env={**ENV, "GIT_NO_LAZY_FETCH": "1"})
        return [ln.split()[0] for ln in p.stdout.splitlines() if ln.endswith(" missing")]

    def prefetch(self, paths: Iterable[str]) -> None:
        """Fetch the blobs of ``paths`` in batches (blobless clone)."""
        shas = [self.files[p] for p in paths if p in self.files and self.files[p] not in self._blobs
                and self.sizes.get(p, 0) <= MAX_BLOB]
        shas = list(dict.fromkeys(shas))
        miss = self._missing(shas)
        for i in range(0, len(miss), 1500):
            batch = miss[i: i + 1500]
            p = subprocess.run(
                ["git", "-C", str(self.dir), "-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin", "--no-tags",
                 "--no-write-fetch-head", "--filter=blob:none", "--stdin"],
                input="\n".join(batch) + "\n", capture_output=True, text=True, env=ENV, timeout=1800)
            if p.returncode != 0:
                logger.warning("%s: blob prefetch failed: %s", self.dir.name, p.stderr[-200:])
        self._load(shas)

    def _load(self, shas: list[str]) -> None:
        todo = [s for s in shas if s not in self._blobs]
        if not todo:
            return
        p = subprocess.run(["git", "-C", str(self.dir), "cat-file", "--batch"], input=("\n".join(todo) + "\n").encode(),
                           capture_output=True, env={**ENV, "GIT_NO_LAZY_FETCH": "1"}, timeout=1800)
        buf, pos = p.stdout, 0
        for s in todo:
            nl = buf.find(b"\n", pos)
            if nl < 0:
                break
            header = buf[pos:nl].split()
            pos = nl + 1
            if len(header) == 3 and header[1] == b"blob":
                size = int(header[2])
                self._blobs[s] = buf[pos: pos + size]
                pos += size + 1
            # "missing" lines have no body

    def read(self, path: str) -> Optional[str]:
        """Text of a tracked file (decoded, BOM stripped) or None if absent / oversize / unavailable."""
        sha = self.files.get(path)
        if sha is None or self.sizes.get(path, 0) > MAX_BLOB:
            return None
        if sha not in self._blobs:
            self.prefetch([path])
        b = self._blobs.get(sha)
        if b is None:
            return None
        return b.decode("utf-8-sig", "replace")
