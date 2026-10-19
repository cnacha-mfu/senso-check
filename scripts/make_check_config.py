"""Build a one-repository corpus and config for the WP2 claim check (Paper 1, RQ2).

Writes ``corpus.json`` (the format ``wp2.common.load_corpus`` reads) and ``config.yaml``
(a copy of configs/wp2_consistency.yaml pointing at that corpus) so that
``wp2_consistency.py --config <out-dir>/config.yaml`` checks the manifest of any public
GitHub repository. The manifest is taken from ``--manifest`` or fetched from the default
branch; the snapshot upper bound is "now", so the check runs against the repo as it is
now, which requires the manifest text to match the committed file (use the committed
file, not an uncommitted edit).

Usage:
    python scripts/make_check_config.py --repo owner/name \
        [--manifest path/to/CLAUDE.md] [--manifest-path CLAUDE.md] [--out-dir DIR] [--local-path DIR]
    (default out-dir: ./senso-out/<owner__name>/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "wp2_consistency.yaml"
DEFAULT_ROOT = Path("senso-out")  # relative to the current directory
FALLBACKS = ["CLAUDE.md", ".claude/CLAUDE.md", "AGENTS.md", ".github/copilot-instructions.md"]

logger = logging.getLogger("make_check_config")


def _get(url: str, token: str | None, raw: bool = False) -> bytes:
    h = {"User-Agent": "senso-check"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if raw:
        h["Accept"] = "application/vnd.github.raw"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return r.read()


def build(repo: str, manifest: str | None = None, manifest_path: str | None = None,
          out_dir: str | None = None, local_path: str | None = None) -> Path:
    """Write corpus.json and config.yaml for ``repo``; returns the config path. Raises RuntimeError."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            from dotenv import dotenv_values
            token = dotenv_values(Path.cwd() / ".env").get("GITHUB_TOKEN")
        except ImportError:
            pass
    out = (Path(out_dir) if out_dir else DEFAULT_ROOT / repo.replace("/", "__")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(_get(f"https://api.github.com/repos/{repo}", token))
    except (urllib.error.URLError, OSError):
        if not local_path:
            raise
        logger.warning("GitHub metadata unavailable for %s; using local checkout only", repo)
        meta = {}
    cands = [manifest_path] if manifest_path else FALLBACKS
    text, mpath = None, None
    for c in cands:
        if manifest:
            text, mpath = Path(manifest).read_text(encoding="utf-8"), c
            break
        if local_path:
            f = Path(local_path) / c
            if f.is_file():
                text, mpath = f.read_text(encoding="utf-8"), c
                break
            continue
        try:
            text = _get(f"https://api.github.com/repos/{repo}/contents/{c}", token, raw=True).decode("utf-8")
            mpath = c
            break
        except urllib.error.HTTPError:
            continue
    if text is None or mpath is None:
        raise RuntimeError(f"no manifest found (tried {cands}); pass --manifest and --manifest-path")
    branch = meta.get("default_branch")
    if local_path and not branch:
        branch = subprocess.run(["git", "-C", local_path, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or None
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    entry = {
        "full_name": repo, "language": (meta.get("language") or "").lower(), "stars": meta.get("stargazers_count"),
        "created_at": meta.get("created_at"), "default_branch": branch,
        "description": meta.get("description"), "manifest_type": "claude_md" if "CLAUDE" in mpath else "other",
        "manifest_path": mpath, "claude_md_content": text, "claude_md_length": len(text),
    }
    (out / "corpus.json").write_text(json.dumps([entry], indent=2), encoding="utf-8")

    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["corpus"] = str(out / "corpus.json")
    cfg["results_dir"] = str(out / "results")
    cfg["clone_dir"] = str(out / "clones")
    cfg["phase1_repos"] = str(out / "no_phase1.json")   # absent file: nothing is a Phase 1 repo
    cfg["snapshot"]["t_upper_new"] = now
    cfg["snapshot"]["t_upper_phase1"] = now
    cfg["snapshot"]["clone_workers"] = 1
    cfg["extraction"]["concurrency_per_model"] = 1
    cfg["extraction"]["prompt_file"] = str(PROJECT_ROOT / cfg["extraction"]["prompt_file"])
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if local_path:  # the pipeline only reads git objects, so a symlink to the checkout replaces the clone
        link = out / "clones" / repo.replace("/", "__")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else None
        if not link.exists():
            link.symlink_to(Path(local_path).resolve())
    logger.info("manifest %s (%d chars), t_upper=%s", mpath, len(text), now)
    logger.info("wrote %s and %s", out / "corpus.json", out / "config.yaml")
    return out / "config.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="owner/name of a public GitHub repository")
    ap.add_argument("--manifest", default=None, help="local manifest file (default: fetch from GitHub)")
    ap.add_argument("--manifest-path", default=None, help="path of the manifest inside the repo (default: first found)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--local-path", default=None, help="existing git checkout to use instead of cloning from GitHub")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        build(a.repo, a.manifest, a.manifest_path, a.out_dir, a.local_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
