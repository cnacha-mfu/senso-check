"""Shared helpers for the WP2 pipeline (Paper 1, RQ2).

Config loading, corpus access, manifest chunking, result paths, JSON helpers.
No measurement or verdict logic lives here.

Usage:
    imported by wp2_consistency.py and wp2/*.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT  # configs/ lives at the repository root
CONFIG_PATH = EXP_DIR / "configs" / "wp2_consistency.yaml"

logger = logging.getLogger("wp2")


def load_env() -> None:
    """Load API keys from ./.env in the current directory (never printed); real environment variables win."""
    load_dotenv(Path.cwd() / ".env")


_ACTIVE_CONFIG: Path = CONFIG_PATH


def set_config_path(path: Path) -> None:
    """Make ``load_config()`` without arguments return this config (``--config``)."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = Path(path).resolve()


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load wp2_consistency.yaml (or the config selected with ``--config``)."""
    with open(path or _ACTIVE_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def results_dir(cfg: Optional[dict[str, Any]] = None) -> Path:
    """Results directory (created)."""
    cfg = cfg or load_config()
    p = PROJECT_ROOT / cfg["results_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def slug(full_name: str) -> str:
    """owner/name -> owner__name."""
    return full_name.replace("/", "__")


def model_slug(model: str) -> str:
    """Model id -> filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model.split("/", 1)[-1])


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically (indent 1, sorted keys off)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_corpus(cfg: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """The 80 JCSSE manifests with cached text, plus ``slug``, ``t_upper`` and ``phase1`` flags."""
    cfg = cfg or load_config()
    corpus = read_json(PROJECT_ROOT / cfg["corpus"])
    p1_path = PROJECT_ROOT / cfg["phase1_repos"]
    p1: set[str] = set()
    if p1_path.exists():
        raw = read_json(p1_path)
        items = raw if isinstance(raw, list) else raw.get("repos", raw)
        for it in items:
            if isinstance(it, dict):
                p1.add(it.get("full_name") or it.get("repo") or it.get("name") or "")
            elif isinstance(it, str):
                p1.add(it)
    out = []
    for e in corpus:
        is_p1 = (e["full_name"] in p1) or ("manifest_path" not in e)
        out.append({
            **e,
            "slug": slug(e["full_name"]),
            "phase1": is_p1,
            "t_upper": cfg["snapshot"]["t_upper_phase1" if is_p1 else "t_upper_new"],
            "manifest_path_guess": e.get("manifest_path"),
        })
    return out


# ---------------------------------------------------------------------------
# Chunking (design 5.2)
# ---------------------------------------------------------------------------
_HEADING = re.compile(r"^#{1,3}\s")


def split_sections(lines: list[str]) -> list[tuple[int, int]]:
    """Split at #/##/### headings (outside code fences); returns [(start, end)) line index ranges."""
    cuts = [0]
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and i > 0 and _HEADING.match(ln):
            cuts.append(i)
    cuts.append(len(lines))
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]


def chunk_manifest(text: str, max_chars: int = 6000) -> list[dict[str, Any]]:
    """Greedy-merge heading sections into chunks of at most ``max_chars``; oversized sections split on lines.

    Each chunk dict: idx, start_line (1-based global), end_line, plain (raw text), prompt (lines prefixed
    ``0123| ``). Deterministic.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    pieces: list[tuple[int, int]] = []
    for s, e in split_sections(lines):
        size = sum(len(x) + 1 for x in lines[s:e])
        if size <= max_chars:
            pieces.append((s, e))
            continue
        cur_s, cur = s, 0
        for i in range(s, e):
            ln = len(lines[i]) + 1
            if cur + ln > max_chars and i > cur_s:
                pieces.append((cur_s, i))
                cur_s, cur = i, 0
            cur += ln
        pieces.append((cur_s, e))
    chunks: list[tuple[int, int]] = []
    for s, e in pieces:
        if chunks:
            ps, pe = chunks[-1]
            if sum(len(x) + 1 for x in lines[ps:e]) <= max_chars:
                chunks[-1] = (ps, e)
                continue
        chunks.append((s, e))
    out = []
    for idx, (s, e) in enumerate(chunks):
        out.append({
            "idx": idx, "start_line": s + 1, "end_line": e,
            "plain": "\n".join(lines[s:e]),
            "prompt": "\n".join(f"{s + k + 1:04d}| {lines[s + k]}" for k in range(e - s)),
        })
    return out
