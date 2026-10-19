"""
NIM model panel loader for v2 experiments.

Reads shared/llm_judge/nim_models_v2.yaml and returns the panel entries whose API key
is present in the environment. JSS-era runners keep their own MODEL_REGISTRY; new
runners (Paper 1 rebuild: WP2 claim extraction, WP5 agent runs) use this instead.

Used by: Paper 1 (JSEP rebuild) and later papers.

Usage:
    from shared.llm_judge.nim_registry import load_panel, make_client
    for entry in load_panel():
        client = make_client(entry)
        client.chat.completions.create(model=entry["model"], messages=[...])

    python shared/llm_judge/nim_registry.py   # list the panel and which keys are present
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "nim_models_v2.yaml"

logger = logging.getLogger(__name__)


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the full registry YAML (panel, reserve, timed_out)."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_panel(path: Path = REGISTRY_PATH, include_missing_keys: bool = False) -> list[dict[str, Any]]:
    """Return panel entries, each with ``base_url`` added; skip entries whose key is absent."""
    load_dotenv(Path.cwd() / ".env")
    registry = load_registry(path)
    panel = []
    for entry in registry["panel"]:
        entry = {**entry, "base_url": registry["nim_base_url"]}
        if not os.getenv(entry["key_env"]) and not include_missing_keys:
            logger.warning("Skipping %s: %s not set", entry["model"], entry["key_env"])
            continue
        panel.append(entry)
    return panel


def make_client(entry: dict[str, Any], timeout: float = 300.0) -> OpenAI:
    """OpenAI-compatible client for one panel entry."""
    return OpenAI(base_url=entry["base_url"], api_key=os.environ[entry["key_env"]], timeout=timeout)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for e in load_panel(include_missing_keys=True):
        present = "key present" if os.getenv(e["key_env"]) else "KEY MISSING"
        print(f"{e['org']:8} {e['model']:45} {e['key_env']:18} {present}")
