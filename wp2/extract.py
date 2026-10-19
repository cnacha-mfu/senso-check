"""WP2 claim extraction with the v2 NIM panel (Paper 1, RQ2; design section 5).

The only LLM step in WP2: each model proposes claim spans per manifest chunk.
Raw responses are checkpointed per (repo, chunk, model) under
``results/real_v2/wp2/extraction/raw/`` and reused on re-run while the prompt
sha256 matches. All validation (grounding, normalization, consensus) is
mechanical and lives in ``normalize.py`` / ``consensus.py``.

Usage:
    python wp2_consistency.py extract [--limit-repos N] [--workers nemotron=2,...]
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from . import common as C

logger = logging.getLogger("wp2.extract")

CLAIM_TYPES = {"path_exists", "path_status", "tool_used", "dependency", "command", "version", "import_rule"}
REPAIR_SUFFIX = "\n\nYour previous answer could not be parsed. Return ONLY one JSON object of the form {\"claims\": [...]}, with no other text and no markdown fence."


class RawClaim(BaseModel):
    """Schema of one extracted claim (design 5.2). Unknown fields are ignored."""

    model_config = ConfigDict(extra="ignore")
    type: str
    quote: Optional[str] = None
    line: Optional[int] = None
    subject: str
    path_components: Optional[list[str]] = None
    mode: str = "asserts_existing"
    status: Optional[str] = None
    tool_category: Optional[str] = None
    runner: Optional[str] = None
    ecosystem_hint: Optional[str] = None
    version: Optional[str] = None
    qualifier: Optional[str] = None
    scope: Optional[str] = None

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v: Any) -> Optional[int]:
        if v is None or str(v).strip() == "":
            return None
        return int(str(v).strip())

    @field_validator("version", "subject", "quote", "scope", mode="before")
    @classmethod
    def _str(cls, v: Any) -> Any:
        return None if v is None else str(v)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```[A-Za-z]*\s*|\s*```\s*$")


def parse_response(text: Optional[str]) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Parse a model response to a list of raw claim dicts.

    Strips ``<think>`` blocks and markdown fences and takes the outermost ``{...}``.
    Returns (claims, None) on success or (None, reason).
    """
    if not text or not text.strip():
        return None, "empty"
    t = _THINK.sub("", text)
    if "</think>" in t:  # unmatched opener stripped by the server
        t = t.split("</think>", 1)[1]
    t = t.strip()
    t = _FENCE.sub("", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None, "no_json_object"
    try:
        obj = json.loads(t[i: j + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_error:{str(exc)[:60]}"
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        return None, "no_claims_list"
    return [c for c in obj["claims"] if isinstance(c, dict)], None


def validate_claims(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Pydantic-validate claims; returns (valid dicts, number schema-invalid)."""
    good, bad = [], 0
    for it in items:
        try:
            rc = RawClaim(**it)
        except (ValidationError, ValueError, TypeError):
            bad += 1
            continue
        if rc.type not in CLAIM_TYPES:
            bad += 1
            continue
        good.append(rc.model_dump())
    return good, bad


def build_user_message(manifest_path: str, chunk: dict[str, Any], n_chunks: int) -> str:
    return (f"Instruction file: {manifest_path}\nChunk {chunk['idx'] + 1} of {n_chunks} "
            f"(lines {chunk['start_line']} to {chunk['end_line']})\n\n{chunk['prompt']}")


class ModelCaller:
    """One panel model with its own client, retry/backoff and statistics."""

    def __init__(self, entry: dict[str, Any], cfg: dict[str, Any]) -> None:
        from shared.llm_judge.nim_registry import make_client

        self.entry = entry
        self.model = entry["model"]
        self.cfg = cfg["extraction"]
        base = make_client(entry, timeout=self.cfg["timeout_s"])
        self.client = base.with_options(max_retries=0, timeout=self.cfg["timeout_s"])
        self.extra_body: Optional[dict[str, Any]] = None
        if "nemotron-3" in self.model:
            self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        elif "gpt-oss" in self.model:
            # default reasoning effort spends ~5k completion tokens and 100-270 s per chunk (probe 2026-10-06)
            self.extra_body = {"reasoning_effort": "low"}
        self.lock = threading.Lock()

    def once(self, system: str, user: str) -> dict[str, Any]:
        """One API call with transport retries (429/5xx/timeout) and exponential backoff."""
        last_err = ""
        for attempt in range(self.cfg["attempts"]):
            t0 = time.perf_counter()
            try:
                kw: dict[str, Any] = dict(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"])
                if self.extra_body:
                    kw["extra_body"] = self.extra_body
                r = self.client.chat.completions.create(**kw)
                msg = r.choices[0].message
                text = msg.content or ""
                u = r.usage
                return {"text": text, "finish_reason": r.choices[0].finish_reason,
                        "latency_s": time.perf_counter() - t0, "transport_attempts": attempt + 1,
                        "usage": {"prompt": getattr(u, "prompt_tokens", None),
                                  "completion": getattr(u, "completion_tokens", None)}}
            except Exception as exc:  # noqa: BLE001 - transport errors are logged and retried
                last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.warning("%s attempt %d failed: %s", self.model, attempt + 1, last_err)
                time.sleep(min(60, 4 * 2 ** attempt))
        return {"text": "", "finish_reason": "error", "latency_s": 0.0,
                "transport_attempts": self.cfg["attempts"], "error": last_err, "usage": {}}

    def extract(self, system: str, user: str) -> dict[str, Any]:
        """Call, parse; on parse failure retry with the repair suffix (at most max_parse_retries)."""
        attempts = []
        res = self.once(system, user)
        attempts.append(res)
        claims, why = parse_response(res["text"])
        retries = 0
        while claims is None and retries < self.cfg["max_parse_retries"]:
            retries += 1
            res = self.once(system, user + REPAIR_SUFFIX)
            attempts.append(res)
            claims, why = parse_response(res["text"])
        return {"attempts": attempts, "parse_ok": claims is not None, "parse_error": why,
                "parse_retries": retries}


def make_callers(cfg: dict[str, Any], models: Optional[list[str]] = None) -> dict[str, ModelCaller]:
    from shared.llm_judge.nim_registry import load_panel

    C.load_env()
    panel = {e["model"]: e for e in load_panel()}
    ex = cfg["extraction"]
    sub = ex["substitution"]
    panel[sub["replacement"]] = {"org": "Meta", "model": sub["replacement"], "key_env": sub["replacement_key_env"],
                                 "base_url": next(iter(panel.values()))["base_url"]}
    out = {}
    for m in models or ex["panel"]:
        if m not in panel or not os.getenv(panel[m]["key_env"]):
            raise RuntimeError(f"model {m} unavailable (missing key or not on the panel)")
        out[m] = ModelCaller(panel[m], cfg)
    return out


def raw_path(res_dir: Path, repo_slug: str, chunk_idx: int, model: str) -> Path:
    return res_dir / "extraction" / "raw" / repo_slug / f"{chunk_idx:03d}.{C.model_slug(model)}.json"


def build_jobs(cfg: dict[str, Any], limit_repos: Optional[int] = None) -> list[dict[str, Any]]:
    """All (repo, chunk) units in corpus order, with the prompt text."""
    corpus = C.load_corpus(cfg)
    if limit_repos:
        corpus = corpus[:limit_repos]
    jobs = []
    for e in corpus:
        mpath = e.get("manifest_path") or "CLAUDE.md"
        chunks = C.chunk_manifest(e["claude_md_content"], cfg["extraction"]["chunk_max_chars"])
        for ch in chunks:
            jobs.append({"repo": e["full_name"], "slug": e["slug"], "manifest_path": mpath,
                         "chunk": ch, "n_chunks": len(chunks),
                         "user": build_user_message(mpath, ch, len(chunks))})
    return jobs


def run_extraction(cfg: dict[str, Any], *, limit_repos: Optional[int] = None,
                   workers: Optional[dict[str, int]] = None, pilot_chunks: int = 20) -> dict[str, Any]:
    """Run the 3-model extraction with checkpoints and the pre-declared substitution rule.

    The first ``pilot_chunks`` chunks run first; a model with more than 10% unparseable
    responses (after retries) on them is replaced by the declared substitute for the whole run.
    """
    res_dir = C.results_dir(cfg)
    ex = cfg["extraction"]
    system = (C.PROJECT_ROOT / ex["prompt_file"]).read_text(encoding="utf-8")
    prompt_sha = C.sha256_text(system)
    jobs = build_jobs(cfg, limit_repos)
    panel = list(ex["panel"])
    callers = make_callers(cfg, panel)
    workers = workers or {}
    subst_log: dict[str, Any] = {"declared_rule": ex["substitution"], "swaps": []}
    sub_path = res_dir / "extraction" / "substitution.json"
    if sub_path.exists():
        subst_log = C.read_json(sub_path)
        for sw in subst_log.get("swaps", []):
            if sw["replaced"] in panel:
                panel[panel.index(sw["replaced"])] = sw["with"]
        callers = make_callers(cfg, panel)

    def do(caller: ModelCaller, job: dict[str, Any]) -> tuple[str, bool]:
        p = raw_path(res_dir, job["slug"], job["chunk"]["idx"], caller.model)
        if p.exists():
            old = C.read_json(p)
            if old.get("prompt_sha256") == prompt_sha and old.get("user_sha256") == C.sha256_text(job["user"]):
                return str(p), True
        r = caller.extract(system, job["user"])
        last = r["attempts"][-1]
        rec = {"repo": job["repo"], "chunk": job["chunk"]["idx"], "n_chunks": job["n_chunks"],
               "model": caller.model, "prompt_sha256": prompt_sha, "user_sha256": C.sha256_text(job["user"]),
               "text": last["text"], "parse_ok": r["parse_ok"], "parse_error": r["parse_error"],
               "parse_retries": r["parse_retries"],
               "latency_s": sum(a["latency_s"] for a in r["attempts"]),
               "usage": {"prompt": sum((a["usage"].get("prompt") or 0) for a in r["attempts"]),
                         "completion": sum((a["usage"].get("completion") or 0) for a in r["attempts"])},
               "n_api_calls": len(r["attempts"]), "finish_reason": last.get("finish_reason"),
               "error": last.get("error"),
               "discarded_attempt_texts": [a["text"] for a in r["attempts"][:-1]]}
        C.write_json(p, rec)
        return str(p), False

    def run_batch(batch: list[dict[str, Any]], models: list[str]) -> None:
        pools = {m: ThreadPoolExecutor(max_workers=workers.get(C.model_slug(m), 1)) for m in models}
        futs = []
        for job in batch:
            for m in models:
                futs.append(pools[m].submit(do, callers[m], job))
        done = 0
        t0 = time.time()
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 25 == 0:
                logger.info("extraction %d/%d (%.0fs)", done, len(futs), time.time() - t0)
        for p in pools.values():
            p.shutdown()

    pilot, rest = jobs[:pilot_chunks], jobs[pilot_chunks:]
    if not sub_path.exists() or not subst_log.get("pilot_checked"):
        run_batch(pilot, panel)
        for i, m in enumerate(list(panel)):
            recs = [C.read_json(raw_path(res_dir, j["slug"], j["chunk"]["idx"], m)) for j in pilot]
            bad = sum(1 for r in recs if not r["parse_ok"]) / max(1, len(recs))
            logger.info("pilot: %s unparseable share %.3f", m, bad)
            subst_log.setdefault("pilot_unparseable", {})[m] = bad
            if bad > ex["substitution"]["replace_if_unparseable_share_above"]:
                rep = ex["substitution"]["replacement"]
                logger.warning("substituting %s with %s", m, rep)
                subst_log["swaps"].append({"replaced": m, "with": rep, "pilot_unparseable_share": bad})
                panel[i] = rep
        subst_log["pilot_checked"] = True
        subst_log["final_panel"] = panel
        C.write_json(sub_path, subst_log)
        callers = make_callers(cfg, panel)
        run_batch(pilot, panel)  # replacement model on the pilot chunks
    run_batch(rest, panel)
    return {"panel": panel, "n_jobs": len(jobs), "substitution": subst_log}
