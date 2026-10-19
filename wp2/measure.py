"""WP2 measurement of claims against a repository snapshot (Paper 1, RQ2; design section 3).

Python may do lookups, parsing and counting against the git snapshot and compute differences of two
measured values; it never compares a count with a constant and never chooses a verdict. Each
``measure_*`` function returns data properties for the claim individual (counts, deltas, strings);
the HermiT verdicts come from seo-claims.owl. T1, T5, T3, T4, T6, T2 are implemented; T7 only if time.

Usage:
    imported by wp2/step_measure.py and wp2/step_drift.py
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tomllib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pathspec
import yaml

from . import common as C
from . import depparse as D
from .gitutil import ENV
from .normalize import tool_vocab
from .repo import RepoAt

logger = logging.getLogger("wp2.measure")

WINDOWS = (90, 180, 365)
BIN_SHORTHAND = {"tsc", "eslint", "prettier", "vitest", "jest", "playwright", "tsx", "ts-node", "nodemon", "vite",
                 "turbo", "nx", "next", "nuxt", "astro", "biome", "prisma", "drizzle-kit", "wrangler", "storybook",
                 "rimraf", "concurrently", "husky", "lint-staged", "changeset", "lerna", "mocha", "cypress", "tsup",
                 "webpack", "rollup", "esbuild", "tailwindcss", "cdk", "expo", "turbo", "pnpm", "yarn", "npm", "npx"}


@lru_cache(maxsize=1)
def version_sources() -> dict[str, Any]:
    return yaml.safe_load((C.EXP_DIR / "configs" / "wp2_version_sources.yaml").read_text(encoding="utf-8"))["subjects"]


@lru_cache(maxsize=1)
def ecosystems() -> dict[str, Any]:
    return yaml.safe_load((C.EXP_DIR / "configs" / "wp2_tool_signatures.yaml").read_text(encoding="utf-8"))["ecosystems"]


def _spec(lines: list[str]) -> pathspec.GitIgnoreSpec:
    return pathspec.GitIgnoreSpec.from_lines(lines)


class Ctx:
    """Per-(repo, rev) measurement context with caches."""

    def __init__(self, repo: RepoAt) -> None:
        self.repo = repo
        self._census: Optional[D.Census] = None
        self._docs: dict[str, Any] = {}
        self._ignore: dict[str, Optional[pathspec.GitIgnoreSpec]] = {}
        self._sig_cache: dict[str, int] = {}
        self._script_cache: Optional[dict[str, Any]] = None

    @property
    def census(self) -> D.Census:
        if self._census is None:
            self._census = D.census(self.repo)
        return self._census

    def doc(self, path: str) -> Any:
        """Parsed JSON / TOML of a tracked file (None if unreadable)."""
        if path not in self._docs:
            text = self.repo.read(path)
            val = None
            if text is not None:
                try:
                    val = json.loads(text) if path.endswith(".json") else tomllib.loads(text)
                except Exception:  # noqa: BLE001
                    val = None
            self._docs[path] = val
        return self._docs[path]

    def ignore_spec(self, d: str) -> Optional[pathspec.GitIgnoreSpec]:
        if d not in self._ignore:
            p = (d + "/" if d else "") + ".gitignore"
            text = self.repo.read(p) if p in self.repo.files else None
            self._ignore[d] = _spec(text.splitlines()) if text else None
        return self._ignore[d]


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def _run_git(repo: RepoAt, *args: str, timeout: int = 300) -> Optional[str]:
    try:
        p = subprocess.run(["git", "-C", str(repo.dir), *args], capture_output=True, env=ENV, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def _pathspec_arg(path: str, glob: bool) -> str:
    return f":(glob){path}" if glob else f":(literal){path}"


# ---------------------------------------------------------------------------
# T1 path existence
# ---------------------------------------------------------------------------

def path_resolution(ctx: Ctx, path: str, glob: bool, bracket: bool) -> dict[str, Any]:
    """Exact and suffix matches of ``path`` at the snapshot (no git history)."""
    repo = ctx.repo
    exact, entities = 0, set()
    if glob:
        sp = _spec([path])
        matched = [f for f in repo.files if sp.match_file(f)]
        exact = len(matched)
        if exact == 0 and "/" in path:
            sp2 = _spec(["**/" + path])
            entities = {f for f in repo.files if sp2.match_file(f)}
        return {"exact": exact, "entities": entities, "resolved": None}
    if path in repo.files:
        exact = 1
    elif path in repo.dir_set:
        exact = repo.count_under(path)
    if exact:
        return {"exact": exact, "entities": set(), "resolved": path}
    base = path.rsplit("/", 1)[-1]
    for f in repo.by_basename.get(base, ()):
        if f != path and f.endswith("/" + path):
            entities.add(f)
    for d in repo.dirs.get(base, ()):
        if d != path and d.endswith("/" + path):
            entities.add(d)
    # drop entities nested inside another matched entity (a file under a matched directory)
    entities = {e for e in entities if not any(e.startswith(o + "/") for o in entities if o != e)}
    resolved = next(iter(entities)) if len(entities) == 1 else None
    return {"exact": 0, "entities": entities, "resolved": resolved}


def gitignored(ctx: Ctx, path: str, is_dir: bool) -> bool:
    """1 if a .gitignore at the snapshot (root or an ancestor directory) matches the path or an ancestor."""
    parts = path.split("/")
    for k in range(1, len(parts) + 1):
        for d_i in range(0, k):  # .gitignore directory = first d_i components
            d = "/".join(parts[:d_i])
            spec = ctx.ignore_spec(d)
            if spec is None:
                continue
            rel = "/".join(parts[d_i:k])
            forms = [rel + "/"] if k < len(parts) else [rel + "/", rel]
            if any(spec.match_file(f) for f in forms):
                return True
    return False


def measure_path(ctx: Ctx, path: str, is_dir: bool, glob: bool, bracket: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """T1 measures. Returns (props, info) with info['resolved'] for T2."""
    res = path_resolution(ctx, path, glob, bracket)
    props: dict[str, Any] = {"pathExactMatchCount": res["exact"], "pathSuffixMatchCount": len(res["entities"])}
    info = {"resolved": res["resolved"] if res["resolved"] else (next(iter(res["entities"])) if len(res["entities"]) == 1 else None)}
    if res["exact"] == 0 and not res["entities"]:
        props["pathIgnoredCount"] = 1 if gitignored(ctx, path, is_dir) else 0
        props["pathSubmoduleCount"] = 1 if ctx.repo.in_gitlink(path) else 0
        if props["pathIgnoredCount"] == 0 and props["pathSubmoduleCount"] == 0:
            out = _run_git(ctx.repo, "log", ctx.repo.rev, "--format=%ct", "-n", "1000", "--", _pathspec_arg(path, glob),
                           timeout=240)
            if out is not None:
                cts = [int(x) for x in out.split()]
                props["pathEverTrackedCount"] = len(cts)
                if cts:
                    props["daysSincePathLastTracked"] = max(0, (ctx.repo.epoch - cts[0]) // 86400)
    return props, info


# ---------------------------------------------------------------------------
# T2 path churn
# ---------------------------------------------------------------------------

def measure_churn(ctx: Ctx, resolved: str) -> dict[str, Any]:
    """Commits and changed lines inside ``resolved`` in the 90/180/365 days before the snapshot."""
    repo = ctx.repo
    e = repo.epoch
    ps = _pathspec_arg(resolved, False)
    base = ["log", repo.rev, "--first-parent", "--no-merges", f"--since={e - 365 * 86400}", f"--until={e}"]
    out = _run_git(repo, *base, "--format=%H\t%ct", "--", ps, timeout=300)
    if out is None:
        return {}
    commits = [(ln.split("\t")[0], int(ln.split("\t")[1])) for ln in out.splitlines() if "\t" in ln]
    props: dict[str, Any] = {}
    for w in WINDOWS:
        props[f"pathCommitsIn{w}d"] = sum(1 for _, ct in commits if ct >= e - w * 86400)
    numstat = _run_git(repo, *base, "-n", "300", "--numstat", "--format=@%H\t%ct", "--", ps, timeout=600)
    lines_by_commit: dict[str, tuple[int, int]] = {}
    cur = None
    for ln in (numstat or "").splitlines():
        if ln.startswith("@"):
            h, ct = ln[1:].split("\t")
            cur = h
            lines_by_commit[cur] = (int(ct), 0)
        elif ln.strip() and cur:
            a, d, _p = (ln.split("\t") + ["", "", ""])[:3]
            n = (int(a) if a.isdigit() else 0) + (int(d) if d.isdigit() else 0)
            lines_by_commit[cur] = (lines_by_commit[cur][0], lines_by_commit[cur][1] + n)
    if numstat is not None:
        for w in WINDOWS:
            props[f"pathLinesChangedIn{w}d"] = sum(n for ct, n in lines_by_commit.values() if ct >= e - w * 86400)
    rc = _run_git(repo, "rev-list", "--count", repo.rev, "--first-parent", "--no-merges",
                  f"--since={e - 180 * 86400}", f"--until={e}")
    if rc is not None and rc.strip().isdigit():
        props["repoCommitsIn180d"] = int(rc.strip())
    last = _run_git(repo, "log", repo.rev, "-n", "1", "--format=%ct", "--", ps)
    if last and last.strip().isdigit():
        props["daysSincePathLastChange"] = max(0, (e - int(last.strip())) // 86400)
    props["_churn_truncated_300"] = int(len(commits) > 300)
    return props


# ---------------------------------------------------------------------------
# T3 tool signatures
# ---------------------------------------------------------------------------

def _glob_count(ctx: Ctx, globs: list[str]) -> int:
    if not globs:
        return 0
    sp = _spec(globs)
    return sum(1 for f in ctx.repo.files if sp.match_file(f) and not any(s in f for s in ("node_modules/", "vendor/")))


def _key_hits(ctx: Ctx, keys: list[str]) -> int:
    n = 0
    for k in keys:
        fname, _, rest = k.partition("::")
        sw = None
        if "^=" in rest:
            rest, sw = rest.split("^=", 1)
        cands = [p for p in ctx.repo.by_basename.get(fname, ()) if not any(s in p for s in D.SKIP_DIRS)]
        cands.sort(key=lambda p: (p.count("/"), p))
        ctx.repo.prefetch(cands[:200])
        for p in cands[:200]:
            doc = ctx.doc(p)
            if not isinstance(doc, dict):
                continue
            cur: Any = doc
            for seg in rest.split("."):
                cur = cur.get(seg) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None and (sw is None or (isinstance(cur, str) and cur.startswith(sw))):
                n += 1
    return n


def _dep_hits(ctx: Ctx, deps: list[str]) -> int:
    if not deps:
        return 0
    want = {D.norm_name(d) for d in deps}
    return sum(1 for d in ctx.census.decls if D.norm_name(d.name) in want)


def sig_hits(ctx: Ctx, tool: str) -> int:
    if tool in ctx._sig_cache:
        return ctx._sig_cache[tool]
    spec = tool_vocab()[tool]
    n = _glob_count(ctx, spec.get("files") or []) + _key_hits(ctx, spec.get("manifest_keys") or []) + _dep_hits(
        ctx, spec.get("deps") or [])
    ctx._sig_cache[tool] = n
    return n


def measure_tool(ctx: Ctx, tool: str) -> dict[str, Any]:
    spec = tool_vocab()[tool]
    props: dict[str, Any] = {"toolSignatureHitCount": sig_hits(ctx, tool)}
    eco = spec.get("ecosystem", "any")
    anchors = ecosystems().get(eco, {}).get("anchors", [])
    props["toolEcosystemAnchorCount"] = 0 if eco == "any" else _glob_count(ctx, anchors)
    props["competingToolSignatureCount"] = sum(sig_hits(ctx, c) for c in spec.get("competitors", []) or [])
    return props


# ---------------------------------------------------------------------------
# T4 dependency
# ---------------------------------------------------------------------------

def measure_dependency(ctx: Ctx, name: str, major: Optional[int]) -> dict[str, Any]:
    c = ctx.census
    n = D.norm_name(name)
    ls = D.last_segment(n)
    exact = [d for d in c.decls if D.norm_name(d.name) == n]
    suffix = [d for d in c.decls if D.norm_name(d.name) != n and ls in D.segments(D.norm_name(d.name))]
    props: dict[str, Any] = {"dependencyExactMatchCount": len(exact), "dependencySuffixMatchCount": len(suffix),
                             "dependencyManifestParsedCount": len(c.parsed)}
    if major is not None:
        props["claimedMajor"] = major
        decls = exact or suffix
        lows, ups, unb, parsed_any = [], [], 0, False
        for d in decls:
            cons = D.parse_constraint(d.spec, bare_caret=d.ecosystem in ("cargo",)) if d.spec not in ("",) else D.parse_constraint("*")
            if d.ecosystem == "go":
                v = D._ver(d.spec)
                cons = {"lower": v, "upper": v, "upper_inclusive": True} if v else None
            if cons is None:
                continue
            parsed_any = True
            lows.append(cons["lower"][0])
            um = D.upper_major(cons)
            if um is None:
                unb = 1
            else:
                ups.append(um)
        if parsed_any:
            props["declaredLowerMajorDelta"] = min(lows) - major
            if unb:
                props["declaredUpperUnboundedCount"] = 1
            else:
                props["declaredUpperMajorDelta"] = max(ups) - major
                props["declaredUpperUnboundedCount"] = 0
    return props


# ---------------------------------------------------------------------------
# T5 scripts and make targets
# ---------------------------------------------------------------------------
_MAKE_TARGET = re.compile(r"^([^\s:#=\t.$%][^:=#]*?)\s*::?(?![=:])", re.M)


def _make_targets(text: str) -> tuple[set[str], int]:
    targets: set[str] = set()
    dynamic = 0
    for m in re.finditer(r"^([^\s#\t.][^:=#\n]*?)\s*::?(?![=])", text, re.M):
        lhs = m.group(1)
        for t in lhs.split():
            if "%" in t or "$(" in t or "${" in t:
                dynamic += 1
            else:
                targets.add(t)
    return targets, dynamic


def _script_index(ctx: Ctx) -> dict[str, Any]:
    if ctx._script_cache is not None:
        return ctx._script_cache
    repo = ctx.repo
    idx: dict[str, Any] = {"script": {"hosts": [], "defs": defaultdict(list)}, "composer": {"hosts": [], "defs": defaultdict(list)},
                           "make": {"hosts": [], "defs": defaultdict(list), "dynamic": 0, "dyn_root": 0}}
    pj = [p for p in repo.files if p.rsplit("/", 1)[-1] == "package.json" and not any(s in p for s in D.SKIP_DIRS)]
    cj = [p for p in repo.files if p.rsplit("/", 1)[-1] == "composer.json" and not any(s in p for s in D.SKIP_DIRS)]
    mk = [p for p in repo.files if (p.rsplit("/", 1)[-1] in ("Makefile", "GNUmakefile", "makefile") or p.endswith(".mk"))
          and not any(s in p for s in D.SKIP_DIRS)]
    repo.prefetch(pj[:400] + cj[:100] + mk[:200])
    for kind, files in (("script", pj[:400]), ("composer", cj[:100])):
        for p in files:
            doc = ctx.doc(p)
            if not isinstance(doc, dict):
                continue
            idx[kind]["hosts"].append(p)
            for s in (doc.get("scripts") or {}):
                idx[kind]["defs"][s].append(p)
    for p in mk[:200]:
        text = repo.read(p)
        if text is None:
            continue
        idx["make"]["hosts"].append(p)
        t, dyn = _make_targets(text)
        idx["make"]["dynamic"] += dyn
        if "/" not in p:
            idx["make"]["dyn_root"] += dyn
        for x in t:
            idx["make"]["defs"][x].append(p)
    ctx._script_cache = idx
    return idx


def measure_command(ctx: Ctx, kind: str, target: str, runner: str) -> dict[str, Any]:
    idx = _script_index(ctx)[kind]
    defs = idx["defs"].get(target, [])
    root = sum(1 for p in defs if "/" not in p)
    if kind == "script" and target == "start" and not defs and "server.js" in ctx.repo.files:
        root = 1  # npm's default start script
    props: dict[str, Any] = {"scriptDefinedRootCount": root, "scriptDefinedAnyCount": max(len(defs), root),
                             "scriptHostFileCount": len(idx["hosts"])}
    dyn = idx.get("dynamic", 0) if kind == "make" else 0
    if kind == "script" and runner != "npm" and not defs:
        # pnpm/yarn/bun fall back to binaries: a name that is a declared dependency or a well-known bin
        # could resolve without a package.json script (counted as a dynamic resolution).
        names = {D.norm_name(d.name) for d in ctx.census.decls}
        if D.norm_name(target) in names or target in BIN_SHORTHAND:
            dyn = 1
    props["dynamicTargetRuleCount"] = dyn
    return props


# ---------------------------------------------------------------------------
# T6 version evidence
# ---------------------------------------------------------------------------

def _unit(p: int) -> int:
    return 10 ** (9 - 3 * (p - 1))


def _code(v: tuple[int, ...], p: int) -> int:
    v = tuple(v) + (0,) * 4
    return sum(v[i] * 10 ** (9 - 3 * i) for i in range(p))


def _get_key(doc: Any, dotted: str) -> Any:
    cur = doc
    for seg in dotted.split("."):
        # keys that themselves contain dots (e.g. "requires-python") are single segments here
        cur = cur.get(seg) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def _raw_from_source(ctx: Ctx, spec: dict[str, Any], path: str) -> Optional[str]:
    parser = spec["parser"]
    if parser in ("json", "toml"):
        doc = ctx.doc(path)
        v = _get_key(doc, spec["key"]) if doc else None
        return None if v is None or isinstance(v, (dict, list)) else str(v)
    text = ctx.repo.read(path)
    if text is None:
        return None
    if parser == "text":
        for ln in text.splitlines():
            ln = ln.split("#")[0].strip()
            if ln:
                return ln
        return None
    if parser == "regex":
        m = re.search(spec["key"], text)
        if not m:
            return None
        return next((g for g in m.groups() if g), None)
    if parser == "ini":
        import configparser

        cp = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            cp.read_string(text)
        except Exception:  # noqa: BLE001
            return None
        sec, _, opt = spec["key"].rpartition(".")
        return cp.get(sec, opt, fallback=None) if (cp.has_section(sec) or sec == "DEFAULT") else None
    if parser == "toolversions":
        for ln in text.splitlines():
            parts = ln.split()
            if len(parts) >= 2 and parts[0] == spec["key"]:
                return parts[1]
        return None
    return None


def _bounds(raw: str, kind: str, subject: str) -> Optional[tuple[tuple[int, ...], Optional[tuple[int, ...]], bool]]:
    """(lower comps, upper comps or None, upper_inclusive) for a declared version."""
    raw = raw.strip()
    if subject == "java":
        raw = re.sub(r"^1\.(\d+)", r"\1", raw.replace("_", "."))
    if kind == "pin" or (kind == "constraint" and re.fullmatch(r"v?\d+(\.\d+){0,3}", raw)):
        v = D._ver(raw)
        return (v, v, True) if v else None
    if kind == "min":
        v = D._ver(raw)
        return (v, None, False) if v else None
    c = D.parse_constraint(raw)
    return (c["lower"], c["upper"], c["upper_inclusive"]) if c else None


def measure_version(ctx: Ctx, subject: str, claim_code_prec: tuple[int, int, tuple[int, ...]]
                    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Version evidence individuals for a T6 claim. ``claim_code_prec`` = (code, precision, components)."""
    _code_full, prec, comps = claim_code_prec
    repo = ctx.repo
    specs = version_sources().get(subject, [])
    evidences: list[dict[str, Any]] = []
    fallback_ok = True
    for sp in specs:
        if sp.get("fallback_only"):
            continue
        pat = sp["path"]
        if pat.startswith("/"):
            glob = pat[1:]
            cands = [p for p in repo.files if "/" not in p and (p == glob or (("*" in glob) and pathspec.GitIgnoreSpec.from_lines(["/" + glob]).match_file(p)))]
        elif "*" in pat:
            s2 = _spec([pat])
            cands = [p for p in repo.files if s2.match_file(p) and not any(x in p for x in D.SKIP_DIRS)]
        else:
            cands = [p for p in repo.by_basename.get(pat, ()) if not any(x in p for x in D.SKIP_DIRS)]
        cands.sort(key=lambda p: (p.count("/"), p))
        cands = cands[:50]
        repo.prefetch(cands)
        for path in cands:
            raw = _raw_from_source(ctx, sp, path)
            if raw is None:
                continue
            b = _bounds(raw, sp["kind"], subject)
            if b is None:
                continue
            lo, up, incl = b
            p_eff = min(prec, len(lo))
            ev: dict[str, Any] = {"versionSourcePath": path, "versionSourceRaw": raw[:120]}
            ev["versionLowerDelta"] = _code(lo, p_eff) - _code(comps, p_eff)
            if up is None:
                ev["versionUpperUnboundedCount"] = 1
            else:
                pu = min(prec, len(up))
                trunc_zero = all(x == 0 for x in up[pu:])
                upper_code = _code(up, pu) - (0 if incl or not trunc_zero else _unit(pu))
                ev["versionUpperDelta"] = upper_code - _code(comps, pu)
                ev["versionUpperUnboundedCount"] = 0
            evidences.append(ev)
            if subject == "project_self":
                fallback_ok = False
    if subject == "project_self" and not evidences and fallback_ok:
        tag = _run_git(repo, "describe", "--tags", "--abbrev=0", repo.rev)
        if tag and tag.strip():
            b = _bounds(tag.strip().lstrip("vV"), "pin", subject)
            if b:
                lo = b[0]
                p_eff = min(prec, len(lo))
                evidences.append({"versionSourcePath": "<git-tag>", "versionSourceRaw": tag.strip()[:120],
                                  "versionLowerDelta": _code(lo, p_eff) - _code(comps, p_eff),
                                  "versionUpperDelta": _code(lo, p_eff) - _code(comps, p_eff),
                                  "versionUpperUnboundedCount": 0})
    return {"versionSourceCount": len(evidences)}, evidences


# ---------------------------------------------------------------------------
# T7 import constraint (stretch)
# ---------------------------------------------------------------------------
_JS_IMPORT = re.compile(
    r"""(?:\bimport\s+(?:[\w*{}\s,]+\s+from\s+)?|\brequire\(\s*|\bimport\(\s*|\bexport\s+[\w*{}\s,]+\s+from\s+)['"]([^'"]+)['"]""")
SRC_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte")


def measure_import(ctx: Ctx, target: str, scope: Optional[str], max_files: int = 3000) -> dict[str, Any]:
    """Source files scanned in scope and import statements that reference the forbidden target."""
    import ast

    repo = ctx.repo
    pre = None
    if scope:
        pre = scope.strip().strip("`'\"").lstrip("./").rstrip("/")
    files = [f for f in repo.files if f.endswith(SRC_EXT) and (pre is None or f == pre or f.startswith(pre + "/")
                                                               or ("/" + pre + "/") in ("/" + f))]
    files = sorted(files)[:max_files]
    repo.prefetch(files)
    scanned = edges = 0
    for f in files:
        text = repo.read(f)
        if text is None:
            continue
        scanned += 1
        specs: list[tuple[str, int]] = []
        if f.endswith(".py"):
            try:
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node, ast.Import):
                        specs += [(a.name, 0) for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        specs.append((node.module or "", node.level))
            except SyntaxError:
                continue
        else:
            specs = [(m.group(1), 1 if m.group(1).startswith(".") else 0) for m in _JS_IMPORT.finditer(text)]
        for spec, level in specs:
            if target.lower() == "relative":
                edges += int(level > 0 or spec.startswith("."))
            else:
                t = target.lstrip("/")
                edges += int(spec == target or spec.startswith(target.rstrip("/") + "/") or spec == t
                             or spec.startswith(t.rstrip("/") + "/"))
    return {"importFilesScanned": scanned, "forbiddenImportEdgeCount": edges}
