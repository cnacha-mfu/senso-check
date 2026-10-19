"""Independent deterministic re-check of WP2 verdicts (Paper 1, RQ2; design section 6.1).

This module re-implements measurement AND the decision rules with plain ``if`` logic, from the
threshold register (thresholds.yaml) and the claim content only. It shares no code with
``measure.py``, ``depparse.py``, ``repo.py`` or ``gitutil.py`` (enforced by ``check_independence``):
its own git calls, its own gitignore and glob translation, its own manifest readers, its own
version-range logic (``packaging`` for PEP 440, hand-written elsewhere).

What it measures is *encoding fidelity*: does the reasoner decide exactly what the rules state? It is not
construct validity (see triangulation).

Usage:
    python wp2_consistency.py recheck
"""

from __future__ import annotations

import ast
import fnmatch
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Optional

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
FORBIDDEN_IMPORTS = ("measure", "depparse", "repo", "gitutil")


def check_independence() -> list[str]:
    """Names of forbidden WP2 modules imported by this file (must be empty)."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom):
            names = [(node.module or "")] + [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        for n in names:
            if any(n == f or n.endswith("." + f) or n.startswith(f + ".") for f in FORBIDDEN_IMPORTS):
                bad.append(n)
    return bad


# ----------------------------------------------------------------------------
# own git access
# ----------------------------------------------------------------------------
class Git:
    def __init__(self, d: Path, rev: str) -> None:
        self.d, self.rev = Path(d), rev
        self._files: Optional[list[str]] = None
        self._links: Optional[list[str]] = None
        self._cache: dict[str, Optional[str]] = {}
        self._sha: dict[str, str] = {}

    def run(self, *a: str, timeout: int = 300) -> Optional[str]:
        try:
            p = subprocess.run(["git", "-C", str(self.d), *a], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None

    def epoch(self) -> int:
        return int((self.run("show", "-s", "--format=%ct", self.rev) or "0").strip())

    def _tree(self) -> None:
        out = self.run("ls-tree", "-r", self.rev, timeout=900) or ""
        self._files, self._links = [], []
        for ln in out.splitlines():
            meta, _, path = ln.partition("\t")
            parts = meta.split()
            mode = parts[0]
            if mode != "160000":
                self._sha[path] = parts[2]
            (self._links if mode == "160000" else self._files).append(path)

    @property
    def files(self) -> list[str]:
        if self._files is None:
            self._tree()
        return self._files  # type: ignore[return-value]

    @property
    def links(self) -> list[str]:
        if self._links is None:
            self._tree()
        return self._links  # type: ignore[return-value]

    _INTERESTING = re.compile(
        r"(^|/)(\.gitignore|package\.json|composer\.json|Makefile|GNUmakefile|makefile|pyproject\.toml|setup\.py|setup\.cfg|"
        r"Pipfile|Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle(\.kts)?|\.python-version|\.node-version|\.nvmrc|"
        r"\.ruby-version|\.tool-versions|global\.json|rust-toolchain(\.toml)?|VERSION|settings\.ini|requirements[^/]*\.txt|"
        r"[^/]*\.(csproj|fsproj|mk|gemspec))$")

    def prefetch_interesting(self, limit: int = 900) -> None:
        """Batch-fetch the blobs of manifest-like files (own implementation; blobless clones)."""
        _ = self.files
        want = [p for p in self._sha if self._INTERESTING.search(p) and not any(s in p for s in _SKIP)]
        want = sorted(want, key=lambda x: (x.count("/"), x))[:limit]
        shas = [self._sha[p] for p in want]
        chk = subprocess.run(["git", "-C", str(self.d), "cat-file", "--batch-check"], input="\n".join(shas) + "\n",
                             capture_output=True, text=True, env={**__import__("os").environ, "GIT_NO_LAZY_FETCH": "1"})
        miss = [ln.split()[0] for ln in chk.stdout.splitlines() if ln.endswith(" missing")]
        for i in range(0, len(miss), 1500):
            subprocess.run(["git", "-C", str(self.d), "-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin", "--no-tags",
                            "--no-write-fetch-head", "--filter=blob:none", "--stdin"],
                           input="\n".join(miss[i:i + 1500]) + "\n", capture_output=True, text=True, timeout=1800)

    def exists(self, path: str) -> bool:
        return self.run("cat-file", "-e", f"{self.rev}:{path}") is not None

    def show(self, path: str) -> Optional[str]:
        if path not in self._cache:
            self._cache[path] = self.run("show", f"{self.rev}:{path}")
        return self._cache[path]

    def touched(self, path: str, glob: bool = False) -> bool:
        spec = f":(glob){path}" if glob else f":(literal){path}"
        out = self.run("rev-list", self.rev, "-n", "1", "--", spec, timeout=240)
        return bool(out and out.strip())


# ----------------------------------------------------------------------------
# own glob / gitignore translation
# ----------------------------------------------------------------------------
def _seg_regex(seg: str) -> str:
    out, i = "", 0
    while i < len(seg):
        c = seg[i]
        if c == "*":
            out += "[^/]*"
        elif c == "?":
            out += "[^/]"
        elif c == "[":
            j = seg.find("]", i + 1)
            if j > i:
                body = seg[i + 1: j].replace("\\", "\\\\")
                out += "[" + ("^" if body.startswith("!") else "") + body.lstrip("!") + "]"
                i = j
            else:
                out += re.escape(c)
        else:
            out += re.escape(c)
        i += 1
    return out


def glob_regex(pat: str, anchored: bool) -> re.Pattern[str]:
    segs = pat.strip("/").split("/")
    parts = []
    for k, s in enumerate(segs):
        if s == "**":
            parts.append("(?:.*/)?" if k < len(segs) - 1 else ".*")
        else:
            parts.append(_seg_regex(s) + ("/" if k < len(segs) - 1 else ""))
    body = "".join(parts)
    return re.compile(("^" if anchored else "^(?:.*/)?") + body + "(?:/.*)?$")


def glob_match(pat: str, f: str) -> bool:
    return bool(glob_regex(pat, "/" in pat.strip("/")).match(f))


def parse_ignore(text: str) -> list[tuple[bool, re.Pattern[str], bool]]:
    rules = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        if not ln or ln.startswith("#"):
            continue
        neg = ln.startswith("!")
        if neg:
            ln = ln[1:]
        dir_only = ln.endswith("/")
        ln = ln.rstrip("/")
        anchored = "/" in ln
        ln = ln.lstrip("/")
        rules.append((neg, glob_regex(ln, anchored), dir_only))
    return rules


def ignored_by(rules: list[tuple[bool, re.Pattern[str], bool]], rel: str, is_dir: bool) -> bool:
    state = False
    for neg, rx, dir_only in rules:
        if dir_only and not is_dir:
            continue
        if rx.match(rel):
            state = not neg
    return state


# ----------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------
def load_thr(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or (_ROOT / "ontology" / "modules" / "thresholds.yaml")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {t["id"]: t["value"] for t in doc["thresholds"]}


# ----------------------------------------------------------------------------
# T1 / T2
# ----------------------------------------------------------------------------
def _entities(files: list[str], p: str) -> set[str]:
    out = set()
    needle = "/" + p
    for f in files:
        full = "/" + f
        i = full.find(needle + "/")
        if i >= 0:
            out.add(full[1: i + len(needle)])
        elif full.endswith(needle):
            out.add(f)
    return out


def _is_ignored(g: Git, path: str) -> bool:
    parts = path.split("/")
    for k in range(1, len(parts) + 1):
        rel_full = "/".join(parts[:k])
        for d_i in range(0, k):
            d = "/".join(parts[:d_i])
            gi = g.show((d + "/" if d else "") + ".gitignore")
            if not gi:
                continue
            rules = parse_ignore(gi)
            rel = "/".join(parts[d_i:k])
            if k < len(parts):
                if ignored_by(rules, rel, True):
                    return True
            else:
                if ignored_by(rules, rel, True) or ignored_by(rules, rel, False):
                    return True
    return False


def path_state(g: Git, a: dict[str, Any]) -> dict[str, Any]:
    p, glob = a["path"], a.get("glob", False)
    if glob:
        exact = any(glob_match(p, f) for f in g.files)
    else:
        exact = g.exists(p) and (p not in g.links)
    if exact:
        return {"exact": True, "suffix": False, "n_entities": 0, "entity": p if not glob else None}
    ents: set[str]
    if glob:
        ents = {f for f in g.files if "/" in p and glob_match("**/" + p, f)}
    else:
        ents = _entities(g.files, p)
        ents = {e for e in ents if not any(e != o and e.startswith(o + "/") for o in ents)}
        ents.discard(p)
    st = {"exact": False, "suffix": bool(ents), "n_entities": len(ents), "entity": next(iter(ents)) if len(ents) == 1 else None}
    if not ents:
        st["ignored"] = _is_ignored(g, p)
        st["submodule"] = any(p == l or p.startswith(l + "/") for l in g.links)
        if not st["ignored"] and not st["submodule"]:
            st["ever"] = g.touched(p, glob)
    return st


def verdict_t1(g: Git, a: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    s = path_state(g, a)
    if not s["exact"] and not s["suffix"] and a.get("alt_path"):
        # tree diagram with a wrapping root: the root-dropped path is used only if it resolves
        alt = path_state(g, {**a, "path": a["alt_path"]})
        if alt["exact"] or alt["suffix"] or alt.get("ignored") or alt.get("submodule") or alt.get("ever"):
            s = alt
    if s["exact"]:
        return "SupportedClaim", "PathFoundExact"
    if s["suffix"]:
        return "SupportedClaim", "PathFoundBySuffix"
    if s.get("ignored"):
        return "UnverifiableClaim", "PathUntrackedByDesign"
    if s.get("submodule"):
        return "UnverifiableClaim", "PathInSubmodule"
    if "ever" not in s:
        return None, None
    return "ContradictedClaim", "PathRemoved" if s["ever"] else "PathNeverExisted"


def verdict_t2(g: Git, a: dict[str, Any], thr: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    s = path_state(g, a)
    if not s["exact"] and s["n_entities"] >= 2:
        return "UnverifiableClaim", "StatusPathAmbiguous"
    if not s["exact"] and not s["suffix"]:
        return "UnverifiableClaim", "StatusPathUnresolved"
    ent = s["entity"]
    if ent is None:
        return None, None
    ep = g.epoch()
    w = int(thr["wp2_status_window_days"])
    since = ep - w * 86400
    spec = f":(literal){ent}"
    log = g.run("log", g.rev, "--first-parent", "--no-merges", f"--since={since}", f"--until={ep}", "--numstat",
                "--format=tformat:@@", "--", spec, timeout=600)
    if log is None:
        return None, None
    commits = log.count("@@")
    lines = 0
    for ln in log.splitlines():
        f = ln.split("\t")
        if len(f) >= 3 and f[0].isdigit() and f[1].isdigit():
            lines += int(f[0]) + int(f[1])
    if commits <= thr["wp2_inactive_support_max_commits"]:
        return "SupportedClaim", "InactivePathQuiet"
    if commits >= thr["wp2_inactive_contradict_min_commits"] and lines >= thr["wp2_inactive_contradict_min_lines"]:
        return "ContradictedClaim", "InactivePathActive"
    return "IndeterminateClaim", "InactivePathModerate"


# ----------------------------------------------------------------------------
# T3 tool
# ----------------------------------------------------------------------------
_DEP_FILES = ("package.json", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "composer.json", "Gemfile", "pom.xml",
              "build.gradle", "build.gradle.kts", "Cargo.toml", "go.mod", "pubspec.yaml", "mix.exs", "build.sbt", "Package.swift", "settings.ini")
_SKIP = ("node_modules/", "vendor/", "third_party/", "third-party/", ".yarn/")


def _dep_text_files(g: Git) -> list[str]:
    out = []
    for f in g.files:
        b = f.rsplit("/", 1)[-1]
        if any(s in f for s in _SKIP):
            continue
        if b in _DEP_FILES or (b.startswith("requirements") and b.endswith(".txt")) or b.endswith((".csproj", ".fsproj")):
            out.append(f)
    return sorted(out, key=lambda x: (x.count("/"), x))[:300]


def _toml_or_json(g: Git, path: str) -> Any:
    t = g.show(path)
    if t is None:
        return None
    try:
        return json.loads(t) if path.endswith(".json") else tomllib.loads(t)
    except Exception:  # noqa: BLE001
        return None


def tool_hits(g: Git, spec: dict[str, Any], cache: dict[str, int], name: str) -> int:
    if name in cache:
        return cache[name]
    n = 0
    for pat in spec.get("files") or []:
        n += sum(1 for f in g.files if glob_match(pat, f) and not any(s in f for s in ("node_modules/", "vendor/")))
    for k in spec.get("manifest_keys") or []:
        fname, _, rest = k.partition("::")
        sw = None
        if "^=" in rest:
            rest, sw = rest.split("^=", 1)
        for f in g.files:
            if f.rsplit("/", 1)[-1] != fname or any(s in f for s in _SKIP):
                continue
            doc = _toml_or_json(g, f)
            cur = doc
            for seg in rest.split("."):
                cur = cur.get(seg) if isinstance(cur, dict) else None
            if cur is not None and (sw is None or (isinstance(cur, str) and cur.startswith(sw))):
                n += 1
    deps = [d.lower() for d in spec.get("deps") or []]
    if deps:
        for f in _dep_text_files(g):
            text = (g.show(f) or "").lower()
            n += sum(1 for d in deps if re.search(r"(?<![\w@/.-])" + re.escape(d) + r"(?![\w.-])", text))
    cache[name] = n
    return n


def verdict_t3(g: Git, a: dict[str, Any], sigs: dict[str, Any], cache: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    tool = a["tool"]
    spec = sigs["tools"][tool]
    hits = tool_hits(g, spec, cache, tool)
    if hits >= 1:
        return "SupportedClaim", "ToolSignatureFound"
    eco = spec.get("ecosystem", "any")
    anchors = 0
    if eco != "any":
        for pat in sigs["ecosystems"][eco]["anchors"]:
            anchors += sum(1 for f in g.files if glob_match(pat, f) and not any(s in f for s in ("node_modules/", "vendor/")))
    comp = sum(tool_hits(g, sigs["tools"][c], cache, c) for c in spec.get("competitors") or [])
    if spec["category"] == "package_manager" and comp >= 1:
        return "ContradictedClaim", "ToolAbsentCompetitorPresent"
    if anchors >= 1:
        return "ContradictedClaim", "ToolAbsentInEcosystem"
    return "UnverifiableClaim", "ToolEcosystemAbsent"


# ----------------------------------------------------------------------------
# T4 dependency and semver
# ----------------------------------------------------------------------------
def _norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n.strip().lower())


def _declared(g: Git) -> tuple[list[tuple[str, str]], int]:
    decls: list[tuple[str, str]] = []
    parsed = 0

    def walk(o: Any, depkey: bool) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                kk = str(k).lower()
                dk = depkey or "depend" in kk or kk in ("require", "require-dev", "requires", "packages", "dev-packages")
                if dk and isinstance(v, str) and kk not in ("python", "php", "version"):
                    decls.append((_norm(str(k)), v))
                elif dk and isinstance(v, dict) and "version" in v and isinstance(v["version"], str):
                    decls.append((_norm(str(k)), v["version"]))
                else:
                    walk(v, dk)
        elif isinstance(o, list) and depkey:
            for s in o:
                if isinstance(s, str):
                    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;#]*)", s)
                    if m:
                        decls.append((_norm(m.group(1)), m.group(2).strip()))
                else:
                    walk(s, depkey)

    for f in _dep_text_files(g):
        b = f.rsplit("/", 1)[-1]
        text = g.show(f)
        if text is None:
            continue
        before = len(decls)
        try:
            if b in ("package.json", "composer.json"):
                walk(json.loads(text), False)
            elif b in ("pyproject.toml", "Pipfile", "Cargo.toml"):
                walk(tomllib.loads(text), False)
            elif b.startswith("requirements") and b.endswith(".txt"):
                for ln in text.splitlines():
                    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([<>=!~][^;#]*)?", ln)
                    if m and not ln.lstrip().startswith(("-", "#")):
                        decls.append((_norm(m.group(1)), (m.group(2) or "").strip()))
            elif b == "go.mod":
                decls += [(m.group(1).lower(), m.group(2)) for m in re.finditer(r"^\s*(?:require\s+)?([\w./-]+\.[\w./-]+)\s+(v[\d.][^\s]*)", text, re.M)]
            elif b == "Gemfile":
                decls += [(m.group(1).lower(), m.group(2) or "") for m in re.finditer(r"gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?", text)]
            elif b.endswith((".csproj", ".fsproj")):
                decls += [(m.group(1).lower(), m.group(2) or "") for m in re.finditer(r'Include="([^"]+)"[^>]*?(?:Version="([^"]*)")?', text)]
            elif b == "pom.xml":
                for m in re.finditer(r"<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>(?:\s*<version>([^<]+)</version>)?", text):
                    decls.append((f"{m.group(1)}:{m.group(2)}".lower().replace(" ", ""), m.group(3) or ""))
            elif b.startswith("build.gradle"):
                decls += [(f"{m.group(1)}:{m.group(2)}".lower(), m.group(3) or "") for m in re.finditer(r"['\"]([\w.-]+):([\w.-]+)(?::([\w.\[\],+-]+))?['\"]", text)]
            elif b == "pubspec.yaml":
                sec = None
                for ln in text.splitlines():
                    if re.match(r"^(dependencies|dev_dependencies|dependency_overrides):", ln):
                        sec = True
                        continue
                    if re.match(r"^\S", ln):
                        sec = None
                    m = re.match(r"^\s+([A-Za-z_][\w]*)\s*:\s*(\S.*)?$", ln)
                    if sec and m:
                        decls.append((_norm(m.group(1)), (m.group(2) or "").strip()))
            elif b == "settings.ini":
                for m in re.finditer(r"(?m)^\s*(?:dev_|conda_)?requirements\s*=\s*(.+)$", text):
                    for tok in m.group(1).split():
                        mm = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", tok)
                        if mm:
                            decls.append((_norm(mm.group(1)), tok[len(mm.group(1)):]))
            elif b == "mix.exs":
                decls += [(m.group(1).lower(), m.group(2) or "") for m in re.finditer(r"\{:(\w+)\s*,\s*(?:\"([^\"]*)\")?", text)]
            elif b == "build.sbt":
                decls += [(f"{m.group(1)}:{m.group(2)}".lower(), m.group(3)) for m in re.finditer(r'"([\w.\-]+)"\s*%%?\s*"([\w.\-]+)"\s*%\s*"([^"]+)"', text)]
            elif b == "Package.swift":
                decls += [(_norm(m.group(1)), "") for m in re.finditer(r'\.package\(\s*url:\s*"[^"]*/([\w.\-]+?)(?:\.git)?"', text)]
            elif b in ("setup.cfg", "setup.py"):
                for s in re.findall(r"['\"]([A-Za-z0-9][A-Za-z0-9._-]*)\s*([<>=!~][^'\"]*)?['\"]", text):
                    decls.append((_norm(s[0]), s[1].strip()))
            else:
                continue
            parsed += 1
        except Exception:  # noqa: BLE001
            del decls[before:]
    return decls, parsed


def _nums(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def major_range(spec: str) -> Optional[tuple[int, Optional[int]]]:
    """(lowest allowed major, highest allowed major or None) for a constraint string, or None if opaque."""
    s = (spec or "").strip()
    if s in ("", "*", "latest"):
        return (0, None)
    if re.match(r"^(workspace:|file:|link:|git|https?:|\.|/)", s):
        return None
    try:
        ss = SpecifierSet(s.replace(" ", ""))  # PEP 440
        lows, ups = [], []
        for sp in ss:
            n = _nums(sp.version)
            if not n:
                continue
            if sp.operator in (">=", ">", "==", "~=", "==="):
                lows.append(n[0])
            if sp.operator in ("==", "===", "<=", "~="):
                ups.append(n[0])
            if sp.operator == "<":
                ups.append(n[0] - 1 if all(x == 0 for x in n[1:]) else n[0])
        if lows or ups:
            return (min(lows) if lows else 0, max(ups) if ups else None)
    except InvalidSpecifier:
        pass
    alts = [a for a in s.split("||")]
    lo, hi, unb = [], [], False
    for alt in alts:
        toks = re.findall(r"(\^|~>|~=|~|>=|<=|>|<|==|=)?\s*v?(\d+(?:\.[\dxX*]+)*)", alt.replace(",", " "))
        if not toks:
            return None
        a_lo, a_hi = None, None
        for op, v in toks:
            n = _nums(v)
            if op in ("^", "~", "~>", "~=", "=", "==", ""):
                a_lo = n[0] if a_lo is None else max(a_lo, n[0])
                a_hi = n[0] if a_hi is None else min(a_hi, n[0])
            elif op in (">=", ">"):
                a_lo = n[0] if a_lo is None else max(a_lo, n[0])
            elif op == "<":
                x = n[0] - 1 if all(y == 0 for y in n[1:]) else n[0]
                a_hi = x if a_hi is None else min(a_hi, x)
            elif op == "<=":
                a_hi = n[0] if a_hi is None else min(a_hi, n[0])
        lo.append(a_lo if a_lo is not None else 0)
        if a_hi is None:
            unb = True
        else:
            hi.append(a_hi)
    return (min(lo), None if unb else max(hi))


def verdict_t4(g: Git, a: dict[str, Any], cache: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    if "decls" not in cache:
        cache["decls"] = _declared(g)
    decls, parsed = cache["decls"]
    n = _norm(a["name"])
    last = re.split(r"[/:]", n)[-1]
    exact = [d for d in decls if d[0] == n]
    suffix = [d for d in decls if d[0] != n and last in [x for x in re.split(r"[/:]", d[0]) if x and not re.fullmatch(r"v\d+", x)]]
    if parsed == 0:
        return "UnverifiableClaim", "DependencyNoManifest"
    if not exact and not suffix:
        return "ContradictedClaim", "DependencyUndeclared"
    major = a.get("major")
    if major is None:
        return "SupportedClaim", "DependencyConfirmed"
    rngs = [major_range(sp) for _, sp in (exact or suffix)]
    rngs = [r for r in rngs if r is not None]
    if not rngs:
        return None, None
    lower = min(r[0] for r in rngs)
    unb = any(r[1] is None for r in rngs)
    upper = None if unb else max(r[1] for r in rngs)
    if lower - major > 0:
        return "ContradictedClaim", "DependencyMajorTooOld"
    if upper is not None and upper - major < 0:
        return "ContradictedClaim", "DependencyMajorTooNew"
    return "SupportedClaim", "DependencyMajorInRange"


# ----------------------------------------------------------------------------
# T5 command
# ----------------------------------------------------------------------------
_BINS = {"tsc", "eslint", "prettier", "vitest", "jest", "playwright", "tsx", "ts-node", "nodemon", "vite", "turbo", "nx",
         "next", "nuxt", "astro", "biome", "prisma", "drizzle-kit", "wrangler", "storybook", "rimraf", "concurrently",
         "husky", "lint-staged", "changeset", "lerna", "mocha", "cypress", "tsup", "webpack", "rollup", "esbuild",
         "tailwindcss", "cdk", "expo", "pnpm", "yarn", "npm", "npx"}


def verdict_t5(g: Git, a: dict[str, Any], cache: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    kind, target = a["kind"], a["target"]
    if "idx" not in cache:
        scripts: dict[str, list[str]] = {}
        comp: dict[str, list[str]] = {}
        make: dict[str, list[str]] = {}
        hosts = {"script": 0, "composer": 0, "make": 0}
        dyn = 0
        for f in g.files:
            b = f.rsplit("/", 1)[-1]
            if any(s in f for s in _SKIP):
                continue
            if b == "package.json":
                t = g.show(f)
                try:
                    d = json.loads(t or "")
                except Exception:  # noqa: BLE001
                    continue
                hosts["script"] += 1
                for k in (d.get("scripts") or {}):
                    scripts.setdefault(k, []).append(f)
            elif b == "composer.json":
                t = g.show(f)
                try:
                    d = json.loads(t or "")
                except Exception:  # noqa: BLE001
                    continue
                hosts["composer"] += 1
                for k in (d.get("scripts") or {}):
                    comp.setdefault(k, []).append(f)
            elif b in ("Makefile", "GNUmakefile", "makefile") or f.endswith(".mk"):
                t = g.show(f)
                if t is None:
                    continue
                hosts["make"] += 1
                for ln in t.splitlines():
                    m = re.match(r"^([^\s#\t.=][^:=#]*?)\s*::?(?!=)", ln)
                    if not m:
                        continue
                    for tg in m.group(1).split():
                        if "%" in tg or "$(" in tg or "${" in tg:
                            dyn += 1
                        else:
                            make.setdefault(tg, []).append(f)
        cache["idx"] = {"script": scripts, "composer": comp, "make": make, "hosts": hosts, "dyn": dyn}
    ix = cache["idx"]
    defs = ix[kind].get(target, [])
    root = [f for f in defs if "/" not in f]
    if kind == "script" and target == "start" and not defs and "server.js" in g.files:
        root = ["server.js"]
    if root or defs:
        return "SupportedClaim", "ScriptDefinedRoot" if root else "ScriptDefinedNested"
    dyn = ix["dyn"] if kind == "make" else 0
    if kind == "script" and a.get("runner") != "npm":
        if "decls" not in cache:
            cache["decls"] = _declared(g)
        names = {d[0] for d in cache["decls"][0]}
        if _norm(target) in names or target in _BINS:
            dyn = 1
    if dyn >= 1:
        return "UnverifiableClaim", "ScriptPossiblyDynamic"
    if ix["hosts"][kind] >= 1:
        return "ContradictedClaim", "ScriptUndefined"
    return "ContradictedClaim", "ScriptHostMissing"


# ----------------------------------------------------------------------------
# T6 version (own parsing; packaging for Python constraints)
# ----------------------------------------------------------------------------
def _vtuple(s: str) -> Optional[tuple[int, ...]]:
    m = re.match(r"^\s*v?(\d+(?:\.\d+){0,3})", s.replace("_", "."))
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _trunc(v: tuple[int, ...], p: int) -> tuple[int, ...]:
    return (tuple(v) + (0,) * 4)[:p]


def _src_values(g: Git, spec: dict[str, Any], subject: str) -> list[tuple[str, str, str]]:
    """(path, raw, kind) for every source; written independently of measure.py."""
    out = []
    for sp in spec:
        if sp.get("fallback_only"):
            continue
        pat = sp["path"]
        root_only = pat.startswith("/")
        pat = pat.lstrip("/")
        for f in g.files:
            if root_only and "/" in f:
                continue
            base = f.rsplit("/", 1)[-1]
            if not (fnmatch.fnmatch(base if not root_only and "*" not in pat else f, pat) if "*" in pat else (f == pat if root_only else base == pat)):
                continue
            if any(s in f for s in _SKIP):
                continue
            text = g.show(f)
            if text is None:
                continue
            raw: Optional[str] = None
            ps = sp["parser"]
            if ps in ("json", "toml"):
                try:
                    cur: Any = json.loads(text) if ps == "json" else tomllib.loads(text)
                except Exception:  # noqa: BLE001
                    continue
                for seg in sp["key"].split("."):
                    cur = cur.get(seg) if isinstance(cur, dict) else None
                raw = cur if isinstance(cur, str) else None
            elif ps == "text":
                raw = next((ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()), None)
            elif ps == "regex":
                m = re.search(sp["key"], text)
                raw = next((x for x in (m.groups() if m else ()) if x), None)
            elif ps == "ini":
                sec, _, opt = sp["key"].rpartition(".")
                m = re.search(r"(?ms)^\[" + re.escape(sec) + r"\]\s*(.*?)(?=^\[|\Z)", text) if sec != "DEFAULT" else re.search(r"(?ms)\A(.*)", text)
                mm = re.search(r"(?m)^" + re.escape(opt) + r"\s*[=:]\s*(.+)$", m.group(1)) if m else None
                raw = mm.group(1).strip() if mm else None
            elif ps == "toolversions":
                mm = re.search(r"(?m)^" + re.escape(sp["key"]) + r"\s+(\S+)", text)
                raw = mm.group(1) if mm else None
            if raw:
                out.append((f, raw, sp["kind"]))
    return out


def verdict_t6(g: Git, a: dict[str, Any], vsrc: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    subject = a["subject_type"]
    claim = _vtuple(a["version"])
    if claim is None:
        return None, None
    prec = a["precision"]
    agree = disagree = 0
    for path, raw, kind in _src_values(g, vsrc[subject], subject):
        raw = raw.strip()
        if subject == "java":
            raw = re.sub(r"^1\.(\d+)", r"\1", raw)
        lo = hi = None
        incl = True
        if kind in ("pin",) or (kind == "constraint" and re.fullmatch(r"v?\d+(\.\d+){0,3}", raw)):
            lo = hi = _vtuple(raw)
        elif kind == "min":
            lo = _vtuple(raw)
            hi = None
        else:
            try:
                ss = SpecifierSet(raw.replace(" ", ""))
                los, his = [], []
                for sp in ss:
                    v = _vtuple(sp.version)
                    if v is None:
                        continue
                    if sp.operator in (">=", ">", "==", "~=", "==="):
                        los.append(v)
                    if sp.operator in ("==", "===", "<=", "~="):
                        his.append(v if sp.operator != "~=" else (v[0],) if len(v) <= 2 else v[:2])
                    if sp.operator == "<":
                        his.append(v)
                        incl = False
                lo = max(los) if los else (0,)
                hi = min(his) if his else None
            except InvalidSpecifier:
                alt_lo, alt_hi, unbounded = [], [], False
                for alt in raw.split("||"):
                    nums = re.findall(r"(\^|~|>=|<=|>|<|=)?\s*v?(\d+(?:\.\d+){0,3})", alt)
                    if not nums:
                        continue
                    los, his = [], []
                    for op, v in nums:
                        t = tuple(int(x) for x in v.split("."))
                        if op in ("^", "~", "=", ""):
                            los.append(t)
                            his.append(t[:1] if op == "^" else t[:2] if op == "~" and len(t) > 1 else t[:1])
                        elif op in (">=", ">"):
                            los.append(t)
                        elif op in ("<", "<="):
                            his.append(t)
                            incl = op == "<="
                    alt_lo.append(max(los) if los else (0,))
                    if his:
                        alt_hi.append(min(his))
                    else:
                        unbounded = True
                if not alt_lo:
                    continue
                lo = min(alt_lo)
                hi = None if unbounded else max(alt_hi)
        if lo is None:
            continue
        pe = min(prec, len(lo))
        ldelta = (_trunc(lo, pe) > _trunc(claim, pe)) - (_trunc(lo, pe) < _trunc(claim, pe))
        if a["qualifier"] == "min":
            ok = ldelta == 0
        else:
            ok = ldelta <= 0
            if ok and hi is not None:
                pu = min(prec, len(hi))
                th, tc = _trunc(hi, pu), _trunc(claim, pu)
                beyond_zero = all(x == 0 for x in hi[pu:])
                if incl or not beyond_zero:
                    ok = th >= tc
                else:
                    ok = th > tc
        agree += int(ok)
        disagree += int(not ok)
    if agree and disagree:
        return "ConflictedClaim", "VersionSourcesConflict"
    if agree:
        return "SupportedClaim", "VersionConfirmed"
    if disagree:
        return "ContradictedClaim", "VersionContradicted"
    return "UnverifiableClaim", "NoVersionSource"


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def recheck_claim(g: Git, c: dict[str, Any], thr: dict[str, Any], sigs: dict[str, Any], vsrc: dict[str, Any],
                  cache: dict[str, Any], tool_cache: dict[str, int]) -> tuple[Optional[str], Optional[str]]:
    t, a = c["tcode"], c["attrs"]
    if int(c["votes"]) < int(thr["wp2_consensus_min_votes"]):
        return None, None
    try:
        if t == "T1" or (t == "T2" and a.get("status_kind") == "protected"):
            return verdict_t1(g, a)
        if t == "T2":
            return verdict_t2(g, a, thr)
        if t == "T3":
            return verdict_t3(g, a, sigs, tool_cache)
        if t == "T4":
            return verdict_t4(g, a, cache)
        if t == "T5":
            return verdict_t5(g, a, cache)
        if t == "T6":
            return verdict_t6(g, a, vsrc)
    except Exception as exc:  # noqa: BLE001
        return None, f"error:{type(exc).__name__}"
    return None, None
