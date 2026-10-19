"""Dependency census and version-constraint parsing for WP2 (Paper 1, RQ2; design T4 and T6).

``census(repo)`` parses every tracked dependency manifest at the snapshot (package.json,
pyproject, requirements*.txt, setup.cfg, Pipfile, go.mod, Cargo.toml, composer.json, Gemfile,
*.csproj, pom.xml, build.gradle) into declarations. ``parse_constraint`` turns a constraint
string (npm, PEP 440, Cargo, composer, Ruby) into numeric bounds. Measurement only: counts and
bounds, never a verdict. recheck.py has its own independent implementations.

Usage:
    imported by wp2/measure.py
"""

from __future__ import annotations

import json
import logging
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Optional

from .repo import RepoAt

logger = logging.getLogger("wp2.depparse")

Ver = tuple[int, ...]
SKIP_DIRS = ("node_modules/", "vendor/", "third_party/", "third-party/", "site-packages/", ".yarn/")
MAX_MANIFESTS = 400


@dataclass
class Decl:
    name: str
    spec: str
    ecosystem: str
    file: str


@dataclass
class Census:
    decls: list[Decl] = field(default_factory=list)
    parsed: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    truncated: bool = False


def norm_name(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n.strip().lower())


def last_segment(n: str) -> str:
    return re.split(r"[/:]", n)[-1]


def segments(n: str) -> list[str]:
    """Path segments of a declared module name (Go ``/vN`` suffixes dropped); a claim that names one segment matches."""
    return [x for x in re.split(r"[/:]", n) if x and not re.fullmatch(r"v\d+", x)]


# ---------------------------------------------------------------------------
# constraint parsing
# ---------------------------------------------------------------------------

def _ver(s: str) -> Optional[Ver]:
    s = s.strip().lstrip("vV=")
    m = re.match(r"^(\d+(?:\.\d+){0,3})", s)
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def _bump(v: Ver, idx: int) -> Ver:
    """Exclusive upper bound: increment component ``idx``, drop the rest."""
    base = list(v[: idx + 1]) + [0] * max(0, idx + 1 - len(v))
    base[idx] += 1
    return tuple(base)


def _caret_upper(v: Ver) -> Ver:
    for i, c in enumerate(v):
        if c != 0:
            return _bump(v, i)
    return _bump(v, max(0, len(v) - 1)) if len(v) > 1 else (1,)


def _alt(alt: str, bare_caret: bool) -> Optional[tuple[Optional[Ver], Optional[Ver], bool]]:
    """One alternative: (lower, upper, upper_inclusive); None if unparseable."""
    alt = alt.strip()
    if alt in ("", "*", "x", "X", "latest"):
        return (0,), None, False
    m = re.match(r"^(\S+)\s+-\s+(\S+)$", alt)
    if m:
        lo, hi = _ver(m.group(1)), _ver(m.group(2))
        return (lo, hi, True) if lo and hi else None
    alt = re.sub(r"([<>=!~^]+)\s+(?=\d|v\d)", r"\1", alt.replace(",", " "))
    lower: Optional[Ver] = None
    upper: Optional[Ver] = None
    incl = False

    def add(lo: Optional[Ver], up: Optional[Ver], up_incl: bool) -> None:
        nonlocal lower, upper, incl
        if lo is not None:
            lower = max(lower, lo) if lower else lo
        if up is not None and (upper is None or up < upper):
            upper, incl = up, up_incl

    for tok in alt.split():
        if tok in ("*", "x", "X"):
            lower = lower or (0,)
            continue
        m = re.match(r"^(\^|~=|~>|~|>=|<=|===|==|=|>|<|!=)?\s*(.+)$", tok)
        if not m:
            return None
        op, rest = m.group(1) or "", m.group(2)
        wild = re.match(r"^v?(\d+(?:\.\d+){0,2})\.(?:[xX*])(?:\.[xX*])?$", rest)
        v = _ver(wild.group(1)) if wild else _ver(rest)
        if v is None:
            return None
        if op == "!=":
            continue
        if wild and op in ("", "=", "==", "==="):
            add(v, _bump(v, len(v) - 1), False)
        elif op == "^" or (op == "" and bare_caret):
            add(v, _caret_upper(v), False)
        elif op == "~":
            add(v, _bump(v, 1) if len(v) >= 2 else _bump(v, 0), False)
        elif op in ("~=", "~>"):
            add(v, _bump(v, len(v) - 2) if len(v) >= 3 else _bump(v, 0), False)
        elif op in (">=", ">"):
            add(v, None, False)
        elif op == "<":
            add(None, v, False)
        elif op == "<=":
            add(None, v, True)
        else:  # pin: bare version, =, ==, ===
            add(v, v, True)
    if lower is None and upper is None:
        return None
    return lower, upper, incl


def parse_constraint(spec: str, bare_caret: bool = False) -> Optional[dict[str, Any]]:
    """Parse a constraint string to ``{lower, upper, upper_inclusive}`` (Ver tuples; upper None = unbounded).

    Alternatives (``||``): lowest lower bound, highest upper bound; unbounded if any alternative is.
    Returns None if the string cannot be parsed (git URLs, workspace:, file:, tags ...).
    """
    if spec is None:
        return None
    s = str(spec).split(";")[0].strip()
    if re.match(r"^(workspace:|file:|link:|git\+|git:|github:|https?:|npm:|catalog:|portal:|patch:|\.|/|[a-z]+://)", s):
        if s.startswith("npm:"):
            s = s.rsplit("@", 1)[-1]
        else:
            return None
    alts = [_alt(a, bare_caret) for a in re.split(r"\|\|", s)]
    if any(a is None for a in alts):
        return None
    lowers = [a[0] if a[0] is not None else (0,) for a in alts]
    ups = [a for a in alts]
    unbounded = any(a[1] is None for a in ups)
    out: dict[str, Any] = {"lower": min(lowers), "upper": None if unbounded else max(a[1] for a in ups),
                           "upper_inclusive": all(a[2] for a in ups if a[1] is not None) if not unbounded else False}
    return out


def upper_major(c: dict[str, Any]) -> Optional[int]:
    """Highest major version allowed by the upper bound (None if unbounded)."""
    u = c["upper"]
    if u is None:
        return None
    if c["upper_inclusive"]:
        return u[0]
    return u[0] - 1 if all(x == 0 for x in u[1:]) else u[0]


# ---------------------------------------------------------------------------
# manifest parsers
# ---------------------------------------------------------------------------

_PEP508 = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _pep508(line: str, eco: str, f: str, out: list[Decl]) -> None:
    line = line.split("#")[0].strip()
    if not line or line.startswith(("-", ".", "/")) or "://" in line.split(";")[0] and "@" not in line:
        return
    m = _PEP508.match(line)
    if m:
        name = m.group(1)
        spec = m.group(2).split(";")[0].strip()
        spec = re.sub(r"^\(|\)$", "", spec).strip()
        if spec.startswith("@"):
            spec = ""
        out.append(Decl(norm_name(name), spec, eco, f))


def _parse_package_json(text: str, f: str) -> list[Decl]:
    d = json.loads(text)
    out = []
    for sec in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        for n, v in (d.get(sec) or {}).items():
            out.append(Decl(n.lower(), str(v), "npm", f))
    return out


def _toml_dep_table(tbl: dict[str, Any], eco: str, f: str, out: list[Decl]) -> None:
    for n, v in (tbl or {}).items():
        if n.lower() == "python":
            continue
        spec = v if isinstance(v, str) else (v.get("version", "") if isinstance(v, dict) else "")
        if isinstance(v, list):
            spec = ""
        out.append(Decl(norm_name(n), str(spec), eco, f))


def _parse_pyproject(text: str, f: str) -> list[Decl]:
    d = tomllib.loads(text)
    out: list[Decl] = []
    proj = d.get("project", {})
    for line in proj.get("dependencies", []) or []:
        _pep508(str(line), "pypi", f, out)
    for _, lst in (proj.get("optional-dependencies") or {}).items():
        for line in lst:
            _pep508(str(line), "pypi", f, out)
    for _, lst in (d.get("dependency-groups") or {}).items():
        for line in lst:
            if isinstance(line, str):
                _pep508(line, "pypi", f, out)
    for line in (d.get("build-system", {}).get("requires") or []):
        _pep508(str(line), "pypi", f, out)
    poetry = d.get("tool", {}).get("poetry", {})
    _toml_dep_table(poetry.get("dependencies"), "pypi", f, out)
    _toml_dep_table(poetry.get("dev-dependencies"), "pypi", f, out)
    for _, g in (poetry.get("group") or {}).items():
        _toml_dep_table(g.get("dependencies"), "pypi", f, out)
    uv = d.get("tool", {}).get("uv", {})
    for line in uv.get("dev-dependencies", []) or []:
        _pep508(str(line), "pypi", f, out)
    for _, lst in (d.get("tool", {}).get("pdm", {}).get("dev-dependencies") or {}).items():
        for line in lst:
            _pep508(str(line), "pypi", f, out)
    return out


def _parse_requirements(text: str, f: str) -> list[Decl]:
    out: list[Decl] = []
    for ln in text.splitlines():
        _pep508(ln, "pypi", f, out)
    return out


def _parse_setup_cfg(text: str, f: str) -> list[Decl]:
    import configparser

    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.read_string(text)
    out: list[Decl] = []
    if cp.has_option("options", "install_requires"):
        for ln in cp.get("options", "install_requires").splitlines():
            _pep508(ln, "pypi", f, out)
    if cp.has_section("options.extras_require"):
        for _, v in cp.items("options.extras_require"):
            for ln in v.splitlines():
                _pep508(ln, "pypi", f, out)
    return out


def _parse_setup_py(text: str, f: str) -> list[Decl]:
    m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
    if not m:
        raise ValueError("no install_requires list")
    out: list[Decl] = []
    for s in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
        _pep508(s, "pypi", f, out)
    return out


def _parse_pipfile(text: str, f: str) -> list[Decl]:
    d = tomllib.loads(text)
    out: list[Decl] = []
    for sec in ("packages", "dev-packages"):
        for n, v in (d.get(sec) or {}).items():
            spec = v if isinstance(v, str) else (v.get("version", "") if isinstance(v, dict) else "")
            out.append(Decl(norm_name(n), "" if spec == "*" else str(spec), "pypi", f))
    return out


def _parse_gomod(text: str, f: str) -> list[Decl]:
    out: list[Decl] = []
    in_block = False
    for ln in text.splitlines():
        s = ln.split("//")[0].strip()
        if s.startswith("require ("):
            in_block = True
            continue
        if in_block and s == ")":
            in_block = False
            continue
        m = re.match(r"^(?:require\s+)?(\S+)\s+(v[0-9][^\s]*)$", s) if (in_block or s.startswith("require ")) else None
        if m:
            out.append(Decl(m.group(1).lower(), m.group(2), "go", f))
    return out


def _parse_cargo(text: str, f: str) -> list[Decl]:
    d = tomllib.loads(text)
    out: list[Decl] = []

    def tab(t: Any) -> None:
        for n, v in (t or {}).items():
            spec = v if isinstance(v, str) else (v.get("version", "") if isinstance(v, dict) else "")
            out.append(Decl(norm_name(n), str(spec), "cargo", f))

    for sec in ("dependencies", "dev-dependencies", "build-dependencies"):
        tab(d.get(sec))
    tab(d.get("workspace", {}).get("dependencies"))
    for _, t in (d.get("target") or {}).items():
        for sec in ("dependencies", "dev-dependencies", "build-dependencies"):
            tab(t.get(sec))
    return out


def _parse_composer(text: str, f: str) -> list[Decl]:
    d = json.loads(text)
    out = []
    for sec in ("require", "require-dev"):
        for n, v in (d.get(sec) or {}).items():
            out.append(Decl(n.lower(), str(v), "composer", f))
    return out


def _parse_gemfile(text: str, f: str) -> list[Decl]:
    out = []
    for m in re.finditer(r"^\s*gem\s+['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?", text, re.M):
        out.append(Decl(m.group(1).lower(), m.group(2) or "", "gem", f))
    return out


def _parse_csproj(text: str, f: str) -> list[Decl]:
    out = []
    for m in re.finditer(r"<(?:PackageReference|PackageVersion)\b([^>]*)>", text):
        a = m.group(1)
        n = re.search(r'Include="([^"]+)"', a)
        v = re.search(r'Version="([^"]*)"', a)
        if n:
            out.append(Decl(n.group(1).lower(), v.group(1) if v else "", "nuget", f))
    return out


def _parse_pom(text: str, f: str) -> list[Decl]:
    out = []
    for m in re.finditer(r"<dependency>(.*?)</dependency>", text, re.S):
        g = re.search(r"<groupId>\s*([^<\s]+)\s*</groupId>", m.group(1))
        a = re.search(r"<artifactId>\s*([^<\s]+)\s*</artifactId>", m.group(1))
        v = re.search(r"<version>\s*([^<\s]+)\s*</version>", m.group(1))
        if a:
            out.append(Decl(f"{g.group(1) if g else ''}:{a.group(1)}".lower(), v.group(1) if v else "", "maven", f))
    return out


def _parse_gradle(text: str, f: str) -> list[Decl]:
    out = []
    for m in re.finditer(r"\b(?:implementation|api|compileOnly|runtimeOnly|testImplementation|testCompileOnly|"
                         r"kapt|annotationProcessor|classpath)\s*\(?\s*['\"]([^:'\"]+):([^:'\"]+)(?::([^'\"]+))?['\"]", text):
        out.append(Decl(f"{m.group(1)}:{m.group(2)}".lower(), m.group(3) or "", "maven", f))
    return out


def _parse_pubspec(text: str, f: str) -> list[Decl]:
    import yaml

    d = yaml.safe_load(text) or {}
    out = []
    for sec in ("dependencies", "dev_dependencies", "dependency_overrides"):
        for n, v in (d.get(sec) or {}).items():
            out.append(Decl(norm_name(str(n)), v if isinstance(v, str) else "", "pub", f))
    return out


def _parse_mix(text: str, f: str) -> list[Decl]:
    return [Decl(m.group(1).lower(), m.group(2) or "", "hex", f)
            for m in re.finditer(r"\{:(\w+)\s*,\s*(?:\"([^\"]*)\")?", text)]


def _parse_sbt(text: str, f: str) -> list[Decl]:
    return [Decl(f"{m.group(1)}:{m.group(2)}".lower(), m.group(3), "maven", f)
            for m in re.finditer(r'"([\w.\-]+)"\s*%%?\s*"([\w.\-]+)"\s*%\s*"([^"]+)"', text)]


def _parse_swiftpm(text: str, f: str) -> list[Decl]:
    return [Decl(norm_name(m.group(1)), "", "swift", f)
            for m in re.finditer(r'\.package\(\s*url:\s*"[^"]*/([\w.\-]+?)(?:\.git)?"', text)]


def _parse_gemspec(text: str, f: str) -> list[Decl]:
    return [Decl(m.group(1).lower(), "", "gem", f)
            for m in re.finditer(r"add_(?:runtime_|development_)?dependency\s*\(?\s*['\"]([^'\"]+)['\"]", text)]


def _parse_settings_ini(text: str, f: str) -> list[Decl]:
    """nbdev settings.ini: space-separated ``requirements`` / ``dev_requirements`` / ``conda_requirements``."""
    out = []
    for m in re.finditer(r"(?m)^\s*(?:dev_|conda_)?requirements\s*=\s*(.+)$", text):
        for tok in m.group(1).split():
            mm = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", tok)
            if mm:
                out.append(Decl(norm_name(mm.group(1)), mm.group(2), "pypi", f))
    return out


def _kind(path: str) -> Optional[str]:
    base = path.rsplit("/", 1)[-1]
    if any(path.startswith(d) or ("/" + d) in path for d in SKIP_DIRS):
        return None
    if base == "package.json":
        return "package.json"
    if base == "pyproject.toml":
        return "pyproject"
    if (base.startswith("requirements") and base.endswith(".txt")) or (
            "/requirements/" in "/" + path and base.endswith(".txt")) or base.endswith("-requirements.txt"):
        return "requirements"
    if base == "setup.cfg":
        return "setup.cfg"
    if base == "setup.py":
        return "setup.py"
    if base == "Pipfile":
        return "pipfile"
    if base == "go.mod":
        return "gomod"
    if base == "Cargo.toml":
        return "cargo"
    if base == "composer.json":
        return "composer"
    if base == "Gemfile":
        return "gemfile"
    if base.endswith((".csproj", ".fsproj")) or base == "Directory.Packages.props":
        return "csproj"
    if base == "pom.xml":
        return "pom"
    if base in ("build.gradle", "build.gradle.kts"):
        return "gradle"
    if base == "settings.ini":
        return "settings.ini"
    if base == "pubspec.yaml":
        return "pubspec"
    if base == "mix.exs":
        return "mix"
    if base == "build.sbt":
        return "sbt"
    if base == "Package.swift":
        return "swiftpm"
    if base.endswith(".gemspec"):
        return "gemspec"
    return None


PARSERS = {"package.json": _parse_package_json, "pyproject": _parse_pyproject, "requirements": _parse_requirements,
           "setup.cfg": _parse_setup_cfg, "setup.py": _parse_setup_py, "pipfile": _parse_pipfile,
           "gomod": _parse_gomod, "cargo": _parse_cargo, "composer": _parse_composer, "gemfile": _parse_gemfile,
           "csproj": _parse_csproj, "pom": _parse_pom, "gradle": _parse_gradle, "pubspec": _parse_pubspec,
           "settings.ini": _parse_settings_ini, "mix": _parse_mix, "sbt": _parse_sbt, "swiftpm": _parse_swiftpm, "gemspec": _parse_gemspec}


def manifest_files(repo: RepoAt) -> list[str]:
    files = [p for p in repo.files if _kind(p)]
    files.sort(key=lambda p: (p.count("/"), p))
    return files


def census(repo: RepoAt) -> Census:
    """Parse every tracked dependency manifest at the snapshot."""
    files = manifest_files(repo)
    c = Census()
    if len(files) > MAX_MANIFESTS:
        files, c.truncated = files[:MAX_MANIFESTS], True
    repo.prefetch(files)
    for p in files:
        text = repo.read(p)
        if text is None:
            c.unparsed.append(p)
            continue
        try:
            c.decls += PARSERS[_kind(p)](text, p)  # type: ignore[index]
            c.parsed.append(p)
        except Exception as exc:  # noqa: BLE001 - one broken manifest must not stop the census
            logger.debug("unparsed %s: %s", p, exc)
            c.unparsed.append(p)
    return c
