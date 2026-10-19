"""WP2 mechanical validation of extracted claims (Paper 1, RQ2; design section 5.3).

Per model, before consensus: span grounding (the quote must be verbatim in the chunk, the
subject verbatim in the quote), type-specific normalization and shape filters, and the
canonical claim key. A claim that fails any step is rejected with a counted reason. No LLM
and no repository access is used here: this is text-only validation.

Usage:
    imported by wp2/consensus.py and wp2/step_consensus.py
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from . import common as C

SYSTEM_PREFIXES = ("/usr", "/etc", "/tmp", "/var", "/opt", "/home", "/dev", "/proc", "/bin", "/sbin", "/lib", "/mnt")
BUILTINS = {"install", "i", "ci", "add", "remove", "exec", "dlx", "x", "create", "init", "publish", "version",
            "audit", "update", "outdated", "link", "pack", "why", "ls", "list", "config", "run", "rm", "uninstall",
            "up", "upgrade", "self-update", "global", "cache", "help", "login", "logout", "whoami", "view", "info",
            "pnpx", "npx", "dedupe", "prune", "rebuild", "root", "bin", "cache-clean", "set", "get"}
SCRIPT_RUNNERS = {"npm", "pnpm", "yarn", "bun"}
KNOWN_ROOT_FILES = {"makefile", "dockerfile", "license", "readme", "changelog", "gemfile", "rakefile", "procfile",
                    "justfile", "vagrantfile", "jenkinsfile", "codeowners", "contributing", "gnumakefile"}
T6_SUBJECTS = {"python", "node", "go", "java", "dotnet", "rust", "php", "ruby", "project_self"}
PATH_REJECT_SUBSTR = ("://", "~", "$", "<", ">", "{", "}", "..", "path/to", "your-", "example", "foo/", "xxx",
                      "/bar", "my-project", "myproject", "my_project", "your_", "yourproject", "your/")
PATH_CHARS = re.compile(r"^[\w.\-/@+*?\[\]#%]+$", re.UNICODE)
FILE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")
KNOWN_EXT = {"py", "js", "ts", "tsx", "jsx", "mjs", "cjs", "json", "jsonc", "yaml", "yml", "toml", "md", "mdx", "txt", "cfg",
             "ini", "xml", "html", "htm", "css", "scss", "sass", "less", "sh", "bash", "zsh", "ps1", "bat", "go", "rs", "java",
             "kt", "kts", "scala", "cs", "fs", "vb", "rb", "php", "c", "h", "cc", "cpp", "hpp", "swift", "m", "mm", "dart",
             "lua", "r", "jl", "ex", "exs", "erl", "hs", "clj", "ml", "mli", "sql", "proto", "gradle", "lock", "env", "conf",
             "csv", "tsv", "ipynb", "vue", "svelte", "astro", "tf", "tfvars", "mod", "sum", "pdf", "png", "svg", "jpg", "jpeg",
             "gif", "ico", "webp", "wasm", "zip", "tar", "gz", "csproj", "sln", "props", "targets", "plist", "pem", "key",
             "rules", "tmpl", "j2", "jinja", "erb", "slim", "pug", "gql", "graphql", "prisma", "http", "mk", "cmake", "bzl",
             "nix", "dockerfile", "editorconfig", "gitignore", "npmrc", "nvmrc", "babelrc", "eslintrc", "prettierrc"}
BRACKET_PLACEHOLDER = re.compile(r"\[(feature|name|component|module|file|path|id|slug|type|x|n|\.\.\.|your[\w-]*|my[\w-]*)\]", re.I)
IMPORT_STATEMENT = re.compile(r"\b(import\s[^\n]*\bfrom\s+['\"]|import\s+['\"]|require\(\s*['\"]|from\s+\S+\s+import\b)")

NPM_ID = re.compile(r"^(@[a-z0-9-~][a-z0-9-._~]*/)?[a-z0-9-~][a-z0-9-._~]*$")
PYPI_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
COMPOSER_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*$")
MAVEN_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_.-]+$")
GO_ID = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}/[^\s]+$")
IMPORT_LINE = re.compile(r"^\s*(import\s|from\s+\S+\s+import\s|require\(|use\s+\S+;|using\s+\S+;|#include|const\s+.*=\s*require)")


@lru_cache(maxsize=1)
def tool_vocab() -> dict[str, dict[str, Any]]:
    """Closed T3 vocabulary: canonical tool -> {category, ...}; aliases resolved by ``canonical_tool``."""
    doc = yaml.safe_load((C.EXP_DIR / "configs" / "wp2_tool_signatures.yaml").read_text(encoding="utf-8"))
    return doc["tools"]


@lru_cache(maxsize=1)
def tool_aliases() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, spec in tool_vocab().items():
        out[name.lower()] = name
        for a in spec.get("aliases", []) or []:
            out[str(a).lower()] = name
    return out


def canonical_tool(s: str) -> Optional[str]:
    return tool_aliases().get(s.strip().lower())


_SMART = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}


def norm_text(s: str) -> str:
    """NFKC, smart quotes to ASCII, drop backticks/asterisks/underscores, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    for k, v in _SMART.items():
        s = s.replace(k, v)
    s = re.sub(r"[`*_]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_prefixes(quote: str) -> str:
    return re.sub(r"(?m)^\s*\d{4}\|\s?", "", quote)


def clean_path_token(s: str) -> str:
    s = s.strip()
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        s = s[2:-2]
    s = s.strip("`'\"").strip()
    s = s.rstrip(".,;:)(")
    return s


def normalize_path(raw: str, manifest_path: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Normalize a path subject; returns ({path, is_dir, glob}, None) or (None, reject_reason)."""
    p = clean_path_token(raw)
    if not p:
        return None, "empty_subject"
    low = p.lower()
    if any(x in low for x in PATH_REJECT_SUBSTR):
        return None, "placeholder_or_url"
    if p.startswith("-") or " " in p or "\t" in p:
        return None, "not_a_path"
    if p.startswith("@"):
        return None, "alias_or_package_specifier"
    if BRACKET_PLACEHOLDER.search(p):
        return None, "bracket_placeholder"
    if p.startswith("/"):
        if p.startswith(SYSTEM_PREFIXES):
            return None, "system_path"
        p = p.lstrip("/")
    while p.startswith("./"):
        p = p[2:]
    p = re.sub(r"/{2,}", "/", p)
    is_dir = p.endswith("/")
    p = p.rstrip("/")
    if not p or p == ".":
        return None, "empty_path"
    if not PATH_CHARS.match(p):
        return None, "bad_chars"
    base = p.rsplit("/", 1)[-1]
    if re.fullmatch(r"\*?\.(py|js|ts|tsx|jsx|md|json|ya?ml|toml|txt|go|rs|java|kt|cs|rb|php|c|h|cpp|swift|sh|css|html)", p):
        return None, "bare_extension"
    if "/" not in p and p.lower() not in KNOWN_ROOT_FILES:
        ext = p.rsplit(".", 1)[-1].lower() if "." in p else ""
        dotfile_ok = p.startswith(".") and p.count(".") == 1  # .gitignore, .nvmrc, .env
        if not (ext in KNOWN_EXT or dotfile_ok) and not (is_dir and "." not in p):
            return None, "not_path_shaped"
    if p.lower() == manifest_path.lower() or (("/" not in p) and p.lower() in {"claude.md", "agents.md"}):
        return None, "manifest_own_path"
    glob = any(ch in p for ch in "*?")
    return {"path": p, "is_dir": is_dir, "glob": glob, "bracket": "[" in p}, None


def quote_shows_wildcard(subj: str, quote: str) -> bool:
    """True if the subject occurs in the raw quote directly preceded by ``*/``, ``**/`` or a single ``*``.

    WP2c: extraction models sometimes return ``**/providers/x.md`` as ``providers/x.md``; the quote still shows the
    wildcard. A bare ``**`` (markdown bold) or an italic ``*x*`` is not a wildcard.
    """
    t = subj.strip().strip("`'\"").rstrip("/")
    i = quote.find(t) if t else -1
    if i <= 0:
        return False
    before, after = quote[:i], quote[i + len(t):]
    if before.endswith("*/"):
        return True
    m = re.search(r"(\*+)$", before)
    return bool(m and len(m.group(1)) == 1 and not after.startswith("*"))


def first_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def version_code(v: str) -> Optional[tuple[int, int]]:
    """(code, precision) for the first dotted number in v; code = M*1e9 + m*1e6 + p*1e3 + b."""
    m = re.search(r"\d+(?:\.\d+){0,3}", v or "")
    if not m:
        return None
    parts = [int(x) for x in m.group(0).split(".")]
    prec = len(parts)
    parts = (parts + [0, 0, 0, 0])[:4]
    return parts[0] * 10**9 + parts[1] * 10**6 + parts[2] * 10**3 + parts[3], prec


def _locate_line(chunk: dict[str, Any], quote_n: str, claimed: int, tol: int) -> int:
    """Global line of the first line of the quote inside the chunk (corrects off-by-more-than-tol)."""
    lines = chunk["plain"].split("\n")
    first = next((q for q in re.split(r"\n", quote_n) if q.strip()), "")
    first = norm_text(first)[:80]
    for i, ln in enumerate(lines):
        if first and first in norm_text(ln):
            actual = chunk["start_line"] + i
            return actual if abs(actual - claimed) > tol else claimed
    return claimed


_TREE_GLYPHS = re.compile(r"^(?:[\s\u2502\u251c\u2514\u2500|+]|[-*](?=\s))*")


def raw_head_slash(line: str) -> bool:
    """True if the tree entry on this line is written with a trailing slash (a directory)."""
    m = _TREE_GLYPHS.match(line)
    text = line[m.end():].strip().strip("`") if m else line.strip()
    head = re.split(r"\s+#|\s{2,}|\s+<-|\s+-\s", text, maxsplit=1)[0].strip().strip("`")
    return head.endswith("/")


def tree_paths(chunk: dict[str, Any], components: list[str], from_line: int) -> list[dict[str, Any]]:
    """Compose full paths for tree-diagram components from the chunk's indentation (mechanical).

    Each component is located on the first line at or after ``from_line`` whose text, after the
    tree glyphs, starts with the component; depth is the column of that text. A single depth-0
    entry that wraps all others is treated as the repository root and dropped.
    """
    lines = chunk["plain"].split("\n")
    rows: list[tuple[int, str, str, int, bool]] = []
    cursor = max(0, from_line - chunk["start_line"])
    for comp in components:
        comp_c = clean_path_token(comp).rstrip("/")
        found = None
        for i in range(cursor, len(lines)):
            raw = lines[i]
            m = _TREE_GLYPHS.match(raw)
            col = m.end() if m else 0
            text = raw[col:].strip().strip("`")
            head = re.split(r"\s+#|\s{2,}|\s+<-|\s+-\s", text, maxsplit=1)[0].strip().strip("`").rstrip("/")
            if head == comp_c or head.startswith(comp_c + " "):
                found = (col, i)
                break
        if found is None:
            continue
        col, i = found
        slash = comp.strip().strip("`").endswith("/") or raw_head_slash(lines[i])
        rows.append((col, comp_c, lines[i], chunk["start_line"] + i, slash))
        cursor = i + 1
    if not rows:
        return []
    min_col = min(r[0] for r in rows)
    # a single depth-0 entry that wraps all others may be the repository root (name of the repo) or a real
    # directory: keep it in the path and give the root-dropped path as ``alt_path`` (measurement tries it
    # only if the root-kept path resolves to nothing)
    wrapper = len([r for r in rows if r[0] == min_col]) == 1 and len(rows) > 2 and rows[0][0] == min_col
    root_name = rows[0][1] if wrapper else None
    out, stack = [], []
    for k, (col, name, raw, gl, slash) in enumerate(rows):
        while stack and stack[-1][0] >= col:
            stack.pop()
        full = "/".join([s[1] for s in stack] + [name])
        has_children = (k + 1 < len(rows) and rows[k + 1][0] > col) or slash
        alt = full[len(root_name) + 1:] if wrapper and k > 0 and full.startswith(root_name + "/") else None
        out.append({"path": full, "is_dir": has_children, "line": gl, "quote": raw.strip(), "alt_path": alt})
        stack.append((col, name))
    return out


def _repair_quote(chunk: dict[str, Any], subj: str, claimed: Optional[int]) -> Optional[tuple[str, int]]:
    """Mechanical repair for a claim with no quote: the manifest line that contains the subject verbatim.

    Uses the line nearest the claimed line number (first occurrence if none). Returns (line text, global line).
    """
    sn = norm_text(subj)
    if not sn:
        return None
    cands = [(chunk["start_line"] + i, ln) for i, ln in enumerate(chunk["plain"].split("\n")) if sn in norm_text(ln)]
    if not cands:
        return None
    gl, ln = min(cands, key=lambda c: (abs(c[0] - claimed) if claimed else 0, c[0]))
    return ln.strip(), gl


def process_claim(raw: dict[str, Any], chunk: dict[str, Any], manifest_path: str, cfg: dict[str, Any],
                  repair: bool = True, repo_name: str = "") -> tuple[list[dict[str, Any]], list[str]]:
    """Validate one extracted claim. Returns (accepted normalized claims, reject reasons).

    A tree-diagram claim expands to several path claims. Each accepted claim carries ``type``,
    ``key`` (tuple as list), ``subject``, ``quote``, ``line``, ``mode`` and type attributes.
    """
    ex = cfg["extraction"]
    typ = raw["type"]
    chunk_n = norm_text(chunk["plain"])
    repaired = False
    quote = strip_prefixes(raw.get("quote") or "")[: ex["quote_max_chars"]]
    quote_n = norm_text(quote)
    if not quote_n and not (typ == "path_exists" and len([c for c in (raw.get("path_components") or []) if c]) >= 2):
        if not repair:
            return [], ["quote_missing"]
        rq = _repair_quote(chunk, raw.get("subject") or "", raw.get("line"))
        if rq is None:
            return [], ["ungrounded_quote"]
        quote, raw = rq[0][: ex["quote_max_chars"]], {**raw, "line": rq[1]}
        quote_n = norm_text(quote)
        repaired = True
    if raw.get("line") is None:
        raw = {**raw, "line": chunk["start_line"]}
    mode = raw.get("mode") or "asserts_existing"
    if mode not in {"asserts_existing", "to_be_created", "example", "negated"}:
        mode = "example"
    comps = [c for c in (raw.get("path_components") or []) if isinstance(c, str) and c.strip()]

    def reject(reason: str) -> tuple[list[dict[str, Any]], list[str]]:
        return [], [reason]

    # ---- tree diagram: expand mechanically ----
    if typ == "path_exists" and len(comps) >= 2:
        for c in comps:
            if norm_text(c.strip().strip("/")) not in chunk_n:
                return reject("ungrounded_component")
        res, why = [], []
        for t in tree_paths(chunk, comps, int(raw["line"])):
            npth, r = normalize_path(t["path"] + ("/" if t["is_dir"] else ""), manifest_path)
            if npth is None:
                why.append(r or "path_reject")
                continue
            npth["is_dir"] = npth["is_dir"] or t["is_dir"]
            alt_n = (normalize_path(t["alt_path"] + ("/" if t["is_dir"] else ""), manifest_path)[0]
                     if t.get("alt_path") else None)
            res.append(_mk("T1", ["T1", npth["path"]], npth["path"], t["quote"], t["line"], mode,
                           {**npth, "tree": True, "alt_path": alt_n["path"] if alt_n else None}))
        return res, why or ([] if res else ["tree_no_paths"])

    if not quote_n or quote_n not in chunk_n:
        return reject("ungrounded_quote")
    subj = (raw.get("subject") or "").strip()
    if not subj:
        return reject("empty_subject")
    line = raw["line"] if repaired else _locate_line(chunk, quote, int(raw["line"]), ex["line_tolerance"])
    subj_n = norm_text(subj)
    vocab_type = typ in {"tool_used", "version"}
    if typ == "version" and subj.strip().lower() == "project_self":
        pass
    elif vocab_type or typ == "dependency":
        if subj_n.lower() not in quote_n.lower():
            return reject("subject_not_in_quote")
    elif subj_n not in quote_n:
        return reject("subject_not_in_quote")

    if typ in ("path_exists", "path_status") and IMPORT_STATEMENT.search(quote):
        return reject("code_import_not_path_claim")

    if typ == "path_exists":
        npth, r = normalize_path(subj, manifest_path)
        if npth is None:
            return reject(r or "path_reject")
        if quote_shows_wildcard(subj, raw.get("quote") or quote):  # WP2c wildcard fix
            npth["glob"] = True
        return [_mk("T1", ["T1", npth["path"]], npth["path"], quote, line, mode, npth)], []

    if typ == "path_status":
        status = (raw.get("status") or "").lower()
        if status in {"legacy", "deprecated", "frozen", "stable"}:
            kind = "inactive"
        elif status in {"generated", "do_not_edit"}:
            kind = "protected"
        else:
            return reject("bad_status")
        npth, r = normalize_path(subj, manifest_path)
        if npth is None:
            return reject(r or "path_reject")
        if quote_shows_wildcard(subj, raw.get("quote") or quote):  # WP2c wildcard fix
            npth["glob"] = True
        return [_mk("T2", ["T2", npth["path"], kind], npth["path"], quote, line, mode,
                    {**npth, "status": status, "status_kind": kind})], []

    if typ == "tool_used":
        tool = canonical_tool(subj)
        if tool is None:
            return reject("tool_out_of_vocabulary")
        cat = tool_vocab()[tool]["category"]
        return [_mk("T3", ["T3", cat, tool], tool, quote, line, mode, {"tool": tool, "category": cat})], []

    if typ == "dependency":
        if "*" in subj:
            return reject("glob_dependency_name")
        name = subj.strip().strip("`'\"").lower()
        ver = raw.get("version")
        m = re.match(r"^(@?[^@\s]+)@(.+)$", name)
        if m:
            name, ver = m.group(1), ver or m.group(2)
        if " " in name or not name:
            return reject("non_identifier_mention")
        ok = any(rx.match(name) for rx in (NPM_ID, PYPI_ID, COMPOSER_ID, MAVEN_ID, GO_ID))
        if not ok:
            return reject("non_identifier_mention")
        if IMPORT_LINE.match(quote):
            return reject("code_import_not_dependency_claim")
        if repo_name and re.sub(r"[-_.]+", "-", name) == re.sub(r"[-_.]+", "-", repo_name.lower()):
            return reject("self_reference")
        in_code = re.search(r"`[^`]*" + re.escape(name) + r"[^`]*`", quote, re.I) is not None
        install_ctx = re.search(r"(pip3?|uv\s+add|uv\s+pip\s+install|poetry\s+add|npm\s+(?:i|install)|yarn\s+add|pnpm\s+(?:add|i)|bun\s+add|"
                                r"composer\s+require|cargo\s+add|go\s+get|gem\s+install|bundle\s+add|dotnet\s+add\s+package)\b[^\n]*"
                                + re.escape(name), quote, re.I) is not None
        if not (in_code or install_ctx or "/" in name or name.startswith("@")):
            return reject("prose_mention_not_code")
        major = first_int(ver)
        return [_mk("T4", ["T4", name, major], name, quote, line, mode,
                    {"name": name, "major": major, "ecosystem_hint": raw.get("ecosystem_hint")})], []

    if typ == "command":
        runner = (raw.get("runner") or "").lower()
        if runner not in SCRIPT_RUNNERS | {"make", "composer"}:
            return reject("bad_runner")
        tgt = subj.strip().strip("`'\"*").strip()
        tgt = re.sub(r"^(run|run-script)\s+", "", tgt)
        tgt = tgt.split()[0] if tgt.split() else ""
        tgt = tgt.rstrip(".,;:)")
        if not tgt or tgt.startswith("-") or any(ch in tgt for ch in "$<>{}%=|&;\\"):
            return reject("bad_target")
        if tgt.lower() in BUILTINS and tgt not in {"test"}:
            return reject("builtin_command")
        if not re.fullmatch(r"[\w.:/@+-]+", tgt, re.UNICODE):
            return reject("bad_target")
        t_re = re.escape(tgt)
        if runner in SCRIPT_RUNNERS:
            rx = rf"(?<![\w-]){runner}\s+(?:run(?:-script)?\s+)?{t_re}(?![\w:-])"
        elif runner == "make":
            rx = rf"(?<![\w-])make\s+(?:-\S+\s+)*{t_re}(?![\w:-])"
        else:
            rx = rf"(?<![\w-])composer\s+(?:run(?:-script)?\s+)?{t_re}(?![\w:-])"
        if not re.search(rx, quote_n):
            return reject("command_not_in_quote")
        kind = "script" if runner in SCRIPT_RUNNERS else runner
        return [_mk("T5", ["T5", kind, tgt], tgt, quote, line, mode, {"runner": runner, "kind": kind, "target": tgt})], []

    if typ == "version":
        sj = subj.strip().lower()
        if sj not in T6_SUBJECTS:
            return reject("bad_version_subject")
        ver = raw.get("version") or ""
        qual = (raw.get("qualifier") or "exact").lower()
        if qual not in {"min", "exact"}:
            return reject("unsupported_qualifier")
        vc = version_code(ver)
        if vc is None:
            return reject("no_version_number")
        if ver.strip().lstrip("vV") not in quote and ver.strip() not in quote:
            m = re.search(r"\d+(?:\.\d+){0,3}", ver)
            if not m or m.group(0) not in quote:
                return reject("version_not_in_quote")
        code, prec = vc
        return [_mk("T6", ["T6", sj, f"{code}@{prec}"], sj, quote, line, mode,
                    {"subject_type": sj, "version": ver, "code": code, "precision": prec, "qualifier": qual})], []

    if typ == "import_rule":
        tgt = subj.strip().strip("`'\"*")
        if not tgt or " " in tgt:
            return reject("non_literal_target")
        scope = (raw.get("scope") or "").strip().strip("`'\"*") or None
        return [_mk("T7", ["T7", tgt, scope], tgt, quote, line, mode, {"target": tgt, "scope": scope})], []

    return reject("unknown_type")


def _mk(tcode: str, key: list[Any], subject: str, quote: str, line: int, mode: str, attrs: dict[str, Any]) -> dict[str, Any]:
    return {"tcode": tcode, "key": key, "key_str": "|".join("" if k is None else str(k) for k in key),
            "subject": subject, "quote": quote, "line": line, "mode": mode, "attrs": attrs}
