#!/usr/bin/env python3
"""Pellet justifications for inferred SEO labels (Paper 1, WP1).

Paper 1 - Owlready2 has no explanation API, so this module calls the Pellet
CLI (``pellet explain --instance ind,Class``) on the jars bundled with
Owlready2, feeding the merged TBox + ABox as RDF/XML on stdin (the CLI's
default loader mis-handles N-Triples), and parses the Manchester-syntax output
into structured records:

  * axioms used: class definitions (``X equivalentTo ...``), SWRL rules
    (``Rule(... -> C(?x))``), subclass axioms;
  * facts used: ``individual property value`` assertions;
  * closures used: ``Functional p`` lines (the source of negation-free
    else-branches) and ``hasMonth only`` style assertions.

The generator can fill a label's ``seo:explanationTemplate`` only from the
facts in a justification, which guarantees every number in a manifest is one
the reasoner used. Each call costs about 2 s (JVM start-up).

Usage:
    python ontology/reasoning/explain.py --selftest
    python ontology/reasoning/explain.py --abox repo.nt --individual IRI --cls GrowthStage
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("senso.explain")

SEO = "http://senso-framework.org/ontology/seo#"
_AXIOM_KEYWORDS = {"equivalentTo", "subClassOf", "disjointWith", "type", "domain", "range",
                   "subPropertyOf", "inverseOf"}


@dataclass
class AxiomRecord:
    """A TBox axiom or SWRL rule in a justification."""

    kind: str  # definition | rule | subclass | other
    subject: str  # defined class, or rule consequent class
    text: str


@dataclass
class FactRecord:
    """An ABox assertion in a justification."""

    subject: str
    prop: str  # property local name, or 'type'
    value: Any
    raw: str


@dataclass
class Justification:
    """One minimal explanation: the axioms and facts that entail the label."""

    axioms: list[AxiomRecord] = field(default_factory=list)
    facts: list[FactRecord] = field(default_factory=list)
    closures: list[str] = field(default_factory=list)
    raw: str = ""

    def fact_values(self) -> dict[str, Any]:
        """Property -> value for the (non-type) facts: input of the template filler."""
        return {f.prop: f.value for f in self.facts if f.prop != "type"}


@dataclass
class Explanation:
    """Result of one explain call."""

    target: str
    justifications: list[Justification] = field(default_factory=list)
    raw_text: str = ""
    seconds: float = 0.0
    returncode: int = 0
    error: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        """Serializable form."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _value(tok: str) -> Any:
    tok = tok.strip()
    m = re.match(r'^"(.*)"\^\^\w+$', tok, re.DOTALL)
    if m:
        return m.group(1)
    if re.match(r"^-?\d+$", tok):
        return int(tok)
    try:
        return float(tok)
    except ValueError:
        return tok


def _classify(line: str) -> tuple[str, Any]:
    line = line.strip()
    if line.startswith("Rule("):
        m = re.search(r"->\s*(\w+)\(\?\w+\)\)\s*$", line)
        return "rule", AxiomRecord("rule", m.group(1) if m else "", line)
    m = re.match(r"^Functional (\S+)$", line)
    if m:
        return "closure", m.group(1)
    toks = line.split(None, 2)
    if len(toks) == 3 and toks[1] in _AXIOM_KEYWORDS:
        if toks[1] == "type":
            return "fact", FactRecord(toks[0], "type", toks[2], line)
        kind = "definition" if toks[1] == "equivalentTo" else (
            "subclass" if toks[1] == "subClassOf" else "other")
        return "axiom", AxiomRecord(kind, toks[0], line)
    if len(toks) == 3:
        return "fact", FactRecord(toks[0], toks[1], _value(toks[2]), line)
    return "other", AxiomRecord("other", "", line)


def parse_explanation(text: str) -> list[Justification]:
    """Parse ``pellet explain`` output into justification records.

    A new axiom starts at the base indent of an explanation block; deeper
    indented lines continue the previous axiom (long ``equivalentTo`` lists).

    Args:
        text: Raw stdout of ``pellet explain``.

    Returns:
        One Justification per numbered explanation.
    """
    just: list[Justification] = []
    cur: Optional[Justification] = None
    chunks: list[str] = []
    base_indent: Optional[int] = None

    def flush_axiom() -> None:
        nonlocal chunks
        if cur is not None and chunks:
            line = " ".join(c.strip() for c in chunks)
            kind, rec = _classify(line)
            if kind == "rule" or kind == "axiom":
                cur.axioms.append(rec)
            elif kind == "fact":
                cur.facts.append(rec)
            elif kind == "closure":
                cur.closures.append(rec)
            else:
                cur.axioms.append(rec)
            cur.raw += (cur.raw and "\n") + line
        chunks = []

    for ln in text.splitlines():
        if not ln.strip():
            continue
        m = re.match(r"^(\d+\))\s+(.*)$", ln)
        if m:
            flush_axiom()
            cur = Justification()
            just.append(cur)
            base_indent = len(m.group(1)) + (len(ln[len(m.group(1)):]) - len(ln[len(m.group(1)):].lstrip()))
            chunks = [m.group(2)]
            continue
        if cur is None:
            continue
        indent = len(ln) - len(ln.lstrip())
        if base_indent is not None and indent <= base_indent:
            flush_axiom()
            chunks = [ln.strip()]
        else:
            chunks.append(ln.strip())
    flush_axiom()
    return just


# ---------------------------------------------------------------------------
# Calling Pellet
# ---------------------------------------------------------------------------

def merged_rdfxml(reasoner: Any, abox: str | bytes | Path) -> bytes:
    """TBox + ABox as RDF/XML bytes (input for the explain CLI).

    Args:
        reasoner: A ``PelletReasoner`` (provides the TBox files).
        abox: N-Triples text, bytes or path.

    Returns:
        RDF/XML bytes of the merged world.
    """
    data = Path(abox).read_bytes() if isinstance(abox, Path) else (
        abox.encode("utf-8") if isinstance(abox, str) else abox)
    world, _ = reasoner.load_world()
    world.get_ontology("urn:senso:abox:explain#").load(fileobj=io.BytesIO(data), format="ntriples")
    buf = io.BytesIO()
    world.save(buf, format="rdfxml")
    return buf.getvalue()


def _run(reasoner: Any, args: Sequence[str], rdfxml: bytes, timeout: int) -> tuple[int, str, str, float]:
    from owlready2.reasoning import _PELLET_CLASSPATH

    cmd = [reasoner.java, f"-Xmx{reasoner.memory_mb}M", "-cp", _PELLET_CLASSPATH,
           "pellet.Pellet", "explain", "--ignore-imports", *args, "/dev/stdin"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=rdfxml, capture_output=True, timeout=timeout, check=False)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"), \
        time.perf_counter() - t0


def explain_instance(
    reasoner: Any,
    rdfxml: bytes,
    individual: str,
    cls: str,
    *,
    max_explanations: int = 1,
    timeout: int = 300,
) -> Explanation:
    """Explain why ``individual`` is an instance of ``cls``.

    Args:
        reasoner: A ``PelletReasoner`` (Java path, memory).
        rdfxml: Output of ``merged_rdfxml``.
        individual: Individual IRI.
        cls: Class IRI or local name in the seo namespace.
        max_explanations: ``--max`` value.
        timeout: Seconds before the call is abandoned.

    Returns:
        Explanation with parsed justifications; errors are captured, not raised.
    """
    cls_iri = cls if "#" in cls else SEO + cls
    rc, out, err, secs = _run(
        reasoner, ["--instance", f"{individual},{cls_iri}", "--max", str(max_explanations)],
        rdfxml, timeout)
    ex = Explanation(target=f"{individual} type {cls_iri}", raw_text=out, seconds=secs, returncode=rc)
    if rc != 0:
        ex.error = (err or out)[:1000]
    else:
        ex.justifications = parse_explanation(out)
    return ex


def explain_inconsistent(reasoner: Any, rdfxml: bytes, *, timeout: int = 300) -> Explanation:
    """Explain why the merged ontology is inconsistent (``--inconsistent``)."""
    rc, out, err, secs = _run(reasoner, ["--inconsistent"], rdfxml, timeout)
    ex = Explanation(target="inconsistent", raw_text=out, seconds=secs, returncode=rc)
    if rc != 0:
        ex.error = (err or out)[:1000]
    else:
        ex.justifications = parse_explanation(out)
    return ex


def justification_ntriples(individual: str, cls_iri: str, j: Justification, node: str) -> str:
    """Render a justification as ``seo:Justification`` N-Triples (design 3.4 step 8).

    The vocabulary (``seo:Justification``, ``seo:justifies``, ``seo:usesAxiom``,
    ``seo:usesFact``, ``seo:usesClosure``, ``seo:explanationText``) is not yet
    declared in the TBox; declaring it is part of the generator rewiring.

    Args:
        individual: Explained individual IRI.
        cls_iri: Explained class IRI.
        j: The justification.
        node: IRI for the justification node.

    Returns:
        N-Triples text.
    """
    def lit(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

    out = [f"<{node}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{SEO}Justification> .",
           f"<{node}> <{SEO}justifies> <{individual}> .",
           f"<{node}> <{SEO}justifiesClass> <{cls_iri}> .",
           f"<{node}> <{SEO}explanationText> {lit(j.raw)} ."]
    for i, a in enumerate(j.axioms):
        out.append(f"<{node}> <{SEO}usesAxiom> {lit(a.text)} .")
    for i, f in enumerate(j.facts):
        b = f"_:{abs(hash((node, i))) % 10**9}"
        out += [f"<{node}> <{SEO}usesFact> {b} .",
                f"{b} <{SEO}onProperty> {lit(f.prop)} .",
                f"{b} <{SEO}value> {lit(str(f.value))} ."]
    for c in j.closures:
        out.append(f"<{node}> <{SEO}usesClosure> {lit(c)} .")
    return "\n".join(out) + "\n"


_SAMPLE = """Axiom: demo_repo type GrowthStage

Explanation(s):
1)   A_locMKTau_gt_0 equivalentTo locMKTau some decimal[> 0.0]
     A_locMKpValue_lt_mk_alpha equivalentTo locMKpValue some decimal[< 0.05]
     demo_repo locMKpValue 8e-06
     GrowthStage equivalentTo A_locMKTau_gt_0
                              and A_locMKpValue_lt_mk_alpha
                              and MKApplicable
                              and SoftwareProject
     demo_repo locMKTau 1.0
     demo_repo type SoftwareProject
     Functional locMKTau
     Rule(Commit(?c), commitSubject(?c, ?s), matches(?s "(?is).*fix.*"^^string ) -> BugFixCommit(?c))
     abc commitSubject "Fix crash"^^string
"""


def _selftest() -> int:
    """Parse a captured Pellet output sample."""
    js = parse_explanation(_SAMPLE)
    assert len(js) == 1, js
    j = js[0]
    assert [a.kind for a in j.axioms] == ["definition", "definition", "definition", "rule"], j.axioms
    assert j.axioms[2].subject == "GrowthStage" and "SoftwareProject" in j.axioms[2].text
    assert j.axioms[3].subject == "BugFixCommit"
    facts = j.fact_values()
    assert facts["locMKpValue"] == 8e-06 and facts["locMKTau"] == 1.0, facts
    assert facts["commitSubject"] == "Fix crash", facts
    assert j.closures == ["locMKTau"], j.closures
    assert [f.prop for f in j.facts if f.prop == "type"] == ["type"]
    print("explain parser selftest ok")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pellet explanations (Paper 1, WP1)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--abox", type=Path)
    ap.add_argument("--individual")
    ap.add_argument("--cls")
    ap.add_argument("--stage-b", action="store_true", help="explain over the stage-B TBox (default: stage A)")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if a.selftest:
        sys.exit(_selftest())
    if a.abox and a.individual and a.cls:
        from ontology.reasoning.reasoner import PelletReasoner

        from ontology.reasoning.reasoner import HermitReasoner

        r = HermitReasoner(legacy=a.legacy) if a.stage_b else PelletReasoner()
        ex = explain_instance(r, merged_rdfxml(r, a.abox), a.individual, a.cls)
        print(json.dumps(ex.to_json(), indent=1, default=str))
    else:
        ap.print_help()
