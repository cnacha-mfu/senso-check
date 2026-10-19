#!/usr/bin/env python3
"""ABox writer for the SEO reasoning pipeline (Paper 1, WP1).

Paper 1 - turns measurements (``ontology/population/measures.py``) into
N-Triples with the datatypes the TBox expects: reals as ``xsd:decimal``
(rounded to 6 places, never scientific notation), counts as ``xsd:integer``.
NaN and inf are not asserted ("undetermined"). Per-month snapshots are linked
with ``seo:hasMonth`` and closed with
``project rdf:type (seo:hasMonth only {m1..mn})`` so universal conditions can
be classified under the open-world assumption.

The writer produces text on purpose: the same text can be PUT to a Fuseki
named graph (``fuseki_graphs.py``) or read straight by ``reasoner.py``.

Usage:
    python ontology/reasoning/abox.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("senso.abox")

SEO = "http://senso-framework.org/ontology/seo#"
DATA_BASE = "http://senso-framework.org/data/"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL = "http://www.w3.org/2002/07/owl#"

_KINDS: Optional[dict[str, str]] = None


def property_kinds() -> dict[str, str]:
    """Return ``{property name: kind}`` from the builder's property table."""
    global _KINDS
    if _KINDS is None:
        from ontology.modules.build_ontology import MEASURED_PROPERTIES

        from ontology.modules.claims_module import CLAIM_PROPERTIES  # WP2 hook

        _KINDS = {m[0]: m[1] for m in MEASURED_PROPERTIES}
        _KINDS.update({m[0]: m[1] for m in CLAIM_PROPERTIES})
    return _KINDS


def fmt_decimal(x: float | int | Decimal) -> str:
    """Format a real as a plain xsd:decimal lexical form (no exponent)."""
    d = Decimal(repr(round(float(x), 6))) if not isinstance(x, Decimal) else x
    s = format(d, "f")
    return s if "." in s else s + ".0"


def _esc(s: str) -> str:
    return (s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            .replace("\r", "\\r").replace("\t", "\\t"))


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


class AboxWriter:
    """Accumulates N-Triples for projects, months and commits (Paper 1, WP1)."""

    def __init__(self, data_base: str = DATA_BASE) -> None:
        self.base = data_base
        self.lines: list[str] = []
        self.month_owner: dict[str, str] = {}
        self.individuals = 0
        self._bn = 0

    # -- low level -------------------------------------------------------
    def _bnode(self) -> str:
        self._bn += 1
        return f"_:b{self._bn}"

    def _t(self, s: str, p: str, o: str) -> None:
        self.lines.append(f"{s} <{p}> {o} .")

    @staticmethod
    def _iri(i: str) -> str:
        return f"<{i}>"

    def _declare(self, subj: str, cls: str) -> None:
        """Type an individual and declare it (owl:NamedIndividual)."""
        self._t(subj, RDF + "type", self._iri(SEO + cls))
        self._t(subj, RDF + "type", self._iri(OWL + "NamedIndividual"))
        self.individuals += 1

    def literal(self, prop: str, value: Any) -> Optional[str]:
        """Typed literal for a measured property value, or None if not assertable."""
        if value is None:
            return None
        kind = property_kinds().get(prop)
        if isinstance(value, datetime):
            return f'"{value.isoformat()}"^^<{XSD}dateTime>'
        if isinstance(value, str):
            return f'"{_esc(value)}"^^<{XSD}string>'
        if isinstance(value, bool):
            raise TypeError(f"{prop}: booleans are not measurements")
        if kind == "int" or (kind is None and isinstance(value, int)):
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value) or value != int(value):
                    return None
                value = int(value)
            return f'"{int(value)}"^^<{XSD}integer>'
        # decimal
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return f'"{fmt_decimal(fv)}"^^<{XSD}decimal>'

    def _props(self, subj: str, values: Mapping[str, Any]) -> None:
        for prop, value in values.items():
            if prop == "month":
                continue
            vals = value if isinstance(value, (list, tuple)) else [value]
            for v in vals:
                lit = self.literal(prop, v)
                if lit is not None:
                    self._t(subj, SEO + prop, lit)

    # -- entities --------------------------------------------------------
    def project(
        self,
        key: str,
        measurements: Mapping[str, Any],
        months: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        close_months: bool = True,
    ) -> str:
        """Add a SoftwareProject with its measurements and month snapshots.

        Args:
            key: Project key (becomes part of the IRI).
            measurements: Project-level property values (``measures.project_measurements``).
            months: Per-month snapshots (``measures.month_snapshots``).
            close_months: Assert the ``hasMonth only {...}`` closure.

        Returns:
            The project IRI.
        """
        iri = self.base + _safe(key)
        s = self._iri(iri)
        self._declare(s, "SoftwareProject")
        self._props(s, measurements)
        if months:
            miris = []
            for snap in months:
                mi = f"{iri}/month/{snap['month']}"
                miris.append(mi)
                self.month_owner[mi] = iri
                ms = self._iri(mi)
                self._declare(ms, "MonthSnapshot")
                self._t(s, SEO + "hasMonth", ms)
                self._props(ms, snap)
            for a, b in zip(miris, miris[1:]):
                self._t(self._iri(a), SEO + "nextMonth", self._iri(b))
            if close_months:
                self._close(s, miris)
        return iri

    def _close(self, subj: str, members: Sequence[str]) -> None:
        r, cls, lst = self._bnode(), self._bnode(), self._bnode()
        self._t(subj, RDF + "type", r)
        self._t(r, RDF + "type", self._iri(OWL + "Restriction"))
        self._t(r, OWL + "onProperty", self._iri(SEO + "hasMonth"))
        self._t(r, OWL + "allValuesFrom", cls)
        self._t(cls, RDF + "type", self._iri(OWL + "Class"))
        self._t(cls, OWL + "oneOf", lst)
        cur = lst
        for i, m in enumerate(members):
            self._t(cur, RDF + "first", self._iri(m))
            if i == len(members) - 1:
                self._t(cur, RDF + "rest", self._iri(RDF + "nil"))
            else:
                nxt = self._bnode()
                self._t(cur, RDF + "rest", nxt)
                cur = nxt

    @staticmethod
    def module_slug(path: str) -> str:
        """IRI-safe, collision-free slug of a module path (readable part plus a 6-hex digest)."""
        return f"{_safe(path)}-{hashlib.sha1(path.encode('utf-8')).hexdigest()[:6]}"

    def module(self, project_iri: str, path: str, measurements: Mapping[str, Any]) -> str:
        """Add a Module of a project with its measurements (WP5 module stability).

        Args:
            project_iri: IRI returned by ``project``.
            path: Module id of the WP5 partition.
            measurements: Module-level property values (``measures.module_abox_properties``).

        Returns:
            The module IRI.
        """
        iri = f"{project_iri}/module/{self.module_slug(path)}"
        s = self._iri(iri)
        self._declare(s, "Module")
        self._props(s, measurements)
        self._t(self._iri(project_iri), SEO + "hasModule", s)
        return iri

    def commit(self, key: str, evidence: Mapping[str, Any]) -> str:
        """Add a Commit with its raw evidence strings (no labels).

        Args:
            key: Commit key (sha).
            evidence: Output of ``measures.extract_commit_evidence``.

        Returns:
            The commit IRI.
        """
        iri = self.base + "commit/" + _safe(key)
        s = self._iri(iri)
        self._declare(s, "Commit")
        self._props(s, evidence)
        return iri

    def ntriples(self) -> str:
        """The accumulated triples as N-Triples text."""
        return "\n".join(self.lines) + "\n"

    @property
    def n_triples(self) -> int:
        """Number of triples written so far."""
        return len(self.lines)


def _demo() -> None:
    from ontology.population.measures import extract_commit_evidence, month_snapshots, project_measurements

    series = [{"month": f"2025-{i + 1:02d}", "commit_count": 5 + i, "loc": 100 + 10 * i,
               "complexity_mean": 2.0} for i in range(6)]
    w = AboxWriter()
    w.project("demo/repo", project_measurements(series), month_snapshots(series))
    w.commit("abc", extract_commit_evidence(author_name="x[bot]", author_email="e", message="Fix bug"))
    print(w.ntriples()[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ABox writer (Paper 1, WP1)")
    ap.add_argument("--demo", action="store_true")
    if ap.parse_args().demo:
        _demo()
