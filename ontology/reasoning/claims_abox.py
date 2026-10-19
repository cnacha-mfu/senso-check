#!/usr/bin/env python3
"""ABox writer for manifest claims (Paper 1, WP2).

Extends ``AboxWriter`` (kept in its own file so the WP1 writer stays untouched) with
``manifest()``, ``claim()``, ``evidence()`` and ``close_evidence()``. IRIs:
``data/manifest/{repo}``, ``data/claim/{repo}/{sha1(key)[:12]}`` and
``.../intro`` for the introduction-commit copy of a claim.

Python asserts claim *content* (type, quote, subject, key, votes) and *measurements*
(counts, deltas). It never asserts a verdict class; ``AboxWriter.literal`` raises on bool.

Usage:
    imported by wp2/step_measure.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ontology.reasoning.abox import DATA_BASE, OWL, RDF, SEO, AboxWriter, _safe  # noqa: E402


def claim_id(key: str) -> str:
    """Stable 12-hex id of a claim key."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class ClaimsAboxWriter(AboxWriter):
    """AboxWriter with manifest/claim/evidence individuals (Paper 1, WP2)."""

    def manifest(self, repo: str, props: Mapping[str, Any]) -> str:
        """A ManifestDocument; returns its IRI."""
        iri = f"{self.base}manifest/{_safe(repo)}"
        s = self._iri(iri)
        self._declare(s, "ManifestDocument")
        self._props(s, props)
        return iri

    def claim(self, repo: str, key: str, types: Sequence[str], props: Mapping[str, Any], *,
              doc: Optional[str] = None, headline: bool = True, suffix: str = "") -> str:
        """A ManifestClaim typed with the asserted claim-type classes ``types``.

        Args:
            repo: Repository (IRI part).
            key: Canonical claim key.
            types: Asserted type classes (e.g. ``["PathExistenceClaim"]``); content, not verdicts.
            props: Claim data properties (quote, subject, key, votes, measurements).
            doc: Document IRI; the claim is linked via makesClaim (headline) or makesExcludedClaim.
                Omit for an introduction-commit copy.
            headline: False routes the link to ``makesExcludedClaim``.
            suffix: ``"/intro"`` for the introduction-commit copy.
        """
        iri = f"{self.base}claim/{_safe(repo)}/{claim_id(key)}{suffix}"
        s = self._iri(iri)
        self._declare(s, "ManifestClaim")
        for t in types:
            self._t(s, RDF + "type", self._iri(SEO + t))
        self._props(s, props)
        if doc:
            self._t(self._iri(doc), SEO + ("makesClaim" if headline else "makesExcludedClaim"), s)
        return iri

    def evidence(self, claim_iri: str, idx: int, props: Mapping[str, Any]) -> str:
        """A VersionEvidence individual linked to its claim in both directions."""
        iri = f"{claim_iri}/ev{idx}"
        s = self._iri(iri)
        self._declare(s, "VersionEvidence")
        self._props(s, props)
        self._t(s, SEO + "evidenceFor", self._iri(claim_iri))
        self._t(self._iri(claim_iri), SEO + "hasEvidence", s)
        return iri

    def introduced_as(self, claim_iri: str, intro_iri: str) -> None:
        """Link a claim to its introduction-commit copy."""
        self._t(self._iri(claim_iri), SEO + "introducedAs", self._iri(intro_iri))

    def close_evidence(self, claim_iri: str, evidence_iris: Sequence[str]) -> None:
        """Assert ``claim rdf:type (hasEvidence only {e1..en})`` (skipped for an empty set)."""
        if not evidence_iris:
            return
        subj = self._iri(claim_iri)
        r, cls, lst = self._bnode(), self._bnode(), self._bnode()
        self._t(subj, RDF + "type", r)
        self._t(r, RDF + "type", self._iri(OWL + "Restriction"))
        self._t(r, OWL + "onProperty", self._iri(SEO + "hasEvidence"))
        self._t(r, OWL + "allValuesFrom", cls)
        self._t(cls, RDF + "type", self._iri(OWL + "Class"))
        self._t(cls, OWL + "oneOf", lst)
        cur = lst
        for i, m in enumerate(evidence_iris):
            self._t(cur, RDF + "first", self._iri(m))
            if i == len(evidence_iris) - 1:
                self._t(cur, RDF + "rest", self._iri(RDF + "nil"))
            else:
                nxt = self._bnode()
                self._t(cur, RDF + "rest", nxt)
                cur = nxt
