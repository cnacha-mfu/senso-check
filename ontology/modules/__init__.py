"""
SENSO Evolution Ontology (SEO) — OWL 2 DL module paths.

Paper 1: Software Evolution Ontology + Agent Manifests

This package provides convenience access to the ontology module files
and the builder that generates them. WP1 added seo-inference.owl (canonical
label rules), seo-legacy.owl (regression artefact, never labels real repos)
and seo-full.owl (imports closure of the canonical modules).

Usage::

    from ontology.modules import MODULE_PATHS, MODULE_DIR
    from ontology.modules.build_ontology import build_all, verify_all
"""

from __future__ import annotations

from pathlib import Path

MODULE_DIR: Path = Path(__file__).resolve().parent
"""Absolute path to the ontology/modules/ directory."""

MODULE_NAMES: list[str] = [
    "seo-entities",
    "seo-processes",
    "seo-metrics",
    "seo-laws",
    "seo-patterns",
    "seo-inference",
    "seo-evidence",
]
"""Ordered list of canonical module names (each imports the previous)."""

LEGACY_MODULE: str = "seo-legacy"
"""Regression artefact (legacy ensemble + paper-table laws); not part of seo-full."""

FULL_MODULE: str = "seo-full"
"""Imports closure of MODULE_NAMES; ``PopulationConfig.owl_file`` points here."""

MODULE_PATHS: dict[str, Path] = {
    name: MODULE_DIR / f"{name}.owl" for name in MODULE_NAMES
}
"""Mapping from module name to its .owl file path (Paper 1)."""

BASE_IRI: str = "http://senso-framework.org/ontology/seo"
"""Base IRI for all SEO ontology modules."""

def load_seo_ontology():
    """Load the full SEO ontology with local file resolution.

    Loads all canonical modules explicitly into the same Owlready2 world so
    inter-module imports resolve via local files without network access.
    Returns a dict of ontology name → owlready2.Ontology.

    Paper 1: Software Evolution Ontology.
    """
    import owlready2

    owlready2.onto_path.append(str(MODULE_DIR))
    world = owlready2.default_world

    ontologies = {}
    for name in MODULE_NAMES:
        owl_path = MODULE_DIR / f"{name}.owl"
        onto = world.get_ontology(f"file://{owl_path}").load()
        ontologies[name] = onto

    return ontologies


__all__: list[str] = [
    "MODULE_DIR",
    "MODULE_NAMES",
    "LEGACY_MODULE",
    "FULL_MODULE",
    "MODULE_PATHS",
    "BASE_IRI",
    "load_seo_ontology",
]
