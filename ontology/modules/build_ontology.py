"""
Build the SENSO Evolution Ontology (SEO) OWL 2 DL modules.

Paper 1: Software Evolution Ontology + Agent Manifests (WP1: the ontology
makes every judgement).

Modules (every module uses the single namespace
``http://senso-framework.org/ontology/seo#``; the ontology IRIs differ only
so that ``owl:imports`` works):

  1. seo-entities.owl   - Evolution Entity Module
  2. seo-processes.owl  - Evolution Process Module
  3. seo-metrics.owl    - Metric Module, incl. the *measured* data properties
                          (WP1 design 3.2). Python asserts these, never labels.
  4. seo-laws.owl       - Evolution Law Module (the eight law individuals)
  5. seo-patterns.owl   - Pattern Module (concept classes of anti-patterns)
  6. seo-inference.owl  - CANONICAL label rules, SWRL-free OWL 2 DL: guards,
                          Mann-Kendall stages (D6), the eight laws with the
                          PIPELINE thresholds (D7), anti-patterns. Thresholds
                          come from thresholds.yaml. Reasoned with HermiT.
  6b. seo-evidence.owl  - commit-level SWRL string rules: AgentCommit,
                          BugFixCommit (D8), FeatureCommit (D9). The only SWRL
                          in the canonical modules. Reasoned with Pellet.
  7. seo-legacy.owl     - REGRESSION ARTEFACT, never labels real repos (D6/D7):
                          the legacy 3-vote ensemble phase encoding
                          (Legacy*Stage) and the paper-table law variant
                          (PaperLaw*). SWRL-free, independent of seo-inference;
                          reasoned with HermiT.
  8. seo-full.owl       - imports closure of 1-6b (what populate.py loads).
                          seo-legacy is deliberately NOT part of it.

Design choices documented here (task step 2):
  * Law variants are emitted as separately named classes, not via a flag:
    ``LawIConformant`` (pipeline, canonical) and ``PaperLawIConformant``
    (paper table, in seo-legacy). A flag would hide which variant a label
    came from; separate names let both be reasoned over in one ABox.
  * The two phase encodings are likewise separate: ``GrowthStage`` (MK) vs
    ``LegacyGrowthStage`` (ensemble).
  * Every else-branch is a POSITIVE complement class (``x >= k`` for
    ``not x < k``) so no OWL negation, no open-world surprise: a missing
    measurement gives no label ("undetermined"), never a false one.
  * Every operand of a union is a NAMED class (Pellet's explanation renderer
    crashes otherwise) and no xsd:pattern facet is used (it crashes Pellet
    2.3.1); string matching is done with SWRL ``matches``. ``lint`` enforces it.

Usage:
    python ontology/modules/build_ontology.py                 # regenerate all
    python ontology/modules/build_ontology.py --thresholds alt_thresholds.yaml
"""

from __future__ import annotations

import argparse
import functools
import logging
import operator
import re
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

import owlready2
from owlready2 import (
    AllDisjoint,
    AnnotationProperty,
    ConstrainedDatatype,
    DataProperty,
    FunctionalProperty,
    Imp,
    ObjectProperty,
    Thing,
    get_ontology,
    locstr,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_IRI: str = "http://senso-framework.org/ontology/seo"
SEO_NS_IRI: str = BASE_IRI + "#"
"""The single namespace shared by every module's entities."""
FUNCTIONAL_MEASURES: bool = True
"""Declare measured data properties Functional (see build_all docstring)."""
THRESHOLDS_FILE: Path = Path(__file__).resolve().parent / "thresholds.yaml"
OUTPUT_DIR: Path = Path(__file__).resolve().parent

MODULE_NAMES: list[str] = [
    "seo-entities",
    "seo-processes",
    "seo-metrics",
    "seo-laws",
    "seo-patterns",
    "seo-inference",
    "seo-evidence",
    "seo-claims",  # WP2: manifest claims and consistency verdicts (imports seo-entities only)
]
LEGACY_MODULE: str = "seo-legacy"
FULL_MODULE: str = "seo-full"
ALL_MODULE_NAMES: list[str] = MODULE_NAMES + [LEGACY_MODULE, FULL_MODULE]


def _iri(name: str) -> str:
    """Return the full IRI for a module name (Paper 1)."""
    return f"{BASE_IRI}/{name}"


class _State:
    """Build-time state: the Owlready world and the output directory."""

    world: Any = None
    out_dir: Path = OUTPUT_DIR


_STATE = _State()


class _SeoOntology(owlready2.namespace.Ontology):
    """Ontology whose attribute access resolves names in the shared seo# IRI."""

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_") or attr in ("world", "storid", "loaded", "name"):
            raise AttributeError(attr)
        return self.world[SEO_NS_IRI + attr]


def _onto(name: str) -> owlready2.namespace.Ontology:
    """Create the ontology for a module in the current build world (Paper 1)."""
    world = _STATE.world or owlready2.default_world
    return world.get_ontology(_iri(name), OntologyClass=_SeoOntology)


class _SeoNamespace(owlready2.namespace.Namespace):
    """The shared seo# namespace bound to one module's ontology.

    Owlready2 constructs such as AllDisjoint treat the *current namespace* as
    an ontology; delegating unknown attributes to the ontology makes a plain
    Namespace usable there, so classes of every module share one IRI base.
    """

    def __getattr__(self, attr: str) -> Any:
        if attr in ("ontology", "world", "_base_iri", "name"):
            raise AttributeError(attr)
        return getattr(self.ontology, attr)


def _ns(onto: owlready2.namespace.Ontology) -> _SeoNamespace:
    """Return the single-IRI namespace of a module ontology (Paper 1)."""
    ns = onto._namespaces.get(SEO_NS_IRI)
    if not isinstance(ns, _SeoNamespace):
        ns = _SeoNamespace(onto, SEO_NS_IRI)
    return ns


def _owl_path(name: str) -> Path:
    """Return the filesystem path for a module (Paper 1)."""
    return _STATE.out_dir / f"{name}.owl"


def load_thresholds(path: Path = THRESHOLDS_FILE) -> dict[str, dict[str, Any]]:
    """Load the threshold register as ``{id: entry}`` (Paper 1, WP1).

    Args:
        path: Path to thresholds.yaml.

    Returns:
        Mapping from threshold id to its register entry (value, unit, rule,
        source, status, sensitivity ...).
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return {t["id"]: t for t in doc["thresholds"]}


# ---------------------------------------------------------------------------
# Measured data properties (WP1 design 3.2). Python asserts these; it never
# compares them with a constant. kind: int -> xsd:integer, dec -> xsd:decimal,
# str -> xsd:string, strs -> multi-valued string, dt -> xsd:dateTime.
# domain: P = SoftwareProject, M = MonthSnapshot, C = Commit, Mo = Module (WP5).
# ---------------------------------------------------------------------------

MEASURED_PROPERTIES: list[tuple[str, str, str, str, str, str]] = [
    # (name, kind, domain, unit, computed_by, comment)
    ("monthsObserved", "int", "P", "months", "measures.series_summary", "Months in the zero-filled lookback window."),
    ("activeMonthCount", "int", "P", "months", "measures.series_summary", "Months with at least one commit."),
    ("activeMonthRatio", "dec", "P", "ratio [0,1]", "measures.series_summary", "activeMonthCount / monthsObserved."),
    ("meanMonthlyCommits", "dec", "P", "commits/month", "measures.series_summary", "Mean commits per month over the window."),
    ("commitsLast3Months", "int", "P", "commits", "measures.series_summary", "Commits in the last three months."),
    ("commitCountCV", "dec", "P", "unitless", "measures.cv", "Population SD of monthly commits / max(mean, 1)."),
    ("commitCountCVSample", "dec", "P", "unitless", "measures.cv", "Sample SD of monthly commits / max(mean, 1) (paper-table variant)."),
    ("churnCV", "dec", "P", "unitless", "measures.cv", "Population SD of monthly churn / max(mean, 1)."),
    ("linesAddedCV", "dec", "P", "unitless", "measures.cv", "Population SD of monthly lines added / max(mean, 1)."),
    ("complexityMonthCount", "int", "P", "months", "measures.series_summary", "Months with mean complexity > 0."),
    ("complexityHalfMeanFirst", "dec", "P", "CCN", "measures.half_means", "Mean complexity of the first half of the months that have complexity."),
    ("complexityHalfMeanSecond", "dec", "P", "CCN", "measures.half_means", "Mean complexity of the second half of the months that have complexity."),
    ("complexityFirst", "dec", "P", "CCN", "measures.series_summary", "Complexity of the first month."),
    ("complexityLast", "dec", "P", "CCN", "measures.series_summary", "Complexity of the last month."),
    ("complexityGrowthRate", "dec", "P", "ratio", "measures.series_summary", "(complexityLast - complexityFirst) / max(complexityFirst, 1)."),
    ("monthlyComplexityGrowth", "dec", "P", "ratio/month", "measures.series_summary", "(last - first) / max(n-1, 1) / max(first, 1) of complexity."),
    ("complexityChange", "dec", "P", "CCN", "measures.series_summary", "complexityLast - complexityFirst (difference of two measurements)."),
    ("complexityToLocGrowthRatio", "dec", "P", "ratio", "measures.series_summary", "complexityGrowthRate / locGrowthRate; asserted only where locGrowthRate is not 0."),
    ("locEndToStartRatio", "dec", "P", "ratio", "measures.loc_endpoints", "locEnd / locStart; asserted only where locStart > 0."),
    ("featureSecondToFirstRatio", "dec", "P", "ratio", "measures.feature_commit_counts", "featureCommitsSecondHalf / featureCommitsFirstHalf; asserted only where the first half has feature commits."),
    ("correctiveToRisingRatio", "dec", "P", "ratio", "post stage M aggregate", "correctiveMonthCount / risingMonthCount; asserted only where risingMonthCount > 0."),
    ("complexityHalfMeanChange", "dec", "P", "CCN", "measures.half_means", "complexityHalfMeanSecond - complexityHalfMeanFirst (difference of two measurements)."),
    ("bugFixHalfRatio", "dec", "P", "ratio", "measures.half_means (post stage A aggregate)", "bugFixRatioSecondHalf / bugFixRatioFirstHalf; asserted only where the first-half ratio is positive."),
    ("locToComplexityGrowthRatio", "dec", "P", "ratio", "measures.series_summary", "locGrowthRate / complexityGrowthRate; asserted only where complexityGrowthRate is not 0."),
    ("locDeltaPerFeature", "dec", "P", "LOC/commit", "measures.feature_commit_counts", "locDelta / max(featureCommitsTotal, 1)."),
    ("bugFixRatioFirstHalf", "dec", "P", "ratio [0,1]", "measures.half_means (post stage A aggregate)", "Mean bug-fix ratio of the first half of active months; needs inferred BugFixCommit (D8)."),
    ("bugFixRatioSecondHalf", "dec", "P", "ratio [0,1]", "measures.half_means (post stage A aggregate)", "Mean bug-fix ratio of the second half of active months."),
    ("mergeMonthRatio", "dec", "P", "ratio [0,1]", "measures.series_summary", "Months with a merge commit / monthsObserved."),
    ("multiAuthorMonthRatio", "dec", "P", "ratio [0,1]", "measures.series_summary", "Months with more than one author / monthsObserved."),
    ("locSeriesLength", "int", "P", "months", "measures.mann_kendall", "Length of the LOC series given to the Mann-Kendall test."),
    ("locMKTau", "dec", "P", "tau", "measures.mann_kendall", "Kendall tau of the monthly LOC series (S / (n(n-1)/2))."),
    ("locMKS", "int", "P", "count", "measures.mann_kendall", "Mann-Kendall S statistic."),
    ("locMKpValue", "dec", "P", "probability", "measures.mann_kendall", "Two-sided Mann-Kendall p-value (pymannkendall.original_test)."),
    ("locSenSlope", "dec", "P", "LOC/month", "measures.mann_kendall", "Theil-Sen slope of monthly LOC."),
    ("monthsSinceLastCommit", "int", "P", "months", "measures.months_since_last_commit", "Whole months between the snapshot month and the month of the latest commit (full history; 0 if a commit falls in the snapshot month). RQ1 C2."),
    ("locRelSenSlopeAnnual", "dec", "P", "fraction/year", "measures.rel_sen_slope_annual", "12 * locSenSlope / max(median monthly LOC, 1). RQ1 C2."),
    ("distinctAuthorsInWindow", "int", "P", "authors", "measures.distinct_authors", "Distinct author e-mails over the whole window (descriptive; not used by a rule)."),
    ("commitMKTau", "dec", "P", "tau", "measures.mann_kendall", "Kendall tau of the zero-filled monthly commit-count series (LOC-free comparator input; not used by a rule)."),
    ("commitMKpValue", "dec", "P", "probability", "measures.mann_kendall", "Mann-Kendall p-value of the monthly commit-count series (comparator input; not used by a rule)."),
    ("locStart", "int", "P", "LOC", "measures.loc_endpoints", "LOC in the first month."),
    ("locEnd", "int", "P", "LOC", "measures.loc_endpoints", "LOC in the last month."),
    ("locDelta", "int", "P", "LOC", "measures.loc_endpoints", "locEnd - locStart."),
    ("locGrowthRate", "dec", "P", "ratio", "measures.loc_endpoints", "(locEnd - locStart) / max(locStart, 1)."),
    ("locChangePct", "dec", "P", "percent", "measures.loc_endpoints", "locGrowthRate * 100 (legacy ensemble unit)."),
    ("locGrowthLast3", "dec", "P", "ratio", "measures.loc_endpoints", "(l3 - l1) / max(l1, 1) over the last three months."),
    ("commitHalfRatio", "dec", "P", "ratio", "measures.half_commit_stats", "Second-half mean commits / max(first-half mean, 1)."),
    ("commitSecondHalfMean", "dec", "P", "commits/month", "measures.half_commit_stats", "Mean commits per month in the second half."),
    ("featureCommitsFirstHalf", "int", "P", "commits", "measures.feature_commit_counts", "Conventional-Commit feat: commits in the first half (D9)."),
    ("featureCommitsSecondHalf", "int", "P", "commits", "measures.feature_commit_counts", "Conventional-Commit feat: commits in the second half (D9)."),
    ("featureCommitsTotal", "int", "P", "commits", "measures.feature_commit_counts", "Total feat: commits."),
    ("refactoringCommits", "int", "P", "commits", "NEW MINING (not mined)", "Refactoring commits; no data in real repos."),
    ("refactoringRatio", "dec", "P", "ratio", "measures.series_summary", "refactoringCommits / max(commits, 1)."),
    ("dependencyCountStart", "int", "P", "dependencies", "NEW MINING (not mined, D9)", "Dependencies at the first snapshot; no data."),
    ("dependencyCountEnd", "int", "P", "dependencies", "NEW MINING (not mined, D9)", "Dependencies at the last snapshot; no data."),
    ("dependencyGrowthRate", "dec", "P", "ratio", "measures.series_summary", "(end - start) / max(start, 1)."),
    ("recentAgentCommitRatio", "dec", "P", "ratio [0,1]", "measures.recent_ratio (post stage A aggregate)", "Mean of the last three monthly agent-commit ratios."),
    ("risingMonthCount", "int", "P", "months", "post stage M aggregate", "Count of inferred RisingMonth (paper-table Law VIII)."),
    ("correctiveMonthCount", "int", "P", "months", "post stage M aggregate", "Count of inferred CorrectiveMonth5."),
    ("responsiveMonthCount", "int", "P", "months", "post stage M aggregate", "Count of inferred ResponsiveRisingMonth."),
    ("correctiveLast2Months", "int", "P", "commits", "measures.month_snapshots", "Refactorings + bug fixes in the last two months."),
    ("monthIndex", "int", "M", "months", "measures.month_snapshots", "0-based index of the month from the start of the window."),
    ("monthIndexFromEnd", "int", "M", "months", "measures.month_snapshots", "0 for the last month, 1 for the one before, ..."),
    ("commitCount", "int", "M", "commits", "measures.month_snapshots", "Commits in the month."),
    ("commitDeltaFromPrev", "int", "M", "commits", "measures.month_snapshots", "commitCount minus the previous month's (absent for the first month)."),
    ("locDeltaFromPrev", "int", "M", "LOC", "measures.month_snapshots", "LOC minus the previous month's (absent for the first month)."),
    ("locChangeRatio", "dec", "M", "ratio", "measures.month_snapshots", "|LOC - previous LOC| / max(previous LOC, 1)."),
    ("complexityDelta", "dec", "M", "CCN", "measures.month_snapshots", "Complexity minus the previous month's."),
    ("correctiveCount", "int", "M", "commits", "measures.month_snapshots", "Refactorings + bug fixes in the month."),
    ("authorName", "str", "C", "text", "measures.extract_commit_evidence", "Commit author name."),
    ("authorEmail", "str", "C", "text", "measures.extract_commit_evidence", "Commit author email."),
    ("commitSubject", "str", "C", "text", "measures.extract_commit_evidence", "First line of the commit message."),
    ("coAuthorTrailer", "strs", "C", "text", "measures.extract_commit_evidence", "Raw Co-authored-by trailer lines (multi-valued)."),
    ("branchName", "strs", "C", "text", "NEW MINING (not mined, D9)", "Pull-request head branch names; no data."),
    ("authorDate", "dt", "C", "xsd:dateTime", "measures.extract_commit_evidence", "Author date."),
    # -- WP5 module-level stability (wp5-analysis-plan.md 3.2); Python measures, OWL labels.
    ("modulePath", "str", "Mo", "path", "measures.module_of", "Module id of the WP5 partition (package, subpackage or module file)."),
    ("moduleLoc", "int", "Mo", "LOC", "populate_v2.mine_modules", "Non-blank lines of the module's .py files at the base tree (LOC_INSTRUMENT)."),
    ("moduleCommitsInWindow", "int", "Mo", "commits", "measures.module_measurements", "Non-merge ancestor commits with author date in (S - W, S] that touch the module."),
    ("moduleChangeMonthsInWindow", "int", "Mo", "months", "measures.module_measurements", "Distinct UTC calendar months among those commits."),
    ("moduleLinesChangedInWindow", "int", "Mo", "lines", "measures.module_measurements", "Sum of insertions and deletions (numstat, no renames) on the module's files in the window."),
    ("moduleRelChurnInWindow", "dec", "Mo", "ratio", "measures.module_measurements", "moduleLinesChangedInWindow / moduleLoc; asserted only where moduleLoc > 0."),
    ("moduleMonthsSinceLastChange", "int", "Mo", "months", "measures.module_measurements", "Months from the month of S back to the month of the last commit touching the module (full history; descriptive, used by no rule)."),
]


# ============================================================================
# Module 1 — seo-entities.owl
# ============================================================================

def build_entities_module() -> owlready2.namespace.Ontology:
    """Create the Evolution Entity Module (Paper 1).

    Defines core software-evolution entities and their agent-aware
    extensions following UFO-A (endurant) / UFO-B (perdurant) distinctions
    where applicable (marked via annotations since the UFO OWL is not
    imported directly).

    Returns:
        The constructed ontology object.
    """
    onto = _onto("seo-entities")

    with _ns(onto):
        # ---- UFO-A inspired endurants ----

        class SoftwareProject(Thing):
            """A software project under evolution study (UFO-A: Kind)."""
            comment = [locstr("Paper 1 — top-level entity representing a software project.", "en")]

        class Version(Thing):
            """A distinct version or release of a software project (UFO-A: Phase)."""
            comment = [locstr("Paper 1 — represents a point-in-time snapshot.", "en")]

        class Module(Thing):
            """A logical module or package within a project (UFO-A: SubKind)."""
            pass

        class File(Thing):
            """A source-code file (UFO-A: SubKind)."""
            pass

        class Function(Thing):
            """A function or method within a file (UFO-A: SubKind)."""
            pass

        class Dependency(Thing):
            """An external or internal dependency (UFO-A: Relator)."""
            pass

        class Developer(Thing):
            """A human developer (UFO-A: Agent)."""
            pass

        class Commit(Thing):
            """A single commit (UFO-B: Event)."""
            comment = [locstr("Paper 1 — atomic change event.", "en")]

        class Release(Thing):
            """A tagged release (UFO-B: complex Event)."""
            pass

        # ---- Agent extensions ----

        class CodingAgent(Thing):
            """An AI coding agent (e.g. Copilot, Claude Code)."""
            comment = [locstr("Paper 1 — agent-aware extension.", "en")]

        class AgentCommit(Commit):
            """A commit authored or co-authored by a coding agent."""
            comment = [locstr(
                "Paper 1 - declared here, DEFINED in seo-inference.owl as the union of "
                "the evidence classes (trailer, bot author, branch). Never asserted.",
                "en",
            )]

        class MonthSnapshot(Thing):
            """One calendar month of a project's history (Paper 1, WP1).

            Carries per-month measurements; linked from the project with
            hasMonth. Closed with `hasMonth only {m1..mn}` by the ABox writer
            so universal conditions can be classified under the open-world
            assumption.
            """
            comment = [locstr("Paper 1 WP1 - per-month measurement snapshot.", "en")]

        # ---- Annotation properties used by the inference module ----

        class explanationTemplate(AnnotationProperty):
            """Text template for rendering a justification of a label class."""

        class thresholdId(AnnotationProperty):
            """Id of a threshold in thresholds.yaml used by a label class."""

        class thresholdSource(AnnotationProperty):
            """file:line of a threshold used by a label class."""

        class computedBy(AnnotationProperty):
            """measures.py function that computes a measured property."""

        class measurementUnit(AnnotationProperty):
            """Unit of a measured property."""

        class AgentManifest(Thing):
            """A machine-readable manifest describing an agent's capabilities."""
            pass

        class AgentGeneratedPR(Thing):
            """A pull request generated by a coding agent."""
            pass

        # ---- Object properties ----

        class hasVersion(ObjectProperty):
            """Links a project to its versions."""
            domain = [SoftwareProject]
            range = [Version]

        class hasModule(ObjectProperty):
            """Links a project to its modules."""
            domain = [SoftwareProject]
            range = [Module]

        class hasFile(ObjectProperty):
            """Links a module to its files."""
            domain = [Module]
            range = [File]

        class hasFunction(ObjectProperty):
            """Links a file to its functions."""
            domain = [File]
            range = [Function]

        class hasDependency(ObjectProperty):
            """Links a project or module to a dependency."""
            domain = [SoftwareProject | Module]
            range = [Dependency]

        class hasMonth(ObjectProperty):
            """Links a project to its per-month snapshots (closed in the ABox)."""
            domain = [SoftwareProject]
            range = [MonthSnapshot]

        class nextMonth(ObjectProperty, FunctionalProperty):
            """Links a month snapshot to the following month."""
            domain = [MonthSnapshot]
            range = [MonthSnapshot]

        class hasCommit(ObjectProperty):
            """Links a version to its commits."""
            domain = [Version]
            range = [Commit]

        class hasRelease(ObjectProperty):
            """Links a project to its releases."""
            domain = [SoftwareProject]
            range = [Release]

        class hasAuthor(ObjectProperty):
            """Links a commit to its author (human or agent)."""
            domain = [Commit]
            range = [Developer | CodingAgent]

        class modifiesFile(ObjectProperty):
            """Links a commit to files it modifies."""
            domain = [Commit]
            range = [File]

        class hasParent(ObjectProperty):
            """Links a version to its predecessor version."""
            domain = [Version]
            range = [Version]

        class dependsOn(ObjectProperty):
            """Links a module/file to what it depends on."""
            domain = [Module | File]
            range = [Module | File | Dependency]

        # Disjointness
        AllDisjoint([SoftwareProject, Version, Module, File, Function,
                     Dependency, Developer, Commit, Release,
                     CodingAgent, AgentManifest, AgentGeneratedPR, MonthSnapshot])

    path = _owl_path("seo-entities")
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)
    return onto


# ============================================================================
# Module 2 — seo-processes.owl
# ============================================================================

def build_processes_module(
    entities_onto: owlready2.namespace.Ontology,
) -> owlready2.namespace.Ontology:
    """Create the Evolution Process Module (Paper 1).

    Defines evolution episodes, phases, and change events.
    Imports seo-entities.

    Args:
        entities_onto: The entities ontology to import.

    Returns:
        The constructed ontology object.
    """
    onto = _onto("seo-processes")
    onto.imported_ontologies.append(entities_onto)

    with _ns(onto):
        ent = entities_onto

        # ---- Core process classes (UFO-B: perdurants) ----

        class EvolutionEpisode(Thing):
            """A discrete episode of software evolution (UFO-B: Event)."""
            comment = [locstr("Paper 1 — groups related changes.", "en")]

        class EvolutionPhase(Thing):
            """Abstract parent for all evolution phases (Paper 1)."""
            comment = [locstr("Paper 1 — base class for evolution phases.", "en")]

        class GrowthPhase(EvolutionPhase):
            """A period of active feature growth."""
            pass

        class StabilizationPhase(EvolutionPhase):
            """A period of bug-fixing and consolidation."""
            pass

        class DeclinePhase(EvolutionPhase):
            """A period of declining activity / maintenance."""
            pass

        class ChangeEvent(Thing):
            """Abstract parent for all change events (Paper 1)."""
            comment = [locstr("Paper 1 — base class for change events.", "en")]

        class RefactoringEvent(ChangeEvent):
            """A detected refactoring operation (UFO-B: atomic Event)."""
            pass

        class ArchitecturalChange(ChangeEvent):
            """A change affecting architectural structure."""
            pass

        # ---- Agent extensions ----

        class AgentAssistedPhase(EvolutionPhase):
            """An evolution phase with significant agent contributions."""
            comment = [locstr("Paper 1 — agent-aware evolution phase.", "en")]

        class AgentAdoptionEvent(ChangeEvent):
            """The event marking initial adoption of a coding agent."""
            pass

        # ---- Object properties ----

        class experiencesPhase(ObjectProperty):
            """Links a project to a phase."""
            domain = [ent.SoftwareProject]
            range = [GrowthPhase | StabilizationPhase | DeclinePhase | AgentAssistedPhase]

        class hasChange(ObjectProperty):
            """Links an episode to its changes."""
            domain = [EvolutionEpisode]
            range = [ent.Commit | RefactoringEvent | ArchitecturalChange]

        class hasRefactoring(ObjectProperty):
            """Links a version to refactoring events within it."""
            domain = [ent.Version]
            range = [RefactoringEvent]

        class occursIn(ObjectProperty):
            """Links an event to the version where it occurs."""
            domain = [EvolutionEpisode | RefactoringEvent | ArchitecturalChange]
            range = [ent.Version]

        AllDisjoint([EvolutionEpisode, EvolutionPhase, ChangeEvent])
        AllDisjoint([GrowthPhase, StabilizationPhase, DeclinePhase, AgentAssistedPhase])
        AllDisjoint([RefactoringEvent, ArchitecturalChange, AgentAdoptionEvent])

    path = _owl_path("seo-processes")
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)
    return onto


def _declare_measured_properties(ent: Any) -> dict[str, Any]:
    """Declare MEASURED_PROPERTIES in the current namespace (Paper 1, WP1).

    Must be called inside ``with onto.get_namespace(...)``. Every property is
    Functional unless kind is ``strs``; reals are xsd:decimal, counts
    xsd:integer. Each gets an rdfs:comment, a unit and a computedBy note.

    Args:
        ent: The entities ontology (for domain classes).

    Returns:
        Mapping property name -> property object.
    """
    import datetime
    import types

    domains = {"P": ent.SoftwareProject, "M": ent.MonthSnapshot, "C": ent.Commit, "Mo": ent.Module}
    ranges = {"int": int, "dec": float, "str": str, "strs": str, "dt": datetime.datetime}
    props: dict[str, Any] = {}
    for name, kind, dom, unit, by, comment in MEASURED_PROPERTIES:
        bases = ((DataProperty,) if kind == "strs" or not FUNCTIONAL_MEASURES
                 else (DataProperty, FunctionalProperty))
        prop = types.new_class(name, bases)
        prop.domain = [domains[dom]]
        prop.range = [ranges[kind]]
        prop.comment = [locstr(f"{comment} Unit: {unit}. Computed by: {by}.", "en")]
        prop.computedBy = [by]
        prop.measurementUnit = [unit]
        props[name] = prop
    return props


# ============================================================================
# Module 3 — seo-metrics.owl
# ============================================================================

def build_metrics_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
) -> owlready2.namespace.Ontology:
    """Create the Metric Module (Paper 1).

    Defines metric classes for Lehman-law operationalization.
    Imports seo-entities and seo-processes.

    Args:
        entities_onto: The entities ontology.
        processes_onto: The processes ontology.

    Returns:
        The constructed ontology object.
    """
    onto = _onto("seo-metrics")
    onto.imported_ontologies.append(entities_onto)
    onto.imported_ontologies.append(processes_onto)

    with _ns(onto):
        ent = entities_onto

        # ---- Metric classes ----

        class Metric(Thing):
            """Abstract base for all metrics (Paper 1)."""
            pass

        class ComplexityMetric(Metric):
            """Cyclomatic / cognitive complexity measurements."""
            pass

        class SizeMetric(Metric):
            """LOC, file count, function count, etc."""
            pass

        class CouplingMetric(Metric):
            """Afferent / efferent coupling measurements."""
            pass

        class CohesionMetric(Metric):
            """LCOM and similar cohesion measurements."""
            pass

        class ActivityMetric(Metric):
            """Commit frequency, churn, bus factor, etc."""
            pass

        # ---- Agent extensions ----

        class AgentContributionRatio(Metric):
            """Ratio of agent-authored commits to total commits (Paper 1)."""
            comment = [locstr("Paper 1 — key metric for agent adoption.", "en")]

        class AgentChurnRatio(Metric):
            """Ratio of agent-authored churn to total churn (Paper 1)."""
            pass

        # ---- Object properties ----

        class hasComplexity(ObjectProperty):
            """Links an entity to a complexity metric."""
            domain = [ent.SoftwareProject | ent.Module | ent.File | ent.Function]
            range = [ComplexityMetric]

        class hasSize(ObjectProperty):
            """Links an entity to a size metric."""
            domain = [ent.SoftwareProject | ent.Module | ent.File]
            range = [SizeMetric]

        class hasCoupling(ObjectProperty):
            """Links an entity to a coupling metric."""
            domain = [ent.Module | ent.File]
            range = [CouplingMetric]

        class hasCohesion(ObjectProperty):
            """Links an entity to a cohesion metric."""
            domain = [ent.Module | ent.File]
            range = [CohesionMetric]

        class hasActivity(ObjectProperty):
            """Links an entity to an activity metric."""
            domain = [ent.SoftwareProject | ent.Module]
            range = [ActivityMetric]

        class measuredAt(ObjectProperty):
            """Links a metric to the version at which it was measured."""
            domain = [Metric]
            range = [ent.Version]

        # ---- Data properties ----

        class metricValue(DataProperty, FunctionalProperty):
            """The numeric value of a metric measurement (Paper 1)."""
            domain = [Metric]
            range = [float]

        _declare_measured_properties(ent)

        AllDisjoint([ComplexityMetric, SizeMetric, CouplingMetric,
                     CohesionMetric, ActivityMetric,
                     AgentContributionRatio, AgentChurnRatio])

    path = _owl_path("seo-metrics")
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)
    return onto


# ============================================================================
# Module 4 — seo-laws.owl
# ============================================================================

def build_laws_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
    metrics_onto: owlready2.namespace.Ontology,
) -> owlready2.namespace.Ontology:
    """Create the Evolution Law Module (Paper 1).

    Encodes Lehman's 8 laws of software evolution as OWL class axioms
    and SWRL rules where possible.

    Args:
        entities_onto: The entities ontology.
        processes_onto: The processes ontology.
        metrics_onto: The metrics ontology.

    Returns:
        The constructed ontology object.
    """
    onto = _onto("seo-laws")
    onto.imported_ontologies.append(entities_onto)
    onto.imported_ontologies.append(processes_onto)
    onto.imported_ontologies.append(metrics_onto)

    with _ns(onto):
        ent = entities_onto
        proc = processes_onto
        met = metrics_onto

        # ---- Law framework classes ----

        class LehmanLaw(Thing):
            """One of Lehman's 8 laws of software evolution (Paper 1)."""
            comment = [locstr("Paper 1 — formal representation of Lehman's laws.", "en")]

        class LawConformance(Thing):
            """Evidence that a project conforms to a Lehman law."""
            pass

        class LawViolation(Thing):
            """Evidence that a project violates a Lehman law."""
            pass

        # ---- Properties ----

        class conformsToLaw(ObjectProperty):
            """Links a project to a law conformance assertion."""
            domain = [ent.SoftwareProject]
            range = [LawConformance]

        class violatesLaw(ObjectProperty):
            """Links a project to a law violation assertion."""
            domain = [ent.SoftwareProject]
            range = [LawViolation]

        class lawType(ObjectProperty):
            """Links a conformance/violation to the law it refers to."""
            domain = [LawConformance | LawViolation]
            range = [LehmanLaw]

        # ---- Helper properties for SWRL rules ----

        class usedInRealWorld(DataProperty):
            """Whether the project is deployed in a real-world setting."""
            domain = [ent.SoftwareProject]
            range = [bool]

        class activeDevelopment(DataProperty):
            """Whether the project is under active development."""
            domain = [ent.SoftwareProject]
            range = [bool]

        class succeeds(ObjectProperty):
            """v2 succeeds v1 (v2 is the next version after v1)."""
            domain = [ent.Version]
            range = [ent.Version]

        class qualityScore(DataProperty, FunctionalProperty):
            """Aggregate quality score for a version."""
            domain = [ent.Version]
            range = [float]

        class growthRate(DataProperty, FunctionalProperty):
            """Growth rate (LOC delta / time) for a version."""
            domain = [ent.Version]
            range = [float]

        class workRate(DataProperty, FunctionalProperty):
            """Work rate (commits / time) for a version."""
            domain = [ent.Version]
            range = [float]

        class incrementalGrowth(DataProperty, FunctionalProperty):
            """Incremental growth (added LOC) for a release."""
            domain = [ent.Release]
            range = [float]

        # ---- Instantiate the 8 laws as named individuals ----

        law_descriptions: list[tuple[str, str]] = [
            ("LawI_ContinuingChange",
             "A program used in a real-world environment must be continually adapted, "
             "else it becomes progressively less satisfactory."),
            ("LawII_IncreasingComplexity",
             "As a program evolves, its complexity increases unless work is done to "
             "maintain or reduce it."),
            ("LawIII_SelfRegulation",
             "The evolution process is self-regulating with statistically determinable "
             "trends and invariances."),
            ("LawIV_ConservationOfOrgStability",
             "The average effective global activity rate on an evolving system is "
             "approximately constant over its lifetime."),
            ("LawV_ConservationOfFamiliarity",
             "The average incremental growth per release tends to remain constant or "
             "decline over a system's lifetime."),
            ("LawVI_ContinuingGrowth",
             "The functional content of a system must be continually increased to "
             "maintain user satisfaction."),
            ("LawVII_DecliningQuality",
             "The quality of a system will appear to be declining unless it is "
             "rigorously maintained and adapted."),
            ("LawVIII_FeedbackSystem",
             "Evolution processes are multi-level, multi-loop, multi-agent feedback "
             "systems."),
        ]

        for name, desc in law_descriptions:
            ind = LehmanLaw(name)
            ind.comment = [locstr(desc, "en")]

        # WP1: the four SWRL rules that used to be here were unsafe (the head
        # introduced an unbound LawConformance variable, so they could never
        # fire) and the "not expressible" annotations on Laws III, IV, V and
        # VIII were wrong. All eight laws are now defined classes / safe rules
        # over asserted measurements in seo-inference.owl.
        onto.metadata.comment.append(locstr(
            "Law individuals only. The conformance rules live in seo-inference.owl "
            "(canonical, pipeline thresholds) and seo-legacy.owl (paper-table variant).",
            "en",
        ))

    path = _owl_path("seo-laws")
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)
    return onto


# ============================================================================
# Module 5 — seo-patterns.owl
# ============================================================================

def build_patterns_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
    metrics_onto: owlready2.namespace.Ontology,
    laws_onto: owlready2.namespace.Ontology,
) -> owlready2.namespace.Ontology:
    """Create the Pattern Module (Paper 1).

    Defines evolution patterns and agent-specific anti-patterns.
    Imports all preceding modules.

    Args:
        entities_onto: The entities ontology.
        processes_onto: The processes ontology.
        metrics_onto: The metrics ontology.
        laws_onto: The laws ontology.

    Returns:
        The constructed ontology object.
    """
    onto = _onto("seo-patterns")
    onto.imported_ontologies.append(entities_onto)
    onto.imported_ontologies.append(processes_onto)
    onto.imported_ontologies.append(metrics_onto)
    onto.imported_ontologies.append(laws_onto)

    with _ns(onto):
        ent = entities_onto

        # ---- Core pattern classes ----

        class EvolutionPattern(Thing):
            """Parent class for all detected evolution patterns (Paper 1)."""
            comment = [locstr("Paper 1 — base class for evolution patterns.", "en")]

        class SustainableGrowth(EvolutionPattern):
            """Healthy growth with balanced complexity management."""
            pass

        class TechnicalDebtAccumulation(EvolutionPattern):
            """Progressive build-up of technical debt."""
            pass

        class BigBangRefactoring(EvolutionPattern):
            """Large, infrequent refactoring episodes."""
            pass

        class GradualDecay(EvolutionPattern):
            """Slow decline in code quality over time."""
            pass

        class ArchitecturalErosion(EvolutionPattern):
            """Degradation of intended architecture over time."""
            pass

        # ---- Agent-specific anti-patterns ----

        class AntiPattern(EvolutionPattern):
            """Abstract parent for all evolution anti-patterns (Paper 1)."""
            comment = [locstr("Paper 1 — base class for anti-patterns, adds depth to hierarchy.", "en")]

        class AgentAssociatedComplexityGrowth(AntiPattern):
            """Complexity growth outpacing size growth while agent commits are frequent.

            Decision D4: renamed from ShadowTechDebt. The label states only an
            association measured in the repository history, not a cause.
            """
            comment = [locstr(
                "Paper 1 - agent-associated complexity growth (formerly "
                "ShadowTechDebt, renamed per decision D4).",
                "en",
            )]

        class DependencySprawl(AntiPattern):
            """Agent-driven excessive addition of new dependencies."""
            pass

        class RefactoringAvoidance(AntiPattern):
            """Pattern where agents avoid refactoring existing code."""
            pass

        class CopyPasteOverReuse(AntiPattern):
            """Agent tendency to duplicate code rather than reuse abstractions."""
            pass

        # ---- Properties ----

        class exhibitsPattern(ObjectProperty):
            """Links a project/version to a detected pattern."""
            domain = [ent.SoftwareProject | ent.Version]
            range = [EvolutionPattern]

        class patternSeverity(DataProperty, FunctionalProperty):
            """Severity score of a detected pattern (0–10 integer)."""
            domain = [EvolutionPattern]
            range = [int]

        class detectedAt(ObjectProperty):
            """Links a pattern to the version where it was detected."""
            domain = [EvolutionPattern]
            range = [ent.Version]

        AllDisjoint([SustainableGrowth, TechnicalDebtAccumulation,
                     BigBangRefactoring, GradualDecay, ArchitecturalErosion,
                     AntiPattern])
        AllDisjoint([AgentAssociatedComplexityGrowth, DependencySprawl,
                     RefactoringAvoidance, CopyPasteOverReuse])

    path = _owl_path("seo-patterns")
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)
    return onto



# ============================================================================
# Inference DSL (Paper 1, WP1)
# ============================================================================

_OPS = {">": "min_exclusive", ">=": "min_inclusive", "<": "max_exclusive", "<=": "max_inclusive"}
_COMP = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_OPNAME = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
_SPEC_RE = re.compile(r"^(-?)([A-Za-z_][A-Za-z_0-9]*)(?:([+-])(\d+(?:\.\d+)?))?$")

# Port of populate.py BUG_FIX_KEYWORDS (lines 56-61), decision D8.
BUG_FIX_KEYWORDS: list[str] = [
    "fix", "bug", "defect", "patch", "repair", "resolve",
    "issue", "error", "fault", "failure", "crash", "exception",
    "hotfix", "hot-fix", "bugfix", "bug-fix", "regression",
    "correct", "workaround",
]
# Word-boundary aware without backslashes: a keyword must not be preceded by
# a word character; any suffix is allowed (fixes, fixed, errors, corrected).
BUG_FIX_REGEX: str = "(?is).*(?<![A-Za-z0-9_])(" + "|".join(BUG_FIX_KEYWORDS) + ").*"
# Conventional Commits prefix (decision D9): feat: / feat(scope): / feat!:
FEATURE_COMMIT_REGEX: str = "(?is)feat([(][^)]*[)])?!?:.*"


def _num(v: float | int) -> str:
    """Format a number for a SWRL rule string (no scientific notation)."""
    return str(v) if isinstance(v, int) else repr(float(v))


class _Rules:
    """Builds named defined classes and safe SWRL rules from the register.

    Every else-branch is a positive complement (`atom_c`), every union operand
    is a named class (`union` asserts it), and every constant is read from
    thresholds.yaml so that WP4 can rebuild the TBox with varied thresholds
    and re-reason an unchanged ABox. Class names never contain threshold
    values (only threshold ids) so they are stable across such rebuilds.
    """

    def __init__(self, props: dict[str, Any], thr: dict[str, dict[str, Any]]) -> None:
        self.props = props
        self.kinds = {m[0]: m[1] for m in MEASURED_PROPERTIES}
        self.thr = thr
        self.cache: dict[tuple, Any] = {}
        self.created: list[Any] = []
        self.names: set[str] = set()
        self.rules: list[Any] = []

    def fork(self) -> "_Rules":
        """A builder with its own atom cache and rule list sharing property map,
        thresholds and the class-name registry (used for the independent legacy module)."""
        r = _Rules(self.props, self.thr)
        r.names = self.names
        return r

    # -- constants -----------------------------------------------------
    def val(self, spec: str | int | float) -> float | int:
        """Resolve a threshold spec: id, '-id', 'id+1', 'id-2' or a number."""
        if isinstance(spec, (int, float)):
            return spec
        m = _SPEC_RE.match(spec)
        if not m or m.group(2) not in self.thr:
            raise KeyError(f"unknown threshold spec {spec!r}")
        base = self.thr[m.group(2)]["value"]
        v = -base if m.group(1) else base
        if m.group(3):
            off = float(m.group(4))
            v = v + off if m.group(3) == "+" else v - off
            if isinstance(base, int):
                v = int(v)
        return v

    @staticmethod
    def _tag(spec: str | int | float) -> str:
        if isinstance(spec, (int, float)):
            return str(spec).replace("-", "neg").replace(".", "p")
        return spec.replace("-", "neg_").replace("+", "_plus_")

    @staticmethod
    def ids(*specs: Any) -> list[str]:
        """Threshold ids referenced by specs (numbers are skipped)."""
        out = []
        for s in specs:
            if isinstance(s, str):
                m = _SPEC_RE.match(s)
                if m:
                    out.append(m.group(2))
        return out

    # -- class factories -------------------------------------------------
    def _new(self, name: str, base: Any = Thing) -> Any:
        import types
        if name in self.names:
            raise ValueError(f"duplicate class name {name}")
        self.names.add(name)
        cls = types.new_class(name, (base,))
        self.created.append(cls)
        return cls

    def atom(self, prop: str, op: str, spec: str | int | float) -> Any:
        """Named class ``p some decimal[op k]`` for the condition ``prop op value`` (cached)."""
        key = (prop, op, spec)
        if key in self.cache:
            return self.cache[key]
        v = self.val(spec)
        name = f"A_{prop}_{_OPNAME[op]}_{self._tag(spec)}"
        text = f"{{{prop}}} {op} {v}"
        # Count properties use xsd:integer facets, real ones xsd:decimal (decimal
        # facets on integer properties made Pellet ~5x slower).
        cast = int if self.kinds[prop] == "int" else float
        if cast is int and float(v) != int(v):
            raise ValueError(f"{prop} {op} {v}: fractional constant on integer property")
        cd = ConstrainedDatatype(cast, **{_OPS[op]: cast(v)})
        cls = self._new(name)
        cls.equivalent_to.append(self.props[prop].some(cd))
        self._annotate(cls, text, self.ids(spec))
        self.cache[key] = cls
        return cls

    def atom_c(self, prop: str, op: str, spec: str | int | float) -> Any:
        """Positive complement of `atom` (x >= k for not x < k)."""
        return self.atom(prop, _COMP[op], spec)

    def between(self, prop: str, lo_op: str, lo: Any, hi_op: str, hi: Any) -> Any:
        """Named class for an interval condition on one property."""
        key = (prop, lo_op, lo, hi_op, hi)
        if key in self.cache:
            return self.cache[key]
        name = f"A_{prop}_{_OPNAME[lo_op]}_{self._tag(lo)}_{_OPNAME[hi_op]}_{self._tag(hi)}"
        text = f"{{{prop}}} within [{self.val(lo)}, {self.val(hi)}]"
        cast = int if self.kinds[prop] == "int" else float
        cd = ConstrainedDatatype(
            cast, **{_OPS[lo_op]: cast(self.val(lo)), _OPS[hi_op]: cast(self.val(hi))})
        cls = self._new(name)
        cls.equivalent_to.append(self.props[prop].some(cd))
        self._annotate(cls, text, self.ids(lo, hi))
        self.cache[key] = cls
        return cls

    def named(self, name: str, expr: Any, text: Optional[str] = None,
              thr_ids: Optional[list[str]] = None) -> Any:
        """Create a named class equivalent to expr."""
        cls = self._new(name)
        cls.equivalent_to.append(expr)
        if text:
            self._annotate(cls, text, thr_ids or [])
        return cls

    def union(self, name: str, *ops: Any, text: Optional[str] = None,
              thr_ids: Optional[list[str]] = None) -> Any:
        """Named union of NAMED classes, encoded as ``op SubClassOf name``.

        Operands must be named classes (Pellet's explanation renderer crashes
        on anonymous union operands). The union is deliberately written in the
        Horn direction (``op SubClassOf name``), not as
        ``name EquivalentTo op1 or op2``: for labelling individuals both entail
        the same memberships, no disjunction is created for the tableau, and
        Pellet 2.3.1 throws ``InternalReasonerException: Adding type to a pruned
        node`` (non-deterministically) while classifying TBoxes with many
        equivalent unions over datatype restrictions. Only the reverse
        subsumption ``name SubClassOf op1 or op2`` is lost, which no label needs.
        """
        for o in ops:
            if not isinstance(o, owlready2.ThingClass):
                raise TypeError(f"union {name}: operand {o!r} is not a named class")
        cls = self._new(name)
        for o in ops:
            o.is_a.append(cls)
        if text:
            self._annotate(cls, text, thr_ids or [])
        return cls

    def swrl(self, name: str, body: str, var: str = "?r", text: Optional[str] = None,
             thr_ids: Optional[list[str]] = None, base: Any = Thing) -> Any:
        """A class with no definition whose members are derived by one safe rule.

        The head variable is always bound in the body, so the rule is safe
        (unlike the four JSS-era rules whose head introduced a fresh variable).
        """
        cls = self._new(name, base)
        self.rule(f"{body} -> {name}({var})")
        if text:
            self._annotate(cls, text, thr_ids or [])
        return cls

    def rule(self, text: str) -> Any:
        """Add one SWRL rule."""
        r = Imp()
        r.set_as_rule(text)
        self.rules.append(r)
        return r

    def _annotate(self, cls: Any, text: str, thr_ids: list[str]) -> None:
        cls.explanationTemplate = [text]
        if thr_ids:
            uniq = list(dict.fromkeys(thr_ids))
            cls.thresholdId = uniq
            cls.thresholdSource = [self.thr[i]["source"] for i in uniq]

    @staticmethod
    def disjoint(*classes: Any) -> None:
        """Declare classes pairwise disjoint (consistency guarantee)."""
        AllDisjoint(list(classes))


# ============================================================================
# Module 6 — seo-inference.owl (canonical labels)
# ============================================================================

def build_evidence_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
    metrics_onto: owlready2.namespace.Ontology,
    rules: _Rules,
) -> owlready2.namespace.Ontology:
    """Create seo-evidence.owl: commit-level SWRL string rules (Paper 1, WP1).

    AgentCommit (trailer, bot author, branch; R1.M3), BugFixCommit (D8) and
    FeatureCommit (D9) are inferred from raw strings with SWRL ``matches`` /
    ``contains``. This is the only SWRL in the canonical modules, so it lives in
    its own module: Pellet reasons over it (stage A, chunked by 1000 commits),
    while seo-inference is SWRL-free and is reasoned with HermiT.

    Args:
        entities_onto: Entities module.
        processes_onto: Processes module.
        metrics_onto: Metrics module (commit evidence properties).
        rules: Shared rule builder.

    Returns:
        The evidence ontology.
    """
    onto = _onto("seo-evidence")
    for o in (entities_onto, processes_onto, metrics_onto):
        onto.imported_ontologies.append(o)
    R = rules

    with _ns(onto):
        Commit = entities_onto.Commit
        AgentCommit = entities_onto.AgentCommit

        # ---------------- commit-level evidence (stage A) ----------------
        by_trailer = R._new("AgentCommitByTrailer", Commit)
        by_bot = R._new("AgentCommitByBotAuthor", Commit)
        by_branch = R._new("AgentCommitByBranch", Commit)
        vendors = {n: R._new(n, Commit) for n in
                   ("ClaudeCodeCommit", "CopilotCommit", "CodexCommit", "DevinCommit")}
        for v in vendors.values():
            v.is_a.append(AgentCommit)
        R._annotate(by_trailer, "Co-authored-by trailer names an agent: {coAuthorTrailer}", [])
        R._annotate(by_bot, "author name marks a bot account: {authorName}", [])
        R._annotate(by_branch, "pull-request branch has an agent prefix: {branchName}", [])
        # Regexes from populate.py AGENT_COAUTHOR_PATTERNS (lines 63-75).
        for vendor, rx in (
            ("ClaudeCodeCommit", "(?is).*co-authored-by:.*claude.*(@anthropic|noreply).*"),
            ("CopilotCommit", "(?is).*co-authored-by:.*(copilot.*@github|github-copilot).*"),
            ("CodexCommit", "(?is).*co-authored-by:.*codex.*@openai.*"),
        ):
            R.rule(f'Commit(?c), coAuthorTrailer(?c, ?t), matches(?t, "{rx}") '
                   f"-> AgentCommitByTrailer(?c), {vendor}(?c)")
        # Bot author: '[bot]' anywhere in the name (the old code lowercased first).
        R.rule('Commit(?c), authorName(?c, ?a), contains(?a, "[bot]") -> AgentCommitByBotAuthor(?c)')
        # Branch prefixes from shared/mining/agent_detector.py BRANCH_PATTERNS.
        # No branch data is mined (D9) so these rules stay inert on real repos.
        for vendor, rx in (
            ("CodexCommit", "(?i)codex[/-].*"),
            ("DevinCommit", "(?i)devin[/-].*"),
            ("ClaudeCodeCommit", "(?i)claude[/-].*"),
            ("CopilotCommit", "(?i)copilot(/|-fix-).*"),
        ):
            R.rule(f'Commit(?c), branchName(?c, ?b), matches(?b, "{rx}") '
                   f"-> AgentCommitByBranch(?c), {vendor}(?c)")
        for evidence_cls in (by_trailer, by_bot, by_branch):
            evidence_cls.is_a.append(AgentCommit)  # union as Horn subclass axioms, see _Rules.union
        AgentCommit.comment.append(locstr(
            "Inferred: each of trailer, bot-author and branch evidence is a subclass. Message markers "
            "and 'touches CLAUDE.md' are deliberately not evidence (WP1). The codex/devin "
            "author-name substring rule of populate.py is dropped pending a WP1 decision.",
            "en"))
        R._annotate(AgentCommit, "commit carries agent evidence (see the evidence class)", [])

        R.swrl("BugFixCommit",
               f'Commit(?c), commitSubject(?c, ?s), matches(?s, "{BUG_FIX_REGEX}")',
               var="?c", base=Commit,
               text='commit subject "{commitSubject}" contains a bug-fix keyword')
        R.swrl("FeatureCommit",
               f'Commit(?c), commitSubject(?c, ?s), matches(?s, "{FEATURE_COMMIT_REGEX}")',
               var="?c", base=Commit,
               text='commit subject "{commitSubject}" has a Conventional-Commit feat prefix')

    _save(onto, "seo-evidence")
    return onto


def build_inference_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
    metrics_onto: owlready2.namespace.Ontology,
    laws_onto: owlready2.namespace.Ontology,
    patterns_onto: owlready2.namespace.Ontology,
    rules: _Rules,
) -> owlready2.namespace.Ontology:
    """Create seo-inference.owl: every label-deciding axiom (Paper 1, WP1).

    Canonical content only (decisions D6-D9): Mann-Kendall stages, the eight
    laws with the pipeline thresholds as disjoint Conformant / NonConformant
    pairs, and the anti-pattern classes. SWRL-free OWL 2 DL: comparisons of one
    measurement with another times a constant use measured ratios of two
    measurements (``bugFixHalfRatio``...), so HermiT can reason over it
    deterministically. Commit-level rules are in seo-evidence.owl; the legacy
    ensemble and the paper-table law variant are in seo-legacy.owl.

    Args:
        entities_onto: Entities module.
        processes_onto: Processes module.
        metrics_onto: Metrics module (measured properties).
        laws_onto: Laws module.
        patterns_onto: Patterns module.
        rules: Shared rule builder holding the property map and thresholds.

    Returns:
        The inference ontology.
    """
    onto = _onto("seo-inference")
    for o in (entities_onto, processes_onto, metrics_onto, laws_onto, patterns_onto):
        onto.imported_ontologies.append(o)
    R = rules
    A, Ac = R.atom, R.atom_c

    with _ns(onto):
        SP = entities_onto.SoftwareProject

        # ---------------- Mann-Kendall stages (D6) ----------------
        MKApplicable = R.named("MKApplicable", A("locSeriesLength", ">=", "mk_min_months"),
                               "{locSeriesLength} months of LOC", ["mk_min_months"])
        p_lt, p_ge = A("locMKpValue", "<", "mk_alpha"), Ac("locMKpValue", "<", "mk_alpha")
        tau_pos, tau_neg = A("locMKTau", ">", 0), A("locMKTau", "<", 0)
        mk_ids = ["mk_alpha", "mk_min_months"]
        GrowthStage = R.named(
            "GrowthStage", SP & MKApplicable & p_lt & tau_pos,
            "significant growing LOC trend (Mann-Kendall tau = {locMKTau}, p = {locMKpValue}, "
            "{locSeriesLength} months)", mk_ids)
        DeclineStage = R.named(
            "DeclineStage", SP & MKApplicable & p_lt & tau_neg,
            "significant shrinking LOC trend (Mann-Kendall tau = {locMKTau}, p = {locMKpValue}, "
            "{locSeriesLength} months)", mk_ids)
        StabilizationStage = R.named(
            "StabilizationStage", SP & MKApplicable & p_ge,
            "no significant LOC trend (Mann-Kendall tau = {locMKTau}, p = {locMKpValue}, "
            "{locSeriesLength} months)", mk_ids)
        R.disjoint(GrowthStage, DeclineStage, StabilizationStage)

        # ---------------- C2: Rajlich-Bennett lifecycle rule set (RQ1 pre-registration 2.2) ----------------
        # C1 above is unchanged; C2 lives in additional classes so both are inferred in one pass.
        rb_d, rb_g = ["rb_dormancy_months"], ["rb_min_rel_loc_growth_per_year"]
        Dormant = R.named(
            "Dormant", SP & A("monthsSinceLastCommit", ">=", "rb_dormancy_months"),
            "no commit for {monthsSinceLastCommit} months", rb_d)
        RecentlyActive = R.named(
            "RecentlyActive", SP & Ac("monthsSinceLastCommit", ">=", "rb_dormancy_months"),
            "last commit {monthsSinceLastCommit} months before the snapshot", rb_d)
        SubstantialLocGrowth = R.named(
            "SubstantialLocGrowth",
            SP & A("locRelSenSlopeAnnual", ">=", "rb_min_rel_loc_growth_per_year"),
            "net LOC growth of {locRelSenSlopeAnnual} of median size per year (Sen slope)", rb_g)
        MinorLocGrowth = R.named(
            "MinorLocGrowth",
            SP & Ac("locRelSenSlopeAnnual", ">=", "rb_min_rel_loc_growth_per_year"),
            "net LOC growth of only {locRelSenSlopeAnnual} of median size per year (Sen slope)", rb_g)
        NonSubstantialChange = R.union(
            "NonSubstantialChange", StabilizationStage, DeclineStage, MinorLocGrowth,
            text="no significant LOC growth, or growth below the substantial-growth rate",
            thr_ids=mk_ids + rb_g)
        RBPhaseOutStage = R.named(
            "RBPhaseOutStage", SP & Dormant,
            "phase-out: no commit for {monthsSinceLastCommit} months", rb_d)
        RBActiveDevelopmentStage = R.named(
            "RBActiveDevelopmentStage", SP & RecentlyActive & GrowthStage & SubstantialLocGrowth,
            "active development: recent commits, significant LOC growth (tau = {locMKTau}, "
            "p = {locMKpValue}) of {locRelSenSlopeAnnual} of median size per year", rb_d + mk_ids + rb_g)
        RBServicingStage = R.named(
            "RBServicingStage", SP & RecentlyActive & NonSubstantialChange,
            "servicing: recent commits, but no substantial LOC growth", rb_d + mk_ids + rb_g)
        R.disjoint(RBActiveDevelopmentStage, RBServicingStage, RBPhaseOutStage)

        # ---------------- the eight laws (pipeline thresholds, D7) ----------------
        def law_pair(roman: str, conf: Any, nc: Any, ct: str, nt: str, ids: list[str]) -> None:
            R._annotate(conf, ct, ids)
            R._annotate(nc, nt, ids)
            R.disjoint(conf, nc)

        def mk_pair(roman: str, conf_expr: Any, nc_expr: Any, ct: str, nt: str,
                    ids: list[str]) -> None:
            c = R.named(f"Law{roman}Conformant", conf_expr)
            n = R.named(f"Law{roman}NonConformant", nc_expr)
            law_pair(roman, c, n, ct, nt, ids)

        # I
        g1 = A("monthsObserved", ">=", "p_law1_min_months")
        mk_pair("I", SP & g1 & A("activeMonthRatio", ">=", "p_law1_active_ratio"),
                SP & g1 & Ac("activeMonthRatio", ">=", "p_law1_active_ratio"),
                "{activeMonthCount} of {monthsObserved} months active (ratio {activeMonthRatio} >= "
                f"{R.val('p_law1_active_ratio')})",
                "{activeMonthCount} of {monthsObserved} months active (ratio {activeMonthRatio} < "
                f"{R.val('p_law1_active_ratio')})",
                ["p_law1_min_months", "p_law1_active_ratio"])
        # II: difference of the two half-means
        g2 = A("complexityMonthCount", ">=", "p_law2_min_complexity_months")
        mk_pair("II", SP & g2 & A("complexityHalfMeanChange", ">=", 0),
                SP & g2 & Ac("complexityHalfMeanChange", ">=", 0),
                "second-half mean complexity {complexityHalfMeanSecond} >= first-half {complexityHalfMeanFirst}",
                "second-half mean complexity {complexityHalfMeanSecond} < first-half {complexityHalfMeanFirst}",
                ["p_law2_min_complexity_months"])
        # III, IV, V: statistical laws over asserted coefficients of variation
        for roman, gid, prop, tid, nm in (
            ("III", "p_law3_min_months", "churnCV", "p_law3_churn_cv", "churn"),
            ("IV", "p_law4_min_months", "commitCountCV", "p_law4_commit_cv", "commit-rate"),
            ("V", "p_law5_min_months", "linesAddedCV", "p_law5_additions_cv", "lines-added"),
        ):
            g = A("monthsObserved", ">=", gid)
            mk_pair(roman, SP & g & A(prop, "<", tid), SP & g & Ac(prop, "<", tid),
                    f"{nm} CV {{{prop}}} < {R.val(tid)} over {{monthsObserved}} months",
                    f"{nm} CV {{{prop}}} >= {R.val(tid)} over {{monthsObserved}} months",
                    [gid, tid])
        # VI: chained inference over the inferred stage
        c6 = R.union("LawVIConformant", GrowthStage, StabilizationStage)
        n6 = R.named("LawVINonConformant", DeclineStage)
        law_pair("VI", c6, n6, "stage is growth or stabilization (see the stage justification)",
                 "stage is decline (see the stage justification)", mk_ids)
        # VII: half ratios come from the post-stage-A aggregate (D8). The condition
        # second >= tolerance * first is the measured ratio second/first >= tolerance
        # when first > 0, and true when first = 0.
        g7 = A("activeMonthCount", ">=", "p_law7_min_active_months")
        first_pos, first_zero = A("bugFixRatioFirstHalf", ">", 0), A("bugFixRatioFirstHalf", "<=", 0)
        c7 = R.union("LawVIIConformant",
                     R.named("LawVIIConformantRatio",
                             SP & g7 & first_pos & A("bugFixHalfRatio", ">=", "p_law7_tolerance")),
                     R.named("LawVIIConformantNoFixesBefore", SP & g7 & first_zero))
        n7 = R.named("LawVIINonConformant", SP & g7 & first_pos & Ac("bugFixHalfRatio", ">=", "p_law7_tolerance"))
        law_pair("VII", c7, n7,
                 "bug-fix ratio second half {bugFixRatioSecondHalf} >= "
                 f"{R.val('p_law7_tolerance')} x first half {{bugFixRatioFirstHalf}}",
                 "bug-fix ratio second half {bugFixRatioSecondHalf} < "
                 f"{R.val('p_law7_tolerance')} x first half {{bugFixRatioFirstHalf}}",
                 ["p_law7_min_active_months", "p_law7_tolerance"])
        # VIII
        g8 = A("monthsObserved", ">=", "p_law8_min_months")
        merge_a = R.named("MergeActive", A("mergeMonthRatio", ">", "p_law8_merge_ratio"))
        merge_i = R.named("MergeInactive", Ac("mergeMonthRatio", ">", "p_law8_merge_ratio"))
        auth_a = R.named("MultiAuthorActive", A("multiAuthorMonthRatio", ">", "p_law8_multiauthor_ratio"))
        auth_i = R.named("MultiAuthorInactive", Ac("multiAuthorMonthRatio", ">", "p_law8_multiauthor_ratio"))
        c8 = R.named("LawVIIIConformant", SP & g8 & merge_a & auth_a)
        nc8a = R.named("LawVIIINonConformantNoMerge", SP & g8 & merge_i)
        nc8b = R.named("LawVIIINonConformantSingleAuthor", SP & g8 & auth_i)
        n8 = R.union("LawVIIINonConformant", nc8a, nc8b)
        ids8 = ["p_law8_min_months", "p_law8_merge_ratio", "p_law8_multiauthor_ratio"]
        law_pair("VIII", c8, n8,
                 f"merge months {{mergeMonthRatio}} > {R.val('p_law8_merge_ratio')} and multi-author "
                 f"months {{multiAuthorMonthRatio}} > {R.val('p_law8_multiauthor_ratio')}",
                 f"merge months {{mergeMonthRatio}} or multi-author months {{multiAuthorMonthRatio}} "
                 "at or below the threshold", ids8)
        R._annotate(nc8a, f"merge months {{mergeMonthRatio}} <= {R.val('p_law8_merge_ratio')}", ids8)
        R._annotate(nc8b, "multi-author months {multiAuthorMonthRatio} <= "
                    f"{R.val('p_law8_multiauthor_ratio')}", ids8)

        # ---------------- anti-patterns (project-level classes) ----------------
        gA = A("monthsObserved", ">=", "a_min_months")

        def recent(spec: str) -> Any:
            return A("recentAgentCommitRatio", ">", spec)

        # cg > f * lg, by the sign of lg (dividing by a negative flips the inequality)
        lg_pos, lg_neg = A("locGrowthRate", ">", 0), A("locGrowthRate", "<", 0)
        lg_zero = R.named("LocGrowthZero", A("locGrowthRate", ">=", 0) & A("locGrowthRate", "<=", 0))
        outpace = R.union(
            "ComplexityOutpacesLocGrowth",
            R.named("ComplexityOutpacesLocGrowthUp",
                    SP & lg_pos & A("complexityToLocGrowthRatio", ">", "a_cx_vs_loc_factor")),
            R.named("ComplexityOutpacesLocGrowthDown",
                    SP & lg_neg & A("complexityToLocGrowthRatio", "<", "a_cx_vs_loc_factor")),
            R.named("ComplexityOutpacesLocGrowthFlat", SP & lg_zero & A("complexityGrowthRate", ">", 0)),
            text="complexity growth {complexityGrowthRate} > "
                 f"{R.val('a_cx_vs_loc_factor')} x LOC growth {{locGrowthRate}}",
            thr_ids=["a_cx_vs_loc_factor"])
        R.named("ExhibitsAgentAssociatedComplexityGrowth",
                SP & gA & recent("a_agent_shadow") & outpace
                & A("complexityGrowthRate", ">", "a_cx_growth_shadow"),
                "recent agent-commit ratio {recentAgentCommitRatio} > "
                f"{R.val('a_agent_shadow')} while complexity growth {{complexityGrowthRate}} "
                f"outpaces LOC growth {{locGrowthRate}} (association only)",
                ["a_agent_shadow", "a_cx_vs_loc_factor", "a_cx_growth_shadow", "a_min_months"])
        R.named("ExhibitsDependencySprawl",
                SP & gA & A("dependencyGrowthRate", ">", "a_dep_growth") & recent("a_agent_dep"),
                "dependency growth {dependencyGrowthRate} > "
                f"{R.val('a_dep_growth')} with recent agent ratio {{recentAgentCommitRatio}} "
                "(no dependency data is mined, D9)",
                ["a_dep_growth", "a_agent_dep", "a_min_months"])
        R.named("ExhibitsRefactoringAvoidance",
                SP & gA & recent("a_agent_refactor")
                & A("refactoringRatio", "<", "a_refactor_ratio")
                & A("complexityGrowthRate", ">", "a_cx_growth_refactor"),
                "recent agent ratio {recentAgentCommitRatio} > "
                f"{R.val('a_agent_refactor')}, refactoring ratio {{refactoringRatio}} < "
                f"{R.val('a_refactor_ratio')}, complexity growth {{complexityGrowthRate}}",
                ["a_agent_refactor", "a_refactor_ratio", "a_cx_growth_refactor", "a_min_months"])
        lpf = R.named("LocPerFeatureHigh", SP & A("locDeltaPerFeature", ">", "a_loc_per_feature"),
                      f"LOC added per feature commit ({{locDeltaPerFeature}}) exceeds {R.val('a_loc_per_feature')}",
                      ["a_loc_per_feature"])
        # lg > f * cg, by the sign of cg
        cg_pos, cg_neg = A("complexityGrowthRate", ">", 0), A("complexityGrowthRate", "<", 0)
        cg_zero = R.named("ComplexityGrowthZero",
                          A("complexityGrowthRate", ">=", 0) & A("complexityGrowthRate", "<=", 0))
        loc_outpaces = R.union(
            "LocOutpacesComplexityGrowth",
            R.named("LocOutpacesComplexityGrowthUp",
                    SP & cg_pos & A("locToComplexityGrowthRatio", ">", "a_loc_vs_cx_factor")),
            R.named("LocOutpacesComplexityGrowthDown",
                    SP & cg_neg & A("locToComplexityGrowthRatio", "<", "a_loc_vs_cx_factor")),
            R.named("LocOutpacesComplexityGrowthFlat", SP & cg_zero & A("locGrowthRate", ">", 0)),
            text=f"LOC growth {{locGrowthRate}} > {R.val('a_loc_vs_cx_factor')} x complexity "
                 "growth {complexityGrowthRate}",
            thr_ids=["a_loc_vs_cx_factor"])
        bloat = R.named("LocBloatBranch",
                        SP & loc_outpaces & A("locGrowthRate", ">", "a_loc_growth_copy2"))
        evidence = R.union("CopyPasteEvidence", lpf, bloat)
        R.named("ExhibitsCopyPasteOverReuse",
                SP & gA & recent("a_agent_copy") & A("locGrowthRate", ">", "a_loc_growth_copy")
                & evidence,
                "recent agent ratio {recentAgentCommitRatio} > "
                f"{R.val('a_agent_copy')} with LOC growth {{locGrowthRate}} > "
                f"{R.val('a_loc_growth_copy')} and bloat evidence",
                ["a_agent_copy", "a_loc_growth_copy", "a_loc_per_feature", "a_loc_vs_cx_factor",
                 "a_loc_growth_copy2", "a_min_months"])

        # ---------------- WP5: module stability (wp5-analysis-plan.md 3.2) ----------------
        # Measurements are asserted by Python (measures.module_measurements); the labels are
        # SWRL-free defined classes. A module without moduleRelChurnInWindow (LOC 0) gets neither
        # Stable nor Low/High; the writer also withholds the other rule inputs for it, so it
        # stays undetermined.
        Mod = entities_onto.Module
        W = R.val("mod_window_months")
        ms_ids = ["mod_stable_max_rel_churn", "mod_stable_max_change_months"]
        LowRel = R.named("LowRelChurnModule", Mod & A("moduleRelChurnInWindow", "<", "mod_stable_max_rel_churn"),
                         f"{{moduleRelChurnInWindow}} of its lines changed in the last {W} months, "
                         f"below {R.val('mod_stable_max_rel_churn')}", ["mod_stable_max_rel_churn"])
        HighRel = R.named("HighRelChurnModule",
                          Mod & Ac("moduleRelChurnInWindow", "<", "mod_stable_max_rel_churn"),
                          f"{{moduleRelChurnInWindow}} of its lines changed in the last {W} months",
                          ["mod_stable_max_rel_churn"])
        Rarely = R.named("RarelyChangedModule",
                         Mod & A("moduleChangeMonthsInWindow", "<=", "mod_stable_max_change_months"),
                         f"changed in {{moduleChangeMonthsInWindow}} of the last {W} months, "
                         f"at most {R.val('mod_stable_max_change_months')}", ["mod_stable_max_change_months"])
        Often = R.named("OftenChangedModule",
                        Mod & Ac("moduleChangeMonthsInWindow", "<=", "mod_stable_max_change_months"),
                        f"changed in {{moduleChangeMonthsInWindow}} of the last {W} months",
                        ["mod_stable_max_change_months"])
        StableModule = R.named(
            "StableModule", LowRel & Rarely,
            f"changed in {{moduleChangeMonthsInWindow}} of the last {W} months; "
            "lines changed = {moduleRelChurnInWindow} of its size", ms_ids)
        ActiveModule = R.union(
            "ActiveModule", HighRel, Often,
            text="{moduleRelChurnInWindow} of its lines changed in the last "
                 f"{W} months, or it changed in {{moduleChangeMonthsInWindow}} of those months",
            thr_ids=ms_ids)
        R.disjoint(StableModule, ActiveModule)

    _save(onto, "seo-inference")
    return onto


# ============================================================================
# Module 7 — seo-legacy.owl (regression artefact, never labels real repos)
# ============================================================================

def build_legacy_module(
    entities_onto: owlready2.namespace.Ontology,
    processes_onto: owlready2.namespace.Ontology,
    metrics_onto: owlready2.namespace.Ontology,
    rules: _Rules,
) -> owlready2.namespace.Ontology:
    """Create seo-legacy.owl (Paper 1, WP1; decisions D6 and D7).

    * ``Legacy{Growth,Decline,Stabilization}Stage``: the exact
      ``test_scenarios.determine_phase_from_snapshot`` 3-vote ensemble with its
      two overrides, as defined classes (votes, 2-of-3 majority, tie to the LOC
      vote, override precedence). It stays so the 20-scenario regression and
      the published kappa remain reproducible; it never labels real repos.
    * ``PaperLaw{I..VIII}{Conformant,NonConformant}``: the paper-table law
      thresholds (``test_scenarios.check_law_conformance``).

    The module is SWRL-free OWL 2 DL and does not import seo-inference, so it
    is reasoned with HermiT (``HermitReasoner``). Pellet 2.3.1 cannot be used:
    it crashes while classifying it ("Adding type to a pruned node") and, with
    rule-derived threshold classes, returns different and incomplete instance
    sets on identical input. Comparisons of one measurement with another times
    a constant (``f2 > 0.7 f1``) therefore use measured RATIOS of two
    measurements (``featureSecondToFirstRatio`` ...) with the constant in the
    class; the denominator-zero case is a separate class. Per-month closure
    classes carry the "last 3 months" and "every month" windows and need
    ``hasMonth only {...}`` closure in the ABox. Law VIII needs counts of
    inferred months, a post-month-stage aggregate (a second reasoning pass).

    Args:
        entities_onto: Entities module.
        processes_onto: Processes module (imported by metrics).
        metrics_onto: Metrics module.
        rules: Rule builder (a fork of the canonical one: own atom cache).

    Returns:
        The legacy ontology.
    """
    onto = _onto("seo-legacy")
    for o in (entities_onto, processes_onto, metrics_onto):
        onto.imported_ontologies.append(o)
    R = rules
    A, Ac = R.atom, R.atom_c

    with _ns(onto):
        SP = entities_onto.SoftwareProject
        MS = entities_onto.MonthSnapshot
        hasMonth = entities_onto.hasMonth
        nextMonth = entities_onto.nextMonth
        n2 = A("monthsObserved", ">=", "e_min_months")      # guard shared by votes
        n4 = A("monthsObserved", ">=", "e_override_min_months")
        n_lt4 = Ac("monthsObserved", ">=", "e_override_min_months")
        ids_min = ["e_min_months"]

        def ratio_cmp(name: str, op: str, factor: str) -> Any:
            """Class for ``featureCommitsSecondHalf op factor * featureCommitsFirstHalf``.

            The measured ratio f2 / f1 exists only for f1 > 0; with f1 = 0 the
            condition reduces to ``f2 op 0`` (a separate class).
            """
            pos = R.named(f"{name}Positive",
                          SP & A("featureCommitsFirstHalf", ">=", 1) & A("featureSecondToFirstRatio", op, factor))
            zero_f1 = A("featureCommitsFirstHalf", "<=", 0)
            parts = [pos]
            if op == ">":
                parts.append(R.named(f"{name}Zero", SP & zero_f1 & A("featureCommitsSecondHalf", ">", 0)))
            elif op == "<=":
                parts.append(R.named(f"{name}Zero", SP & zero_f1 & A("featureCommitsSecondHalf", "<=", 0)))
            elif op == ">=":
                parts.append(R.named(f"{name}Zero", SP & zero_f1))
            return R.union(name, *parts,
                           text=f"second-half features {{featureCommitsSecondHalf}} {op} {R.val(factor)} x "
                                "first-half {featureCommitsFirstHalf}", thr_ids=[factor])

        # ---------------- LOC vote ----------------
        LG = R.named("LocVoteGrowth", SP & n2 & A("locMKTau", ">", "e_tau")
                     & A("locChangePct", ">", "e_loc_growth_pct"),
                     "LOC vote: growth (tau {locMKTau}, change {locChangePct}%)",
                     ["e_tau", "e_loc_growth_pct"])
        LD = R.named("LocVoteDecline", SP & n2 & A("locMKTau", "<", "-e_tau")
                     & A("locChangePct", "<", "e_loc_decline_pct"),
                     "LOC vote: decline (tau {locMKTau}, change {locChangePct}%)",
                     ["e_tau", "e_loc_decline_pct"])
        not_lg = R.union("LocNotGrowth", Ac("locMKTau", ">", "e_tau"),
                         Ac("locChangePct", ">", "e_loc_growth_pct"))
        not_ld = R.union("LocNotDecline", Ac("locMKTau", "<", "-e_tau"),
                         Ac("locChangePct", "<", "e_loc_decline_pct"))
        LS = R.named("LocVoteStabilization", SP & n2 & not_lg & not_ld,
                     "LOC vote: stabilization (tau {locMKTau}, change {locChangePct}%)",
                     ["e_tau", "e_loc_growth_pct", "e_loc_decline_pct"])

        # ---------------- commit vote ----------------
        r_lt05, r_ge05 = A("commitHalfRatio", "<", "e_commit_collapse"), Ac("commitHalfRatio", "<", "e_commit_collapse")
        r_gt115, r_le115 = A("commitHalfRatio", ">", "e_commit_growth"), Ac("commitHalfRatio", ">", "e_commit_growth")
        m2_gt20, m2_le20 = (A("commitSecondHalfMean", ">", "e_commit_growth_min_mean"),
                            Ac("commitSecondHalfMean", ">", "e_commit_growth_min_mean"))
        r_lt07, r_ge07 = A("commitHalfRatio", "<", "e_commit_wind_down"), Ac("commitHalfRatio", "<", "e_commit_wind_down")
        m2_lt15, m2_ge15 = A("commitSecondHalfMean", "<", "e_commit_low_mean"), Ac("commitSecondHalfMean", "<", "e_commit_low_mean")
        cd1 = R.named("CommitCollapse", SP & n2 & r_lt05)
        cd2 = R.named("CommitWindDownLow", SP & n2 & r_ge05 & r_lt07 & m2_lt15)
        CD = R.union("CommitVoteDecline", cd1, cd2,
                     text="commit vote: decline (second/first half ratio {commitHalfRatio}, "
                          "second-half mean {commitSecondHalfMean})",
                     thr_ids=["e_commit_collapse", "e_commit_wind_down", "e_commit_low_mean"])
        CG = R.named("CommitVoteGrowth", SP & n2 & r_gt115 & m2_gt20,
                     "commit vote: growth (ratio {commitHalfRatio}, second-half mean {commitSecondHalfMean})",
                     ["e_commit_growth", "e_commit_growth_min_mean"])
        cs1 = R.named("CommitWindDownKept", SP & n2 & r_ge05 & r_lt07 & m2_ge15)
        not_cg = R.union("CommitNotGrowth", r_le115, m2_le20)
        cs2 = R.named("CommitSteady", SP & n2 & r_ge07 & not_cg)
        CS = R.union("CommitVoteStabilization", cs1, cs2,
                     text="commit vote: stabilization (ratio {commitHalfRatio})",
                     thr_ids=["e_commit_wind_down", "e_commit_growth", "e_commit_growth_min_mean"])

        # ---------------- feature vote ----------------
        f2_above = ratio_cmp("FeatSecondAboveGrowthShare", ">", "e_feat_second_vs_first")
        f2_atmost = ratio_cmp("FeatSecondAtMostGrowthShare", "<=", "e_feat_second_vs_first")
        f2_below_stab = ratio_cmp("FeatSecondBelowStabShare", "<", "e_feat_stab_factor")
        f2_atleast_stab = ratio_cmp("FeatSecondAtLeastStabShare", ">=", "e_feat_stab_factor")
        f2_below_half = ratio_cmp("FeatSecondBelowHalfOfFirst", "<", "e_decl_feat_factor")
        f2_atleast_half = ratio_cmp("FeatSecondAtLeastHalfOfFirst", ">=", "e_decl_feat_factor")

        FA = R.named("FeatGrowthPrimary", SP & n2 & A("featureCommitsTotal", ">", "e_feat_total_growth") & f2_above)
        not_fa = R.union("FeatNotGrowthPrimary", Ac("featureCommitsTotal", ">", "e_feat_total_growth"), f2_atmost)
        fb = R.union("FeatLowActivity", A("featureCommitsTotal", "<", "e_feat_total_low"),
                     A("featureCommitsSecondHalf", "<", "e_feat_second_low"))
        not_fb = R.named("FeatNotLowActivity", Ac("featureCommitsTotal", "<", "e_feat_total_low")
                         & Ac("featureCommitsSecondHalf", "<", "e_feat_second_low"))
        m2f_lt15, m2f_ge15 = (A("commitSecondHalfMean", "<", "e_feat_low_mean"),
                              Ac("commitSecondHalfMean", "<", "e_feat_low_mean"))
        rf_lt07, rf_ge07 = A("commitHalfRatio", "<", "e_feat_low_ratio"), Ac("commitHalfRatio", "<", "e_feat_low_ratio")
        FD = R.named("FeatureVoteDecline", SP & n2 & not_fa & fb & m2f_lt15 & rf_lt07,
                     "feature vote: decline (features {featureCommitsFirstHalf} -> {featureCommitsSecondHalf})",
                     ["e_feat_total_low", "e_feat_second_low", "e_feat_low_mean", "e_feat_low_ratio"])
        not_winding = R.union("FeatCommitsNotWindingDown", m2f_ge15, rf_ge07)
        fs1 = R.named("FeatLowActivityKept", SP & n2 & not_fa & fb & not_winding)
        fs2 = R.named("FeatModerateBelowStab", SP & n2 & not_fa & not_fb & f2_below_stab)
        fg2 = R.named("FeatModerateAtLeastStab", SP & n2 & not_fa & not_fb & f2_atleast_stab)
        FG = R.union("FeatureVoteGrowth", FA, fg2,
                     text="feature vote: growth (features {featureCommitsFirstHalf} -> {featureCommitsSecondHalf})",
                     thr_ids=["e_feat_total_growth", "e_feat_second_vs_first", "e_feat_stab_factor"])
        FS = R.union("FeatureVoteStabilization", fs1, fs2,
                     text="feature vote: stabilization (features {featureCommitsFirstHalf} -> {featureCommitsSecondHalf})",
                     thr_ids=["e_feat_stab_factor", "e_feat_total_low", "e_feat_second_low"])

        # ---------------- majority, tie to LOC vote ----------------
        def majority(tag: str, a: Any, b: Any, c: Any) -> Any:
            ab = R.named(f"Majority{tag}_ab", a & b)
            ac = R.named(f"Majority{tag}_ac", a & c)
            bc = R.named(f"Majority{tag}_bc", b & c)
            return R.union(f"Majority{tag}", ab, ac, bc)

        MG, MD, MS_ = majority("Growth", LG, CG, FG), majority("Decline", LD, CD, FD), majority("Stabilization", LS, CS, FS)
        perms = [(LG, CD, FS), (LG, CS, FD), (LD, CG, FS), (LD, CS, FG), (LS, CG, FD), (LS, CD, FG)]
        NM = R.union("NoMajority", *[R.named(f"NoMajority_{i}", a & b & c)
                                     for i, (a, b, c) in enumerate(perms)])

        # ---------------- per-month classes and overrides ----------------
        FirstMonth = R.named("FirstMonth", A("monthIndex", "<=", 0))
        CommitIncreaseMonth = A("commitDeltaFromPrev", ">", 0)
        CommitNonIncreaseMonth = Ac("commitDeltaFromPrev", ">", 0)
        non_inc_or_first = R.union("CommitNonIncreaseOrFirstMonth", CommitNonIncreaseMonth, FirstMonth)
        Monotone = R.named("CommitsMonotoneNonIncreasing",
                           SP & hasMonth.only(non_inc_or_first) & hasMonth.some(MS),
                           "commits never increase month on month", [])
        NotMonotone = R.named("CommitsNotMonotone", SP & hasMonth.some(CommitIncreaseMonth))
        r_lt065, r_ge065 = A("commitHalfRatio", "<", "e_decl_ratio"), Ac("commitHalfRatio", "<", "e_decl_ratio")
        DO = R.named("LegacyDeclineOverride", SP & n2 & n4 & Monotone & r_lt065 & f2_below_half,
                     "decline override: commits fall monotonically (ratio {commitHalfRatio}) and features "
                     "{featureCommitsFirstHalf} -> {featureCommitsSecondHalf}",
                     ["e_override_min_months", "e_decl_ratio", "e_decl_feat_factor"])
        NotDO = R.union("LegacyNoDeclineOverride", NotMonotone, r_ge065, f2_atleast_half)

        OlderMonth = A("monthIndexFromEnd", ">=", "e_grow_recent_months-1")
        LocRoseMonth = A("locDeltaFromPrev", ">", 0)
        LocNotRoseMonth = Ac("locDeltaFromPrev", ">", 0)
        older_or_rose = R.union("OlderOrLocRoseMonth", OlderMonth, LocRoseMonth)
        RecentRising = R.named("RecentLocRising",
                               SP & hasMonth.only(older_or_rose) & hasMonth.some(MS),
                               "LOC rises in each of the last months", ["e_grow_recent_months"])
        RecentNonRisingMonth = R.named(
            "RecentNonRisingMonth",
            A("monthIndexFromEnd", "<=", "e_grow_recent_months-2") & LocNotRoseMonth)
        NotRecentRising = R.named("RecentLocNotRising", SP & hasMonth.some(RecentNonRisingMonth))
        g3_gt, g3_le = A("locGrowthLast3", ">", "e_grow_recent_growth"), Ac("locGrowthLast3", ">", "e_grow_recent_growth")
        f2_gt30, f2_le30 = A("featureCommitsSecondHalf", ">", "e_grow_features"), Ac("featureCommitsSecondHalf", ">", "e_grow_features")
        GOcore = R.named("LegacyGrowthOverrideCore", SP & n2 & n4 & RecentRising & g3_gt & f2_gt30)
        GO = R.named("LegacyGrowthOverride", GOcore & NotDO,
                     "growth override: LOC rising over the last months ({locGrowthLast3}), features "
                     "{featureCommitsSecondHalf}",
                     ["e_grow_recent_months", "e_grow_recent_growth", "e_grow_features"])
        NotGO = R.union("LegacyNoGrowthOverrideCore", NotRecentRising, g3_le, f2_le30)
        no_large = R.named("LegacyNoOverrideLarge", SP & n2 & n4 & NotDO & NotGO)
        no_small = R.named("LegacyNoOverrideSmall", SP & n2 & n_lt4)
        NoOverride = R.union("LegacyNoOverride", no_large, no_small)

        # ---------------- final stage classes ----------------
        growth_maj = R.named("LegacyGrowthByMajority", NoOverride & MG)
        growth_tie = R.named("LegacyGrowthByLocTieBreak", NoOverride & NM & LG)
        decl_maj = R.named("LegacyDeclineByMajority", NoOverride & MD)
        decl_tie = R.named("LegacyDeclineByLocTieBreak", NoOverride & NM & LD)
        stab_maj = R.named("LegacyStabilizationByMajority", NoOverride & MS_)
        stab_tie = R.named("LegacyStabilizationByLocTieBreak", NoOverride & NM & LS)
        LGS = R.union("LegacyGrowthStage", GO, growth_maj, growth_tie,
                      text="legacy ensemble: growth", thr_ids=ids_min)
        LDS = R.union("LegacyDeclineStage", DO, decl_maj, decl_tie,
                      text="legacy ensemble: decline", thr_ids=ids_min)
        LSS = R.union("LegacyStabilizationStage", stab_maj, stab_tie,
                      text="legacy ensemble: stabilization", thr_ids=ids_min)
        R.disjoint(LGS, LDS, LSS)

        # ---------------- paper-table laws ----------------
        pg = A("monthsObserved", ">=", "t_law1_min_months")

        def plaw(roman: str, conf: Any, nc: Any, ct: str, nt: str, ids: list[str]) -> None:
            R._annotate(conf, ct, ids)
            R._annotate(nc, nt, ids)
            R.disjoint(conf, nc)

        def pmk(roman: str, conf_expr: Any, nc_expr: Any, ct: str, nt: str, ids: list[str]) -> None:
            plaw(roman, R.named(f"PaperLaw{roman}Conformant", conf_expr),
                 R.named(f"PaperLaw{roman}NonConformant", nc_expr), ct, nt, ids)

        # I
        m_ge, m_lt = A("meanMonthlyCommits", ">=", "t_law1_mean_commits"), Ac("meanMonthlyCommits", ">=", "t_law1_mean_commits")
        l3_gt, l3_le = A("commitsLast3Months", ">", "t_law1_recent_commits"), Ac("commitsLast3Months", ">", "t_law1_recent_commits")
        pmk("I", SP & pg & m_ge & l3_gt,
            R.union("PaperLawINCAny", R.named("PaperLawINCMean", SP & pg & m_lt),
                    R.named("PaperLawINCRecent", SP & pg & l3_le)),
            "mean {meanMonthlyCommits} commits/month and {commitsLast3Months} in the last 3 months",
            "mean {meanMonthlyCommits} commits/month or {commitsLast3Months} in the last 3 months too low",
            ["t_law1_min_months", "t_law1_mean_commits", "t_law1_recent_commits"])
        # II: complexity change (last - first, a difference of two measurements)
        pmk("II", SP & pg & A("complexityChange", ">=", "-t_law2_tolerance"),
            SP & pg & Ac("complexityChange", ">=", "-t_law2_tolerance"),
            "complexity change {complexityChange} >= -tolerance",
            "complexity change {complexityChange} < -tolerance", ["t_law1_min_months", "t_law2_tolerance"])
        # III
        grow = A("locGrowthRate", ">", "t_law3_loc_up")
        shrink = A("locGrowthRate", "<", "t_law3_loc_down")
        stable = R.between("locGrowthRate", ">=", "t_law3_loc_down", "<=", "t_law3_loc_up")
        prop_ok = A("complexityToLocGrowthRatio", "<=", "t_law3_cx_factor")   # only used where LOC growth > 0
        runaway = Ac("complexityToLocGrowthRatio", "<=", "t_law3_cx_factor")
        shr_ok, shr_bad = A("complexityGrowthRate", "<=", "t_law3_cx_shrink_max"), Ac("complexityGrowthRate", "<=", "t_law3_cx_shrink_max")
        stab_ok = R.between("complexityGrowthRate", ">", "-t_law3_cx_stable_abs", "<", "t_law3_cx_stable_abs")
        stab_bad = R.union("PaperCxDriftLarge", A("complexityGrowthRate", "<=", "-t_law3_cx_stable_abs"),
                           A("complexityGrowthRate", ">=", "t_law3_cx_stable_abs"))
        cvu, cvu_bad = A("commitCountCVSample", "<", "t_law3_cv_up"), Ac("commitCountCVSample", "<", "t_law3_cv_up")
        cvd, cvd_bad = A("commitCountCVSample", "<", "t_law3_cv_down"), Ac("commitCountCVSample", "<", "t_law3_cv_down")
        cvs, cvs_bad = A("commitCountCVSample", "<", "t_law3_cv_stable"), Ac("commitCountCVSample", "<", "t_law3_cv_stable")
        ok3 = R.union("PaperLawIIIOk",
                      R.named("PaperLawIIIGrowOk", SP & pg & grow & prop_ok & cvu),
                      R.named("PaperLawIIIShrinkOk", SP & pg & shrink & shr_ok & cvd),
                      R.named("PaperLawIIIStableOk", SP & pg & stable & stab_ok & cvs))
        bad3 = R.union("PaperLawIIIBad",
                       R.named("PaperLawIIIGrowRunaway", SP & pg & grow & runaway),
                       R.named("PaperLawIIIGrowCV", SP & pg & grow & cvu_bad),
                       R.named("PaperLawIIIShrinkCx", SP & pg & shrink & shr_bad),
                       R.named("PaperLawIIIShrinkCV", SP & pg & shrink & cvd_bad),
                       R.named("PaperLawIIIStableCx", SP & pg & stable & stab_bad),
                       R.named("PaperLawIIIStableCV", SP & pg & stable & cvs_bad))
        ids3 = ["t_law3_loc_up", "t_law3_loc_down", "t_law3_cx_factor", "t_law3_cx_shrink_max",
                "t_law3_cx_stable_abs", "t_law3_cv_up", "t_law3_cv_down", "t_law3_cv_stable"]
        plaw("III", R.named("PaperLawIIIConformant", ok3), R.named("PaperLawIIINonConformant", bad3),
             "LOC growth {locGrowthRate}, complexity growth {complexityGrowthRate}, commit CV {commitCountCVSample} within the branch limits",
             "LOC growth {locGrowthRate}, complexity growth {complexityGrowthRate}, commit CV {commitCountCVSample} outside the branch limits", ids3)
        # IV
        pmk("IV", SP & pg & A("commitCountCVSample", "<", "t_law4_cv"),
            SP & pg & Ac("commitCountCVSample", "<", "t_law4_cv"),
            f"commit CV {{commitCountCVSample}} < {R.val('t_law4_cv')}",
            f"commit CV {{commitCountCVSample}} >= {R.val('t_law4_cv')}",
            ["t_law1_min_months", "t_law4_cv"])
        # V (per-month closure)
        break_month = A("locChangeRatio", ">=", "t_law5_max_loc_change")
        familiar = R.union("FamiliarOrFirstMonth", A("locChangeRatio", "<", "t_law5_max_loc_change"), FirstMonth)
        cx5, cx5_bad = A("monthlyComplexityGrowth", "<", "t_law5_cx_growth"), Ac("monthlyComplexityGrowth", "<", "t_law5_cx_growth")
        plaw("V",
             R.named("PaperLawVConformant", SP & pg & hasMonth.only(familiar) & hasMonth.some(MS) & cx5),
             R.named("PaperLawVNonConformant", R.union(
                 "PaperLawVBad", R.named("PaperLawVBreak", SP & hasMonth.some(break_month)),
                 R.named("PaperLawVComplexity", SP & pg & cx5_bad))),
             f"every month's LOC change < {R.val('t_law5_max_loc_change')} and complexity growth {{monthlyComplexityGrowth}} < threshold",
             "a month's LOC change reached the limit, or complexity growth {monthlyComplexityGrowth} too high",
             ["t_law1_min_months", "t_law5_max_loc_change", "t_law5_cx_growth"])
        # VI: LOC_last > factor * LOC_first as a ratio; LOC_first = 0 is a separate case
        start_pos, start_zero = A("locStart", ">=", 1), A("locStart", "<=", 0)
        grew = R.union("PaperLocGrew",
                       R.named("PaperLocGrewRatio", SP & start_pos & A("locEndToStartRatio", ">", "t_law6_loc_factor")),
                       R.named("PaperLocGrewFromZero", SP & start_zero & A("locEnd", ">", 0)),
                       text="LOC {locStart} -> {locEnd} grew by more than the factor", thr_ids=["t_law6_loc_factor"])
        not_grew = R.union("PaperLocNotGrew",
                           R.named("PaperLocNotGrewRatio", SP & start_pos & Ac("locEndToStartRatio", ">", "t_law6_loc_factor")),
                           R.named("PaperLocNotGrewFromZero", SP & start_zero & A("locEnd", "<=", 0)),
                           text="LOC {locStart} -> {locEnd} did not grow by the factor", thr_ids=["t_law6_loc_factor"])
        f6, f6_bad = A("featureCommitsSecondHalf", ">", "t_law6_features"), Ac("featureCommitsSecondHalf", ">", "t_law6_features")
        plaw("VI", R.named("PaperLawVIConformant", SP & pg & grew & f6),
             R.named("PaperLawVINonConformant", R.union(
                 "PaperLawVIBad", R.named("PaperLawVIBadLoc", SP & pg & not_grew),
                 R.named("PaperLawVIBadFeatures", SP & pg & f6_bad))),
             "LOC {locStart} -> {locEnd} grew and {featureCommitsSecondHalf} recent feature commits",
             "LOC {locStart} -> {locEnd} or {featureCommitsSecondHalf} recent feature commits too low",
             ["t_law1_min_months", "t_law6_loc_factor", "t_law6_features"])
        # VII: complexity did not decrease: change >= 0
        cx_nd = A("complexityChange", ">=", 0)
        cx_d = Ac("complexityChange", ">=", 0)
        ref_lo, ref_hi = A("refactoringRatio", "<", "t_law7_refactor_ratio"), Ac("refactoringRatio", "<", "t_law7_refactor_ratio")
        plaw("VII",
             R.named("PaperLawVIIConformant", R.union(
                 "PaperLawVIIOk", R.named("PaperLawVIIOkComplexity", SP & pg & cx_nd),
                 R.named("PaperLawVIIOkRefactoring", SP & pg & ref_lo))),
             R.named("PaperLawVIINonConformant", SP & pg & cx_d & ref_hi),
             "complexity did not decrease (change {complexityChange}) or refactoring ratio {refactoringRatio} is low",
             "complexity decreased and refactoring ratio {refactoringRatio} is high",
             ["t_law1_min_months", "t_law7_refactor_ratio"])
        # VIII (month stage, then counts, then a ratio of the counts)
        NonFirst = A("monthIndex", ">=", 1)
        Rising = R.named("RisingMonth", A("complexityDelta", ">", "t_law8_rising_delta"),
                         "complexity rose by {complexityDelta}", ["t_law8_rising_delta"])
        Corr5 = R.named("CorrectiveMonth5", NonFirst & A("correctiveCount", ">", "t_law8_corrective"),
                        "{correctiveCount} corrective commits", ["t_law8_corrective"])
        Corr3 = R.named("CorrectiveMonth3", A("correctiveCount", ">", "t_law8_responsive"),
                        "{correctiveCount} corrective commits", ["t_law8_responsive"])
        R.named("ResponsiveRisingMonth", Rising & nextMonth.some(Corr3),
                "rising month followed by a corrective month", ["t_law8_rising_delta", "t_law8_responsive"])
        g8 = A("monthsObserved", ">=", "t_law8_min_months")
        no_rising = A("risingMonthCount", "<=", 0)
        some_rising = A("risingMonthCount", ">=", 1)
        match8 = R.union(
            "PaperCorrectiveMatchesRising",
            R.named("PaperNoRisingMonths", SP & no_rising),
            R.named("PaperCorrectiveRatioOk", SP & some_rising & A("correctiveToRisingRatio", ">=", "t_law8_factor")),
            text="{correctiveMonthCount} corrective months vs {risingMonthCount} rising", thr_ids=["t_law8_factor"])
        mism8 = R.named("PaperCorrectiveBelowRising",
                        SP & some_rising & Ac("correctiveToRisingRatio", ">=", "t_law8_factor"),
                        "{correctiveMonthCount} corrective months vs {risingMonthCount} rising", ["t_law8_factor"])
        rec_hi, rec_lo = A("correctiveLast2Months", ">", "t_law8_recent"), Ac("correctiveLast2Months", ">", "t_law8_recent")
        fb_seen = R.union("PaperFeedbackSeen", A("responsiveMonthCount", ">", 0), no_rising)
        no_fb = R.named("PaperNoFeedback", A("responsiveMonthCount", "<=", 0) & some_rising)
        plaw("VIII",
             R.named("PaperLawVIIIConformant", SP & g8 & match8 & rec_hi & fb_seen),
             R.named("PaperLawVIIINonConformant", R.union(
                 "PaperLawVIIIBad", R.named("PaperLawVIIIBadMatch", SP & g8 & mism8),
                 R.named("PaperLawVIIIBadRecent", SP & g8 & rec_lo),
                 R.named("PaperLawVIIIBadFeedback", SP & g8 & no_fb))),
             "{correctiveMonthCount} corrective months vs {risingMonthCount} rising, "
             "{correctiveLast2Months} recent corrective commits, {responsiveMonthCount} responsive months",
             "corrective response to complexity rises is insufficient",
             ["t_law8_min_months", "t_law8_factor", "t_law8_recent", "t_law8_rising_delta",
              "t_law8_corrective", "t_law8_responsive"])

    _save(onto, "seo-legacy")
    return onto


# ============================================================================
# Module 8 — seo-full.owl (imports closure of the canonical modules)
# ============================================================================

def build_full_module(modules: list[owlready2.namespace.Ontology]) -> owlready2.namespace.Ontology:
    """Create seo-full.owl, importing every canonical module (Paper 1, WP1).

    populate.py's ``PopulationConfig.owl_file`` points here. seo-legacy is not
    imported on purpose: the legacy encodings must never label real repos.

    Args:
        modules: The canonical module ontologies, in dependency order.

    Returns:
        The full ontology.
    """
    onto = _onto(FULL_MODULE)
    for m in modules:
        onto.imported_ontologies.append(m)
    onto.metadata.comment.append(locstr(
        "Imports closure of the canonical SEO modules (entities .. inference). "
        "seo-legacy.owl is a regression artefact and is deliberately not imported.", "en"))
    _save(onto, FULL_MODULE)
    return onto


def _save(onto: owlready2.namespace.Ontology, name: str) -> None:
    path = _owl_path(name)
    onto.save(file=str(path), format="rdfxml")
    _canonicalize_rdfxml(path)
    logger.info("Saved %s", path)


def _canonicalize_rdfxml(path: Path) -> None:
    """Make the saved RDF/XML byte-stable across builds (the TBox sha256 is frozen by RQ1).

    Owlready2 writes the property elements of one description in an order that
    depends on hash-randomised insertion order (annotation values such as
    ``thresholdId`` swap between runs). RDF has no triple order, so the child
    elements of every top-level description are sorted as text units (a unit is
    a child element at indent 2 together with its nested lines, so ``rdf:List``
    order inside a unit is untouched). Top-level block order is already stable.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^<([\w:.-]+)(\s[^>]*)?>$", ln)
        if not m or ln.endswith("/>") or ln.startswith("<?") or ln.startswith("<rdf:RDF"):
            out.append(ln)
            i += 1
            continue
        close = f"</{m.group(1)}>"
        j = i + 1
        while j < len(lines) and lines[j] != close:
            j += 1
        units: list[list[str]] = []
        for child in lines[i + 1: j]:
            if (re.match(r"^  <", child) and not child.startswith("  </")) or not units:
                units.append([child])
            else:
                units[-1].append(child)
        units.sort(key=lambda u: "\n".join(u))
        out.append(ln)
        out.extend(c for u in units for c in u)
        out.append(close)
        i = j + 1
    path.write_text("\n".join(out), encoding="utf-8")


# ============================================================================
# Lint
# ============================================================================

def lint_ontology(rules: _Rules, out_dir: Path) -> list[str]:
    """Check the structural rules the explanation renderer needs (Paper 1, WP1).

    * every operand of a union and of a complement is a NAMED class;
    * no xsd:pattern facet appears in any generated file (crashes Pellet 2.3.1).

    Args:
        rules: The rule builder (for the list of created classes).
        out_dir: Directory with the generated .owl files.

    Returns:
        List of violation messages (empty when clean).
    """
    from owlready2 import And, Not, Or, Restriction

    problems: list[str] = []

    def walk(expr: Any, owner: str) -> None:
        if isinstance(expr, Or):
            for c in expr.Classes:
                if not isinstance(c, owlready2.ThingClass):
                    problems.append(f"{owner}: union operand {c!r} is not a named class")
                else:
                    continue
                walk(c, owner)
        elif isinstance(expr, Not):
            if not isinstance(expr.Class, owlready2.ThingClass):
                problems.append(f"{owner}: complement operand {expr.Class!r} is not a named class")
        elif isinstance(expr, And):
            for c in expr.Classes:
                if not isinstance(c, owlready2.ThingClass):
                    walk(c, owner)

    for cls in rules.created:
        for expr in list(cls.equivalent_to) + list(cls.is_a):
            walk(expr, cls.name)
    for path in out_dir.glob("seo-*.owl"):
        text = path.read_text(encoding="utf-8")
        if "XMLSchema#pattern" in text:
            problems.append(f"{path.name}: contains an xsd:pattern facet")
    return problems


# ============================================================================
# Main
# ============================================================================

def build_all(
    thresholds_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Build every SEO module and return their file paths (Paper 1, WP1).

    A fresh Owlready world is used per call, so WP4's sensitivity analysis
    can call this repeatedly with different threshold files and output
    directories.

    Args:
        thresholds_path: Alternative threshold register (default thresholds.yaml).
        out_dir: Output directory (default ontology/modules/).

    Returns:
        Dictionary mapping module names to their .owl file paths.

    Raises:
        RuntimeError: If the lint finds a structural violation.
    """
    logger.info("Building SENSO Evolution Ontology modules ...")
    _STATE.world = owlready2.World()
    _STATE.world.set_backend()
    _STATE.out_dir = Path(out_dir) if out_dir else OUTPUT_DIR
    _STATE.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        thr = load_thresholds(Path(thresholds_path) if thresholds_path else THRESHOLDS_FILE)
        entities = build_entities_module()
        processes = build_processes_module(entities)
        metrics = build_metrics_module(entities, processes)
        laws = build_laws_module(entities, processes, metrics)
        patterns = build_patterns_module(entities, processes, metrics, laws)
        world = _STATE.world
        props = {m[0]: world[SEO_NS_IRI + m[0]] for m in MEASURED_PROPERTIES}
        rules = _Rules(props, thr)
        inference = build_inference_module(entities, processes, metrics, laws, patterns, rules)
        evidence = build_evidence_module(entities, processes, metrics, rules)
        legacy_rules = rules.fork()
        build_legacy_module(entities, processes, metrics, legacy_rules)
        try:  # WP2 hook (design 4); also works when this file runs as a script
            from ontology.modules.claims_module import build_claims_module
        except ModuleNotFoundError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from ontology.modules.claims_module import build_claims_module
        build_claims_module(entities, rules)
        build_full_module([entities, processes, metrics, laws, patterns, inference, evidence])
        rules.created += legacy_rules.created
        rules.rules += legacy_rules.rules
        problems = lint_ontology(rules, _STATE.out_dir)
        if problems:
            for p in problems:
                logger.error("lint: %s", p)
            raise RuntimeError(f"{len(problems)} lint violation(s)")
        logger.info("Lint clean: %d defined/derived classes, %d SWRL rules.",
                    len(rules.created), len(rules.rules))
        return {name: _owl_path(name) for name in ALL_MODULE_NAMES}
    finally:
        _STATE.world = None
        _STATE.out_dir = OUTPUT_DIR


def verify_all(out_dir: Optional[Path] = None) -> bool:
    """Reload every .owl file in a fresh world to verify it parses (Paper 1).

    Args:
        out_dir: Directory holding the files (default ontology/modules/).

    Returns:
        True if all modules load without error.
    """
    logger.info("Verifying generated OWL files ...")
    base = Path(out_dir) if out_dir else OUTPUT_DIR
    world = owlready2.World()
    ok = True
    for name in ALL_MODULE_NAMES:
        path = base / f"{name}.owl"
        try:
            onto = world.get_ontology(path.as_uri()).load()
            logger.info("  OK %s: %d classes, %d properties",
                        name, len(list(onto.classes())), len(list(onto.properties())))
        except Exception:
            logger.exception("  FAILED to load %s", name)
            ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the SEO OWL modules (Paper 1, WP1)")
    ap.add_argument("--thresholds", type=Path, default=None, help="alternative thresholds.yaml")
    ap.add_argument("--out-dir", type=Path, default=None, help="output directory")
    cli = ap.parse_args()
    build_all(cli.thresholds, cli.out_dir)
    if not verify_all(cli.out_dir):
        logger.error("Verification failed for one or more modules.")
        sys.exit(1)
    logger.info("Done.")
