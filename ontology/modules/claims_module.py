"""seo-claims.owl: manifest claims and their consistency verdicts (Paper 1, WP2 / RQ2).

Design: papers/paper1/wp2-design.md sections 2-4. The module is SWRL-free OWL 2 DL and
imports ``seo-entities`` only (it must not import ``seo-inference``, so the WP1 labels
and their timings are unchanged). Python measures facts about a repository snapshot
(counts, deltas, strings) and asserts them on claim individuals; every verdict class,
threshold and consensus rule lives here, and HermiT decides which verdict a claim is in.

Verdict lattice: ``SupportedClaim``, ``ContradictedClaim``, ``UnverifiableClaim``,
``IndeterminateClaim``, ``ConflictedClaim`` are pairwise disjoint (``AllDisjoint``); each
is the Horn union of named leaf classes (``LEAF_VERDICT`` maps leaf -> verdict). A claim
that gets no measurement falls into none of them ("unmeasured"), never into
``ContradictedClaim`` (open-world rule). Chained classes: ``ManifestWithContradiction``,
``DriftedClaim``, ``BornWrongClaim``.

Usage:
    imported by ontology/modules/build_ontology.py (``build_all`` calls ``build_claims_module``)
    python wp2_consistency.py selftest
"""

from __future__ import annotations

import types
from typing import Any

import owlready2
from owlready2 import DataProperty, FunctionalProperty, ObjectProperty, Thing, locstr

CLAIMS_MODULE_NAME = "seo-claims"

# (name, kind, domain, unit, computed_by, comment). kind: int | dec | str | strs.
# domain: CL = ManifestClaim, E = ClaimEvidence, D = ManifestDocument.
CLAIM_PROPERTIES: list[tuple[str, str, str, str, str, str]] = [
    # common
    ("claimQuote", "str", "CL", "text", "wp2.consensus", "Verbatim quote from the manifest (context)."),
    ("claimSubject", "str", "CL", "text", "wp2.normalize", "Normalized subject of the claim (context)."),
    ("claimKey", "str", "CL", "text", "wp2.normalize", "Canonical claim key (type, normalized subject, ...)."),
    ("claimLine", "int", "CL", "line", "wp2.consensus", "Global line number of the quote in the manifest."),
    ("extractionVotes", "int", "CL", "models", "wp2.consensus", "Number of extraction models (1 to 3) that proposed this claim key."),
    ("snapshotCommit", "str", "D", "sha", "wp2.snapshot", "Commit S at which the manifest and repository are examined."),
    ("manifestPath", "str", "D", "path", "wp2.snapshot", "Repository path of the manifest."),
    # T1 path existence
    ("pathExactMatchCount", "int", "CL", "paths", "wp2.measure.path_counts", "Tracked files equal to the path, under the directory, or matching the glob at S."),
    ("pathSuffixMatchCount", "int", "CL", "paths", "wp2.measure.path_counts", "Tracked paths that end with '/' + path (monorepo-relative mentions)."),
    ("pathIgnoredCount", "int", "CL", "0/1", "wp2.measure.path_counts", "1 if a .gitignore at S matches the path, else 0."),
    ("pathSubmoduleCount", "int", "CL", "0/1", "wp2.measure.path_counts", "1 if a gitlink (mode 160000) is a prefix of the path, else 0."),
    ("pathEverTrackedCount", "int", "CL", "commits", "wp2.measure.path_counts", "Commits up to S that touched the path."),
    ("daysSincePathLastTracked", "int", "CL", "days", "wp2.measure.path_counts", "Days between the last commit touching the path and S (context)."),
    # T2 path status
    ("pathCommitsIn90d", "int", "CL", "commits", "wp2.measure.path_churn", "Non-merge first-parent commits touching the path in the 90 days before S."),
    ("pathCommitsIn180d", "int", "CL", "commits", "wp2.measure.path_churn", "Same for 180 days."),
    ("pathCommitsIn365d", "int", "CL", "commits", "wp2.measure.path_churn", "Same for 365 days."),
    ("pathLinesChangedIn90d", "int", "CL", "lines", "wp2.measure.path_churn", "Lines added plus deleted inside the path, 90 days before S."),
    ("pathLinesChangedIn180d", "int", "CL", "lines", "wp2.measure.path_churn", "Same for 180 days."),
    ("pathLinesChangedIn365d", "int", "CL", "lines", "wp2.measure.path_churn", "Same for 365 days."),
    ("repoCommitsIn180d", "int", "CL", "commits", "wp2.measure.path_churn", "Repository commits in the 180 days before S (context)."),
    ("daysSincePathLastChange", "int", "CL", "days", "wp2.measure.path_churn", "Days since the path last changed before S (context)."),
    # T3 tool usage
    ("toolSignatureHitCount", "int", "CL", "signatures", "wp2.measure.tool_hits", "Tracked files, manifest keys and declared dependencies that match the tool's signature at S."),
    ("toolEcosystemAnchorCount", "int", "CL", "files", "wp2.measure.tool_hits", "Tracked ecosystem anchor files (package.json, pyproject.toml, ...) at S."),
    ("competingToolSignatureCount", "int", "CL", "signatures", "wp2.measure.tool_hits", "Signature hits of competing tools of the same category and ecosystem."),
    # T4 dependency
    ("dependencyExactMatchCount", "int", "CL", "declarations", "wp2.depparse", "Declarations of the dependency (normalized name) across tracked manifests at S."),
    ("dependencySuffixMatchCount", "int", "CL", "declarations", "wp2.depparse", "Declarations whose name matches after the last '/'."),
    ("dependencyManifestParsedCount", "int", "CL", "files", "wp2.depparse", "Dependency manifests parsed at S."),
    ("claimedMajor", "int", "CL", "major", "wp2.depparse", "Major version written in the claim."),
    ("declaredLowerMajorDelta", "int", "CL", "major", "wp2.depparse", "Lowest declared lower-bound major minus claimedMajor."),
    ("declaredUpperMajorDelta", "int", "CL", "major", "wp2.depparse", "Highest declared upper-bound major minus claimedMajor; asserted only if bounded."),
    ("declaredUpperUnboundedCount", "int", "CL", "0/1", "wp2.depparse", "1 if some declaration has no upper bound."),
    # T5 command
    ("scriptDefinedRootCount", "int", "CL", "definitions", "wp2.measure.script_counts", "Definitions of the script/target in root package.json, composer.json or Makefile."),
    ("scriptDefinedAnyCount", "int", "CL", "definitions", "wp2.measure.script_counts", "Definitions anywhere in tracked package.json, composer.json, Makefile and *.mk files."),
    ("scriptHostFileCount", "int", "CL", "files", "wp2.measure.script_counts", "Tracked files that could define the script (package.json / composer.json / Makefile ...)."),
    ("dynamicTargetRuleCount", "int", "CL", "rules", "wp2.measure.script_counts", "Make pattern rules or targets containing $(...)."),
    ("ciInvocationCount", "int", "CL", "files", "wp2.measure.script_counts", "CI configs at S invoking the script (context, triangulation)."),
    # T6 version
    ("versionSourceCount", "int", "CL", "sources", "wp2.measure.version_evidence", "Parseable version sources for the claim's subject at S."),
    ("versionLowerDelta", "int", "E", "version code", "wp2.measure.version_evidence", "Declared lower bound minus claimed version, truncated to the claim's precision."),
    ("versionUpperDelta", "int", "E", "version code", "wp2.measure.version_evidence", "Declared upper bound minus claimed version; asserted only if bounded."),
    ("versionUpperUnboundedCount", "int", "E", "0/1", "wp2.measure.version_evidence", "1 if the source has no upper bound."),
    ("versionSourcePath", "str", "E", "path", "wp2.measure.version_evidence", "Source file (context)."),
    ("versionSourceRaw", "str", "E", "text", "wp2.measure.version_evidence", "Raw declared version (context)."),
    # T7 import rule
    ("importFilesScanned", "int", "CL", "files", "wp2.measure.import_edges", "Tracked source files scanned inside the claim's scope."),
    ("forbiddenImportEdgeCount", "int", "CL", "edges", "wp2.measure.import_edges", "Import statements that reference the forbidden target."),
]

# Leaf class -> verdict. Filled by build_claims_module (module-level so reports can read it).
LEAF_VERDICT: dict[str, str] = {}
# Leaf class -> claim type code (T1..T7)
LEAF_TYPE: dict[str, str] = {}
VERDICTS = ["SupportedClaim", "ContradictedClaim", "UnverifiableClaim", "IndeterminateClaim", "ConflictedClaim"]
CHAINED = ["ManifestWithContradiction", "DriftedClaim", "BornWrongClaim"]
WINDOWS = (90, 180, 365)


def claim_kinds() -> dict[str, str]:
    """``{property: kind}`` for the claim properties (merged into ``abox.property_kinds``)."""
    return {m[0]: m[1] for m in CLAIM_PROPERTIES}


def _declare_properties(ent_ns_classes: dict[str, Any]) -> dict[str, Any]:
    """Declare CLAIM_PROPERTIES and the object properties in the current namespace."""
    import datetime  # noqa: F401  (kept for parity with the measured-property declarer)

    domains = {"CL": ent_ns_classes["ManifestClaim"], "E": ent_ns_classes["ClaimEvidence"],
               "D": ent_ns_classes["ManifestDocument"]}
    ranges = {"int": int, "dec": float, "str": str, "strs": str}
    props: dict[str, Any] = {}
    for name, kind, dom, unit, by, comment in CLAIM_PROPERTIES:
        bases = (DataProperty,) if kind == "strs" else (DataProperty, FunctionalProperty)
        prop = types.new_class(name, bases)
        prop.domain = [domains[dom]]
        prop.range = [ranges[kind]]
        prop.comment = [locstr(f"{comment} Unit: {unit}. Computed by: {by}.", "en")]
        prop.computedBy = [by]
        prop.measurementUnit = [unit]
        props[name] = prop
    return props


def build_claims_module(entities_onto: owlready2.namespace.Ontology, rules: Any) -> owlready2.namespace.Ontology:
    """Create and save seo-claims.owl (Paper 1, WP2).

    Called from ``build_all`` after the WP1 modules. Forks ``rules`` (own atom cache, shared
    class-name registry and thresholds) with ``props``/``kinds`` extended by CLAIM_PROPERTIES,
    and appends the classes it creates to ``rules.created`` so the lint covers them.

    Args:
        entities_onto: The seo-entities ontology (the only import).
        rules: The canonical ``_Rules`` builder.

    Returns:
        The claims ontology.
    """
    import sys

    bo = sys.modules[type(rules).__module__]  # the builder module actually running (script or package)
    _ns, _onto, _save = bo._ns, bo._onto, bo._save

    LEAF_VERDICT.clear()
    LEAF_TYPE.clear()
    onto = _onto(CLAIMS_MODULE_NAME)
    onto.imported_ontologies.append(entities_onto)
    R = rules.fork()
    R.kinds = {**R.kinds, **claim_kinds()}
    thr = R.thr
    window = int(thr["wp2_status_window_days"]["value"])
    if window not in WINDOWS:
        raise ValueError(f"wp2_status_window_days must be one of {WINDOWS}, got {window}")

    with _ns(onto):
        # ---- core classes ----
        Doc = R._new("ManifestDocument")
        Claim = R._new("ManifestClaim")
        Evid = R._new("ClaimEvidence")
        VEvid = R._new("VersionEvidence", Evid)
        R.props = {**R.props, **_declare_properties(
            {"ManifestClaim": Claim, "ClaimEvidence": Evid, "ManifestDocument": Doc})}
        makesClaim = types.new_class("makesClaim", (ObjectProperty,))
        makesClaim.domain, makesClaim.range = [Doc], [Claim]
        makesExcluded = types.new_class("makesExcludedClaim", (ObjectProperty,))
        makesExcluded.domain, makesExcluded.range = [Doc], [Claim]
        hasEvidence = types.new_class("hasEvidence", (ObjectProperty,))
        hasEvidence.domain, hasEvidence.range = [Claim], [Evid]
        evidenceFor = types.new_class("evidenceFor", (ObjectProperty,))
        evidenceFor.domain, evidenceFor.range = [Evid], [Claim]
        evidenceFor.inverse_property = hasEvidence
        introducedAs = types.new_class("introducedAs", (ObjectProperty, FunctionalProperty))
        introducedAs.domain, introducedAs.range = [Claim], [Claim]

        # ---- asserted claim-type classes (extracted content, not verdicts) ----
        T1c = R._new("PathExistenceClaim", Claim)
        T2c = R._new("PathStatusClaim", Claim)
        InactiveC = R._new("InactiveStatusClaim", T2c)
        R._new("ProtectedStatusClaim", Claim)
        T3c = R._new("ToolUsageClaim", Claim)
        PkgMgrC = R._new("PackageManagerClaim", T3c)
        NonPkgC = R._new("NonPackageManagerToolClaim", T3c)
        R.disjoint(PkgMgrC, NonPkgC)
        T4c = R._new("DependencyClaim", Claim)
        UnverDep = R._new("UnversionedDependencyClaim", T4c)
        VerDep = R._new("VersionedDependencyClaim", T4c)
        R.disjoint(UnverDep, VerDep)
        T5c = R._new("CommandClaim", Claim)
        T6c = R._new("VersionClaim", Claim)
        MinV = R._new("MinVersionClaim", T6c)
        ExactV = R._new("ExactVersionClaim", T6c)
        R.disjoint(MinV, ExactV)
        T7c = R._new("ImportConstraintClaim", Claim)
        R.disjoint(T1c, T2c, T3c, T4c, T5c, T6c, T7c)

        A, Ac, B = R.atom, R.atom_c, R.between
        win = f"{window}d"
        p_commits, p_lines = f"pathCommitsIn{win}", f"pathLinesChangedIn{win}"

        # ---- consensus (TBox parameter) ----
        Checked = R.named("CheckedClaim", Claim & A("extractionVotes", ">=", "wp2_consensus_min_votes"),
                          "extracted by {extractionVotes} models (consensus threshold)", ["wp2_consensus_min_votes"])

        def leaf(name: str, verdict: str, typ: str, expr: Any, text: str, thr_ids: list[str] | None = None) -> Any:
            cls = R.named(name, expr, text, thr_ids or [])
            LEAF_VERDICT[name] = verdict
            LEAF_TYPE[name] = typ
            return cls

        S, C_, U, I, K = "SupportedClaim", "ContradictedClaim", "UnverifiableClaim", "IndeterminateClaim", "ConflictedClaim"

        # ============ T1 path existence ============
        ex0 = B("pathExactMatchCount", ">=", 0, "<=", 0)
        sf0 = B("pathSuffixMatchCount", ">=", 0, "<=", 0)
        ig0 = B("pathIgnoredCount", ">=", 0, "<=", 0)
        sm0 = B("pathSubmoduleCount", ">=", 0, "<=", 0)
        ever0 = B("pathEverTrackedCount", ">=", 0, "<=", 0)
        base1 = Checked & T1c
        leaf("PathFoundExact", S, "T1", base1 & A("pathExactMatchCount", ">=", 1),
             "{claimSubject} matches {pathExactMatchCount} tracked path(s) at the snapshot")
        leaf("PathFoundBySuffix", S, "T1", base1 & ex0 & A("pathSuffixMatchCount", ">=", 1),
             "{claimSubject} is not tracked at the root but {pathSuffixMatchCount} tracked path(s) end with it (relocated)")
        leaf("PathUntrackedByDesign", U, "T1", base1 & ex0 & sf0 & A("pathIgnoredCount", ">=", 1),
             "{claimSubject} is not tracked but is matched by a .gitignore at the snapshot")
        leaf("PathInSubmodule", U, "T1", base1 & ex0 & sf0 & A("pathSubmoduleCount", ">=", 1),
             "{claimSubject} lies inside a git submodule")
        leaf("PathRemoved", C_, "T1", base1 & ex0 & sf0 & ig0 & sm0 & A("pathEverTrackedCount", ">=", 1),
             "{claimSubject} is not tracked at the snapshot and not git-ignored; {pathEverTrackedCount} earlier commit(s) touched it")
        leaf("PathNeverExisted", C_, "T1", base1 & ex0 & sf0 & ig0 & sm0 & ever0,
             "{claimSubject} is not tracked at the snapshot, not git-ignored, and no commit up to the snapshot ever touched it")

        # ============ T2 path status ============
        base2 = Checked & InactiveC
        leaf("StatusPathUnresolved", U, "T2", base2 & ex0 & sf0,
             "{claimSubject} does not resolve to a tracked path, so churn is not measured")
        leaf("StatusPathAmbiguous", U, "T2", base2 & ex0 & A("pathSuffixMatchCount", ">=", 2),
             "{claimSubject} matches {pathSuffixMatchCount} tracked paths by suffix, so churn is not measured")
        leaf("InactivePathQuiet", S, "T2", base2 & A(p_commits, "<=", "wp2_inactive_support_max_commits"),
             "labelled inactive, and only {" + p_commits + "} commit(s) touched it in the window",
             ["wp2_inactive_support_max_commits", "wp2_status_window_days"])
        leaf("InactivePathActive", C_, "T2",
             base2 & A(p_commits, ">=", "wp2_inactive_contradict_min_commits")
             & A(p_lines, ">=", "wp2_inactive_contradict_min_lines"),
             "labelled inactive, yet {" + p_commits + "} commits and {" + p_lines + "} changed lines fell in the window",
             ["wp2_inactive_contradict_min_commits", "wp2_inactive_contradict_min_lines", "wp2_status_window_days"])
        not_active = R.union("InactiveNotActive", A(p_commits, "<", "wp2_inactive_contradict_min_commits"),
                             A(p_lines, "<", "wp2_inactive_contradict_min_lines"))
        leaf("InactivePathModerate", I, "T2",
             base2 & Ac(p_commits, "<=", "wp2_inactive_support_max_commits") & not_active,
             "labelled inactive; {" + p_commits + "} commit(s) in the window is between the quiet and active bands",
             ["wp2_inactive_support_max_commits", "wp2_inactive_contradict_min_commits",
              "wp2_inactive_contradict_min_lines", "wp2_status_window_days"])

        # ============ T3 tool usage ============
        base3 = Checked & T3c
        hits0 = B("toolSignatureHitCount", ">=", 0, "<=", 0)
        leaf("ToolSignatureFound", S, "T3", base3 & A("toolSignatureHitCount", ">=", 1),
             "{claimSubject} has {toolSignatureHitCount} matching signature(s) in the repository")
        leaf("ToolAbsentCompetitorPresent", C_, "T3",
             base3 & PkgMgrC & hits0 & A("competingToolSignatureCount", ">=", 1),
             "{claimSubject} has no signature while {competingToolSignatureCount} signature(s) of a competing package manager exist")
        no_strong = R.union("NoCompetingPackageManagerEvidence", NonPkgC, A("competingToolSignatureCount", "<=", 0))
        leaf("ToolAbsentInEcosystem", C_, "T3",
             base3 & hits0 & A("toolEcosystemAnchorCount", ">=", 1) & no_strong,
             "{claimSubject} has no signature although the ecosystem is present ({toolEcosystemAnchorCount} anchor file(s))")
        leaf("ToolEcosystemAbsent", U, "T3",
             base3 & hits0 & B("toolEcosystemAnchorCount", ">=", 0, "<=", 0),
             "{claimSubject} has no signature and the tool's ecosystem has no anchor file, so it cannot be checked")

        # ============ T4 dependency ============
        base4 = Checked & T4c
        d_exact = A("dependencyExactMatchCount", ">=", 1)
        d_exact0 = B("dependencyExactMatchCount", ">=", 0, "<=", 0)
        d_suf0 = B("dependencySuffixMatchCount", ">=", 0, "<=", 0)
        decl_exact = R.named("DependencyDeclaredExact", base4 & d_exact)
        decl_suffix = R.named("DependencyDeclaredBySuffix", base4 & d_exact0 & A("dependencySuffixMatchCount", ">=", 1))
        Declared = R.union("DependencyDeclared", decl_exact, decl_suffix)
        leaf("DependencyConfirmed", S, "T4", UnverDep & Declared,
             "{claimSubject} is declared ({dependencyExactMatchCount} declaration(s))")
        leaf("DependencyUndeclared", C_, "T4",
             base4 & d_exact0 & d_suf0 & A("dependencyManifestParsedCount", ">=", 1),
             "{claimSubject} is declared in none of the {dependencyManifestParsedCount} parsed manifests")
        leaf("DependencyNoManifest", U, "T4", base4 & B("dependencyManifestParsedCount", ">=", 0, "<=", 0),
             "no dependency manifest could be parsed at the snapshot")
        leaf("DependencyMajorTooOld", C_, "T4", VerDep & Declared & A("declaredLowerMajorDelta", ">", 0),
             "{claimSubject}: declared lower bound is {declaredLowerMajorDelta} major(s) above the claimed major")
        leaf("DependencyMajorTooNew", C_, "T4", VerDep & Declared & A("declaredUpperMajorDelta", "<", 0),
             "{claimSubject}: declared upper bound is {declaredUpperMajorDelta} major(s) below the claimed major")
        upper_ok = R.union("DependencyUpperCompatible", A("declaredUpperMajorDelta", ">=", 0),
                           A("declaredUpperUnboundedCount", ">=", 1))
        leaf("DependencyMajorInRange", S, "T4",
             VerDep & Declared & A("declaredLowerMajorDelta", "<=", 0) & upper_ok,
             "{claimSubject}: the claimed major lies within the declared range")

        # ============ T5 command ============
        base5 = Checked & T5c
        any0 = B("scriptDefinedAnyCount", ">=", 0, "<=", 0)
        leaf("ScriptDefinedRoot", S, "T5", base5 & A("scriptDefinedRootCount", ">=", 1),
             "{claimSubject} is defined {scriptDefinedRootCount} time(s) at the repository root")
        leaf("ScriptDefinedNested", S, "T5", base5 & B("scriptDefinedRootCount", ">=", 0, "<=", 0)
             & A("scriptDefinedAnyCount", ">=", 1),
             "{claimSubject} is defined only in nested files ({scriptDefinedAnyCount} definition(s))")
        leaf("ScriptUndefined", C_, "T5",
             base5 & any0 & A("scriptHostFileCount", ">=", 1) & B("dynamicTargetRuleCount", ">=", 0, "<=", 0),
             "{claimSubject} is defined in none of {scriptHostFileCount} script file(s)")
        leaf("ScriptHostMissing", C_, "T5", base5 & any0 & B("scriptHostFileCount", ">=", 0, "<=", 0),
             "the repository has no file that could define {claimSubject}")
        leaf("ScriptPossiblyDynamic", U, "T5", base5 & any0 & A("dynamicTargetRuleCount", ">=", 1),
             "{claimSubject} is undefined but {dynamicTargetRuleCount} dynamic make rule(s) could generate it")

        # ============ T6 version (evidence individuals) ============
        base6 = Checked & T6c
        low_eq = B("versionLowerDelta", ">=", 0, "<=", 0)
        upper_fine = R.union("VersionUpperCompatible", A("versionUpperDelta", ">=", 0),
                             A("versionUpperUnboundedCount", ">=", 1))
        min_agree = R.named("MinVersionAgreement", VEvid & evidenceFor.some(MinV) & low_eq,
                            "declared lower bound equals the claimed minimum")
        exact_agree = R.named("ExactVersionAgreement", VEvid & evidenceFor.some(ExactV)
                              & A("versionLowerDelta", "<=", 0) & upper_fine,
                              "claimed version lies within the declared range")
        Agreeing = R.union("AgreeingVersionEvidence", min_agree, exact_agree)
        min_dis_lo = R.named("MinVersionLowerThanClaimed", VEvid & evidenceFor.some(MinV) & A("versionLowerDelta", "<", 0),
                             "declared lower bound is below the claimed minimum")
        min_dis_hi = R.named("MinVersionHigherThanClaimed", VEvid & evidenceFor.some(MinV) & A("versionLowerDelta", ">", 0),
                             "declared lower bound is above the claimed minimum")
        ex_dis_lo = R.named("ExactVersionBelowRange", VEvid & evidenceFor.some(ExactV) & A("versionLowerDelta", ">", 0),
                            "claimed version is below the declared lower bound")
        ex_dis_hi = R.named("ExactVersionAboveRange", VEvid & evidenceFor.some(ExactV) & A("versionUpperDelta", "<", 0),
                            "claimed version is above the declared upper bound")
        Disagreeing = R.union("DisagreeingVersionEvidence", min_dis_lo, min_dis_hi, ex_dis_lo, ex_dis_hi)
        NonDis = R._new("NonDisagreeingVersionEvidence", VEvid)   # Horn target: Agreeing is a subclass
        Agreeing.is_a.append(NonDis)
        NonAgr = R._new("NonAgreeingVersionEvidence", VEvid)      # Horn target: Disagreeing is a subclass
        Disagreeing.is_a.append(NonAgr)
        leaf("VersionConfirmed", S, "T6", base6 & hasEvidence.some(Agreeing) & hasEvidence.only(NonDis),
             "every parseable source ({versionSourceCount}) agrees with the claimed version")
        leaf("VersionContradicted", C_, "T6", base6 & hasEvidence.some(Disagreeing) & hasEvidence.only(NonAgr),
             "every parseable source ({versionSourceCount}) disagrees with the claimed version")
        leaf("VersionSourcesConflict", K, "T6", base6 & hasEvidence.some(Agreeing) & hasEvidence.some(Disagreeing),
             "sources disagree with each other: some agree and some contradict the claimed version")
        leaf("NoVersionSource", U, "T6", base6 & B("versionSourceCount", ">=", 0, "<=", 0),
             "no parseable version source exists for the claim's subject at the snapshot")

        # ============ T7 import constraint (stretch) ============
        base7 = Checked & T7c
        leaf("ImportRuleViolated", C_, "T7", base7 & A("forbiddenImportEdgeCount", ">=", "wp2_import_violation_min_edges"),
             "{forbiddenImportEdgeCount} import edge(s) reference the forbidden target",
             ["wp2_import_violation_min_edges"])
        leaf("ImportRuleHeld", S, "T7", base7 & B("forbiddenImportEdgeCount", ">=", 0, "<=", 0)
             & A("importFilesScanned", ">=", 1), "no forbidden import in {importFilesScanned} scanned file(s)")
        leaf("ImportRuleModerate", I, "T7", base7 & A("forbiddenImportEdgeCount", ">", 0)
             & A("forbiddenImportEdgeCount", "<", "wp2_import_violation_min_edges"),
             "{forbiddenImportEdgeCount} forbidden import(s), below the violation threshold",
             ["wp2_import_violation_min_edges"])
        leaf("ImportScopeEmpty", U, "T7", base7 & B("importFilesScanned", ">=", 0, "<=", 0),
             "no source file was found in the claim's scope")

        # ============ verdict classes ============
        # classes by name from the created list
        by_name = {c.name: c for c in R.created}
        vcls = {}
        for v in VERDICTS:
            leaves = [by_name[n] for n, vv in LEAF_VERDICT.items() if vv == v]
            vcls[v] = R.union(v, *leaves, text=f"{v}: a claim in one of its leaf classes")
            vcls[v].is_a.append(Claim)
        R.disjoint(*[vcls[v] for v in VERDICTS])

        # ============ chained classes ============
        R.named("ManifestWithContradiction", Doc & makesClaim.some(vcls[C_]),
                "the manifest makes at least one contradicted claim")
        R.named("DriftedClaim", vcls[C_] & introducedAs.some(vcls[S]),
                "contradicted at the snapshot but supported when the claim text was introduced")
        R.named("BornWrongClaim", vcls[C_] & introducedAs.some(vcls[C_]),
                "contradicted both at the snapshot and when the claim text was introduced")

        onto.metadata.comment.append(locstr(
            "WP2 claim vocabulary and consistency verdicts. SWRL-free; imports seo-entities only.", "en"))
    rules.created += R.created
    _save(onto, CLAIMS_MODULE_NAME)
    return onto
