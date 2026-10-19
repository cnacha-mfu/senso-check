#!/usr/bin/env python3
"""Reasoning over the SEO TBox and a measured ABox: Pellet and HermiT (Paper 1, WP1).

Paper 1 - every label (stage, law conformance, anti-pattern, AgentCommit,
BugFixCommit) is inferred here, none is computed in Python. Both reasoners run
through Owlready2's bundled jars with the Java from the ``SENSO_JAVA``
environment variable (default: ``$JAVA_HOME/bin/java``, else ``java`` on PATH; tested with Temurin 25).

Which engine reasons over what, and why (WP1 findings):

  * seo-evidence.owl (commit strings, SWRL ``matches``/``contains``) needs SWRL
    built-ins: **Pellet** (``PelletReasoner``, stage A). HermiT rejects them.
  * seo-inference.owl (stages, laws, anti-patterns) and seo-legacy.owl are
    SWRL-free OWL 2 DL: **HermiT** (``HermitReasoner``, stage B). Pellet 2.3.1
    is the wrong tool there: on the datatype-heavy TBox it needs minutes for 40
    repositories (HermiT: seconds), crashes on the legacy module
    ("Adding type to a pruned node") and returns different, incomplete label
    sets on identical input across runs.

Pipeline units (design 3.4), per repository:

  * stage A, commit level: ``PelletReasoner.stage_a`` classifies commit
    individuals in chunks of 1000 (AgentCommit and BugFixCommit are local to a
    commit, so chunking is sound);
  * aggregation (counting, in Python, see ``measures.py``) over the inferred
    commit labels;
  * stage B, repo level: ``HermitReasoner.infer`` on the project ABox (without
    month individuals; only the legacy regression module uses them).

Inferred labels are returned as data (and as N-Triples of the *inferred*
ontology, separate from the asserted ABox). An inconsistent ABox is a
reportable outcome, captured in the result and never raised.

At most 2-3 reasoner JVMs may run concurrently on the shared VM; this module
does not spawn parallel work itself.

Usage:
    python ontology/reasoning/reasoner.py --demo
    python ontology/reasoning/reasoner.py --abox repo.nt              # Pellet, stage A
    python ontology/reasoning/reasoner.py --abox repo.nt --hermit     # HermiT, stage B
    python ontology/reasoning/reasoner.py --abox repo.nt --hermit --legacy
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import owlready2  # noqa: E402

logger = logging.getLogger("senso.reasoner")

MODULES_DIR = PROJECT_ROOT / "ontology" / "modules"
STAGE_A_MODULES = ["seo-entities", "seo-processes", "seo-metrics", "seo-evidence"]
"""Commit-level SWRL string rules: reasoned with Pellet (the only SWRL in the canonical modules)."""
STAGE_B_MODULES = ["seo-entities", "seo-processes", "seo-metrics", "seo-laws", "seo-patterns",
                   "seo-inference"]
"""Repo-level labels, SWRL-free OWL 2 DL: reasoned with HermiT."""
LEGACY_MODULE = "seo-legacy"
LEGACY_MODULES = ["seo-entities", "seo-processes", "seo-metrics", LEGACY_MODULE]
"""Regression artefact (never labels real repos): reasoned with HermiT."""
CLAIMS_MODULES = ["seo-entities", "seo-claims"]
"""WP2 manifest-claim verdicts, SWRL-free: reasoned with HermiT (HermitReasoner(modules=CLAIMS_MODULES))."""
LABEL_MODULES = ("seo-inference", "seo-evidence", "seo-legacy", "seo-claims")
SEO = "http://senso-framework.org/ontology/seo#"


def _strip_progress(msg: str) -> str:
    """Drop Pellet's classification progress bar from an error message."""
    return re.sub(r"Classifying:\s*\d+% complete in \d+:\d+", "", msg)


def default_java() -> str:
    """Java executable: ``SENSO_JAVA``, else ``$JAVA_HOME/bin/java``, else ``java`` on PATH."""
    if os.environ.get("SENSO_JAVA"):
        return os.path.expanduser(os.environ["SENSO_JAVA"])
    jh = os.environ.get("JAVA_HOME")
    return str(Path(jh) / "bin" / "java") if jh else "java"


@dataclass
class InferenceResult:
    """Outcome of one Pellet pass (Paper 1, WP1)."""

    labels: dict[str, list[str]] = field(default_factory=dict)
    """individual IRI -> sorted label class local names (atoms excluded)."""
    consistent: bool = True
    error: Optional[str] = None
    timings: dict[str, float] = field(default_factory=dict)
    abox_triples: int = 0
    abox_individuals: int = 0
    inferred_ntriples: str = ""
    merged_ntriples: str = ""
    """TBox + ABox, kept only when inconsistent (input for explain.py)."""
    tbox_sha256: str = ""

    def labels_of(self, iri: str) -> set[str]:
        """Label class names inferred for an individual."""
        return set(self.labels.get(iri, ()))

    def to_json(self) -> dict[str, Any]:
        """Serializable summary without the triple dumps."""
        return {
            "consistent": self.consistent, "error": self.error, "timings": self.timings,
            "abox_triples": self.abox_triples, "abox_individuals": self.abox_individuals,
            "n_labelled": len(self.labels), "tbox_sha256": self.tbox_sha256,
        }


@dataclass
class StageAResult:
    """Merged outcome of chunked commit-level reasoning."""

    labels: dict[str, list[str]] = field(default_factory=dict)
    chunk_timings: list[dict[str, float]] = field(default_factory=list)
    n_commits: int = 0
    inconsistent_chunks: int = 0

    def flagged(self, class_name: str) -> set[str]:
        """IRIs of individuals inferred to be in a class."""
        return {iri for iri, ls in self.labels.items() if class_name in ls}


class PelletReasoner:
    """Run Pellet (Owlready2 bundle) over TBox files plus an ABox (default modules: stage A)."""

    def __init__(
        self,
        tbox_dir: Path = MODULES_DIR,
        *,
        java: Optional[str] = None,
        temp_dir: Optional[Path] = None,
        memory_mb: int = 2000,
        chunk_size: int = 1000,
        modules: Optional[Sequence[str]] = None,
        engine: str = "pellet",
    ) -> None:
        """Configure the reasoner.

        Args:
            tbox_dir: Directory with the generated module files.
            java: Java executable (default: see ``default_java``).
            temp_dir: Directory for Owlready's temporary files.
            memory_mb: Pellet ``-Xmx`` in MB.
            chunk_size: Commit individuals per stage-A reasoning call.
            modules: Override the module list.
            engine: ``pellet`` or ``hermit``.
        """
        self.engine = engine
        self.tbox_dir = Path(tbox_dir).resolve()
        self.modules = list(modules or STAGE_A_MODULES)
        self.java = java or default_java()
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.memory_mb = memory_mb
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    def tbox_sha256(self) -> str:
        """SHA-256 over the module files in load order (provenance)."""
        h = hashlib.sha256()
        for name in self.modules:
            h.update((self.tbox_dir / f"{name}.owl").read_bytes())
        return h.hexdigest()

    def _configure(self) -> Optional[str]:
        owlready2.JAVA_EXE = self.java
        owlready2.reasoning.JAVA_MEMORY = self.memory_mb
        old = tempfile.tempdir
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(self.temp_dir)
        return old

    def load_world(self) -> tuple[Any, dict[str, Any]]:
        """Load the TBox into a fresh in-memory Owlready world."""
        world = owlready2.World()
        world.set_backend()
        onts: dict[str, Any] = {}
        for name in self.modules:
            path = self.tbox_dir / f"{name}.owl"
            onts[name] = world.get_ontology(path.as_uri()).load()
        return world, onts

    # ------------------------------------------------------------------
    def infer(
        self,
        abox: str | bytes | Path,
        *,
        include_atoms: bool = False,
        keep_merged_on_inconsistency: bool = True,
    ) -> InferenceResult:
        """One Pellet pass over the TBox plus an N-Triples ABox.

        Args:
            abox: N-Triples text, bytes or a path.
            include_atoms: Also report the ``A_*`` helper classes.
            keep_merged_on_inconsistency: Keep the merged dump for explain.py.

        Returns:
            InferenceResult; inconsistency is captured in ``consistent``/``error``.
        """
        data = Path(abox).read_bytes() if isinstance(abox, Path) else (
            abox.encode("utf-8") if isinstance(abox, str) else abox)
        res = InferenceResult(tbox_sha256=self.tbox_sha256())
        res.abox_triples = sum(1 for ln in data.splitlines() if ln.strip())
        old_tmp = self._configure()
        t_all = time.perf_counter()
        try:
            t0 = time.perf_counter()
            world, onts = self.load_world()
            tbox_individuals = {i.iri for i in world.individuals()}
            res.timings["tbox_load_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            abox_iri = f"urn:senso:abox:{uuid.uuid4().hex}#"
            abox_onto = world.get_ontology(abox_iri)
            abox_onto.load(fileobj=io.BytesIO(data), format="ntriples")
            abox_individuals = [i for i in world.individuals() if i.iri not in tbox_individuals]
            res.abox_individuals = len(abox_individuals)
            res.timings["abox_load_s"] = time.perf_counter() - t0

            inferred = world.get_ontology(f"urn:senso:inferred:{uuid.uuid4().hex}#")
            t0 = time.perf_counter()
            try:
                with inferred:
                    if self.engine == "hermit":
                        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
                    else:
                        owlready2.sync_reasoner_pellet(world, infer_property_values=False, debug=0)
            except owlready2.OwlReadyInconsistentOntologyError as exc:
                res.consistent = False
                res.error = str(exc)[:2000]
                res.timings["reason_s"] = time.perf_counter() - t0
                if keep_merged_on_inconsistency:
                    res.merged_ntriples = self.merged_ntriples(world)
                return res
            except owlready2.OwlReadyJavaError as exc:
                res.consistent = False
                res.error = "java error: " + _strip_progress(str(exc))[:3000]
                res.timings["reason_s"] = time.perf_counter() - t0
                return res
            res.timings["reason_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            label_iris = self._label_class_iris(onts, include_atoms)
            for ind in abox_individuals:
                names = sorted({
                    c.name for c in ind.INDIRECT_is_a
                    if isinstance(c, owlready2.ThingClass) and c.iri in label_iris
                })
                if names:
                    res.labels[ind.iri] = names
            buf = io.BytesIO()
            inferred.save(buf, format="ntriples")
            res.inferred_ntriples = buf.getvalue().decode("utf-8")
            res.timings["extract_s"] = time.perf_counter() - t0
            return res
        finally:
            res.timings["total_s"] = time.perf_counter() - t_all
            tempfile.tempdir = old_tmp

    @staticmethod
    def _label_class_iris(onts: Mapping[str, Any], include_atoms: bool) -> set[str]:
        out: set[str] = {SEO + "AgentCommit"}  # declared in seo-entities, defined in seo-inference
        for name in LABEL_MODULES:
            onto = onts.get(name)
            if onto is None:
                continue
            for c in onto.classes():
                if include_atoms or not c.name.startswith("A_"):
                    out.add(c.iri)
        return out

    @staticmethod
    def merged_ntriples(world: Any) -> str:
        """Serialize the whole world (TBox + ABox) as N-Triples, as Pellet saw it."""
        buf = io.BytesIO()
        python_name = world._abbreviate(
            "http://www.lesfleursdunormal.fr/static/_downloads/owlready_ontology.owl#python_name")
        world.save(buf, format="ntriples",
                   filter=lambda g, s, p, o, d: p != python_name)
        return buf.getvalue().decode("utf-8")

    # ------------------------------------------------------------------
    def stage_a(self, commits: Sequence[Mapping[str, Any]]) -> StageAResult:
        """Commit-level reasoning in chunks (AgentCommit, BugFixCommit, ...).

        Args:
            commits: Dicts with ``id`` (sha) and evidence properties
                (``measures.extract_commit_evidence``).

        Returns:
            Merged labels and per-chunk timings.
        """
        from ontology.reasoning.abox import AboxWriter

        out = StageAResult(n_commits=len(commits))
        for i in range(0, len(commits), self.chunk_size):
            chunk = commits[i: i + self.chunk_size]
            w = AboxWriter()
            for c in chunk:
                w.commit(c["id"], {k: v for k, v in c.items() if k != "id"})
            r = self.infer(w.ntriples())
            if not r.consistent:
                out.inconsistent_chunks += 1
                logger.error("stage A chunk %d inconsistent: %s", i // self.chunk_size, r.error)
            out.labels.update(r.labels)
            out.chunk_timings.append({**r.timings, "commits": len(chunk)})
        return out

    def jvm_baseline(self) -> float:
        """Reasoner wall time on an empty ABox (JVM start-up plus TBox classification)."""
        return self.infer("").timings["reason_s"]


class HermitReasoner(PelletReasoner):
    """HermiT over the SWRL-free modules (stage B and the legacy regression module).

    HermiT is sound and complete for OWL 2 DL and deterministic. Pellet 2.3.1 is
    not: on the datatype-heavy stage-B TBox it takes minutes and returns
    different, incomplete label sets on identical input (WP1 finding), so
    repo-level labels are computed with HermiT. HermiT rejects SWRL built-ins,
    hence it cannot load seo-evidence.
    """

    def __init__(self, tbox_dir: Path = MODULES_DIR, *, legacy: bool = False, **kw: Any) -> None:
        """Args: legacy: reason over seo-legacy instead of seo-inference."""
        kw.setdefault("modules", LEGACY_MODULES if legacy else STAGE_B_MODULES)
        super().__init__(tbox_dir, engine="hermit", **kw)


def count_month_labels(
    labels: Mapping[str, Sequence[str]],
    month_owner: Mapping[str, str],
    class_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Count inferred month classes per project (post month-stage aggregate).

    Counting is measurement. The results feed ``risingMonthCount`` etc. in a
    second pass (paper-table Law VIII, seo-legacy).

    Args:
        labels: ``InferenceResult.labels`` of the month-stage pass.
        month_owner: ``AboxWriter.month_owner``.
        class_names: Month classes to count.

    Returns:
        project IRI -> class name -> count.
    """
    out: dict[str, dict[str, int]] = {}
    for month, proj in month_owner.items():
        d = out.setdefault(proj, {c: 0 for c in class_names})
        for c in class_names:
            if c in labels.get(month, ()):
                d[c] += 1
    return out


def _demo() -> None:
    from ontology.population.measures import extract_commit_evidence, month_snapshots, project_measurements
    from ontology.reasoning.abox import AboxWriter

    series = [{"month": f"2025-{i + 1:02d}", "commit_count": 5 + i, "loc": 100 + 25 * i,
               "complexity_mean": 2.0 + 0.1 * i} for i in range(12)]
    w = AboxWriter()
    iri = w.project("demo/repo", project_measurements(series), month_snapshots(series))
    c = w.commit("abc", extract_commit_evidence(
        author_name="dependabot[bot]", author_email="e", message="Fix crash\n\nCo-authored-by: Claude <noreply@anthropic.com>"))
    ws = AboxWriter()
    ws.project("demo/repo", project_measurements(series), month_snapshots(series), close_months=False)
    rb = HermitReasoner().infer(ws.ntriples())            # stage B: repo-level labels
    ra = PelletReasoner().infer(w.ntriples())             # stage A: commit-level labels
    print(json.dumps({"stage_b": rb.to_json(), "stage_a": ra.to_json()}, indent=1))
    print(iri, sorted(rb.labels_of(ws.base + "demo_repo")))
    print(c, sorted(ra.labels_of(c)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pellet reasoning for SEO (Paper 1, WP1)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--abox", type=Path, help="N-Triples ABox to reason over")
    ap.add_argument("--hermit", action="store_true", help="HermiT over stage-B (or legacy) instead of Pellet over stage A")
    ap.add_argument("--legacy", action="store_true", help="with --hermit: the legacy module instead of seo-inference")
    ap.add_argument("--temp-dir", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if args.demo:
        _demo()
    elif args.abox:
        cls = HermitReasoner if args.hermit else PelletReasoner
        res = (cls(legacy=args.legacy, temp_dir=args.temp_dir) if args.hermit
               else cls(temp_dir=args.temp_dir)).infer(args.abox)
        print(json.dumps({**res.to_json(), "labels": res.labels}, indent=1))
    else:
        ap.print_help()
