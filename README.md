# senso-check

Does your `CLAUDE.md` or `AGENTS.md` still match your repository?

`senso-check` reads the instruction file of a public GitHub repository, finds the factual claims in it, checks each
claim against the repository's git history, and tells you which claims are supported, contradicted or
unverifiable. For a contradicted claim it also says whether the claim **drifted** (it was true when it was written
and is false now) or was **born wrong** (it was never true), and for how many days it has been false.

This is the checker evaluated in the paper "SENSO: Checking Coding-Agent Manifests Against Repository History",
from senso-framework commit `ba8fbdd6eddc09dc21f5e6e7f4f88b1440a9ce5b`. Rules, thresholds, prompts, models, the
ontology and the claim normalisation (including the wildcard fix of senso-framework commit `0d546f5c`) are unchanged.
Only paths and packaging differ.

No human takes part in a check. Language models only **extract** claims from the text; they never decide whether a
claim is true. Verdicts come from measuring the repository and from an OWL 2 ontology that the HermiT reasoner
applies to those measurements.

## What it checks

| Type | Claim | Example |
|------|-------|---------|
| T1 | A file or directory path exists | ``The tests are in `tests/Unit`.`` |
| T2 | A path is inactive (legacy, deprecated, frozen) | "`old/` is deprecated" |
| T3 | A tool is used | "We lint with ruff" |
| T4 | A dependency is declared (optionally with a major version) | "Uses laravel/framework v12" |
| T5 | A script or build target exists | "Run `npm run build`" |
| T6 | A runtime or project version | "Requires Python 3.10+" |

Each claim gets one of five results: **supported**, **contradicted**, **unverifiable**, **indeterminate** or
**conflicted**. Only claims that at least two of the three extraction models find, as an assertion with a verbatim
quote from the manifest, receive a result. Claims without a result are hidden unless you pass `--all-claims`.

## Requirements

- Python 3.12 (3.11 or newer should work; 3.12 is what was tested).
- Java, for HermiT and Pellet. Set `JAVA_HOME`, or have `java` on your `PATH`. Tested with Temurin 25
  (OpenJDK 25.0.4). The jars bundled with owlready2 are old bytecode, so other recent JDKs probably work too;
  that is untested. `SENSO_JAVA` can point to a specific `java` executable.
- `git`, and network access to github.com and `integrate.api.nvidia.com`.
- Free NVIDIA NIM API keys (see below).

## Install

```bash
git clone https://github.com/cnacha-mfu/senso-check
cd senso-check
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Requirements are pinned to the versions used for the paper (owlready2 0.51, openai 3.24.0, pydantic 2.13.5, ...).

## Set up the NIM keys

All language-model calls go to NVIDIA NIM (`https://integrate.api.nvidia.com/v1`, free tier) through the `openai`
SDK. Get a key at https://build.nvidia.com. The extraction uses three models, each read from its own
environment variable:

| Model | Variable |
|-------|----------|
| `nvidia/nemotron-3-super-120b-a12b` | `NVIDIA_API_KEY_2` |
| `openai/gpt-oss-20b` | `NVIDIA_API_KEY_5` |
| `google/diffusiongemma-26b-a4b-it` | `NVIDIA_API_KEY_4` |

NIM keys belong to an account, not to a vendor, so one key works for all three: set the same value in all three
variables. A fourth variable, `NVIDIA_API_KEY`, is used only if a model has to be replaced because more than 10% of
its answers cannot be parsed (the fallback is `meta/llama-3.2-11b-vision-instruct`). The check stops with an error if
one of the three keys is missing.

`GITHUB_TOKEN` is optional. Without it the tool uses unauthenticated GitHub API calls (a few per run, 60 per hour
limit). With it, the rate limit is higher.

Put the variables in a file named `.env` **in the directory where you run the tool** (it is read from the current
directory; variables already in the environment take precedence). `.env` is in `.gitignore`.

```
NVIDIA_API_KEY_2=nvapi-...
NVIDIA_API_KEY_4=nvapi-...
NVIDIA_API_KEY_5=nvapi-...
GITHUB_TOKEN=...            # optional
```

## Usage

```bash
python senso_check.py --repo owner/name          # same as: python -m senso_check --repo owner/name
```

The tool clones the repository (blobless clone), finds the commit at which the manifest is current, extracts and
measures the claims, reasons, and prints one line per checked claim. Progress goes to stderr, the report to stdout.
Example (laravel-package-ocean, abridged; the full output is in `docs/senso-check-examples/`):

```
SENSO claim check: HassanZahirnia/laravel-package-ocean @ ed5108251b  manifest CLAUDE.md
  L12   T4  supported     - filament/filament (FILAMENT) - v4
  L17   T4  contradicted  - laravel/mcp (MCP) - v0
         rule: DependencyUndeclared  [born wrong]
         fact: dependencyExactMatchCount=0, dependencySuffixMatchCount=0, dependencyManifestParsedCount=2, claimedMajor=0
  L168  T1  supported     - Middleware are configured declaratively in `bootstrap/app.php` using
  L699  T1  contradicted  - [**/input-modalities/images.md] Use these docs for passing images as
         rule: PathNeverExisted  [born wrong]
         fact: pathExactMatchCount=0, pathSuffixMatchCount=0, pathIgnoredCount=0, pathSubmoduleCount=0, pathEverTrackedCount=0
...
summary: supported=18, contradicted=22
time: 288.8s  results: senso-out/HassanZahirnia__laravel-package-ocean/results
```

Reading a line: `L17` is the manifest line, `T4` the claim type, then the result and the quoted claim. For a contradicted
claim, `rule` is the ontology class that fired, the label in brackets says drifted, born wrong or origin unresolved
(with "false for N days" when known), and `fact` is what was measured in the repository.

Note: this example was recorded before the wildcard fix. Paths written with a leading `**/` (the `L699` line)
are treated as patterns now, so the current version would not report them as contradicted.

### Options

| Option | Meaning |
|--------|---------|
| `--repo owner/name` | Required. A public GitHub repository. |
| `--manifest FILE` | Use this local text as the manifest. It must equal the committed file, or the snapshot fails (exit 2). |
| `--manifest-path PATH` | Manifest path inside the repository. Default: first found of `CLAUDE.md`, `.claude/CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`. |
| `--local-path DIR` | Use an existing local checkout (read-only, git objects only) instead of cloning. Commit the manifest first. |
| `--out DIR` | Output directory. Default: `./senso-out/<owner__name>/`. |
| `--format text\|json` | Text report (default) or JSON with per-claim fields and per-step timings. |
| `--fail-on contradicted` | Exit with code 1 if any claim is contradicted (for CI). |
| `--skip-explain` | Skip the Pellet explanation step (saves about a minute). |
| `--all-claims` | Also list claims that received no result. |

Exit codes: `0` ok, `1` a claim is contradicted and `--fail-on contradicted` was given, `2` pipeline error (bad
repository, missing manifest, missing key, failed step; see `senso-out/<owner__name>/run.log`).

### Output files

Everything goes under `./senso-out/<owner__name>/` (never into your repository): `config.yaml`, `corpus.json`,
`run.log`, and `results/` with `snapshots/` (commit examined), `claims/consensus.jsonl` (claims at least two models
found), `measurements/` (measured facts), `verdicts/` (result and rule per claim), `drift/` (drifted, born wrong,
days; only if a claim is contradicted) and `explanations/` (Pellet justifications). A re-run reuses the cached
extraction. The ontology is built once into `~/.cache/senso-check/tbox` (change with `SENSO_TBOX_DIR`).

### Time

A small manifest takes 1 to 2 minutes, mostly for the extraction calls. A 35,000-character manifest took about
5 minutes with explanations. Large repositories take longer (clone and history walks). Re-runs are fast.

## Use in CI (GitHub Actions)

`examples/senso-check-ci.yml` is a workflow for your repository: it checks out `cnacha-mfu/senso-check`, installs the
requirements, sets up Java and fails the pull request if `CLAUDE.md` or `AGENTS.md` contains a contradicted claim.
It needs the three NIM keys as repository secrets. Because of the false alarms described below, consider running it
as advisory (drop `--fail-on contradicted`) first. The example is documentation and has not been run on GitHub.

## Session-start hook

`docs/session-start-hook.md` shows a Claude Code `SessionStart` hook that prints contradicted claims at the start of
each agent session. The command was tested from a shell, not inside Claude Code.

## Limitations

- A `contradicted` result means "check this line", not "proven error". On 30 new repositories, only 14 of 41
  contradicted results were real errors. The false alarms were extraction errors: two files written as one
  (`src/elf.c/h`), Go package paths read as file paths, naming templates (`NNNN-name.md`), absolute system paths,
  build outputs, a repository name at the root of a tree diagram, and the version of another tool.
- Dependency (T4) and tool (T3) results are the least reliable. Path (T1) and script (T5) results were confirmed
  most often. Example: `laravel/mcp` above is reported as undeclared although it is installed as a dependency
  of another package.
- Extraction misses claims: in the evaluation it covered 56% of the paths and 84% of the commands written in code spans.
- The extraction is a language-model step, so results can vary between runs. In our check of this package, two runs
  on the same manifest found different claim sets (for example 9 versus 20 claims for `kantord/headson`).
- The check examines the repository as it is when you run it. A `--manifest` that differs from the committed file
  makes the snapshot fail.
- Public GitHub repositories and local checkouts only; one manifest per run.
- The extraction depends on three models served on the NVIDIA NIM free tier. NIM retires models over time; if one is
  gone, extraction fails or the declared fallback is used, and results change.

## How it works

`senso_check.py` writes a one-repository config (`scripts/make_check_config.py`) and runs the steps of
`wp2_consistency.py`: `snapshot` (find the commit where the manifest is current), `extract` (three NIM models), `consensus`
(keep claims at least two models found and that quote the text), `measure` (measure each claim in the repository at
that commit), `reason` (HermiT classifies each claim with the ontology in `ontology/`), `drift` (drifted or born wrong,
only if a claim is contradicted) and `explain` (Pellet justifications). The ontology is generated from
`ontology/modules/build_ontology.py` and `claims_module.py` with the thresholds in `ontology/modules/thresholds.yaml`.
The extraction prompt is `configs/wp2_extraction_prompt.md`.

## How to cite

> Nacha Chondamrongkul and Jing Sun. "SENSO: Checking Coding-Agent Manifests Against Repository History." [Venue], 2026.

See `CITATION.cff`. (The venue and DOI will be added once the paper is published.)

## Licence

MIT, see `LICENSE`.
