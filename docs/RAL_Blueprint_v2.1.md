\## Changelog \& Adversarial Audit Summary (v2.0 $\\rightarrow$ v2.1)



A line-by-line systems, mathematical, and security audit of RAL Blueprint v2.0 identified \*\*11 concrete defects\*\* (6 targeted by your audit + 5 newly uncovered structural bugs). Every defect has been deterministically resolved in \*\*RAL Blueprint v2.1\*\* below:



| # | Defect Found in v2.0 | Root Cause / Failure Impact | Deterministic Fix in v2.1 |

| :--- | :--- | :--- | :--- |

| \*\*1\*\* | \*\*Hardcoded `-pro` Exclusion\*\* | `FRONTIER\_DENY\_PATTERNS` blindly matched `r"-pro($|-)"`, discarding usable legacy free-tier Pro models (e.g., `gemini-2.5-pro`) and preventing easy edits for active projects. | Moved all exclusion rules into `config/model\_policy.yaml` (`frontier\_exclude\_patterns` blocking active frontier series like `gemini-4`, `carbon`, `neon`, `argon`, `hydrogen`, `ultra`, `deep-research`, plus `user\_reserved\_patterns`). Older Pro/Flash/Gemma models remain eligible. |

| \*\*2\*\* | \*\*1-Token Canary Conflated with Capability Ranking\*\* | `discover\_free\_swarm()` ranked reasoning strength from a 1-token liveness ping, which measures only HTTP 200 latency—not symbolic/epistemic rigor. | Split into (a) nightly 1-token JSON liveness probe and (b) a 20-problem local \*\*Calibration Benchmark\*\* (`config/calibration\_suite.json`) run \*\*once per newly discovered `model\_id`\*\* and persisted in `metrics/model\_benchmarks.json`. |

| \*\*3\*\* | \*\*Hardcoded Embedding Model \& Vector-Space Mixing\*\* | Hardcoded `gemini-embedding-2`; switching between `gemini-embedding-2` and `gemini-embedding-001` silently corrupts cosine math because they produce \*\*incompatible vector spaces\*\* (and `gemini-embedding-2` aggregates list inputs into a single vector). | Dynamic `embedContent` discovery via `config/model\_policy.yaml` with live 1-string canary, single-document API calls, Matryoshka truncation (`768`d), and strict `meta.embedding\_model\_id` guard in SQLite triggering atomic staged `vectors\_next` re-embedding instead of mixing spaces. |

| \*\*4\*\* | \*\*Lean 4 / Mathlib Sandbox Bloat\*\* | Installing Lean 4 + Mathlib inside `ci/sandbox.Dockerfile` adds multi-GB bloat and stalls nightly builds. | Stripped Lean 4 from the core v1 container (`python:3.12-slim` + `sympy`, `scipy`, `numpy`, `z3-solver`, `hypothesis`, `pint` only, <350 MB). Added an extensible `OraclePlugin` interface (`src/ral/pipeline/oracles/`) for future optional verifiers. |

| \*\*5\*\* | \*\*Director Math Edge Cases (Backoff, Anisotropy, Day-0 Ties, Epoch Rounding)\*\* | (a) Paywalled `DanglingPrereqs` kept $\\sigma\_d(t)$ permanently elevated (infinite retry loop); (b) raw centroid cosines suffer narrow-band anisotropy; (c) $\\tau\_d(0)=0$ zeroed out Day-0 exploration and `w\_audit`/`w\_exploit` wasted Day-0 epochs on an empty vault; (d) partial-epoch ($K<6$) edge cases unspecified. | Added exponential prerequisite backoff ($t^{\\text{cool}}\_p = t\_p + \\min(90, 2^{k\_p})$) with 5-attempt quarantine; global mean-centering + top-2 PC isotropic whitening; initialized unvisited $\\tau\_d = \\infty$ with unbiased `sha256(date:d)` tie-breaking; gated `exploit`/`audit`/`synth` on $|V\_t|$; hardened Hamilton largest-remainder allocator for all $K \\in \[0, 6]$. |

| \*\*6\*\* | \*\*Public-Repo Boundary \& GitHub Rate/Retention Traps\*\* | Lacked hard CI enforcement preventing non-OA licenses or private local paths from entering the public repo; didn't budget `GITHUB\_TOKEN`'s 1,000 req/hr limit or default 90-day Actions artifact accumulation. | Added mandatory SPDX/OA `source\_license` allowlist + private-path/token regex bans to `ci/validate\_vault.py`; bounded nightly GitHub API usage to <40 calls/night with `X-RateLimit-Remaining` backoff; enforced `retention-days: 1` on inter-job handoff artifacts (never committing `plan.json` to Git). |

| \*\*7\*\* | \*\*Self-Discovered #1: Non-Atomic `--clobber` Release Asset Destruction\*\* | `gh release upload --clobber` deletes the existing `ral-index-latest.sqlite.zst` \*before\* uploading the replacement; any network error destroys the index. | \*\*Blue-Green Release Swap:\*\* Upload versioned `ral-index-YYYYMMDD-{run\_id}.sqlite.zst` $\\rightarrow$ verify checksum $\\rightarrow$ update 300-byte `ral-index-latest.json` pointer $\\rightarrow$ prune keeping last 2 snapshots. |

| \*\*8\*\* | \*\*Self-Discovered #2: Branch Protection vs. `GITHUB\_TOKEN` Push Deadlock\*\* | Requiring PR status checks on `main` blocks `ral\_nightly.yml` `git push` with `GH006`, while `GITHUB\_TOKEN`-created PRs do not trigger CI workflows. | Run `ci/validate\_node.py` + `ci/validate\_vault.py` synchronously as a \*\*blocking pre-push local gate\*\* before atomic `git push origin main`; restrict GitHub Ruleset on `main` strictly to blocking force-pushes and deletion. |

| \*\*9\*\* | \*\*Self-Discovered #3: Premature Drive Upload \& 2PC Orphan Leakage\*\* | Uploading Ore in Stage 1 polluted `RAL\_Ore\_Warehouse` with Stage 3/4 rejects and left permanent Drive orphans on mid-epoch runner timeouts. | \*\*Verify-Before-Archive 2PC:\*\* Keep Ore in runner `/tmp/ore\_cache` until Stages 1–5 pass, then upload to Drive immediately before Git commit. Added weekly \*\*Drive Orphan GC\*\* (`maintenance.py`) quarantining >7-day unreferenced files into `\_quarantine\_orphans/` (auto-purged at 30d). |

| \*\*10\*\* | \*\*Self-Discovered #4: Unverifiable `assumptions\_hash` \& Unnormalized `quote\_sha256`\*\* | `assumptions` existed only as prose bullets (impossible for CI to re-hash); `quote\_sha256` lacked canonical normalization rules. | Promoted structured `assumptions: list\[str]` and `source\_license: str` into YAML frontmatter; specified deterministic `NFKC` + lowercase + whitespace-collapse + LaTeX-comment-strip canonicalization for `quote\_sha256` and `assumptions\_hash`. |

| \*\*11\*\* | \*\*Self-Discovered #5: Underspecified MCP Server Security \& Epistemic Hygiene\*\* | `ral\_mcp\_server.py` lacked read-only SQLite locking, injection guards, cache-freshness verification, and `T3`/`T4` warnings. | Hardened MCP server with `mode=ro` SQLite URI, strict `ral\_id` regex validation, parameterized queries, `ral-index-latest.json` SHA256 sync check, and mandatory co-retrieval of `epistemic\_tier` + linked `NEGATIVE\_RESULT` counterexamples. |



\---



Below is the complete, standalone \*\*RAL Blueprint v2.1\*\* ready to save directly as `RAL\_Blueprint\_v2.1.md` and hand to \*\*Antigravity\*\*.



\---



\# Rayhan's Autonomous Lab (RAL) — Architecture \& Implementation Blueprint v2.1



> \*\*Document Status:\*\* Authoritative Engineering \& Epistemology Contract (Self-Contained)

> \*\*Target Implementer:\*\* Antigravity Engineering Agent

> \*\*Repository:\*\* `rayhans-autonomous-lab` (Public GitHub Repository)

> \*\*Cold Ore Archive:\*\* `RAL\_Ore\_Warehouse` (Google Drive, 4.8 TB)

> \*\*Operating Schedule:\*\* Nightly, 10:13 PM – 5:00 AM+ MUT (`13 18 \* \* \*` UTC), fully autonomous, zero local hardware.



\---



\## 0. Foundational Architectural Axiom



RAL runs on \*\*free-tier, non-frontier Gemini/Gemma models\*\* so that \*\*future frontier AI models\*\* (Gemini 4/5/6+ class) can ingest canonical, zero-redundancy scientific and mathematical knowledge for downstream R\&D.



To defeat the \*\*Weak-Teacher / Strong-Student Paradox\*\* (where prose written by weaker models degrades stronger future models through synthetic compression errors), RAL enforces \*\*Generator–Verifier Asymmetry\*\*:

1\. Weak/cheap LLMs are used strictly as \*\*stochastic search, extraction, formalization, and red-teaming engines\*\*.

2\. \*\*Prose summaries are rejected.\*\* Every canonical Claim Node must be anchored to primary-source markup (`.tex`, JATS XML, verified PDF) via deterministic verbatim quote hashes AND bounded by \*\*non-LLM deterministic oracles\*\* (SymPy symbolic algebra, dimensional analysis, SciPy numerical boundary fuzzing, Z3 SMT satisfiability, property-based testing) inside an air-gapped Docker sandbox.

3\. No claim may reach `T1\_CANONICAL` solely on LLM judgment.



\---



\## 1. System Topology \& Infrastructure Contracts



\### 1.1 Dual-Storage Split (Ore vs. Gold)

| Layer | Location | Contents | Storage \& Integrity Policy |

| :--- | :--- | :--- | :--- |

| \*\*Distilled Vault ("Gold")\*\* | GitHub Repo `rayhans-autonomous-lab` (\*\*PUBLIC\*\*) | Canonical Markdown + LaTeX + verified code kernels + YAML frontmatter + sharded append-only NDJSON ledgers/edges + nightly telemetry. | Pure compressible text only. \*\*Zero monolithic mutable JSONs, SQLite DBs, or embeddings committed to Git.\*\* Target size < 250 MB at 50,000+ nodes. |

| \*\*Compiled Ephemeral Index\*\* | GitHub Release tag `nightly-index` (Blue-Green assets) | `ral-index-YYYYMMDD-{run\_id}.sqlite.zst` (FTS5 + Graph adjacency + `ral\_id` path map + int8 embeddings) + 300-byte pointer `ral-index-latest.json`. | Versioned upload $\\rightarrow$ verify checksum $\\rightarrow$ update pointer $\\rightarrow$ prune keeping last \*\*2 snapshots\*\*. Never uses destructive `--clobber` on the binary asset. |

| \*\*Raw Source Archive ("Ore")\*\* | Google Drive `RAL\_Ore\_Warehouse` (4.8 TB) | Content-addressed raw `.tex.tar.gz`, JATS XML, Open-Access PDFs, and open datasets at `by-sha256/{h\[0:2]}/{h\[2:4]}/{sha256}.{ext}`. | \*\*Verify-Before-Archive 2PC:\*\* Uploaded \*only\* after a claim passes all verification stages. Weekly orphan GC moves >7-day unreferenced files to `\_quarantine\_orphans/` (purged at 30 days). |



\### 1.2 Public Repo Boundary Invariant (Strictly Enforced in CI)

Because `rayhans-autonomous-lab` is a \*\*public\*\* repository (unlocking unmetered standard Ubuntu Linux GitHub Actions minutes instead of the 2,000-minute/month private Free cliff) and uses the free-tier Gemini API:

1\. \*\*Open-Access License Allowlist Only:\*\* Every ingested source MUST carry an allowlisted Open-Access/Public-Domain license in YAML frontmatter (`CC-BY-4.0`, `CC-BY-3.0`, `CC-BY-SA-4.0`, `CC0-1.0`, `arXiv-non-exclusive-1.0`, `MIT`, `BSD-3-Clause`, `Apache-2.0`, `Public-Domain`, `US-Gov-NIST`, `PMC-OA`). Paywalled, proprietary, or unlicensed sources are hard-rejected at Stage 1 and blocked by `ci/validate\_vault.py`.

2\. \*\*Zero Private Data / Secret Leakage:\*\* `ci/validate\_vault.py` runs regex and Shannon-entropy scans blocking any local filesystem paths (`/Users/`, `C:\\Users\\`, `/home/rayhan`), private API keys, OAuth tokens, or emails prior to every commit.



\### 1.3 API \& Compute Isolation

\* \*\*Billing-Disabled GCP Project:\*\* RAL’s free-tier Gemini key (`RAL\_GEMINI\_FREE\_KEY`) MUST reside in a dedicated GCP project with \*\*no billing account attached\*\*. Because Gemini rate limits are project-scoped (not key-scoped), this makes billing charges or quota interference with paid frontier keys physically impossible.

\* \*\*Quota Window Alignment:\*\* Gemini Free Tier RPD resets at Midnight Pacific Time (`08:00 UTC` PST / `07:00 UTC` PDT = `12:00 PM` / `11:00 AM MUT`). The entire 10:13 PM – 5:00 AM+ MUT run executes cleanly within one unbroken quota window.

\* \*\*GitHub Actions `GITHUB\_TOKEN` Budget \& Artifact Retention:\*\*

&#x20; \* Primary limit: 1,000 REST requests/hour per repository; secondary limit: 80 content-creating requests/minute.

&#x20; \* RAL's nightly footprint across 8 jobs is bounded to \*\*< 40 GitHub API calls total\*\* (\~14 artifact upload/download ops, \~7 Git pushes, \~4 Release asset ops, \~1 conditional Issue check). All GitHub REST calls use a centralized backoff wrapper inspecting `X-RateLimit-Remaining` and `Retry-After`.

&#x20; \* Inter-job ephemeral handoff files (`plan.json`, `model\_router\_state.json`, `circuit\_breaker.json`) are passed exclusively via `actions/upload-artifact@v4` with \*\*`retention-days: 1`\*\* (never committed to `.git`; auto-deleted after 24 hours).



\### 1.4 Pre-Push Validation Gate \& Repository Ruleset (No Deadlock)

\* Do \*\*NOT\*\* require external PR status checks on `main` (which blocks automated `GITHUB\_TOKEN` commits with `GH006` and prevents `GITHUB\_TOKEN` PRs from triggering CI).

\* Instead, every execution epoch stages mutations locally, runs `python ci/validate\_node.py` + `python ci/validate\_vault.py` as a \*\*mandatory blocking pre-push step\*\*, rebases against `origin/main`, and fast-forward pushes with `permissions: contents: write`.

\* Configure a GitHub \*\*Repository Ruleset\*\* on `main` restricting \*\*force-pushes\*\* and \*\*branch deletion\*\* only.



\---



\## 2. Repository Directory Tree



```text

rayhans-autonomous-lab/

├── .github/

│   └── workflows/

│       └── ral\_nightly.yml               # Single workflow: Job 0 -> Epochs 1..6 -> Job 7

├── config/

│   ├── model\_policy.yaml                 # Configurable frontier deny-list + embedding preferences

│   ├── calibration\_suite.json            # Fixed deterministic 20-problem local capability benchmark

│   ├── taxonomy\_seeds.yaml               # Day-0 root domains (MSC2020, PhySH, ACM, IEEE, JEL, IUPAC, MeSH)

│   ├── curriculum\_params.yaml            # U(d,t), S(di,dj,t), backoff, discipline floor/ceiling

│   ├── license\_allowlist.yaml            # Mandatory SPDX / Open-Access source licenses

│   └── oracle\_plugins.yaml               # Enabled/disabled verification plugins (lean4 disabled in v1)

├── ci/

│   ├── sandbox.Dockerfile                # Lean Python 3.12 + sympy, scipy, numpy, z3-solver, hypothesis, pint

│   ├── validate\_node.py                  # Pydantic YAML schema, hashes, <=350w prose cap, AST linter

│   └── validate\_vault.py                 # Full DAG acyclicity, OA license gate, secret/path scan, ref integrity

├── src/

│   └── ral/

│       ├── \_\_init\_\_.py

│       ├── canonicalize.py               # Deterministic text/LaTeX normalization for quote \& assumption hashes

│       ├── preflight.py                  # Drive canary + dynamic model/embedding discovery + live 1-token probes

│       ├── model\_router.py               # Cached capability ranking, RPD tracking, cross-family assignment, fallback

│       ├── director.py                   # U(d,t), whitened S(di,dj,t), prereq backoff, epoch allocator, Leiden split

│       ├── github\_client.py              # Rate-limit-aware GitHub REST/Release Blue-Green helper

│       ├── sources/

│       │   ├── openalex.py               # OpenAlex metadata, OA filter, citation half-life fetcher

│       │   ├── arxiv\_tex.py              # Native .tex tarball downloader \& LaTeX AST/section parser

│       │   ├── pmc\_jats.py               # PubMed Central JATS XML fetcher \& parser

│       │   ├── crossref\_retract.py       # Crossref / Retraction Watch firewall \& DOI normalizer

│       │   └── reference\_dbs.py          # NIST constants, OEIS, DLMF deterministic fetchers

│       ├── storage/

│       │   ├── drive\_ore.py              # Content-addressed 2PC upload, checksum verify, canary, orphan GC

│       │   └── index\_builder.py          # Ephemeral SQLite (FTS5 + Graph + int8 vectors) \& space migration

│       ├── pipeline/

│       │   ├── scout.py                  # Dedup + native source extract + verbatim quote verifier

│       │   ├── formalizer.py             # Structured YAML + LaTeX + verify.py kernel generator

│       │   ├── ast\_linter.py             # Anti-tautology / anti-cheating Python AST verifier

│       │   ├── oracle\_sandbox.py         # Air-gapped Docker execution harness (--network=none)

│       │   ├── oracles/

│       │   │   ├── base.py               # Abstract OraclePlugin interface + registry

│       │   │   ├── sympy\_numeric.py      # Symbolic identity + pint units + 1,000-pt boundary fuzzing

│       │   │   ├── z3\_smt.py             # Z3 SMT solver + hypothesis property testing

│       │   │   └── empirical\_stats.py    # Unit/conservation + CI/p-value arithmetic reproduction

│       │   ├── skeptic.py                # Blinded cross-family adversarial red-teamer (falsify.py)

│       │   └── architect.py              # Hybrid dedup merge + Append-Only Theory-Upgrade engine

│       └── maintenance.py                # Zero-LLM fallback (retraction sweep, kernel regression, Drive GC)

├── mcp/

│   └── ral\_mcp\_server.py                 # Read-only MCP server (dependency cones, formal queries, failure modes)

├── vault/                                # Canonical Distilled Gold (dynamically sharded by taxonomy)

├── ledger/

│   ├── sources/                          # Sharded append-only NDJSON: YYYY-MM.ndjson

│   ├── rejected/                         # Sharded append-only NDJSON: YYYY-MM.ndjson

│   └── prereq\_debt.ndjson                # Backoff \& quarantine tracker for dangling prerequisites

├── graph/

│   ├── edges/                            # Sharded append-only NDJSON: YYYY-MM.ndjson

│   └── lineage/                          # Sharded append-only NDJSON: YYYY-MM.ndjson (SUPERSEDES history)

├── metrics/

│   ├── model\_benchmarks.json             # Persisted one-time calibration scores per discovered model\_id

│   └── telemetry/                        # Sharded append-only NDJSON: YYYY-MM.ndjson

├── INDEX.md                              # Auto-generated hierarchical vault map (<15,000 tokens)

└── llms.txt                              # Machine-readable entry point for future frontier AI models

```



\---



\## 3. Configuration Contracts \& Dynamic Model Governance



\### 3.1 Configurable Model Isolation Policy (`config/model\_policy.yaml`)

All frontier exclusions and embedding preferences are externalized—\*\*legacy free-tier Pro models (e.g., `gemini-2.5-pro`) are NOT blocked\*\*.



```yaml

schema\_version: "2.1"



\# Regex patterns (case-insensitive) matched against normalized model\_id.

\# Blocks active frontier tiers \& non-text modalities without blocking legacy Pro/Flash/Gemma models.

frontier\_exclude\_patterns:

&#x20; - "^gemini-4"

&#x20; - "carbon"

&#x20; - "neon"

&#x20; - "argon"

&#x20; - "hydrogen"

&#x20; - "ultra"

&#x20; - "deep-research"

&#x20; - "antigravity"

&#x20; - "imagen"

&#x20; - "veo"

&#x20; - "lyria"

&#x20; - "live"

&#x20; - "tts"

&#x20; - "audio"

&#x20; - "robotics"

&#x20; - "computer-use"



\# User-extensible list for reserving specific models for active projects at any time.

user\_reserved\_patterns: \[]



\# Minimum quality floors from calibration\_suite.json (0.0 to 1.0)

min\_json\_reliability: 0.90

min\_formalizer\_score: 0.55

min\_scout\_score: 0.40



\# Capability epoch jump required to trigger Generational Re-Audit Ratchet

epoch\_upgrade\_delta: 0.05



\# Dynamic embedding model discovery preference order (must support embedContent)

preferred\_embedding\_patterns:

&#x20; - "^gemini-embedding-2"

&#x20; - "^gemini-embedding-001"

&#x20; - "^text-embedding-"

embedding\_output\_dim: 768

embedding\_migration\_grace\_nights: 3  # Only migrate vector space if primary is unavailable >= 3 consecutive nights

```



\### 3.2 Liveness Probing vs. Persisted Calibration Benchmark (`src/ral/preflight.py` \& `src/ral/model\_router.py`)



\*\*Design Rule:\*\* Never rank reasoning capability from a 1-token canary. Separate fast nightly liveness verification from persisted one-time capability benchmarking.



```python

\# src/ral/preflight.py — Authoritative Implementation Contract

from \_\_future\_\_ import annotations

import json, re, time

from dataclasses import dataclass

from pathlib import Path

import yaml



@dataclass(frozen=True)

class LiveModelStatus:

&#x20;   model\_id: str

&#x20;   family: str           # e.g., "gemini-3", "gemini-2.5", "gemma-4"

&#x20;   live: bool

&#x20;   json\_ok: bool

&#x20;   latency\_ms: int

&#x20;   error: str | None = None



def infer\_family(model\_id: str) -> str:

&#x20;   m = re.match(r"^(gemini-\\d+(?:\\.\\d+)?|gemma-\\d+)", model\_id.lower())

&#x20;   return m.group(1) if m else model\_id.split("-")\[0].lower()



def load\_policy(path: Path = Path("config/model\_policy.yaml")) -> dict:

&#x20;   return yaml.safe\_load(path.read\_text(encoding="utf-8"))



def is\_excluded(model\_id: str, policy: dict) -> bool:

&#x20;   patterns = policy\["frontier\_exclude\_patterns"] + policy.get("user\_reserved\_patterns", \[])

&#x20;   return any(re.search(pat, model\_id, re.IGNORECASE) for pat in patterns)



def probe\_generate\_liveness(client, model\_id: str, timeout\_s: int = 12) -> LiveModelStatus:

&#x20;   """1-token strict-JSON liveness probe ONLY. Does NOT rank capability."""

&#x20;   t0 = time.monotonic()

&#x20;   try:

&#x20;       resp = client.models.generate\_content(

&#x20;           model=model\_id,

&#x20;           contents='Reply with exact JSON: {"ok":true}',

&#x20;           config={"max\_output\_tokens": 16, "temperature": 0.0, "response\_mime\_type": "application/json"},

&#x20;       )

&#x20;       raw = (resp.text or "").strip()

&#x20;       parsed = json.loads(raw)

&#x20;       json\_ok = isinstance(parsed, dict) and parsed.get("ok") is True

&#x20;       return LiveModelStatus(

&#x20;           model\_id=model\_id, family=infer\_family(model\_id),

&#x20;           live=True, json\_ok=json\_ok, latency\_ms=int((time.monotonic() - t0) \* 1000)

&#x20;       )

&#x20;   except Exception as exc:

&#x20;       return LiveModelStatus(

&#x20;           model\_id=model\_id, family=infer\_family(model\_id),

&#x20;           live=False, json\_ok=False, latency\_ms=int((time.monotonic() - t0) \* 1000),

&#x20;           error=type(exc).\_\_name\_\_

&#x20;       )



def discover\_embedding\_model(client, policy: dict) -> dict | None:

&#x20;   """Dynamically discovers the highest-preference live free-tier embedding model."""

&#x20;   available = \[]

&#x20;   for m in client.models.list():

&#x20;       mid = m.name.removeprefix("models/")

&#x20;       actions = getattr(m, "supported\_actions", None) or getattr(m, "supported\_generation\_methods", \[]) or \[]

&#x20;       if "embedContent" not in actions or is\_excluded(mid, policy):

&#x20;           continue

&#x20;       available.append(mid)



&#x20;   target\_dim = int(policy\["embedding\_output\_dim"])

&#x20;   for pattern in policy\["preferred\_embedding\_patterns"]:

&#x20;       matches = sorted(\[m for m in available if re.search(pattern, m, re.IGNORECASE)], reverse=True)

&#x20;       for mid in matches:

&#x20;           try:

&#x20;               # Single-string invocation (required for gemini-embedding-2 so list inputs aren't aggregated)

&#x20;               res = client.models.embed\_content(

&#x20;                   model=mid,

&#x20;                   contents="RAL embedding liveness canary",

&#x20;                   config={"output\_dimensionality": target\_dim},

&#x20;               )

&#x20;               vec = res.embeddings\[0].values

&#x20;               if len(vec) == target\_dim:

&#x20;                   return {"model\_id": mid, "dim": target\_dim, "live": True}

&#x20;           except Exception:

&#x20;               continue

&#x20;   return None

```



```python

\# src/ral/model\_router.py — Capability Ranking, Epoch Ratchet \& Dynamic Fallback Contract

from \_\_future\_\_ import annotations

import json

from datetime import datetime, timezone

from pathlib import Path



BENCHMARK\_PATH = Path("metrics/model\_benchmarks.json")

SUITE\_PATH = Path("config/calibration\_suite.json")



def ensure\_benchmarked(client, live\_models: list\[dict], policy: dict) -> tuple\[dict, bool]:

&#x20;   """

&#x20;   Loads cached benchmark scores from metrics/model\_benchmarks.json.

&#x20;   Runs the deterministic 20-problem local calibration suite ONLY for newly discovered model\_ids

&#x20;   (or if suite\_version changed). Returns (benchmarks\_db, capability\_epoch\_upgraded).

&#x20;   """

&#x20;   suite = json.loads(SUITE\_PATH.read\_text(encoding="utf-8"))

&#x20;   suite\_ver = suite\["suite\_version"]

&#x20;   db = (

&#x20;       json.loads(BENCHMARK\_PATH.read\_text(encoding="utf-8"))

&#x20;       if BENCHMARK\_PATH.exists()

&#x20;       else {"model\_capability\_epoch": 1, "historical\_max\_score": 0.0, "suite\_version": suite\_ver, "models": {}}

&#x20;   )

&#x20;   if db.get("suite\_version") != suite\_ver:

&#x20;       db\["models"] = {}

&#x20;       db\["suite\_version"] = suite\_ver



&#x20;   changed = False

&#x20;   for m in live\_models:

&#x20;       mid = m\["model\_id"]

&#x20;       if not (m\["live"] and m\["json\_ok"]):

&#x20;           continue

&#x20;       if mid not in db\["models"]:

&#x20;           # Runs 20 deterministic local checks: 8 SymPy/math formalizations, 6 counterexample traps, 6 strict JSON tasks

&#x20;           res = run\_local\_calibration\_suite(client, mid, suite\["problems"])

&#x20;           db\["models"]\[mid] = {

&#x20;               "overall\_score": res\["overall\_score"],

&#x20;               "math\_score": res\["math\_score"],

&#x20;               "skeptic\_score": res\["skeptic\_score"],

&#x20;               "json\_reliability": res\["json\_reliability"],

&#x20;               "family": m\["family"],

&#x20;               "benchmarked\_at": datetime.now(timezone.utc).isoformat(),

&#x20;           }

&#x20;           changed = True



&#x20;   live\_scores = \[

&#x20;       db\["models"]\[m\["model\_id"]]\["overall\_score"]

&#x20;       for m in live\_models

&#x20;       if m\["live"] and m\["json\_ok"] and m\["model\_id"] in db\["models"]

&#x20;   ]

&#x20;   epoch\_upgraded = False

&#x20;   if live\_scores:

&#x20;       best\_live = max(live\_scores)

&#x20;       if db\["historical\_max\_score"] == 0.0:

&#x20;           db\["historical\_max\_score"] = best\_live

&#x20;           changed = True

&#x20;       elif best\_live >= db\["historical\_max\_score"] + float(policy\["epoch\_upgrade\_delta"]):

&#x20;           db\["model\_capability\_epoch"] += 1

&#x20;           db\["historical\_max\_score"] = best\_live

&#x20;           epoch\_upgraded = True

&#x20;           changed = True



&#x20;   if changed:

&#x20;       BENCHMARK\_PATH.parent.mkdir(parents=True, exist\_ok=True)

&#x20;       BENCHMARK\_PATH.write\_text(json.dumps(db, indent=2, sort\_keys=True) + "\\n", encoding="utf-8")

&#x20;   return db, epoch\_upgraded



def build\_routing\_table(live\_models: list\[dict], db: dict, policy: dict) -> dict:

&#x20;   """

&#x20;   Assigns ranked fallback chains for each swarm role using persisted benchmark scores.

&#x20;   Enforces orthogonal model-family diversity between Formalizer and Adversarial Skeptic when available.

&#x20;   """

&#x20;   eligible = \[]

&#x20;   for m in live\_models:

&#x20;       mid = m\["model\_id"]

&#x20;       if not (m\["live"] and m\["json\_ok"]) or mid not in db\["models"]:

&#x20;           continue

&#x20;       rec = db\["models"]\[mid]

&#x20;       if rec\["json\_reliability"] >= policy\["min\_json\_reliability"]:

&#x20;           eligible.append({"model\_id": mid, "latency\_ms": m\["latency\_ms"], \*\*rec})



&#x20;   if not eligible:

&#x20;       return {"status": "ZERO\_HEALTHY\_MODELS", "roles": {}}



&#x20;   # Rank Formalizer/Architect by math + overall capability

&#x20;   formalizer\_chain = sorted(eligible, key=lambda x: (-x\["math\_score"], -x\["overall\_score"], x\["latency\_ms"]))

&#x20;   top\_family = formalizer\_chain\[0]\["family"]



&#x20;   # Rank Skeptic preferring orthogonal model family (e.g., Gemma-4 vs Gemini Flash/Pro) then skeptic\_score

&#x20;   skeptic\_chain = sorted(

&#x20;       eligible,

&#x20;       key=lambda x: (0 if x\["family"] != top\_family else 1, -x\["skeptic\_score"], -x\["overall\_score"])

&#x20;   )



&#x20;   # Rank Scout preferring high throughput / lowest latency above minimum score floor

&#x20;   scout\_pool = \[x for x in eligible if x\["overall\_score"] >= policy\["min\_scout\_score"]] or eligible

&#x20;   scout\_chain = sorted(scout\_pool, key=lambda x: (x\["latency\_ms"], -x\["overall\_score"]))



&#x20;   return {

&#x20;       "status": "OK",

&#x20;       "model\_capability\_epoch": db\["model\_capability\_epoch"],

&#x20;       "roles": {

&#x20;           "formalizer": \[x\["model\_id"] for x in formalizer\_chain],

&#x20;           "architect":  \[x\["model\_id"] for x in formalizer\_chain],

&#x20;           "skeptic":    \[x\["model\_id"] for x in skeptic\_chain],

&#x20;           "scout":      \[x\["model\_id"] for x in scout\_chain],

&#x20;       },

&#x20;       "exhausted\_models": \[],  # Dynamically appended on HTTP 429 RPD exhaustion during epochs

&#x20;   }

```



\### 3.3 Vector-Space Integrity Guard (`src/ral/storage/index\_builder.py`)

1\. Every compiled SQLite index stamps `meta(key, value)` with `embedding\_model\_id` and `embedding\_dim`.

2\. \*\*Never mix embeddings from different models in one table.\*\*

&#x20;  \* If `discovered\_embed\_model != meta.embedding\_model\_id` for $< 3$ consecutive nights (transient outage/429): keep existing vectors intact, skip embedding writes and Leiden folder splits for that night, and serve hybrid retrieval via \*\*FTS5 BM25 + `formal\_signature` Jaccard\*\*.

&#x20;  \* If `discovered\_embed\_model != meta.embedding\_model\_id` for $\\ge 3$ consecutive nights (deprecation or generational upgrade): create a staging table `vectors\_next`, incrementally embed batches across nightly maintenance slots until 100% of `nodes` are populated in the new space, then atomically swap `vectors\_next -> vectors` and update `meta.embedding\_model\_id`.



\---



\## 4. Autonomous Curriculum \& Curiosity Engine (`src/ral/director.py`)



\### 4.1 Taxonomy State \& Dynamic Sub-Branch Splitting

\* \*\*Day-0 Roots (`config/taxonomy\_seeds.yaml`):\*\* 7 macro-disciplines seeded once with zero human topic schedule:

&#x20; `math` (MSC2020), `phys` (PhySH), `cs` (ACM CCS), `eng` (IEEE), `econ` (JEL), `chem` (IUPAC), `bio` (MeSH/NCBI), each with 4–6 foundational Level-1 branches (\~35 initial leaves).

\* \*\*Autonomous Splitting:\*\* Whenever leaf folder $d \\in \\mathcal{L}(\\mathcal{T}\_t)$ exceeds $n\_d > 40$ verified nodes AND embeddings are in a consistent vector space, run deterministic Leiden/Ward clustering over node embeddings ($k \\in \[2, 5]$, requiring silhouette score $> 0.25$). Relocate files via `git mv`. Because every graph edge references immutable `ral\_id` (resolved via SQLite `nodes.rel\_path`), \*\*zero links break\*\*.



\### 4.2 Unresolvable Prerequisite Exponential Backoff

Every dangling prerequisite stub $p \\in \\text{Dangling}(d)$ (a `DEPENDS\_ON` target referenced in a proof that lacks a canonical `T1` node) is tracked in `ledger/prereq\_debt.ndjson` with `(stub\_id, domain\_d, attempt\_count k\_p, last\_attempt\_night t\_p, status)`:

\* On each failed open-access resolution attempt, increment $k\_p \\leftarrow k\_p + 1$ and compute cooldown:

&#x20; $$t^{\\text{cool}}\_p = t\_p + \\min\\!\\big(90,\\; 2^{k\_p}\\big)$$

\* If $k\_p \\ge 5$, transition stub to `status: UNRESOLVABLE\_PAYWALLED\_OR\_MISSING` (tension contribution = $0$; eligible for passive re-check once every 180 nights or if resolved opportunistically by an incoming paper).

\* \*\*Effective Epistemic Tension\*\* $\\sigma\_d(t)$:

&#x20; $$\\sigma\_d(t) = w\_1 \\sum\_{\\substack{p \\in \\text{Dangling}(d) \\\\ \\text{status}=\\text{OPEN}}} \\mathbb{I}\\!\\left\[t \\ge t^{\\text{cool}}\_p\\right] \\cdot e^{-0.5\\,k\_p} \\;+\\; w\_2\\,|\\text{OpenConflicts}(d)| \\;+\\; w\_3\\,|\\text{ExpiredT3}(d)|$$

&#x20; \*(Defaults: $w\_1 = 1.0,\\; w\_2 = 1.5,\\; w\_3 = 0.5$.)\*



\### 4.3 Domain Priority Score $U(d, t)$ \& Deterministic Day-0 Tie-Breaking

For every active leaf domain $d \\in \\mathcal{L}(\\mathcal{T}\_t)$ at night $t$:

\* $n\_d(t)$: Verified (`T1` + `T2`) node count.

\* $\\rho\_d(t) = 0.5 \\cdot \\widetilde{\\text{PR}}\_{\\text{cross}}(d) + 0.5 \\cdot \\frac{\\ln(1 + h\_d)}{\\ln(1 + h\_{\\max})} \\in \[0, 1]$: \*\*Foundational Leverage\*\*, combining normalized cross-domain reverse PageRank and OpenAlex median citation half-life $h\_d$ ($h\_{\\max} = 50\\text{ yrs}$; default $h\_d = 10\\text{ yrs}$ before first fetch).

\* $\\tau\_d(t)$: Nights since last work on $d$. \*\*Day-0 Invariant:\*\* For any never-visited domain ($t^{\\text{last}}\_d = \\text{None}$), define $\\tau\_d(t) = +\\infty$ so that $1 - e^{-\\tau\_d(t)/\\tau\_0} = 1.0$ identically (preventing Day-0 exploration collapse).

\* $c\_d(t)$: Compute epochs spent on $d$ in the trailing 30 nights.

\* $\\hat{r}\_d(t)$: EWMA of verified nodes yielded per 100k tokens in $d$ (initialized to global mean $\\bar{r}(t)$).

\* $T = \\sum\_{d' \\in \\mathcal{L}(\\mathcal{T}\_t)} (n\_{d'}(t) + 1)$.



$$U(d, t) = \\underbrace{\\alpha \\cdot \\frac{\\rho\_d(t) \\cdot \\big(1 + \\ln(1 + \\sigma\_d(t))\\big)}{1 + \\lambda\_{\\text{sat}} \\cdot c\_d(t)}}\_{\\text{Exploitation: Foundational Depth \\\& Tension}} \\;+\\; \\underbrace{\\beta \\cdot \\sqrt{\\frac{\\ln(T + 1)}{n\_d(t) + 1}} \\cdot \\left(1 - e^{-\\tau\_d(t)/\\tau\_0}\\right)}\_{\\text{Exploration: UCB1 Coverage \\\& Staleness}} \\;+\\; \\underbrace{\\gamma \\cdot \\operatorname{clip}\\!\\left(\\hat{r}\_d(t) - \\bar{r}(t),\\, -0.2,\\, 0.2\\right)}\_{\\text{Yield Efficiency Feedback}}$$



\*(Defaults in `config/curriculum\_params.yaml`: $\\alpha = 1.0,\\; \\beta = 0.85,\\; \\gamma = 0.5,\\; \\lambda\_{\\text{sat}} = 0.35,\\; \\tau\_0 = 14$.)\*



\* \*\*Macro-Discipline Balance Guard:\*\* Any leaf whose root macro-discipline $M(d)$ consumed $\\ge 20\\%$ of trailing 30-day epochs is masked (unless all macro-disciplines are masked); any macro-discipline below $4\\%$ receives a $2.0\\times$ priority multiplier on $U(d, t)$.

\* \*\*Unbiased Deterministic Tie-Breaking:\*\* Sort candidate domains by:

&#x20; $$\\text{key}(d) = \\Big(-\\operatorname{round}\\!\\big(U(d, t),\\, 9\\big),\\;\\; -\\text{seed\\\_prior}(d),\\;\\; \\operatorname{sha256}\\!\\big(\\text{"}\\{ \\text{run\\\_date\\\_iso} \\}\\text{:}\\{d\\}\\text{"}\\big)\[:8]\\Big)$$

&#x20; \*(Eliminates alphabetical bias on Day 0 while remaining 100% reproducible for any given date.)\*



\### 4.4 Isotropic Cross-Domain Synthesis Score $S(d\_i, d\_j, t)$

To eliminate narrow-cone embedding anisotropy before measuring cross-discipline similarity:

1\. For every verified node $u \\in V\_t$, let $\\mathbf{v}\_u \\in \\mathbb{R}^{768}$ be the L2-normalized embedding of its \*\*domain-stripped `formal\_signature` + operator AST\*\* (not raw prose).

2\. Compute the global vault mean $\\boldsymbol{\\mu}\_{\\text{vault}} = \\frac{1}{|V\_t|} \\sum\_{u \\in V\_t} \\mathbf{v}\_u$, center $\\tilde{\\mathbf{v}}\_u = \\mathbf{v}\_u - \\boldsymbol{\\mu}\_{\\text{vault}}$, and compute the top $m = 2$ principal directions $\\mathbf{P}\_2 \\in \\mathbb{R}^{2 \\times 768}$ via truncated SVD on $\\tilde{\\mathbf{V}}$.

3\. Project out common anisotropy ("All-but-the-Top") and re-normalize:

&#x20;  $$\\mathbf{v}'\_u = \\tilde{\\mathbf{v}}\_u - (\\tilde{\\mathbf{v}}\_u \\mathbf{P}\_2^\\top)\\mathbf{P}\_2, \\qquad \\hat{\\mathbf{v}}\_u = \\frac{\\mathbf{v}'\_u}{\\max(\\|\\mathbf{v}'\_u\\|\_2,\\, 10^{-8})}$$

4\. Compute isotropic domain centroids $\\hat{\\mathbf{e}}\_{\\text{struct}}(d) = \\frac{\\sum\_{u \\in d} \\hat{\\mathbf{v}}\_u}{\\max(\\|\\sum\_{u \\in d} \\hat{\\mathbf{v}}\_u\\|\_2,\\, 10^{-8})}$ and score cross-discipline pairs ($M(d\_i) \\neq M(d\_j)$):



$$S(d\_i, d\_j, t) = \\underbrace{\\max\\!\\Big(0,\\; \\hat{\\mathbf{e}}\_{\\text{struct}}(d\_i)^\\top \\hat{\\mathbf{e}}\_{\\text{struct}}(d\_j)\\Big)}\_{\\text{Whitened Isotropic Similarity}} \\;\\cdot\\; \\underbrace{\\left(1 - \\frac{|E\_{\\text{bridge}}(d\_i, d\_j)|}{1 + \\min(n\_{d\_i}, n\_{d\_j})}\\right)}\_{\\text{Unbridged Structural Hole}} \\;\\cdot\\; \\underbrace{\\sqrt{\\bar{q}\_{d\_i}(t) \\cdot \\bar{q}\_{d\_j}(t)}}\_{\\text{Minimum Rigor Gate}}$$



Synthesis triggers only for pairs satisfying $S(d\_i, d\_j, t) \\ge 0.35$ and $\\min(\\bar{q}\_{d\_i}, \\bar{q}\_{d\_j}) \\ge 0.70$, producing a \*\*Bridge Node (`BRIDGE\_ISOMORPHISM`)\*\* with an explicit variable-mapping dictionary—zero duplicated prose.



\### 4.5 Continuous Portfolio Allocation \& Zero/Partial-Epoch Safe Implementation (`src/ral/director.py`)

Let $H\_{\\text{cov}}(t) \\in \[0, 1]$ be normalized Shannon entropy of node counts across the 7 macro-disciplines ($H\_{\\text{cov}} = 0$ if $|V\_t| = 0$), $D\_{\\text{debt}}(t) = \\min\\!\\big(1,\\; \\frac{\\sum\_d \\sigma\_d(t)}{0.25\\,|V\_t| + 10}\\big)$, and $\\bar{Q}(t) \\in \[0, 1]$ global oracle pass rate:



\* $w\_{\\text{explore}} = 0.15 + 0.45 \\cdot (1 - H\_{\\text{cov}}(t))$

\* $w\_{\\text{exploit}} = \\big(0.25 + 0.30 \\cdot D\_{\\text{debt}}(t)\\big) \\cdot \\mathbb{I}\[|V\_t| \\ge 10]$

\* $w\_{\\text{audit}} = \\big(0.15 + 0.20 \\cdot \\mathbb{I}\[\\text{epoch\\\_upgraded} \\lor |\\text{ExpiredT3}| > 20]\\big) \\cdot \\mathbb{I}\[|V\_t| \\ge 25]$

\* $w\_{\\text{synth}} = 0.25 \\cdot H\_{\\text{cov}}(t) \\cdot \\bar{Q}(t) \\cdot \\mathbb{I}\[|V\_t| \\ge 150 \\land \\text{embed\\\_space\\\_consistent}]$



```python

\# src/ral/director.py — Safe Hamilton Largest-Remainder Allocator (K in 0..6)

import math



def allocate\_epochs(weights: dict\[str, float], available\_epochs: int) -> dict\[str, int]:

&#x20;   """

&#x20;   Deterministically partitions `available\_epochs` (0 <= K <= 6) across modes.

&#x20;   Zero-division safe when K == 0 or all weights degenerate to 0.

&#x20;   """

&#x20;   modes = \["exploit", "explore", "audit", "synth"]

&#x20;   if available\_epochs <= 0:

&#x20;       return {m: 0 for m in modes}



&#x20;   clean = {m: max(0.0, float(weights.get(m, 0.0))) for m in modes}

&#x20;   total\_w = sum(clean.values())

&#x20;   if total\_w <= 1e-12:

&#x20;       clean = {"explore": 1.0, "exploit": 0.0, "audit": 0.0, "synth": 0.0}

&#x20;       total\_w = 1.0



&#x20;   exact = {m: (clean\[m] / total\_w) \* available\_epochs for m in modes}

&#x20;   floors = {m: math.floor(exact\[m]) for m in modes}

&#x20;   remainder = available\_epochs - sum(floors.values())

&#x20;   tie\_order = sorted(modes, key=lambda m: (-(exact\[m] - floors\[m]), modes.index(m)))

&#x20;   for m in tie\_order\[:remainder]:

&#x20;       floors\[m] += 1



&#x20;   assert sum(floors.values()) == available\_epochs

&#x20;   return floors

```



\*(Result on Day 0 with $|V\_t| = 0$, $K = 6$: `{"explore": 6, "exploit": 0, "audit": 0, "synth": 0}`—zero wasted epochs on empty audit/exploit states. On $K = 0$ circuit-breaker trip: all zeros $\\rightarrow$ immediate pivot to Zero-LLM maintenance.)\*



\---



\## 5. Nightly Orchestrator State Machine (`.github/workflows/ral\_nightly.yml`)



```text

\[TRIGGER: cron '13 18 \* \* \*' UTC (22:13 MUT) | workflow\_dispatch | Cloudflare Backup]

\[GLOBAL LOCK: concurrency: { group: ral-nightly-pipeline, cancel-in-progress: false }]

&#x20;                                  │

&#x20;                                  ▼

┌──────────────────────────────────────────────────────────────────────┐

│  JOB 0: preflight\_and\_plan (timeout: 15 min)                         │

│  1. Pull latest index via ral-index-latest.json (Blue-Green pointer) │

│  2. Build/load cached ci/sandbox.Dockerfile (GHA Docker layer cache) │

│  3. Drive 16-byte write/read/delete canary -> if FAIL: set           │

│     circuit\_breaker="DRIVE\_AUTH\_FAIL", open alert Issue, K\_llm=0     │

│  4. models.list() -> filter config/model\_policy.yaml -> 1-token JSON │

│     liveness probes + dynamic embedding discovery                    │

│  5. Run 20-problem calibration ONLY for brand-new model\_ids ->       │

│     commit metrics/model\_benchmarks.json if updated                  │

│  6. Run director.py -> compute U(d,t), whitened S(di,dj,t),          │

│     Leiden splits (if n\_d > 40) -> emit plan.json + router\_state.json│

│  7. Upload handoff artifact (retention-days: 1)                      │

└──────────────────────────────────┬───────────────────────────────────┘

&#x20;                                  │ needs: \[preflight\_and\_plan]

&#x20;                                  ▼

┌──────────────────────────────────────────────────────────────────────┐

│  JOBS 1..6: epoch\_1 .. epoch\_6 (timeout: 55m hard / 48m soft stop)   │

│  Sequential chain: epoch\_k has `needs: \[epoch\_{k-1}]` \&              │

│                    `if: always() \&\& !cancelled()`                    │

│  1. Download handoff artifact (plan.json, router\_state.json)         │

│  2. If circuit\_breaker == DRIVE\_AUTH\_FAIL -> skip Ore ops, run       │

│     Zero-LLM local validation only; exit cleanly                     │

│  3. If all LLM models exhausted (429 RPD) -> pivot remaining minutes │

│     to maintenance.py (retraction sweep, kernel regression, Drive GC)│

│  4. Execute assigned mode: Scout -> Formalizer -> AST Linter ->      │

│     Air-Gapped Docker Sandbox -> Blinded Skeptic -> Architect        │

│  5. Verify-Before-Archive 2PC:                                       │

│     (a) Upload ONLY verified Ore to Drive by-sha256 + verify MD5     │

│     (b) Stage Markdown + NDJSON locally -> run validate\_node.py \&    │

│         validate\_vault.py (BLOCKING pre-push gate)                   │

│     (c) git pull --rebase \&\& git push origin main                    │

│  6. Update router\_state.json artifact (retention-days: 1)            │

└──────────────────────────────────┬───────────────────────────────────┘

&#x20;                                  │ needs: \[epoch\_6], if: always()

&#x20;                                  ▼

┌──────────────────────────────────────────────────────────────────────┐

│  JOB 7: consolidate\_and\_watchdog (timeout: 15 min)                   │

│  1. Run full ci/validate\_vault.py (DAG acyclicity, OA licenses,      │

│     secret/path scan, referential integrity, .git size telemetry)    │

│  2. Compile SQLite index (FTS5 + Graph + int8 vectors)               │

│  3. Blue-Green Release Swap: upload ral-index-YYYYMMDD-{id}.sqlite.zst│

│     -> verify checksum -> update ral-index-latest.json -> keep last 2│

│  4. Regenerate /INDEX.md (<15k tokens) \& /llms.txt -> commit         │

│  5. Append metrics/telemetry/YYYY-MM.ndjson                          │

│  6. Dead-Man's Switch: if Drive failed, .git > 700 MB, or 0 verified │

│     nodes for 3 consecutive nights -> open/update single Issue       │

│     labeled `ral-critical-alert`                                     │

└──────────────────────────────────────────────────────────────────────┘

```



\---



\## 6. Epistemic Verification, Sandbox \& Theory-Upgrade Protocol



\### 6.1 Lean Air-Gapped Docker Sandbox (`ci/sandbox.Dockerfile`)

Lean 4 / Mathlib is intentionally excluded from v1 to keep image size < 350 MB and build time < 45 s.



```dockerfile

\# ci/sandbox.Dockerfile

FROM python:3.12-slim



ENV PYTHONDONTWRITEBYTECODE=1 \\

&#x20;   PYTHONUNBUFFERED=1 \\

&#x20;   PIP\_NO\_CACHE\_DIR=1



RUN pip install --no-cache-dir \\

&#x20;   sympy==1.13.3 \\

&#x20;   scipy==1.14.1 \\

&#x20;   numpy==2.1.3 \\

&#x20;   z3-solver==4.13.3.0 \\

&#x20;   hypothesis==6.119.0 \\

&#x20;   pint==0.24.4



RUN useradd --uid 65534 --no-create-home --shell /usr/sbin/nologin sandboxuser

WORKDIR /work

USER 65534:65534

```



\*\*Mandatory Execution Invocation (`src/ral/pipeline/oracle\_sandbox.py`):\*\*

```bash

docker run --rm \\

&#x20; --network=none \\

&#x20; --read-only \\

&#x20; --cap-drop=ALL \\

&#x20; --security-opt=no-new-privileges \\

&#x20; --memory=1g --memory-swap=1g \\

&#x20; --cpus=1 --pids-limit=64 \\

&#x20; --user=65534:65534 \\

&#x20; --tmpfs /tmp:rw,noexec,nosuid,size=64m \\

&#x20; -v "${EPHEMERAL\_SCRIPT\_DIR}:/work:ro" \\

&#x20; ral-sandbox:latest \\

&#x20; timeout -k 5 30 python3 /work/verify.py

```

\*Zero secrets are mounted or passed via `-e`. Untrusted PDF/LaTeX text extraction ALSO runs inside this container.\*



\### 6.2 Pluggable Oracle Architecture (`src/ral/pipeline/oracles/base.py`)

```python

from abc import ABC, abstractmethod

from dataclasses import dataclass



@dataclass(frozen=True)

class OracleResult:

&#x20;   passed: bool

&#x20;   plugin\_name: str

&#x20;   stdout: str

&#x20;   stderr: str

&#x20;   elapsed\_ms: int



class OraclePlugin(ABC):

&#x20;   name: str

&#x20;   enabled\_by\_default: bool = True



&#x20;   @abstractmethod

&#x20;   def lint(self, source\_code: str) -> list\[str]:

&#x20;       """Returns list of static anti-cheating violations; empty means pass."""



&#x20;   @abstractmethod

&#x20;   def execute(self, source\_code: str, timeout\_s: int = 30) -> OracleResult:

&#x20;       """Executes verification kernel inside air-gapped Docker sandbox."""

```

\*Enabled v1 Plugins (`config/oracle\_plugins.yaml`):\* `sympy\_numeric`, `z3\_smt`, `empirical\_stats`. Future plugins (e.g., `lean4\_external`) register against `OraclePlugin` without modifying the pipeline or bloating the core image.



\### 6.3 Five-Stage Adversarial Verification Loop

1\. \*\*Stage 1 — Scout \& Native-Markup Extractor (`scout.py`):\*\*

&#x20;  \* Queries OpenAlex, arXiv (`.tex` source tarball first), PubMed Central (JATS XML first), Crossref/Retraction Watch, NIST, OEIS.

&#x20;  \* \*\*License \& Retraction Gate:\*\* Hard-drops any source not matching `config/license\_allowlist.yaml` or flagged retracted/withdrawn.

&#x20;  \* \*\*Deterministic Deduplication:\*\* Checks `sha256`, `canonical\_doi`, and `arxiv\_base\_id` against `ledger/sources/\*.ndjson` and `ledger/rejected/\*.ndjson` \*before\* calling any LLM. (Exception: peer-reviewed publication or newer arXiv version superseding an older preprint routes to the \*\*Theory-Upgrade Track\*\*.)

&#x20;  \* \*\*Prompt-Injection Isolation:\*\* Raw extracted text is wrapped inside randomized per-call boundary tags `<ore\_payload\_{uuid4}>...</ore\_payload\_{uuid4}>`; extraction LLMs have zero tool-call permissions.

&#x20;  \* \*\*Verbatim Quote Gate:\*\* Extracts candidate claim + `quote\_text` + `source\_locator`. Python canonicalizes both raw source and `quote\_text` (`canonicalize.py`: Unicode `NFKC` $\\rightarrow$ lowercase $\\rightarrow$ strip `%` LaTeX comments $\\rightarrow$ collapse `\\s+` to single space) and verifies normalized substring match ratio $\\ge 0.95$. Computes `quote\_sha256` over the canonicalized quote. Fail = immediate discard.

2\. \*\*Stage 2 — Formalizer (`formalizer.py`):\*\*

&#x20;  \* Emits structured JSON matching the v2.1 YAML frontmatter + Markdown sections + executable `verify.py` targeting an enabled `OraclePlugin`.

3\. \*\*Stage 3 — Anti-Cheating AST Linter \& Docker Oracle (`ast\_linter.py` + `oracle\_sandbox.py`):\*\*

&#x20;  \* Static Python AST linter \*\*hard-rejects\*\*: `assert True`, tautological comparisons (`x == x`), empty/bare `except:` handlers, forbidden imports (`os`, `sys`, `subprocess`, `socket`, `urllib`, `requests`, `http`, `pathlib`, `open`), or fewer than 3 independent symbolic/numeric assertions testing distinct inputs/boundary cases.

&#x20;  \* Runs `verify.py` in the air-gapped Docker sandbox. Up to \*\*2\*\* automated repair attempts using stderr feedback; persistent failure logs source hash into `ledger/rejected/YYYY-MM.ndjson` with `retry\_on\_model\_upgrade: true`.

4\. \*\*Stage 4 — Blinded Adversarial Skeptic (`skeptic.py`):\*\*

&#x20;  \* Uses the \*\*cross-family `skeptic` model\*\* (e.g., Gemma-4 challenging Gemini Flash/Pro).

&#x20;  \* Prompt strictly instructs: \*"Assume this formalization has a hidden defect or unstated assumption. Write a standalone `falsify.py` script that obeys every item in `assumptions\[]` while breaking the `Formal Statement`, OR cite exact missing hypotheses from the primary quote."\*

&#x20;  \* Executes `falsify.py` inside the same air-gapped sandbox (passes through the same forbidden-import AST linter). If `falsify.py` returns a machine-verified counterexample: Formalizer must tighten `assumptions\[]` and re-pass Stages 2–4 (max 1 tightening loop) OR discard the claim.

&#x20;  \* \*\*Deterministic Epistemic Tier Rules Engine (Non-LLM):\*\*

&#x20;    \* `T1\_CANONICAL`: Passes `sympy\_numeric` or `z3\_smt` oracle + peer-reviewed non-retracted source (age $\\ge 2$ yrs OR citations $\\ge 50$ OR canonical reference DB NIST/OEIS/DLMF) + survives Blinded Skeptic.

&#x20;    \* `T2\_ESTABLISHED`: Peer-reviewed non-retracted empirical result passing `empirical\_stats` sanity oracle with explicit uncertainty bounds + survives Blinded Skeptic.

&#x20;    \* `T3\_SPECULATIVE`: Unrefereed preprint, single uncorroborated study, or open conjecture. Hard-capped at `T3`; assigned `re\_evaluate\_after = +180 days`; \*\*forbidden\*\* as a `DEPENDS\_ON` prerequisite for any `T1` node.

&#x20;    \* `T4\_FALSIFIED`: Retracted results or machine-verified disproved conjectures/pathologies stored explicitly as `NEGATIVE\_RESULT` nodes linked via `BREAKS\_WITHOUT\_ASSUMPTION`.

5\. \*\*Stage 5 — Taxonomy Architect \& Verify-Before-Archive 2PC (`architect.py`):\*\*

&#x20;  \* Hybrid retrieval against SQLite (`FTS5` + `formal\_signature` Jaccard + consistent-space vector cosine).

&#x20;  \* \*\*Zero-Redundancy Merge:\*\* Cosine $\\ge 0.88$ (or exact `formal\_signature` + symbolic equivalence) $\\rightarrow$ append new source entry to existing node's `provenance\[]`; zero duplicate node created.

&#x20;  \* \*\*Conflict / Generalization:\*\* Execute Theory-Upgrade Protocol (Section 6.4).

&#x20;  \* \*\*Atomic 2PC Commit:\*\* ONLY now upload raw Ore from `/tmp/ore\_cache` to Google Drive `by-sha256/{aa}/{bb}/{sha256}.{ext}` $\\rightarrow$ verify remote Drive checksum $\\rightarrow$ write node + append NDJSON rows $\\rightarrow$ run blocking `validate\_node.py` \& `validate\_vault.py` $\\rightarrow$ push to Git.



\### 6.4 Append-Only Theory-Upgrade Protocol

When verified candidate $C\_{\\text{new}}$ conflicts with or improves existing node $N\_{\\text{old}}$:

1\. \*\*Preprint Firewall:\*\* If $\\text{tier}(C\_{\\text{new}}) > \\text{tier}(N\_{\\text{old}})$ (e.g., `T3` challenging `T1`/`T2`), \*\*never modify Active Consensus\*\*. Emit a `CHALLENGES` edge and append a one-line pointer under `## Active Challenges \& Open Anomalies`.

2\. \*\*Full Supersession Gate:\*\* Requires $\\text{tier}(C\_{\\text{new}}) \\le \\text{tier}(N\_{\\text{old}})$ AND passing in sandbox \*\*either\*\*:

&#x20;  \* \*Correspondence Test:\* $C\_{\\text{new}}$ reproduces $N\_{\\text{old}}$ across $N\_{\\text{old}}$'s valid domain while generalizing assumptions or tightening bounds, \*\*OR\*\*

&#x20;  \* \*Verified Falsification / Retraction:\* Crossref retraction record or sandbox-executed counterexample disproving $N\_{\\text{old}}$.

3\. \*\*Append-Only Mutation:\*\* Keep `ral\_id` immutable, increment `consensus\_version` ($v \\to v+1$), write new `## Active Consensus`, move the entire prior Active Consensus verbatim into `<details><summary>v{old} (YYYY-MM-DD to YYYY-MM-DD) | Reason: {ENUM}</summary>...</details>` under `## Superseded \& Archived`, and append a `SUPERSEDES` record to `graph/lineage/YYYY-MM.ndjson`.



\---



\## 7. Exact AI-Native Vault Schemas (v2.1)



\### 7.1 Canonical Text \& Hash Rules (`src/ral/canonicalize.py`)

\* `canonical\_text(s)`: `re.sub(r"\\s+", " ", re.sub(r"(?<!\\\\)%.\*", "", unicodedata.normalize("NFKC", s)).lower()).strip()`

\* `assumptions\_hash`: `"sha256:" + sha256("\\n".join(sorted(canonical\_text(a) for a in assumptions)))`

\* `quote\_sha256`: `"sha256:" + sha256(canonical\_text(quote\_text))`

\* `oracle\_script\_sha256`: `"sha256:" + sha256(verify\_py\_code.strip())`



\### 7.2 Canonical Node File (`vault/{macro}/{subpath}/{ral\_id}.md`)

```markdown

\---

ral\_schema: "2.1"

ral\_id: "RAL-PHYS-01J9X8K2M4"

node\_type: "THEOREM" # AXIOM | THEOREM | EMPIRICAL\_LAW | ALGORITHM | BRIDGE\_ISOMORPHISM | NEGATIVE\_RESULT

epistemic\_tier: "T1\_CANONICAL" # T1\_CANONICAL | T2\_ESTABLISHED | T3\_SPECULATIVE | T4\_FALSIFIED

consensus\_version: 2

taxonomy\_code: "phys.statistical-mechanics.thermodynamics"

title: "Fluctuation-Dissipation Theorem (Classical Linear Response)"

formal\_signature:

&#x20; - "operator:autocorrelation"

&#x20; - "operator:fourier\_transform"

&#x20; - "symmetry:time\_translation"

&#x20; - "regime:linear\_response"

assumptions:

&#x20; - "system is in canonical thermal equilibrium with temperature T > 0"

&#x20; - "external perturbation satisfies linear response |delta\_H| << k\_B \* T"

&#x20; - "classical high-temperature limit hbar \* omega << k\_B \* T"

assumptions\_hash: "sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"

verification:

&#x20; oracle\_plugin: "sympy\_numeric"

&#x20; oracle\_status: "PASS"

&#x20; oracle\_script\_sha256: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

&#x20; verified\_at: "2026-11-24T20:14:02Z"

&#x20; formalizer\_model: "gemini-3.8-flash"

&#x20; skeptic\_model: "gemma-4-27b-it"

&#x20; model\_capability\_epoch: 1

provenance:

&#x20; - source\_id: "arxiv:2104.01234v2"

&#x20;   source\_license: "arXiv-non-exclusive-1.0"

&#x20;   source\_format: "arxiv\_tex" # arxiv\_tex | pmc\_jats | oa\_pdf | reference\_db

&#x20;   source\_locator: "tex:sec2.3:eq14"

&#x20;   ore\_uri: "gdrive://RAL\_Ore\_Warehouse/by-sha256/4a/9c/4a9c01...f12.tar.gz"

&#x20;   ore\_sha256: "4a9c01...f12"

&#x20;   quote\_sha256: "sha256:8b1a9953c4611296a827abf8c47804d7"

&#x20;   quote\_char\_len: 214

&#x20;   pub\_year: 2021

&#x20;   peer\_reviewed: true

&#x20;   retracted: false

re\_evaluate\_after: "2028-11-24"

\---



\## Active Consensus (v2 — Effective 2026-11-24)



\### Formal Statement

$$\\chi''(\\omega) = \\frac{\\omega}{2 k\_B T} S\_{AA}(\\omega)$$



| Symbol | Meaning | Domain | SI Units |

| :--- | :--- | :--- | :--- |

| $\\chi''(\\omega)$ | Imaginary part of generalized susceptibility | $\\mathbb{R}$ | $\[A]^2 \\cdot \\mathrm{J}^{-1}$ |

| $S\_{AA}(\\omega)$ | Equilibrium power spectral density of observable $A$ | $\\mathbb{R}\_{\\ge 0}$ | $\[A]^2 \\cdot \\mathrm{s}$ |

| $k\_B T$ | Thermal energy scale | $\\mathbb{R}\_{>0}$ | $\\mathrm{J}$ |



\### Failure Boundaries \& Known Pathologies

\- \*\*Breaks when:\*\* Driven non-equilibrium steady states violating detailed balance (see `RAL-PHYS-01J9Z1P0Q2`); quantum regime $\\hbar\\omega \\gtrsim k\_B T$ requiring $\\coth(\\hbar\\omega / 2 k\_B T)$ correction.



\### Executable Verification Kernel

```python

\# RAL\_ORACLE: sympy\_numeric | TIMEOUT: 15s

import sympy as sp

import numpy as np



w, kT, hbar = sp.symbols("w kT hbar", positive=True, real=True)

quantum\_factor = hbar \* sp.coth(hbar \* w / (2 \* kT))

classical\_limit = sp.limit(quantum\_factor, hbar, 0)

assert sp.simplify(classical\_limit - (2 \* kT / w)) == 0



\# Boundary numerical verification across 5 decades in kT / (hbar \* w)

for ratio in np.logspace(1, 5, 50):

&#x20;   q\_val = 1.0 / np.tanh(1.0 / (2.0 \* ratio))

&#x20;   c\_val = 2.0 \* ratio

&#x20;   assert np.isclose(q\_val, c\_val, rtol=1e-2)



\# Verify divergence warning in quantum regime (ratio << 1)

assert abs((1.0 / np.tanh(1.0 / (2.0 \* 0.05))) - (2.0 \* 0.05)) > 0.8

```



\### Canonical Provenance Anchors

\- `arxiv:2104.01234v2` (`tex:sec2.3:eq14`): Classical linear-response limit of Callen–Welton FDT.



\## Active Challenges \& Open Anomalies

\- \*(None)\*



\## Superseded \& Archived

<details>

<summary>v1 (2026-04-10 to 2026-11-24) | Reason: DOMAIN\_GENERALIZATION</summary>



\*(Verbatim archived v1 Active Consensus block)\*

</details>

```



\### 7.3 Sharded NDJSON Schemas

\* \*\*Graph Edges (`graph/edges/YYYY-MM.ndjson`):\*\*

&#x20; ```json

&#x20; {"edge\_id":"E-01J9X8K9ZZ","src":"RAL-PHYS-01J9X8K2M4","dst":"RAL-ECON-01J9Y4M8B1","relation":"ISOMORPHIC\_TO","formal\_mapping":{"k\_B\*T":"liquidity\_temperature","chi''(omega)":"impact\_response\_kernel"},"confidence":0.89,"evidence\_sha256":"7c2f...81a","created\_at":"2026-11-24T22:41:10Z","created\_by\_model":"gemini-3.8-flash"}

&#x20; ```

&#x20; \*Allowed `relation` enum:\* `DEPENDS\_ON | GENERALIZES | INSTANTIATES | ISOMORPHIC\_TO | CONTRADICTS | CHALLENGES | SUPERSEDES | BREAKS\_WITHOUT\_ASSUMPTION`.

\* \*\*Source Ledger (`ledger/sources/YYYY-MM.ndjson`):\*\*

&#x20; ```json

&#x20; {"ore\_sha256":"4a9c01...f12","canonical\_doi":"10.1103/physrev.83.34","arxiv\_base\_id":"2104.01234","arxiv\_version":2,"source\_license":"arXiv-non-exclusive-1.0","ore\_uri":"gdrive://RAL\_Ore\_Warehouse/by-sha256/4a/9c/4a9c01...f12.tar.gz","linked\_ral\_ids":\["RAL-PHYS-01J9X8K2M4"],"ingested\_at":"2026-11-24T20:14:05Z"}

&#x20; ```

\* \*\*Rejected Ledger (`ledger/rejected/YYYY-MM.ndjson`):\*\*

&#x20; ```json

&#x20; {"source\_sha256":"9e12...aa4","canonical\_doi":null,"arxiv\_base\_id":"2610.99999","stage\_failed":"STAGE3\_ORACLE","reason":"AST\_CHEAT\_OR\_ASSERT\_FAIL","retry\_on\_model\_upgrade":true,"model\_capability\_epoch":1,"rejected\_at":"2026-11-24T21:05:11Z"}

&#x20; ```

\* \*\*Prerequisite Debt Ledger (`ledger/prereq\_debt.ndjson`):\*\*

&#x20; ```json

&#x20; {"stub\_id":"STUB-MATH-HAHN-BANACH-COMPLEX","domain":"math.analysis.functional","cited\_by":\["RAL-PHYS-01J9X8K2M4"],"attempt\_count":2,"last\_attempt\_night":14,"cooldown\_until\_night":18,"status":"OPEN"}

&#x20; ```



\---



\## 8. Blocking CI Validation Rules (`ci/validate\_node.py` \& `ci/validate\_vault.py`)



Every nightly push MUST pass all deterministic checks below; any failure aborts the commit:

1\. \*\*Schema \& Hash Integrity (`validate\_node.py`):\*\*

&#x20;  \* Pydantic strict validation of v2.1 YAML frontmatter.

&#x20;  \* Recomputes `assumptions\_hash` from `assumptions\[]` and `oracle\_script\_sha256` from the `Executable Verification Kernel` code block—exact match required.

&#x20;  \* Natural-language prose word count outside LaTeX/tables/code blocks \*\*$\\le 350$ words\*\*; rejects banned conversational filler phrases.

&#x20;  \* Re-runs `ast\_linter.py` on the embedded Python kernel.

2\. \*\*Public Repo \& Epistemic Hygiene (`validate\_vault.py`):\*\*

&#x20;  \* \*\*Open-Access License Gate:\*\* Every `provenance\[].source\_license` MUST exist in `config/license\_allowlist.yaml`.

&#x20;  \* \*\*Private Path \& Secret Leak Gate:\*\* Scans full diff and vault for high-entropy secrets, API key prefixes (`AIza`, `ghp\_`, `gho\_`, `ya29.`), OAuth JSON keys, and local machine paths (`/Users/`, `C:\\Users\\`, `/home/`).

&#x20;  \* \*\*Unique `ral\_id` \& Referential Integrity:\*\* Zero duplicate `ral\_id` across `vault/`; every edge `src` and `dst` resolves to an existing `ral\_id` (or registered `STUB-\*` in `prereq\_debt.ndjson`); every `ore\_sha256` exists in `ledger/sources/\*.ndjson`.

&#x20;  \* \*\*DAG Acyclicity \& Tier Firewall:\*\* `DEPENDS\_ON` subgraph MUST be a strict DAG (zero cycles via Kahn's algorithm); no `T1\_CANONICAL` node may have a `DEPENDS\_ON` edge pointing to a `T3\_SPECULATIVE` or `T4\_FALSIFIED` node.

&#x20;  \* \*\*Repo Size Telemetry:\*\* Logs `.git` size; flags critical alert if $> 700\\text{ MB}$.



\---



\## 9. Hardened Read-Only MCP Server (`mcp/ral\_mcp\_server.py`)



Ships inside the repository so future frontier agents (Gemini 4/5/6 / Antigravity) can mount RAL deterministically with one command:

\* \*\*Index Synchronization:\*\* On startup, fetches `ral-index-latest.json` from the `nightly-index` GitHub Release (or local offline cache), verifies SHA256 before decompressing `ral-index-YYYYMMDD-{run\_id}.sqlite.zst` to a local read-only SQLite file.

\* \*\*Strict Security Invariants:\*\*

&#x20; \* Opens database strictly read-only: `sqlite3.connect(f"file:{db\_path}?mode=ro", uri=True)`.

&#x20; \* Parameterized SQL queries only (zero string interpolation).

&#x20; \* Strict regex guard on all ID inputs: `^RAL-\[A-Z]{2,6}-\[0-9A-Z]{10}$`.

&#x20; \* Path resolution strictly anchored inside repo `vault/` via `Path.resolve().is\_relative\_to(VAULT\_ROOT)`.

\* \*\*Exposed Deterministic MCP Tools:\*\*

&#x20; 1. `ral\_search(query, formal\_signature\_tags=None, max\_tier="T2\_ESTABLISHED", limit=10)` — Hybrid FTS5 + `formal\_signature` Jaccard (+ vector search if query embedder configured); always returns `ral\_id`, `epistemic\_tier`, `taxonomy\_code`, and `consensus\_version`.

&#x20; 2. `ral\_get\_node(ral\_id)` — Returns canonical Markdown + \*\*automatically attaches\*\* all linked `BREAKS\_WITHOUT\_ASSUMPTION` (`NEGATIVE\_RESULT`) failure modes and explicit `WARNING` banner if `epistemic\_tier` is `T3\_SPECULATIVE` or `T4\_FALSIFIED`.

&#x20; 3. `ral\_get\_dependency\_cone(ral\_id, max\_depth=5)` — Traverses the acyclic `DEPENDS\_ON` DAG and returns the minimal prerequisite axiom chain in \*\*topological order\*\* (foundations first).

&#x20; 4. `ral\_find\_isomorphisms(domain\_a=None, domain\_b=None)` — Queries `ISOMORPHIC\_TO` edges and returns structured `formal\_mapping` variable dictionaries across sciences.



\---



\## 10. One-Time Human Setup Checklist (Zero Maintenance Post-Launch)



1\. \*\*Create Public GitHub Repository:\*\* Initialize `rayhans-autonomous-lab` as \*\*Public\*\*. Under \*Settings $\\rightarrow$ Rules $\\rightarrow$ Rulesets\*, protect `main` against \*\*force-pushes\*\* and \*\*deletion\*\* only (allow fast-forward pushes from Actions). Under \*Settings $\\rightarrow$ Actions $\\rightarrow$ General\*, enable \*Read and write permissions\*.

2\. \*\*Create Billing-Disabled GCP Project for Free Gemini:\*\* Create project `ral-free-epistemic-engine` with \*\*NO billing account linked\*\*. Enable Generative Language API, mint key, store as GitHub Actions secret `RAL\_GEMINI\_FREE\_KEY`.

3\. \*\*Configure Google Drive (`RAL\_Ore\_Warehouse`):\*\*

&#x20;  \* \*Option A (Workspace Shared Drive):\* Create Shared Drive `RAL\_Ore\_Warehouse`, add Service Account as `Content Manager`, store JSON key in secret `RAL\_GDRIVE\_SA\_JSON` and Drive ID in `RAL\_GDRIVE\_ROOT\_FOLDER\_ID`.

&#x20;  \* \*Option B (Personal 4.8 TB Drive):\* Create OAuth2 Desktop credentials in GCP, set OAuth Consent Screen Publishing Status to \*\*Production\*\* (prevents 7-day token expiration), authorize once with narrow `https://www.googleapis.com/auth/drive.file` scope, store `RAL\_GDRIVE\_CLIENT\_ID`, `RAL\_GDRIVE\_CLIENT\_SECRET`, `RAL\_GDRIVE\_REFRESH\_TOKEN`, `RAL\_GDRIVE\_ROOT\_FOLDER\_ID`.

4\. \*\*Optional Backup Cron Heartbeat (5-Minute Setup):\*\* Deploy a free Cloudflare Worker scheduled at `45 18 \* \* \*` UTC (`22:45 MUT`) that checks `GET /repos/{owner}/rayhans-autonomous-lab/actions/runs` and triggers `workflow\_dispatch` only if GitHub’s native cron failed to fire.



\---



\## 11. Antigravity Phased Implementation Milestones



| Phase | Deliverables | Mandatory Automated Acceptance Tests |

| :--- | :--- | :--- |

| \*\*M1: Foundation \& Security Rails\*\* | `config/\*.yaml`, `config/calibration\_suite.json`, `ci/sandbox.Dockerfile`, `src/ral/canonicalize.py`, `src/ral/pipeline/ast\_linter.py`, `src/ral/pipeline/oracle\_sandbox.py`, `ci/validate\_node.py`, `ci/validate\_vault.py`, `src/ral/storage/drive\_ore.py`. | (1) Sandbox blocks `import os; os.environ` \& network socket attempts; (2) AST linter rejects `assert True` \& bare `except:`; (3) `validate\_vault.py` rejects non-OA license, private `/Users/` path, and `DEPENDS\_ON` cycle; (4) Drive canary passes. |

| \*\*M2: Dynamic Swarm \& Epistemic Pipeline\*\* | `src/ral/preflight.py`, `src/ral/model\_router.py`, `src/ral/sources/\*`, `src/ral/pipeline/{scout,formalizer,skeptic,architect}.py`, `src/ral/pipeline/oracles/\*`. | (1) `model\_policy.yaml` blocks `gemini-4-argon` while allowing `gemini-2.5-pro`; (2) Calibration runs once and caches to `metrics/model\_benchmarks.json`; (3) Skeptic `falsify.py` forces assumption tightening; (4) `T3` preprint fails to overwrite `T1` consensus; (5) Valid upgrade archives v1 into `<details>`. |

| \*\*M3: Director, Index \& Maintenance\*\* | `src/ral/director.py`, `src/ral/storage/index\_builder.py`, `src/ral/github\_client.py`, `src/ral/maintenance.py`. | (1) `allocate\_epochs` passes unit tests for $K \\in \\{0..6\\}$ \& Day-0 ($|V\_t|=0 \\implies 6$ explore); (2) Prereq backoff quarantines stub at $k\_p=5$; (3) Embedding mismatch halts vector overwrite; (4) Blue-Green Release swap keeps 2 snapshots; (5) 429 RPD triggers graceful Zero-LLM maintenance pivot. |

| \*\*M4: Workflow Ignition \& MCP Server\*\* | `.github/workflows/ral\_nightly.yml`, `mcp/ral\_mcp\_server.py`, `/INDEX.md`, `/llms.txt`. | (1) Full 7-job dry run via `workflow\_dispatch` with `retention-days: 1` artifacts and <40 GitHub API calls; (2) MCP server passes read-only \& injection unit tests and serves topological `ral\_get\_dependency\_cone()`; (3) Dead-Man's Switch verified against simulated Drive canary failure. |

