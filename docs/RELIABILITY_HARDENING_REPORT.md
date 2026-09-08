# ScholarEval reliability hardening — implementation and acceptance report

Status: implemented and tested offline. Full live acceptance remains open because
the execution environment has no `S2_API_KEY`, `API_KEY`, or `API_ENDPOINT`.
The canonical full command was exercised and stopped in `BLOCKED_CONFIG` during
preflight, before retrieval or Codex startup. This is not a claim of production
readiness or a completed live evaluation.

## Architecture

Previously the CLI was a chain of batch subprocesses: any nonzero exit stopped the
pipeline, and recomputation could overwrite a completed artifact and its sidecar.
The [source audit and failure matrix](reliability-design.md) map the 17 stages and
their dependency boundaries.

The CLI now builds an explicit DAG and walks backward from requested final
artifacts. Valid committed downstream artifacts stop traversal. Pairwise comparison
also explicitly depends on contribution dimensions, even when relevance provides
a boundary for the literature-search branch.

`StageCheckpoint` runs each candidate in a unique `<output>.generations/gen-*`
directory. Scientific structure is validated, files are flushed, a generation
commit record is written, and the completed sidecar pointer is atomically replaced.
Compatibility output files are then materialized. If that final copy is interrupted,
resume repairs them from the committed generation. `<output>.attempt.json` stores
running/failure state separately. Old generations and old completed metadata remain
available. The canonical CLI holds an OS process lock on the save directory;
standalone stages lock their output location.

`ItemStore` uses SQLite transactions and content hashes for immutable completed
items. It backs relevance, methods/reference synthesis, query loops, meta-review,
pairwise comparison, embeddings, citation formatting, related-work extraction,
and retrieval operations. Corrupt item records are quarantined before replacement.
Retrieval's existing journal remains a compatible progress view.

`Supervisor` records categorized states, retries stage subprocesses with resume,
waits for quota, and recreates the managed Codex session on recovery. Ordinary
dependency recovery defaults to 24 hours per stage invocation. Quota waits default
to 14 days, starting at 15 minutes and rising to hourly retries. Repeated top-level
schema failures stop after three supervisor attempts. All limits are configurable;
there is no tight infinite retry loop.

S2 uses a persistent response cache keyed by method, URL/API version, parameters,
and payload. Its shared SQLite pacer reserves requests before HTTP, uses a lease
to serialize requests across processes, records global cooldowns, and supports
CLOSED/OPEN/HALF_OPEN circuit states. The interval starts at five seconds, increases
after throttling, and recovers gradually. Numeric and HTTP-date Retry-After values
are honored, including cooldowns surviving client or stage restarts. Individual
malformed records are counted and skipped; batch positions remain aligned using
null placeholders. Invalid top-level responses get sanitized diagnostic snapshots.

The official managed-auth Codex path and existing exact response-cache keys remain.
Exact response hits return zero new tokens, including while quota is exhausted.
Completed raw turns are additionally stored before format correction, so a later
correction failure does not require paying again for an already received raw turn.
New cache entries commit their text and SHA-256 checksum in one transaction.
Checksum mismatches or deletion of an attested entry stop with scientific validation
failure, preserving the database and avoiding a replacement inference. Legacy
entries retain their exact keys/text and acquire checksums when first read; this
detects subsequent alteration, not historical corruption before attestation.
There is no automatic paid inference fallback. The implementation follows the
[official app-server interface](https://learn.chatgpt.com/docs/app-server).

## Scientific behavior and fingerprint policy

Scientific prompts, relevance/novelty rubrics, model selection restrictions, and
normal scoring formulas were preserved. Prompt templates, explicit algorithm
versions, input hashes, cutoff, model/reasoning configuration, and scientific
parameters determine semantic identity. DAG boundaries include transitive semantic
dependencies. Retry counts, timeout, worker count, sleep intervals, concurrency,
logs, destination paths, and transport source changes do not invalidate results.
Algorithm changes require an explicit version bump; prompt template changes are
detected automatically. Citation-formatting prompt identity is also included in
the final-review semantics.

These intentional correctness fixes affect formerly faulty cases:

- Invalid relevance output no longer becomes score zero.
- Embedding failures no longer become zero vectors.
- Pairwise dimensions/scores and final Soundness structure are validated.
- Relevance output order follows its input order, preventing thread scheduling
  from changing score ties in paper sampling.
- Augmentation now applies an explicit cutoff to references as well as
  recommendations. Under a cutoff, undated or uninterpretable dates are excluded.
  Augmentation algorithm version is bumped; the migrated current run has no cutoff.
- An empty set of contribution comparisons produces an undetermined score instead
  of failing while formatting `None`. No-evidence meta-review Markdown is explicit.
- Optional related-work augmentation parses only its selected downloaded files;
  unrelated historical PDFs cannot enter the evidence set.

Embeddings still use the separately configured Titan Text Embeddings V2 service.
Replacing it with a local model would change paper selection and was not assumed.
`API_KEY`/`API_ENDPOINT` must explicitly identify that service; no default paid
OpenAI endpoint is selected. The managed Codex backend does not provide embeddings.

## Existing-run migration

Pre-change snapshot: `.reliability-snapshots/20260907-164012/`.
It contains source/checkpoint/progress copies, a consistent SQLite backup of the
Codex cache, and hashes of PDF/XML files. No old state was deleted.

For `demo_data/astra-test-1`:

| Preservation check | Result |
| --- | --- |
| Completed checkpoints migrated | 6 |
| Completed augmentation retrieval operations retained | 9 |
| Scientific/artifact/PDF/XML files hash-verified | 86; zero mismatches |
| Original Codex response entries | 193 before and after; contents identical |
| Assessed contribution papers | 129 |
| Papers entering augmentation at relevance >= 3 | 56 |
| Scientific output bytes changed by migration | 0 |

The migrated checkpoints are method extraction, Soundness query generation,
final Soundness, contribution dimensions, contribution queries, and initial
contribution relevance. Damaged methods analysis and unverified historical
checkpoints remain preserved but were not falsely declared successful.

Each replaced sidecar has an adjacent `.legacy-*` copy. The migration verifies
pre-change checkpoint validity, unchanged prompt templates, output hashes, and
current structural validity. Existing raw retrieval results are retained with
their hashes. PDF/XML progress remains in place and uses content hashes, rather
than timeout/worker settings, to decide reuse.

Machine-readable evidence is in the run directory:

- `reliability-migration.json`
- `reliability-verification.json`
- `reliability-tests.json`
- `dry-run.json`
- `workflow.sqlite3` and `run-summary.json`

Migration/verification commands, already executed for this run:

```powershell
.\.venv\Scripts\python.exe scripts/migrate_reliability_state.py --snapshot .reliability-snapshots/20260907-164012 --run demo_data/astra-test-1
.\.venv\Scripts\python.exe scripts/verify_reliability_migration.py --snapshot .reliability-snapshots/20260907-164012 --run demo_data/astra-test-1
```

Do not import arbitrary legacy artifacts as successful. A previously recorded,
hash-verified snapshot is required by the migration tool.

## Test report

Windows, Python 3.13.14 in `.venv`: **117 tests passed in 32.866 seconds**. `scikit-learn` was missing
from both the requirements and this environment; it was added and installed.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m compileall -q ScholarEval scripts tests
```

The unit/fault tests cover atomic-write interruption, invalid generation rejection,
old-generation preservation, interruption after pointer commit before multi-file
materialization, semantic/operational fingerprint separation, item corruption,
per-item quota resume, Retry-After parsing, circuit state transitions, persistent
S2 cache identity, sanitized anomaly snapshots, reference pagination, malformed
records and positional batches, exact Codex cache hits during quota, cache alteration
and legacy attestation, terminal state propagation after retry exhaustion,
categorized no-LLM refusal, and explicit UTF-8 file I/O. Existing tests cover 37 candidates/31 full texts/6 unavailable,
isolated parsing errors, systemic GROBID errors, and Codex protocol/transport faults.

The integrated chaos test runs **all 17 actual stage main functions** through the
new checkpoint wrapper and supervisor, with fake S2, Codex protocol processes,
embeddings, and Docker/GROBID dependencies. It injects 20 HTTP 429s, three 503s,
malformed references, three unavailable PDFs, a managed-container restart, a Codex
disconnect, quota exhaustion, and a real child-process kill during relevance.
It completes both final reviews and a second DAG traversal requires no stages.
Separately, a killed five-item worker reuses exactly its three committed items.

**Test-isolation incident:** an early integration-test attempt failed to mock the
citation helper's separate LLM constructor and made one real managed-Codex call
using synthetic fixture text. No paid API fallback or production-run cache was
used. Its temporary-run metrics imply 4,592 input and 118 output tokens after
subtracting the known fake-call counts. The helper is now covered by the fake
engine, and an explicit offline guard prevents real app-server startup. A regression
test verifies that guard. The subsequent passing chaos runs used fake services.

The real saved-directory dry run made zero network/inference calls and selected
only `c5,c6,c7,c8,c9,c10,c11`. Final Soundness, initial relevance, and separately
required dimensions formed valid boundaries. The canonical live command then
stopped correctly in preflight on the missing S2 key; a live full run has not been
completed. Tests of recovery use simulated clocks and do not establish real
service availability, actual quota reset timing, or model scientific quality.

## Normal command and behavior

Configure `S2_API_KEY` and the existing embedding service in the launching shell.
Keep credentials out of source files and reports. The normal command is:

```powershell
.\.venv\Scripts\python.exe -m ScholarEval.ScholarEval --research_idea test_idea.txt --llm_engine_name auto --save_to demo_data/astra-test-1 --resume
```

Append `--dry-run` for offline DAG, checkpoint, and local preflight inspection.
`--no-llm` prohibits inference stages without completed checkpoints and does not
fall through to embeddings. `--stop-after-retrieval` selects Soundness retrieval
as the target, independent of whether a final Soundness artifact exists.

| Situation | Expected behavior |
| --- | --- |
| S2 429 | Local retries plus shared adaptive cooldown; exhausted local budget becomes WAITING_RATE_LIMIT; stage resumes its items |
| Codex quota | WAITING_QUOTA; persist progress, wait slowly, recreate session, retry next pending work; exact cache hits remain usable |
| Codex disconnect | RETRYABLE_FAILED; recreate run session; resume items/cache |
| GROBID unhealthy | Restart only the named compatible ScholarEval container; retry pending parses; never remove user containers |
| Unavailable PDF | Retain metadata/abstract/snippet evidence and explicit availability/failure information under available-evidence |
| Temporary PDF host failure | Bounded download attempts; record PDF_DOWNLOAD_RETRYABLE and a retry time; later retrieval attempts can retry |
| Ctrl+C | CANCELLED; preserve old generation and committed items |
| Process kill | OS releases locks; abandoned RUNNING child is retryable; rerun the same command after parent death |
| Bad final structure | FATAL_VALIDATION; candidate stays unpromoted with failure diagnostics |

Successful evidence generations intentionally freeze their observed availability.
They are not silently refreshed merely because an optional PDF might later appear.
S2's response cache similarly keeps successful request snapshots for reproducibility;
use a separate run/cache location when intentionally collecting a fresh corpus.

Operational settings include `SCHOLAREVAL_RECOVERY_MAX_SECONDS` (86400),
`SCHOLAREVAL_QUOTA_MAX_WAIT_SECONDS` (1209600), `SCHOLAREVAL_RETRY_SECONDS` (30),
`SCHOLAREVAL_QUOTA_POLL_SECONDS` (900), `SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS` (5),
`SCHOLAREVAL_S2_MAX_RETRIES` (6), `SCHOLAREVAL_GROBID_WORKERS` (1), and
`SCHOLAREVAL_PDF_MAX_ATTEMPTS` (3). GROBID's existing configured timeout is retained
unless `SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS` explicitly overrides it.

Inspect `run-summary.json` for states and metrics, `workflow.sqlite3` for events and
items, `diagnostics/s2-*.json` for sanitized HTTP anomalies, and generation
`failure.json` for exception type/message and stack locations without locals.

## Remaining limits and stopping conditions

- Full live acceptance still requires valid S2 and embedding credentials and a
  controlled Astra-backed run. The legacy Streamlit orchestrator is not covered
  by the new CLI DAG acceptance test; its syntax/UTF-8 fixes do not migrate its UI.
- Revoked credentials, unavailable model/account access, invalid configuration,
  missing dependencies, an occupied incompatible Docker port, disk exhaustion,
  unreadable/corrupt required input, or scientific validation failure can stop work.
- A permanently unavailable dependency, a quota that never resets, or repeated
  breaking upstream schema changes eventually exhaust recovery budgets.
- Recovery cannot guarantee exactly-once remote inference if the server completed
  a response that was never received locally. Already received and durably committed
  responses/items are reused. Cloud model revisions and provider behavior are not
  controlled by ScholarEval.
- SQLite/atomic replacement rely on a working local filesystem. Hardware loss,
  external deletion, and corruption of all retained generations require backups.
  Generations are intentionally retained and consume disk space; no automatic
  deletion policy was introduced.
- S2 pacing coordinates only processes sharing the local pacer database. Requests
  made by other applications or machines can still exhaust a shared provider limit.
- Structural tests do not prove scientific correctness of an LLM's claims.
  Missing evidence remains a recorded limitation, never proof of novelty or validity.

## File-by-file change inventory

Paths are relative to the repository root. Existing uncommitted edits in the four
originally modified files were preserved or incorporated into the new design.

| File | Purpose / behavior change |
| --- | --- |
| `.gitignore` | Exclude private reliability snapshots and local caches |
| `README.md` | Canonical durable CLI command and configuration/report links |
| `requirements.txt` | Declare previously missing scikit-learn dependency |
| `ScholarEval/ScholarEval.py` | Replace batch-script orchestration with durable CLI entry point and categorized startup errors |
| `ScholarEval/workflow.py` | New 17-node DAG, dependency boundaries, preflight, supervisor, recovery budgets, run telemetry |
| `ScholarEval/ScholarEval_app.py` | Explicit UTF-8 and repair pre-existing malformed except/finally syntax; legacy UI orchestration retained |
| `ScholarEval/utils/durable.py` | Atomic binary/text/JSON writes, SQLite state/events/metrics, immutable item store, OS locks |
| `ScholarEval/utils/workflow_errors.py` | Shared states/exit codes, existing-exception adapters, sanitized diagnostics |
| `ScholarEval/utils/checkpoints.py` | Generation transactions, completed-pointer recovery, scientific validators, semantic fingerprints |
| `ScholarEval/utils/retrieval_http.py` | Persistent S2 cache, adaptive shared pacing/leases/circuit, retries, tolerant list parsing, anomaly snapshots |
| `ScholarEval/utils/retrieval_progress.py` | Back existing journals with common item storage; preserve stale journals; allow corrected auth on resume |
| `ScholarEval/utils/semantic_scholar.py` | Preserve metadata reuse fixes; tolerant references; pagination alignment; batch chunking; safe authors |
| `ScholarEval/utils/grobid.py` | Owned-container recovery, atomic TEI files, retryable transient parses, UTF-8 Docker output, metrics |
| `ScholarEval/utils/pdf_utils.py` | Atomic PDFs, explicit availability/retry states, conservative concurrency, bounded retries, disk-error propagation |
| `ScholarEval/utils/citation_check.py` | Durable item caching for both citation-formatting entry points |
| `ScholarEval/engine/codex_app_server_engine.py` | Preserve exact keys; cache during quota; durable raw turns and transactional checksums; telemetry; offline startup guard |
| `ScholarEval/engine/litellm_engine.py` | Enforce offline/no-inference guard on explicitly selected API backend |
| `ScholarEval/soundness/extract_methods.py` | Explicit UTF-8 artifact and cost-log I/O |
| `ScholarEval/soundness/make_queries.py` | Durable method query items and UTF-8 |
| `ScholarEval/soundness/snippet_search.py` | Keep parse journal outside candidate generations; reuse PDF/XML hashes across operational tuning |
| `ScholarEval/soundness/methods_and_results_synthesis.py` | Durable method/reference items, complete method coverage, propagate failures, conservative Codex workers, UTF-8 |
| `ScholarEval/soundness/meta_review.py` | Durable per-method reviews, deterministic citation order, batched bibliography metadata, explicit no-evidence output, UTF-8 |
| `ScholarEval/soundness/tldr_soundness.py` | Validate structured summary before rendering; UTF-8 cost/output I/O |
| `ScholarEval/contribution/extract_dimensions_and_contributions.py` | Explicit UTF-8 cost logging |
| `ScholarEval/contribution/queries_generator.py` | Durable per-contribution query items and UTF-8 |
| `ScholarEval/contribution/relevance_assessor.py` | Replace unscoped partial-file recovery with durable validated items; no fabricated scores; deterministic ordering |
| `ScholarEval/contribution/paper_augmentation.py` | Preserve returned metadata reuse, stable retrieval journal, bounded selected-PDF parsing, per-XML items, cutoff enforcement, causal failures, UTF-8 |
| `ScholarEval/contribution/embedding_filter.py` | Durable validated embeddings, explicit endpoint requirement, typed transient failures, no zero-vector substitution |
| `ScholarEval/contribution/pairwise_comparator.py` | Durable validated paper comparisons, failure propagation, deterministic output, UTF-8 |
| `ScholarEval/contribution/prepare_final_contribution_context.py` | Explicit UTF-8 I/O |
| `ScholarEval/contribution/contribution_review_synthesis.py` | UTF-8 logs and correct handling of no comparable dimensions |
| `scripts/check_retrieval_backend.py` | Handle classified missing-Docker configuration errors |
| `scripts/snapshot_reliability_state.py` | Non-destructive snapshot and baseline checkpoint audit |
| `scripts/migrate_reliability_state.py` | Verified, repeatable legacy migration with backups and no re-attestation of partial data |
| `scripts/verify_reliability_migration.py` | Compare retained file/cache contents and relevance counts to snapshot |
| `tests/test_codex_backend.py` | Quota-at-startup regression updated for cache-first pause behavior |
| `tests/test_retrieval_reliability.py` | Update expected generation repair, DAG boundary, malformed-record, and corrected-auth behavior |
| `tests/test_durable_workflow.py` | New transaction/item/pacer/quota/UTF-8/kill tests and actual 17-stage chaos integration |
| `docs/reliability-design.md` | Source architecture audit, failure matrix, and redesign policy |
| `docs/RETRIEVAL_RELIABILITY.md` | Updated retrieval/resume operating guide |
| `docs/RELIABILITY_HARDENING_REPORT.md` | This implementation, migration, test, limitations, and change report |
