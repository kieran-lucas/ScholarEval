# Retrieval reliability and resume

The canonical CLI uses the durable DAG and supervisor described in the
[hardening report](RELIABILITY_HARDENING_REPORT.md). That report contains the
migration record, failure matrix, tests, and remaining live acceptance work.

## One-command operation

Configure `S2_API_KEY` and the explicitly selected Titan embedding service's
`API_KEY`/`API_ENDPOINT` in the launching PowerShell session, then run:

```powershell
.\.venv\Scripts\python.exe -m ScholarEval.ScholarEval --research_idea test_idea.txt --llm_engine_name auto --save_to demo_data/astra-test-1 --resume
```

The CLI defaults to managed Codex/Astra. No paid inference fallback is selected.
Append `--dry-run` to inspect the DAG and local state without network or model
calls. `--no-llm --stop-after-retrieval` allows retrieval only when prerequisite
inference stages already have valid checkpoints.

## Semantic Scholar

The full workflow requires the S2 key; standalone client use still allows anonymous
access. The key is sent in the request header, never stored in request identity
or diagnostic headers. All wrapper endpoints use the shared HTTP policy.

`SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS` defaults to 5 seconds. A shared SQLite pacer
reserves requests before sending, serializes them with expiring leases, persists
global cooldowns, and uses CLOSED/OPEN/HALF_OPEN circuit states. 429s increase
spacing; sustained successes gradually recover it. `SCHOLAREVAL_S2_RATE_DB`
overrides the local pacer location. Other machines/apps are not coordinated.

`SCHOLAREVAL_S2_MAX_RETRIES` defaults to 6 local retries. 429, 5xx, timeout, and
connection failures escalate to categorized supervisor waits after local retries.
Numeric and HTTP-date Retry-After values are honored. Long cooldowns remain
recorded when control returns to the supervisor. Malformed top-level JSON/schema
gets a sanitized diagnostic and bounded retries; three repeated supervisor schema
failures stop for investigation.

Successful responses, including empty results, are cached in `s2-responses.sqlite3`
by method, API URL/version, parameters and payload. Operational tuning does not
invalidate the cache. Malformed individual records are skipped and counted. Batch
results retain null positions; reference pagination uses the original page count.
Recommendations/references reuse returned metadata; bibliography metadata is
batched. Failed requests never become successful empty scientific results.

## PDFs and GROBID

PDFs and TEI XML use atomic writes. Existing readable PDFs and matching PDF/XML
hash pairs are reused. PDF fallback sequences default to three attempts
(`SCHOLAREVAL_PDF_MAX_ATTEMPTS`), with bounded backoff. Temporary host failures
carry `PDF_DOWNLOAD_RETRYABLE` and a retry time. Terminal/unavailable PDFs remain
distinct from metadata evidence. Completed evidence generations freeze coverage;
resume does not silently refresh a completed corpus.

The default `SCHOLAREVAL_FULL_TEXT_POLICY=available-evidence` retains metadata,
abstracts, snippets and citation provenance when full text is unavailable.
`require-full-text` in Soundness requires at least one parsed document, as before.
Coverage never establishes absence of prior art or scientific validity.

GROBID readiness uses `/api/isalive`, HTTP 200 and body `true`. The existing image
is `lfoppiano/grobid:latest-crf`. ScholarEval can start compatible containers or
create `scholareval-grobid`, but automatic restart is limited to that named,
compatible container. Unrelated containers are never removed or restarted.
Transient parsing failures remain retryable and successful parses are retained.
Default concurrency is one; timeout/worker overrides are operational and do not
invalidate completed scientific outputs. Cold preflight checks required GROBID
readiness before inference. A parsing call with zero PDFs starts no Docker/client.

## State and diagnostics

`workflow.sqlite3` stores items, states, metrics and events. `run-summary.json`
provides a machine summary. Immutable generations and an atomic completed pointer
are separate from attempt state. Interrupted output materialization is repaired
from the committed generation on resume.

S2 anomalies are in `diagnostics/s2-*.json`. Failed generations contain
`failure.json` with sanitized message/type and stack locations, without locals.
Existing progress journals and the exact Codex response cache are retained.

Default recovery budget: 24 hours per stage invocation; quota pauses: up to 14 days
with slow polling. Revoked credentials, invalid configuration, scientific validation
failures, permanent outages, disk problems or exhausted recovery budgets can still
require intervention. See the report for settings and preservation limits.
