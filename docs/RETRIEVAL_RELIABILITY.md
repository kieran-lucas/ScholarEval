# Retrieval reliability and resume

The existing Codex app-server / ChatGPT Plus / Astra backend, prompts, extraction
section selection, contribution scoring, and scientific retrieval sources are
unchanged. No new search source or model fallback was added.

## Semantic Scholar

`S2_API_KEY` is optional for public access and recommended for reliable use. It is
sent in the `x-api-key` header; credentials are never printed or stored in stage
metadata. Anonymous requests use best effort / shared public rate limits. Some
endpoints can require authentication. The official service describes an
introductory authenticated limit of one request per second:
[Semantic Scholar API](https://www.semanticscholar.org/product/api).

PowerShell, in the same terminal that runs ScholarEval:

```powershell
$env:S2_API_KEY = '<your-key>'
$env:SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS = '1.1'
$env:SCHOLAREVAL_S2_MAX_RETRIES = '6'
```

Every S2 endpoint in the existing wrapper uses the same HTTP policy. A SQLite lock
in the OS temporary directory serializes requests across local ScholarEval
processes, including contribution stages. Monotonic start timestamps enforce a
minimum 1.1-second interval by default. `SCHOLAREVAL_S2_RATE_DB` can override the
local database path; all processes must use the same path to share the limiter.
This database contains only a timestamp, never a key. It does not coordinate
requests made from other computers or applications.

429, 500, 502, 503, 504, connection errors and timeouts get at most six retries
(seven attempts total). Backoff is `min(60, 2 * 2**retry_index) + uniform(0, 1)`
seconds. A valid numeric or HTTP-date `Retry-After` raises the wait to at least the
server's requested delay. If that delay exceeds 180 seconds, the operation fails
retryably immediately instead of retrying early or waiting indefinitely. Request
timeouts are 30 seconds, or 60 for snippet search. There is no retry for 400,
401/403, malformed JSON, or invalid response schemas.

Failures raise typed exceptions and stop the stage. Only a validated successful
response can produce an empty list. Exhausted 429 can never mean "no prior art."

## Docker and GROBID

On Windows, install Docker Desktop and keep its Linux-container daemon running.
The existing image remains `lfoppiano/grobid:latest-crf`, with host port 8070
mapped to container port 8070. `GROBID_config.json` selects the server URL.

ScholarEval reuses a compatible running container on that port, or starts a
compatible stopped container. If none exists, it creates `scholareval-grobid`.
An incompatible container, occupied port, or conflicting container name causes
a clear failure; ScholarEval never kills or removes user containers.

After starting/reusing the container, ScholarEval polls `/api/isalive` until HTTP
200 with body `true`, then constructs the Python client. Defaults:

```powershell
$env:SCHOLAREVAL_GROBID_READY_TIMEOUT_SECONDS = '180'
$env:SCHOLAREVAL_GROBID_POLL_SECONDS = '3'
```

See the documented [GROBID service checks](https://grobid.readthedocs.io/en/latest/Grobid-service/).
Timeouts report container state, exit code, port bindings, and at most 20 log
lines / 4,000 characters. Exit 137 or `OOMKilled` suggests Docker memory pressure;
increase Docker Desktop's memory allocation and inspect its resource usage.
There are no automatic restart loops.

Zero PDFs means no Docker startup, no client construction, and no localhost
request. Search/metadata HTTP or network failures still stop retrieval; they never
mean "no prior art." Individual exhausted PDF downloads are `full_text_unavailable`.
Every candidate remains in `snippet_papers.json`, with its corpus ID, S2 paper ID,
metadata, abstract, snippets and their source/citation provenance, attempted URLs,
download history, failure reason, and separate download/parse/evidence statuses.

The default `SCHOLAREVAL_FULL_TEXT_POLICY=available-evidence` permits partial
coverage, including metadata/abstract/snippet evidence when no full texts can be
obtained. `require-full-text` optionally requires **at least one successfully parsed
document**, never 100% or an arbitrary percentage. Successful empty searches are
explicitly `no_candidates`; neither empty results nor coverage establish scientific
validity or absence of prior art. `snippet_coverage.json` reports discovered,
downloaded, parsed, unavailable, parse-failed and metadata-only counts and the policy.
The stage checkpoint validates this report and requires every reference in the
candidate artifact. Downstream synthesis receives evidence limitations explicitly.

Downloads reuse readable existing PDFs. `pdfs/download_progress.json` bounds
fallback sequences per paper/URL across resumes, using
`SCHOLAREVAL_PDF_MAX_ATTEMPTS` (default 1). Each sequence retains the existing DOI,
aiohttp and requests fallbacks. Raising the limit deliberately permits additional
attempts; ordinary resume does not retry exhausted PDFs. Legacy files without a
download journal are reused when valid; missing files get one bounded sequence
because historical per-paper failures were not journaled.

Available PDFs proceed to GROBID after the download summary. Individual failures
are `parse_failed`, with metadata evidence retained. `snippet_parse_progress.json`
saves results immediately and reuses parsed XML or document failures when PDF and
parser/config hashes match. GROBID 503 retries are bounded to two calls per document.
Unhealthy service or all documents returning infrastructure errors fails the stage
with failed parses marked retryable. Successfully parsed documents remain saved.
Transport exceptions are classified before the installed client's broad IOError
handler can incorrectly turn a read timeout into a permanent HTTP 400 failure.
For small Docker memory allocations, set `SCHOLAREVAL_GROBID_WORKERS=1` (1..10;
default 10) and `SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS=300` (positive, at most
3600; otherwise the GROBID config timeout applies). These settings are checkpointed.

The old downloader explicitly used `verify=False` in the requests fallback and
`ssl=False` in aiohttp/DOI/Unpaywall paths, apparently as blanket compatibility for
difficult sites; the code gave no certificate-specific justification. The current
fork has restored verification in all these paths. No warnings are suppressed and
there is no insecure fallback. Certificate errors are recorded as download failures;
configure a trusted CA bundle (for requests, `REQUESTS_CA_BUNDLE`) if a legitimate
institutional proxy requires it.

## Resume

The default CLI behavior still recomputes stages. Opt in with `--resume`:

```powershell
.venv/Scripts/python.exe -m ScholarEval.ScholarEval --research_idea test_idea.txt --llm_engine_name auto --save_to demo_data/my-run --resume
```

Every stage subprocess checks its own checkpoint before entering its main
function or constructing an LLM client. Standalone stage commands also accept
`--resume`. The Streamlit sidebar accepts an existing **stage directory** for
resume (for example `demo_data/my-run/soundness`). Use one writer per run directory.

`<output>.stage.json` records completion, input fingerprint, output hashes,
completion time, and provenance. Checks include:

- JSON/JSONL parses, expected types/fields, query coverage of every method,
  and applicable downstream coverage checks.
- SHA-256 of source inputs, relevant stage source/helper versions, model and
  reasoning configuration, and applicable cutoff/GROBID configuration.
- Matching hashes for every declared output. Empty methods, missing queries,
  error records, truncated artifacts, and interrupted stages are not reusable.

Infrastructure helper changes do not invalidate upstream extraction/query
checkpoints. Changing the idea, methods, prompts/stage code, or relevant model
configuration invalidates affected stages. The Codex backend's legacy
`--llm_engine_name` argument is ignored in fingerprints just as it is by that
backend; the actual configured Codex model and reasoning are fingerprinted.

Search progress is an atomic JSON journal with `pending`, `succeeded`,
`failed-retryable`, and `failed-terminal` items, attempt counts, result hashes,
returned IDs, metrics, and timestamps. It saves on every attempt and completed
query/operation. Resume reuses successful items and retries pending/retryable
items. Terminal items require correcting configuration and running that retrieval
stage once **without** `--resume`. Query deduplication is deterministic. Legacy
monitor-only progress and existence-only references are never trusted.

Legacy methods/queries have no historical fingerprint. To reuse an explicitly
verified successful run, attest that they came from the supplied saved plan and
current configuration. The import validates both artifacts, the matching saved
plan, and cost records; it never imports old references or modifies the artifacts:

```powershell
$env:SCHOLAREVAL_LLM_BACKEND = 'codex'
$env:SCHOLAREVAL_CODEX_MODEL = 'auto'
$env:SCHOLAREVAL_CODEX_REASONING = 'high'
.venv/Scripts/python.exe scripts/import_legacy_checkpoints.py --soundness-dir demo_data/astra-test-1/soundness --research-idea demo_data/astra-test-1/soundness/research_plan.txt --llm-engine-name auto --attest-successful-run
```

This is a one-time operation and refuses to overwrite existing sidecars. It
explicitly records that the historical prompt hash is unavailable. The known
successful `astra-test-1` methods/queries were imported during this reliability
pass; do not import them again.

## Retrieval-only smoke procedure

No command in this block invokes Astra:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/check_retrieval_backend.py --no-network
.venv/Scripts/python.exe scripts/check_retrieval_backend.py --check-s2 --start-grobid
.venv/Scripts/python.exe -m ScholarEval.soundness.snippet_search --queries_file demo_data/astra-test-1/soundness/queries.json --methods_file demo_data/astra-test-1/soundness/methods.json --output_file demo_data/astra-test-1/soundness/snippet --pdf_dir demo_data/astra-test-1/soundness/pdfs --progress_file demo_data/astra-test-1/soundness/snippet_progress.json --resume
```

Preflight is read-only by default. `--check-s2` makes at most one cheap request,
with retries disabled. `--start-grobid` explicitly starts/reuses and waits for the
service. `--pull-image` permits pulling if the expected image is absent.
`--no-network` forbids external requests/pulls but permits local Docker and health
checks. A local-ready result without `--check-s2` does not prove S2 accessibility.
Even a passing metadata request cannot guarantee snippet endpoint access.

Only after direct retrieval succeeds, verify top-level resume:

```powershell
$env:SCHOLAREVAL_LLM_BACKEND = 'codex'
$env:SCHOLAREVAL_CODEX_MODEL = 'auto'
$env:SCHOLAREVAL_CODEX_REASONING = 'high'
.venv/Scripts/python.exe -m ScholarEval.ScholarEval --research_idea demo_data/astra-test-1/soundness/research_plan.txt --llm_engine_name auto --save_to demo_data/astra-test-1 --resume --no-llm --stop-after-retrieval
```

Expect `method extraction skipped` and `query generation skipped`. `--no-llm`
blocks stages requiring uncached model work, including embedding filtering;
it also prevents startup of the parent Codex session. `--stop-after-retrieval`
stops before soundness synthesis and the contribution pipeline.

For the first complete idea evaluation, use the same command with `--resume`
but remove `--no-llm --stop-after-retrieval`. Downstream evaluation will then use
Astra. The original contribution embedding stage still requires its separate
Titan/API configuration; the working Codex integration does not replace it.

## Metrics and debugging

`snippet_metrics.json`, contribution `*.metrics.json`, and retrieval progress
contain S2 attempts/successes/retries/429s/wait seconds, completed queries, unique
papers, PDF coverage, and GROBID action/startup seconds where applicable.
Progress metrics accumulate across resumes with matching inputs. A request
interrupted after server success but before its atomic local completion may be
repeated; exactly-once remote delivery cannot be guaranteed.

The existing `soundness_costs.jsonl`, `contribution_costs.jsonl`, and Codex
usage/checkpoint logging remain separate and unchanged. Retrieval metrics are
under a `retrieval` key; they do not claim LLM calls or convert tokens to Plus
allowance. A skipped stage writes no new LLM usage entry.

| Message | Meaning / action |
| --- | --- |
| 429 exhausted | Search did not complete. Configure/verify S2 key or wait, then resume. |
| 401/403 | Key invalid or endpoint not permitted. Fix access, then restart the failed retrieval stage without resume. |
| GROBID startup timeout | Service never passed its health check; inspect bounded diagnostics. |
| Docker unavailable | CLI missing or Docker Desktop daemon stopped/unreachable. |
| Exit 137 / OOM | Possible Docker memory exhaustion; adjust resources and inspect logs. |
| Stale checkpoint | Inputs/configuration/source changed, output changed, or completion cannot be proved. Recompute; no-LLM mode blocks this before spending tokens. |
| PDF/XML incomplete | Search may have succeeded, but reading coverage is incomplete. Resume retries infrastructure. |

See [the reliability audit](RETRIEVAL_RELIABILITY_AUDIT.md) for the observed live
results and remaining blockers from this pass.
