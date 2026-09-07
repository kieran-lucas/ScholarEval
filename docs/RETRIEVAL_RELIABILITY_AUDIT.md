# Reliability pass: 2026-09-06

Branch: `feat/codex-plus-astra-backend`. Existing uncommitted Codex backend work
was present before this task and was preserved. No commit, push, benchmark,
model fallback, or live Astra inference was performed.

## Source map before editing

| Area | Observed behavior |
| --- | --- |
| CLI orchestration | Sequential subprocesses; no stage resume; successful stdout hidden. Copies plan and reruns upstream stages after any later failure. |
| Streamlit orchestration | Creates timestamped directories each click; progress monitors do not resume. Reading child pipes only after process exit risks a full-pipe deadlock. |
| Method extraction | Writes `methods.json` after a successful LLM response, then overwrites it on rerun. No schema gate or input provenance. |
| Query generation | Writes `queries.json` after all method queries; no resume or provenance. The supplied Astra artifacts are nonempty and cover all three methods. |
| Snippet search | Loads existing references using only existence. Otherwise catches all search failures and assigns empty method lists. Progress is display-only, overwritten, never loaded. |
| Metadata retrieval | Existing batches of 500 IDs, but no status/schema validation in the batch wrapper. IDs deduplicated with unordered sets. |
| S2 HTTP | All API calls live in `utils/semantic_scholar.py`; sporadic blind sleeps and inconsistent status checks. Snippet 429 returns `None`; only 504 has one special retry. No `Retry-After` support. |
| Credentials | Existing convention is `S2_API_KEY` with `x-api-key`; one bibliography helper uses `S2_API_KEY_2`, and title match mentions obsolete `s2_key` in an error. |
| PDF downloader | Existing download fallbacks remain intact. File existence alone was enough to reuse a downloaded PDF. |
| GROBID lifecycle | Snippet and augmentation stages start `docker run -d --rm -p 8070:8070 lfoppiano/grobid:latest-crf` before discovering whether PDFs exist. Container presence is treated as readiness. |
| GROBID waits | Snippet waits indefinitely for any image-matching container, then constructs the client immediately. Augmentation's readiness loop starts after client construction and is also unbounded. |
| Zero PDFs | Snippet still creates the client and later divides XML elapsed time by zero. Default contribution augmentation starts Docker despite using API references rather than PDFs. |
| Contribution search | `paper_extractor` catches failed queries and continues. Augmentation catches failed recommendation/reference calls and emits partial results. |
| Later outputs | Analysis/review JSON, contribution JSON/JSONL, and text/Markdown reviews are written but not used for stage resume. Some analysis files are saved incrementally, so existence cannot prove completion. |
| Codex checkpoints | Existing response cache, usage logs, run-owned session and error handling already work; these were not redesigned. |

## Changes

- `ScholarEval/ScholarEval.py`: `--resume`, `--no-llm`,
  `--stop-after-retrieval`, visible skip messages, UTF-8 child output, safe plan copy.
- `ScholarEval/ScholarEval_app.py`: existing-stage directory selection, resume
  propagation, and concurrent pipe draining during progress monitoring.
- New `utils/checkpoints.py`: subprocess-local completion gates, validation,
  fingerprints/output hashes, atomic metadata, failed-stage status.
- Every soundness/contribution stage entry point: calls the checkpoint gate
  before main/LLM initialization. Later existing artifacts become reusable only
  after validated completion metadata is recorded; legacy partial files are not
  inferred to be complete.
- `utils/semantic_scholar.py` and new `utils/retrieval_http.py`: shared rate
  limiter/retry policy and typed HTTP/schema failures across existing endpoints.
- New `utils/retrieval_progress.py`, rewritten `soundness/snippet_search.py`
  and `contribution/paper_extractor.py`: durable query/operation state,
  deterministic IDs, failure propagation, metrics, and completion-only outputs.
- `contribution/paper_augmentation.py`: persisted recommendation/reference
  operations; errors propagate; GROBID starts only for downloaded PDFs in the
  existing related-work mode. Default API-reference augmentation skips GROBID.
- New `utils/grobid.py`, updated `utils/pdf_utils.py`: bounded readiness polling,
  deterministic reuse/start, diagnostics, zero-PDF gate, validation of existing
  downloaded PDFs. Snippet parsing isolates the current PDF set and requires XML
  output for every downloaded PDF.
- `soundness/meta_review.py`: preserve S2 failures and use the existing key name
  consistently in bibliography retrieval.
- New `scripts/check_retrieval_backend.py`,
  `scripts/import_legacy_checkpoints.py`, and `tests/test_retrieval_reliability.py`.
- `requirements.txt`, `README.md`, and the retrieval guide document dependencies,
  configuration, commands, metrics, and failure recovery.

The section keyword list and text extraction logic were retained. Extraction
and query prompt string equality against the original source was checked.
Contribution scores, thresholds, retrieval limits/batching, and novelty reasoning
were not modified.

## Validation and live results

- **79 offline tests passed**, including the existing 47 Codex backend tests.
  Mocks reproduce repeated 429, recovery after 429, real successful empty
  responses, malformed payloads, auth failures, server/network retries, numeric
  and HTTP-date Retry-After, shared limiting, interruption/query resume, zero
  PDFs, delayed readiness, timeout/OOM diagnostics, port conflicts, stale input,
  and upstream skipping.
- A real **top-level CLI subprocess with offline fixture checkpoints** passed
  under `--resume --no-llm --stop-after-retrieval`. It logged method extraction,
  query generation, and snippet search skipped; created no usage log or Codex DB.
- **Local preflight: LOCAL READY.** Docker CLI/daemon, the existing expected
  image/container, port 8070, and `/api/isalive` passed. Python dependencies pass.
  PyPDF2 was already declared in requirements but missing from `.venv`; version
  3.0.1 was installed there during this task.
- **S2 preflight:** no key configured. Its one anonymous metadata request
  returned **429**, so the connectivity-enabled preflight correctly reported
  **NOT READY**. The existing healthy GROBID container was reused; no duplicate
  was created.
- **Direct live snippet smoke:** used the existing three methods/queries at
  `demo_data/astra-test-1/soundness/`, with isolated outputs in
  `demo_data/astra-test-1/retrieval-smoke/`. It made **7 attempts, 6 retries,
  7 HTTP 429 responses, 0 successes**, with **125.18 seconds** of backoff.
  It exited nonzero with `RetrievalRateLimitError`; the progress journal contains
  one `failed-retryable` query with 7 attempts and two `pending` queries with 0
  attempts. It produced **no references or papers success artifacts** and made
  **no GROBID/client or LLM calls**. Metrics/progress remain available locally.
- The live top-level resume smoke was **not run**, because direct retrieval did
  not succeed. Zero live Astra calls were consumed throughout this pass.
- The known successful saved methods/queries were explicitly imported with
  legacy provenance sidecars after structural validation and saved-plan matching.
  Original artifacts and timestamps were preserved. The old failed references
  were not imported or trusted.
- `git diff --check` passed. A scan of tracked/new nonignored files found no
  configured secret values, API-key-shaped credentials, or auth-data files.

## Remaining blockers / next run

The code now fails safely at infrastructure boundaries. A complete scientific
idea evaluation has **not** been demonstrated and should wait for successful S2
snippet retrieval. Configure `S2_API_KEY` with appropriate endpoint access, then
follow the exact retrieval-only and guarded top-level commands in
[the retrieval guide](RETRIEVAL_RELIABILITY.md). Anonymous access remains allowed,
but is currently rate-limited in this environment.

No live PDF parsing coverage was exercised because retrieval never produced
papers. Readiness, reuse and diagnostics were tested, including live health.
The original contribution embedding stage still needs its separate Titan/API
configuration; that requirement was not changed or live-tested here.
