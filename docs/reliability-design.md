# Reliability audit and design

## Source audit (7 September 2026)

The CLI launches six Soundness and eleven Contribution modules via `run_command`.
Every module uses `checked_main`, which previously overwrote both published output
and completed sidecar during recomputation. Subprocess failures became `False`.

| Stages | Inputs / dependencies | Cost and persistence before hardening |
| --- | --- | --- |
| Soundness extract_methods → make_queries | idea → methods | Astra; final files only |
| snippet_search | queries + methods | S2 snippets/batch; PDF hosts/Unpaywall; Docker/GROBID; retrieval and PDF progress |
| methods_and_results_synthesis | idea + references + paper evidence | Astra per method/reference; destructive per-method writes |
| meta_review → tldr_soundness | analysis + references → meta review + idea | Astra per method then final summary; S2 citations; final files only |
| Contribution dimensions → queries → paper_extractor | idea → contributions → queries | Astra then S2; retrieval progress |
| relevance_assessor | idea + extracted papers | Astra per abstract; partial JSON trusted by paperId alone |
| paper_augmentation | relevance papers (score >= 3) | recommendations + paginated references; optional related-work PDF/GROBID; operation progress |
| embedding_filter → relevance_assessor | augmented papers + idea | separate API_KEY/API_ENDPOINT Titan embedding service, then Astra |
| paper_sampler | final relevance | local score sort and top 25 |
| pairwise_comparator | sample + dimensions + idea | Astra per paper; final file only |
| prepare_final_contribution_context → contribution_review_synthesis | comparisons → context + idea | local transform then Astra |

Final contribution review also calls `utils/citation_check.py`, which constructs
its own LLMEngine for a second inference. Soundness can use the same helper when
a bibliography is supplied. These calls must participate in caching, fingerprint
identity, and offline test isolation just like the visible stage-level calls.

S2 calls live in `semantic_scholar.py`; all already route through RetrievalHTTP,
but the old limiter stores only last request time. There is no persistent response
cache or shared cooldown. List validation rejects malformed individual records.
Recommendations and references already have a local uncommitted redundancy fix;
preserve it. Bibliography still performs per-ID metadata lookups. Codex uses the
official app-server with managed ChatGPT auth, an SQLite response cache, and a
run-owned loopback session. A failure latches the engine until restart. Quota at
startup prevents cache access. Embedding errors become fabricated zero vectors.
GROBID has bounded document parsing but no recovery of an unhealthy running
container. PDF download attempts persist, including exhaustion as unavailable.

Main artifact writes are in stage `main` functions, retrieval coverage/progress,
PDF/XML writers, Codex SQLite, and cost JSONL. Stage artifacts need isolated
generations; progress and item results need atomic commits; append-only cost logs
need UTF-8. Existing broad handlers in synthesis/relevance/comparisons must not
turn infrastructure errors into scientific scores or accept incomplete stages.

## Failure matrix

| Failure | Classification/state | Recovery and escalation | Preserved state |
| --- | --- | --- | --- |
| S2 429 | WAITING_RATE_LIMIT | shared adaptive cooldown, half-open probe, bounded supervisor duration | all completed items/generations |
| 5xx, timeout, disconnect | RETRYABLE_FAILED | client retry then bounded stage restart | all completed items/generations |
| Codex allowance exhausted | WAITING_QUOTA | slow exponential retry of next pending operation; session restart; configurable maximum wait | exact responses and items |
| Auth revoked/missing config | BLOCKED_AUTH / BLOCKED_CONFIG | stop with actionable category; next invocation retries after correction | all |
| Invalid top-level S2 JSON/schema | RETRYABLE_FAILED then FATAL_SCHEMA | sanitized quarantine, small repeated-failure budget | successful responses only |
| Malformed list record | skipped record | count and skip; preserve batch positions with null | usable records |
| Optional PDF unavailable | metadata evidence | continue under available-evidence, record reason | abstracts/snippets and full texts |
| GROBID unavailable | RETRYABLE_FAILED | recover only identified ScholarEval container, then bounded retry | parsed documents |
| Bad scientific result | FATAL_VALIDATION | refuse promotion; no fabricated score | last completed generation |
| Ctrl+C/process kill | CANCELLED / abandoned RUNNING | next invocation resolves committed generation and resumes items | last completed generation |
| Disk full/corrupt required idea | BLOCKED_CONFIG / FATAL_VALIDATION | stop; never substitute scientific input | old generation where storage still readable |

## Coherent implementation

Use SQLite for work items, events, metrics, and stage states. Store each stage's
candidate outputs in a unique generation directory. Validate/fsync every output,
commit an immutable generation record, atomically replace the completed sidecar
pointer, then materialize compatibility filenames. A restart repairs compatibility
files from that pointer before any consumer runs. Running/failure state lives in
a separate attempt record, never overwrites completed metadata. A process lock
prevents concurrent publishers in one run. Never remove prior generations.

The explicit DAG selects required stages backward from final targets and stops at
valid downstream boundaries. Boundary identity covers the idea, cutoff, scientific
parameters, model/reasoning, prompt and algorithm versions transitively. Direct
stage identity additionally covers immediate input content. Missing historical
inputs do not invalidate an already committed downstream boundary. A separately
consumed dimensions artifact is still required by pairwise comparison.

Semantic identity excludes timeouts, retries, worker counts, polling, logging,
file destinations, and transport source files. Prompt templates and scientific
algorithm versions belong in semantic identity. Infrastructure-only refactors
use a verified compatibility migration, never re-attest unknown partial data.

Use one reusable SQLite item store for expensive stage loops and retrieval
operations. Namespace by stage semantic identity; key by complete item inputs.
Commit validated results before proceeding. Exact Astra responses retain existing
cache keys and zero-new-token hits; no API fallback. Embeddings remain separately
configured unless the user authorizes a scientific change.

Preflight checks only dependencies required by the selected DAG, before uncached
inference. Tests run entirely offline first. A no-LLM/no-network dry run must never
spend allowance. A controlled real run is gated on tests and actual credentials.
