# Engine audit (before implementation)

Inspected the current Python source, not just the README. Worktree initially
clean on `main`; implementation branch: `feat/codex-plus-astra-backend`.

## Architecture map

`ScholarEval/ScholarEval.py` runs sequential soundness and contribution stages
as `python -m` subprocesses. `ScholarEval_app.py` independently launches the same
stages from Streamlit. Worker pools inside stages create thread-local engines.

There is one text engine: `engine/litellm_engine.py::LLMEngine`. Despite its name,
it uses `openai.OpenAI(base_url=api_endpoint).chat.completions.create`, not
`litellm.completion`. LiteLLM supplies pricing tables in the callers.

Public contract: constructor `(llm_engine_name, api_key, api_endpoint)`;
synchronous `respond(user_input, temperature=0.7, top_p=0.95, max_tokens=40000)`
returns `(response_text, prompt_tokens, completion_tokens)`. Only temperature
is forwarded; top_p and max_tokens are dropped. No explicit engine retries or
timeouts; these inherit installed OpenAI SDK defaults. No async text calls.
The local validation environment installed OpenAI SDK 2.54.0: its inspected
defaults are two retries, 5-second connection timeout and 600-second read/write/
pool timeouts. The repository does not pin the SDK, so these are environment
observations rather than new ScholarEval settings.

## Call sites and parsing

| Module | Engine creation / response contract |
| --- | --- |
| soundness/extract_methods | main; Python list via StringUtils.extract_python_list |
| soundness/make_queries | main; JSON object with query |
| soundness/methods_and_results_synthesis | thread-local, 8 workers; JSON method/results/context |
| soundness/meta_review | thread-local worker pool; JSON support/contradictions/suggested_action/soundness_score |
| soundness/tldr_soundness | main; JSON summaries/suggestions, then existing Markdown conversion |
| contribution/extract_dimensions_and_contributions | main; JSON dimension-to-statements mapping, saved as JSONL |
| contribution/queries_generator | main; nonempty lines, capped to n_queries |
| contribution/relevance_assessor | thread-local worker pool; regex JSON extraction then score/rationale |
| contribution/pairwise_comparator | thread-local, 5 workers; JSON comparison and dimension scores |
| contribution/contribution_review_synthesis | main; plain Markdown; optional citation checker |
| utils/citation_check | engine per function invocation; plain text, two entry points |

`prepare_final_contribution_context` imports the engine without constructing it.
All current calls pass only messages plus temperature (0, 0.1, 0.2, or 0.3).
Messages contain plain string user content or system then user content. No
response_format, stop, tool, stream, top_p, or max_tokens caller overrides.
All histories are supplied in each call; no persistent conversation is needed.

StringUtils JSON parsing strips JSON fences, escapes some backslashes, and
returns None on failure (or the first element of a list). Its JSONL helper can
skip malformed records. Python-list parsing uses a non-greedy bracket regex
and ast.literal_eval, returning [] on failure. Relevance parsing can assign
score 0 on malformed output. Several worker loops catch all Exception values
and continue; backend failures must escape these handlers to avoid producing
scientific scores from transport failures. No caller-level LLM retry loops.

## Configuration and separate backends

All text-stage constructors read API_KEY and API_ENDPOINT. Citation helpers
read API_KEY_1. The Streamlit sidebar displays API_KEY and S2_API_KEY presence
and hardcodes a small-model name for some stages. The CLI requires
llm_engine_name and defaults litellm_name to a Claude pricing entry.

`dataset_creation/extract_research_plan` reads API_KEY; `extract_review_summary`
and `evaluation/llm_metrics` read API_KEY_1, all with API_ENDPOINT. They import
`..engine` outside ScholarEval's package (pre-existing packaging problem).
`evaluation/coverage` independently uses PrometheusEval's LiteLLM('openai/gpt-4o')
and is not the ScholarEval engine. Benchmark/evaluation code remains outside
this integration.

`contribution/embedding_filter` directly calls the OpenAI SDK embeddings endpoint
with model `Titan Text Embeddings V2`, API_KEY and API_ENDPOINT. This separate
retrieval dependency cannot be replaced by a Codex text turn without changing
retrieval. Semantic Scholar uses S2_API_KEY (bibliography also S2_API_KEY_2);
PDF/GROBID dependencies remain separate. Retrieval includes async downloads.

## Minimal integration boundary

Keep the legacy import as a backend factory, retain the original API engine,
and put the Codex transport, discovery, lifecycle, and checkpoints in engine/.
The multi-process orchestrators need a small run-scoped local connection to
share one native stdio app-server. It is not an OpenAI HTTP compatibility proxy.
Only exception propagation is needed inside scientific stages. Existing prompts,
parsers, retrieval, comparison rules, scoring, and benchmark logic are retained.

The CLI saves stage artifacts but does not skip all completed stages on rerun.
Per-response engine checkpoints allow those calls to be replayed after quota
reset; retrieval may still rerun using its existing progress files.

Protocol examined: installed official `codex-cli 0.153.4`, both stable and
experimental schemas generated locally with `app-server generate-json-schema`.
The stable thread API has baseInstructions/developerInstructions and text user
inputs, but no arbitrary role-history input or universal tool_choice=none.
Experimental environments=[] disables environment access; explicit opt-in is
required. Native outputSchema is available on turn/start.
