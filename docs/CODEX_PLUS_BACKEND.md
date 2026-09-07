# Codex / ChatGPT Plus backend

ScholarEval can use its original API engine (the default), or the official
Codex app-server authenticated by Codex's ChatGPT login. There is no Platform
API key, token extraction, private HTTP integration, or community proxy in the
Codex backend. Account access and the exact Astra wire model ID are discovered
through the official executable.

## Requirements and authentication

- Official stable Codex CLI **>= 0.153.0**. Developed against **0.153.4**.
- A ChatGPT account exposing GPT-6 Astra in Codex. Availability is account-dependent
  during rollout; Plus alone is not a guarantee of model visibility.
- Python and ScholarEval dependencies: `python -m pip install -r requirements.txt`.
  Existing pipeline modules also import `pandas`, `aiohttp`, and `scikit-learn`;
  install these if missing. The stdio adapter itself uses the standard library;
  explicit JSON Schema validation uses `jsonschema`.

```powershell
codex.cmd --version
codex.cmd login
# Alternatively, the official device flow:
codex.cmd login --device-auth
```

On macOS/Linux, use `codex` instead of `codex.cmd`. The Windows `.cmd` launcher
works without changing PowerShell execution policy. Log in using ChatGPT.
The adapter calls `account/read(refreshToken=true)`; only Codex accesses or
refreshes credentials. It never initiates login, logout, purchases, or quota
reset-credit consumption. It rejects API-key accounts. The app-server process
forces the built-in OpenAI provider and ChatGPT login mode, and does not accept
externally supplied OAuth tokens.

Official references: [app-server](https://learn.chatgpt.com/docs/app-server),
[configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[model availability](https://learn.chatgpt.com/docs/models),
[Work usage and cost](https://learn.chatgpt.com/docs/enterprise/chatgpt-work-usage-and-cost).

## Configuration

```powershell
$env:SCHOLAREVAL_LLM_BACKEND = 'codex'
$env:SCHOLAREVAL_CODEX_MODEL = 'auto'
$env:SCHOLAREVAL_CODEX_REASONING = 'high'
$env:SCHOLAREVAL_CODEX_MAX_CONCURRENCY = '1'
$env:SCHOLAREVAL_CODEX_REQUEST_TIMEOUT = '300'
```

These are the Codex defaults apart from `SCHOLAREVAL_LLM_BACKEND`, whose default
is `litellm`. `auto` selects a visible Astra entry, preferring current/default
entries over ones advertising an upgrade. An explicit model must match a
returned Astra catalog ID or wire ID. Other families and hidden entries are
rejected. Unsupported reasoning values fail with the advertised supported list;
the adapter never silently changes model or effort. It does not default to MAX.

Optional variables:

| Variable | Purpose |
| --- | --- |
| `SCHOLAREVAL_CODEX_EXECUTABLE` | Official executable path if Codex is not on PATH |
| `SCHOLAREVAL_CODEX_CACHE` | Persistent SQLite response checkpoint path for standalone stages or resuming UI runs |

The CLI automatically checkpoints to `<save_to>/codex-responses.sqlite3` unless
overridden. The UI creates a checkpoint in each run directory. Standalone engine
calls do not persist responses unless a cache path is configured. Do not share
checkpoint files with people who should not see the model's research responses.
`SCHOLAREVAL_CODEX_CONNECTION` is an internal, temporary local IPC capability;
do not set, print, or persist it yourself. It is unrelated to ChatGPT credentials.

## Preflight and smallest smoke test

From the repository root, using the Python environment with ScholarEval installed:

```powershell
python scripts/check_codex_backend.py --no-inference
python scripts/check_codex_backend.py
```

The first command performs no inference. The second checks version, initialization,
ChatGPT account/plan, model discovery, effort, allowance status, and one tiny
`OK` response. It fails closed if allowance status is unavailable. Only allowlisted
status fields are printed; stderr and raw protocol/auth errors are suppressed.

To additionally exercise the existing method-extraction prompt, engine factory,
parser, and output artifact in one process lifecycle:

```powershell
python scripts/check_codex_backend.py --scholareval-smoke --output-dir .scholareval-codex/smoke-1
```

This performs **two small inferences total**, no literature search, and no
benchmark. The synthetic `idea.txt` is exclusively created; use a fresh output
directory on each rerun. API keys are removed from the smoke stage's environment.
The backend smoke and ScholarEval stage share one app-server process and use
separate ephemeral threads. Preflight bypasses checkpoints so PASS means live.

For this workspace, the prepared Python environment is `.venv/Scripts/python.exe`.
Use that in place of `python`, or activate `.venv` before running commands.

## Running ScholarEval

The original CLI works with the new backend; its required legacy engine-name
argument is ignored for Codex model selection:

```powershell
python -m ScholarEval.ScholarEval --research_idea path/to/idea.txt --llm_engine_name auto --save_to demo_data/astra-baseline
```

This runs the full original pipeline for **one** idea and can consume substantial
allowance. It was not run as part of integration testing. For a small actual stage:

```powershell
$env:SCHOLAREVAL_CODEX_CACHE = 'demo_data/astra-baseline/codex-responses.sqlite3'
python -m ScholarEval.soundness.extract_methods --input_file path/to/idea.txt --output_file demo_data/astra-baseline/methods.json --llm_engine_name auto
```

Create the output directory first. Do not pass `--litellm_name` to standalone
Codex stages: API price tables do not describe included Codex usage. The main
CLI automatically disables this price computation in Codex mode. Existing cost
fields of zero mean **no calculated API cost**, not unlimited/free model access.

The Streamlit entry point also shares one run-owned engine across its subprocesses:
`python -m streamlit run ScholarEval/ScholarEval_app.py`. Model-picker API choices
are disabled when the Codex backend is selected. Each button run owns and closes
its app-server; the sidebar setting does not prove account access until preflight.

## Architecture and lifecycle

Before: stage -> `LLMEngine` -> OpenAI SDK -> configured API endpoint.

After: stage -> same `LLMEngine` factory -> `CodexAppServerEngine` -> official
`codex app-server` over stdio -> ChatGPT-managed auth -> discovered Astra.
For multi-process runs, a private loopback JSON connection forwards only
`respond` calls to the parent-owned engine. It is not HTTP, an OpenAI compatibility
layer, or a separately installed proxy. It is necessary because the original CLI
and UI launch a new Python process for each stage. Thread-local engine creation
within stages shares the same engine. Direct single-process usage needs no IPC.

Each call starts an ephemeral thread, subscribes before starting its turn,
waits for `turn/completed`, and unsubscribes. The process stays alive. Final
agent-message items are authoritative; deltas fill streaming text, and commentary,
reasoning and tool events never become the return value. Return type remains
`(text, input_tokens, output_tokens)`. Missing token counts return zero with a
warning (unknown, not measured). Checkpoint replays return zero new usage.

System and developer prefixes map to `baseInstructions` and
`developerInstructions`; a single user input retains its exact text. The normal
non-realtime API has no arbitrary assistant-role history input, so unusual
multi-message histories use an ordered JSON transcript with explicit roles and
escaped content, alongside the separate instruction fields. This is an
approximation for historical assistant roles; no conversation persists between
independent calls. Original task prompts remain unchanged.

EOF shuts down app-server; a bounded process-tree termination is used only if
shutdown hangs. There is no automatic process restart: failures stop the run
and preserve completed checkpoints. This avoids repeating an uncertain billed
turn. Timeouts include the full inference/retry/format-correction deadline;
waiting for a concurrency slot is separately bounded by the same timeout.
Only explicit capacity/overload errors are retried, at most twice, with exponential
delay and jitter. Quota, auth, rate-limit, invalid request and protocol failures
are not retried. Backend exceptions escape existing worker error handlers.

## Parameter and output compatibility

| Parameter | Codex policy |
| --- | --- |
| Messages | Text roles translated as described above; tools/images/extra message fields rejected |
| Reasoning | Forwarded as turn effort; validated against model/list |
| `output_schema` | Forwarded as outputSchema; schema and result validated |
| `temperature`, `top_p` | Ignored by design; one warning per engine; Codex exposes no sampling control |
| Default `max_tokens=40000` | Legacy compatibility sentinel, ignored with that warning; **not an output cap** |
| Explicit nondefault `max_tokens` | Rejected; Codex exposes no equivalent hard output cap |
| `stop`, `response_format`, other arguments | Rejected instead of silently ignored |

The API backend now forwards its already accepted `top_p` and `max_tokens`.
This isolated bug fix means its old default 40000 cap now actually reaches the
API. Adjust it explicitly for API models with different limits when calling the
engine directly; this does not add sampling control to Astra.

For existing prompts with a final JSON/Python fenced format example, the adapter
checks syntax before handing off to unchanged parsers. Valid JSON outer fences
are normalized to protect the existing relevance parser; fields, scores, escapes
and scientific content are not repaired. Explicit output_schema returns raw JSON.
One extra inference may request **syntax-only** correction after parse failure.
Failure after that is a distinct `CodexOutputFormatError`. Prompt-based format
detection is deliberately narrow; semantic field completeness, query-line quality,
and scientific correctness still belong to the existing pipeline.

## Permissions and remaining limitations

The generated 0.153.4 stable protocol has no universal no-tools switch. This
adapter opts into its generated experimental schema for `environments=[]`
(disable environment access) and `allowProviderModelFallback=false`. It combines
these with read-only/network-disabled sandbox policy, never approvals, an empty
temporary working directory, disabled shell/exec/patch, browsing, MCP servers,
apps, plugins, hooks, memory, skills discovery and delegation features. Any
server tool/approval request or observed non-text action stops inference.

This is a constrained Codex agent, not a raw completion endpoint. Codex may still
add runtime instructions or expose internal protocol facilities; tool absence is
not asserted as a universal future-version guarantee. Codex itself necessarily
uses the network for authentication/model inference and may write its own auth
refresh, cache or diagnostics outside ephemeral conversation history. Those are
official runtime operations, not research-model workspace actions. Recheck the
generated schema after upgrades. Managed machine policy may cause startup to
fail if it conflicts with these restrictions.

ChatGPT Plus allowance is limited. Work and Codex share the applicable agentic
allowance; exact limits and Astra access are account-dependent. This backend is
**not OpenAI Platform API usage**. It never switches to an API key, another model,
Fast tier, or purchased/reset credits. The initial rate-limit read and subsequent
notifications stop work at reported exhaustion, conservatively checking all
reported buckets. The compact `limited` label means at least one reported window
has used 90% or more; `exhausted` means a reported limit or 100% usage. It cannot
reserve allowance or know that another client will
consume quota concurrently; an already-running turn can cross a usage boundary.
Server/account billing policy remains authoritative, including any extra credits
you configured independently. No numeric remaining message count is invented.

Completed text responses are checkpointed immediately, keyed by exact messages,
model, effort, adapter format revision, CLI version and output schema. Rerun the
same command with the same cache after quota reset. Incomplete/failed responses
are never checkpointed. Existing intermediate files remain in place. The original
orchestrator may rerun retrieval; this is not a new whole-pipeline scheduler or
scientific-result cache. Use a fresh output/cache path for an independent benchmark
replicate. Changing prompts/model/effort prevents reuse.

Semantic Scholar keys, PDF/GROBID, and **Titan embedding API_KEY/API_ENDPOINT**
remain separate retrieval dependencies. A Codex text call needs no API key; the
unchanged full contribution pipeline may still require its embedding API key.
The adapter does not improve retrieval recall, contribution comparison, or
evaluation methodology. Dataset-creation and independent benchmark scripts are
not migrated; their pre-existing import/dependency issues are outside scope.

Allowance is spent on method extraction, query generation, each paper/method
analysis, relevance assessment, each pairwise comparison, synthesis and citation
checking. Format correction and capacity retries may add attempts. Long evidence
contexts and high reasoning cost more than the tiny smoke. Checkpoint reuse and
account/model/status discovery do not perform new model inference.

## Switching back and testing

```powershell
$env:SCHOLAREVAL_LLM_BACKEND = 'litellm'
$env:API_KEY = 'your-existing-gateway-key'
$env:API_ENDPOINT = 'your-existing-gateway-endpoint'
python -m ScholarEval.ScholarEval --research_idea path/to/idea.txt --llm_engine_name YOUR_API_MODEL --save_to demo_data/api-baseline
```

Provide the correct optional `--litellm_name` for your gateway's pricing, as before.
There is no automatic fallback between backends.

Offline tests and local protocol contract checks:

```powershell
codex.cmd app-server generate-json-schema --out .codex-schema
codex.cmd app-server generate-json-schema --experimental --out .codex-schema-experimental
python -m unittest discover -s tests -v
git diff --check
```

The optional generated-schema contract test skips if no schema has been generated.
The mocked suite does not spend allowance. See [the source audit](CODEX_BACKEND_AUDIT.md)
for the original architecture and call-site/parser inventory.

## Validation in this fork (2026-09-06)

- Official CLI: **0.153.4**; stable and experimental protocol schemas generated
  from that executable. No upgrade or community proxy needed.
- `account/read`: **ChatGPT managed**, plan **plus**.
- `model/list` wire model: **gpt-6-astra**; supported efforts:
  **low, medium, high, xhigh, max, ultra**. Both real calls requested **high**.
- Live backend smoke: **PASS**, exact `OK` final response.
- Live unchanged ScholarEval method-extraction stage: **PASS**, two parsed methods
  saved to `.scholareval-codex/smoke/methods.json`; reported **4207 input / 147
  output tokens** for this stage.
- **Two successful live inference calls total**; no format corrections, full
  retrieval run, or benchmark. Later failure-path/shutdown hardening was checked
  offline and with a final no-inference preflight, avoiding another inference
  turn. The final status check reports **limited** quota (initially available).
- **47 offline tests**, including installed-schema request validation; Python
  compilation and `git diff --check`. No repository lint/type configuration exists.
- No API key used for the smoke; no authentication material logged; no GitHub push.
