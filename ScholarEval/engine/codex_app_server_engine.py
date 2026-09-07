"""Stateless text inference over one long-lived, officially authenticated Codex."""

import ast
import atexit
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

from .codex_transport import (
    CodexAuthError, CodexCapacityError, CodexError, CodexModelError,
    CodexOutputFormatError, CodexProcessError, CodexProtocolError,
    CodexQuotaError, CodexTimeoutError, JsonRpcClient, codex_version,
    find_codex, server_error,
)

LOG = logging.getLogger(__name__)
TRANSPORT_INSTRUCTIONS = (
    'Answer the supplied ScholarEval inference task using only the supplied text. '
    'Do not use tools, inspect files, browse, delegate, or perform actions. '
    'Return the requested final answer without progress commentary.'
)

# Official configuration fields plus schema-gated environments=[] on thread/start.
# No personal config file is read by this adapter; config/read is app-server-owned.
RESTRICTED_CONFIG = {
    'model_provider': 'openai', 'forced_login_method': 'chatgpt',
    'approval_policy': 'never', 'approvals_reviewer': 'user',
    'sandbox_mode': 'read-only', 'web_search': 'disabled',
    'project_doc_max_bytes': 0, 'history.persistence': 'none',
    'skills.bundled.enabled': False, 'skills.include_instructions': False,
    'include_apps_instructions': False, 'include_environment_context': False,
    'include_collaboration_mode_instructions': False,
    'tools.update_plan.enabled': False,
    'features.shell_tool': False, 'features.unified_exec': False,
    'features.apply_patch_freeform': False, 'features.hooks': False,
    'features.codex_hooks': False, 'features.plugin_hooks': False,
    'features.plugins': False, 'features.remote_plugin': False,
    'features.apps': False, 'features.connectors': False,
    'features.memories': False, 'features.chronicle': False,
    'features.multi_agent': False, 'features.multi_agent_v2': False,
    'features.browser_use': False, 'features.computer_use': False,
    'features.image_generation': False, 'features.view_image': False,
    'features.code_mode': False, 'features.code_mode_host': False,
    'features.js_repl': False, 'features.goals': False,
    'features.skill_search': False, 'features.skip_host_skill_discovery': True,
    'features.skill_mcp_dependency_install': False,
    'features.tool_search': False, 'features.tool_suggest': False,
    'features.shell_snapshot': False,
    'features.unbounded_connection_retries': False,
}


def translate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise CodexProtocolError('Expected a nonempty list of text messages.')
    for msg in messages:
        if (not isinstance(msg, dict) or set(msg) - {'role', 'content'}
                or msg.get('role') not in ('system', 'developer', 'user', 'assistant')
                or not isinstance(msg.get('content'), str)):
            raise CodexProtocolError('Codex supports text system/developer/user/assistant messages only.')
    system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
    developer = '\n\n'.join(m['content'] for m in messages if m['role'] == 'developer')
    # The common system/developer prefix + single user call maps natively.
    prefix = messages[:-1]
    native = (messages[-1]['role'] == 'user'
              and all(m['role'] in ('system', 'developer') for m in prefix)
              and [m['role'] for m in prefix] == sorted(
                  [m['role'] for m in prefix], key=lambda role: role != 'system'))
    if native:
        user = messages[-1]['content']
    else:
        # Standard non-realtime turns have no assistant-history input. JSON escaping
        # keeps delimiters in evidence from changing the serialized role boundaries.
        user = ('[ORDERED CONVERSATION]\n'
                'Continue this conversation, preserving the roles and order shown.\n'
                + json.dumps(messages, ensure_ascii=False) + '\n[END CONVERSATION]')
    return system or TRANSPORT_INSTRUCTIONS, '\n\n'.join(filter(None, [developer, TRANSPORT_INSTRUCTIONS])), user


def discover_astra(models, requested='auto', effort='high'):
    def is_astra(model):
        name = ' '.join(str(model.get(k, '')) for k in ('id', 'model', 'displayName'))
        return bool(re.search(r'gpt[\s-]*6[\s-]+astra\b', name, re.I))

    available = ', '.join(f"{m.get('displayName', '?')} ({m.get('model', '?')})" for m in models)
    candidates = [m for m in models if is_astra(m) and not m.get('hidden')]
    if requested != 'auto':
        candidates = [m for m in candidates if requested in (m.get('id'), m.get('model'))]
    if not candidates:
        raise CodexModelError('BLOCKED BY ACCOUNT AVAILABILITY: requested GPT-6 Astra is not visible. '
                              'Available models: ' + (available or '(none)'))
    # Advertised upgrades identify superseded entries; prefer non-retiring/default.
    candidates.sort(key=lambda m: (bool(m.get('upgrade') or m.get('upgradeInfo')),
                                   not m.get('isDefault'), m.get('model', '')))
    selected = candidates[0]
    supported = [r['reasoningEffort'] for r in selected['supportedReasoningEfforts']]
    if effort not in supported:
        raise CodexModelError(f'Reasoning {effort!r} is unsupported. Supported values: {", ".join(supported)}')
    return selected, supported


def quota_state(payload):
    """Conservatively honor all reported buckets; never invent remaining messages."""
    buckets = list((payload.get('rateLimitsByLimitId') or {}).values())
    if payload.get('rateLimits'):
        buckets.append(payload['rateLimits'])
    observed = []
    for bucket in buckets:
        if bucket.get('rateLimitReachedType') or bucket.get('spendControlReached'):
            return 'exhausted'
        for window in ('primary', 'secondary'):
            entry = bucket.get(window)
            if entry is not None:
                used = entry.get('usedPercent')
                if isinstance(used, (int, float)):
                    if used >= 100:
                        return 'exhausted'
                    observed.append(used)
    if not observed:
        return 'unknown'
    return 'limited' if max(observed) >= 90 else 'available'


def validate_output(text, messages, output_schema=None):
    """Syntax only; leave scientific content and existing downstream parsers alone."""
    template = '\n'.join(m['content'] for m in messages)
    formats = re.findall(r'```(jsonl|json|python)\b', template)
    declared = [(m.group(1) or 'json').lower() for m in re.finditer(
        r'parseable\s+(jsonl|json|python)\s+block|JSON formatting requirements:', template, re.I)]
    # Explicit format instructions take precedence over code examples in evidence.
    formats = declared or formats
    kind = 'json' if output_schema is not None else (formats[-1] if formats else None)
    try:
        if kind in ('json', 'jsonl'):
            match = re.fullmatch(r'\s*```(?:jsonl?|)\s*\n?(.*?)\s*```\s*', text, re.S)
            raw = match.group(1) if match else text.strip()
            if kind == 'jsonl':
                objects = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if not objects or not all(isinstance(obj, dict) for obj in objects):
                    raise ValueError
            else:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError
                if output_schema is not None:
                    import jsonschema
                    try:
                        jsonschema.validate(obj, output_schema)
                    except jsonschema.ValidationError:
                        raise ValueError from None
            # Preserve a JSON fence for the existing relevance parser, whose raw
            # brace regex cannot handle braces inside rationale strings.
            # No escapes, fields, or scientific values are repaired.
            return raw if output_schema is not None or kind == 'jsonl' else '```json\n' + raw + '\n```'
        if kind == 'python':
            match = re.search(r'\[\s*(.*?)\s*\]', text, re.S)
            value = ast.literal_eval('[' + match.group(1).strip() + ']')
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError
        if not text.strip():
            raise ValueError
        return text
    except (ValueError, SyntaxError, AttributeError):
        raise CodexOutputFormatError('Model output format failure: expected the requested parseable text format.') from None


class CodexAppServerEngine:
    def __init__(self, *, client=None, status=None, cache_path=None):
        try:
            self.timeout = float(os.environ.get('SCHOLAREVAL_CODEX_REQUEST_TIMEOUT', '300'))
            concurrency = int(os.environ.get('SCHOLAREVAL_CODEX_MAX_CONCURRENCY', '1'))
        except ValueError:
            raise CodexError('Codex timeout and concurrency must be numeric.') from None
        if not math.isfinite(self.timeout) or self.timeout <= 0 or not 1 <= concurrency <= 8:
            raise CodexError('Use a positive finite timeout and max concurrency between 1 and 8.')
        self.effort = os.environ.get('SCHOLAREVAL_CODEX_REASONING', 'high')
        self.requested_model = os.environ.get('SCHOLAREVAL_CODEX_MODEL', 'auto')
        self._slots = threading.BoundedSemaphore(concurrency)
        self._cache_lock = threading.Lock()
        self._failure = None
        self._closed = False
        self._warned = False
        self._temp = tempfile.TemporaryDirectory(prefix='scholareval-codex-')
        self._cache = None
        self.client = client
        self.status = {}
        self._reporter = status or self._log_status
        try:
            if client is None:
                executable = find_codex()
                self.version = codex_version(executable)
                self._report('Codex version', self.version)
                command = [executable, 'app-server', '--listen', 'stdio://']
                for key, value in RESTRICTED_CONFIG.items():
                    command.extend(['-c', key + '=' + json.dumps(value)])
                child_env = {k: v for k, v in os.environ.items() if k not in (
                    'OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'API_KEY',
                    'API_KEY_1', 'API_ENDPOINT', 'SCHOLAREVAL_CODEX_CONNECTION',
                )}
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                           errors='strict', bufsize=1, cwd=self._temp.name, env=child_env,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                self.client = JsonRpcClient(process)
            else:
                self.version = 'mock'
            self.client.on_notification = self._notification
            self.client.initialize()
            self._report('App-server', 'PASS')
            self._report('Backend', 'Codex app-server')
            account = self.client.request('account/read', {'refreshToken': True}).get('account')
            if not account or account.get('type') != 'chatgpt':
                raise CodexAuthError('ChatGPT-managed login required. Run codex login (or codex login --device-auth).')
            self._report('Auth', 'ChatGPT managed')
            self._report('Plan', account.get('planType', 'unknown'))
            models, cursor, seen = [], None, set()
            while True:
                result = self.client.request('model/list', {'includeHidden': False, 'cursor': cursor})
                models.extend(result['data'])
                cursor = result.get('nextCursor')
                if not cursor:
                    break
                if cursor in seen:
                    raise CodexProtocolError('Repeated cursor in model/list.')
                seen.add(cursor)
            self.models = models
            self.model_info, self.supported_efforts = discover_astra(models, self.requested_model, self.effort)
            self.model = self.model_info['model']  # Discovered wire ID, never inferred from a label.
            self.llm_engine_name = self.model
            self._report('Astra', self.model)
            self._report('Reasoning', self.effort)
            self._report('Supported reasoning', ', '.join(self.supported_efforts))
            limits = self.client.request('account/rateLimits/read')
            self._report('Quota', quota_state(limits))
            if self.status['Quota'] == 'exhausted':
                raise server_error({'codexErrorInfo': 'usageLimitExceeded'})
            if self.status['Quota'] == 'unknown':
                raise CodexQuotaError('Codex did not expose allowance state; cannot verify included usage before inference.')
            # Read only through the official executable, never token/config files.
            config = self.client.request('config/read', {'includeLayers': False})['config']
            self.thread_config = dict(RESTRICTED_CONFIG)
            for section in ('mcp_servers', 'plugins'):
                self.thread_config[section] = {name: {'enabled': False} for name in config.get(section) or {}}
            self.thread_config['apps'] = {'_default': {'enabled': False}, **{
                name: {'enabled': False} for name in (config.get('apps') or {}) if name != '_default'
            }}
            path = os.environ.get('SCHOLAREVAL_CODEX_CACHE') if cache_path is None else cache_path
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self._cache = sqlite3.connect(path, check_same_thread=False)
                self._cache.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, text TEXT NOT NULL)')
                self._cache.commit()
            atexit.register(self.close)
        except (KeyError, TypeError, ValueError, AttributeError):
            self.close()
            raise CodexProtocolError('Codex preflight returned an unexpected schema; raw payload suppressed.') from None
        except (OSError, sqlite3.Error):
            self.close()
            raise CodexProcessError('Could not start Codex or open its local checkpoint; check executable and filesystem permissions.') from None
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _log_status(key, value):
        print(f'{key}: {value}', file=sys.stderr, flush=True)

    def _report(self, key, value):
        self.status[key] = value
        self._reporter(key, value)

    def _notification(self, method, params):
        if method == 'account/rateLimits/updated':
            if quota_state(params) == 'exhausted':
                self._failure = server_error({'codexErrorInfo': 'usageLimitExceeded'})
        elif method == 'account/updated' and params.get('authMode') != 'chatgpt':
            self._failure = CodexAuthError('Codex authentication changed; stop and log in with ChatGPT.')
        elif method == 'model/rerouted':
            self._failure = CodexModelError('Codex reported a model reroute; no substituted-model output will be accepted.')

    def respond(self, user_input, temperature=0.7, top_p=0.95, max_tokens=40000, *, output_schema=None, **kwargs):
        if kwargs:
            raise CodexProtocolError('Unsupported Codex parameters: ' + ', '.join(sorted(kwargs)))
        if max_tokens != 40000:
            raise CodexProtocolError('Codex app-server has no max_tokens control. Explicit output-token caps are unsupported.')
        translate_messages(user_input)  # Validate before spending allowance.
        if output_schema is not None:
            import jsonschema
            try:
                jsonschema.Draft202012Validator.check_schema(output_schema)
            except jsonschema.SchemaError:
                raise CodexProtocolError('Invalid output_schema; no inference sent.') from None
        if not self._slots.acquire(timeout=self.timeout):
            raise CodexTimeoutError('Timed out waiting for the Codex concurrency slot; no inference sent.')
        try:
            if self._closed:
                raise CodexProcessError('Codex engine is closed.')
            if self._failure:
                raise self._failure
            with self._cache_lock:
                if not self._warned:
                    LOG.warning('Codex ignores temperature, top_p and the legacy default max_tokens=40000; no sampling/output cap is promised.')
                    self._warned = True
            key = hashlib.sha256(json.dumps([2, self.version, self.model, self.effort,
                                            user_input, output_schema], sort_keys=True).encode()).hexdigest()
            if self._cache:
                with self._cache_lock:
                    cached = self._cache.execute('SELECT text FROM responses WHERE key=?', (key,)).fetchone()
                if cached:
                    return cached[0], 0, 0  # No new usage on resume.
            deadline = time.monotonic() + self.timeout
            try:
                result = self._with_capacity_retry(user_input, output_schema, deadline)
                try:
                    text = validate_output(result[0], user_input, output_schema)
                except CodexOutputFormatError:
                    correction = user_input + [
                        {'role': 'assistant', 'content': result[0]},
                        {'role': 'user', 'content': 'The previous answer did not parse. Return the same scientific content '
                         'in the originally requested format. Correct syntax only; do not change claims, scores, or evidence.'},
                    ]
                    fixed = self._with_capacity_retry(correction, output_schema, deadline)
                    text = validate_output(fixed[0], user_input, output_schema)
                    result = text, result[1] + fixed[1], result[2] + fixed[2]
                if self._cache:
                    with self._cache_lock:
                        self._cache.execute('INSERT OR REPLACE INTO responses VALUES (?, ?)', (key, text))
                        self._cache.commit()
                return text, result[1], result[2]
            except CodexError as error:
                # Fail all queued work immediately; do not repeatedly consume quota.
                self._failure = error
                raise
        except sqlite3.Error:
            self._failure = CodexError('Codex response checkpoint could not be read or saved; stop to preserve resumability.')
            raise self._failure from None
        finally:
            self._slots.release()

    def _with_capacity_retry(self, messages, schema, deadline):
        for attempt in range(3):
            try:
                return self._infer(messages, schema, deadline)
            except CodexCapacityError:
                if attempt == 2:
                    raise
                delay = 2 ** attempt + random.uniform(0, 0.5)
                if time.monotonic() + delay >= deadline:
                    raise CodexTimeoutError('Codex inference deadline reached during capacity backoff.') from None
                time.sleep(delay)

    def _infer(self, messages, schema, deadline):
        base, developer, user = translate_messages(messages)
        thread_id = None
        turn_id = None
        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise CodexTimeoutError('Codex inference timed out; no partial answer was accepted.')
            return seconds
        try:
            started = self.client.request('thread/start', {
                'model': self.model, 'modelProvider': 'openai', 'ephemeral': True,
                'approvalPolicy': 'never', 'approvalsReviewer': 'user', 'sandbox': 'read-only',
                'cwd': self._temp.name, 'baseInstructions': base, 'developerInstructions': developer,
                'config': self.thread_config, 'environments': [], 'allowProviderModelFallback': False,
            }, timeout=remaining())
            thread_id = started['thread']['id']
            if started['model'] != self.model or started['modelProvider'] != 'openai':
                raise CodexModelError('Codex changed the requested model/provider; inference refused.')
            if (started['approvalPolicy'] != 'never' or started['sandbox']['type'] != 'readOnly'
                    or started['sandbox'].get('networkAccess', False)):
                raise CodexProtocolError('Codex did not apply the required permissions.')
            events = self.client.subscribe(thread_id)
            params = {'threadId': thread_id, 'model': self.model, 'effort': self.effort,
                      'summary': 'none', 'serviceTierForTurn': 'default',
                      'sandboxPolicy': {'type': 'readOnly', 'networkAccess': False},
                      'input': [{'type': 'text', 'text': user}]}
            if schema is not None:
                params['outputSchema'] = schema
            turn_id = self.client.request('turn/start', params, timeout=remaining())['turn']['id']
            items, deltas, usage = {}, {}, {}
            while True:
                if self._failure:
                    raise self._failure
                try:
                    event = events.get(timeout=min(remaining(), 0.5))
                except queue.Empty:
                    continue
                if isinstance(event, BaseException):
                    raise event
                method, data = event
                if data.get('turnId') not in (None, turn_id):
                    continue
                if method == 'item/agentMessage/delta':
                    item_id = data['itemId']
                    deltas[item_id] = deltas.get(item_id, '') + data['delta']
                elif method in ('item/started', 'item/completed'):
                    item = data['item']
                    if item['type'] == 'agentMessage':
                        items[item['id']] = item
                    elif item['type'] not in ('userMessage', 'reasoning'):
                        raise CodexProtocolError('Codex attempted a non-text tool/action; inference stopped.')
                elif method == 'thread/tokenUsage/updated':
                    usage = data['tokenUsage']['total']
                elif method == 'thread/closed':
                    raise CodexProcessError('Codex closed the inference thread before completion.')
                elif method == 'error' and not data.get('willRetry', False):
                    raise server_error(data.get('error', {}))
                elif method == 'turn/completed':
                    turn = data['turn']
                    if turn['id'] != turn_id:
                        continue
                    if turn['status'] == 'failed':
                        raise server_error(turn.get('error') or {})
                    if turn['status'] != 'completed':
                        raise CodexError('Codex turn was interrupted; partial output discarded.')
                    for item in turn.get('items', []):
                        if item['type'] == 'agentMessage':
                            items[item['id']] = item
                        elif item['type'] not in ('userMessage', 'reasoning'):
                            raise CodexProtocolError('Codex completed a non-text action; result refused.')
                    finals = [i for i in items.values() if i.get('phase') == 'final_answer' and not i.get('delivery')]
                    if not finals:
                        # Phase is optional in the official protocol; last unphased
                        # agent message is the legacy terminal-answer convention.
                        finals = [i for i in items.values() if i.get('phase') is None and not i.get('delivery')][-1:]
                    text = '\n'.join(i.get('text') or deltas.get(i['id'], '') for i in finals)
                    if not text.strip():
                        raise CodexOutputFormatError('Codex completed without final answer text.')
                    if not usage:
                        LOG.warning('Codex supplied no token usage; returning zero as unknown, not a measured count.')
                    return text, usage.get('inputTokens', 0), usage.get('outputTokens', 0)
        except (KeyError, TypeError, ValueError, AttributeError):
            self.client.close()
            raise CodexProtocolError('Malformed Codex event/result schema; no partial output was accepted.') from None
        except (CodexTimeoutError, CodexProcessError, CodexProtocolError):
            # A timed-out thread/start may have created an unknown thread. Closing
            # the process also prevents late orphan turns from continuing to bill.
            self.client.close()
            raise
        except CodexError:
            if turn_id:
                try:
                    self.client.request('turn/interrupt', {'threadId': thread_id, 'turnId': turn_id}, timeout=5)
                except CodexError:
                    self.client.close()
            raise
        finally:
            if thread_id:
                self.client.unsubscribe(thread_id)
                try:
                    self.client.request('thread/unsubscribe', {'threadId': thread_id}, timeout=5)
                except CodexError:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.client:
            self.client.close()
        if self._cache:
            self._cache.close()
        self._temp.cleanup()
        atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
