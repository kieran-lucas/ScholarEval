"""Offline protocol/engine tests. No Codex login or model usage required."""

import io
import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from ScholarEval.engine.codex_app_server_engine import (
    CodexAppServerEngine, RESTRICTED_CONFIG, discover_astra, quota_state,
    translate_messages, validate_output,
)
from ScholarEval.engine import codex_session
from ScholarEval.engine.codex_transport import (
    CodexAuthError, CodexCapacityError, CodexError, CodexModelError,
    CodexOutputFormatError, CodexProcessError, CodexProtocolError,
    CodexQuotaError, CodexRateLimitError, CodexTimeoutError,
    JsonRpcClient, codex_version, server_error,
)

ASTRA = {'id': 'catalog-astra', 'model': 'gpt-6-astra', 'displayName': 'GPT-6 Astra',
         'hidden': False, 'isDefault': True, 'description': 'fixture',
         'defaultReasoningEffort': 'medium',
         'supportedReasoningEfforts': [{'reasoningEffort': x, 'description': x} for x in ('low', 'high')]}


class ReadPipe:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(timeout=10)
        if item is None:
            raise StopIteration
        return item

    def close(self):
        pass


class WritePipe:
    def __init__(self, process):
        self.process = process

    def write(self, line):
        self.process.receive(json.loads(line))

    def flush(self):
        pass

    def close(self):
        self.process.crash()


class FakeProcess:
    def __init__(self, *, account=True, models=None, turns=None, stalled=False):
        self.stdout = ReadPipe()
        self.stderr = io.StringIO('sensitive diagnostics MUST NOT be logged')
        self.stdin = WritePipe(self)
        self.requests = []
        self.exit_code = None
        self.account = account
        self.models = models if models is not None else [ASTRA]
        self.turns = list(turns or ['OK'])
        self.stalled = stalled
        self.counter = 0
        self.turn_count = 0

    def emit(self, value):
        self.stdout.queue.put(json.dumps(value) + '\n')

    def event(self, method, data):
        self.emit({'method': method, 'params': data})

    def receive(self, request):
        self.requests.append(request)
        if 'method' not in request or 'id' not in request:
            return
        method = request['method']
        result = {}
        if method == 'account/read':
            result = {'account': {'type': 'chatgpt', 'email': 'not-logged@example.test', 'planType': 'plus'} if self.account else None}
        elif method == 'model/list':
            result = {'data': self.models, 'nextCursor': None}
        elif method == 'account/rateLimits/read':
            result = {'rateLimits': {'primary': {'usedPercent': 12}, 'secondary': None}}
        elif method == 'config/read':
            result = {'config': {'mcp_servers': {'unsafe': {}}, 'apps': None, 'plugins': None}}
        elif method == 'thread/start':
            self.counter += 1
            result = {'thread': {'id': f'thread-{self.counter}'}, 'model': ASTRA['model'],
                      'modelProvider': 'openai', 'approvalPolicy': 'never', 'sandbox': {'type': 'readOnly'}}
        elif method == 'turn/start':
            self.turn_count += 1
            result = {'turn': {'id': f'turn-{self.counter}', 'status': 'inProgress', 'items': []}}
        self.emit({'id': request['id'], 'result': result})
        if method == 'turn/start' and not self.stalled:
            thread_id, turn_id = f'thread-{self.counter}', f'turn-{self.counter}'
            value = self.turns[min(self.turn_count - 1, len(self.turns) - 1)]
            if callable(value):
                value(self, thread_id, turn_id)
                return
            self.complete(thread_id, turn_id, value)

    def complete(self, thread_id, turn_id, text):
        base = {'threadId': thread_id, 'turnId': turn_id}
        final = {'id': 'answer', 'type': 'agentMessage', 'phase': 'final_answer', 'text': text}
        self.event('item/completed', dict(base, item={'id': 'thinking', 'type': 'reasoning', 'text': 'HIDDEN REASONING'}))
        self.event('item/completed', dict(base, item={'id': 'commentary', 'type': 'agentMessage', 'phase': 'commentary', 'text': 'PROGRESS'}))
        self.event('item/started', dict(base, item=dict(final, text='')))
        for part in (text[:len(text)//2], text[len(text)//2:]):
            self.event('item/agentMessage/delta', dict(base, itemId='answer', delta=part))
        self.event('item/completed', dict(base, item=final))
        self.event('thread/tokenUsage/updated', dict(base, tokenUsage={'total': {'inputTokens': 11, 'outputTokens': 7}}))
        self.event('turn/completed', {'threadId': thread_id, 'turn': {'id': turn_id, 'status': 'completed', 'items': [final]}})

    def crash(self):
        if self.exit_code is None:
            self.exit_code = 0
            self.stdout.queue.put(None)

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        self.crash()
        return self.exit_code

    def kill(self):
        self.crash()


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'SCHOLAREVAL_CODEX_MODEL': 'auto',
                                         'SCHOLAREVAL_CODEX_REASONING': 'high',
                                         'SCHOLAREVAL_CODEX_REQUEST_TIMEOUT': '1',
                                         'SCHOLAREVAL_CODEX_MAX_CONCURRENCY': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def engine(self, process=None, cache_path=False):
        process = process or FakeProcess()
        client = JsonRpcClient(process, timeout=0.2)
        self.addCleanup(client.close)
        engine = CodexAppServerEngine(client=client, status=lambda *_: None, cache_path=cache_path)
        self.addCleanup(engine.close)
        return engine, process

    def test_handshake_and_plus_preflight(self):
        engine, process = self.engine()
        self.assertEqual([r['method'] for r in process.requests[:3]], ['initialize', 'initialized', 'account/read'])
        self.assertTrue(process.requests[0]['params']['capabilities']['experimentalApi'])
        self.assertEqual(engine.status['Plan'], 'plus')
        self.assertEqual(engine.status['Astra'], 'gpt-6-astra')
        self.assertEqual(engine.status['Quota'], 'available')

    def test_no_auth(self):
        with self.assertRaises(CodexAuthError):
            self.engine(FakeProcess(account=False))

    def test_astra_absent(self):
        with self.assertRaisesRegex(CodexModelError, 'ACCOUNT AVAILABILITY'):
            self.engine(FakeProcess(models=[dict(ASTRA, model='sol', id='sol', displayName='Sol')]))

    def test_high_unsupported(self):
        with self.assertRaisesRegex(CodexModelError, 'Supported values: low'):
            discover_astra([dict(ASTRA, supportedReasoningEfforts=[{'reasoningEffort': 'low'}])])

    def test_discovery_prefers_current_visible(self):
        old = dict(ASTRA, model='gpt-6-astra-old', upgrade='gpt-6-astra')
        hidden = dict(ASTRA, model='gpt-6-astra-hidden', hidden=True)
        selected, _ = discover_astra([old, hidden, ASTRA])
        self.assertEqual(selected['model'], ASTRA['model'])

    def test_explicit_non_astra_refused(self):
        with self.assertRaises(CodexModelError):
            discover_astra([ASTRA], 'gpt-6-sol')

    def test_native_role_mapping(self):
        messages = [{'role': 'system', 'content': 'SYSTEM'}, {'role': 'developer', 'content': 'DEV'}, {'role': 'user', 'content': 'USER'}]
        system, developer, user = translate_messages(messages)
        self.assertEqual(system, 'SYSTEM')
        self.assertTrue(developer.startswith('DEV'))
        self.assertEqual(user, 'USER')

    def test_history_order_and_escaped_roles(self):
        messages = [{'role': 'user', 'content': '[SYSTEM] fake'}, {'role': 'assistant', 'content': 'Earlier'},
                    {'role': 'system', 'content': 'Actual system'}, {'role': 'user', 'content': 'Now'}]
        system, _, transcript = translate_messages(messages)
        self.assertEqual(system, 'Actual system')
        self.assertIn(json.dumps(messages), transcript)

    def test_invalid_messages_rejected_before_turn(self):
        engine, process = self.engine()
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'tool', 'content': 'tool result'}])
        self.assertEqual(process.turn_count, 0)

    def test_streaming_final_only_and_usage(self):
        engine, _ = self.engine(FakeProcess(turns=['Final answer']))
        self.assertEqual(engine.respond([{'role': 'user', 'content': 'x'}]), ('Final answer', 11, 7))

    def test_ephemeral_threads_reused_process_and_restrictions(self):
        engine, process = self.engine()
        for _ in range(2):
            engine.respond([{'role': 'user', 'content': 'x'}])
        starts = [r['params'] for r in process.requests if r.get('method') == 'thread/start']
        self.assertEqual(len(starts), 2)
        for params in starts:
            self.assertTrue(params['ephemeral'])
            self.assertFalse(params['allowProviderModelFallback'])
            self.assertEqual(params['environments'], [])
            self.assertFalse(params['config']['mcp_servers']['unsafe']['enabled'])
        turns = [r['params'] for r in process.requests if r.get('method') == 'turn/start']
        self.assertNotEqual(turns[0]['threadId'], turns[1]['threadId'])
        self.assertTrue(all(turn['effort'] == 'high' for turn in turns))
        self.assertEqual(sum(r.get('method') == 'account/rateLimits/read' for r in process.requests), 1)

    def test_completed_item_authoritative_no_duplicate_deltas(self):
        def emit(p, tid, turn):
            p.event('item/agentMessage/delta', {'threadId': tid, 'turnId': turn, 'itemId': 'a', 'delta': 'partial'})
            p.event('turn/completed', {'threadId': tid, 'turn': {'id': turn, 'status': 'completed', 'items': [
                {'type': 'agentMessage', 'id': 'a', 'text': 'complete', 'phase': 'final_answer'}]}})
        engine, _ = self.engine(FakeProcess(turns=[emit]))
        self.assertEqual(engine.respond([{'role': 'user', 'content': 'x'}])[0], 'complete')

    def test_legacy_unphased_last_message(self):
        def emit(p, tid, turn):
            p.event('turn/completed', {'threadId': tid, 'turn': {'id': turn, 'status': 'completed', 'items': [
                {'type': 'agentMessage', 'id': 'a', 'text': 'preamble'}, {'type': 'agentMessage', 'id': 'b', 'text': 'answer'}]}})
        engine, _ = self.engine(FakeProcess(turns=[emit]))
        self.assertEqual(engine.respond([{'role': 'user', 'content': 'x'}])[0], 'answer')

    def test_turn_failure_and_interruption(self):
        for status in ('failed', 'interrupted'):
            with self.subTest(status=status):
                def emit(p, tid, turn):
                    p.event('turn/completed', {'threadId': tid, 'turn': {'id': turn, 'status': status, 'error': {'message': 'SECRET'}}})
                engine, _ = self.engine(FakeProcess(turns=[emit]))
                with self.assertRaises(CodexError) as result:
                    engine.respond([{'role': 'user', 'content': 'x'}])
                self.assertNotIn('SECRET', str(result.exception))

    def test_timeout_closes_process(self):
        engine, process = self.engine(FakeProcess(stalled=True))
        engine.timeout = 0.05
        with self.assertRaises(CodexTimeoutError):
            engine.respond([{'role': 'user', 'content': 'x'}])
        self.assertIsNotNone(process.poll())

    def test_quota_failure_not_retried_and_latched(self):
        def emit(p, tid, turn):
            p.event('turn/completed', {'threadId': tid, 'turn': {'id': turn, 'status': 'failed',
                    'error': {'codexErrorInfo': 'usageLimitExceeded', 'message': 'SECRET'}}})
        engine, process = self.engine(FakeProcess(turns=[emit]))
        for _ in range(2):
            with self.assertRaises(CodexQuotaError):
                engine.respond([{'role': 'user', 'content': 'x'}])
        self.assertEqual(process.turn_count, 1)

    def test_quota_notification_blocks_new_turn(self):
        engine, process = self.engine()
        engine._notification('account/rateLimits/updated', {'rateLimits': {'primary': {'usedPercent': 100}}})
        with self.assertRaises(CodexQuotaError):
            engine.respond([{'role': 'user', 'content': 'x'}])
        self.assertEqual(process.turn_count, 0)

    def test_capacity_bounded_backoff(self):
        engine, _ = self.engine()
        with patch.object(engine, '_infer', side_effect=CodexCapacityError('capacity')) as infer, patch('time.sleep') as sleep:
            with self.assertRaises(CodexCapacityError):
                engine._with_capacity_retry([], None, time.monotonic() + 20)
            self.assertEqual(infer.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_crash(self):
        def emit(p, *_):
            p.crash()
        engine, _ = self.engine(FakeProcess(turns=[emit]))
        with self.assertRaises(CodexProcessError):
            engine.respond([{'role': 'user', 'content': 'x'}])

    def test_malformed_json_rpc(self):
        engine, process = self.engine()
        process.stdout.queue.put('not json\n')
        engine.client._reader.join(timeout=1)
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'user', 'content': 'x'}])

    def test_clean_shutdown_idempotent(self):
        engine, process = self.engine()
        engine.close()
        engine.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(engine.client._reader.is_alive())
        self.assertFalse(Path(engine._temp.name).exists())

    def test_id_correlation_out_of_order(self):
        process = FakeProcess()
        process.receive = lambda request: process.requests.append(request)
        client = JsonRpcClient(process, timeout=1)
        self.addCleanup(client.close)
        results = {}
        workers = [threading.Thread(target=lambda key=k: results.update({key: client.request(key)})) for k in ('one', 'two')]
        for worker in workers:
            worker.start()
        limit = time.monotonic() + 1
        while len(process.requests) < 2 and time.monotonic() < limit:
            time.sleep(0.001)
        for request in reversed(process.requests):
            process.emit({'id': request['id'], 'result': {'method': request['method']}})
        for worker in workers:
            worker.join(timeout=1)
        self.assertEqual(results, {'one': {'method': 'one'}, 'two': {'method': 'two'}})

    def test_json_format_retry_once_and_checkpoint(self):
        messages = [{'role': 'user', 'content': 'Return ```json\n{"query": "..."}\n```'}]
        with tempfile.TemporaryDirectory() as folder:
            engine, process = self.engine(FakeProcess(turns=['invalid', '```json\n{"query": "valid"}\n```']), str(Path(folder) / 'cache.sqlite3'))
            result = engine.respond(messages)
            self.assertEqual(result, ('```json\n{"query": "valid"}\n```', 22, 14))
            self.assertEqual(process.turn_count, 2)
            self.assertEqual(engine.respond(messages), ('```json\n{"query": "valid"}\n```', 0, 0))
            engine.close()

    def test_format_failure_distinct_and_bounded(self):
        engine, process = self.engine(FakeProcess(turns=['invalid']))
        with self.assertRaises(CodexOutputFormatError):
            engine.respond([{'role': 'user', 'content': 'Return ```json\n{}\n```'}])
        self.assertEqual(process.turn_count, 2)

    def test_no_content_repair(self):
        with self.assertRaises(CodexOutputFormatError):
            validate_output('{"score": 3,}', [{'role': 'user', 'content': '```json\n{}\n```'}])

    def test_parameter_policy(self):
        engine, process = self.engine()
        for kwargs in ({'max_tokens': 10}, {'stop': ['end']}, {'response_format': {'type': 'json_object'}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CodexProtocolError):
                engine.respond([{'role': 'user', 'content': 'x'}], **kwargs)
        self.assertEqual(process.turn_count, 0)

    def test_error_categories(self):
        for kind, cls in [('usageLimitExceeded', CodexQuotaError), ('rateLimitExceeded', CodexRateLimitError),
                          ('serverOverloaded', CodexCapacityError), ('unauthorized', CodexAuthError), ('badRequest', CodexError)]:
            self.assertIsInstance(server_error({'codexErrorInfo': kind, 'message': 'SECRET'}), cls)

    def test_quota_buckets_and_unknown(self):
        self.assertEqual(quota_state({}), 'unknown')
        self.assertEqual(quota_state({'rateLimits': {'primary': {'usedPercent': 91}}}), 'limited')
        self.assertEqual(quota_state({'rateLimits': {'primary': {'usedPercent': 1}},
                                     'rateLimitsByLimitId': {'astra': {'primary': {'usedPercent': 100}}}}), 'exhausted')

    def test_tool_attempt_refused(self):
        def emit(p, tid, turn):
            p.event('item/started', {'threadId': tid, 'turnId': turn, 'item': {'id': 'bad', 'type': 'commandExecution'}})
        engine, process = self.engine(FakeProcess(turns=[emit]))
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'user', 'content': 'x'}])
        self.assertIsNotNone(process.poll())

    def test_local_run_connection_uses_same_engine(self):
        engine, process = self.engine()
        with codex_session.CodexRun(engine=engine) as run:
            connection = run.environment()[codex_session.CONNECTION_ENV]
            for _ in range(2):
                remote = codex_session._RemoteEngine(connection)
                self.assertEqual(remote.respond([{'role': 'user', 'content': 'x'}]), ('OK', 11, 7))
        self.assertEqual(process.turn_count, 2)
        self.assertEqual(sum(r.get('method') == 'initialize' for r in process.requests), 1)

    def test_version_gate(self):
        for version, accepted in [('0.146.0', False), ('0.153.0', True), ('0.153.4', True), ('0.153.0-alpha', False)]:
            with patch('subprocess.run', return_value=Mock(returncode=0, stdout='codex-cli ' + version)):
                if accepted:
                    self.assertEqual(codex_version('codex'), version)
                else:
                    with self.assertRaises(CodexProcessError):
                        codex_version('codex')

    def test_litellm_fallback_parameter_forwarding(self):
        from ScholarEval.engine.litellm_engine import LLMEngine, LiteLLMEngine
        openai = Mock()
        openai.OpenAI.return_value.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='api answer'))], usage=Mock(prompt_tokens=2, completion_tokens=3))
        with patch.dict('sys.modules', {'openai': openai}), patch.dict(os.environ, {'SCHOLAREVAL_LLM_BACKEND': 'litellm'}):
            engine = LLMEngine('api-model', 'test-key', 'https://example.test')
            self.assertIsInstance(engine, LiteLLMEngine)
            self.assertEqual(engine.respond([], temperature=0.1, top_p=0.5, max_tokens=25), ('api answer', 2, 3))
            openai.OpenAI.return_value.chat.completions.create.assert_called_once_with(
                model='api-model', messages=[], temperature=0.1, top_p=0.5, max_tokens=25)

    def test_output_schema_forwarded_and_validated(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest('Install requirements.txt for JSON Schema tests')
        engine, process = self.engine(FakeProcess(turns=['{"ok": true}']))
        schema = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok'], 'additionalProperties': False}
        self.assertEqual(engine.respond([{'role': 'user', 'content': 'Return JSON'}], output_schema=schema)[0], '{"ok": true}')
        turn = next(r for r in process.requests if r.get('method') == 'turn/start')
        self.assertEqual(turn['params']['outputSchema'], schema)
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'user', 'content': 'x'}], output_schema={'type': 'invalid'})

    @unittest.skipUnless(Path('.codex-schema-experimental/v2/ThreadStartParams.json').exists(),
                         'Generate the installed Codex schema for optional local contract checks')
    def test_requests_match_installed_schema(self):
        import jsonschema
        engine, process = self.engine()
        engine.respond([{'role': 'user', 'content': 'x'}])
        names = {'initialize': 'v1/InitializeParams', 'thread/start': 'v2/ThreadStartParams',
                 'turn/start': 'v2/TurnStartParams', 'model/list': 'v2/ModelListParams'}
        for request in process.requests:
            name = names.get(request.get('method'))
            if name:
                schema = json.loads(Path(f'.codex-schema-experimental/{name}.json').read_text())
                self.assertFalse(list(jsonschema.Draft7Validator(schema).iter_errors(request['params'])), name)
                self.assertFalse(set(request['params']) - set(schema['properties']), name)

    def test_approval_request_rejected_without_execution(self):
        def emit(p, tid, turn):
            p.emit({'id': 'approval-1', 'method': 'item/commandExecution/requestApproval',
                    'params': {'threadId': tid, 'turnId': turn}})
        engine, process = self.engine(FakeProcess(turns=[emit]))
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'user', 'content': 'x'}])
        replies = [r for r in process.requests if r.get('id') == 'approval-1']
        # Immediate shutdown may cancel the queued rejection; neither path grants
        # approval, and the server must be stopped before any result is accepted.
        self.assertTrue(all('error' in reply for reply in replies))
        self.assertIsNotNone(process.poll())

    def test_api_auth_is_rejected(self):
        process = FakeProcess()
        original = process.receive
        def receive(request):
            if request.get('method') == 'account/read':
                process.emit({'id': request['id'], 'result': {'account': {'type': 'apiKey'}}})
            else:
                original(request)
        process.receive = receive
        with self.assertRaises(CodexAuthError):
            self.engine(process)

    def test_discovery_pagination(self):
        process = FakeProcess()
        original = process.receive
        cursors = []
        def receive(request):
            if request.get('method') == 'model/list':
                cursor = request['params']['cursor']
                cursors.append(cursor)
                process.emit({'id': request['id'], 'result': {
                    'data': [] if cursor is None else [ASTRA], 'nextCursor': 'page2' if cursor is None else None}})
            else:
                original(request)
        process.receive = receive
        self.engine(process)
        self.assertEqual(cursors, [None, 'page2'])

    def test_default_concurrency_serializes_turns(self):
        def delayed(p, tid, turn):
            timer = threading.Timer(0.03, p.complete, args=(tid, turn, 'OK'))
            timer.start()
        engine, process = self.engine(FakeProcess(turns=[delayed]))
        answers = []
        workers = [threading.Thread(target=lambda: answers.append(engine.respond([{'role': 'user', 'content': 'x'}]))) for _ in range(3)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)
        self.assertEqual(answers, [('OK', 11, 7)] * 3)
        operations = [r['method'] for r in process.requests if r.get('method') in ('turn/start', 'thread/unsubscribe')]
        self.assertEqual(operations, ['turn/start', 'thread/unsubscribe'] * 3)

    def test_model_reroute_refused(self):
        def emit(p, tid, turn):
            p.event('model/rerouted', {'threadId': tid, 'turnId': turn, 'fromModel': ASTRA['model'], 'toModel': 'other'})
            p.complete(tid, turn, 'wrong model output')
        engine, _ = self.engine(FakeProcess(turns=[emit]))
        with self.assertRaises(CodexModelError):
            engine.respond([{'role': 'user', 'content': 'x'}])

    def test_malformed_event_schema_refused(self):
        def emit(p, tid, turn):
            p.event('turn/completed', {'threadId': tid, 'turn': {}})
        engine, process = self.engine(FakeProcess(turns=[emit]))
        with self.assertRaises(CodexProtocolError):
            engine.respond([{'role': 'user', 'content': 'x'}])
        self.assertIsNotNone(process.poll())

    def test_json_fence_protects_existing_relevance_parser(self):
        import re
        raw = '{"rationale": "Compare {x} with {y}", "score": 3}'
        answer = validate_output(raw, [{'role': 'user', 'content': 'Return ```json\n{}\n```'}])
        # Exact parser regex used by the unchanged relevance_assessor.
        match = re.search(r'```json\s*(\{.*?\})\s*```', answer, re.DOTALL)
        self.assertEqual(json.loads(match.group(1)), json.loads(raw))

    def test_checkpoint_survives_engine_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = str(Path(folder) / 'responses.sqlite3')
            messages = [{'role': 'user', 'content': 'x'}]
            engine, _ = self.engine(cache_path=cache)
            engine.respond(messages)
            engine.close()
            resumed, process = self.engine(cache_path=cache)
            self.assertEqual(resumed.respond(messages), ('OK', 0, 0))
            self.assertEqual(process.turn_count, 0)
            resumed.close()

    def test_exhausted_preflight_consumes_no_turns(self):
        process = FakeProcess()
        original = process.receive
        def receive(request):
            if request.get('method') == 'account/rateLimits/read':
                process.emit({'id': request['id'], 'result': {'rateLimits': {'primary': {'usedPercent': 100}}}})
            else:
                original(request)
        process.receive = receive
        engine, _ = self.engine(process)
        with self.assertRaises(CodexQuotaError):
            engine.respond([{'role': 'user', 'content': 'uncached'}])
        self.assertEqual(process.turn_count, 0)

    def test_no_auth_data_in_status(self):
        process = FakeProcess()
        client = JsonRpcClient(process)
        records = []
        with CodexAppServerEngine(client=client, cache_path=False, status=lambda key, val: records.append((key, val))):
            pass
        self.assertNotIn('not-logged@example.test', str(records))
        self.assertNotIn('sensitive diagnostics', str(records))

    def test_unknown_backend_rejected(self):
        from ScholarEval.engine.litellm_engine import LLMEngine
        with patch.dict(os.environ, {'SCHOLAREVAL_LLM_BACKEND': 'typo'}), self.assertRaises(ValueError):
            LLMEngine()

    def test_closed_engine_does_not_replay_cache(self):
        engine, _ = self.engine()
        engine.close()
        with self.assertRaises(CodexProcessError):
            engine.respond([{'role': 'user', 'content': 'x'}])

    def test_format_instruction_wins_over_evidence_code(self):
        messages = [{'role': 'system', 'content': 'Return a parseable JSON block.'},
                    {'role': 'user', 'content': 'Research idea includes ```python\nx = 1\n```'}]
        self.assertIn('{"score": 3}', validate_output('{"score": 3}', messages))


if __name__ == '__main__':
    unittest.main()
