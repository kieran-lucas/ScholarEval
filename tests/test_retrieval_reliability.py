"""Offline failure reproductions. No Docker, network, or model calls."""
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests

from ScholarEval.utils.checkpoints import StageCheckpoint, atomic_json, checked_main
from ScholarEval.utils.grobid import GrobidService, GrobidStartupError, parse_pdfs
from ScholarEval.utils.retrieval_http import (RateLimiter, RetrievalHTTP, RetrievalError,
    RetrievalRateLimitError, RetrievalAuthError, RetrievalServerError,
    RetrievalNetworkError, RetrievalResponseError)
from ScholarEval.utils.retrieval_progress import RetrievalProgress


SEARCH = 'https://api.semanticscholar.org/graph/v1/paper/search'


def response(status=200, data=None, headers=None):
    return Mock(status_code=status, json=Mock(return_value={'data': []} if data is None else data), headers=headers or {})


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.waits = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay


class HTTPTests(unittest.TestCase):
    def client(self, responses, retries=2):
        timer = FakeTime()
        request = Mock(side_effect=responses)
        client = RetrievalHTTP('test-placeholder', request=request, limiter=Mock(slot=lambda _: nullcontext()),
                               sleep=timer.sleep, jitter=lambda: 0, max_retries=retries)
        return client, request, timer

    def test_repeated_429_is_distinct_failure_never_empty(self):
        client, request, timer = self.client([response(429)] * 3)
        with self.assertRaises(RetrievalRateLimitError) as caught:
            client.request('GET', SEARCH)
        self.assertEqual(caught.exception.attempts, 3)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(timer.waits, [2, 4])
        self.assertEqual(client.metrics['requests_succeeded'], 0)
        self.assertEqual(client.metrics['rate_limit_429'], 3)

    def test_429_then_success_preserves_result_and_header(self):
        data = {'data': [{'paperId': 'abc'}]}
        client, request, timer = self.client([response(429, headers={'Retry-After': '7'}), response(data=data)])
        self.assertEqual(client.request('GET', SEARCH).json(), data)
        self.assertEqual(timer.waits, [7])
        self.assertEqual(request.call_args.kwargs['headers']['x-api-key'], 'test-placeholder')

    def test_retry_after_http_date(self):
        client, _, timer = self.client([response(429, headers={'Retry-After': 'Thu, 01 Jan 1970 00:00:10 GMT'}), response()])
        with patch('ScholarEval.utils.retrieval_http.time.time', return_value=0):
            client.request('GET', SEARCH)
        self.assertEqual(timer.waits, [10])

    def test_excessive_retry_after_fails_without_early_retry(self):
        client, request, timer = self.client([response(429, headers={'Retry-After': '99999'})])
        with self.assertRaises(RetrievalRateLimitError):
            client.request('GET', SEARCH)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(timer.waits, [])

    def test_successful_empty_is_valid(self):
        client, _, _ = self.client([response()])
        self.assertEqual(client.request('GET', SEARCH).json(), {'data': []})
        self.assertEqual(client.metrics['requests_succeeded'], 1)

    def test_terminal_status_no_retry(self):
        for code, kind in [(400, RetrievalError), (401, RetrievalAuthError), (403, RetrievalAuthError)]:
            client, request, timer = self.client([response(code)])
            with self.assertRaises(kind):
                client.request('GET', SEARCH)
            self.assertEqual(request.call_count, 1)
            self.assertFalse(timer.waits)

    def test_server_and_network_retries(self):
        for failed in [response(x) for x in (500, 502, 503, 504)] + [requests.Timeout(), requests.ConnectionError()]:
            client, request, _ = self.client([failed, response()])
            client.request('GET', SEARCH)
            self.assertEqual(request.call_count, 2)

    def test_failure_categories(self):
        for failed, kind in [(response(503), RetrievalServerError), (requests.Timeout(), RetrievalNetworkError)]:
            client, _, _ = self.client([failed], retries=0)
            with self.assertRaises(kind):
                client.request('GET', SEARCH)

    def test_malformed_success_is_not_empty(self):
        for data in ({}, {'data': None}, {'data': [{}]}, {'error': 'unavailable'}, []):
            client, request, _ = self.client([response(data=data)])
            with self.assertRaises(RetrievalResponseError):
                client.request('GET', SEARCH)
            self.assertEqual(request.call_count, 1)
        invalid = response()
        invalid.json.side_effect = ValueError('not json')
        client, _, _ = self.client([invalid])
        with self.assertRaises(RetrievalResponseError):
            client.request('GET', SEARCH)

    def test_batch_schema_and_length(self):
        client, _, _ = self.client([response(data=[])])
        with self.assertRaises(RetrievalResponseError):
            client.request('POST', SEARCH.replace('/search', '/batch'), json={'ids': ['a']})

    def test_malformed_references_not_empty(self):
        client, _, _ = self.client([response(data={'data': [{}]})])
        with self.assertRaises(RetrievalResponseError):
            client.request('GET', 'https://api.semanticscholar.org/graph/v1/paper/p/references')

    def test_anonymous_header_omitted(self):
        with patch.dict(os.environ, {'S2_API_KEY': ''}):
            client, request, _ = self.client([response()])
            client.api_key = None
            client.request('GET', SEARCH, headers={'x-api-key': None})
        self.assertNotIn('x-api-key', request.call_args.kwargs['headers'])

    def test_shared_limiter_and_failed_attempt_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            timer = FakeTime()
            path = str(Path(tmp) / 'rate.db')
            metrics = {'total_wait_seconds': 0}
            a = RateLimiter(1.1, timer.clock, timer.sleep, path)
            b = RateLimiter(1.1, timer.clock, timer.sleep, path)
            with self.assertRaises(RuntimeError), a.slot(metrics):
                raise RuntimeError('failure')
            with b.slot(metrics):
                pass
            self.assertAlmostEqual(timer.waits[0], 1.1)


class GrobidTests(unittest.TestCase):
    def service(self):
        timer = FakeTime()
        service = GrobidService(clock=timer.clock, sleep=timer.sleep)
        service.timeout = 10
        service.interval = 3
        return service, timer

    def container(self, running=True, image=GrobidService.image):
        return {'Id': 'fixture', 'Config': {'Image': image}, 'State': {'Running': running, 'Status': 'running' if running else 'exited'},
                'HostConfig': {'PortBindings': {'8070/tcp': [{'HostPort': '8070'}]}}}

    def test_zero_pdfs_never_construct_client_or_contact_docker(self):
        with patch('ScholarEval.utils.grobid.GrobidService') as service, patch('grobid_client.grobid_client.GrobidClient') as client:
            self.assertEqual(parse_pdfs(Mock(), [], 'unused')['grobid_action'], 'not-needed')
        service.assert_not_called()
        client.assert_not_called()

    def test_delayed_start_waits_before_client(self):
        service, timer = self.service()
        service.containers = Mock(return_value=[self.container(False)])
        service.port_open = Mock(return_value=False)
        service.docker = Mock()
        service.healthy = Mock(side_effect=[False, False, True])
        metrics = service.ensure_ready()
        self.assertEqual(metrics['grobid_action'], 'started-existing')
        self.assertEqual(timer.waits, [3, 3])
        service.docker.assert_called_once_with('start', 'fixture')

    def test_healthy_running_container_reused(self):
        service, _ = self.service()
        service.containers = Mock(return_value=[self.container()])
        service.healthy = Mock(return_value=True)
        service.docker = Mock()
        self.assertEqual(service.ensure_ready()['grobid_action'], 'reused')
        service.docker.assert_not_called()

    def test_timeout_bounded_with_diagnostics(self):
        service, timer = self.service()
        service.healthy = Mock(return_value=False)
        service.diagnostics = Mock(return_value='container status: exited; exit code: 137; possible cause: Docker memory limit / OOM')
        with self.assertRaisesRegex(GrobidStartupError, '137'):
            service.wait_ready()
        self.assertEqual(timer.now, 110)
        service.diagnostics.assert_called_once()

    def test_health_requires_true_body(self):
        service, _ = self.service()
        with patch('ScholarEval.utils.grobid.requests.get', return_value=Mock(status_code=200, text='false')):
            self.assertFalse(service.healthy())

    def test_incompatible_container_never_mutated(self):
        service, _ = self.service()
        service.containers = Mock(return_value=[self.container(image='unrelated')])
        service.docker = Mock()
        with self.assertRaisesRegex(GrobidStartupError, 'incompatible'):
            service.ensure_ready()
        service.docker.assert_not_called()

    def test_unverified_process_never_mutated(self):
        service, _ = self.service()
        service.containers = Mock(return_value=[])
        service.port_open = Mock(return_value=True)
        service.docker = Mock()
        with self.assertRaisesRegex(GrobidStartupError, 'unverified'):
            service.ensure_ready()
        service.docker.assert_not_called()

    def test_diagnostic_log_tail_oom_and_secret_redaction(self):
        service, _ = self.service()
        service.container = 'fixture'
        info = self.container(False)
        info['State'].update(ExitCode=137, OOMKilled=True)
        service.docker = Mock(return_value=json.dumps([info]))
        with patch.dict(os.environ, {'S2_API_KEY': 'test-secret-placeholder'}), patch(
            'ScholarEval.utils.grobid.subprocess.run', return_value=Mock(stdout='test-secret-placeholder', stderr='OOM')):
            report = service.diagnostics()
        self.assertIn('137', report)
        self.assertIn('Docker memory limit / OOM', report)
        self.assertNotIn('test-secret-placeholder', report)

    def test_client_created_only_after_health_polling(self):
        service, timer = self.service()
        service.containers = Mock(return_value=[self.container()])
        service.healthy = Mock(side_effect=[False, True])
        s2 = Mock()
        with patch('ScholarEval.utils.grobid.GrobidService', return_value=service), patch(
            'grobid_client.grobid_client.GrobidClient', side_effect=lambda **kw: self.assertEqual(timer.waits, [3]) or Mock()) as client:
            parse_pdfs(s2, ['one.pdf'], 'fixture-dir')
        client.assert_called_once()
        s2.extract_sections_from_pdf.assert_called_once()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root / 'plan.txt'
        self.plan.write_text('the idea', encoding='utf-8')
        self.methods = self.root / 'methods.json'
        self.queries = self.root / 'queries.json'
        atomic_json(self.methods, {'clean_methods': ['method']})
        atomic_json(self.queries, {'queries': {'method': 'query'}})
        self.argv = ['--input_file', str(self.plan), '--output_file', str(self.methods), '--llm_engine_name', 'fixture']
        self.qargv = ['--research_plan', str(self.plan), '--methods_file', str(self.methods), '--output_file', str(self.queries), '--llm_engine_name', 'fixture']

    def test_resume_both_upstream_stages_never_calls_main(self):
        for stage, argv in [('extract_methods', self.argv), ('make_queries', self.qargv)]:
            module = 'ScholarEval.soundness.' + stage
            cp = StageCheckpoint(module, argv)
            cp.complete()
            main = Mock(side_effect=AssertionError('LLM must never be called'))
            with patch.object(sys, 'argv', [stage, *argv, '--resume']), patch.dict(os.environ, {'SCHOLAREVAL_NO_LLM': '1'}):
                checked_main(main, module)
            main.assert_not_called()

    def test_idea_change_recomputes(self):
        cp = StageCheckpoint('ScholarEval.soundness.extract_methods', self.argv)
        cp.complete()
        self.plan.write_text('changed idea')
        self.assertFalse(StageCheckpoint(cp.module, self.argv).valid())
        main = Mock(side_effect=lambda: atomic_json(self.methods, {'clean_methods': ['new method']}))
        with patch.object(sys, 'argv', ['stage', *self.argv, '--resume']), patch.dict(os.environ, {'SCHOLAREVAL_NO_LLM': '0'}):
            checked_main(main, cp.module)
        main.assert_called_once()

    def test_changed_methods_invalidate_queries(self):
        cp = StageCheckpoint('ScholarEval.soundness.make_queries', self.qargv)
        cp.complete()
        atomic_json(self.methods, {'clean_methods': ['other']})
        self.assertFalse(StageCheckpoint(cp.module, self.qargv).valid())

    def test_corrupt_partial_and_unproven_artifacts_rejected(self):
        cp = StageCheckpoint('ScholarEval.soundness.extract_methods', self.argv)
        self.assertFalse(cp.valid())
        cp.complete()
        for value in ('{', '{}', '{"clean_methods": []}', '{"clean_methods": [null]}'):
            self.methods.write_text(value)
            self.assertFalse(cp.valid())
        atomic_json(self.queries, {'queries': {}})
        self.assertFalse(StageCheckpoint('ScholarEval.soundness.make_queries', self.qargv).validate())

    def test_no_llm_guard_blocks_stale_stage(self):
        main = Mock()
        with patch.object(sys, 'argv', ['stage', *self.argv, '--resume']), patch.dict(os.environ, {'SCHOLAREVAL_NO_LLM': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'No-LLM guard'):
                checked_main(main, 'ScholarEval.soundness.extract_methods')
        main.assert_not_called()

    def test_top_level_resume_subprocess_zero_llm(self):
        # Genuine CLI orchestration with fixture checkpoints and no external work.
        run = self.root / 'run'
        soundness = run / 'soundness'
        soundness.mkdir(parents=True)
        plan = soundness / 'research_plan.txt'
        plan.write_bytes(self.plan.read_bytes())
        methods = soundness / 'methods.json'
        queries = soundness / 'queries.json'
        atomic_json(methods, {'clean_methods': ['method']})
        atomic_json(queries, {'queries': {'method': 'query'}})
        common = ['--llm_engine_name', 'auto']
        env = {**os.environ, 'SCHOLAREVAL_LLM_BACKEND': 'codex', 'PYTHONIOENCODING': 'utf-8'}
        with patch.dict(os.environ, env):
            StageCheckpoint('ScholarEval.soundness.extract_methods', ['--input_file', str(plan), '--output_file', str(methods), *common]).complete()
            StageCheckpoint('ScholarEval.soundness.make_queries', ['--research_plan', str(plan), '--methods_file', str(methods), '--output_file', str(queries), *common]).complete()
            atomic_json(soundness / 'snippet_references.json', {'method': []})
            atomic_json(soundness / 'snippet_papers.json', {})
            atomic_json(soundness / 'snippet_coverage.json', {'status': 'no_candidates'})
            StageCheckpoint('ScholarEval.soundness.snippet_search', ['--queries_file', str(queries), '--methods_file', str(methods),
                            '--output_file', str(soundness / 'snippet')]).complete()
        result = subprocess.run([sys.executable, '-m', 'ScholarEval.ScholarEval', '--research_idea', str(self.plan),
                                 '--save_to', str(run), '--llm_engine_name', 'auto', '--resume', '--no-llm', '--stop-after-retrieval'],
                                 env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('method extraction skipped', result.stdout)
        self.assertIn('query generation skipped', result.stdout)
        self.assertIn('snippet_search skipped', result.stdout)
        self.assertFalse((soundness / 'soundness_costs.jsonl').exists())
        self.assertFalse((run / 'codex-responses.sqlite3').exists())


class ProgressTests(unittest.TestCase):
    def test_query_resume_after_rate_exhaustion_and_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'progress.json'
            http = RetrievalHTTP(max_retries=0)
            progress = RetrievalProgress(path, 'fingerprint', http)
            progress.run('successful', lambda: {'data': []})
            def fail():
                http.on_attempt()
                raise RetrievalRateLimitError('429')
            with self.assertRaises(RetrievalRateLimitError):
                progress.run('retry', fail)
            self.assertEqual(progress.state['items']['retry']['attempts'], 1)
            self.assertEqual(progress.state['items']['retry']['status'], 'failed-retryable')
            resumed = RetrievalProgress(path, 'fingerprint', RetrievalHTTP(), resume=True)
            never = Mock(side_effect=AssertionError('repeated successful query'))
            self.assertEqual(resumed.run('successful', never), {'data': []})
            resumed.run('retry', lambda: {'data': [{'paperId': 'p'}]}, lambda d: [p['paperId'] for p in d['data']])
            self.assertEqual(resumed.state['items']['retry']['paper_ids'], ['p'])
            with self.assertRaises(KeyboardInterrupt):
                resumed.run('interrupted', Mock(side_effect=KeyboardInterrupt))
            self.assertEqual(json.loads(path.read_text())['items']['interrupted']['status'], 'pending')
            reset = RetrievalProgress(path, 'different-idea', RetrievalHTTP(), resume=True)
            self.assertFalse(reset.state['items'])

    def test_terminal_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = RetrievalProgress(Path(tmp) / 'p.json', 'f', RetrievalHTTP())
            with self.assertRaises(RetrievalAuthError):
                progress.run('q', Mock(side_effect=RetrievalAuthError('403')))
            never = Mock()
            with self.assertRaises(RetrievalError):
                progress.run('q', never)
            never.assert_not_called()

    def test_snippet_successful_zero_does_not_start_grobid(self):
        from ScholarEval.soundness.snippet_search import retrieve
        from ScholarEval.utils.semantic_scholar import SemanticScholar
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / 'm.json', {'clean_methods': ['method']})
            atomic_json(root / 'q.json', {'queries': {'method': 'query'}})
            args = Mock(pdf_dir=str(root / 'pdfs'), queries_file=str(root / 'q.json'), methods_file=str(root / 'm.json'),
                        output_file=str(root / 'snippet'), cutoff_date=None, research_title='', research_abstract='')
            s2 = SemanticScholar(None)
            s2.search_snippets = Mock(return_value={'data': []})
            progress = RetrievalProgress(root / 'progress.json', 'f', s2.http)
            with patch('ScholarEval.utils.grobid.GrobidService') as service:
                retrieve(args, s2, progress, progress.state['metrics'])
            service.assert_not_called()
            self.assertEqual(json.loads((root / 'snippet_references.json').read_text()), {'method': []})
            self.assertEqual(json.loads((root / 'snippet_papers.json').read_text()), {})

    def test_snippet_429_never_writes_success_artifacts(self):
        from ScholarEval.soundness.snippet_search import retrieve
        from ScholarEval.utils.semantic_scholar import SemanticScholar
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / 'm.json', {'clean_methods': ['method']})
            atomic_json(root / 'q.json', {'queries': {'method': 'query'}})
            args = Mock(pdf_dir=str(root / 'pdfs'), queries_file=str(root / 'q.json'), methods_file=str(root / 'm.json'),
                        output_file=str(root / 'snippet'), cutoff_date=None, research_title='', research_abstract='')
            s2 = SemanticScholar(None)
            s2.http = HTTPTests().client([response(429)] * 3)[0]
            progress = RetrievalProgress(root / 'progress.json', 'f', s2.http)
            with patch('ScholarEval.utils.grobid.GrobidService') as service, self.assertRaises(RetrievalRateLimitError):
                retrieve(args, s2, progress, progress.state['metrics'])
            service.assert_not_called()
            self.assertFalse((root / 'snippet_references.json').exists())
            self.assertFalse((root / 'snippet_papers.json').exists())


if __name__ == '__main__':
    unittest.main()
