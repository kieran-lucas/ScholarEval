"""Real durability boundaries with injected failures; no account/network required."""
import argparse
import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from ScholarEval.utils.checkpoints import StageCheckpoint, checked_main, file_hash
from ScholarEval.utils.durable import ItemStore, StateDB, atomic_json, atomic_bytes, digest
from ScholarEval.utils.retrieval_http import RetrievalHTTP, RateLimiter, retry_after_seconds, RetrievalResponseError
from ScholarEval.utils.workflow_errors import ScientificValidationError, QuotaPause
from ScholarEval.workflow import Supervisor, build_graph, required_stages
from test_retrieval_reliability import FakeTime, response, SEARCH
from test_codex_backend import FakeProcess, JsonRpcClient, CodexAppServerEngine, CodexQuotaError


class DurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'SCHOLAREVAL_STATE_DB': str(self.root/'state.sqlite3'),
            'SCHOLAREVAL_LLM_BACKEND': 'codex', 'SCHOLAREVAL_RESUME': '0',
            'SCHOLAREVAL_NO_LLM': '0', 'SCHOLAREVAL_SUPERVISED': '0',
            'SCHOLAREVAL_DIAGNOSTICS_DIR': str(self.root/'diagnostics')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.plan = self.root/'plan.txt'
        self.plan.write_text('idea ✓ tiếng Việt', encoding='utf-8')
        self.output = self.root/'methods.json'
        self.argv = ['--input_file', str(self.plan), '--output_file', str(self.output), '--llm_engine_name', 'auto']
        self.module = 'ScholarEval.soundness.extract_methods'

    def test_atomic_interruption_preserves_previous_bytes(self):
        atomic_bytes(self.output, b'old')
        with patch('ScholarEval.utils.durable.os.replace', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            atomic_bytes(self.output, b'new')
        self.assertEqual(self.output.read_bytes(), b'old')

    def test_offline_guard_rejects_unmocked_codex_startup(self):
        from ScholarEval.utils.workflow_errors import ConfigurationError
        with patch.dict(os.environ, {'SCHOLAREVAL_OFFLINE':'1'}), \
             patch('ScholarEval.engine.codex_app_server_engine.find_codex') as executable:
            with self.assertRaises(ConfigurationError):
                CodexAppServerEngine(cache_path=str(self.root/'never-created.sqlite3'))
            executable.assert_not_called()

    def test_augmentation_reuses_metadata_without_detail_amplification(self):
        import asyncio
        from ScholarEval.contribution import paper_augmentation as stage
        source = self.root/'relevant.json'
        atomic_json(source, {'papers':[{'paperId':'seed','relevance_score':4,'abstract':'x'}]})
        argv = ['--relevant_papers', str(source), '--output_file', str(self.output)]
        s2 = Mock(http=RetrievalHTTP(request=Mock(), limiter=Mock(slot=lambda _:nullcontext())))
        s2.get_recommendations_multi_seed.return_value = [{'paperId':'rec','abstract':'r'}]
        s2.get_references.return_value = [{'paperId':'ref','abstract':'f'}]
        with patch.object(stage, 'SemanticScholar', return_value=s2), patch.object(sys, 'argv', ['stage',*argv]):
            checked_main(lambda: asyncio.run(stage.main()), stage.__name__)
        s2.get_paper_details.assert_not_called()
        self.assertEqual({p['paperId'] for p in json.loads(self.output.read_text(encoding='utf-8'))}, {'seed','rec','ref'})

    def test_corrupt_item_is_quarantined_and_recomputed(self):
        store = ItemStore(self.root/'items.sqlite3', 'corruption')
        store.commit('a', {'score':4})
        with store.state.connect() as db:
            db.execute('UPDATE items SET result=?', ('{',))
        self.assertEqual(store.run('a', lambda:{'score':4}), {'score':4})
        with store.state.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM events WHERE kind='corrupt_item_quarantined'").fetchone()[0], 1)

    def test_reference_filter_pagination_uses_original_count_and_batch_positions(self):
        from ScholarEval.utils.semantic_scholar import SemanticScholar
        url = 'https://api.semanticscholar.org/graph/v1/paper/p/references'
        first = [None]*99 + [{'citedPaper':{'paperId':'first'}}]
        second = [{'citedPaper':{'paperId':'second'}}]
        client = RetrievalHTTP(request=Mock(side_effect=[response(data={'data':first}), response(data={'data':second})]),
                               limiter=Mock(slot=lambda _:nullcontext()), max_retries=0)
        s2 = SemanticScholar(None)
        s2.http = client
        self.assertEqual([p['paperId'] for p in s2.get_references('p')], ['first','second'])
        self.assertEqual(client.request_fn.call_args.kwargs['params']['offset'],100)
        data, skipped = RetrievalHTTP.filter_records([None, 7, {'paperId':'last'}], url.replace('/p/references','/batch'))
        self.assertEqual(data,[None,None,{'paperId':'last'}])
        self.assertEqual(skipped,1)

    def test_last_known_good_survives_interrupted_recomputation(self):
        atomic_json(self.output, {'clean_methods': ['old', 'complete']})
        cp = StageCheckpoint(self.module, self.argv)
        cp.complete()
        old_meta, old_output = cp.sidecar.read_bytes(), self.output.read_bytes()
        def interrupted():
            output = sys.argv[sys.argv.index('--output_file') + 1]
            atomic_json(output, {'clean_methods': ['partial']})
            raise KeyboardInterrupt
        with patch.object(sys, 'argv', ['stage', *self.argv]), self.assertRaises(KeyboardInterrupt):
            checked_main(interrupted, self.module)
        self.assertEqual(cp.sidecar.read_bytes(), old_meta)
        self.assertEqual(self.output.read_bytes(), old_output)
        self.assertTrue(cp.valid())
        attempt = json.loads(Path(str(cp.primary)+'.attempt.json').read_text(encoding='utf-8'))
        self.assertEqual(attempt['status'], 'CANCELLED')

    def test_kill_after_pointer_repairs_multiple_outputs(self):
        meta = self.root/'meta.json'
        atomic_json(meta, {'analysis': {}})
        out, md = self.root/'final.txt', self.root/'final.md'
        args = ['--input_file', str(self.plan), '--meta_review_file', str(meta), '--output_file', str(out), '--markdown_file', str(md), '--llm_engine_name', 'auto']
        cp = StageCheckpoint('ScholarEval.soundness.tldr_soundness', args)
        old = b'## Strengths\nold\n## Weaknesses\nlimits\n## Top 3 Suggestions\na,b,c'
        new = old.replace(b'old', b'new')
        atomic_bytes(out, old); atomic_bytes(md, b'old')
        cp.complete()
        candidate = cp.generations/'gen-new'
        candidate.mkdir()
        atomic_bytes(candidate/out.name, new); atomic_bytes(candidate/md.name, b'new')
        with patch('ScholarEval.utils.checkpoints.atomic_bytes', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            cp.promote(candidate)
        self.assertEqual(out.read_bytes(), old)
        self.assertTrue(cp.valid())
        self.assertEqual(out.read_bytes(), new)
        self.assertEqual(md.read_bytes(), b'new')

    def test_invalid_generation_never_promoted(self):
        atomic_json(self.output, {'clean_methods': ['complete']})
        cp = StageCheckpoint(self.module, self.argv)
        cp.complete()
        before = cp.sidecar.read_bytes()
        candidate = cp.generations/'bad'
        candidate.mkdir()
        atomic_bytes(candidate/self.output.name, b'{')
        with self.assertRaises(ScientificValidationError):
            cp.promote(candidate)
        self.assertEqual(cp.sidecar.read_bytes(), before)

    def test_operational_changes_preserve_identity_scientific_changes_do_not(self):
        cp = StageCheckpoint(self.module, self.argv)
        with patch.dict(os.environ, {'SCHOLAREVAL_GROBID_WORKERS': '9', 'SCHOLAREVAL_CODEX_REQUEST_TIMEOUT': '10', 'SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS': '100'}):
            self.assertEqual(cp.fingerprint, StageCheckpoint(self.module, self.argv+['--max_workers','6']).fingerprint)
        with patch.dict(os.environ, {'SCHOLAREVAL_CODEX_REASONING': 'low'}):
            self.assertNotEqual(cp.fingerprint, StageCheckpoint(self.module, self.argv).fingerprint)
        self.plan.write_text('different idea', encoding='utf-8')
        self.assertNotEqual(cp.fingerprint, StageCheckpoint(self.module, self.argv).fingerprint)

    def test_item_integrity_and_namespace(self):
        store = ItemStore(self.root/'items.sqlite3', 'idea1')
        store.commit('a', {'score': 4})
        operation = Mock(side_effect=AssertionError('must reuse'))
        self.assertEqual(store.run('a', operation), {'score': 4})
        operation.assert_not_called()
        with self.assertRaises(ScientificValidationError):
            store.commit('a', {'score': 0})
        with self.assertRaises(KeyError):
            ItemStore(self.root/'items.sqlite3', 'idea2').load('a')

    def test_relevance_real_loop_quota_on_third_call_resumes_two_items(self):
        from ScholarEval.contribution import relevance_assessor as stage
        from ScholarEval.engine.codex_transport import CodexQuotaError
        papers = [{'paperId': str(i), 'title': str(i), 'abstract': 'evidence'} for i in range(5)]
        input_path = self.root/'papers.json'
        atomic_json(input_path, papers)
        out = self.root/'scores.json'
        argv = ['--research_plan', str(self.plan), '--papers_file', str(input_path), '--output_file', str(out), '--llm_engine','auto']
        response_ok = ('```json\n{"score":4,"rationale":"Supported"}\n```', 1, 1)
        first = Mock(respond=Mock(side_effect=[response_ok, response_ok, CodexQuotaError('quota')]))
        with patch.object(sys, 'argv', ['stage', *argv]), patch.object(stage, 'LLMEngine', return_value=first), self.assertRaises(CodexQuotaError):
            checked_main(stage.main, stage.__name__)
        self.assertFalse(out.exists())
        second = Mock(respond=Mock(return_value=response_ok))
        with patch.object(sys, 'argv', ['stage', *argv, '--resume']), patch.object(stage, 'LLMEngine', return_value=second):
            checked_main(stage.main, stage.__name__)
        self.assertEqual(second.respond.call_count, 3)
        self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))['papers']), 5)

    def test_dag_final_dominates_damaged_history_and_tracks_idea_cutoff(self):
        for group in ('soundness', 'contribution'):
            (self.root/group).mkdir()
            atomic_bytes(self.root/group/'research_plan.txt', self.plan.read_bytes())
        graph = build_graph(self.root)
        cp = graph['s6'].checkpoint()
        for path in cp.outputs:
            if path.suffix == '.txt':
                atomic_bytes(path, b'## Strengths\nsupported\n## Weaknesses\nlimits\n## Top 3 Suggestions\na,b,c')
            else:
                atomic_bytes(path, b'validated final')
        cp.complete()
        self.assertEqual(required_stages(graph, ['s6']), [])
        atomic_bytes(self.root/'soundness'/'meta_review.json', b'broken history')
        self.assertEqual(required_stages(build_graph(self.root), ['s6']), [])
        self.assertTrue(required_stages(build_graph(self.root, cutoff='2020-01-01'), ['s6']))
        atomic_bytes(self.root/'soundness'/'research_plan.txt', b'changed idea')
        self.assertTrue(required_stages(build_graph(self.root), ['s6']))

    def test_kill_restart_actual_process_commits_only_finished_items(self):
        script = self.root/'worker.py'
        script.write_text('''
import os, sys
from pathlib import Path
from ScholarEval.utils.durable import ItemStore, atomic_json
from ScholarEval.utils.checkpoints import checked_main
def main():
    store = ItemStore.current('kill-test')
    for i in range(5):
        store.run(i, lambda: {'method': str(i)})
        if i == 2 and os.environ.get('INJECT_KILL') == '1': os._exit(99)
    atomic_json(sys.argv[sys.argv.index('--output_file')+1], {'clean_methods': [str(i) for i in range(5)]})
checked_main(main, 'ScholarEval.soundness.extract_methods')
''', encoding='utf-8')
        env = dict(os.environ, PYTHONPATH=str(Path.cwd()), INJECT_KILL='1')
        first = subprocess.run([sys.executable, str(script), *self.argv], env=env, capture_output=True)
        self.assertEqual(first.returncode, 99)
        self.assertFalse(self.output.exists())
        env['INJECT_KILL'] = '0'
        second = subprocess.run([sys.executable, str(script), *self.argv, '--resume'], env=env, capture_output=True)
        self.assertEqual(second.returncode, 0, second.stderr.decode('utf-8'))
        summary = StateDB(self.root/'state.sqlite3').summary(self.root/'summary.json')
        self.assertEqual(summary['metrics']['workflow.item_commits'], 5)
        self.assertEqual(summary['metrics']['workflow.item_cache_hits'], 3)

    def test_s2_twenty_429_three_503_then_malformed_records_recovers(self):
        timer = FakeTime()
        raw = {'data': [None, {}, {'citedPaper': None}, {'citedPaper': {}}, {'citedPaper': {'paperId': 'good'}}]}
        request = Mock(side_effect=[response(429)]*20 + [response(503)]*3 + [response(data=raw)])
        limiter = RateLimiter(1, timer.clock, timer.sleep, str(self.root/'rate.sqlite3'))
        client = RetrievalHTTP(request=request, limiter=limiter, sleep=timer.sleep, jitter=lambda: 0, max_retries=2)
        supervisor = Supervisor(self.root, sleep=timer.sleep, clock=timer.clock)
        values = []
        def operation():
            values.append(client.request('GET', 'https://api.semanticscholar.org/graph/v1/paper/p/references').json())
            return 0
        self.assertTrue(supervisor.recover('s2-chaos', operation))
        self.assertEqual(request.call_count, 24)
        self.assertEqual(client.metrics['malformed_records_skipped'], 4)
        self.assertEqual(values[0]['data'][0]['citedPaper']['paperId'], 'good')
        self.assertGreater(sum(timer.waits), 0)
        with limiter.connection() as db:
            self.assertEqual(db.execute('SELECT state FROM adaptive').fetchone()[0], 'CLOSED')

    def test_retry_after_and_global_circuit_survive_client_restart(self):
        self.assertEqual(retry_after_seconds('NaN'), 0)
        self.assertEqual(retry_after_seconds('Thu, 01 Jan 1970 00:00:10 GMT', now=0), 10)
        timer = FakeTime()
        path = str(self.root/'rate.sqlite3')
        limiter = RateLimiter(1, timer.clock, timer.sleep, path)
        for _ in range(3): limiter.observe(429, 999)
        with limiter.connection() as db:
            self.assertEqual(db.execute('SELECT state FROM adaptive').fetchone()[0], 'OPEN')
        other = RateLimiter(1, timer.clock, timer.sleep, path)
        with other.slot({'total_wait_seconds': 0}):
            with other.connection() as db:
                self.assertEqual(db.execute('SELECT state FROM adaptive').fetchone()[0], 'HALF_OPEN')
        self.assertGreaterEqual(timer.now, 1099)

    def test_s2_cache_identity_and_sanitized_quarantine(self):
        with patch.dict(os.environ, {'SCHOLAREVAL_S2_CACHE_DB': str(self.root/'s2.sqlite3')}):
            first = RetrievalHTTP(api_key='VERY-SECRET', request=Mock(return_value=response(data={'data':[{'paperId':'p'}]})), limiter=Mock(slot=lambda _:nullcontext()))
            first.request('GET', SEARCH, params={'query':'idea'}, timeout=2)
            second = RetrievalHTTP(request=Mock(side_effect=AssertionError('cache missed')), limiter=Mock(slot=lambda _:nullcontext()))
            second.request('GET', SEARCH, params={'query':'idea'}, timeout=99)
            self.assertEqual(second.metrics['cache_hits'], 1)
            bad = RetrievalHTTP(api_key='VERY-SECRET', request=Mock(return_value=response(data={'bad':'VERY-SECRET'})), limiter=Mock(slot=lambda _:nullcontext()), max_retries=0)
            with self.assertRaises(RetrievalResponseError): bad.request('GET', SEARCH, params={'query':'bad'})
            snapshot = next((self.root/'diagnostics').glob('*.json')).read_text(encoding='utf-8')
            self.assertNotIn('VERY-SECRET', snapshot)
            self.assertIn('[REDACTED]', snapshot)

    def test_cached_codex_response_available_during_quota_pause(self):
        process = FakeProcess(turns=['valid answer'])
        engine = CodexAppServerEngine(client=JsonRpcClient(process), status=lambda *_:None, cache_path=str(self.root/'codex.sqlite3'))
        self.addCleanup(engine.close)
        prompt = [{'role':'user','content':'Question'}]
        engine.respond(prompt)
        engine._failure = CodexQuotaError('quota')
        self.assertEqual(engine.respond(prompt), ('valid answer', 0, 0))
        with self.assertRaises(CodexQuotaError): engine.respond([{'role':'user','content':'Another'}])
        self.assertEqual(process.turn_count, 1)

    def test_codex_cache_integrity_preserves_evidence_without_new_calls(self):
        process = FakeProcess(turns=['valid answer'])
        engine = CodexAppServerEngine(client=JsonRpcClient(process), status=lambda *_:None,
                                     cache_path=str(self.root/'integrity.sqlite3'))
        self.addCleanup(engine.close)
        prompt = [{'role':'user','content':'Question'}]
        engine.respond(prompt)
        with engine._cache:
            engine._cache.execute("UPDATE responses SET text='altered answer'")
        with self.assertRaises(ScientificValidationError):
            engine.respond(prompt)
        self.assertEqual(engine._cache.execute('SELECT text FROM responses').fetchone()[0], 'altered answer')
        raw_key = engine._cache.execute('SELECT key FROM completed_turns').fetchone()[0]
        with engine._cache:
            engine._cache.execute('DELETE FROM completed_turns')
        with self.assertRaises(ScientificValidationError):
            engine._cached_text('completed_turns', raw_key)
        self.assertEqual(process.turn_count, 1)

    def test_codex_legacy_cache_attestation_preserves_exact_response(self):
        process = FakeProcess(turns=['valid answer'])
        engine = CodexAppServerEngine(client=JsonRpcClient(process), status=lambda *_:None,
                                     cache_path=str(self.root/'legacy.sqlite3'))
        self.addCleanup(engine.close)
        prompt = [{'role':'user','content':'Question'}]
        engine.respond(prompt)
        before = engine._cache.execute('SELECT key,text FROM responses').fetchall()
        with engine._cache:
            engine._cache.execute('DELETE FROM cache_integrity')
        engine._failure = CodexQuotaError('quota')
        self.assertEqual(engine.respond(prompt), ('valid answer', 0, 0))
        self.assertEqual(engine._cache.execute('SELECT key,text FROM responses').fetchall(), before)
        self.assertEqual(engine._cache.execute('SELECT count(*) FROM cache_integrity').fetchone()[0], 1)
        self.assertEqual(process.turn_count, 1)

    def test_no_llm_stage_refusal_has_configuration_exit_and_diagnostic(self):
        main = Mock()
        with patch.dict(os.environ, {'SCHOLAREVAL_NO_LLM':'1', 'SCHOLAREVAL_SUPERVISED':'1'}), \
             patch.object(sys, 'argv', ['stage', *self.argv]), self.assertRaises(SystemExit) as caught:
            checked_main(main, self.module)
        self.assertEqual(caught.exception.code, 79)
        main.assert_not_called()
        attempt = json.loads(Path(str(self.output)+'.attempt.json').read_text(encoding='utf-8'))
        self.assertEqual(attempt['status'], 'BLOCKED_CONFIG')
        self.assertFalse(self.output.exists())

    def test_quota_wait_and_schema_escalation_are_bounded(self):
        timer = FakeTime()
        supervisor = Supervisor(self.root, sleep=timer.sleep, clock=timer.clock)
        operation = Mock(side_effect=[QuotaPause('quota'), 0])
        self.assertTrue(supervisor.recover('quota', operation))
        bad = Mock(side_effect=RetrievalResponseError('breaking schema'))
        self.assertFalse(supervisor.recover('schema', bad))
        self.assertEqual(bad.call_count, 3)
        self.assertEqual(supervisor.last_failure.state, 'FATAL')
        self.assertEqual(supervisor.last_failure.reason, 'repeated_schema_failure')
        supervisor.quota_seconds = 1
        self.assertFalse(supervisor.recover('quota-expired', Mock(side_effect=QuotaPause('quota'))))
        self.assertEqual(supervisor.last_failure.state, 'BLOCKED_CONFIG')
        self.assertFalse(supervisor.last_failure.retryable)

    def test_main_package_text_io_has_explicit_utf8(self):
        violations = []
        for path in Path('ScholarEval').rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call): continue
                fn = n.func
                is_open = isinstance(fn, ast.Name) and fn.id == 'open'
                is_text = isinstance(fn, ast.Attribute) and fn.attr in {'read_text', 'write_text'}
                if not (is_open or is_text): continue
                mode = n.args[1] if is_open and len(n.args)>1 else None
                if isinstance(mode, ast.Constant) and 'b' in str(mode.value): continue
                if not any(k.arg == 'encoding' for k in n.keywords): violations.append(f'{path}:{n.lineno}')
        self.assertEqual(violations, [])

    def test_full_17_stage_chaos_actual_stage_mains(self):
        """Run actual science-stage code with deterministic fake external services."""
        import asyncio
        import importlib
        from ScholarEval.utils import semantic_scholar
        from ScholarEval.utils.grobid import GrobidService
        for group in ('soundness', 'contribution'):
            (self.root/group).mkdir()
            atomic_bytes(self.root/group/'research_plan.txt', self.plan.read_bytes())
        def paper(pid):
            return {'paperId': pid, 'title': 'Paper '+pid, 'abstract': 'Evidence '+pid,
                    'authors':[{'name':'Researcher'}], 'publicationDate':'2020-01-01',
                    'venue':'Journal', 'citationCount':1, 'openAccessPdf':None, 'url':'https://example.test/'+pid}
        timer = FakeTime()
        faults = {'429':0, '503':0, 'malformed':0, 'kill':0, 'restart':0, 'grobid':0}
        def request(method, url, **kwargs):
            if '/recommendations/' in url:
                if faults['429'] < 20:
                    faults['429'] += 1
                    return response(429)
                if faults['503'] < 3:
                    faults['503'] += 1
                    return response(503)
                return response(data={'recommendedPapers':[paper('r'+str(i)) for i in range(3)]})
            if url.endswith('/references'):
                faults['malformed'] += 1
                return response(data={'data':[None, {'citedPaper':None}, {}, {'citedPaper':paper('r3')}, {'citedPaper':paper('r4')}]})
            if '/snippet/' in url:
                return response(data={'data':[{'paper':{'corpusId':str(i)}, 'snippet':{'text':'evidence', 'annotations':{'refMentions':[]}}} for i in range(3)]})
            if url.endswith('/batch'):
                return response(data=[paper(pid.split(':')[-1]) for pid in kwargs['json']['ids']])
            if '/search' in url:
                return response(data={'data':[paper('p0'),paper('p1')]})
            raise AssertionError('Unexpected external request: '+url)
        limiter = RateLimiter(0.1, timer.clock, timer.sleep, str(self.root/'rate.sqlite3'))
        def http(*args, **kwargs):
            return RetrievalHTTP(request=request, limiter=limiter, sleep=timer.sleep, jitter=lambda:0, max_retries=2)
        scientific = {
            's1': "```python\n['Method']\n```",
            's2': '```json\n{"query":"method search"}\n```',
            's4': json.dumps({'method':'Detailed experimental method supported by the supplied evidence record.',
                              'results':'Quantitative results remain uncertain without complete full text.',
                              'context':'The abstract is the available evidence for this fixture.'}),
            's5': json.dumps({'support':'Evidence supports a limited claim.', 'contradictions':'Unknown.',
                              'suggested_action':'Validate with an experiment.', 'soundness_score':5}),
            's6': json.dumps({'strengths_summary':'Limited supporting evidence.', 'weaknesses_summary':'Full text unavailable.',
                              'top_3_suggestions':['Validate','Measure','Replicate']}),
            'c1': '```json\n{"Dimension":["Contribution"]}\n```',
            'c2': 'focused contribution query',
            'c4': '```json\n{"score":4,"rationale":"Related contribution"}\n```',
            'c7': '```json\n{"score":4,"rationale":"Related contribution"}\n```',
            'c9': json.dumps({'overall_comparison':'Partial novelty overlap.', 'dimension_comparisons':{
                'Dimension':{'comparison':'The idea adds a distinct hypothesis.', 'score':1}}}),
            'c11': 'Final contribution review with evidence limitations.',
        }
        attempts = {}
        killed_script = self.root/'kill_relevance.py'
        killed_script.write_text('''
import json, os, sys
from ScholarEval.contribution import relevance_assessor as stage
from ScholarEval.utils.durable import ItemStore
from ScholarEval.utils.checkpoints import checked_main
class Engine:
    def __init__(self, **kwargs): pass
    def respond(self, *args, **kwargs): return ('```json\\n{"score":4,"rationale":"Related contribution"}\\n```',1,1)
stage.LLMEngine = Engine
original = ItemStore.commit
count = 0
def commit(self, *args, **kwargs):
    global count
    value = original(self, *args, **kwargs)
    count += 1
    if count == 2: os._exit(99)
    return value
ItemStore.commit = commit
checked_main(stage.main, stage.__name__)
''', encoding='utf-8')
        def execute(stage, env):
            attempts[stage.id] = attempts.get(stage.id, 0) + 1
            attempt = attempts[stage.id]
            if stage.id == 'c7' and attempt == 1:
                result = subprocess.run([sys.executable, str(killed_script), *stage.argv],
                    env=dict(env, PYTHONPATH=str(Path.cwd())), capture_output=True)
                self.assertEqual(result.returncode, 99, result.stderr.decode('utf-8'))
                faults['kill'] += 1
                return 75
            if stage.id == 's3':
                service = GrobidService(clock=timer.clock, sleep=timer.sleep)
                container = {'Id':'fixture', 'Name':'/scholareval-grobid', 'Config':{'Image':service.image},
                    'State':{'Running':True}, 'HostConfig':{'PortBindings':{'8070/tcp':[{'HostPort':str(service.port)}]}}}
                service.containers = Mock(return_value=[container])
                service.healthy = Mock(side_effect=[False,True])
                service.docker = Mock()
                service.ensure_ready()
                service.docker.assert_called_once_with('restart','fixture')
                faults['grobid'] += 1
            module = importlib.import_module(stage.module)
            turns = [scientific.get(stage.id, 'OK')]
            if stage.id == 'c7' and attempt == 2:
                def quota(process, thread, turn):
                    process.event('turn/completed', {'threadId':thread,'turn':{'id':turn,'status':'failed',
                        'error':{'codexErrorInfo':'usageLimitExceeded'}}})
                turns = [scientific['c7'], scientific['c7'], quota]
            if stage.id == 'c9' and attempt == 1:
                def disconnect(process, thread, turn): process.crash()
                turns = [disconnect]
                faults['restart'] += 1
            engine = None
            if hasattr(module, 'LLMEngine') and stage.id not in {'c6','c10'}:
                engine = CodexAppServerEngine(client=JsonRpcClient(FakeProcess(turns=turns)), status=lambda *_:None,
                                              cache_path=str(self.root/'codex.sqlite3'))
            engine_patch = patch.object(module, 'LLMEngine', return_value=engine) if engine else nullcontext()
            citation_patch = patch('ScholarEval.utils.citation_check.LLMEngine', return_value=engine) if engine else nullcontext()
            embedding_patch = patch.object(module, 'get_thread_client', return_value=Mock(
                embeddings=Mock(create=Mock(return_value=Mock(data=[Mock(embedding=[1.0,0.5])]))))) if stage.id=='c6' else nullcontext()
            try:
                with patch.dict(os.environ, env), patch.object(sys, 'argv', [stage.module,*stage.argv]), engine_patch, embedding_patch, citation_patch:
                    try:
                        main = (lambda: asyncio.run(module.main())) if stage.id == 'c5' else module.main
                        checked_main(main, stage.module)
                        return 0
                    except SystemExit as error:
                        return error.code
            finally:
                if engine: engine.close()
        env = {'SCHOLAREVAL_OFFLINE':'1', 'SCHOLAREVAL_S2_CACHE_DB':str(self.root/'s2.sqlite3'), 'SCHOLAREVAL_CODEX_CACHE':str(self.root/'codex.sqlite3'),
               'SCHOLAREVAL_CODEX_MAX_CONCURRENCY':'1', 'SCHOLAREVAL_CODEX_REASONING':'high',
               'SCHOLAREVAL_FULL_TEXT_POLICY':'available-evidence', 'API_KEY':'fixture', 'API_ENDPOINT':'https://example.test'}
        with patch.dict(os.environ, env), patch.object(semantic_scholar, 'RetrievalHTTP', side_effect=http), \
             patch('ScholarEval.engine.codex_session.run_environment', return_value={}), \
             patch('ScholarEval.engine.codex_session.close_run'):
            graph = build_graph(self.root)
            supervisor = Supervisor(self.root, sleep=timer.sleep, clock=timer.clock, execute=execute)
            for stage in required_stages(graph, ['s6','c11'], resume=False):
                self.assertTrue(supervisor.run_stage(stage, os.environ.copy()), stage.id)
            self.assertEqual(required_stages(build_graph(self.root), ['s6','c11']), [])
        self.assertEqual(faults['429'], 20)
        self.assertEqual(faults['503'], 3)
        self.assertGreater(faults['malformed'], 0)
        self.assertEqual(faults['kill'], 1)
        self.assertEqual(faults['restart'], 1)
        self.assertEqual(faults['grobid'], 1)
        self.assertEqual(attempts['c7'], 3)
        self.assertEqual(len(attempts), 17)
        self.assertTrue((self.root/'contribution'/'contribution_review.txt').exists())


if __name__ == '__main__':
    unittest.main()
