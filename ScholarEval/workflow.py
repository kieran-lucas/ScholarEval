"""Explicit stage DAG and bounded durable subprocess supervision."""
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Callable

from .utils.checkpoints import StageCheckpoint, OUTPUTS, IGNORED, file_hash, semantic_config
from .utils.durable import StateDB, atomic_bytes, atomic_json, digest, run_lock, telemetry
from .utils.workflow_errors import ConfigurationError, Outcome, classify, from_exit, diagnostic


@dataclass
class Stage:
    id: str
    module: str
    argv: list[str]
    dependencies: tuple[str, ...] = ()
    boundary: str = ''

    def checkpoint(self) -> StageCheckpoint:
        cp = StageCheckpoint(self.module, self.argv)
        cp.boundary = self.boundary
        return cp


def build_graph(save_dir, model='auto', cutoff=None, litellm=None) -> dict[str, Stage]:
    root = Path(save_dir)
    graph = {}

    def add(group, name, module, deps, arguments):
        folder = root / group
        argv = []
        for key, value in arguments.items():
            if value is not None:
                argv.extend(['--' + key, str(value)])
        if 'llm_engine' in arguments or 'llm_engine_name' in arguments:
            argv += ['--cost_log_file', str(folder / (group + '_costs.jsonl'))]
            if litellm:
                argv += ['--litellm_name', litellm]
        graph[name] = Stage(name, f'ScholarEval.{group}.{module}', argv, tuple(deps))

    s = root / 'soundness'
    c = root / 'contribution'
    sp, cp = s / 'research_plan.txt', c / 'research_plan.txt'
    add('soundness', 's1', 'extract_methods', [], dict(input_file=sp, output_file=s/'methods.json', llm_engine_name=model))
    add('soundness', 's2', 'make_queries', ['s1'], dict(research_plan=sp, methods_file=s/'methods.json', output_file=s/'queries.json', llm_engine_name=model))
    add('soundness', 's3', 'snippet_search', ['s1', 's2'], dict(queries_file=s/'queries.json', methods_file=s/'methods.json', output_file=s/'snippet', pdf_dir=s/'pdfs', progress_file=s/'snippet_progress.json', cutoff_date=cutoff))
    add('soundness', 's4', 'methods_and_results_synthesis', ['s3'], dict(research_plan=sp, methods_and_ref_file=s/'snippet_references.json', ref_and_paper_file=s/'snippet_papers.json', output_file=s/'methods_analysis.json', llm_engine_name=model))
    add('soundness', 's5', 'meta_review', ['s3', 's4'], dict(research_plan=sp, mr_analysis_file=s/'methods_analysis.json', methods_and_ref_file=s/'snippet_references.json', output_file=s/'meta_review.json', markdown_output=s/'meta_review.md', llm_engine_name=model))
    add('soundness', 's6', 'tldr_soundness', ['s5'], dict(input_file=sp, meta_review_file=s/'meta_review.json', output_file=s/'tldr_soundness.txt', markdown_file=s/'tldr_soundness.md', llm_engine_name=model))
    add('contribution', 'c1', 'extract_dimensions_and_contributions', [], dict(input_file=cp, output_file=c/'dimensions_contributions.jsonl', llm_engine=model))
    add('contribution', 'c2', 'queries_generator', ['c1'], dict(research_plan=cp, contrib_file=c/'dimensions_contributions.jsonl', output_file=c/'contribution_queries.json', llm_engine_name=model))
    add('contribution', 'c3', 'paper_extractor', ['c2'], dict(queries_file=c/'contribution_queries.json', output_file=c/'contribution_papers.json', progress_file=c/'paper_progress.json', cutoff_date=cutoff))
    add('contribution', 'c4', 'relevance_assessor', ['c3'], dict(research_plan=cp, papers_file=c/'contribution_papers.json', output_file=c/'filtered_contribution_papers.json', llm_engine=model))
    add('contribution', 'c5', 'paper_augmentation', ['c4'], dict(relevant_papers=c/'filtered_contribution_papers.json', output_file=c/'augmented_contribution_papers.json', cutoff_date=cutoff))
    add('contribution', 'c6', 'embedding_filter', ['c5'], dict(research_plan=cp, papers_json=c/'augmented_contribution_papers.json', output=c/'filtered_augmented_contribution_papers.json', top_k=100))
    add('contribution', 'c7', 'relevance_assessor', ['c6'], dict(research_plan=cp, papers_file=c/'filtered_augmented_contribution_papers.json', output_file=c/'final_contribution_papers.json', llm_engine=model))
    add('contribution', 'c8', 'paper_sampler', ['c7'], dict(input_file=c/'final_contribution_papers.json', output_file=c/'sampled_final_contribution_papers.json'))
    add('contribution', 'c9', 'pairwise_comparator', ['c1', 'c8'], dict(research_plan=cp, papers_metadata=c/'sampled_final_contribution_papers.json', dimensions_file=c/'dimensions_contributions.jsonl', output_file=c/'pairwise_comparisons.json', llm_engine=model))
    add('contribution', 'c10', 'prepare_final_contribution_context', ['c9'], dict(input_file=c/'pairwise_comparisons.json', output_file=c/'contribution_context.json'))
    add('contribution', 'c11', 'contribution_review_synthesis', ['c10'], dict(research_plan=cp, comparisons_file=c/'contribution_context.json', output_file=c/'contribution_review.txt', llm_engine=model))
    plan = file_hash(sp) if sp.exists() else 'missing-required-plan'
    for stage in graph.values():
        opts = dict(zip(stage.argv[::2], stage.argv[1::2]))
        scientific = {k: v for k, v in opts.items() if k not in OUTPUTS | IGNORED
                      and k in {'--top_k', '--cutoff_date'}}
        stage.boundary = digest({'plan': plan, 'semantics': semantic_config(stage.module, opts),
                                 'parameters': scientific,
                                 'dependencies': [graph[d].boundary for d in stage.dependencies]})
    return graph


def required_stages(graph: dict[str, Stage], targets: list[str], resume: bool = True) -> list[Stage]:
    needed, seen = [], set()
    def visit(name):
        if name in seen:
            return
        seen.add(name)
        stage = graph[name]
        if resume and stage.checkpoint().valid():
            print(f'[resume] {name} {stage.module.rsplit(".", 1)[-1]} valid; dependency boundary', flush=True)
            state = telemetry()
            if state:
                state.stage(name, 'COMPLETED', reused=True, boundary=stage.boundary)
            return
        for dependency in stage.dependencies:
            visit(dependency)
        needed.append(stage)
    for target in targets:
        visit(target)
    return needed


class Supervisor:
    def __init__(self, root, *, sleep=time.sleep, clock=time.monotonic, execute=None):
        self.root = Path(root)
        self.state = StateDB(self.root / 'workflow.sqlite3')
        self.sleep, self.clock = sleep, clock
        self.execute = execute or self.subprocess
        self.last_failure = None
        self.max_seconds = float(os.environ.get('SCHOLAREVAL_RECOVERY_MAX_SECONDS', '86400'))
        self.quota_seconds = float(os.environ.get('SCHOLAREVAL_QUOTA_MAX_WAIT_SECONDS', '1209600'))
        self.retry_base = float(os.environ.get('SCHOLAREVAL_RETRY_SECONDS', '30'))
        self.quota_base = float(os.environ.get('SCHOLAREVAL_QUOTA_POLL_SECONDS', '900'))
        values = (self.max_seconds, self.quota_seconds, self.retry_base, self.quota_base)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ConfigurationError('Supervisor waits/budgets must be positive and finite')

    def transition(self, stage: str, status: str, **detail) -> None:
        self.state.stage(stage, status, **detail)
        self.state.summary(self.root / 'run-summary.json')
        reason = ': ' + detail['reason'] if detail.get('reason') else ''
        print(f'[workflow] {stage}: {status}{reason}', flush=True)

    def wait(self, seconds: float) -> None:
        # Heartbeat while waiting; Ctrl+C remains responsive.
        remaining = seconds
        while remaining > 0:
            delay = min(remaining, 60)
            self.sleep(delay)
            remaining -= delay
            if remaining > 0:
                print(f'[workflow] waiting; next retry in {remaining:.0f}s', flush=True)

    def recover(self, name: str, operation: Callable[[], int], restart: Callable[[], None] | None = None) -> bool:
        started, failures, last_reason, repeated = self.clock(), 0, None, 0
        while True:
            self.transition(name, 'RUNNING', attempt=failures + 1)
            try:
                code = operation()
                outcome = from_exit(code) if code else None
            except BaseException as error:
                outcome = classify(error)
                details = diagnostic(error)
                self.state.event('failure', stage=name, **details)
                print(f'[recovery] {details["message"]}', file=sys.stderr, flush=True)
                if isinstance(error, KeyboardInterrupt):
                    self.transition(name, 'CANCELLED', reason=outcome.reason)
                    raise
            if outcome is None:
                self.transition(name, 'COMPLETED')
                return True
            failures += 1
            repeated = repeated + 1 if last_reason == outcome.reason else 1
            last_reason = outcome.reason
            self.transition(name, outcome.state, reason=outcome.reason, exit_code=outcome.exit_code)
            quota = outcome.state == 'WAITING_QUOTA'
            budget = self.quota_seconds if quota else self.max_seconds
            if (not outcome.retryable or self.clock() - started >= budget
                    or outcome.exit_code == 80 and repeated >= 3):
                if outcome.retryable:
                    reason = 'repeated_schema_failure' if outcome.exit_code == 80 and repeated >= 3 else 'recovery_budget_exhausted'
                    previous_reason = outcome.reason
                    outcome = Outcome('BLOCKED_CONFIG' if quota else 'FATAL', 79 if quota else 70, False, reason)
                    self.transition(name, outcome.state, reason=reason, last_failure=previous_reason)
                self.last_failure = outcome
                return False
            base, cap = (self.quota_base, 3600) if quota else (self.retry_base, 900)
            delay = min(cap, base * 2 ** min(failures - 1, 8), budget - (self.clock() - started))
            self.state.metric('workflow.retry_loops')
            if quota:
                self.state.metric('codex.quota_pauses')
            print(f'[recovery] {outcome.reason}; retry in {delay:.0f}s; completed work preserved', flush=True)
            self.wait(delay)
            if restart:
                restart()

    @staticmethod
    def subprocess(stage: Stage, env: dict[str, str]) -> int:
        # Inherit stdout/stderr for live progress; never buffer an entire long stage.
        process = subprocess.Popen([sys.executable, '-m', stage.module, *stage.argv], env=env)
        try:
            code = process.wait()
            if code:
                attempt_path = Path(str(stage.checkpoint().primary) + '.attempt.json')
                try:
                    attempt = json.loads(attempt_path.read_text(encoding='utf-8'))
                    if attempt.get('status') == 'RUNNING':
                        return 75  # Abrupt child death; OS locks have been released.
                except (OSError, ValueError):
                    pass
            return code
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

    def run_stage(self, stage: Stage, env: dict[str, str]) -> bool:
        from .engine.codex_session import close_run, run_environment
        attempt_env = dict(env, SCHOLAREVAL_SUPERVISED='1', SCHOLAREVAL_STAGE_BOUNDARY=stage.boundary,
                           SCHOLAREVAL_STATE_DB=str(self.state.path.resolve()))
        def operation():
            if stage.checkpoint().valid() and attempt_env.get('SCHOLAREVAL_RESUME') == '1':
                return 0
            if attempt_env.get('SCHOLAREVAL_NO_LLM') != '1':
                attempt_env.update(run_environment(cache_path=env.get('SCHOLAREVAL_CODEX_CACHE')))
            return self.execute(stage, attempt_env)
        def restart():
            close_run()
            attempt_env.pop('SCHOLAREVAL_CODEX_CONNECTION', None)
            attempt_env['SCHOLAREVAL_RESUME'] = '1'
            self.state.metric('codex.process_restarts')
        return self.recover(stage.id, operation, restart)


def preflight(root: Path | str, stages: list[Stage], no_llm: bool = False, offline: bool = False) -> None:
    """No inference, no quota probes with prompts; fail before expensive work."""
    from importlib.util import find_spec
    root = Path(root)
    probe = root / '.write-probe'
    atomic_bytes(probe, 'UTF-8 kiểm tra ✓'.encode('utf-8'))
    if probe.read_text(encoding='utf-8') != 'UTF-8 kiểm tra ✓':
        raise ConfigurationError('UTF-8 filesystem check failed')
    probe.unlink()
    if shutil.disk_usage(root).free < 512 * 1024 * 1024:
        raise ConfigurationError('Less than 512 MiB free in run directory')
    modules = ['requests', 'numpy', 'litellm', 'aiohttp', 'grobid_client', 'sklearn',
               'PyPDF2', 'dateutil', 'pandas', 'jsonschema']
    missing = [m for m in modules if find_spec(m) is None]
    if stages and missing:
        raise ConfigurationError('Missing Python dependencies: ' + ', '.join(missing))
    for setting in ('SCHOLAREVAL_CODEX_CACHE', 'SCHOLAREVAL_S2_CACHE_DB'):
        cache = os.environ.get(setting)
        if not cache:
            continue
        cache_path = Path(cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists() and (no_llm or offline):
            probe = cache_path.with_name('.' + cache_path.name + '.write-probe')
            atomic_bytes(probe, b'cache directory writable')
            probe.unlink()
            continue
        try:
            db = sqlite3.connect(cache_path, timeout=10)
            try:
                db.execute('BEGIN IMMEDIATE')
                db.execute('CREATE TABLE IF NOT EXISTS _scholareval_write_probe (id INTEGER)')
                db.execute('INSERT INTO _scholareval_write_probe VALUES(1)')
                db.rollback()  # Verify writes without changing existing cache data/schema.
            finally:
                db.close()
        except sqlite3.Error as error:
            raise ConfigurationError(f'{setting} is not a writable SQLite cache') from error
    if offline:
        print('[preflight] local checks PASS; external checks omitted for offline dry run', flush=True)
        return
    ids = {s.id for s in stages}
    if ids & {'s3', 's5', 'c3', 'c5'} and not os.environ.get('S2_API_KEY'):
        raise ConfigurationError('S2_API_KEY is required for this full workflow; set it in the launching shell')
    if 'c6' in ids and not no_llm:
        if not os.environ.get('API_KEY') or not os.environ.get('API_ENDPOINT'):
            raise ConfigurationError('Embedding filter requires explicit API_KEY and API_ENDPOINT for Titan Text Embeddings V2; no paid fallback is selected')
    if 's3' in ids:
        from .utils.grobid import GrobidService
        GrobidService().ensure_ready()
    if not no_llm and stages:
        from .engine.codex_session import run_environment
        run_environment(cache_path=os.environ.get('SCHOLAREVAL_CODEX_CACHE'))
    print('[preflight] PASS', flush=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='Durable ScholarEval evaluation workflow')
    parser.add_argument('--research_idea', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--llm_engine_name', required=True)
    parser.add_argument('--cutoff_date')
    parser.add_argument('--litellm_name', default='claude-sonnet-4-20250514')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Offline DAG/resume/preflight inspection; no inference or network')
    parser.add_argument('--stop-after-retrieval', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('SCHOLAREVAL_LLM_BACKEND', 'codex')
    root = Path(args.save_to).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from .engine.codex_session import close_run
    with run_lock(root / '.workflow.lock'):
        supervisor = Supervisor(root)
        try:
            content = Path(args.research_idea).read_text(encoding='utf-8')
            if not content.strip():
                raise ConfigurationError('Research idea is empty')
            os.environ.update(SCHOLAREVAL_RESUME='1' if args.resume else '0',
                              SCHOLAREVAL_NO_LLM='1' if args.no_llm or args.dry_run else '0',
                              SCHOLAREVAL_STATE_DB=str(supervisor.state.path.resolve()),
                              PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            os.environ.setdefault('SCHOLAREVAL_CODEX_CACHE', str(root / 'codex-responses.sqlite3'))
            os.environ.setdefault('SCHOLAREVAL_S2_CACHE_DB', str(root / 's2-responses.sqlite3'))
            os.environ.setdefault('SCHOLAREVAL_DIAGNOSTICS_DIR', str(root / 'diagnostics'))
            for group in ('soundness', 'contribution'):
                target = root / group / 'research_plan.txt'
                if not target.exists() or target.read_bytes() != content.encode('utf-8'):
                    atomic_bytes(target, content.encode('utf-8'))
            cutoff = None if not args.cutoff_date or args.cutoff_date.lower() == 'none' else args.cutoff_date
            if cutoff:
                from datetime import date
                date.fromisoformat(cutoff)
            litellm = None if os.environ.get('SCHOLAREVAL_LLM_BACKEND') == 'codex' else args.litellm_name
            graph = build_graph(root, args.llm_engine_name, cutoff, litellm)
            targets = ['s3'] if args.stop_after_retrieval else ['s6', 'c11']
            stages = required_stages(graph, targets, args.resume)
            supervisor.state.metric('workflow.resumes', int(args.resume))
            for stage in stages:
                supervisor.transition(stage.id, 'PENDING')
            def check():
                preflight(root, stages, args.no_llm, args.dry_run)
                return 0
            if not supervisor.recover('preflight', check, close_run):
                failure = supervisor.last_failure
                supervisor.transition('workflow', failure.state, reason=failure.reason)
                return 1
            if args.dry_run:
                atomic_json(root / 'dry-run.json', {'pending': [s.id for s in stages],
                            'targets': targets, 'network_calls': 0, 'llm_calls': 0})
                print('[dry-run] pending stages: ' + ', '.join(s.id for s in stages))
                supervisor.transition('workflow', 'PENDING', dry_run=True)
                return 0
            for stage in stages:
                if not supervisor.run_stage(stage, os.environ.copy()):
                    failure = supervisor.last_failure
                    supervisor.transition('workflow', failure.state, reason=failure.reason)
                    return 1
            supervisor.transition('workflow', 'COMPLETED')
            return 0
        except BaseException as error:
            outcome = classify(error)
            supervisor.transition('workflow', outcome.state, reason=outcome.reason)
            print(f'[workflow] {outcome.state}: {error}', file=sys.stderr)
            return outcome.exit_code
        finally:
            close_run()
            supervisor.state.summary(root / 'run-summary.json')
