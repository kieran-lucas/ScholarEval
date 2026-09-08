"""Small, content-addressed stage checkpoints; no model or network imports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import ast
import copy
import uuid

from .durable import atomic_json, atomic_bytes, digest, run_lock, telemetry
from .workflow_errors import ScientificValidationError, ConfigurationError, classify, diagnostic


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


OUTPUTS = {'--output_file', '--output', '--markdown_output', '--markdown_file', '--bibliography_file'}
IGNORED = {'--cost_log_file', '--progress_file', '--pdf_dir', '--litellm_name',
           '--max_workers', '--sleep_between_calls', '--grobid_config'}
RETRIEVAL = {'snippet_search', 'paper_extractor', 'paper_augmentation'}
NON_LLM = RETRIEVAL | {'paper_sampler', 'prepare_final_contribution_context'}


def semantic_config(module, opts):
    """Explicit algorithm version plus long scientific prompt templates.

    Bump algorithm_version for changes to scoring/selection semantics. Do not hash
    transport code: retries, logging and generation storage are operational.
    """
    root = Path(__file__).resolve().parents[2]
    source = root / (module.replace('.', '/') + '.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    prompts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
               and isinstance(n.value, str) and len(n.value) > 180
               and ('{rp}' in n.value or 'JSON' in n.value or 'research' in n.value.lower())]
    stage = module.rsplit('.', 1)[-1]
    config = {'algorithm_version': 2 if stage == 'paper_augmentation' else 1, 'prompt_templates': digest(prompts)}
    if stage not in NON_LLM and stage != 'embedding_filter':
        backend = os.environ.get('SCHOLAREVAL_LLM_BACKEND', 'litellm').lower()
        config['model'] = {'backend': backend,
            'identity': os.environ.get('SCHOLAREVAL_CODEX_MODEL', 'auto') if backend == 'codex'
                        else opts.get('--llm_engine', opts.get('--llm_engine_name')),
            'reasoning': os.environ.get('SCHOLAREVAL_CODEX_REASONING', 'high') if backend == 'codex' else None}
    if stage == 'embedding_filter':
        config['embedding'] = {'model': 'Titan Text Embeddings V2',
                               'endpoint': os.environ.get('API_ENDPOINT')}
    if stage == 'snippet_search':
        config['full_text_policy'] = os.environ.get('SCHOLAREVAL_FULL_TEXT_POLICY', 'available-evidence')
    if stage in {'tldr_soundness', 'contribution_review_synthesis'}:
        helper = ast.parse((root/'ScholarEval/utils/citation_check.py').read_text(encoding='utf-8'))
        config['citation_prompts'] = digest([n.value for n in ast.walk(helper) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and len(n.value) > 180])
    return config


def options(argv):
    return {arg: argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg.startswith('--')}


class StageCheckpoint:
    def __init__(self, module, argv):
        self.module = module
        self.stage = module.rsplit('.', 1)[-1]
        self.opts = options(argv)
        self.output_flags = OUTPUTS - {'--bibliography_file'} if self.stage == 'tldr_soundness' else OUTPUTS
        primary = self.opts.get('--output_file', self.opts.get('--output'))
        self.outputs = [Path(v) for k, v in self.opts.items() if k in self.output_flags]
        if self.stage == 'snippet_search':
            self.outputs = [Path(primary + suffix) for suffix in ('_references.json', '_papers.json', '_coverage.json')]
        self.sidecar = Path(primary + '.stage.json')
        self.primary = Path(primary)
        self.generations = Path(primary + '.generations')
        backend = os.environ.get('SCHOLAREVAL_LLM_BACKEND', 'litellm').lower()
        inputs = {'checkpoint_version': 2, 'semantics': semantic_config(module, self.opts)}
        for key, value in self.opts.items():
            if key in self.output_flags | IGNORED:
                continue
            if backend == 'codex' and key in {'--llm_engine', '--llm_engine_name'}:
                continue  # Existing Codex backend selects its model from config, not this legacy argument.
            inputs[key] = file_hash(value) if Path(value).is_file() else value
        self.inputs = inputs
        self.fingerprint = digest(inputs)
        self.boundary = os.environ.get('SCHOLAREVAL_STAGE_BOUNDARY')

    def validate(self):
        try:
            values = []
            for path in self.outputs:
                content = path.read_text(encoding='utf-8')
                if not content.strip():
                    return False
                if path.suffix == '.jsonl':
                    value = [json.loads(line) for line in content.splitlines() if line.strip()]
                elif path.suffix == '.json':
                    value = json.loads(content)
                else:
                    value = content
                if has_error(value):
                    return False
                values.append(value)
            data = values[0]
            if self.stage == 'extract_methods':
                return isinstance(data, dict) and strings(data.get('clean_methods'))
            if self.stage == 'tldr_soundness':
                return isinstance(data, str) and all(label in data for label in
                    ('## Strengths', '## Weaknesses', '## Top 3 Suggestions'))
            if self.stage == 'make_queries':
                methods = json.loads(Path(self.opts['--methods_file']).read_text(encoding='utf-8'))['clean_methods']
                return (isinstance(data, dict) and isinstance(data.get('queries'), dict)
                        and set(data['queries']) == set(methods) and bool(methods)
                        and all(isinstance(v, str) and v.strip() for v in data['queries'].values()))
            if self.stage == 'snippet_search':
                methods = json.loads(Path(self.opts['--methods_file']).read_text(encoding='utf-8'))['clean_methods']
                return (isinstance(data, dict) and set(data) == set(methods)
                        and all(isinstance(v, list) and all(isinstance(i, str) and i for i in v) for v in data.values())
                        and values[2].get('status') in {'completed', 'no_candidates'}
                        and set(values[1]) == {cid for refs in data.values() for cid in refs}
                        and isinstance(values[1], dict) and all(isinstance(p, dict) and isinstance(p.get('paper'), str)
                            and p.get('evidence_status') in {'full_text_available', 'metadata_abstract_only'}
                            and {'n_words', 'n_char', 'sections_used', 'sections_all'} <= p.keys() for p in values[1].values()))
            if self.stage == 'extract_dimensions_and_contributions':
                return isinstance(data, list) and bool(data) and all(isinstance(p, dict) and p.get('dimension') and p.get('contributions') for p in data)
            if self.stage == 'queries_generator':
                source = [json.loads(line) for line in Path(self.opts['--contrib_file']).read_text(encoding='utf-8').splitlines() if line.strip()]
                expected = {c for entry in source for c in entry['contributions']}
                return isinstance(data, dict) and bool(data.get('queries')) and set(data['queries']) == expected and all(strings(v) for v in data['queries'].values())
            if self.stage in {'methods_and_results_synthesis', 'meta_review'}:
                if not isinstance(data, dict) or not isinstance(data.get('analysis'), dict):
                    return False
                if self.stage == 'methods_and_results_synthesis':
                    refs = json.loads(Path(self.opts['--methods_and_ref_file']).read_text(encoding='utf-8'))
                    return set(data['analysis']) == set(refs) and all(isinstance(data['analysis'].get(m, []), list) and
                               sorted(str(a['corpus_id']) for a in data['analysis'].get(m, [])) == sorted(ids)
                               for m, ids in refs.items())
                previous = json.loads(Path(self.opts['--mr_analysis_file']).read_text(encoding='utf-8'))['analysis']
                return set(data['analysis']) == set(previous) and all(valid_meta_review(v) for v in data['analysis'].values())
            if self.stage == 'relevance_assessor':
                data = data.get('papers') if isinstance(data, dict) else None
                source = json.loads(Path(self.opts['--papers_file']).read_text(encoding='utf-8'))
                source = source.get('papers', []) if isinstance(source, dict) else source
                return (isinstance(data, list) and {p['paperId'] for p in data} == {p['paperId'] for p in source if p.get('abstract')}
                        and len(data) == len({p['paperId'] for p in data})
                        and all(valid_relevance(p) for p in data))
            if self.stage == 'pairwise_comparator':
                source = json.loads(Path(self.opts['--papers_metadata']).read_text(encoding='utf-8'))
                expected = [p for p in source if p.get('abstract')]
                dimensions = [json.loads(line)['dimension'] for line in Path(self.opts['--dimensions_file']).read_text(encoding='utf-8').splitlines() if line.strip()]
                return isinstance(data, dict) and isinstance(data.get('comparisons'), list) and all(valid_comparison(p, dimensions) for p in data['comparisons']) and len(data['comparisons']) == len(expected)
            if self.stage == 'prepare_final_contribution_context':
                source = json.loads(Path(self.opts['--input_file']).read_text(encoding='utf-8'))['comparisons']
                return isinstance(data, list) and len(data) == len(source) and all(p.get('paperReference') and p.get('comparison') for p in data)
            if self.stage in {'paper_extractor', 'paper_augmentation', 'embedding_filter', 'paper_sampler'}:
                return isinstance(data, list) and all(isinstance(p, dict) and p.get('paperId') for p in data)
            return isinstance(data, str) and bool(data.strip())
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            return False

    def valid(self):
        try:
            meta = json.loads(self.sidecar.read_text(encoding='utf-8'))
            identity = meta['fingerprint'] == self.fingerprint
            if self.boundary and meta.get('boundary') == self.boundary:
                identity = True
            if meta['status'] != 'completed' or not identity:
                return False
            if meta.get('generation'):
                generation = self.generations / meta['generation']
                if generation.resolve().parent != self.generations.resolve():
                    return False
                files = [generation / p.name for p in self.outputs]
                if meta['output_hashes'] != [file_hash(p) for p in files]:
                    return False
                # The atomic pointer is the commit. Compatibility files can be
                # repaired after a kill between individual materializations.
                for source, target, expected in zip(files, self.outputs, meta['output_hashes']):
                    if not target.exists() or file_hash(target) != expected:
                        atomic_bytes(target, source.read_bytes())
                return True  # Validated at commit; upstream history is not required.
            return self.validate() and meta['output_hashes'] == [file_hash(p) for p in self.outputs]
        except (OSError, ValueError, KeyError):
            return False

    def complete(self, provenance='stage completed successfully'):
        if not self.validate():
            raise ScientificValidationError(f'{self.stage}: invalid or partial output; checkpoint not committed')
        generation = self.generations / ('gen-' + uuid.uuid4().hex)
        generation.mkdir(parents=True)
        for path in self.outputs:
            atomic_bytes(generation / path.name, path.read_bytes())
        self.promote(generation, provenance)

    def promote(self, generation, provenance='stage completed successfully'):
        candidate = copy.copy(self)
        candidate.outputs = [generation / p.name for p in self.outputs]
        if not candidate.validate():
            raise ScientificValidationError(f'{self.stage}: candidate generation failed validation')
        meta = {'version': 2, 'status': 'completed', 'stage': self.module,
                'fingerprint': self.fingerprint, 'semantic_inputs': self.inputs,
                'boundary': self.boundary, 'generation': generation.name,
                'output_hashes': [file_hash(p) for p in candidate.outputs],
                'completed_at': time.time(), 'provenance': provenance}
        # Fsync stage-produced files before making their generation visible.
        for path in candidate.outputs:
            with path.open('r+b') as stream:
                os.fsync(stream.fileno())
        atomic_json(generation / 'commit.json', meta)
        atomic_json(self.sidecar, meta)
        for source, target in zip(candidate.outputs, self.outputs):
            atomic_bytes(target, source.read_bytes())
        state = telemetry()
        if state:
            state.metric('workflow.generation_commits')


def valid_relevance(p):
    return (isinstance(p, dict) and isinstance(p.get('relevance_score'), int)
            and not isinstance(p.get('relevance_score'), bool) and 0 <= p['relevance_score'] <= 5
            and isinstance(p.get('relevance_rationale'), str) and bool(p['relevance_rationale'].strip())
            and not p['relevance_rationale'].startswith(('Processing error:', 'Failed to parse assessment:')))


def valid_comparison(p, dimensions):
    comparison = p.get('comparison') if isinstance(p, dict) else None
    if not isinstance(comparison, dict) or not isinstance(comparison.get('overall_comparison'), str) or not comparison['overall_comparison'].strip():
        return False
    values = comparison.get('dimension_comparisons')
    return (isinstance(values, dict) and set(values) == set(dimensions)
            and all(isinstance(v, dict) and isinstance(v.get('comparison'), str) and bool(v['comparison'].strip())
                    and type(v.get('score')) is int and v['score'] in {-1, 0, 1} for v in values.values()))


def valid_meta_review(value):
    if not isinstance(value, dict) or not all(isinstance(value.get(k), str) and value[k].strip()
        for k in ('support', 'contradictions', 'suggested_action')):
        return False
    if value.get('soundness_score') is None:
        return value.get('n_related_work') == 0
    try:
        score = float(value['soundness_score'])
        return not isinstance(value['soundness_score'], bool) and score.is_integer() and 0 <= score <= 10
    except (TypeError, ValueError):
        return False


def strings(value):
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) and x.strip() for x in value)


def has_error(value):
    if isinstance(value, dict):
        return bool(value.get('error')) or any(has_error(v) for v in value.values())
    return isinstance(value, list) and any(has_error(v) for v in value)


def checked_main(main, module):
    resume = '--resume' in sys.argv or os.environ.get('SCHOLAREVAL_RESUME') == '1'
    if resume:
        os.environ['SCHOLAREVAL_RESUME'] = '1'
    sys.argv[:] = [a for a in sys.argv if a != '--resume']
    if '--help' in sys.argv or '-h' in sys.argv:
        return main()
    checkpoint = StageCheckpoint(module, sys.argv[1:])
    if resume and checkpoint.valid():
        label = {'extract_methods': 'method extraction', 'make_queries': 'query generation'}.get(checkpoint.stage, checkpoint.stage)
        print(f'[RESUME] {checkpoint.outputs[0].name} valid - {label} skipped', flush=True)
        return
    if resume:
        print(f'[RESUME] {checkpoint.stage}: missing, stale, or invalid checkpoint; recomputing', flush=True)
    with run_lock(str(checkpoint.primary) + '.lock'):
        generation = checkpoint.generations / ('gen-' + uuid.uuid4().hex)
        generation.mkdir(parents=True)
        argv = sys.argv[:]
        prior_env = {k: os.environ.get(k) for k in ('SCHOLAREVAL_STAGE_PRIMARY', 'SCHOLAREVAL_STAGE_FINGERPRINT')}
        os.environ['SCHOLAREVAL_STAGE_PRIMARY'] = str(checkpoint.primary)
        os.environ['SCHOLAREVAL_STAGE_FINGERPRINT'] = checkpoint.fingerprint
        # Re-route every explicit output (and outputs derived from its basename)
        # before argparse runs. Progress/caches keep their original durable paths.
        for i, value in enumerate(sys.argv[:-1]):
            if value in checkpoint.output_flags:
                sys.argv[i + 1] = str(generation / Path(sys.argv[i + 1]).name)
        attempt_path = Path(str(checkpoint.primary) + '.attempt.json')
        attempt = {'status': 'RUNNING', 'active_generation': generation.name,
                   'fingerprint': checkpoint.fingerprint, 'started_at': time.time()}
        atomic_json(attempt_path, attempt)
        try:
            if os.environ.get('SCHOLAREVAL_NO_LLM') == '1' and checkpoint.stage not in NON_LLM:
                raise ConfigurationError(f'No-LLM guard: {checkpoint.stage} requires a valid checkpoint')
            result = main()
            if result not in (None, 0):
                raise ScientificValidationError(f'{checkpoint.stage} failed: {result}')
            checkpoint.promote(generation)
            attempt['status'] = 'COMPLETED'
            return result
        except BaseException as error:
            outcome = classify(error)
            atomic_json(generation / 'failure.json', diagnostic(error))
            attempt.update(status=outcome.state, failure_type=outcome.reason, exit_code=outcome.exit_code)
            state = telemetry()
            if state:
                state.stage(str(checkpoint.primary), outcome.state, reason=outcome.reason)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if os.environ.get('SCHOLAREVAL_SUPERVISED') == '1':
                print(f'[stage] {outcome.state}: {outcome.reason}', file=sys.stderr, flush=True)
                raise SystemExit(outcome.exit_code) from None
            raise
        finally:
            atomic_json(attempt_path, attempt)
            sys.argv[:] = argv
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
