"""Small, content-addressed stage checkpoints; no model or network imports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


OUTPUTS = {'--output_file', '--output', '--markdown_output', '--markdown_file', '--bibliography_file'}
IGNORED = {'--cost_log_file', '--progress_file', '--pdf_dir', '--litellm_name'}
RETRIEVAL = {'snippet_search', 'paper_extractor', 'paper_augmentation'}
NON_LLM = RETRIEVAL | {'paper_sampler', 'prepare_final_contribution_context'}


def options(argv):
    return {arg: argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg.startswith('--')}


class StageCheckpoint:
    def __init__(self, module, argv):
        self.module = module
        self.stage = module.rsplit('.', 1)[-1]
        self.opts = options(argv)
        primary = self.opts.get('--output_file', self.opts.get('--output'))
        self.outputs = [Path(v) for k, v in self.opts.items() if k in OUTPUTS]
        if self.stage == 'snippet_search':
            self.outputs = [Path(primary + suffix) for suffix in ('_references.json', '_papers.json', '_coverage.json')]
        self.sidecar = Path(primary + '.stage.json')
        backend = os.environ.get('SCHOLAREVAL_LLM_BACKEND', 'litellm').lower()
        inputs = {'checkpoint_version': 1}
        for key, value in self.opts.items():
            if key in OUTPUTS | IGNORED:
                continue
            if backend == 'codex' and key in {'--llm_engine', '--llm_engine_name'}:
                continue  # Existing Codex backend selects its model from config, not this legacy argument.
            inputs[key] = file_hash(value) if Path(value).is_file() else value
        root = Path(__file__).resolve().parents[2]
        sources = [root / (module.replace('.', '/') + '.py')]
        # Infrastructure changes must not invalidate successful upstream Astra calls.
        helpers = ['string_utils.py']
        if self.stage in RETRIEVAL or self.stage == 'meta_review':
            helpers += ['semantic_scholar.py', 'retrieval_http.py', 'retrieval_progress.py']
        if self.stage in {'snippet_search', 'paper_augmentation'}:
            helpers += ['grobid.py', 'pdf_utils.py']
        sources += [root / 'ScholarEval/utils' / name for name in helpers]
        inputs['source_versions'] = {p.name: file_hash(p) for p in sources}
        if self.stage not in NON_LLM:
            inputs['llm_config'] = {'backend': backend}
            if backend == 'codex':
                inputs['llm_config'].update(model=os.environ.get('SCHOLAREVAL_CODEX_MODEL', 'auto'),
                                           reasoning=os.environ.get('SCHOLAREVAL_CODEX_REASONING', 'high'))
        if self.stage in RETRIEVAL:
            inputs['grobid_config'] = file_hash(root / 'GROBID_config.json')
        if self.stage == 'snippet_search':
            inputs['full_text_policy'] = os.environ.get('SCHOLAREVAL_FULL_TEXT_POLICY', 'available-evidence')
            inputs['parse_timeout'] = os.environ.get('SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS')
            inputs['parse_workers'] = os.environ.get('SCHOLAREVAL_GROBID_WORKERS', '10')
        self.fingerprint = digest(inputs)

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
                    return all(isinstance(data['analysis'].get(m, []), list) and
                               sorted(str(a['corpus_id']) for a in data['analysis'].get(m, [])) == sorted(ids)
                               for m, ids in refs.items())
                previous = json.loads(Path(self.opts['--mr_analysis_file']).read_text(encoding='utf-8'))['analysis']
                return set(data['analysis']) == set(previous)
            if self.stage == 'relevance_assessor':
                data = data.get('papers') if isinstance(data, dict) else None
                source = json.loads(Path(self.opts['--papers_file']).read_text(encoding='utf-8'))
                source = source.get('papers', []) if isinstance(source, dict) else source
                return (isinstance(data, list) and {p['paperId'] for p in data} == {p['paperId'] for p in source if p.get('abstract')}
                        and all('relevance_score' in p and not str(p.get('relevance_rationale', '')).startswith(('Processing error:', 'Failed to parse assessment:')) for p in data))
            if self.stage == 'pairwise_comparator':
                source = json.loads(Path(self.opts['--papers_metadata']).read_text(encoding='utf-8'))
                expected = [p for p in source if p.get('abstract')]
                return isinstance(data, dict) and isinstance(data.get('comparisons'), list) and all(p.get('comparison') for p in data['comparisons']) and len(data['comparisons']) == len(expected)
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
            return (meta['status'] == 'completed' and meta['fingerprint'] == self.fingerprint
                    and self.validate() and meta['output_hashes'] == [file_hash(p) for p in self.outputs])
        except (OSError, ValueError, KeyError):
            return False

    def complete(self, provenance='stage completed successfully'):
        if not self.validate():
            raise ValueError(f'{self.stage}: invalid or partial output; checkpoint not committed')
        atomic_json(self.sidecar, {'status': 'completed', 'stage': self.module,
                    'fingerprint': self.fingerprint, 'output_hashes': [file_hash(p) for p in self.outputs],
                    'completed_at': time.time(), 'provenance': provenance})


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
    if os.environ.get('SCHOLAREVAL_NO_LLM') == '1' and checkpoint.stage not in NON_LLM:
        raise RuntimeError(f'No-LLM guard: {checkpoint.stage} requires a valid checkpoint')
    atomic_json(checkpoint.sidecar, {'status': 'running', 'fingerprint': checkpoint.fingerprint})
    try:
        result = main()
        if result not in (None, 0):
            raise RuntimeError(f'{checkpoint.stage} failed: {result}')
        checkpoint.complete()
    except BaseException as error:
        atomic_json(checkpoint.sidecar, {'status': 'failed', 'fingerprint': checkpoint.fingerprint,
                    'failure_type': type(error).__name__, 'failed_at': time.time()})
        raise
    return result
