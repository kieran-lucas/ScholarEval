"""Original snippet retrieval, with resumable infrastructure operations."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ..utils.checkpoints import StageCheckpoint, atomic_json, checked_main, digest, file_hash
from ..utils.grobid import parse_pdf_corpus
from ..utils.pdf_utils import FastPDFDownloader
from ..utils.retrieval_http import RetrievalError, RetrievalResponseError
from ..utils.retrieval_progress import RetrievalProgress
from ..utils.semantic_scholar import SemanticScholar
from ..utils.string_utils import GrobidXMLParser


def main():
    parser = argparse.ArgumentParser(description="Search for snippets related to method. Supports --resume.")
    parser.add_argument("--queries_file", required=True)
    parser.add_argument("--methods_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pdf_dir", default="./pdfs")
    parser.add_argument("--research_title", default="")
    parser.add_argument("--research_abstract", default="")
    parser.add_argument("--cutoff_date")
    parser.add_argument("--progress_file")
    args = parser.parse_args()
    checkpoint = StageCheckpoint('ScholarEval.soundness.snippet_search', sys.argv[1:])
    s2 = SemanticScholar(os.environ.get('S2_API_KEY'))
    progress = RetrievalProgress(args.progress_file or args.output_file + '_progress.json',
        checkpoint.fingerprint, s2.http, resume=os.environ.get('SCHOLAREVAL_RESUME') == '1')
    metrics = progress.state['metrics']
    for name in ('search_queries_completed', 'unique_papers_found', 'pdfs_discovered', 'pdfs_downloaded', 'pdfs_parsed'):
        metrics.setdefault(name, 0)
    metrics.setdefault('grobid_startup_seconds', 0.0)
    metrics.setdefault('grobid_action', 'not-needed')
    atomic_json(args.output_file + '_coverage.json', {'status': 'processing'})
    try:
        retrieve(args, s2, progress, metrics)
        progress.state['status'] = 'completed'
    except BaseException as error:
        progress.state['status'] = 'failed'
        coverage_path = Path(args.output_file + '_coverage.json')
        coverage = json.loads(coverage_path.read_text(encoding='utf-8'))
        coverage.update(status='retrieval_failed', failure_reason=str(error), failure_type=type(error).__name__)
        atomic_json(coverage_path, coverage)
        raise
    finally:
        progress.save()
        atomic_json(args.output_file + '_metrics.json', {'retrieval': metrics, 'status': progress.state['status']})
        print('Retrieval metrics: ' + json.dumps(metrics), flush=True)


def retrieve(args, s2, progress, metrics):
    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    queries = json.loads(Path(args.queries_file).read_text(encoding='utf-8'))['queries']
    methods = json.loads(Path(args.methods_file).read_text(encoding='utf-8'))['clean_methods']
    if not methods or set(queries) != set(methods) or not all(isinstance(queries[m], str) and queries[m].strip() for m in methods):
        raise ValueError('Methods/queries are empty, malformed, or incomplete')
    pub_date = args.cutoff_date or ''
    if not pub_date and args.research_title and args.research_abstract:
        matched = progress.run('evaluation-paper', lambda: s2.get_single_paper(args.research_title))
        if matched['data'] and s2.is_right_paper(matched['data'][0]['abstract'], args.research_abstract):
            pub_date = s2.get_safe_pub_date(matched['data'][0]['publicationDate'])
    keys = [digest([m, queries[m], pub_date]) for m in methods]
    for key in keys:
        progress.pending(key)
    progress.save()
    references = {}
    candidate_snippets = {}
    for method, key in zip(methods, keys):
        snippets = progress.run(key, lambda: s2.search_snippets(queries[method], year=['', pub_date], limit=8),
                                lambda data: s2.steal_cite_from_snip(data['data']))
        references[method] = sorted(s2.steal_cite_from_snip(snippets['data']))
        for snippet in snippets['data']:
            source_id = str(snippet['paper']['corpusId'])
            for cid in s2.steal_cite_from_snip([snippet]):
                candidate_snippets.setdefault(cid, []).append({
                    'method': method, 'source_corpus_id': source_id,
                    'relationship': 'source' if cid == source_id else 'cited_by_source',
                    'result': snippet,
                })
        metrics['search_queries_completed'] = sum(progress.state['items'][k]['status'] == 'succeeded' for k in keys)
        progress.save()
    # This file is written only after EVERY search succeeded; never load legacy references on existence alone.
    atomic_json(args.output_file + '_references.json', references)
    paper_ids = sorted({ref for refs in references.values() for ref in refs})
    metrics['unique_papers_found'] = len(paper_ids)
    metadata = []
    for i in range(0, len(paper_ids), 500):
        batch = ['CorpusId:' + pid for pid in paper_ids[i:i + 500]]
        def validate_metadata(data):
            if not isinstance(data, list) or len(data) != len(batch):
                raise RetrievalResponseError('Incomplete metadata batch response')
            return [p['paperId'] for p in data if p]
        metadata.extend(progress.run('metadata-' + digest(batch), lambda: s2.get_paper_bulk(batch), validate_metadata))
    downloader = FastPDFDownloader(pdf_dir=str(pdf_dir), email=os.environ.get('email'))
    pdf_data, candidates = [], {}
    for paper, cid in zip(metadata, paper_ids):
        info = (paper or {}).get('openAccessPdf') or {}
        url = info.get('url') or downloader.extract_url(info.get('disclaimer') or '')
        if url:
            pdf_data.append((url, cid))
        candidates[cid] = {
            'corpus_id': cid, 'paper_id': (paper or {}).get('paperId'),
            'metadata': paper or {}, 'snippets': candidate_snippets.get(cid, []),
            'download_status': 'pending' if url else 'full_text_unavailable',
            'parse_status': 'not_attempted', 'evidence_status': 'metadata_abstract_only',
            'attempted_urls': [], 'failure_reason': None if url else 'No PDF URL supplied by Semantic Scholar',
            'sections_used': [], 'sections_all': [],
        }
    metrics['pdfs_discovered'] = len(pdf_data)
    results = asyncio.run(downloader.download_pdfs_batch_async(pdf_data)) if pdf_data else []
    valid = []
    for (url, cid), result in zip(pdf_data, results):
        candidate = candidates[cid]
        record = downloader.download_record(cid, url)
        candidate.update(attempted_urls=record.get('attempted_urls', []), download=record)
        if result and not isinstance(result, Exception):
            valid.append(Path(result))
            candidate.update(download_status='downloaded', failure_reason=None)
        else:
            candidate.update(download_status='full_text_unavailable', failure_reason=record.get(
                'failure_reason', type(result).__name__ if isinstance(result, Exception) else 'No readable PDF after bounded attempts'))
    # A valid file remains usable even when the provider no longer supplies its URL.
    for cid, candidate in candidates.items():
        pdf = pdf_dir / (cid + '.pdf')
        if candidate['download_status'] != 'downloaded' and downloader.valid_pdf(pdf):
            valid.append(pdf)
            candidate.update(download_status='downloaded', failure_reason=None)
    metrics['pdfs_downloaded'] = len(valid)
    metrics['pdfs_download_failed'] = sum(candidates[cid]['download_status'] != 'downloaded' for _, cid in pdf_data)
    metrics['pdfs_unavailable'] = len(candidates) - len(valid)
    policy = os.environ.get('SCHOLAREVAL_FULL_TEXT_POLICY', 'available-evidence')
    if policy not in {'available-evidence', 'require-full-text'}:
        raise ValueError('SCHOLAREVAL_FULL_TEXT_POLICY must be available-evidence or require-full-text')
    print(f'{len(candidates)} papers discovered\n{len(valid)} full text available\n'
          f'{metrics["pdfs_unavailable"]} full text unavailable\ncontinuing with available evidence', flush=True)
    parse_path = Path(args.output_file + '_parse_progress.json')
    parse_version = digest([file_hash(Path(__file__).resolve().parents[1] / 'utils/grobid.py'),
                            file_hash('GROBID_config.json'),
                            os.environ.get('SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS'),
                            os.environ.get('SCHOLAREVAL_GROBID_WORKERS', '10')])
    try:
        parse_state = json.loads(parse_path.read_text(encoding='utf-8'))
        if parse_state.get('version') != parse_version:
            parse_state = {}
    except (OSError, ValueError):
        parse_state = {}
    parse_state.setdefault('version', parse_version)
    parse_records = parse_state.setdefault('papers', {})

    def save_evidence(status):
        for candidate in candidates.values():
            if candidate['evidence_status'] != 'full_text_available':
                candidate['paper'] = ('FULL TEXT NOT AVAILABLE FOR ANALYSIS. Metadata/abstract and search snippets only; '
                    'cited_by_source snippets describe a citing paper and are not this paper\'s own full text.\n' +
                    json.dumps({'metadata': candidate['metadata'], 'snippets': candidate['snippets']}, ensure_ascii=False))
            candidate['n_char'] = len(candidate['paper'])
            candidate['n_words'] = len(candidate['paper'].split())
        metrics['pdfs_parsed'] = sum(c['evidence_status'] == 'full_text_available' for c in candidates.values())
        metrics['pdfs_parse_failed'] = sum(c['parse_status'] == 'parse_failed' for c in candidates.values())
        metrics['metadata_abstract_only'] = len(candidates) - metrics['pdfs_parsed']
        atomic_json(args.output_file + '_papers.json', candidates)
        atomic_json(args.output_file + '_coverage.json', {
            'status': status, 'policy': policy, 'papers_discovered': len(candidates),
            'full_text_downloaded': len(valid), 'full_text_unavailable': metrics['pdfs_unavailable'],
            'parsed_successfully': metrics['pdfs_parsed'], 'parse_failed': metrics['pdfs_parse_failed'],
            'metadata_abstract_only': metrics['metadata_abstract_only'],
            'parsed_fraction': metrics['pdfs_parsed'] / len(candidates) if candidates else None,
            'scientific_conclusion': 'Coverage describes evidence availability, not absence of prior art or scientific validity.',
        })

    def on_parse(cid, record):
        parse_records[cid] = record
        candidates[cid]['parse_status'] = record['status']
        candidates[cid]['parse'] = record
        if record.get('failure_reason'):
            candidates[cid]['failure_reason'] = record['failure_reason']
        atomic_json(parse_path, parse_state)

    pending = []
    for pdf in valid:
        record = parse_records.get(pdf.stem, {})
        xml_path = pdf_dir / (pdf.stem + '.grobid.tei.xml')
        same_pdf = record.get('pdf_sha256') == file_hash(pdf)
        if same_pdf and (record.get('status') == 'parse_failed' or
                        (record.get('status') == 'parsed' and xml_path.is_file() and
                         record.get('xml_sha256') == file_hash(xml_path))):
            on_parse(pdf.stem, record)
        else:
            pending.append(pdf)
    save_evidence('processing')
    try:
        # Only these explicit candidate paths enter GROBID; old corpus files cannot leak in.
        parse_pdf_corpus(pending, pdf_dir, on_parse, metrics=metrics)
    finally:
        for pdf in valid:
            cid = pdf.stem
            if candidates[cid]['parse_status'] != 'parsed':
                continue
            xml_path = pdf_dir / (cid + '.grobid.tei.xml')
            try:
                ET.parse(xml_path)
                parser = GrobidXMLParser(xml_path.read_text(encoding='utf-8'))
                sections = parser.extract_sections()
            except (OSError, ET.ParseError, ValueError, AttributeError) as error:
                on_parse(cid, {**parse_records[cid], 'status': 'parse_failed',
                               'failure_reason': f'TEI extraction failed: {type(error).__name__}'})
                continue
            key_sections = ['method', 'task', 'baseline', 'methodology', 'approach', 'procedure', 'protocol', 'technique', 'design', 'framework', 'implementation', 'algorithm', 'process', 'workflow', 'strategy', 'preparation', 'synthesis', 'fabrication', 'construction','setup', 'apparatus', 'equipment', 'instrumentation', 'system', 'configuration', 'experimental', 'experiment', 'procedure', 'protocol', 'preparation', 'sample', 'specimen', 'material', 'device', 'platform', 'facility', 'laboratory', 'condition', 'parameter', 'result', 'results', 'finding', 'findings', 'outcome', 'observation', 'data', 'measurement', 'performance', 'evaluation', 'validation', 'testing', 'characterization', 'analysis', 'assessment', 'output', 'response', 'behavior', 'effect', 'impact']
            paper = ""
            for section in sections:
                if any([x in section['header'].lower() for x in key_sections]) and section['section_number']:
                    paper += f"# {section['header']}\n\t"
                    paper += '\n\t'.join(section['paragraphs']) + '\n'
            used = [section['header'] for section in sections if
                    any(x in section['header'].lower() for x in key_sections) and section['section_number']]
            if not paper.strip():
                paper = '\n'.join('\n'.join(section['paragraphs']) for section in sections)
                used = [section['header'] for section in sections]
            if not paper.strip():
                on_parse(cid, {**parse_records[cid], 'status': 'parse_failed',
                               'failure_reason': 'GROBID XML contains no extractable section text'})
                continue
            candidates[cid].update({
                'evidence_status': 'full_text_available',
                'n_char': len(paper),
                'n_words': len(paper.split()),
                'sections_used': used,
                'sections_all': [section['header'] for section in sections],
                'paper': paper,
            })
        save_evidence('processing')
    if candidates and policy == 'require-full-text' and not metrics['pdfs_parsed']:
        raise RetrievalError('Coverage policy requires at least one parsed full text; candidate evidence is saved')
    save_evidence('completed' if candidates else 'no_candidates')


if __name__ == '__main__':
    try:
        checked_main(main, 'ScholarEval.soundness.snippet_search')
    except (RetrievalError, RuntimeError, ValueError) as error:
        print(f'Retrieval stopped: {error}', file=sys.stderr)
        sys.exit(1)
