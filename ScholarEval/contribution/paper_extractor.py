"""Contribution paper search with durable per-query progress."""
import argparse
import json
import os
from pathlib import Path
import sys

from ..utils.semantic_scholar import SemanticScholar
from ..utils.checkpoints import StageCheckpoint, atomic_json, checked_main, digest
from ..utils.retrieval_progress import RetrievalProgress


def main():
    parser = argparse.ArgumentParser(description='Extract top papers. Supports --resume.')
    parser.add_argument('--queries_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--cutoff_date')
    parser.add_argument('--progress_file')
    args = parser.parse_args()
    checkpoint = StageCheckpoint('ScholarEval.contribution.paper_extractor', sys.argv[1:])
    s2 = SemanticScholar(os.environ.get('S2_API_KEY'))
    progress = RetrievalProgress(args.progress_file or args.output_file + '.progress.json',
        checkpoint.fingerprint, s2.http, resume=os.environ.get('SCHOLAREVAL_RESUME') == '1')
    query_map = json.loads(Path(args.queries_file).read_text(encoding='utf-8'))['queries']
    tasks = [(digest([contrib, query, args.cutoff_date]), query) for contrib, queries in query_map.items() for query in queries]
    for key, _ in tasks:
        progress.pending(key)
    progress.save()
    papers = {}
    try:
        for key, query in tasks:
            result = progress.run(key, lambda: s2.search_top_papers(query, max_date=args.cutoff_date),
                                  lambda data: [p['paperId'] for p in data])
            for paper in result:
                papers.setdefault(paper['paperId'], s2.extract_metadata(paper))
            progress.state['metrics'].update(search_queries_completed=sum(
                progress.state['items'][k]['status'] == 'succeeded' for k, _ in tasks), unique_papers_found=len(papers))
            progress.save()
        atomic_json(args.output_file, list(papers.values()))
        progress.state['status'] = 'completed'
    except BaseException:
        progress.state['status'] = 'failed'
        raise
    finally:
        progress.save()
        atomic_json(args.output_file + '.metrics.json', {'retrieval': progress.state['metrics'], 'status': progress.state['status']})


if __name__ == '__main__':
    checked_main(main, 'ScholarEval.contribution.paper_extractor')
