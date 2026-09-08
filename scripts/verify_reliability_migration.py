"""Verify scientific output, PDF/XML and Codex cache preservation against a snapshot."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScholarEval.utils.checkpoints import file_hash
from ScholarEval.utils.durable import atomic_json, digest


def verify(snapshot, run):
    snapshot, run = Path(snapshot), Path(run)
    data = json.loads((snapshot/'snapshot.json').read_text(encoding='utf-8'))
    checked, failures = [], []
    for entry in data['files']:
        path = Path(entry['path'])
        if not path.is_relative_to(run):
            continue
        if path.suffix not in {'.pdf', '.xml', '.txt', '.md', '.json', '.jsonl'}:
            continue
        if path.name.endswith('.stage.json') or 'progress' in path.name:
            continue
        if entry.get('sha256'):
            checked.append(str(path))
            if not path.exists() or file_hash(path) != entry['sha256']:
                failures.append(str(path))
    def rows(path):
        with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as db:
            return dict(db.execute('SELECT key,text FROM responses'))
    old = rows(snapshot/run/'codex-responses.sqlite3')
    current = rows(run/'codex-responses.sqlite3')
    cache_preserved = all(current.get(key) == text for key, text in old.items())
    relevance = json.loads((run/'contribution'/'filtered_contribution_papers.json').read_text(encoding='utf-8'))['papers']
    report = {'files_verified': len(checked), 'mismatches': failures,
              'codex_entries_before': len(old), 'codex_entries_now': len(current),
              'codex_original_entries_preserved': cache_preserved,
              'codex_original_content_hash': digest(old),
              'assessed_papers': len(relevance),
              'augmentation_seeds': sum(p['relevance_score'] >= 3 for p in relevance)}
    atomic_json(run/'reliability-verification.json', report)
    print(json.dumps(report, indent=2))
    if failures or not cache_preserved:
        raise ValueError('Preservation verification failed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--run', default='demo_data/astra-test-1')
    args = parser.parse_args()
    verify(args.snapshot, args.run)
