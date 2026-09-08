"""Import ONLY checkpoints verified by a pre-change, hash-verified snapshot.

No inference, downloads, deletion, or re-attestation of partial scientific output.
Run from the repository root. The original sidecars remain in the snapshot and in
an adjacent .legacy-* file. New generations contain byte-identical output files.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScholarEval.utils.checkpoints import StageCheckpoint, file_hash
from ScholarEval.utils.durable import atomic_json, run_lock
from ScholarEval.workflow import build_graph


def prompts(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and len(n.value) > 180
            and ('{rp}' in n.value or 'JSON' in n.value or 'research' in n.value.lower())]


def migrate(snapshot, run):
    snapshot, run = Path(snapshot), Path(run)
    record = json.loads((snapshot/'snapshot.json').read_text(encoding='utf-8'))
    verified, preserved, retrieval_items = [], [], 0
    with run_lock(run/'.workflow.lock'), patch.dict(os.environ, record['semantic_environment']):
        if record['plan_sha256'] != file_hash(run/'soundness'/'research_plan.txt'):
            raise ValueError('Snapshot plan does not match saved plan')
        graph = build_graph(run)
        for old in record['checkpoints']:
            cp = StageCheckpoint(old['module'], old['argv'])
            source = Path(old['module'].replace('.', '/') + '.py')
            if prompts(source) != prompts(snapshot/source):
                raise ValueError('Scientific prompt changed; legacy migration refused: ' + old['module'])
            if cp.stage == 'paper_augmentation':
                progress = Path(str(cp.primary) + '.progress.json')
                if progress.exists():
                    data = json.loads(progress.read_text(encoding='utf-8'))
                    if data.get('fingerprint') == old['fingerprint']:
                        if file_hash(progress) != file_hash(snapshot/progress):
                            raise ValueError('Retrieval progress changed after snapshot')
                        from ScholarEval.utils.durable import digest
                        if any(v.get('status') == 'succeeded' and v.get('result_hash') != digest(v.get('result'))
                               for v in data['items'].values()):
                            raise ValueError('Retrieval item hash mismatch')
                        backup = Path(str(progress) + '.legacy-' + old['fingerprint'][:16])
                        if not backup.exists():
                            shutil.copy2(progress, backup)
                        data['fingerprint'] = cp.fingerprint
                        atomic_json(progress, data)
                        retrieval_items += sum(v.get('status') == 'succeeded' for v in data['items'].values())
                    elif data.get('fingerprint') == cp.fingerprint:
                        retrieval_items += sum(v.get('status') == 'succeeded' for v in data['items'].values())
            if not old['valid']:
                preserved.append(str(cp.primary))
                continue
            stage = next(s for s in graph.values() if s.checkpoint().primary.resolve() == cp.primary.resolve())
            cp.boundary = stage.boundary
            if cp.valid():
                verified.append(str(cp.primary))
                continue
            if file_hash(cp.sidecar) != file_hash(snapshot/cp.sidecar):
                raise ValueError('Sidecar changed after snapshot: ' + str(cp.sidecar))
            if [file_hash(p) for p in cp.outputs] != old['output_hashes'] or not cp.validate():
                raise ValueError('Output changed/invalid since snapshot: ' + str(cp.primary))
            backup = Path(str(cp.sidecar) + '.legacy-' + old['fingerprint'][:16])
            if not backup.exists():
                shutil.copy2(cp.sidecar, backup)
            cp.complete('Hash-verified pre-hardening snapshot: ' + str(snapshot))
            assert [file_hash(p) for p in cp.outputs] == old['output_hashes']
            assert stage.checkpoint().valid()
            verified.append(str(cp.primary))
        report = {'snapshot': str(snapshot), 'migrated_or_verified': verified,
                  'unverified_preserved_without_attestation': preserved,
                  'retrieval_items_migrated': retrieval_items,
                  'scientific_outputs_changed': 0, 'cache_entries_deleted': 0}
        atomic_json(run/'reliability-migration.json', report)
        print(json.dumps(report, indent=2))
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--run', default='demo_data/astra-test-1')
    args = parser.parse_args()
    migrate(args.snapshot, args.run)
