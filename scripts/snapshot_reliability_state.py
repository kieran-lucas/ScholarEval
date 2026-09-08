"""Non-destructive, hash-verified snapshot before reliability migration (no network)."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScholarEval import ScholarEval as runner
from ScholarEval.utils.checkpoints import StageCheckpoint, atomic_json, file_hash


def snapshot(run: Path) -> Path:
    destination = Path('.reliability-snapshots') / time.strftime('%Y%m%d-%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for root in [Path('ScholarEval'), Path('tests'), Path('scripts'), run]:
        for source in root.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            if source.suffix in {'.pdf', '.xml', '.log'}:
                entries.append({'path': str(source), 'sha256': file_hash(source), 'copied': False})
                continue
            target = destination / source
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == '.sqlite3':
                with sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True) as old, sqlite3.connect(target) as new:
                    old.backup(new)
                entries.append({'path': str(source), 'snapshot_sha256': file_hash(target), 'sqlite_backup': True})
            else:
                shutil.copy2(source, target)
                assert file_hash(source) == file_hash(target)
                entries.append({'path': str(source), 'sha256': file_hash(source), 'copied': True})
    from ScholarEval.workflow import build_graph
    env = {'SCHOLAREVAL_LLM_BACKEND': 'codex', 'SCHOLAREVAL_CODEX_MODEL': 'auto',
           'SCHOLAREVAL_CODEX_REASONING': 'high'}
    checkpoints = []
    with patch.dict(os.environ, env):
        for stage in build_graph(run).values():
            cp = stage.checkpoint()
            checkpoints.append({'module': stage.module, 'argv': stage.argv, 'fingerprint': cp.fingerprint,
                                'valid': cp.valid(), 'structurally_valid': cp.validate(),
                                'output_hashes': [file_hash(p) if p.exists() else None for p in cp.outputs]})
    atomic_json(destination / 'snapshot.json', {'files': entries, 'checkpoints': checkpoints,
                'plan_sha256': file_hash('test_idea.txt'), 'semantic_environment': env})
    print(destination)
    for cp in checkpoints:
        print(cp['module'], 'valid=' + str(cp['valid']), 'structure=' + str(cp['structurally_valid']))
    return destination


if __name__ == '__main__':
    snapshot(Path(sys.argv[1] if len(sys.argv) > 1 else 'demo_data/astra-test-1'))
