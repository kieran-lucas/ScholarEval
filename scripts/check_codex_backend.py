"""One tiny inference, optionally followed by one unchanged ScholarEval stage."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ScholarEval.engine.codex_app_server_engine import CodexAppServerEngine
from ScholarEval.engine.codex_session import CodexRun
from ScholarEval.engine.codex_transport import CodexError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-inference', action='store_true', help='Account/model/quota checks only; consumes no inference allowance.')
    parser.add_argument('--scholareval-smoke', action='store_true', help='Also run one short method-extraction stage; no retrieval or benchmark.')
    parser.add_argument('--output-dir', default='.scholareval-codex/smoke')
    args = parser.parse_args()
    if args.scholareval_smoke and not args.no_inference:
        output_dir = Path(args.output_dir).resolve()
        if any((output_dir / name).exists() for name in ('idea.txt', 'methods.json')):
            parser.error('Smoke output already exists; choose a fresh --output-dir before spending allowance.')
    print('## CODEX BACKEND PREFLIGHT', flush=True)
    def report(key, value):
        print(f'{key}: {value}', flush=True)
    try:
        with CodexAppServerEngine(status=report, cache_path=False) as engine:
            if args.no_inference:
                print('Inference: NOT RUN\nOVERALL: PREFLIGHT ONLY')
                return 0
            text, _, _ = engine.respond([{'role': 'user', 'content': 'Reply with exactly OK.'}])
            if text.strip() != 'OK':
                print('Inference: FAIL (unexpected final text)\nOVERALL: NOT READY')
                return 1
            print('Inference: PASS', flush=True)
            if args.scholareval_smoke:
                output_dir = Path(args.output_dir).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                idea = output_dir / 'idea.txt'
                # Exclusive creation protects existing user research inputs.
                with idea.open('x', encoding='utf-8') as stream:
                    stream.write('Compare a sample mean with a sample median on one synthetic dataset containing outliers. Measure absolute estimation error.\n')
                with CodexRun(engine=engine) as run:
                    env = os.environ.copy()
                    env.update(run.environment())
                    env['SCHOLAREVAL_LLM_BACKEND'] = 'codex'
                    for key in ('API_KEY', 'API_KEY_1', 'API_ENDPOINT', 'OPENAI_API_KEY'):
                        env.pop(key, None)
                    command = [sys.executable, '-m', 'ScholarEval.soundness.extract_methods',
                               '--input_file', str(idea), '--output_file', str(output_dir / 'methods.json'),
                               '--llm_engine_name', 'auto']
                    result = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[1])
                    if result.returncode:
                        raise CodexError('ScholarEval smoke stage failed.')
                methods = json.loads((output_dir / 'methods.json').read_text())['clean_methods']
                if not methods or not all(isinstance(method, str) for method in methods):
                    raise CodexError('ScholarEval existing parser returned no valid methods.')
                print(f'ScholarEval engine -> existing parser -> methods.json: PASS ({len(methods)} methods)')
            print('OVERALL: READY')
            return 0
    except CodexError as error:
        print(f'FAIL: {error}\nOVERALL: NOT READY', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
