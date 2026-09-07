"""Explicitly attest known successful methods/queries from a pre-checkpoint run.

This cannot reconstruct old prompt provenance. The caller must confirm the saved
artifacts came from the given plan and current extraction/query configuration.
Never imports references or downstream artifacts, which may hide old failures.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScholarEval.utils.checkpoints import StageCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soundness-dir', required=True)
    parser.add_argument('--research-idea', required=True)
    parser.add_argument('--llm-engine-name', required=True)
    parser.add_argument('--attest-successful-run', action='store_true', required=True)
    args = parser.parse_args()
    folder = Path(args.soundness_dir)
    plan = folder / 'research_plan.txt'
    if plan.read_bytes() != Path(args.research_idea).read_bytes():
        raise ValueError('Saved research plan differs from supplied idea; refusing legacy import')
    costs = [json.loads(line) for line in (folder / 'soundness_costs.jsonl').read_text().splitlines() if line.strip()]
    stages = [('extract_methods', ['--input_file', str(plan)]),
              ('make_queries', ['--research_plan', str(plan), '--methods_file', str(folder / 'methods.json')])]
    checkpoints = []
    for stage, inputs in stages:
        if not any(c.get('step') == stage for c in costs):
            raise ValueError(f'Missing successful run cost record for {stage}')
        output = folder / ('methods.json' if stage == 'extract_methods' else 'queries.json')
        cp = StageCheckpoint('ScholarEval.soundness.' + stage, inputs + ['--output_file', str(output), '--llm_engine_name', args.llm_engine_name])
        if cp.sidecar.exists():
            raise ValueError(f'{cp.sidecar} already exists; legacy import cannot overwrite provenance')
        if not cp.validate():
            raise ValueError(f'{output} is incomplete or invalid')
        checkpoints.append(cp)
    for cp in checkpoints:
        cp.complete('explicit legacy attestation: successful saved run; plan matched; original prompt hash unavailable')
        print(f'Imported {cp.outputs[0].name}; original artifact/timestamp preserved')


if __name__ == '__main__':
    main()
