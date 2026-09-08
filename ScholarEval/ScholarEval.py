"""Canonical CLI entry point for the durable ScholarEval DAG."""
from ScholarEval.workflow import main
from ScholarEval.utils.workflow_errors import classify, diagnostic
import sys

if __name__ == '__main__':
    try:
        result = main()
    except (OSError, RuntimeError, ValueError) as error:
        outcome = classify(error)
        print(f'[workflow] {outcome.state}: {diagnostic(error)["message"]}', file=sys.stderr)
        result = outcome.exit_code
    raise SystemExit(result)
