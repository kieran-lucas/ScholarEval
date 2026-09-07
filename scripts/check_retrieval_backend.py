"""Read-only by default. Never imports an LLM engine or uses Astra."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ScholarEval.utils.grobid import GrobidService, GrobidStartupError
from ScholarEval.utils.retrieval_http import RetrievalHTTP, RetrievalError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-network', action='store_true', help='No external requests/pulls; local Docker/health checks allowed')
    parser.add_argument('--check-s2', action='store_true', help='Make exactly one cheap S2 request, without retries')
    parser.add_argument('--start-grobid', action='store_true', help='Start/reuse GROBID and poll readiness')
    parser.add_argument('--pull-image', action='store_true', help='Pull the expected image if absent')
    args = parser.parse_args()
    if args.no_network and (args.check_s2 or args.pull_image):
        parser.error('--no-network cannot be combined with --check-s2/--pull-image')
    ready = True
    print('## RETRIEVAL PREFLIGHT')
    for name in ('grobid_client', 'requests', 'aiohttp', 'numpy', 'dateutil', 'bs4', 'PyPDF2'):
        found = importlib.util.find_spec(name) is not None
        print(f'Python {name}: ' + ('PASS' if found else 'MISSING'))
        ready &= found
    print('Docker CLI: ' + ('PASS' if shutil.which('docker') else 'MISSING'))
    service = GrobidService()
    try:
        service.docker('info', '--format', '{{.ServerVersion}}')
        print('Docker daemon: PASS')
        try:
            service.docker('image', 'inspect', service.image)
        except GrobidStartupError:
            if args.pull_image:
                service.docker('pull', service.image, timeout=600)
            else:
                raise GrobidStartupError('GROBID image missing; use --pull-image when network is allowed')
        print('GROBID image: PASS (' + service.image + ')')
        if args.start_grobid:
            # Image inspected above, so --no-network cannot accidentally pull.
            service.ensure_ready()
        containers = service.containers()
        compatible = [c for c in containers if service.compatible(c)]
        for container in compatible:
            print(f'GROBID container: {container["State"]["Status"].upper()}; port mapping: {service.mapping(container)}')
        if not compatible:
            print('GROBID container: NONE')
        healthy = bool(compatible) and any(c['State'].get('Running') for c in compatible) and service.healthy()
        print('GROBID health: ' + ('PASS' if healthy else 'NOT READY'))
        ready &= healthy
    except GrobidStartupError as error:
        print(str(error))
        ready = False
    http = RetrievalHTTP(max_retries=0)
    print(f'S2 rate policy: 1 request / {http.interval:g}s; configured retries: {os.environ.get("SCHOLAREVAL_S2_MAX_RETRIES", "6")}')
    if args.check_s2:
        try:
            http.request('GET', 'https://api.semanticscholar.org/graph/v1/paper/CorpusId:215416146', params={'fields': 'paperId'})
            print('Semantic Scholar connectivity: PASS (' + ('authenticated' if http.api_key else 'anonymous') + ')')
        except RetrievalError as error:
            print(f'Semantic Scholar connectivity: FAIL ({error.category})')
            ready = False
    else:
        print('Semantic Scholar connectivity: NOT TESTED (use --check-s2 for one request)')
    print('OVERALL: ' + ('LOCAL READY' if ready and not args.check_s2 else 'READY' if ready else 'NOT READY'))
    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())
