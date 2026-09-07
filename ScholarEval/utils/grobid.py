"""Bounded Docker/GROBID startup. Never stop or remove user containers."""
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests


class GrobidStartupError(RuntimeError):
    pass


class GrobidService:
    image = 'lfoppiano/grobid:latest-crf'
    name = 'scholareval-grobid'

    def __init__(self, config='./GROBID_config.json', *, clock=time.monotonic, sleep=time.sleep):
        with open(config, encoding='utf-8') as stream:
            self.url = json.load(stream)['grobid_server'].rstrip('/')
        parsed = urlparse(self.url)
        self.host, self.port = parsed.hostname, parsed.port or 8070
        self.clock, self.sleep = clock, sleep
        self.timeout = float(os.environ.get('SCHOLAREVAL_GROBID_READY_TIMEOUT_SECONDS', '180'))
        self.interval = float(os.environ.get('SCHOLAREVAL_GROBID_POLL_SECONDS', '3'))
        if not 0 < self.timeout <= 3600 or not 0 < self.interval <= 60:
            raise ValueError('GROBID timeout/poll interval must be positive and bounded')
        self.container = None
        self.metrics = {'grobid_startup_seconds': 0.0, 'grobid_action': 'not-needed'}

    def docker(self, *args, timeout=20):
        if not shutil.which('docker'):
            raise GrobidStartupError('Docker CLI unavailable; install Docker Desktop and reopen the shell')
        try:
            result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            raise GrobidStartupError('Docker command unavailable/timed out; check Docker Desktop daemon') from None
        if result.returncode:
            # Do not expose inspect configuration/env; only selected diagnostics below.
            raise GrobidStartupError(f'Docker {args[0]} failed; check daemon, image, and port {self.port}')
        return result.stdout.strip()

    def containers(self):
        ids = self.docker('ps', '-aq').split()
        return json.loads(self.docker('inspect', *ids)) if ids else []

    def mapping(self, container):
        return container.get('HostConfig', {}).get('PortBindings', {}) or {}

    def owns_port(self, container):
        return any(str(binding.get('HostPort')) == str(self.port)
                   for bindings in self.mapping(container).values() for binding in (bindings or []))

    def compatible(self, container):
        return (container.get('Config', {}).get('Image') == self.image and
                any(str(p.get('HostPort')) == str(self.port)
                    for p in (self.mapping(container).get('8070/tcp') or [])))

    def healthy(self, timeout=3):
        try:
            response = requests.get(self.url + '/api/isalive', timeout=timeout)
            return response.status_code == 200 and response.text.strip().lower() == 'true'
        except requests.RequestException:
            return False

    def port_open(self):
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def diagnostics(self):
        if not self.container:
            return f'port: {self.port}; no managed container'
        try:
            info = json.loads(self.docker('inspect', self.container))[0]
            state = info.get('State', {})
            # Docker logs often go to stderr; obtain both streams, bounded.
            result = subprocess.run(['docker', 'logs', '--tail', '20', self.container],
                                    capture_output=True, text=True, timeout=10)
            logs = (result.stdout + result.stderr)[-4000:]
            for name, value in os.environ.items():
                if value and len(value) >= 6 and any(x in name.upper() for x in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                    logs = logs.replace(value, '[REDACTED]')
            oom = state.get('OOMKilled') or state.get('ExitCode') == 137
            return (f'container status: {state.get("Status")}; exit code: {state.get("ExitCode")}; '
                    f'port mapping: {self.mapping(info)}\n'
                    + ('possible cause: Docker memory limit / OOM\n' if oom else '')
                    + f'container log tail:\n{logs}')
        except (GrobidStartupError, OSError, subprocess.TimeoutExpired, ValueError):
            return f'container {self.container}; port {self.port}; Docker diagnostics unavailable'

    def wait_ready(self):
        start = self.clock()
        while self.clock() - start < self.timeout:
            remaining = self.timeout - (self.clock() - start)
            if self.healthy(timeout=max(0.01, min(3, remaining))):
                return
            remaining = self.timeout - (self.clock() - start)
            if remaining > 0:
                self.sleep(min(self.interval, remaining))
        raise GrobidStartupError(f'GROBID failed to become ready after {self.timeout:g}s\n{self.diagnostics()}')

    def ensure_ready(self):
        start = self.clock()
        try:
            if self.host not in ('localhost', '127.0.0.1', '::1'):
                self.metrics['grobid_action'] = 'external'
                self.wait_ready()
                return self.metrics
            containers = self.containers()
            owners = [c for c in containers if self.owns_port(c) and c.get('State', {}).get('Running')]
            if any(not self.compatible(c) for c in owners) or len(owners) > 1:
                raise GrobidStartupError(f'Port {self.port} is occupied by an incompatible/ambiguous container; no containers changed')
            candidates = sorted([c for c in containers if self.compatible(c)], key=lambda c: c['Id'])
            if owners:
                chosen = owners[0]
                self.metrics['grobid_action'] = 'reused'
            else:
                if self.port_open():
                    raise GrobidStartupError(f'Port {self.port} is occupied by an unverified process; no containers changed')
                chosen = candidates[0] if candidates else None
                if chosen:
                    self.container = chosen['Id']
                    self.docker('start', self.container)
                    self.metrics['grobid_action'] = 'started-existing'
                else:
                    if any(c.get('Name') == '/' + self.name for c in containers):
                        raise GrobidStartupError(f'Container name {self.name} belongs to an incompatible container')
                    self.container = self.docker('run', '-d', '--name', self.name,
                        '-p', f'{self.port}:8070', self.image, timeout=180)
                    self.metrics['grobid_action'] = 'created'
            if chosen:
                self.container = chosen['Id']
            print(f'GROBID {self.metrics["grobid_action"]}; waiting for {self.url}/api/isalive', flush=True)
            self.wait_ready()
            return self.metrics
        finally:
            self.metrics['grobid_startup_seconds'] = round(self.clock() - start, 3)


def parse_pdfs(s2, pdf_paths, pdf_dir, config='./GROBID_config.json', metrics=None):
    """No Docker, client construction or localhost traffic for an empty PDF set."""
    if not pdf_paths:
        print('No PDFs to parse; GROBID skipped', flush=True)
        return {'grobid_startup_seconds': 0.0, 'grobid_action': 'not-needed'}
    service = GrobidService(config)
    try:
        service.ensure_ready()
        from grobid_client.grobid_client import GrobidClient
        client = GrobidClient(config_path=config)
        s2.extract_sections_from_pdf(client, str(pdf_dir))
        return service.metrics
    finally:
        if metrics is not None:
            metrics.update(service.metrics)


def parse_pdf_corpus(pdf_paths, pdf_dir, on_result, config='./GROBID_config.json', metrics=None):
    """Isolate document failures; fail if the service is unavailable/systemically failing.

    The installed client's recursive 503 retry has no bound. Return 503 to this
    caller instead, which makes at most two attempts per document per invocation.
    """
    if not pdf_paths:
        return {}
    service = GrobidService(config)
    from .checkpoints import file_hash
    try:
        service.ensure_ready()
        from grobid_client.grobid_client import GrobidClient

        class BoundedClient(GrobidClient):
            def _handle_server_busy_retry(self, file_path, *args, **kwargs):
                return file_path, 503, None

            def post(self, *args, **kwargs):
                # Some client versions catch IOError before RequestException in
                # process_pdf; requests timeouts inherit OSError and become 400.
                # Keep transport failures distinguishable from document errors.
                try:
                    return super().post(*args, **kwargs)
                except requests.RequestException as error:
                    response = requests.Response()
                    response.status_code = 408 if isinstance(error, requests.Timeout) else 503
                    response._content = f'GROBID transport failure: {type(error).__name__}'.encode()
                    return response, response.status_code

        client = BoundedClient(config_path=config)
        workers = int(os.environ.get('SCHOLAREVAL_GROBID_WORKERS', '10'))
        timeout = float(os.environ.get('SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS', '60'))
        if not 1 <= workers <= 10 or not 0 < timeout <= 3600:
            raise ValueError('GROBID workers must be 1..10 and parse timeout must be > 0 and <= 3600 seconds')
        # Preserve the configured timeout unless the caller explicitly overrides it.
        if 'SCHOLAREVAL_GROBID_PARSE_TIMEOUT_SECONDS' in os.environ:
            client.config['timeout'] = timeout

        def process(pdf):
            for attempt in range(1, 3):
                try:
                    _, status, content = client.process_pdf('processFulltextDocument', str(pdf))
                except (requests.RequestException, OSError):
                    status, content = 503, None
                if status != 503 or attempt == 2:
                    break
                time.sleep(3)
            record = {'parse_attempts': attempt, 'http_status': status, 'pdf_sha256': file_hash(pdf)}
            if status == 200 and content:
                try:
                    root = ET.fromstring(content)
                    if root.tag.split('}')[-1] != 'TEI':
                        raise ValueError('not TEI XML')
                    target = Path(pdf_dir) / (pdf.stem + '.grobid.tei.xml')
                    target.write_text(content, encoding='utf-8')
                    record.update(status='parsed', xml_sha256=file_hash(target))
                    return record
                except (ET.ParseError, ValueError):
                    record.update(status='parse_failed', failure_reason='Malformed GROBID TEI XML')
                    return record
            record.update(status='parse_failed', failure_reason=f'GROBID HTTP {status}; no usable full text',
                          failure_detail=content[:1000] if isinstance(content, str) else None)
            return record

        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(process, Path(pdf)): Path(pdf) for pdf in pdf_paths}
            for future in as_completed(jobs):
                pdf = jobs[future]
                record = future.result()
                results[pdf.stem] = record
                on_result(pdf.stem, record)
                print(f'GROBID {pdf.stem}: {record["status"]}', flush=True)
        healthy = service.healthy()
        systemic = (not any(r['status'] == 'parsed' for r in results.values()) and
                    all(r.get('http_status', 0) in (408, 429, 500, 502, 503, 504) for r in results.values()))
        if not healthy or systemic:
            # Infrastructure failures remain retryable on resume, even when health
            # is superficially green (e.g. every document returned HTTP 500).
            for cid, record in results.items():
                if record['status'] != 'parsed':
                    record['status'] = 'parse_retryable'
                    on_result(cid, record)
            raise GrobidStartupError('GROBID service/systemic parsing failure; partial progress saved\n' + service.diagnostics())
        return results
    finally:
        if metrics is not None:
            metrics.update(service.metrics)
