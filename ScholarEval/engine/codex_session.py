"""Share the native engine across ScholarEval's existing subprocess stages.

Private loopback JSON IPC, not HTTP or an OpenAI-compatible proxy. Only text
inference is exposed. The random run capability is unrelated to Codex auth.
"""

import atexit
import hmac
import json
import os
import secrets
import socket
import socketserver
import threading

from . import codex_transport
from .codex_app_server_engine import CodexAppServerEngine
from .codex_transport import CodexError, CodexProcessError, CodexProtocolError

CONNECTION_ENV = 'SCHOLAREVAL_CODEX_CONNECTION'
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_lock = threading.RLock()
_engine = None
_host = None


def _read_json(stream):
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line or len(line) > MAX_MESSAGE_BYTES:
        raise CodexProtocolError('Invalid ScholarEval engine IPC response size.')
    value = json.loads(line)
    if not isinstance(value, dict):
        raise CodexProtocolError('Invalid ScholarEval engine IPC object.')
    return value


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(self.server.engine.timeout * 2 + 30)
        try:
            request = _read_json(self.rfile)
            if not hmac.compare_digest(str(request.get('capability', '')), self.server.capability):
                return
            if request.get('method') != 'respond':
                raise CodexProtocolError('Unsupported ScholarEval engine IPC method.')
            result = self.server.engine.respond(**request['params'])
            response = {'result': result}
        except CodexError as error:
            response = {'error': {'type': type(error).__name__, 'message': str(error)}}
        except Exception:
            response = {'error': {'type': 'CodexProtocolError', 'message': 'ScholarEval engine IPC failed; details suppressed.'}}
        try:
            self.wfile.write((json.dumps(response) + '\n').encode())
        except OSError:
            pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


class CodexRun:
    def __init__(self, cache_path=None, engine=None):
        self.engine = engine or CodexAppServerEngine(cache_path=cache_path)
        self._closed = False
        try:
            self.server = _Server(('127.0.0.1', 0), _Handler)
            self.server.engine = self.engine
            self.server.capability = secrets.token_hex(32)
            self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='scholareval-engine')
            self._thread.start()
            atexit.register(self.close)
        except BaseException:
            self.engine.close()
            raise

    def environment(self):
        return {CONNECTION_ENV: json.dumps({
            'port': self.server.server_address[1], 'capability': self.server.capability,
            'timeout': self.engine.timeout, 'model': self.engine.model,
        })}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self.engine.close()
        self._thread.join(timeout=2)
        atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _RemoteEngine:
    def __init__(self, connection):
        self.connection = json.loads(connection)
        self.llm_engine_name = self.connection['model']

    def respond(self, user_input, temperature=0.7, top_p=0.95, max_tokens=40000, **kwargs):
        params = dict(user_input=user_input, temperature=temperature, top_p=top_p, max_tokens=max_tokens, **kwargs)
        try:
            with socket.create_connection(('127.0.0.1', self.connection['port']), timeout=5) as connection:
                connection.settimeout(self.connection['timeout'] * 2 + 30)
                request = {'capability': self.connection['capability'], 'method': 'respond', 'params': params}
                connection.sendall((json.dumps(request) + '\n').encode())
                with connection.makefile('rb') as stream:
                    response = _read_json(stream)
        except (OSError, ValueError):
            raise CodexProcessError('The run-owned ScholarEval Codex engine connection failed. Resume the run.') from None
        if 'error' in response:
            error = response['error']
            error_type = getattr(codex_transport, error['type'], CodexError)
            if not isinstance(error_type, type) or not issubclass(error_type, CodexError):
                error_type = CodexError
            raise error_type(error['message'])
        return tuple(response['result'])


def get_codex_engine():
    global _engine
    with _lock:
        if _engine is None:
            connection = os.environ.get(CONNECTION_ENV)
            _engine = _RemoteEngine(connection) if connection else CodexAppServerEngine()
        return _engine


def run_environment(cache_path=None):
    """Called by the CLI parent; stages inherit only the local run connection."""
    global _host
    if os.environ.get('SCHOLAREVAL_LLM_BACKEND', 'litellm').lower() != 'codex':
        return {}
    with _lock:
        if _host is None:
            _host = CodexRun(cache_path=cache_path)
        return _host.environment()


def close_run():
    global _host
    with _lock:
        if _host:
            _host.close()
            _host = None
