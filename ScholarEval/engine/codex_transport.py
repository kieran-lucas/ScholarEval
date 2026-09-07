"""Native Codex app-server JSON-RPC stdio transport (schema: CLI 0.153.4)."""

import json
import os
import queue
import re
import shutil
import subprocess
import threading


class CodexError(RuntimeError):
    """Safe-to-display backend failure; never contains raw server payloads."""


class CodexProtocolError(CodexError):
    pass


class CodexTimeoutError(CodexError):
    pass


class CodexProcessError(CodexError):
    pass


class CodexAuthError(CodexError):
    pass


class CodexModelError(CodexError):
    pass


class CodexQuotaError(CodexError):
    pass


class CodexRateLimitError(CodexError):
    pass


class CodexCapacityError(CodexError):
    pass


class CodexOutputFormatError(CodexError):
    pass


def server_error(payload):
    """Classify documented codes; deliberately do not echo upstream messages."""
    if not isinstance(payload, dict):
        return CodexProtocolError('Malformed Codex error payload; details suppressed.')
    info = payload.get('codexErrorInfo')
    if not info and isinstance(payload.get('data'), dict):
        info = payload['data'].get('codexErrorInfo')
    if isinstance(info, dict):
        kind = next(iter(info), '')
        details = info.get(kind)
        status = details.get('httpStatusCode') if isinstance(details, dict) else None
    else:
        kind, status = info, None
    if kind == 'usageLimitExceeded':
        return CodexQuotaError('ChatGPT Work/Codex allowance is exhausted. Resume after reset; no API fallback was used.')
    if kind == 'sessionBudgetExceeded':
        return CodexQuotaError('Codex session budget exhausted; stop and review the configured budget before resuming.')
    if kind == 'unauthorized' or status in (401, 403):
        return CodexAuthError('Codex authentication failed. Run codex login with ChatGPT, then retry.')
    if kind == 'rateLimitExceeded' or status == 429:
        return CodexRateLimitError('Codex rate limit reached. Stop and retry later; no automatic quota retry.')
    if kind == 'serverOverloaded' or status in (502, 503, 504) or payload.get('code') == -32001:
        return CodexCapacityError('Codex server capacity is temporarily unavailable.')
    if payload.get('code') in (-32700, -32600, -32601, -32602):
        return CodexProtocolError('Codex rejected the protocol request. Check the installed CLI schema and configuration.')
    return CodexError('Codex request/turn failed. No result was accepted; no model or API fallback was used.')


def find_codex(executable=None):
    requested = executable or os.environ.get('SCHOLAREVAL_CODEX_EXECUTABLE') or ('codex.cmd' if os.name == 'nt' else 'codex')
    found = shutil.which(requested)
    if not found:
        raise CodexProcessError('Codex executable not found. Install official Codex CLI >= 0.153.0.')
    return found


def codex_version(executable):
    try:
        result = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        raise CodexProcessError('Could not execute codex --version.') from None
    match = re.search(r'codex-cli (\d+)\.(\d+)\.(\d+)([^\s]*)', result.stdout)
    if result.returncode or not match:
        raise CodexProcessError('Unrecognized Codex CLI version response.')
    version = tuple(map(int, match.group(1, 2, 3)))
    if version < (0, 153, 0) or match.group(4):
        raise CodexProcessError('Codex CLI >= 0.153.0 (stable) is required for Astra; upgrade the official CLI.')
    return '.'.join(map(str, version))


class JsonRpcClient:
    """One reader correlates replies and routes notifications by ephemeral thread."""

    def __init__(self, process, timeout=30):
        self.process = process
        self.timeout = timeout
        self._lock = threading.RLock()
        self._outgoing = queue.Queue()
        self._pending = {}
        self._threads = {}
        self._next_id = 0
        self._failure = None
        self._closed = False
        self.on_notification = None
        self._reader = threading.Thread(target=self._read, daemon=True, name='codex-rpc')
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True, name='codex-stderr')
        self._writer = threading.Thread(target=self._write, daemon=True, name='codex-input')
        self._reader.start()
        self._stderr_reader.start()
        self._writer.start()

    def initialize(self):
        result = self.request('initialize', {
            'clientInfo': {'name': 'scholareval', 'title': 'ScholarEval', 'version': '0.1.0'},
            # environments=[] and allowProviderModelFallback are schema-gated.
            'capabilities': {'experimentalApi': True},
        })
        self.notify('initialized', {})
        return result

    def _send(self, message):
        with self._lock:
            if self._failure:
                raise self._failure
            self._outgoing.put(json.dumps(message, ensure_ascii=False) + '\n')

    def _write(self):
        # A stalled child must not block the requesting thread in pipe.write().
        try:
            while True:
                line = self._outgoing.get()
                if line is None or self._closed:
                    break
                self.process.stdin.write(line)
                self.process.stdin.flush()
        except (OSError, ValueError):
            self._fail(CodexProcessError('Codex app-server closed its input stream.'))
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass

    def request(self, method, params=None, timeout=None):
        reply = queue.Queue()
        with self._lock:
            if self._failure:
                raise self._failure
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = reply
        try:
            self._send({'id': request_id, 'method': method, 'params': params or {}})
            try:
                value = reply.get(timeout=self.timeout if timeout is None else max(0, timeout))
            except queue.Empty:
                raise CodexTimeoutError('Codex app-server request timed out.') from None
            if isinstance(value, BaseException):
                raise value
            if 'error' in value:
                raise server_error(value['error'])
            if not isinstance(value.get('result'), dict):
                raise CodexProtocolError('Codex returned an invalid result object.')
            return value['result']
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def notify(self, method, params):
        self._send({'method': method, 'params': params})

    def subscribe(self, thread_id):
        with self._lock:
            if self._failure:
                raise self._failure
            events = queue.Queue()
            self._threads[thread_id] = events
            return events

    def unsubscribe(self, thread_id):
        with self._lock:
            self._threads.pop(thread_id, None)

    def _fail(self, error):
        with self._lock:
            if self._failure:
                return
            self._failure = error
            for recipient in list(self._pending.values()) + list(self._threads.values()):
                recipient.put(error)

    def _read(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError
                if 'method' in message:
                    method, params = message['method'], message.get('params', {})
                    if not isinstance(method, str) or not isinstance(params, dict):
                        raise ValueError
                    if 'id' in message:
                        # Never approve tools, provide tokens, or execute server requests.
                        self._send({'id': message['id'], 'error': {
                            'code': -32601, 'message': 'ScholarEval does not execute tools or approvals.',
                        }})
                        with self._lock:
                            target = self._threads.get(params.get('threadId'))
                        if target:
                            target.put(CodexProtocolError('Codex requested a tool/approval during text inference.'))
                        else:
                            self._fail(CodexProtocolError('Codex requested an unsupported client operation.'))
                    else:
                        if self.on_notification:
                            self.on_notification(method, params)
                        with self._lock:
                            target = self._threads.get(params.get('threadId'))
                        if target:
                            target.put((method, params))
                elif 'id' in message and (('result' in message) != ('error' in message)):
                    if 'error' in message and not isinstance(message['error'], dict):
                        raise ValueError
                    with self._lock:
                        target = self._pending.get(message['id'])
                    if target:
                        target.put(message)
                else:
                    raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError):
            self._fail(CodexProtocolError('Malformed JSON-RPC from Codex app-server; raw payload suppressed.'))
        except (OSError, UnicodeError, CodexError):
            self._fail(CodexProcessError('Codex app-server output stream failed.'))
        finally:
            self._fail(CodexProcessError('Codex app-server exited or closed stdout. Restart the ScholarEval run to resume.'))

    def _drain_stderr(self):
        # Drain without retaining or logging potentially sensitive auth diagnostics.
        try:
            while self.process.stderr.read(4096):
                pass
        except (OSError, ValueError):
            pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._fail(CodexProcessError('Codex app-server was closed.'))
        self._outgoing.put(None)  # Writer closes stdin; EOF requests shutdown.
        try:
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if os.name == 'nt' and self.process.poll() is None:
                # The official npm .cmd launcher has child processes on Windows.
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            elif self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
        for reader in (self._reader, self._stderr_reader, self._writer):
            if reader is not threading.current_thread():
                reader.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()
