"""Durable atomic I/O, SQLite work items and machine-readable run telemetry."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Callable

from .workflow_errors import ConfigurationError, ScientificValidationError


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def atomic_bytes(path: Path | str, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path: Path | str, text: str) -> None:
    atomic_bytes(path, text.encode('utf-8'))


def atomic_json(path: Path | str, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


class StateDB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS items (
                    namespace TEXT, key TEXT, result TEXT NOT NULL, sha256 TEXT NOT NULL,
                    completed REAL NOT NULL, PRIMARY KEY(namespace, key));
                CREATE TABLE IF NOT EXISTS stages (
                    id TEXT PRIMARY KEY, state TEXT NOT NULL, detail TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, time REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metrics (name TEXT PRIMARY KEY, value REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def event(self, kind: str, **detail: Any) -> None:
        with self.connect() as db:
            db.execute('INSERT INTO events(time,kind,detail) VALUES(?,?,?)',
                       (time.time(), kind, json.dumps(detail, ensure_ascii=False)))

    def metric(self, name: str, value: float = 1) -> None:
        with self.connect() as db:
            db.execute('INSERT INTO metrics VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value', (name, value))

    def metrics_snapshot(self, prefix: str, values: dict) -> None:
        with self.connect() as db:
            db.executemany('INSERT OR REPLACE INTO metrics VALUES(?,?)',
                [(prefix + key, value) for key, value in values.items() if isinstance(value, (int, float))])

    def stage(self, stage: str, state: str, **detail: Any) -> None:
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO stages VALUES(?,?,?,?)',
                       (stage, state, json.dumps(detail), time.time()))
        self.event('stage_state', stage=stage, state=state, **detail)

    def summary(self, path: Path | str) -> dict:
        with self.connect() as db:
            value = {
                'stages': {r[0]: {'state': r[1], 'detail': json.loads(r[2]), 'updated': r[3]}
                           for r in db.execute('SELECT * FROM stages')},
                'metrics': dict(db.execute('SELECT * FROM metrics')),
                'completed_items': dict(db.execute('SELECT namespace,count(*) FROM items GROUP BY namespace')),
                'timestamp': time.time(),
            }
        atomic_json(path, value)
        return value


def telemetry() -> StateDB | None:
    path = os.environ.get('SCHOLAREVAL_STATE_DB')
    return StateDB(path) if path else None


def metric(name: str, value: float = 1) -> None:
    state = telemetry()
    if state:
        state.metric(name, value)


class ItemStore:
    """Content-addressed immutable results. Invalid/uncommitted items are retried."""
    def __init__(self, path: Path | str, namespace: str):
        self.state = StateDB(path)
        self.namespace = namespace

    @classmethod
    def current(cls, name: str = 'items') -> 'ItemStore':
        primary = os.environ.get('SCHOLAREVAL_STAGE_PRIMARY', '.scholareval/items')
        path = os.environ.get('SCHOLAREVAL_STATE_DB', primary + '.items.sqlite3')
        namespace = os.environ.get('SCHOLAREVAL_STAGE_FINGERPRINT', 'standalone') + ':' + name
        return cls(path, namespace)

    def load(self, key: Any, validate: Callable[[Any], bool] = lambda _: True) -> Any:
        with self.state.connect() as db:
            row = db.execute('SELECT result,sha256 FROM items WHERE namespace=? AND key=?',
                             (self.namespace, digest(key))).fetchone()
        if row:
            try:
                value = json.loads(row[0])
                if digest(value) == row[1] and validate(value):
                    return value
            except (ValueError, TypeError, KeyError):
                pass
        raise KeyError(key)

    def commit(self, key: Any, value: Any, validate: Callable[[Any], bool] = lambda _: True) -> Any:
        if not validate(value):
            raise ScientificValidationError('Item result failed validation; not committed')
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self.state.connect() as db:
            row = db.execute('SELECT result,sha256 FROM items WHERE namespace=? AND key=?',
                             (self.namespace, digest(key))).fetchone()
            try:
                intact = row and row[1] == digest(json.loads(row[0]))
            except ValueError:
                intact = False
            if intact:
                if row[1] != digest(value):
                    raise ScientificValidationError('Attempt to overwrite a completed work item')
                return value
            if row:
                db.execute('INSERT INTO events(time,kind,detail) VALUES(?,?,?)',
                    (time.time(), 'corrupt_item_quarantined', json.dumps({'namespace': self.namespace,
                        'key': digest(key), 'result': row[0], 'sha256': row[1]})))
            db.execute('INSERT OR REPLACE INTO items VALUES(?,?,?,?,?)',
                       (self.namespace, digest(key), encoded, digest(value), time.time()))
        self.state.metric('workflow.item_commits')
        return value

    def run(self, key: Any, operation: Callable[[], Any], validate: Callable[[Any], bool] = lambda _: True) -> Any:
        try:
            value = self.load(key, validate)
        except KeyError:
            return self.commit(key, operation(), validate)
        self.state.metric('workflow.item_cache_hits')
        return value


@contextmanager
def run_lock(path: Path | str):
    """OS-owned lock: released on hard kill; never use a stale PID as a lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ConfigurationError('Another process owns this run/stage; use a separate save directory') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
