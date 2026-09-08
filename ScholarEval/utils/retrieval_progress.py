"""Persist completed retrieval operations, including successful empty responses."""
import json
from pathlib import Path
import time
import os
import shutil
from .durable import ItemStore, telemetry

from .checkpoints import atomic_json, digest
from .retrieval_http import RetrievalError, RetrievalResponseError


class RetrievalProgress:
    def __init__(self, path, fingerprint, http, resume=False):
        self.path, self.http = Path(path), http
        self.state = {'version': 1, 'fingerprint': fingerprint, 'status': 'pending', 'items': {}, 'metrics': {}}
        self.store = ItemStore(os.environ.get('SCHOLAREVAL_STATE_DB', str(path) + '.items.sqlite3'),
                               'retrieval:' + fingerprint)
        if resume:
            try:
                old = json.loads(self.path.read_text(encoding='utf-8'))
                if old.get('version') == 1 and old.get('fingerprint') == fingerprint and isinstance(old.get('items'), dict):
                    self.state = old
                elif self.path.exists():
                    shutil.copy2(self.path, str(self.path) + '.preserved-' + str(time.time_ns()))
            except (OSError, ValueError):
                pass
        for key in http.metrics:
            http.metrics[key] = self.state.get('metrics', {}).get(key, 0)
        self.save()

    def save(self):
        self.state['metrics'].update(self.http.metrics)
        self.state['timestamp'] = time.time()
        self.state['current'] = sum(v.get('status') == 'succeeded' for v in self.state['items'].values())
        self.state['total'] = len(self.state['items'])
        atomic_json(self.path, self.state)
        state = telemetry()
        if state:
            state.metrics_snapshot('retrieval.' + self.path.stem + '.', self.state['metrics'])

    def pending(self, key):
        self.state['items'].setdefault(key, {'status': 'pending', 'attempts': 0, 'paper_ids': []})

    def run(self, key, operation, ids=lambda data: []):
        self.pending(key)
        item = self.state['items'][key]
        if item.get('status') == 'succeeded' and item.get('result_hash') == digest(item.get('result')):
            print(f'[RESUME] retrieval item {key[:16]} succeeded - skipped', flush=True)
            return item['result']
        # Auth/configuration can be corrected between invocations. Never make a
        # transient credential problem a permanently poisoned item checkpoint.
        item['status'] = 'pending'
        self.state['status'] = 'processing'

        def attempted():
            item['attempts'] += 1
            self.save()

        self.http.on_attempt = attempted
        self.save()
        try:
            def validated_operation():
                value = operation()
                list(ids(value))  # Validate result shape before durable commit.
                return value
            result = self.store.run(key, validated_operation)
            paper_ids = sorted(set(str(p) for p in ids(result)))
            item.update(status='succeeded', result=result, result_hash=digest(result), paper_ids=paper_ids)
            return result
        except RetrievalError as error:
            item.update(status='failed-retryable' if error.retryable else 'failed-terminal', category=error.category)
            self.state['status'] = 'failed'
            raise
        except (ValueError, KeyError, TypeError, AttributeError):
            item.update(status='failed-terminal', category='malformed response')
            self.state['status'] = 'failed'
            raise RetrievalResponseError('Retrieval operation returned malformed data') from None
        finally:
            self.http.on_attempt = None
            self.save()
