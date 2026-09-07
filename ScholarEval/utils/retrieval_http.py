"""Secret-free, bounded HTTP retries and a process-shared S2 rate limiter."""
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import tempfile
import threading
import time

import requests


class RetrievalError(RuntimeError):
    category = 'retrieval'
    retryable = False


class RetrievalRateLimitError(RetrievalError):
    category = 'rate-limit (429)'
    retryable = True


class RetrievalAuthError(RetrievalError):
    category = 'authentication/authorization'


class RetrievalServerError(RetrievalError):
    category = 'transient server'
    retryable = True


class RetrievalNetworkError(RetrievalError):
    category = 'network timeout/connection'
    retryable = True


class RetrievalResponseError(RetrievalError):
    category = 'malformed response'


def number(name, default, minimum=0):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return value


class RateLimiter:
    """SQLite lock spans the HTTP attempt; monotonic timestamps survive subprocesses.

    The local machine shares one limiter across stages/runs. A clock reset (reboot)
    resets the stored timestamp; no keys or URLs are written to this database.
    """
    def __init__(self, interval, clock=time.monotonic, sleep=time.sleep, path=None):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.path = path or os.environ.get('SCHOLAREVAL_S2_RATE_DB') or str(Path(tempfile.gettempdir()) / 'scholareval-s2-rate.sqlite3')

    @contextmanager
    def slot(self, metrics):
        conn = sqlite3.connect(self.path, timeout=120)
        try:
            conn.execute('CREATE TABLE IF NOT EXISTS rate (id INTEGER PRIMARY KEY, last REAL)')
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT last FROM rate WHERE id=1').fetchone()
            now = self.clock()
            last = row[0] if row and row[0] <= now else now - self.interval
            wait = max(0, last + self.interval - now)
            if wait:
                self.sleep(wait)
                metrics['total_wait_seconds'] += wait
            # Commit even on HTTP failure, so subsequent processes cannot burst.
            conn.execute('INSERT OR REPLACE INTO rate VALUES (1, ?)', (self.clock(),))
            try:
                yield
            finally:
                conn.commit()
        finally:
            conn.close()


class RetrievalHTTP:
    _announced = False
    _announce_lock = threading.Lock()

    def __init__(self, api_key=None, *, request=None, limiter=None, sleep=time.sleep, jitter=None, max_retries=None):
        self.api_key = api_key or os.environ.get('S2_API_KEY')
        self.interval = number('SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS', 1.1)
        configured_retries = number('SCHOLAREVAL_S2_MAX_RETRIES', 6)
        if not configured_retries.is_integer():
            raise ValueError('SCHOLAREVAL_S2_MAX_RETRIES must be an integer')
        self.max_retries = int(configured_retries) if max_retries is None else max_retries
        self.request_fn = request or requests.request
        self.sleep = sleep
        self.jitter = jitter or (lambda: random.uniform(0, 1))
        self.limiter = limiter or RateLimiter(self.interval)
        self.metrics = dict(requests_attempted=0, requests_succeeded=0, retries=0,
                            rate_limit_429=0, total_wait_seconds=0.0)
        self.on_attempt = None
        with self._announce_lock:
            if not self.__class__._announced:
                print('Semantic Scholar API key: ' + ('configured' if self.api_key else 'not configured'), flush=True)
                if not self.api_key:
                    print('Semantic Scholar anonymous access: best effort / shared public rate limit; 429 may occur.', flush=True)
                self.__class__._announced = True

    def request(self, method, url, **kwargs):
        headers = {k: v for k, v in kwargs.pop('headers', {}).items() if k.lower() != 'x-api-key' and v is not None}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        kwargs.setdefault('timeout', 30)
        for attempt in range(self.max_retries + 1):
            response = None
            with self.limiter.slot(self.metrics):
                self.metrics['requests_attempted'] += 1
                if self.on_attempt:
                    self.on_attempt()
                try:
                    response = self.request_fn(method, url, headers=headers, **kwargs)
                except (requests.Timeout, requests.ConnectionError):
                    error = RetrievalNetworkError('Semantic Scholar connection/timeout failure')
                except requests.RequestException:
                    raise RetrievalError('Semantic Scholar request configuration/transport failure') from None
                else:
                    status = response.status_code
                    if 200 <= status < 300:
                        self.validate_response(response, url, kwargs)
                        self.metrics['requests_succeeded'] += 1
                        return response
                    if status == 429:
                        self.metrics['rate_limit_429'] += 1
                        error = RetrievalRateLimitError('Semantic Scholar rate limit exhausted (HTTP 429)')
                    elif status in (401, 403):
                        error = RetrievalAuthError(f'Semantic Scholar HTTP {status}; check S2_API_KEY and endpoint access')
                    elif status in (500, 502, 503, 504):
                        error = RetrievalServerError(f'Semantic Scholar HTTP {status}')
                    else:
                        error = RetrievalError(f'Semantic Scholar HTTP {status}; request failed')
            if not error.retryable or attempt == self.max_retries:
                error.attempts = attempt + 1
                raise error from None
            delay = min(60, 2 * 2 ** min(attempt, 10)) + self.jitter()
            if response is not None and response.headers.get('Retry-After'):
                raw = response.headers['Retry-After']
                try:
                    retry_after = float(raw)
                except ValueError:
                    try:
                        retry_after = parsedate_to_datetime(raw).timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        retry_after = 0
                if math.isfinite(retry_after):
                    # Do not retry earlier than requested. Long server waits fail
                    # retryably instead of blocking a run for unbounded time.
                    if retry_after > 180:
                        error.attempts = attempt + 1
                        raise error from None
                    delay = max(delay, retry_after)
            self.metrics['retries'] += 1
            logging.warning('S2 attempt %d/%d: %s; retry in %.2fs', attempt + 1,
                            self.max_retries + 1, error.category, delay)
            self.sleep(delay)
            self.metrics['total_wait_seconds'] += delay

    @staticmethod
    def validate_response(response, url, kwargs):
        try:
            data = response.json()
        except ValueError:
            raise RetrievalResponseError('Semantic Scholar returned invalid JSON') from None
        if url.endswith('/batch'):
            ids = kwargs.get('json', {}).get('ids', [])
            valid = isinstance(data, list) and len(data) == len(ids) and all(p is None or isinstance(p, dict) and p.get('paperId') for p in data)
        elif '/search' in url or url.endswith('/references') or '/recommendations/' in url:
            key = 'recommendedPapers' if '/recommendations/' in url else 'data'
            valid = isinstance(data, dict) and isinstance(data.get(key), list) and all(isinstance(p, dict) for p in data[key])
            if valid and '/snippet/' in url:
                try:
                    valid = all(isinstance(p['paper']['corpusId'], (str, int))
                                and isinstance(p['snippet']['annotations']['refMentions'], (list, type(None)))
                                and all(isinstance(ref, dict) and 'matchedPaperCorpusId' in ref
                                        for ref in (p['snippet']['annotations']['refMentions'] or []))
                                for p in data[key])
                except (KeyError, TypeError):
                    valid = False
            elif valid and url.endswith('/references'):
                valid = all('citedPaper' in p and (p['citedPaper'] is None or
                            isinstance(p['citedPaper'], dict) and p['citedPaper'].get('paperId')) for p in data[key])
            elif valid and '/references' not in url:
                valid = all(p.get('paperId') for p in data[key])
        else:
            valid = isinstance(data, dict) and bool(data.get('paperId'))
        if not valid:
            raise RetrievalResponseError('Semantic Scholar response has an unexpected schema')
