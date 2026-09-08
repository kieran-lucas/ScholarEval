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
import json
from urllib.parse import urlsplit
import uuid

from .durable import ItemStore, atomic_json, metric

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
    retryable = True


def retry_after_seconds(raw, now=None):
    try:
        value = float(raw)
    except (ValueError, TypeError):
        try:
            value = parsedate_to_datetime(raw).timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, value) if math.isfinite(value) else 0.0


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
    def __init__(self, interval, clock=time.time, sleep=time.sleep, path=None):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.lease_seconds = 120
        self.path = path or os.environ.get('SCHOLAREVAL_S2_RATE_DB') or str(Path(tempfile.gettempdir()) / 'scholareval-s2-rate.sqlite3')

    @contextmanager
    def slot(self, metrics):
        # Reserve durably BEFORE HTTP: process death cannot roll pacing back.
        owner = uuid.uuid4().hex
        while True:
            with self.connection() as conn:
                row = conn.execute('SELECT last,interval,cooldown,failures,streak,state FROM adaptive WHERE id=1').fetchone()
                last, interval, cooldown, failures, streak, state = row
                now = self.clock()
                lease = conn.execute('SELECT until FROM lease WHERE id=1').fetchone()[0]
                wait = max(0, last + max(interval, self.interval) - now, cooldown - now, lease - now)
                if wait <= 0:
                    conn.execute('UPDATE adaptive SET last=?,state=? WHERE id=1',
                                 (now, 'HALF_OPEN' if state == 'OPEN' else state))
                    conn.execute('UPDATE lease SET until=?,owner=? WHERE id=1', (now + self.lease_seconds, owner))
            if wait <= 0:
                break
            self.sleep(min(wait, 60))
            metrics['total_wait_seconds'] += min(wait, 60)
            metric('s2.cooldown_seconds', min(wait, 60))
        try:
            yield
        finally:
            with self.connection() as conn:
                conn.execute('UPDATE lease SET until=0,owner=NULL WHERE id=1 AND owner=?', (owner,))

    @contextmanager
    def connection(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=60)
        try:
            conn.execute('PRAGMA synchronous=FULL')
            conn.execute('CREATE TABLE IF NOT EXISTS adaptive (id INTEGER PRIMARY KEY,last REAL,interval REAL,cooldown REAL,failures INTEGER,streak INTEGER,state TEXT)')
            conn.execute('CREATE TABLE IF NOT EXISTS lease (id INTEGER PRIMARY KEY,until REAL,owner TEXT)')
            conn.commit()
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('INSERT OR IGNORE INTO adaptive VALUES(1,?,?,?,?,?,?)',
                         (self.clock() - self.interval, self.interval, 0, 0, 0, 'CLOSED'))
            conn.execute('INSERT OR IGNORE INTO lease VALUES(1,0,NULL)')
            yield conn
            conn.commit()
        finally:
            conn.close()

    def observe(self, status, retry_after=0):
        with self.connection() as conn:
            interval, failures, streak = conn.execute('SELECT interval,failures,streak FROM adaptive WHERE id=1').fetchone()
            if status == 429 or status >= 500:
                failures += 1
                interval = min(120, max(self.interval, interval) * (2 if status == 429 else 1.25))
                cooldown = max(retry_after, interval, min(900, 15 * 2 ** min(failures, 6)))
                state = 'OPEN' if failures >= 3 else 'CLOSED'
                conn.execute('UPDATE adaptive SET interval=?,cooldown=?,failures=?,streak=0,state=? WHERE id=1',
                             (interval, self.clock() + cooldown, failures, state))
            elif 200 <= status < 300:
                streak += 1
                if streak >= 10:
                    interval, streak = max(self.interval, interval * 0.9), 0
                conn.execute('UPDATE adaptive SET interval=?,failures=0,streak=?,state="CLOSED" WHERE id=1', (interval, streak))


class RetrievalHTTP:
    _announced = False
    _announce_lock = threading.Lock()

    def __init__(self, api_key=None, *, request=None, limiter=None, sleep=time.sleep, jitter=None, max_retries=None):
        self.api_key = api_key or os.environ.get('S2_API_KEY')
        self.interval = number('SCHOLAREVAL_S2_MIN_INTERVAL_SECONDS', 5.0)
        configured_retries = number('SCHOLAREVAL_S2_MAX_RETRIES', 6)
        if not configured_retries.is_integer():
            raise ValueError('SCHOLAREVAL_S2_MAX_RETRIES must be an integer')
        self.max_retries = int(configured_retries) if max_retries is None else max_retries
        self.request_fn = request or requests.request
        self.injected_request = request is not None
        self.sleep = sleep
        self.jitter = jitter or (lambda: random.uniform(0, 1))
        self.limiter = limiter or RateLimiter(self.interval)
        self.metrics = dict(requests_attempted=0, requests_succeeded=0, retries=0,
                            rate_limit_429=0, remote_5xx=0, malformed_records_skipped=0,
                            cache_hits=0, total_wait_seconds=0.0)
        cache_path = os.environ.get('SCHOLAREVAL_S2_CACHE_DB')
        self.cache = ItemStore(cache_path or '.scholareval/s2-responses.sqlite3', 's2-http-v1') if cache_path or request is None else None
        self.on_attempt = None
        with self._announce_lock:
            if not self.__class__._announced:
                print('Semantic Scholar API key: ' + ('configured' if self.api_key else 'not configured'), flush=True)
                if not self.api_key:
                    print('Semantic Scholar anonymous access: best effort / shared public rate limit; 429 may occur.', flush=True)
                self.__class__._announced = True

    def request(self, method, url, **kwargs):
        identity = [method.upper(), url, kwargs.get('params', {}), kwargs.get('json')]
        if self.cache:
            try:
                data = self.cache.load(identity)
            except KeyError:
                pass
            else:
                self.metrics['cache_hits'] += 1
                metric('s2.cache_hits')
                return self.json_response(data)
        if os.environ.get('SCHOLAREVAL_OFFLINE') == '1' and not self.injected_request:
            from .workflow_errors import ConfigurationError
            raise ConfigurationError('Offline guard: uncached S2 request refused')
        headers = {k: v for k, v in kwargs.pop('headers', {}).items() if k.lower() != 'x-api-key' and v is not None}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        kwargs.setdefault('timeout', 30)
        if isinstance(self.limiter, RateLimiter):
            timeout = kwargs['timeout']
            duration = sum(timeout) if isinstance(timeout, tuple) else timeout
            self.limiter.lease_seconds = max(120, float(duration) + 10)
        for attempt in range(self.max_retries + 1):
            response = None
            with self.limiter.slot(self.metrics):
                self.metrics['requests_attempted'] += 1
                metric('s2.requests')
                metric('s2.endpoint.' + urlsplit(url).path)
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
                        try:
                            self.validate_response(response, url, kwargs)
                            data, skipped = self.filter_records(response.json(), url)
                        except RetrievalResponseError as invalid:
                            self.quarantine(response, url, str(invalid))
                            error = invalid
                        else:
                            self.metrics['malformed_records_skipped'] += skipped
                            metric('s2.malformed_records_skipped', skipped)
                            if skipped:
                                logging.warning('S2 skipped %d malformed records on %s', skipped, urlsplit(url).path)
                            if self.cache:
                                self.cache.commit(identity, data)
                            if isinstance(self.limiter, RateLimiter):
                                self.limiter.observe(status)
                            self.metrics['requests_succeeded'] += 1
                            return self.json_response(data)
                    elif status == 429:
                        self.metrics['rate_limit_429'] += 1
                        metric('s2.429')
                        error = RetrievalRateLimitError('Semantic Scholar rate limit exhausted (HTTP 429)')
                    elif status in (401, 403):
                        error = RetrievalAuthError(f'Semantic Scholar HTTP {status}; check S2_API_KEY and endpoint access')
                    elif status in (500, 502, 503, 504):
                        self.metrics['remote_5xx'] += 1
                        metric('s2.5xx')
                        error = RetrievalServerError(f'Semantic Scholar HTTP {status}')
                    else:
                        error = RetrievalError(f'Semantic Scholar HTTP {status}; request failed')
            retry_after = retry_after_seconds(response.headers.get('Retry-After')) if response is not None else 0
            if response is not None and isinstance(self.limiter, RateLimiter):
                self.limiter.observe(response.status_code if response.status_code >= 400 else 503, retry_after)
            if not error.retryable or attempt == self.max_retries or retry_after > 180:
                error.attempts = attempt + 1
                error.retry_after = retry_after
                raise error from None
            delay = min(60, 2 * 2 ** min(attempt, 10)) + self.jitter()
            delay = max(delay, retry_after)
            self.metrics['retries'] += 1
            metric('s2.retries')
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
            valid = isinstance(data, list) and len(data) == len(ids)
        elif '/search' in url or url.endswith('/references') or '/recommendations/' in url:
            key = 'recommendedPapers' if '/recommendations/' in url else 'data'
            valid = isinstance(data, dict) and isinstance(data.get(key), list)
        else:
            valid = isinstance(data, dict) and bool(data.get('paperId'))
        if not valid:
            raise RetrievalResponseError('Semantic Scholar response has an unexpected schema')

    @staticmethod
    def json_response(data):
        response = requests.Response()
        response.status_code = 200
        response.encoding = 'utf-8'
        response._content = json.dumps(data, ensure_ascii=False).encode('utf-8')
        return response

    @staticmethod
    def filter_records(data, url):
        def usable(p):
            if not isinstance(p, dict):
                return False
            if '/snippet/' in url:
                try:
                    mentions = p['snippet']['annotations'].get('refMentions') or []
                    if not isinstance(mentions, list):
                        return False
                    p['snippet']['annotations']['refMentions'] = [r for r in mentions if isinstance(r, dict) and 'matchedPaperCorpusId' in r]
                    return isinstance(p['paper']['corpusId'], (str, int))
                except (KeyError, TypeError, AttributeError):
                    return False
            if url.endswith('/references'):
                p = p.get('citedPaper')
            valid = isinstance(p, dict) and isinstance(p.get('paperId'), str) and bool(p['paperId'])
            if valid:
                for field in ('title', 'abstract', 'venue', 'publicationDate'):
                    if field in p and not isinstance(p[field], (str, type(None))):
                        p[field] = None
                if 'authors' in p:
                    authors = p['authors'] if isinstance(p['authors'], list) else []
                    p['authors'] = [a for a in authors if isinstance(a, dict) and isinstance(a.get('name'), str)]
                if 'openAccessPdf' in p and not isinstance(p['openAccessPdf'], (dict, type(None))):
                    p['openAccessPdf'] = None
            return valid
        if url.endswith('/batch'):
            return [p if usable(p) else None for p in data], sum(p is not None and not usable(p) for p in data)
        key = 'recommendedPapers' if '/recommendations/' in url else 'data'
        if isinstance(data, dict) and isinstance(data.get(key), list):
            original = data[key]
            filtered = [p for p in original if usable(p)]
            result = dict(data, **{key: filtered})
            if url.endswith('/references'):
                result['_raw_count'] = len(original)
            return result, len(original) - len(filtered)
        return data, 0

    def quarantine(self, response, url, reason):
        folder = Path(os.environ.get('SCHOLAREVAL_DIAGNOSTICS_DIR', '.scholareval/diagnostics'))
        try:
            raw = json.dumps(response.json(), ensure_ascii=False)[:4000]
        except ValueError:
            raw = str(response.text)[:4000]
        secrets = [self.api_key] + [v for k, v in os.environ.items() if any(t in k.upper() for t in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CONNECTION'))]
        for value in secrets:
            if value:
                raw = raw.replace(value, '[REDACTED]')
        atomic_json(folder / ('s2-' + uuid.uuid4().hex + '.json'),
                    {'endpoint': urlsplit(url).path, 'status': response.status_code,
                     'snippet': raw, 'timestamp': time.time(), 'reason': reason})
