"""Shared, secret-free workflow outcomes across subprocess boundaries."""
from dataclasses import dataclass
import errno
import os
import traceback


class WorkflowError(RuntimeError):
    state = 'FATAL'
    exit_code = 70
    retryable = False


class RetryableError(WorkflowError):
    state, exit_code, retryable = 'RETRYABLE_FAILED', 75, True


class NetworkTransientError(RetryableError):
    pass


class RateLimitError(RetryableError):
    state, exit_code = 'WAITING_RATE_LIMIT', 76


class QuotaPause(RetryableError):
    state, exit_code = 'WAITING_QUOTA', 77


class AuthenticationError(WorkflowError):
    state, exit_code = 'BLOCKED_AUTH', 78


class ConfigurationError(WorkflowError):
    state, exit_code = 'BLOCKED_CONFIG', 79


class ScientificValidationError(WorkflowError):
    state, exit_code = 'FATAL_VALIDATION', 81


class UnsupportedSchemaError(RetryableError):
    state, exit_code = 'RETRYABLE_FAILED', 80


@dataclass(frozen=True)
class Outcome:
    state: str
    exit_code: int
    retryable: bool
    reason: str


def classify(error: BaseException) -> Outcome:
    """Keep existing public exception classes compatible, without traceback parsing."""
    if isinstance(error, WorkflowError):
        return Outcome(error.state, error.exit_code, error.retryable, type(error).__name__)
    name = type(error).__name__
    mapping = {
        'CodexQuotaError': QuotaPause,
        'CodexAuthError': AuthenticationError, 'RetrievalAuthError': AuthenticationError,
        'CodexModelError': ConfigurationError,
        'CodexProcessError': NetworkTransientError, 'CodexTimeoutError': NetworkTransientError,
        'CodexCapacityError': NetworkTransientError, 'CodexRateLimitError': RateLimitError,
        'RetrievalRateLimitError': RateLimitError, 'RetrievalNetworkError': NetworkTransientError,
        'RetrievalServerError': NetworkTransientError, 'GrobidStartupError': NetworkTransientError,
        'RetrievalResponseError': UnsupportedSchemaError,
        'CodexOutputFormatError': ScientificValidationError,
        'CodexProtocolError': ConfigurationError,
    }
    if isinstance(error, KeyboardInterrupt):
        return Outcome('CANCELLED', 130, False, name)
    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EACCES, errno.EROFS}:
        kind = ConfigurationError
    elif isinstance(error, (ModuleNotFoundError, ImportError)):
        kind = ConfigurationError
    elif isinstance(error, (FileNotFoundError, UnicodeError, ValueError, KeyError, TypeError)):
        kind = ScientificValidationError
    else:
        kind = mapping.get(name, WorkflowError)
    return Outcome(kind.state, kind.exit_code, kind.retryable, name)


def diagnostic(error: BaseException) -> dict:
    """Useful failure context without headers, environment values, or locals."""
    message = str(error)[:4000]
    for key, value in os.environ.items():
        if value and any(token in key.upper() for token in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CONNECTION')):
            message = message.replace(value, '[REDACTED]')
    return {'type': type(error).__name__, 'message': message,
            'frames': [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                       for f in traceback.extract_tb(error.__traceback__)]}


def from_exit(code: int) -> Outcome:
    for kind in (RetryableError, RateLimitError, QuotaPause, AuthenticationError,
                 ConfigurationError, UnsupportedSchemaError, ScientificValidationError):
        if kind.exit_code == code:
            return Outcome(kind.state, code, kind.retryable, kind.__name__)
    return Outcome('CANCELLED' if code == 130 else 'FATAL', code, False, 'subprocess_exit')
