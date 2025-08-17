#!/usr/bin/env python3
"""
Retry mechanisms and backoff strategies for vLLM CLI operations.

Provides decorators and utilities for handling transient failures
with exponential backoff and configurable retry policies.
"""
import functools
import logging
import time
from typing import Callable, List, Optional, Type

from .base import VLLMCLIError
from .system import FileSystemError, NetworkError

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        retriable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """
        Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds before first retry
            max_delay: Maximum delay in seconds between retries
            backoff_multiplier: Multiplier for exponential backoff
            retriable_exceptions: List of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retriable_exceptions = retriable_exceptions or [
            ConnectionError,
            TimeoutError,
            FileSystemError,
            NetworkError,
        ]


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator to retry operations with exponential backoff.

    Args:
        config: RetryConfig instance (uses default if None)

    Returns:
        Decorated function with retry behavior
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = config.base_delay

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this exception type is retriable
                    is_retriable = any(
                        isinstance(e, exc_type)
                        for exc_type in config.retriable_exceptions
                    )

                    if not is_retriable or attempt >= config.max_attempts - 1:
                        # Don't retry non-retriable exceptions or on last attempt
                        break

                    logger.warning(
                        f"Attempt {attempt + 1} of {config.max_attempts} failed "
                        f"for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)
                    delay = min(delay * config.backoff_multiplier, config.max_delay)

            # All retries exhausted
            if isinstance(last_exception, VLLMCLIError):
                raise last_exception
            else:
                raise VLLMCLIError(
                    f"Operation failed after {config.max_attempts} attempts: {last_exception}",
                    error_code="RETRY_EXHAUSTED",
                    context={
                        "original_error": str(last_exception),
                        "attempts": config.max_attempts,
                    },
                ) from last_exception

        return wrapper

    return decorator


def retry_on_condition(
    condition_func: Callable[[Exception], bool],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
):
    """
    Decorator to retry operations based on a custom condition.

    Args:
        condition_func: Function that returns True if exception should trigger retry
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        backoff_multiplier: Multiplier for exponential backoff

    Returns:
        Decorated function with conditional retry behavior
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not condition_func(e) or attempt >= max_attempts - 1:
                        break

                    logger.warning(
                        f"Retrying {func.__name__} (attempt {attempt + 1}/{max_attempts}): {e}"
                    )

                    time.sleep(delay)
                    delay *= backoff_multiplier

            # Re-raise the last exception
            raise last_exception

        return wrapper

    return decorator


class RetryableOperation:
    """
    Context manager for retryable operations with detailed control.

    Provides more control over retry logic than decorators,
    allowing for mid-operation decisions about retry attempts.
    """

    def __init__(self, operation_name: str, config: Optional[RetryConfig] = None):
        """
        Initialize retryable operation.

        Args:
            operation_name: Name of the operation for logging
            config: RetryConfig or None for default
        """
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.attempt = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return

        self.last_exception = exc_val
        self.attempt += 1

        # Check if we should retry
        if self.should_retry(exc_val):
            delay = self._calculate_delay()
            logger.warning(
                f"Operation '{self.operation_name}' failed (attempt {self.attempt}/"
                f"{self.config.max_attempts}): {exc_val}. Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            return True  # Suppress the exception to allow retry

        return False  # Let the exception propagate

    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        if self.attempt >= self.config.max_attempts:
            return False

        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.retriable_exceptions
        )

    def _calculate_delay(self) -> float:
        """Calculate delay before next retry attempt."""
        delay = self.config.base_delay * (
            self.config.backoff_multiplier ** (self.attempt - 1)
        )
        return min(delay, self.config.max_delay)


def with_retries(operation_name: str, max_attempts: int = 3) -> Callable:
    """
    Simple retry decorator for common use cases.

    Args:
        operation_name: Name for logging purposes
        max_attempts: Maximum retry attempts

    Returns:
        Decorated function
    """
    config = RetryConfig(max_attempts=max_attempts)
    return retry_with_backoff(config)


# Pre-configured retry decorators for common scenarios
network_retry = retry_with_backoff(
    RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        retriable_exceptions=[ConnectionError, TimeoutError, NetworkError],
    )
)

file_operation_retry = retry_with_backoff(
    RetryConfig(
        max_attempts=2,
        base_delay=0.5,
        retriable_exceptions=[FileSystemError, PermissionError, OSError],
    )
)

server_operation_retry = retry_with_backoff(
    RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=10.0,
        retriable_exceptions=[ConnectionError, TimeoutError],
    )
)


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
) -> float:
    """
    Calculate exponential backoff delay.

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        multiplier: Backoff multiplier

    Returns:
        Delay in seconds
    """
    delay = base_delay * (multiplier**attempt)
    return min(delay, max_delay)


def jittered_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: float = 0.1
) -> float:
    """
    Calculate backoff delay with random jitter to avoid thundering herd.

    Args:
        attempt: Attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Jitter factor (0.0 to 1.0)

    Returns:
        Delay in seconds with jitter
    """
    import random

    base_delay_with_attempt = base_delay * (2**attempt)
    jitter_amount = base_delay_with_attempt * jitter * random.random()
    delay = base_delay_with_attempt + jitter_amount
    return min(delay, max_delay)
