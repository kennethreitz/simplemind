import logging
import time
from typing import Any, Callable

import logfire

from .settings import settings

_verbose_logger = logging.getLogger("simplemind")


def log_verbose(message: str, enabled: bool | None = None) -> None:
    """Log a message at INFO level when verbose mode is on.

    `enabled` is a per-call override; when it is None, `settings.VERBOSE` decides.
    Messages go to the stdlib logger "simplemind". If nothing in the logger's
    chain has a handler yet, a stderr handler is added to that logger, so the
    root logger is left alone.
    """
    if not (settings.VERBOSE if enabled is None else enabled):
        return

    if not _verbose_logger.hasHandlers():
        _verbose_logger.addHandler(logging.StreamHandler())
    _verbose_logger.setLevel(logging.INFO)
    _verbose_logger.info(message)


def logger(func: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that logs the function parameters, function returns,
    and exceptions raised if logging is enabled, using logfire.
    """

    def wrapper(*args, **kwargs) -> Any:
        if not settings.logging.is_enabled:
            return func(*args, **kwargs)

        logfire.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        t1 = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            t2 = time.perf_counter()
            logfire.info(f"{func.__name__} returned: {result} in {t2-t1} seconds")

            return result

        except Exception as e:
            t2 = time.perf_counter()
            logfire.error(f"Error in {func.__name__}: {e} in {t2-t1} seconds")
            raise e

    return wrapper
