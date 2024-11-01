import time
from typing import Any, Callable

import logfire

from .settings import settings


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
