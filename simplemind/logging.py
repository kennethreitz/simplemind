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

        # See logfire manual tracing docs: https://logfire.pydantic.dev/docs/guides/onboarding-checklist/add-manual-tracing/#exceptions
        with logfire.span("{event}", event="function called", function=func.__name__, args=args, kwargs=kwargs):
            
            t1 = time.perf_counter()

            try:
                is_streaming = "generate_stream_text" in func.__name__ or kwargs.get("stream")
                if is_streaming:
                    chunks = []
                    for chunk in func(*args, **kwargs):
                        chunks.append(chunk)
                        yield chunk
                    result = "".join(chunks)
                    # note: no need to log the function name here, as it's already in the span
                    logfire.info(
                        "{event}",
                        event="function completed",
                        result=result[:2000],
                        chunk_count=len(chunks),
                        duration=time.perf_counter() - t1
                    )
                else:
                    result = func(*args, **kwargs)
                    logfire.info(
                        "{event}",
                        event="function completed",
                        result=str(result)[:2000],
                        duration=time.perf_counter() - t1
                    )
                    return result

            except Exception as e:
                logfire.error(
                    "{event}",
                    event="function failed",
                    error=str(e),
                    duration=time.perf_counter() - t1,
                    streamed_chunks=len(chunks) if is_streaming else None,
                    partial_streaming_result="".join(chunks)[:2000] if is_streaming else None,
                )
                raise e

    return wrapper
