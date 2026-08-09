"""Datastar integration helpers.

``datastar_py`` ships a ``@datastar_response`` decorator that relies on
``functools.wraps``. Newer FastAPI versions follow ``__wrapped__`` when
detecting generator endpoints and therefore mistake the wrapped route for a
raw async generator, breaking its SSE/JSONL handling. This local decorator
preserves the original signature via ``__signature__`` (so FastAPI can still
bind path/query params) without exposing ``__wrapped__``.
"""

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from datastar_py.starlette import DatastarResponse


def datastar_action(
    func: Callable[..., Any],
) -> Callable[..., Awaitable[DatastarResponse]]:
    """Wrap an async generator route so it returns a :class:`DatastarResponse`.

    The decorated function may be an async generator (yielding SSE events) or
    an async function returning an iterable of SSE events; the result is
    wrapped in a Datastar SSE response without confusing FastAPI's generator
    detection.
    """

    async def wrapper(*args: Any, **kwargs: Any) -> DatastarResponse:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return DatastarResponse(result)

    wrapper.__name__ = getattr(func, "__name__", wrapper.__name__)
    wrapper.__qualname__ = getattr(func, "__qualname__", wrapper.__qualname__)
    wrapper.__module__ = getattr(func, "__module__", wrapper.__module__)
    wrapper.__doc__ = getattr(func, "__doc__", wrapper.__doc__)
    wrapper.__annotations__ = {**getattr(func, "__annotations__", {})}
    wrapper.__annotations__["return"] = DatastarResponse
    wrapper.__signature__ = inspect.signature(func)
    return wrapper
