import os
import asyncio
import time
import functools
from typing import Callable, Any, TypeVar

from logging_utils import logger

key_file = os.environ.get("HTB_KEY_FILE")
exchange_enabled = True if key_file is not None and os.path.isfile(key_file) else False


OPERATION_CANCELLED = 'Operation cancelled'


def fmt_price(price: float) -> str:
    if price > 1000:
        return f"{price:,.0f}"
    if price > 100:
        return f"{price:,.2f}"
    if price > 1:
        return f"{price:,.3f}"
    if price > 0.1:
        return f"{price:,.4f}"
    if price > 0.01:
        return f"{price:,.5f}"
    if price > 0.001:
        return f"{price:,.6f}"
    return f"{price:,.7f}"


def fmt(value: float) -> str:
    return format(value, ',.2f')


F = TypeVar('F', bound=Callable[..., Any])


def log_execution_time(func: F) -> F:
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore
