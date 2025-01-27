import os
import asyncio
import time

from logging_utils import logger

exchange_enabled = True if os.environ.get("HTB_KEY_FILE") is not None and os.path.isfile(os.environ.get("HTB_KEY_FILE")) else False


OPERATION_CANCELLED = 'Operation cancelled'


def fmt_price(price: float) -> str:
    if price > 1000:
        return format(price, ',.0f')
    if price > 1:
        return format(price, ',.2f')
    if price > 0.1:
        return format(price, ',.4f')
    if price > 0.01:
        return format(price, ',.5f')
    if price > 0.001:
        return format(price, ',.6f')
    return format(price, ',.7f')


def fmt(value: float) -> str:
    return format(value, ',.2f')


def px_round(value: float) -> str:
    return round(float(f"{value:.5g}"), 6)


def log_execution_time(func):
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
            return result
        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
            return result
        return sync_wrapper