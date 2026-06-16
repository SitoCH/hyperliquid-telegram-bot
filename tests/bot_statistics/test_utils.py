import pytest
from bot_statistics.utils import get_historical_price, format_timestamp
from datetime import datetime


def test_get_historical_price():
    price_history = {
        1000: 10.0,
        2000: 20.0,
        3000: 30.0
    }
    
    # Exact match
    assert get_historical_price(1000, price_history) == 10.0
    assert get_historical_price(2000, price_history) == 20.0
    
    # Before earliest
    assert get_historical_price(500, price_history) == 10.0
    
    # After latest
    assert get_historical_price(4000, price_history) == 30.0
    
    # Closest (before)
    assert get_historical_price(1400, price_history) == 10.0
    
    # Closest (after)
    assert get_historical_price(1600, price_history) == 20.0
    
    # Midpoint (should return before)
    assert get_historical_price(1500, price_history) == 10.0


def test_get_historical_price_empty():
    assert get_historical_price(1000, {}) is None


def test_format_timestamp():
    # 2023-01-01 00:00:00 UTC is 1672531200000 ms
    ts = 1672531200000
    
    # Default format
    formatted = format_timestamp(ts)
    # The exact result depends on the local timezone of the environment
    # but we can verify it's a string of the right length
    assert len(formatted) == 10
    assert formatted.count('-') == 2
    
    # Custom format
    formatted_custom = format_timestamp(ts, '%Y/%m/%d')
    assert formatted_custom.count('/') == 2
