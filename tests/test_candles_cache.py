import pytest
from pathlib import Path
import shutil
from technical_analysis.candles_cache import (
    get_candles_with_cache, _round_timestamp, CACHE_DIR, clear_cache
)

@pytest.fixture
def mock_fetch():
    def fetch(coin: str, timeframe: str, start_ts: int, end_ts: int):
        # Generate candles for the exact requested time range
        hours = int((end_ts - start_ts) / 3600000)
        return [
            {'T': _round_timestamp(start_ts + (i * 3600000), timeframe), 'o': 100 + i, 'h': 101 + i, 'l': 99 + i, 'c': 100 + i}
            for i in range(hours)
        ]
    return fetch

@pytest.fixture(autouse=True)
def setup_teardown():
    """Create cache directory before tests and clean it up after"""
    # Setup
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Teardown
    clear_cache()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

def test_get_candles_empty_cache(mock_fetch):
    now = 1000000000000
    end_ts = _round_timestamp(now, '1h')
    start_ts = _round_timestamp(end_ts - 86400000, '1h')
    result = get_candles_with_cache('BTC', '1h', now, 1, mock_fetch)
    assert len(result) == 24
    assert result[0]['T'] == start_ts
    assert result[-1]['T'] == end_ts - 3600000

def test_get_candles_update_cache(mock_fetch):
    now = 1000000000000
    end_ts = _round_timestamp(now, '1h')
    first_result = get_candles_with_cache('BTC', '1h', now, 1, mock_fetch)
    assert len(first_result) == 24
    
    new_now = now + 3600000
    new_end_ts = _round_timestamp(new_now, '1h')
    result = get_candles_with_cache('BTC', '1h', new_now, 1, mock_fetch)
    assert len(result) == 24
    assert result[-1]['T'] == new_end_ts - 3600000

def test_get_candles_cache_hit(mock_fetch):
    now = 1000000000000
    end_ts = _round_timestamp(now, '1h')
    # First call to populate cache
    get_candles_with_cache('BTC', '1h', now, 1, mock_fetch)
    
    # Second call with same timestamp should use cache
    mock_fetch_with_tracking = lambda *args: (mock_fetch(*args), pytest.fail("Should not be called"))[-1]
    result = get_candles_with_cache('BTC', '1h', now, 1, mock_fetch_with_tracking)
    assert len(result) == 24
    assert result[-1]['T'] == end_ts - 3600000
