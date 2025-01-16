import pytest
from pathlib import Path
import shutil
from technical_analysis.candles_cache import (
    get_candles_with_cache, _round_timestamp, CACHE_DIR, clear_cache
)
from technical_analysis.wyckoff_types import Timeframe

@pytest.fixture
def mock_fetch():
    def fetch(coin: str, timeframe_str: str, start_ts: int, end_ts: int):
        # Find the timeframe enum from timeframe_str
        timeframe = next(tf for tf in Timeframe if tf.name == timeframe_str)
        
        # Calculate number of intervals based on timeframe minutes
        interval_ms = timeframe.minutes * 60 * 1000
        intervals = int((end_ts - start_ts) / interval_ms)
        
        return [
            {
                'T': _round_timestamp(start_ts + (i * interval_ms), timeframe),
                'o': 100 + i,
                'h': 101 + i,
                'l': 99 + i,
                'c': 100 + i
            }
            for i in range(intervals)
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
    end_ts = _round_timestamp(now, Timeframe.HOUR_1)
    start_ts = _round_timestamp(end_ts - 86400000, Timeframe.HOUR_1)
    result = get_candles_with_cache('BTC', Timeframe.HOUR_1, now, 1, mock_fetch)
    assert len(result) == 24
    assert result[0]['T'] == start_ts
    assert result[-1]['T'] == end_ts - 3600000

def test_get_candles_update_cache(mock_fetch):
    now = 1000000000000
    end_ts = _round_timestamp(now, Timeframe.HOUR_1)
    first_result = get_candles_with_cache('BTC', Timeframe.HOUR_1, now, 1, mock_fetch)
    assert len(first_result) == 24
    
    new_now = now + 3600000
    new_end_ts = _round_timestamp(new_now, Timeframe.HOUR_1)
    result = get_candles_with_cache('BTC', Timeframe.HOUR_1, new_now, 1, mock_fetch)
    assert len(result) == 24
    assert result[-1]['T'] == new_end_ts - 3600000

def test_get_candles_cache_hit(mock_fetch):
    now = 1000000000000
    first_result = get_candles_with_cache('BTC', Timeframe.HOUR_1, now, 1, mock_fetch)
    
    # Second call just 30 seconds later
    slightly_later = now + 30000  # 30 seconds later
    
    # Mock get_cached_candles to return data with a very recent candle
    def mock_get_cached_candles(*args):
        candles = first_result.copy()
        # Add a very recent candle
        candles[-1] = {**candles[-1], 'T': slightly_later - 30000}  # Candle from 30s ago
        return candles
    
    import technical_analysis.candles_cache as cc
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cc, 'get_cached_candles', mock_get_cached_candles)
    
    mock_fetch_with_tracking = lambda *args: (mock_fetch(*args), pytest.fail("Should not be called"))[-1]
    result = get_candles_with_cache('BTC', Timeframe.HOUR_1, slightly_later, 1, mock_fetch_with_tracking)
    
    assert len(result) == len(first_result)
    assert result[-1]['T'] >= slightly_later - 60000  # Last candle should be within 1 minute
