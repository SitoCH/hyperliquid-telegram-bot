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
                'c': 100 + i,
                'v': 1000 + i
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


def test_partial_candle_updates(mock_fetch):
    """Test that partial candles are properly updated when fetching multiple times"""
    base_ts = 1000000000000
    timeframe = Timeframe.HOUR_1
    
    # Create a mock fetch that returns different values for the current candle
    def fetch_with_updates(coin: str, timeframe_str: str, start_ts: int, end_ts: int):
        candles = mock_fetch(coin, timeframe_str, start_ts, end_ts)
        if candles:
            candles[-1].update({
                'h': 999.99,  # Significantly different value to verify update
                'c': 998.88
            })
        return candles
    
    # First fetch at exactly the start of a candle
    now = _round_timestamp(base_ts, timeframe)
    first_result = get_candles_with_cache('BTC', timeframe, now, 1, mock_fetch)
    print(first_result)
    
    # Second fetch after the 60-second cache window
    now += 65000  # 65 seconds later (just over the cache window)
    second_result = get_candles_with_cache('BTC', timeframe, now, 1, fetch_with_updates)
    print()
    print(second_result)
    
    assert first_result[-1]['h'] != second_result[-1]['h'], "Last candle should be updated"
    assert second_result[-1]['h'] == 999.99, "Last candle should have updated high value"
    assert second_result[-1]['c'] == 998.88, "Last candle should have updated close value"
