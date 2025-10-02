import pytest
from pathlib import Path
import shutil
from technical_analysis.candles_cache import (
    get_candles_with_cache, _round_timestamp, CACHE_DIR, clear_cache
)
from technical_analysis.wyckoff.wyckoff_types import Timeframe

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

@pytest.mark.asyncio
async def test_get_candles_empty_cache(mock_fetch):
    now = 1000000000000
    end_ts = _round_timestamp(now, Timeframe.HOUR_1)
    start_ts = _round_timestamp(end_ts - 86400000, Timeframe.HOUR_1)
    result = await get_candles_with_cache('BTC', Timeframe.HOUR_1, now, 1, mock_fetch)
    assert len(result) == 24
    assert result[0]['T'] == start_ts
    assert result[-1]['T'] == end_ts - 3600000

@pytest.mark.asyncio
async def test_get_candles_update_cache(mock_fetch):
    now = 1000000000000
    first_result = await get_candles_with_cache('BTC', Timeframe.HOUR_1, now, 1, mock_fetch)
    assert len(first_result) == 24
    
    new_now = now + 3600000
    new_end_ts = _round_timestamp(new_now, Timeframe.HOUR_1)
    result = await get_candles_with_cache('BTC', Timeframe.HOUR_1, new_now, 1, mock_fetch)
    assert len(result) == 24
    assert result[-1]['T'] == new_end_ts - 3600000


@pytest.mark.asyncio
async def test_partial_candle_updates(mock_fetch):
    """Test that partial candles are properly updated when fetching multiple times"""
    base_ts = 1000000000000
    timeframe = Timeframe.HOUR_1
    
    # First fetch using standard mock
    now = _round_timestamp(base_ts, timeframe)
    first_result = await get_candles_with_cache('BTC', timeframe, now, 1, mock_fetch)
    
    # Create a modified mock that returns different values
    def updated_mock(coin: str, timeframe_str: str, start_ts: int, end_ts: int):
        candles = mock_fetch(coin, timeframe_str, start_ts, end_ts)
        # Modify all candles to have different values
        for candle in candles:
            candle.update({
                'h': 999.99,
                'c': 998.88,
                'v': 9999  # Increased volume to ensure update
            })
        return candles
    
    # Second fetch with modified data
    now += 65000  # 65 seconds later
    second_result = await get_candles_with_cache('BTC', timeframe, now, 1, updated_mock)
    
    assert abs(first_result[-1]['h'] - second_result[-1]['h']) > 0.01, "Last candle should be updated"
    assert abs(second_result[-1]['h'] - 999.99) < 0.01, "Last candle should have updated high value"
    assert abs(second_result[-1]['c'] - 998.88) < 0.01, "Last candle should have updated close value"


@pytest.mark.asyncio
async def test_include_incomplete_candles(mock_fetch):
    timeframe = Timeframe.HOUR_1
    interval_ms = timeframe.minutes * 60 * 1000
    now = 1000000000000 + interval_ms // 2  # Halfway through the current candle

    def fetch_with_incomplete(coin: str, timeframe_str: str, start_ts: int, end_ts: int):
        candles = mock_fetch(coin, timeframe_str, start_ts, end_ts)
        timeframe = next(tf for tf in Timeframe if tf.name == timeframe_str)
        current_candle_start = _round_timestamp(end_ts, timeframe)
        candles.append({
            'T': current_candle_start,
            'o': 200,
            'h': 210,
            'l': 190,
            'c': 205,
            'v': 1500
        })
        return candles

    # Default behaviour excludes incomplete candles
    complete_only = await get_candles_with_cache('BTC', timeframe, now, 1, fetch_with_incomplete)
    current_candle_start = _round_timestamp(now, timeframe)
    assert all(c['T'] < current_candle_start for c in complete_only)

    # Optional flag returns the incomplete candle without caching it
    with_incomplete = await get_candles_with_cache('BTC', timeframe, now, 1, fetch_with_incomplete, include_incomplete=True)
    assert len(with_incomplete) == len(complete_only) + 1
    assert with_incomplete[-1]['T'] == current_candle_start

    # Cache should still only contain complete candles
    final_complete = await get_candles_with_cache('BTC', timeframe, now, 1, fetch_with_incomplete)
    assert all(c['T'] < current_candle_start for c in final_complete)
