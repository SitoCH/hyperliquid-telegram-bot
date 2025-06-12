import os
import json
import asyncio
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from utils import log_execution_time
from .wyckoff.wyckoff_types import Timeframe
from logging_utils import logger

# Calculate cache directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'cache' / 'candles'

def _get_cache_file_path(coin: str, timeframe: Timeframe) -> Path:
    """Get the path for the cache file of a specific coin and timeframe"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{coin}_{timeframe}_candles.json"

def _load_from_disk(coin: str, timeframe: Timeframe) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """Load candles data from disk"""
    cache_file = _get_cache_file_path(coin, timeframe)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return tuple(json.load(f))
        except ValueError as e:
            logger.error(f"Failed to load cache for {coin} {timeframe.name}: {e}")
            cache_file.unlink()
    return None

def _save_to_disk(coin: str, timeframe: Timeframe, data: Tuple[int, List[Dict[str, Any]]]) -> None:
    """Save candles data to disk"""
    cache_file = _get_cache_file_path(coin, timeframe)
    with open(cache_file, 'w') as f:
        json.dump(list(data), f)

def get_cached_candles(coin: str, timeframe: Timeframe, start_time: int, end_time: int) -> Optional[List[Dict[str, Any]]]:
    """Get candles from cache if available and not too old"""
    # Try disk cache
    cached_data = _load_from_disk(coin, timeframe)
    if cached_data is None:
        return None
    
    _, candles = cached_data
    # Filter candles within requested time range
    return [c for c in candles if start_time <= c['T'] <= end_time]

def trim_candles(candles: List[Dict[str, Any]], lookback_days: int) -> List[Dict[str, Any]]:
    """Keep only the candles within lookback period"""
    if not candles:
        return candles
    latest_ts = max(c['T'] for c in candles)
    min_ts = latest_ts - (lookback_days * 86400000)
    return [c for c in candles if c['T'] >= min_ts]

def merge_candles(old_candles: List[Dict[str, Any]], new_candles: List[Dict[str, Any]], lookback_days: int) -> List[Dict[str, Any]]:
    """Merge old and new candles, always replacing old with new if timestamp matches"""
    merged = {c['T']: c for c in old_candles}
    for candle in new_candles:
        merged[candle['T']] = candle  # Always replace with new candle
    sorted_candles = sorted(merged.values(), key=lambda x: x['T'])
    if lookback_days > 0:
        sorted_candles = trim_candles(sorted_candles, lookback_days)
    return sorted_candles

def verify_candles(coin: str, candles: List[Dict[str, Any]], timeframe: Timeframe, is_initial_load: bool = False) -> Tuple[bool, str]:
    """
    Verify the integrity of candles data.
    Returns (is_valid, error_message)
    """
    def _clear_all_timeframe_caches(coin: str):
        """Clear cache for all timeframes of a given coin"""
        for tf in Timeframe:
            cache_file = _get_cache_file_path(coin, tf)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    logger.warning(f"Cleared cache for {coin} {tf.name} due to data integrity issue")
                except Exception as e:
                    logger.error(f"Failed to clear cache for {coin} {tf.name}: {e}")

    if not candles:
        return True, ""
    
    # Sort candles by timestamp to ensure chronological order
    sorted_candles = sorted(candles, key=lambda x: x['T'])
    
    # Check required fields
    required_fields = {'T', 'o', 'h', 'l', 'c', 'v'}
    for candle in sorted_candles:
        missing_fields = required_fields - set(candle.keys())
        if missing_fields:
            _clear_all_timeframe_caches(coin)
            return False, f"Missing required fields: {missing_fields}"
    
    # Skip interval checks for initial data load
    if not is_initial_load:
        # Check timeframe intervals
        interval_ms = timeframe.minutes * 60 * 1000
        for i in range(1, len(sorted_candles)):
            time_diff = sorted_candles[i]['T'] - sorted_candles[i-1]['T']
            if time_diff != interval_ms:
                time_diff_minutes = time_diff / (60 * 1000)
                _clear_all_timeframe_caches(coin)
                return False, f"Invalid interval at index {i}: expected {timeframe.minutes}min, got {time_diff_minutes}min"
    
    # Check for valid numerical values
    for candle in sorted_candles:
        try:
            o = float(candle['o'])
            h = float(candle['h'])
            l = float(candle['l'])
            c = float(candle['c'])
            # Check for NaN
            if any(math.isnan(x) for x in [o, h, l, c]):
                _clear_all_timeframe_caches(coin)
                return False, "Invalid numerical values (NaN) found"
        except (ValueError, TypeError):
            _clear_all_timeframe_caches(coin)
            return False, "Non-numeric values found in OHLCV data"
        # Verify OHLC relationships
        if not (l <= o <= h and l <= c <= h):
            _clear_all_timeframe_caches(coin)
            return False, f"Invalid OHLC relationships at timestamp {candle['T']}"
    
    return True, ""

def update_cache(coin: str, timeframe: Timeframe, candles: List[Dict[str, Any]], current_time: int) -> None:
    """Update disk cache with new candles"""
    # Don't update cache if no new candles
    if not candles:
        return

    cached_data = _load_from_disk(coin, timeframe)
    is_initial_load = cached_data is None

    if cached_data is not None:
        _, existing_candles = cached_data
        # Only merge if we have existing candles with data
        if existing_candles and len(existing_candles) > 0:
            min_timestamp = min(c['T'] for c in existing_candles)
            lookback_days = (current_time - min_timestamp) // 86400000
            candles = merge_candles(existing_candles, candles, lookback_days)
    
    # Verify merged data
    is_valid, error_msg = verify_candles(coin, candles, timeframe, is_initial_load)
    if not is_valid:
        # Remove corrupt cache before raising error
        cache_file = _get_cache_file_path(coin, timeframe)
        if cache_file.exists():
            cache_file.unlink()
        raise ValueError(f"Invalid merged candles data for {coin}: {error_msg}")
    
    _save_to_disk(coin, timeframe, (current_time, candles))

def clear_cache() -> None:
    """Clear disk cache"""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*_candles.json'):
            cache_file.unlink()

def _round_timestamp(ts: int, timeframe: Timeframe) -> int:
    """Round timestamp down to nearest interval based on timeframe"""
    ms_interval = timeframe.minutes * 60 * 1000
    return ts - (ts % ms_interval)

async def get_candles_with_cache(coin: str, timeframe: Timeframe, now: int, lookback_days: int, fetch_fn) -> List[Dict[str, Any]]:
    """
    Get candles using cache, handling the incomplete current candle appropriately.
    
    Args:
        coin: Trading pair symbol
        timeframe: Timeframe enum value
        now: Current timestamp in milliseconds
        lookback_days: Number of days to look back
        fetch_fn: Sync function to fetch candles from the exchange (will be run in thread pool)
    """
    try:
        end_ts = now
        start_ts = _round_timestamp(end_ts - lookback_days * 86400000, timeframe)
        current_candle_start = _round_timestamp(now, timeframe)
        
        cached = get_cached_candles(coin, timeframe, start_ts, end_ts)
        if cached and len(cached) > 0:
            # Find the timestamp of the last cached candle
            last_cached_ts = max(c['T'] for c in cached)

            # Run sync fetch_fn in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            new_candles = await loop.run_in_executor(None, fetch_fn, coin, timeframe.name, last_cached_ts, end_ts)
            # Merge all candles
            merged = merge_candles(cached, new_candles, lookback_days)
            
            # Filter out incomplete candles for both cache and return
            complete_candles = [c for c in merged if c['T'] < current_candle_start]
            if complete_candles:
                update_cache(coin, timeframe, complete_candles, now)
            
            return complete_candles

        # Run sync fetch_fn in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        candles = await loop.run_in_executor(None, fetch_fn, coin, timeframe.name, start_ts, end_ts)
        if candles:
            # Filter out incomplete candles for both cache and return
            complete_candles = [c for c in candles if c['T'] < current_candle_start]
            if complete_candles:
                update_cache(coin, timeframe, complete_candles, now)
            return complete_candles

        return candles
        
    except Exception as e:
        logger.error(f"Error in get_candles_with_cache for {coin}: {str(e)}")
        raise