import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from utils import log_execution_time

# Calculate cache directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'cache' / 'candles'

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

def _get_cache_file_path(coin: str, timeframe: str) -> Path:
    """Get the path for the cache file of a specific coin and timeframe"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{coin}_{timeframe}_candles.json"

def _load_from_disk(coin: str, timeframe: str) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """Load candles data from disk"""
    cache_file = _get_cache_file_path(coin, timeframe)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return tuple(json.load(f))
        except (ValueError):
            # Handle corrupted cache files
            cache_file.unlink()
    return None

def _save_to_disk(coin: str, timeframe: str, data: Tuple[int, List[Dict[str, Any]]]) -> None:
    """Save candles data to disk"""
    cache_file = _get_cache_file_path(coin, timeframe)
    with open(cache_file, 'w') as f:
        json.dump(list(data), f)

def get_cached_candles(coin: str, timeframe: str, start_time: int, end_time: int) -> Optional[List[Dict[str, Any]]]:
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
    """Merge old and new candles, removing duplicates and keeping only within lookback period"""
    merged = {c['T']: c for c in old_candles}
    merged.update({c['T']: c for c in new_candles})
    sorted_candles = sorted(merged.values(), key=lambda x: x['T'])
    return trim_candles(sorted_candles, lookback_days)

def update_cache(coin: str, timeframe: str, candles: List[Dict[str, Any]], current_time: int) -> None:
    """Update disk cache with new candles"""
    cached_data = _load_from_disk(coin, timeframe)
    if cached_data is not None:
        # Merge with existing candles
        _, existing_candles = cached_data
        lookback_days = (current_time - min(c['T'] for c in existing_candles)) // 86400000
        candles = merge_candles(existing_candles, candles, lookback_days)
    
    _save_to_disk(coin, timeframe, (current_time, candles))

def clear_cache() -> None:
    """Clear disk cache"""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob('*_candles.json'):
            cache_file.unlink()

def _round_timestamp(ts: int, timeframe: str) -> int:
    """Round timestamp down to nearest interval based on timeframe"""
    minutes = TIMEFRAME_MINUTES[timeframe]
    ms_interval = minutes * 60 * 1000
    return ts - (ts % ms_interval)


def get_candles_with_cache(coin: str, timeframe: str, now: int, lookback_days: int, fetch_fn) -> List[Dict[str, Any]]:
    """Get candles using cache, fetching only newer data if needed"""
    end_ts = _round_timestamp(now, timeframe)
    start_ts = _round_timestamp(end_ts - lookback_days * 86400000, timeframe)
    
    cached = get_cached_candles(coin, timeframe, start_ts, end_ts)
    if cached:
        # If we have cached data, check if it's up to date
        last_cached_ts = max(c['T'] for c in cached)
        interval_ms = TIMEFRAME_MINUTES[timeframe] * 60 * 1000
        if last_cached_ts >= end_ts - interval_ms:  # One interval behind is acceptable
            return cached
            
        # Only fetch missing candles
        fetch_start = _round_timestamp(last_cached_ts + 1, timeframe)
        new_candles = fetch_fn(coin, timeframe, fetch_start, end_ts)
        merged = merge_candles(cached, new_candles, lookback_days)
        update_cache(coin, timeframe, merged, now)
        return merged
    else:
        # No cache, fetch all candles
        candles = fetch_fn(coin, timeframe, start_ts, end_ts)
        update_cache(coin, timeframe, candles, now)
        return candles
