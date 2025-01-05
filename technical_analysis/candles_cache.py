from typing import Dict, List, Any, Optional, Tuple

# Cache structure: {coin: {timeframe: (last_update_ts, candles)}}
_candles_cache: Dict[str, Dict[str, Tuple[int, List[Dict[str, Any]]]]] = {}

# Timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

def get_cached_candles(coin: str, timeframe: str, start_time: int, end_time: int) -> Optional[List[Dict[str, Any]]]:
    """Get candles from cache if available and not too old"""
    if coin not in _candles_cache or timeframe not in _candles_cache[coin]:
        return None
    
    _, candles = _candles_cache[coin][timeframe]
   
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
    """Update the cache with new candles"""
    if coin not in _candles_cache:
        _candles_cache[coin] = {}
    
    if timeframe in _candles_cache[coin]:
        # Merge with existing candles
        _, existing_candles = _candles_cache[coin][timeframe]
        lookback_days = (current_time - min(c['T'] for c in existing_candles)) // 86400000
        candles = merge_candles(existing_candles, candles, lookback_days)
    
    _candles_cache[coin][timeframe] = (current_time, candles)

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
