from typing import Dict, List, Any, Optional, Tuple

# Cache structure: {coin: {timeframe: (last_update_ts, candles)}}
_candles_cache: Dict[str, Dict[str, Tuple[int, List[Dict[str, Any]]]]] = {}


def get_cached_candles(coin: str, timeframe: str, start_time: int, end_time: int) -> Optional[List[Dict[str, Any]]]:
    """Get candles from cache if available and not too old"""
    if coin not in _candles_cache or timeframe not in _candles_cache[coin]:
        return None
    
    last_update, candles = _candles_cache[coin][timeframe]
    if not candles or last_update < end_time - 300000:  # Cache expires after 5 minutes
        return None
    
    # Filter candles within requested time range
    return [c for c in candles if start_time <= c['T'] <= end_time]


def merge_candles(old_candles: List[Dict[str, Any]], new_candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge old and new candles, removing duplicates and sorting by timestamp"""
    # Create a dictionary of candles keyed by timestamp
    merged = {c['T']: c for c in old_candles}
    # Update with new candles
    merged.update({c['T']: c for c in new_candles})
    # Convert back to list and sort by timestamp
    return sorted(merged.values(), key=lambda x: x['T'])


def update_cache(coin: str, timeframe: str, candles: List[Dict[str, Any]], current_time: int) -> None:
    """Update the cache with new candles"""
    if coin not in _candles_cache:
        _candles_cache[coin] = {}
    
    if timeframe in _candles_cache[coin]:
        # Merge with existing candles
        _, existing_candles = _candles_cache[coin][timeframe]
        candles = merge_candles(existing_candles, candles)
    
    _candles_cache[coin][timeframe] = (current_time, candles)


def get_candles_with_cache(coin: str, timeframe: str, now: int, lookback_days: int, fetch_fn) -> List[Dict[str, Any]]:
    """Get candles using cache, fetching only newer data if needed"""
    start_ts = now - lookback_days * 86400000
    cached = get_cached_candles(coin, timeframe, start_ts, now)
    if cached:
        # If we have cached data, only fetch newer candles
        start_ts = max(c['T'] for c in cached) + 1
        if start_ts >= now:
            return cached
        new_candles = fetch_fn(coin, timeframe, start_ts, now)
        merged = merge_candles(cached, new_candles)
        update_cache(coin, timeframe, merged, now)
        return merged
    else:
        # No cache, fetch all candles
        candles = fetch_fn(coin, timeframe, start_ts, now)
        update_cache(coin, timeframe, candles, now)
        return candles
