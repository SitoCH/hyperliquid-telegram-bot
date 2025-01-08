import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Callable

# Calculate cache directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'cache' / 'funding_rates'

class FundingRateEntry(TypedDict):
    time: int
    fundingRate: float

def _get_funding_cache_file_path(coin: str) -> Path:
    """Get the path for the funding rate cache file of a specific coin"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{coin}_funding_rates.json"

def _load_funding_from_disk(coin: str) -> Optional[Tuple[int, List[FundingRateEntry]]]:
    """Load funding rate data from disk"""
    cache_file = _get_funding_cache_file_path(coin)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted cache files
            cache_file.unlink()
    return None


def _save_funding_to_disk(coin: str, data: Tuple[int, List[FundingRateEntry]]) -> None:
    """Save funding rate data to disk"""
    cache_file = _get_funding_cache_file_path(coin)
    with open(cache_file, 'w') as f:
        json.dump(data, f)

def _convert_funding_rate(rate: Dict[str, Any]) -> FundingRateEntry:
    """Convert raw funding rate data to FundingRateEntry with consistent types"""
    return {
        'time': int(str(rate['time'])),  # Handle both int and str inputs
        'fundingRate': float(str(rate['fundingRate']))  # Handle both float and str inputs
    }

def get_funding_with_cache(coin: str, now: int, lookback_days: int, fetch_fn) -> List[FundingRateEntry]:
    """Get funding rates using cache, fetching only newer data if needed"""
    end_ts = now
    start_ts = end_ts - lookback_days * 86400000

    # Try disk cache
    cached_data = _load_funding_from_disk(coin)
    if cached_data:
        last_update, funding_rates = cached_data
        
        # Filter cached rates within requested time range
        cached_rates = [r for r in funding_rates if start_ts <= r['time'] <= end_ts]
        
        # Check if cache is recent enough
        if cached_rates and last_update >= end_ts - 3600000 / 4:  # 15 minues in milliseconds
            return cached_rates
            
        # Only fetch missing data
        if cached_rates:
            fetch_start = max(r['time'] for r in cached_rates) + 1
            new_rates =[_convert_funding_rate(rate) for rate in fetch_fn(coin, fetch_start, end_ts)]
            merged = merge_funding_rates(cached_rates, new_rates, lookback_days)
            _save_funding_to_disk(coin, (now, merged))
            return merged

    # No cache or outdated, fetch all data
    funding_rates = [_convert_funding_rate(rate) for rate in fetch_fn(coin, start_ts, end_ts)]
    _save_funding_to_disk(coin, (now, funding_rates))
    return funding_rates

def merge_funding_rates(
    old_rates: List[FundingRateEntry], 
    new_rates: List[FundingRateEntry], 
    lookback_days: int
) -> List[FundingRateEntry]:
    """Merge old and new funding rates, removing duplicates and keeping only within lookback period"""
    # Use timestamp as key to avoid duplicates
    merged = {r['time']: r for r in old_rates}
    merged.update({r['time']: r for r in new_rates})
    
    # Sort by timestamp and filter by lookback period
    sorted_rates = sorted(merged.values(), key=lambda x: x['time'])
    
    if not sorted_rates:
        return []
        
    latest_ts = max(r['time'] for r in sorted_rates)
    min_ts = latest_ts - (lookback_days * 86400000)
    
    return [r for r in sorted_rates if r['time'] >= min_ts]
