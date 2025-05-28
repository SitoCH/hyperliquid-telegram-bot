import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from dacite import from_dict # type: ignore
from hyperliquid_utils.utils import hyperliquid_utils
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Callable
from utils import log_execution_time

# Calculate cache directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / 'cache' / 'funding_rates'


@dataclass
class FundingRateEntry:
    time: int
    funding_rate: float
    premium: float


@dataclass
class FundingRateCache:
    last_update: int
    rates: List[FundingRateEntry]



def _get_funding_cache_file_path(coin: str) -> Path:
    """Get the path for the funding rate cache file of a specific coin"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{coin}_funding_rates.json"

def _load_funding_from_disk(coin: str) -> Optional[FundingRateCache]:
    """Load funding rate data from disk"""
    cache_file = _get_funding_cache_file_path(coin)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return from_dict(data_class=FundingRateCache, data=data)
        except (json.JSONDecodeError, FileNotFoundError):
            # Handle corrupted cache files
            cache_file.unlink()
    return None


def _save_funding_to_disk(coin: str, rates_cache: FundingRateCache) -> None:
    """Save funding rate data to disk"""
    cache_file = _get_funding_cache_file_path(coin)
    with open(cache_file, 'w') as f:
        json.dump(asdict(rates_cache), f)


def _convert_funding_rate(rate: Dict[str, Any]) -> FundingRateEntry:
    """Convert raw funding rate data to FundingRateEntry with consistent types"""
    return FundingRateEntry(
        int(str(rate['time'])),
        float(str(rate['fundingRate'])),
        float(str(rate['premium']))
    )

def _fetch_new_funding_rates(coin: str, start_ts: int, end_ts: int) -> List[FundingRateEntry]:
    """Fetch new funding rates from HyperLiquid API"""
    try:
        funding_rates = hyperliquid_utils.info.funding_history(coin, start_ts, end_ts)
        return [_convert_funding_rate(rate) for rate in funding_rates]
    except KeyError:
        return []


def get_funding_with_cache(coin: str, now: int, lookback_days: int) -> List[FundingRateEntry]:
    """Get funding rates using cache, fetching only newer data if needed"""
    end_ts = now
    start_ts = end_ts - lookback_days * 86400000

    # Try disk cache
    funding_rate_cache = _load_funding_from_disk(coin)
    if funding_rate_cache:
        
        # Filter cached rates within requested time range
        cached_rates = [r for r in funding_rate_cache.rates if start_ts <= r.time <= end_ts]
        
        # Check if cache is recent enough
        if cached_rates and funding_rate_cache.last_update >= end_ts - 3600000 / 12:  # 5 minutes in milliseconds
            return cached_rates
            
        # Only fetch missing data
        if cached_rates:
            fetch_start = max(r.time for r in cached_rates) + 1
            new_rates = _fetch_new_funding_rates(coin, fetch_start, end_ts)
            if new_rates:
                merged = merge_funding_rates(cached_rates, new_rates, lookback_days)
                _save_funding_to_disk(coin, FundingRateCache(now, merged))
                return merged
            return cached_rates

    # No cache or outdated, fetch all data
    funding_rates = _fetch_new_funding_rates(coin, start_ts, end_ts)
    if funding_rates:
        _save_funding_to_disk(coin, FundingRateCache(now, funding_rates))
    return funding_rates

def merge_funding_rates(
    old_rates: List[FundingRateEntry], 
    new_rates: List[FundingRateEntry], 
    lookback_days: int
) -> List[FundingRateEntry]:
    """Merge old and new funding rates, removing duplicates and keeping only within lookback period"""
    # Use timestamp as key to avoid duplicates
    merged = {r.time: r for r in old_rates}
    merged.update({r.time: r for r in new_rates})
    
    # Sort by timestamp and filter by lookback period
    sorted_rates = sorted(merged.values(), key=lambda x: x.time)
    
    if not sorted_rates:
        return []
        
    latest_ts = max(r.time for r in sorted_rates)
    min_ts = latest_ts - (lookback_days * 86400000)
    
    return [r for r in sorted_rates if r.time >= min_ts]


def analyze_funding_rate_patterns(funding_rates: List[FundingRateEntry]) -> Dict[str, Any]:
    """Analyze funding rate patterns and provide thresholds and insights"""
    if not funding_rates:
        return {}
    
    rates = [r.funding_rate for r in funding_rates]
    
    # Calculate statistics
    current_rate = rates[-1] if rates else 0.0
    avg_24h = sum(rates[-24:]) / len(rates[-24:]) if len(rates) >= 24 else current_rate
    avg_7d = sum(rates[-168:]) / len(rates[-168:]) if len(rates) >= 168 else current_rate
    
    # Define funding rate thresholds
    thresholds = {
        'extremely_bullish': 0.001,      # 0.1% (very high positive funding)
        'very_bullish': 0.0005,          # 0.05%
        'bullish': 0.0002,               # 0.02%
        'neutral_high': 0.0001,          # 0.01%
        'neutral_low': -0.0001,          # -0.01%
        'bearish': -0.0002,              # -0.02%
        'very_bearish': -0.0005,         # -0.05%
        'extremely_bearish': -0.001      # -0.1% (very high negative funding)
    }
    
    # Determine current sentiment
    sentiment = 'neutral'
    if current_rate >= thresholds['extremely_bullish']:
        sentiment = 'extremely_bullish'
    elif current_rate >= thresholds['very_bullish']:
        sentiment = 'very_bullish'
    elif current_rate >= thresholds['bullish']:
        sentiment = 'bullish'
    elif current_rate >= thresholds['neutral_high']:
        sentiment = 'neutral_bullish'
    elif current_rate <= thresholds['extremely_bearish']:
        sentiment = 'extremely_bearish'
    elif current_rate <= thresholds['very_bearish']:
        sentiment = 'very_bearish'
    elif current_rate <= thresholds['bearish']:
        sentiment = 'bearish'
    elif current_rate <= thresholds['neutral_low']:
        sentiment = 'neutral_bearish'
    
    # Calculate trend
    trend = 'stable'
    if len(rates) >= 8:
        recent_avg = sum(rates[-8:]) / 8
        older_avg = sum(rates[-16:-8]) / 8 if len(rates) >= 16 else recent_avg
        if recent_avg > older_avg * 1.5:
            trend = 'increasing'
        elif recent_avg < older_avg * 0.5:
            trend = 'decreasing'
    
    # Calculate volatility
    if len(rates) >= 24:
        rate_changes = [abs(rates[i] - rates[i-1]) for i in range(1, min(25, len(rates)))]
        volatility = sum(rate_changes) / len(rate_changes)
    else:
        volatility = 0.0
    
    return {
        'current_rate': current_rate,
        'avg_24h': avg_24h,
        'avg_7d': avg_7d,
        'sentiment': sentiment,
        'trend': trend,
        'volatility': volatility,
        'thresholds': thresholds,
        'extremes': {
            'is_extreme': abs(current_rate) >= thresholds['very_bullish'],
            'direction': 'bullish' if current_rate > 0 else 'bearish' if current_rate < 0 else 'neutral',
            'magnitude': abs(current_rate)
        },
        'mean_reversion_signal': {
            'likely': abs(current_rate) > abs(avg_7d) * 2,
            'direction': 'down' if current_rate > avg_7d * 2 else 'up' if current_rate < avg_7d * 2 else 'none'
        }
    }
