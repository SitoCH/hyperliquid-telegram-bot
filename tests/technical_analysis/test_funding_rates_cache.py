import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import time

from technical_analysis.funding_rates_cache import (
    FundingRateEntry,
    FundingRateCache,
    get_funding_with_cache,
    merge_funding_rates,
    CACHE_DIR
)

# Test data
MOCK_FUNDING_RATES = [
    {'time': '1700000000000', 'fundingRate': '0.0001', 'premium': '0.0002'},
    {'time': '1700001000000', 'fundingRate': '0.0002', 'premium': '0.0003'},
    {'time': '1700002000000', 'fundingRate': '0.0003', 'premium': '0.0004'}
]

@pytest.fixture
def mock_api():
    with patch('technical_analysis.funding_rates_cache.hyperliquid_utils.info.funding_history') as mock:
        mock.return_value = MOCK_FUNDING_RATES
        yield mock

@pytest.fixture
def clean_cache():
    # Clear cache before and after tests
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob('*funding_rates.json'):
            file.unlink()
    yield
    if CACHE_DIR.exists():
        for file in CACHE_DIR.glob('*funding_rates.json'):
            file.unlink()

def test_get_funding_no_cache(mock_api, clean_cache):
    now = 1700002000000
    lookback_days = 1
    rates = get_funding_with_cache('BTC', now, lookback_days)
    
    assert len(rates) == 3
    assert rates[0].time == 1700000000000
    assert rates[0].funding_rate == 0.0001
    assert rates[0].premium == 0.0002
    mock_api.assert_called_once()

def test_get_funding_with_existing_cache(mock_api, clean_cache):
    # First call to create cache
    now = 1700002000000
    get_funding_with_cache('BTC', now, 1)
    
    # Second call within cache validity period
    new_now = now + 100000  # Small time difference
    rates = get_funding_with_cache('BTC', new_now, 1)
    
    # API should only be called once (for the first call)
    assert mock_api.call_count == 1

def test_merge_funding_rates():
    old_rates = [
        FundingRateEntry(1700000000000, 0.0001, 0.0002),
        FundingRateEntry(1700001000000, 0.0002, 0.0003)
    ]
    new_rates = [
        FundingRateEntry(1700001000000, 0.0002, 0.0003),  # Duplicate
        FundingRateEntry(1700002000000, 0.0003, 0.0004)   # New entry
    ]
    
    merged = merge_funding_rates(old_rates, new_rates, 1)
    assert len(merged) == 3
    assert merged[-1].time == 1700002000000

def test_cache_invalidation(mock_api, clean_cache):
    now = 1700002000000
    # First call
    get_funding_with_cache('BTC', now, 1)
    
    # Call after cache invalidation period (> 5 minutes)
    now += 3600000  # 1 hour later
    mock_api.return_value = [
        {'time': '1700003000000', 'fundingRate': '0.0004', 'premium': '0.0005'}
    ]
    
    rates2 = get_funding_with_cache('BTC', now, 1)
    assert mock_api.call_count == 2
    assert len(rates2) > 0
    assert max(r.time for r in rates2) == 1700003000000

def test_invalid_cache_handling(clean_cache):
    # Create invalid cache file
    cache_file = CACHE_DIR / 'BTC_funding_rates.json'
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        f.write('invalid json')
    
    with patch('technical_analysis.funding_rates_cache.hyperliquid_utils.info.funding_history') as mock:
        mock.return_value = MOCK_FUNDING_RATES
        rates = get_funding_with_cache('BTC', 1700002000000, 1)
        
    assert len(rates) == 3

def test_lookback_period_filtering():
    now = 1700002000000
    day_in_ms = 86400000
    old_rates = [
        FundingRateEntry(now - 3 * day_in_ms, 0.0001, 0.0002),
        FundingRateEntry(now - 2 * day_in_ms, 0.0002, 0.0003),
        FundingRateEntry(now - day_in_ms, 0.0003, 0.0004),
        FundingRateEntry(now, 0.0004, 0.0005)
    ]
    
    # Test with 2 days lookback
    merged = merge_funding_rates(old_rates, [], 2)
    assert len(merged) == 3  # Should only include last 2 days of data
    assert min(r.time for r in merged) >= now - 2 * day_in_ms
