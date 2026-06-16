import pytest
from bot_statistics.btc_price_utils import calculate_btc_hold_performance
from bot_statistics.sp500_price_utils import calculate_sp500_hold_performance


def test_calculate_btc_hold_performance():
    price_history = {
        1000: 20000.0,
        2000: 21000.0
    }
    
    # Positive performance
    result = calculate_btc_hold_performance(1000, 22000.0, price_history)
    assert result is not None
    assert result['starting_price'] == 20000.0
    assert result['current_price'] == 22000.0
    assert result['pct_change'] == 10.0
    
    # Negative performance
    result = calculate_btc_hold_performance(2000, 18900.0, price_history)
    assert result is not None
    assert result['starting_price'] == 21000.0
    assert result['current_price'] == 18900.0
    assert result['pct_change'] == -10.0
    
    # Missing current price
    assert calculate_btc_hold_performance(1000, None, price_history) is None
    
    # Empty history
    assert calculate_btc_hold_performance(1000, 22000.0, {}) is None


def test_calculate_sp500_hold_performance():
    price_history = {
        1000: 4000.0,
        2000: 4100.0
    }
    
    # Positive performance
    result = calculate_sp500_hold_performance(1000, 4400.0, price_history)
    assert result is not None
    assert result['starting_price'] == 4000.0
    assert result['current_price'] == 4400.0
    assert result['pct_change'] == 10.0
    
    # Missing current price
    assert calculate_sp500_hold_performance(1000, None, price_history) is None
    
    # Empty history
    assert calculate_sp500_hold_performance(1000, 4400.0, {}) is None
