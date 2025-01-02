import pandas as pd
import pytest
from datetime import datetime, timezone
from technical_analysis.significant_levels import find_significant_levels

def test_find_significant_levels_with_single_data_point():
    # Create a DataFrame with just one data point
    single_data = {
        'Open': [100.0],
        'High': [105.0],
        'Low': [95.0],
        'Close': [102.0],
        'Volume': [1000.0],
        'ATR': [2.0]
    }
    
    # Create DataFrame with single timestamp index
    df = pd.DataFrame(single_data, 
                     index=[pd.Timestamp('2025-01-02 21:45:58', tz=timezone.utc)])
    
    # Test the function
    resistance_levels, support_levels = find_significant_levels(df)
    
    # Assert that we get empty lists for both levels when there's insufficient data
    assert len(resistance_levels) == 0
    assert len(support_levels) == 0

def test_find_significant_levels_with_multiple_data_points():
    # Create a DataFrame with multiple data points
    multi_data = {
        'Open': [100.0, 102.0],
        'High': [105.0, 107.0],
        'Low': [95.0, 97.0],
        'Close': [102.0, 104.0],
        'Volume': [1000.0, 1100.0],
        'ATR': [2.0, 2.1]
    }
    
    # Create DataFrame with multiple timestamp indices
    df = pd.DataFrame(multi_data, 
                     index=[
                         pd.Timestamp('2025-01-02 21:45:58', tz=timezone.utc),
                         pd.Timestamp('2025-01-02 22:45:58', tz=timezone.utc)
                     ])
    
    # Test the function
    resistance_levels, support_levels = find_significant_levels(df)
    
    # Assert that we get some levels when there's sufficient data
    assert isinstance(resistance_levels, list)
    assert isinstance(support_levels, list)