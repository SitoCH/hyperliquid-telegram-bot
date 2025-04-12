import requests
import os
from datetime import datetime
from logging_utils import logger
from bot_statistics.btc_price_utils import get_btc_historical_price

def get_sp500_price_history(start_timestamp):
    """
    Fetch S&P500 price history for a given time range using Alpha Vantage API.
    
    Args:
        start_timestamp: Start timestamp in milliseconds
        
    Returns:
        Dictionary with timestamps as keys and prices as values
    """
    try:
        start_date = datetime.fromtimestamp(start_timestamp / 1000).strftime('%Y-%m-%d')
        
        api_key = os.environ.get("HTB_ALPHAVANTAGE_API_KEY")
        if not api_key:
            logger.warn("Alpha Vantage API key not found in environment variables")
            return {}

        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": "SPY",
                "outputsize": "full",
                "apikey": api_key,
                "startdate": start_date
            }
        )
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            logger.error(f"Invalid response from Alpha Vantage: {data}")
            return {}
            
        daily_data = data["Time Series (Daily)"]
        
        price_history = {}
        
        for date_str, values in daily_data.items():
            date_timestamp = int(datetime.strptime(date_str, '%Y-%m-%d').timestamp() * 1000)
            if date_timestamp >= start_timestamp:
                price_history[date_timestamp] = float(values["4. close"])
                
        return price_history
    
    except Exception as e:
        logger.error(f"Error fetching S&P500 price history: {str(e)}", exc_info=True)
        return {}

def get_sp500_current_price():
    """
    Get current S&P500 price.
    
    Returns:
        S&P500 price in USD
    """
    try:
        api_key = os.environ.get("HTB_ALPHAVANTAGE_API_KEY")
        if not api_key:
            logger.warn("Alpha Vantage API key not found in environment variables")
            return None
            
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": "SPY",
                "apikey": api_key
            }
        )
        data = response.json()
        
        if "Global Quote" not in data:
            logger.error(f"Invalid response from Alpha Vantage: {data}")
            return None
            
        return float(data["Global Quote"]["05. price"])
    except Exception as e:
        logger.error(f"Error fetching current S&P500 price: {str(e)}", exc_info=True)
        return None

def get_sp500_historical_price(timestamp, price_history):
    """
    Get S&P500 price at a specific historical timestamp.
    
    Args:
        timestamp: Timestamp in milliseconds
        price_history: Dictionary of historical prices
    
    Returns:
        S&P500 price in USD
    """
    return get_btc_historical_price(timestamp, price_history)

def calculate_sp500_hold_performance(start_timestamp, sp500_current_price, price_history):
    """
    Calculate the performance of simply holding S&P500 from the start timestamp until now.
    
    Args:
        start_timestamp: Start timestamp in milliseconds
        sp500_current_price: Current S&P500 price
        price_history: Dictionary of historical prices
    
    Returns:
        Dictionary with S&P500 hold performance
    """
    try:
        starting_price = get_sp500_historical_price(start_timestamp, price_history)
        
        if starting_price and sp500_current_price:
            pct_change = ((sp500_current_price - starting_price) / starting_price) * 100
            return {
                'starting_price': starting_price,
                'current_price': sp500_current_price,
                'pct_change': pct_change
            }
        return None
    except Exception as e:
        logger.error(f"Error calculating S&P500 hold performance: {str(e)}", exc_info=True)
        return None