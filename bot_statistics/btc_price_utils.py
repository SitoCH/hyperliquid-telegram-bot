import time
import requests
from logging_utils import logger
from .utils import get_historical_price


def get_btc_price_history(start_timestamp):
    """
    Fetch BTC price history for a given time range.

    Args:
        start_timestamp: Start timestamp in milliseconds

    Returns:
        Dictionary with timestamps as keys and prices as values
    """

    try:
        # Convert to seconds for CoinGecko API
        start_sec = int(start_timestamp / 1000)
        end_sec = time.time()

        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
            params={
                "vs_currency": "usd",
                "from": start_sec,
                "to": end_sec
            }
        )
        data = response.json()

        # Create a dictionary mapping timestamps to prices
        price_history = {}
        for timestamp_ms, price in data["prices"]:
            price_history[int(timestamp_ms)] = price

        return price_history

    except Exception as e:
        logger.error(f"Error fetching BTC price history: {str(e)}", exc_info=True)
        return {}


def get_btc_current_price():
    """
    Get current BTC price.

    Returns:
        BTC price in USD
    """
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"}
        )
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception as e:
        logger.error(f"Error fetching current BTC price: {str(e)}", exc_info=True)
        return None


def calculate_btc_hold_performance(start_timestamp, btc_current_price, price_history):
    """
    Calculate the performance of simply holding BTC from the start timestamp until now.

    Args:
        start_timestamp: Start timestamp in milliseconds
        btc_current_price: Current BTC price
        price_history: Dictionary of historical prices (optional)

    Returns:
        Dictionary with BTC hold performance
    """
    try:
        starting_price = get_historical_price(start_timestamp, price_history)

        if starting_price and btc_current_price:
            pct_change = ((btc_current_price - starting_price) / starting_price) * 100
            return {
                'starting_price': starting_price,
                'current_price': btc_current_price,
                'pct_change': pct_change
            }
        return None
    except Exception as e:
        logger.error(f"Error calculating BTC hold performance: {str(e)}", exc_info=True)
        return None
