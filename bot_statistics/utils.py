import bisect
from datetime import datetime


def get_historical_price(timestamp, price_history):
    """
    Get price at a specific historical timestamp from a price history dictionary.

    Args:
        timestamp: Timestamp in milliseconds
        price_history: Dictionary of historical prices (timestamp_ms -> price)

    Returns:
        Price at the closest timestamp, or None if history is empty
    """
    if not price_history:
        return None

    timestamps = sorted(price_history.keys())
    
    # Find the index where timestamp would be inserted
    idx = bisect.bisect_left(timestamps, timestamp)

    if idx == 0:
        # If timestamp is before the earliest available data, use the earliest
        return price_history[timestamps[0]]
    elif idx == len(timestamps):
        # If timestamp is after the latest available data, use the latest
        return price_history[timestamps[-1]]
    else:
        # Find the closest timestamp (either before or after)
        before = timestamps[idx - 1]
        after = timestamps[idx]

        if timestamp - before <= after - timestamp:
            return price_history[before]
        else:
            return price_history[after]


def format_timestamp(timestamp_ms, format_str='%Y-%m-%d'):
    """
    Format a millisecond timestamp into a string.

    Args:
        timestamp_ms: Timestamp in milliseconds
        format_str: Strftime format string

    Returns:
        Formatted date string
    """
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime(format_str)
