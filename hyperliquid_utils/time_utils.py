from datetime import datetime


def format_timestamp(timestamp_ms: int, format_str: str = '%Y-%m-%d') -> str:
    """
    Format a millisecond timestamp into a string.

    Args:
        timestamp_ms: Timestamp in milliseconds
        format_str: Strftime format string

    Returns:
        Formatted date string
    """
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime(format_str)
