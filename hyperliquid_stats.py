import time
import requests
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from tabulate import simple_separated_format, tabulate
import bisect

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt
from logging_utils import logger

def get_btc_price_history(start_timestamp):
    """
    Fetch BTC price history for a given time range.
    
    Args:
        start_timestamp: Start timestamp in milliseconds
        end_timestamp: End timestamp in milliseconds
        
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

def get_btc_historical_price(timestamp, price_history):
    """
    Get BTC price at a specific historical timestamp.
    
    Args:
        timestamp: Timestamp in milliseconds
        price_history: Dictionary of historical prices (optional)
    
    Returns:
        BTC price in USD
    """
    timestamps = sorted(price_history.keys())
    if not timestamps:
        return None
        
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
        before = timestamps[idx-1]
        after = timestamps[idx]
        
        if timestamp - before <= after - timestamp:
            return price_history[before]
        else:
            return price_history[after]

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
        starting_price = get_btc_historical_price(start_timestamp, price_history)
        
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

def calculate_trading_stats(user_fills, cutoff_timestamp):
    """
    Calculate trading statistics for trades after the cutoff timestamp.
    
    Args:
        user_fills: List of trading fills
        cutoff_timestamp: Timestamp in milliseconds to filter trades
        
    Returns:
        Dictionary containing total_trades, win_rate, and pnl
    """
    all_trades = [trade for trade in user_fills if trade['time'] >= cutoff_timestamp]
    
    total_fees = sum(float(trade.get('fee', 0)) for trade in all_trades)
    
    # Calculate maximum capital employed based on trade sizes
    max_notional_value = 0
    total_notional_value = 0
    
    for trade in all_trades:
        # Only consider opening positions for capital calculation
        if 'dir' in trade and not trade['dir'].startswith('Close'):
            if 'sz' in trade and 'px' in trade:
                notional_value = float(trade['sz']) * float(trade['px'])
                total_notional_value += notional_value
                max_notional_value = max(max_notional_value, notional_value)
    
    # Rest of the aggregation code for PnL calculation
    aggregated_trades = {}
    for trade in all_trades:
        trade_hash = trade.get('hash', 'unknown')
        if trade_hash not in aggregated_trades:
            aggregated_trades[trade_hash] = {
                'is_close': False,
                'closed_pnl': 0.0,
            }
        
        if 'dir' in trade and trade['dir'].startswith('Close'):
            aggregated_trades[trade_hash]['is_close'] = True
        
        if 'closedPnl' in trade:
            aggregated_trades[trade_hash]['closed_pnl'] += float(trade['closedPnl'])
    
    closed_positions = {
        hash_id: data for hash_id, data in aggregated_trades.items() 
        if data['is_close']
    }
    
    winning_trades = {
        hash_id: data for hash_id, data in closed_positions.items() 
        if data['closed_pnl'] > 0
    }
    
    win_rate = (len(winning_trades) / len(closed_positions)) * 100 if closed_positions else 0
    
    closed_pnl = sum(data['closed_pnl'] for data in closed_positions.values())
    
    adjusted_pnl = closed_pnl - total_fees
    
    # Calculate percentage return using actual capital employed
    initial_capital = max_notional_value
    if initial_capital <= 0:
        # Fallback if we can't determine capital from trades
        initial_capital = total_notional_value / len(all_trades) if all_trades else 1000
    
    if initial_capital > 0:
        pct_return = (adjusted_pnl / initial_capital) * 100
    else:
        pct_return = 0
    
    return {
        'winning_trades': len(winning_trades),
        'total_trades': len(closed_positions),
        'win_rate': win_rate,
        'pnl': closed_pnl,
        'adjusted_pnl': adjusted_pnl,
        'total_fees': total_fees,
        'pct_return': pct_return
    }

def format_stats_table(stats, btc_hold):
    """
    Format trading statistics as a tabulated table.
    
    Args:
        stats: Dictionary containing trading statistics
        btc_hold: Dictionary containing BTC hold performance stats (optional)
        
    Returns:
        Formatted table as a string
    """
    tablefmt = simple_separated_format('  ')
    
    table_data = [
        ["Win rate", f"{stats['winning_trades']} / {stats['total_trades']} ({fmt(stats['win_rate'])}%)"],
        ["PnL", f"{fmt(stats['pnl'])} USDC"],
        ["Total fees", f"{fmt(stats['total_fees'])} USDC"],
        ["Net PnL (ex fees)", f"{fmt(stats['adjusted_pnl'])} USDC"]
    ]
    
    if 'total_trades' in stats and stats['total_trades'] > 0:
        table_data.append(["Bot return", f"{fmt(stats['pct_return'])}%"])
        table_data.append(["BTC hold return", f"{fmt(btc_hold['pct_change'])}%"])
        
        relative_performance = stats['pct_return'] - btc_hold['pct_change']
        performance_text = f"{fmt(relative_performance)}%"
        table_data.append(["Bot vs BTC perf.", performance_text])
    
    table = tabulate(table_data, headers=["", ""], tablefmt=tablefmt, colalign=("left", "right"))
    
    return f"<pre>{table}</pre>"

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message_lines = [
            "<b>Trading Statistics:</b>"
        ]

        current_time = int(time.time() * 1000)
        three_months_ago = current_time - (90 * 24 * 60 * 60 * 1000)
        one_month_ago = current_time - (30 * 24 * 60 * 60 * 1000)
        one_week_ago = current_time - (7 * 24 * 60 * 60 * 1000)
        one_day_ago = current_time - (24 * 60 * 60 * 1000)

        user_fills = hyperliquid_utils.info.user_fills_by_time(hyperliquid_utils.address, three_months_ago)
        
        stats_1d = calculate_trading_stats(user_fills, one_day_ago)
        stats_7d = calculate_trading_stats(user_fills, one_week_ago)
        stats_30d = calculate_trading_stats(user_fills, one_month_ago)
        stats_90d = calculate_trading_stats(user_fills, three_months_ago)
        
        btc_price_history = get_btc_price_history(one_month_ago)
        btc_current_price = get_btc_current_price()
        btc_hold_1d = calculate_btc_hold_performance(one_day_ago, btc_current_price, btc_price_history)
        btc_hold_7d = calculate_btc_hold_performance(one_week_ago, btc_current_price, btc_price_history)
        btc_hold_30d = calculate_btc_hold_performance(one_month_ago, btc_current_price, btc_price_history)
        btc_hold_90d = calculate_btc_hold_performance(three_months_ago, btc_current_price, btc_price_history)
        
        # Format tables using the helper function
        message_lines.append("Last day:")
        message_lines.append(format_stats_table(stats_1d, btc_hold_1d))
        message_lines.append("")
        message_lines.append("Last 7 days:")
        message_lines.append(format_stats_table(stats_7d, btc_hold_7d))
        message_lines.append("")
        message_lines.append("Last 30 days:")
        message_lines.append(format_stats_table(stats_30d, btc_hold_30d))
        message_lines.append("")
        message_lines.append("Last 90 days:")
        message_lines.append(format_stats_table(stats_90d, btc_hold_90d))
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")
