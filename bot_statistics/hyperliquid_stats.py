import time
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from tabulate import simple_separated_format, tabulate

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt
from logging_utils import logger
from bot_statistics.btc_price_utils import (
    get_btc_price_history, 
    get_btc_current_price,
    calculate_btc_hold_performance
)
from bot_statistics.sp500_price_utils import (
    get_sp500_price_history,
    get_sp500_current_price,
    calculate_sp500_hold_performance
)

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

def format_stats_table(stats, btc_hold, sp500_hold=None):
    """
    Format trading statistics as a tabulated table.
    
    Args:
        stats: Dictionary containing trading statistics
        btc_hold: Dictionary containing BTC hold performance stats
        sp500_hold: Dictionary containing S&P500 hold performance stats (optional)
        
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
        
        if btc_hold:
            table_data.append(["BTC:", ""])
            table_data.append(["HODL return", f"{fmt(btc_hold['pct_change'])}%"])
            relative_btc_performance = stats['pct_return'] - btc_hold['pct_change']
            table_data.append(["Bot vs HODL perf.", f"{fmt(relative_btc_performance)}%"])
        
        if sp500_hold:
            table_data.append(["S&P500:", ""])
            table_data.append(["HODL return", f"{fmt(sp500_hold['pct_change'])}%"])
            relative_sp500_performance = stats['pct_return'] - sp500_hold['pct_change']
            table_data.append(["Bot vs HODL perf.", f"{fmt(relative_sp500_performance)}%"])
    
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

        user_fills = hyperliquid_utils.info.user_fills_by_time(hyperliquid_utils.address, three_months_ago, None, True)
        
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
        
        sp500_price_history = get_sp500_price_history(one_month_ago)
        sp500_current_price = get_sp500_current_price()
        sp500_hold_1d = calculate_sp500_hold_performance(one_day_ago, sp500_current_price, sp500_price_history)
        sp500_hold_7d = calculate_sp500_hold_performance(one_week_ago, sp500_current_price, sp500_price_history)
        sp500_hold_30d = calculate_sp500_hold_performance(one_month_ago, sp500_current_price, sp500_price_history)
        sp500_hold_90d = calculate_sp500_hold_performance(three_months_ago, sp500_current_price, sp500_price_history)
        
        # Format tables using the helper function
        message_lines.append("Last day:")
        message_lines.append(format_stats_table(stats_1d, btc_hold_1d, sp500_hold_1d))
        message_lines.append("")
        message_lines.append("Last 7 days:")
        message_lines.append(format_stats_table(stats_7d, btc_hold_7d, sp500_hold_7d))
        message_lines.append("")
        message_lines.append("Last 30 days:")
        message_lines.append(format_stats_table(stats_30d, btc_hold_30d, sp500_hold_30d))
        message_lines.append("")
        message_lines.append("Last 90 days:")
        message_lines.append(format_stats_table(stats_90d, btc_hold_90d, sp500_hold_90d))
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")
