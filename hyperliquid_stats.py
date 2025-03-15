import time
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from tabulate import simple_separated_format, tabulate

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt

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
    
    return {
        'winning_trades': len(winning_trades),
        'total_trades': len(closed_positions),
        'win_rate': win_rate,
        'pnl': closed_pnl,
        'adjusted_pnl': adjusted_pnl,
        'total_fees': total_fees
    }

def format_stats_table(stats):
    """
    Format trading statistics as a tabulated table.
    
    Args:
        stats: Dictionary containing trading statistics
        title: Title for the table (e.g., "Last 7 days")
        
    Returns:
        Formatted table as a string
    """
    tablefmt = simple_separated_format('  ')
    
    table = tabulate([
        ["Win rate", f"{stats['winning_trades']} / {stats['total_trades']} ({fmt(stats['win_rate'])}%)"],
        ["PnL", f"{fmt(stats['pnl'])} USDC"],
        ["Total fees", f"{fmt(stats['total_fees'])} USDC"],
        ["Net PnL (after fees)", f"{fmt(stats['adjusted_pnl'])} USDC"]
    ], headers=["", ""], tablefmt=tablefmt, colalign=("left", "right"))
    
    return f"<pre>{table}</pre>"

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message_lines = [
            "<b>Trading Statistics:</b>"
        ]

        current_time = int(time.time() * 1000)
        one_month_ago = current_time - (30 * 24 * 60 * 60 * 1000)

        user_fills = hyperliquid_utils.info.user_fills_by_time(hyperliquid_utils.address, one_month_ago)
        
        # Calculate stats for both time periods
        stats_1d = calculate_trading_stats(user_fills, current_time - (24 * 60 * 60 * 1000))
        stats_7d = calculate_trading_stats(user_fills, current_time - (7 * 24 * 60 * 60 * 1000))
        stats_30d = calculate_trading_stats(user_fills, one_month_ago)
        
        # Format tables using the helper function
        message_lines.append("Last day:")
        message_lines.append(format_stats_table(stats_1d))
        message_lines.append("Last 7 days:")
        message_lines.append(format_stats_table(stats_7d))
        message_lines.append("")
        message_lines.append("Last 30 days:")
        message_lines.append(format_stats_table(stats_30d))
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")