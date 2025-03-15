import time
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils

def calculate_trading_stats(user_fills, cutoff_timestamp):
    """
    Calculate trading statistics for trades after the cutoff timestamp.
    
    Args:
        user_fills: List of trading fills
        cutoff_timestamp: Timestamp in milliseconds to filter trades
        
    Returns:
        Dictionary containing total_trades, win_rate, and pnl
    """
    # Filter all trades after cutoff timestamp
    all_trades = [trade for trade in user_fills if trade['time'] >= cutoff_timestamp]
    
    # Filter closed positions after cutoff timestamp
    closed_positions = [trade for trade in all_trades if 
                       'dir' in trade and 
                       trade['dir'].startswith('Close')]
    
    # Calculate win rate
    winning_trades = [trade for trade in closed_positions if float(trade['closedPnl']) > 0]
    win_rate = (len(winning_trades) / len(closed_positions)) * 100 if closed_positions else 0
    
    # Calculate total PnL from closed positions
    closed_pnl = sum(float(trade['closedPnl']) for trade in closed_positions)
    
    # Add fees from all trades to get complete PnL picture
    total_fees = sum(float(trade.get('fee', 0)) for trade in all_trades)
    
    # Calculate adjusted PnL (closed PnL minus all fees)
    adjusted_pnl = closed_pnl - total_fees
    
    return {
        'winning_trades': len(winning_trades),
        'total_trades': len(closed_positions),
        'win_rate': win_rate,
        'pnl': closed_pnl,
        'adjusted_pnl': adjusted_pnl,
        'total_fees': total_fees
    }

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message_lines = [
            "<b>Statistics:</b>"
        ]

        current_time = int(time.time() * 1000)
        one_month_ago = current_time - (30 * 24 * 60 * 60 * 1000)

        user_fills = hyperliquid_utils.info.user_fills_by_time(hyperliquid_utils.address, one_month_ago)
        
        stats_7d = calculate_trading_stats(user_fills, current_time - (7 * 24 * 60 * 60 * 1000))
        stats_30d = calculate_trading_stats(user_fills, one_month_ago)
        
        message_lines.append("Last 7 days:")
        message_lines.append(f"Win rate: {stats_7d['winning_trades']} / {stats_7d['total_trades']} ({stats_7d['win_rate']:.2f}%)")
        message_lines.append(f"PnL: {stats_7d['pnl']:.2f} USDC")
        message_lines.append(f"Total fees: {stats_7d['total_fees']:.2f} USDC")
        message_lines.append(f"Net PnL (after fees): {stats_7d['adjusted_pnl']:.2f} USDC")
        message_lines.append("")
        message_lines.append("Last 30 days:")
        message_lines.append(f"Win rate: {stats_30d['winning_trades']} / {stats_30d['total_trades']} ({stats_30d['win_rate']:.2f}%)")
        message_lines.append(f"PnL: {stats_30d['pnl']:.2f} USDC")
        message_lines.append(f"Total fees: {stats_30d['total_fees']:.2f} USDC") 
        message_lines.append(f"Net PnL (after fees): {stats_30d['adjusted_pnl']:.2f} USDC")
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")