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
    # Filter closed positions after cutoff timestamp
    closed_positions = [trade for trade in user_fills if 
                       trade['time'] >= cutoff_timestamp and 
                       'dir' in trade and 
                       trade['dir'].startswith('Close')]
    
    # Calculate win rate
    winning_trades = [trade for trade in closed_positions if float(trade['closedPnl']) > 0]
    win_rate = (len(winning_trades) / len(closed_positions)) * 100 if closed_positions else 0
    
    # Calculate total PnL
    pnl = sum(float(trade['closedPnl']) for trade in closed_positions)
    
    return {
        'total_trades': len(closed_positions),
        'win_rate': win_rate,
        'pnl': pnl
    }

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message_lines = [
            "<b>Statistics:</b>"
        ]
        user_fills = hyperliquid_utils.info.user_fills(hyperliquid_utils.address)
        
        current_time = int(time.time() * 1000)
        stats_7d = calculate_trading_stats(user_fills, current_time - (7 * 24 * 60 * 60 * 1000))
        stats_30d = calculate_trading_stats(user_fills, current_time - (30 * 24 * 60 * 60 * 1000))
        
        message_lines.append("Last 7 Days:")
        message_lines.append(f"Total Closed Positions: {stats_7d['total_trades']}")
        message_lines.append(f"Win Rate: {stats_7d['win_rate']:.2f}%")
        message_lines.append(f"PnL: {stats_7d['pnl']:.2f} USDC")
        message_lines.append("")
        message_lines.append("Last 30 Days:")
        message_lines.append( f"Total Closed Positions: {stats_30d['total_trades']}")
        message_lines.append(f"Win Rate: {stats_30d['win_rate']:.2f}%")
        message_lines.append(f"PnL: {stats_30d['pnl']:.2f} USDC")
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")