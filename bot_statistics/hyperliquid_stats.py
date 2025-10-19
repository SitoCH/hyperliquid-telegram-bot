import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

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

TradeFill = Dict[str, Any]


class StatsResult(TypedDict):
    winning_trades: int
    total_trades: int
    win_rate: float
    pnl: float
    adjusted_pnl: float
    total_fees: float
    pct_return: float


HoldPerformance = Dict[str, float]


def calculate_trading_stats(user_fills: List[TradeFill], cutoff_timestamp: int) -> StatsResult:
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
    max_notional_value: float = 0.0
    total_notional_value: float = 0.0
    
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
    initial_capital: float = max_notional_value
    if initial_capital <= 0:
        # Fallback if we can't determine capital from trades
        initial_capital = float(total_notional_value / len(all_trades)) if all_trades else 1000.0
    
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

def _fmt_signed_pct(value: float) -> str:
    sign = "+" if value > 0 else ("" if value == 0 else "")
    return f"{sign}{fmt(value)}%"


def _format_kv_block(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return "<pre>-</pre>"
    max_label = max(len(label) for label, _ in pairs)
    max_value = max(len(value) for _, value in pairs)
    lines = [f"{label.ljust(max_label)}  {value.rjust(max_value)}" for label, value in pairs]
    return f"<pre>" + "\n".join(lines) + "</pre>"


def format_stats_table(
    stats: StatsResult,
    btc_hold: Optional[HoldPerformance],
    sp500_hold: Optional[HoldPerformance] = None,
) -> str:
    """
    Format trading statistics as a clean, monospaced key-value block for Telegram.
    """
    pairs: List[Tuple[str, str]] = [
        ("Trades", f"{stats['winning_trades']} / {stats['total_trades']} ({fmt(stats['win_rate'])}%)"),
        ("PnL", f"{fmt(stats['pnl'])} USDC"),
        ("Fees", f"{fmt(stats['total_fees'])} USDC"),
        ("Net PnL", f"{fmt(stats['adjusted_pnl'])} USDC"),
    ]

    if stats.get('total_trades', 0) > 0:
        pairs.append(("Return", f"{fmt(stats['pct_return'])}%"))

    if btc_hold:
        pairs.append(("BTC HODL", f"{fmt(btc_hold['pct_change'])}%"))
        pairs.append(("vs BTC", _fmt_signed_pct(stats.get('pct_return', 0) - btc_hold['pct_change'])))

    if sp500_hold:
        pairs.append(("S&P 500 HODL", f"{fmt(sp500_hold['pct_change'])}%"))
        pairs.append(("vs S&P 500", _fmt_signed_pct(stats.get('pct_return', 0) - sp500_hold['pct_change'])))

    return _format_kv_block(pairs)

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        message_lines = [
            "📊 <b>Trading statistics</b>"
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
        message_lines.append("\n📅 <b>Last day</b>")
        message_lines.append(format_stats_table(stats_1d, btc_hold_1d, sp500_hold_1d))
        message_lines.append("📅 <b>Last 7 days</b>")
        message_lines.append(format_stats_table(stats_7d, btc_hold_7d, sp500_hold_7d))
        message_lines.append("📅 <b>Last 30 days</b>")
        message_lines.append(format_stats_table(stats_30d, btc_hold_30d, sp500_hold_30d))
        message_lines.append("📅 <b>Last 90 days</b>")
        message_lines.append(format_stats_table(stats_90d, btc_hold_90d, sp500_hold_90d))
        
        # Send the response
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
    except Exception as e:
        await telegram_utils.reply(update, f"Error getting stats: {str(e)}")
