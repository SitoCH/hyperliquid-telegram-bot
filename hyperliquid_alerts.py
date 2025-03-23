from telegram.constants import ParseMode
from telegram.ext import ContextTypes
import time
from datetime import datetime, timedelta

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt
from logging_utils import logger

async def check_profit_percentage(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        total_balance = float(user_state['marginSummary']['accountValue'])
        available_balance = float(user_state['withdrawable'])
        
        if available_balance > 100:
            message = [
                "💰 <b>Available balance alert</b> 💰",
                f"Total balance: {fmt(total_balance)} USDC",
                f"Available balance: {fmt(available_balance)} USDC",
            ]
            await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
        
        if user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in user_state["assetPositions"]
            )
            
            pnl_percentage = (total_pnl / total_balance) * 100
            
            if abs(pnl_percentage) > 50:
                alert_info = {
                    True: ("🚀", "profit"),
                    False: ("📉", "loss")
                }[pnl_percentage > 50]
                
                message = [
                    f"{alert_info[0]} <b>Unrealized {alert_info[1]} alert</b> {alert_info[0]}",
                    f"Total balance: {fmt(total_balance)} USDC",
                    f"Unrealized profit: {fmt(total_pnl)} USDC ({fmt(pnl_percentage)}%)",
                ]
                await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.critical(e, exc_info=True)


async def check_positions_to_close(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        
        if not user_state["assetPositions"]:
            return
            
        trade_history = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
        
        current_time = time.time()
        one_day_ago = (current_time - (24 * 60 * 60)) * 1000
        
        positions_to_close = []
        
        for asset_position in user_state["assetPositions"]:
            position = asset_position["position"]
            coin = position["coin"]
            pnl = float(position["unrealizedPnl"])

            position_trades = [
                trade for trade in trade_history
                if trade["coin"] == coin
            ]
            
            if position_trades:
                position_trades.sort(key=lambda x: int(x["timestamp"]))
                oldest_trade = position_trades[0]
                trade_time = int(oldest_trade["timestamp"])
                
                if trade_time < one_day_ago and pnl > 0.25:
                    positions_to_close.append({
                        "coin": coin,
                        "size": position["szi"],
                        "entry_price": position["entryPx"],
                        "pnl": pnl
                    })

        if positions_to_close:
            for pos in positions_to_close:
                message = [
                    f"⏰ <b>Stale position on {pos['coin']}</b>",
                    f"Current PnL: {fmt(pos['pnl'])} USDC"
                ]
                await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
                
    except Exception as e:
        logger.critical(e, exc_info=True)
