from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hyperliquid_utils import hyperliquid_utils
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
            
            if abs(pnl_percentage) > 30:
                alert_info = {
                    True: ("🚀", "profit"),
                    False: ("📉", "loss")
                }[pnl_percentage > 30]
                
                message = [
                    f"{alert_info[0]} <b>Unrealized {alert_info[1]} alert</b> {alert_info[0]}",
                    f"Total balance: {fmt(total_balance)} USDC",
                    f"Unrealized profit: {fmt(total_pnl)} USDC ({fmt(pnl_percentage)}%)",
                ]
                await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.critical(e, exc_info=True)
