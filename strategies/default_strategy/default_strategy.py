import datetime
import random
from typing import Any
from telegram.ext import ContextTypes, CommandHandler
from strategies.base_strategy.base_strategy import BaseStrategy
from logging_utils import logger
from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils


class DefaultStrategy(BaseStrategy):
    """
    A basic trading strategy that serves as a default implementation.
    It implementation is minimal and meant as a starting point.
    """

    def get_strategy_params(self) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
        """Get standard parameters for the default strategy."""
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 10,
            "sparkline": "false",
            "price_change_percentage": "24h,30d,1y",
        }
        cryptos = hyperliquid_utils.fetch_cryptos(params)
        all_mids = hyperliquid_utils.info.all_mids()
        meta = hyperliquid_utils.info.meta()
        return cryptos, all_mids, meta

    def filter_top_cryptos(
        self,
        cryptos: list[dict[str, Any]],
        all_mids: dict[str, str],
        meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Filter to return available coins on Hyperliquid."""
        filtered = []
        for coin in cryptos:
            symbol = coin["symbol"]
            if symbol in all_mids:
                filtered.append({
                    "name": coin["name"],
                    "symbol": symbol,
                    "market_cap": coin["market_cap"],
                    "price_change_percentage_1y_in_currency": coin.get("price_change_percentage_1y_in_currency"),
                })
        return filtered

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Initialize with basic rebalance and analyze commands."""
        try:
            logger.info("Initializing default strategy")

            rebalance_button_text = "rebalance"
            telegram_utils.add_buttons([f"/{rebalance_button_text}"], 1)
            telegram_utils.add_handler(CommandHandler(rebalance_button_text, self.rebalance))

            analyze_button_text = "analyze"
            telegram_utils.add_buttons([f"/{analyze_button_text}"], 1)
            telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))

            telegram_utils.run_repeating(
                self.check_position_allocation_drifts,
                interval=datetime.timedelta(minutes=random.randint(12 * 60 - 10, 12 * 60 + 10))
            )
        except Exception as e:
            logger.error(f"Error executing default strategy: {str(e)}")
