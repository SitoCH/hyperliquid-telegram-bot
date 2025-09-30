from dataclasses import dataclass
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from strategies.base_strategy.base_strategy import BaseStrategy, BaseStrategyConfig
from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from telegram.constants import ParseMode
from utils import fmt
from typing import List, Dict, Set, Tuple


@dataclass
class AlphaGConfig:
    """Configuration specific to AlphaG strategy."""
    pass


class AlphaGStrategy(BaseStrategy):
    """AlphaG strategy stub - implementation to be added."""

    def __init__(self):
        self._config = BaseStrategyConfig(
            leverage=5,
            min_yearly_performance=20.0
        )
        
        self._g_alpha_config = AlphaGConfig()

    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        """Get strategy parameters including filtered crypto data and exchange info."""
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "sparkline": "false",
            "price_change_percentage": "24h,30d,1y",
        }
        
        cryptos = hyperliquid_utils.fetch_cryptos(params, page_count=2)
        all_mids = hyperliquid_utils.info.all_mids()
        meta = hyperliquid_utils.info.meta()
        
        return cryptos, all_mids, meta

    def filter_top_cryptos(
        self,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> List[Dict]:
        """Filter and sort cryptos according to fixed token strategy criteria."""
        filtered_cryptos = []
        asset_info_map = {
            info["name"]: int(info["maxLeverage"])
            for info in meta.get("universe", [])
        }
        
        for coin in cryptos:
            symbol = coin["symbol"]
            yearly_change = coin["price_change_percentage_1y_in_currency"]
                            
            if symbol not in all_mids:
                logger.info(f"Excluding {symbol}: not available on Hyperliquid")
                continue

            max_leverage = asset_info_map.get(symbol)
            if max_leverage is not None and self.config.leverage > max_leverage:
                logger.info(f"Excluding {symbol}: strategy leverage {self.config.leverage} exceeds max allowed leverage {max_leverage}")
                continue

            filtered_cryptos.append({
                "name": coin["name"],
                "symbol": symbol,
                "market_cap": coin["market_cap"],
                "price_change_percentage_1y_in_currency": yearly_change,
            })

        return sorted(filtered_cryptos, key=lambda x: x["market_cap"], reverse=True)

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Analyze current portfolio state and display information."""
        try:
            # Get current user state from Hyperliquid
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            
            if not user_state.get("assetPositions"):
                await telegram_utils.reply(update, "No open positions found.")
                return

            # Categorize positions as long or short
            long_positions = 0
            short_positions = 0
            total_long_margin = 0.0
            total_short_margin = 0.0
            total_long_position_value = 0.0
            total_short_position_value = 0.0
            
            for asset_position in user_state["assetPositions"]:
                position = asset_position["position"]
                size = float(position["szi"])
                position_value = float(position["positionValue"])
                margin_used = float(position["marginUsed"])

                if size > 0:  # Long position
                    long_positions += 1
                    total_long_margin += margin_used
                    total_long_position_value += position_value
                else:  # Short position
                    short_positions += 1
                    total_short_margin += margin_used
                    total_short_position_value += position_value
            
            # Calculate balance metrics
            total_margin = total_long_margin + total_short_margin
            total_position_value = total_long_position_value + total_short_position_value
            long_margin_percentage = (total_long_margin / total_margin * 100) if total_margin > 0 else 0
            short_margin_percentage = (total_short_margin / total_margin * 100) if total_margin > 0 else 0
            margin_difference = total_long_margin - total_short_margin

            message_lines = [
                "<b>📊 Portfolio Long/Short Analysis</b>",
                "",
                f"<b>Summary:</b>",
                f"🟢 Long Positions: {long_positions} positions",
                f"   • Margin Used: {fmt(total_long_margin)} USDC ({fmt(long_margin_percentage)}%)",
                f"   • Position Value: {fmt(total_long_position_value)} USDC",
                "",
                f"🔴 Short Positions: {short_positions} positions",
                f"   • Margin Used: {fmt(total_short_margin)} USDC ({fmt(short_margin_percentage)}%)",
                f"   • Position Value: {fmt(total_short_position_value)} USDC",
                "",
                f"⚖️ <b>Totals:</b>",
                f"   • Total Margin Used: {fmt(total_margin)} USDC",
                f"   • Total Position Value: {fmt(total_position_value)} USDC",
                f"   • Margin Difference (L-S): {fmt(margin_difference)} USDC",
                f"   • Long/Short Margin Ratio: {fmt(total_long_margin / total_short_margin) if total_short_margin > 0 else '∞'}"
            ]
            
            await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)
            logger.info(f"Portfolio analysis completed: {long_positions} longs, {short_positions} shorts")
            
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}", exc_info=True)
            await telegram_utils.reply(update, f"Error analyzing portfolio: {str(e)}")

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE):
        """Initialize strategy - implementation to be added."""

        analyze_button_text = "analyze"
        telegram_utils.add_buttons([f"/{analyze_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))

        logger.info("AlphaG strategy initialized")
