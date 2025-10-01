import time

from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from logging_utils import logger
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.wyckoff.wyckoff_types import Timeframe
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from telegram.constants import ParseMode
from utils import fmt
from typing import List, Dict, Tuple

class AlphaGStrategy():
    """AlphaG strategy stub - implementation to be added."""

    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str]]:
        """Get strategy parameters including filtered crypto data and exchange info."""
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 200,
            "sparkline": "false"
        }
        
        coins = hyperliquid_utils.fetch_cryptos(params, page_count=2)
        all_mids = hyperliquid_utils.info.all_mids()
        
        return coins, all_mids

    def filter_top_coins(
        self,
        coins: List[Dict],
        all_mids: Dict[str, str]
    ) -> List[Dict]:
        """Filter and sort coins according to fixed token strategy criteria."""
        filtered_coins = []
        
        for coin in coins:
            symbol = coin["symbol"]
                            
            if symbol not in all_mids:
                logger.info(f"Excluding {symbol}: not available on Hyperliquid")
                continue

            filtered_coins.append({
                "name": coin["name"],
                "symbol": symbol,
                "market_cap": coin["market_cap"],
            })

        return sorted(filtered_coins, key=lambda x: x["market_cap"], reverse=True)

    async def detect_price_movements(
        self,
        coins: List[Dict],
        lookback_days: int,
        threshold_pct: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect coins with price surges and crashes based on first candle open and last candle close.
        
        Args:
            coins: List of coin dictionaries with 'symbol' and 'name' keys
            analysis_hours: Number of hours to analyze for movement detection (default: 48)
            threshold_pct: Absolute price change percentage threshold (default: 10.0)
                          Surge if > threshold_pct, crash if < -threshold_pct
            
        Returns:
            Tuple of (surge_coins, crash_coins) lists containing movement information
        """
        now = int(time.time() * 1000)
        surge_coins = []
        crash_coins = []
        
        for entry in coins:
            start_time = time.time()
            coin = entry['symbol']
            logger.info(f"Analyzing {coin} for price movements")
            
            try:
                candles = await get_candles_with_cache(
                    coin, 
                    Timeframe.DAY_1, 
                    now, 
                    lookback_days, 
                    hyperliquid_utils.info.candles_snapshot
                )
                
                if len(candles) >= lookback_days:
                    first_open = float(candles[0]['o'])
                    last_close = float(candles[-1]['c'])
                    
                    price_change_pct = ((last_close - first_open) / first_open) * 100
                    
                    if price_change_pct > threshold_pct:
                        surge_coins.append({
                            'symbol': coin,
                            'name': entry['name'],
                            'price_change_pct': price_change_pct,
                            'current_price': last_close
                        })
                        logger.info(f"🚀 Surge detected in {coin}: {price_change_pct:.2f}% change")
                    
                    elif price_change_pct < -threshold_pct:
                        crash_coins.append({
                            'symbol': coin,
                            'name': entry['name'],
                            'price_change_pct': price_change_pct,
                            'current_price': last_close
                        })
                        logger.info(f"📉 Crash detected in {coin}: {price_change_pct:.2f}% change")
                    
                    logger.info(f"Analysis for {coin} done in {(time.time() - start_time):.2f} seconds")
                else:
                    logger.warning(f"Insufficient candle data for {coin}: {len(candles)} days available, {lookback_days} required")
                    
            except Exception as e:
                logger.error(f"Error analyzing {coin}: {str(e)}", exc_info=True)
                continue
        
        return surge_coins, crash_coins

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
            long_margin_percentage = (total_long_margin / total_margin * 100) if total_margin > 0 else 0
            short_margin_percentage = (total_short_margin / total_margin * 100) if total_margin > 0 else 0

            message_lines = [
                "<b>⚖️ Portfolio Long / Short Analysis</b>",
                "",
                f"<b>Summary:</b>",
                f"Long Positions: {long_positions} positions",
                f" • Margin Used: {fmt(total_long_margin)} USDC ({fmt(long_margin_percentage)}%)",
                f" • Position Value: {fmt(total_long_position_value)} USDC",
                "",
                f"Short Positions: {short_positions} positions",
                f" • Margin Used: {fmt(total_short_margin)} USDC ({fmt(short_margin_percentage)}%)",
                f" • Position Value: {fmt(total_short_position_value)} USDC",
                "",
                f"<b>Totals:</b>",
                f" • Total Margin Used: {fmt(total_margin)} USDC",
                f" • Long/Short Margin Ratio: {fmt(total_long_margin / total_short_margin) if total_short_margin > 0 else '∞'}"
            ]
            
            await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)

            message = await telegram_utils.send(f"Analyzing coins...") # type: ignore
            coins, all_mids = self.get_strategy_params()
            filtered_coins = self.filter_top_coins(coins, all_mids)
            
            coins_to_analyze = filtered_coins[2:75]
            surge_coins, crash_coins = await self.detect_price_movements(coins_to_analyze, 2, 15.0)
            
            await message.delete() # type: ignore
            
            if surge_coins:
                surge_message_lines = [
                    "<b>🚀 Price Surge Detection</b>",
                    ""
                ]
                for surge_coin in sorted(surge_coins, key=lambda x: x['price_change_pct'], reverse=True):
                    surge_message_lines.append(
                        f"<b>{surge_coin['symbol']}</b> ({surge_coin['name']})\n"
                        f"  • Price Change: {fmt(surge_coin['price_change_pct'])}%\n"
                        f"  • Current Price: ${fmt(surge_coin['current_price'])}"
                    )
                
                await telegram_utils.reply(update, '\n'.join(surge_message_lines), parse_mode=ParseMode.HTML)
            
            if crash_coins:
                crash_message_lines = [
                    "<b>📉 Price Crash Detection</b>",
                    ""
                ]
                for crash_coin in sorted(crash_coins, key=lambda x: x['price_change_pct']):
                    crash_message_lines.append(
                        f"<b>{crash_coin['symbol']}</b> ({crash_coin['name']})\n"
                        f"  • Price Change: {fmt(crash_coin['price_change_pct'])}%\n"
                        f"  • Current Price: ${fmt(crash_coin['current_price'])}"
                    )
                
                await telegram_utils.reply(update, '\n'.join(crash_message_lines), parse_mode=ParseMode.HTML)
            
            if not surge_coins and not crash_coins:
                await telegram_utils.reply(update, "No significant price movements detected in the analyzed coins.")

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}", exc_info=True)
            await telegram_utils.reply(update, f"Error analyzing portfolio: {str(e)}")

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE):
        """Initialize strategy - implementation to be added."""

        analyze_button_text = "analyze"
        telegram_utils.add_buttons([f"/{analyze_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))

        logger.info("AlphaG strategy initialized")
