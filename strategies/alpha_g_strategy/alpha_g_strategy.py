import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from telegram.constants import ParseMode

from logging_utils import logger
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.wyckoff.wyckoff_types import Timeframe
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from utils import fmt


@dataclass
class PriceMovement:
    """Represents a detected price movement for a coin."""
    symbol: str
    name: str
    price_change_pct: float
    current_price: float


@dataclass
class ReversalSignal:
    """Represents a potential reversal signal in a partial candle."""
    symbol: str
    name: str
    movement_type: str  # 'surge' or 'crash'
    full_candles_change_pct: float  
    current_change_pct: float
    current_price: float


@dataclass
class MovementAnalysisResult:
    """Container for all detected price movements."""
    surges: List[PriceMovement]
    crashes: List[PriceMovement]
    reversals: List[ReversalSignal]
    
    def has_any_movements(self) -> bool:
        """Check if any movements were detected."""
        return bool(self.surges or self.crashes or self.reversals)

class AlphaGStrategy():
    """AlphaG strategy stub - implementation to be added."""

    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str]]:
        """Get strategy parameters including filtered crypto data and exchange info."""
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
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
    ) -> MovementAnalysisResult:
        """
        Detect coins with price surges, crashes, and potential reversals.
        
        Args:
            coins: List of coin dictionaries with 'symbol' and 'name' keys
            lookback_days: Number of days to analyze
            threshold_pct: Absolute price change percentage threshold
            
        Returns:
            MovementAnalysisResult containing surges, crashes, and reversals
        """
        now = int(time.time() * 1000)
        current_candle_start = now - (now % (Timeframe.DAY_1.minutes * 60 * 1000))
        
        surges: List[PriceMovement] = []
        crashes: List[PriceMovement] = []
        reversals: List[ReversalSignal] = []
        
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
                    hyperliquid_utils.info.candles_snapshot,
                    True
                )
                
                last_two_full, partial_candle = self._extract_recent_candles(
                    candles, current_candle_start, coin
                )
                if last_two_full is None:
                    continue

                # Classify the movement
                movement = self._classify_movement(last_two_full, entry, threshold_pct)
                if movement:
                    movement_type, price_movement = movement
                    if movement_type == 'surge':
                        surges.append(price_movement)
                    else:  # crash
                        crashes.append(price_movement)
                    
                    # Check for reversal signal
                    reversal = self._detect_partial_reversal(
                        movement_type, last_two_full[-1], partial_candle, entry, price_movement.price_change_pct
                    )
                    if reversal:
                        reversals.append(reversal)

                logger.info(f"Analysis for {coin} done in {(time.time() - start_time):.2f}s")
                    
            except Exception as e:
                logger.error(f"Error analyzing {coin}: {str(e)}", exc_info=True)
                continue
        
        return MovementAnalysisResult(surges=surges, crashes=crashes, reversals=reversals)

    @staticmethod
    def _extract_recent_candles(
        candles: List[Dict],
        current_candle_start: int,
        coin_symbol: str
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        if not candles:
            logger.warning(f"No candle data returned for {coin_symbol}")
            return None, None

        candles = sorted(candles, key=lambda x: x['T'])[-3:]
        complete = [c for c in candles if c['T'] < current_candle_start]
        partial = next((c for c in candles if c['T'] >= current_candle_start), None)

        if len(complete) < 2:
            logger.warning(
                f"Insufficient complete candles for {coin_symbol}: {len(complete)} full candles available"
            )
            return None, partial

        return complete[-2:], partial

    @staticmethod
    def _classify_movement(
        last_two_full: List[Dict],
        coin_entry: Dict,
        threshold_pct: float,
    ) -> Optional[Tuple[str, PriceMovement]]:
        """Classify price movement as surge or crash if threshold is exceeded."""
        first_open = float(last_two_full[0]['o'])
        last_close = float(last_two_full[-1]['c'])
        
        if first_open == 0:
            return None
            
        price_change_pct = ((last_close - first_open) / first_open) * 100

        movement = PriceMovement(
            symbol=coin_entry['symbol'],
            name=coin_entry['name'],
            price_change_pct=price_change_pct,
            current_price=last_close
        )

        if price_change_pct > threshold_pct:
            logger.info(f"🚀 Surge detected in {coin_entry['symbol']}: {price_change_pct:.2f}%")
            return 'surge', movement

        if price_change_pct < -threshold_pct:
            logger.info(f"📉 Crash detected in {coin_entry['symbol']}: {price_change_pct:.2f}%")
            return 'crash', movement

        return None

    @staticmethod
    def _detect_partial_reversal(
        movement_type: str,
        last_full_candle: Dict,
        partial_candle: Optional[Dict],
        coin_entry: Dict,
        full_candles_change_pct: float,
    ) -> Optional[ReversalSignal]:
        """Detect if the partial candle shows a potential reversal signal."""
        if not partial_candle:
            return None

        prev_open = float(last_full_candle['o'])
        prev_close = float(last_full_candle['c'])
        current_open = float(partial_candle['o'])
        current_close = float(partial_candle['c'])

        prev_direction = prev_close - prev_open
        current_direction = current_close - current_open

        if prev_direction == 0 or current_direction == 0:
            return None

        # Check if direction flipped
        direction_flipped = (
            (prev_direction > 0 and current_direction < 0) or
            (prev_direction < 0 and current_direction > 0)
        )
        if not direction_flipped:
            return None

        prev_change_pct = ((prev_close - prev_open) / prev_open) * 100 if prev_open else 0.0
        current_change_pct = ((current_close - current_open) / current_open) * 100 if current_open else 0.0

        logger.info(
            f"🔄 Reversal signal in {coin_entry['symbol']}: "
            f"full candles {full_candles_change_pct:.2f}%, "
            f"prev candle {prev_change_pct:.2f}%, "
            f"current partial {current_change_pct:.2f}%"
        )

        return ReversalSignal(
            symbol=coin_entry['symbol'],
            name=coin_entry['name'],
            movement_type=movement_type,
            full_candles_change_pct=full_candles_change_pct,
            current_change_pct=current_change_pct,
            current_price=current_close
        )

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
            
            coins_to_analyze = filtered_coins[2:80]
            result = await self.detect_price_movements(coins_to_analyze, 2, 15.0)
            
            await message.delete() # type: ignore
            
            if result.surges:
                surge_message_lines = [
                    "<b>🚀 Price Surge Detection</b>",
                    ""
                ]
                for surge in sorted(result.surges, key=lambda x: x.price_change_pct, reverse=True):
                    surge_message_lines.append(
                        f"<b>{surge.symbol}</b> ({surge.name})\n"
                        f"  • Price Change: {fmt(surge.price_change_pct)}%\n"
                    )
                
                await telegram_utils.reply(update, '\n'.join(surge_message_lines), parse_mode=ParseMode.HTML)
            
            if result.crashes:
                crash_message_lines = [
                    "<b>📉 Price Crash Detection</b>",
                    ""
                ]
                for crash in sorted(result.crashes, key=lambda x: x.price_change_pct):
                    crash_message_lines.append(
                        f"<b>{crash.symbol}</b> ({crash.name})\n"
                        f"  • Price change: {fmt(crash.price_change_pct)}%\n"
                    )
                
                await telegram_utils.reply(update, '\n'.join(crash_message_lines), parse_mode=ParseMode.HTML)
            
            if result.reversals:
                reversal_message_lines = [
                    "<b>🔄 Partial Candle Reversal Signals</b>",
                    ""
                ]
                for reversal in sorted(result.reversals, key=lambda x: abs(x.current_change_pct), reverse=True):
                    initial_signal = "Surge" if reversal.movement_type == 'surge' else "Crash"
                    reversal_message_lines.append(
                        f"<b>{reversal.symbol}</b> ({reversal.name})\n"
                        f"  • Initial signal: {initial_signal} ({fmt(reversal.full_candles_change_pct)}%)\n"
                        f"  • Current candle: {fmt(reversal.current_change_pct)}%\n"
                    )

                await telegram_utils.reply(update, '\n'.join(reversal_message_lines), parse_mode=ParseMode.HTML)

            if not result.has_any_movements():
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
