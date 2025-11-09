import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

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
class ReversalSignal:
    """Represents a potential reversal signal in a partial candle."""
    symbol: str
    name: str
    movement_type: str  # 'surge' or 'crash'
    full_candles_change_pct: float  
    current_change_pct: float
    current_price: float
    confirmed: bool = False

class AlphaGStrategy():
    """AlphaG strategy."""

    COIN_MIN_VOLUME = 2_500_000

    def __init__(self) -> None:
        self._lookback_days: int = int(os.getenv("HTB_ALPHA_G_STRATEGY_LOOKBACK_DAYS", "3"))
        self._threshold_pct: float = float(os.getenv("HTB_ALPHA_G_STRATEGY_THRESHOLD_PCT", "20.0"))

    def filter_top_coins(
        self,
        meta: Any,
        all_mids: Dict
    ) -> List[Dict]:
        """Filter and sort coins according to fixed token strategy criteria."""

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "sparkline": "false"
        }

        coins = hyperliquid_utils.fetch_cryptos(params, page_count=2)

        universe, coin_data = meta[0]['universe'], meta[1]
        coin_volume = {u["name"]: float(c["dayNtlVlm"]) for u, c in zip(universe, coin_data)}

        filtered_coins = []

        for coin in coins:
            symbol = coin["symbol"]
            
            if symbol not in all_mids:
                logger.info(f"Excluding {symbol}: not available on Hyperliquid")
                continue

            volume = coin_volume.get(symbol, 0)
            if volume <= self.COIN_MIN_VOLUME:
                logger.info(f"Excluding {symbol}: 24h volume {fmt(volume)} USDC <= {fmt(self.COIN_MIN_VOLUME)} USDC")
                continue

            filtered_coins.append({
                "name": coin["name"],
                "symbol": symbol,
                "market_cap": coin["market_cap"],
            })

        return sorted(filtered_coins, key=lambda x: x["market_cap"], reverse=True)

    @staticmethod
    def _coin_volume_map_from_meta(meta: Any) -> Dict[str, float]:
        """Return mapping symbol -> 24h notional volume (USDC) using provided meta."""
        universe, coin_data = meta[0]['universe'], meta[1]
        return {u["name"]: float(c.get("dayNtlVlm", 0.0)) for u, c in zip(universe, coin_data)}

    def _compute_low_liquidity_positions(self, user_state: Dict, coin_volume_map: Dict[str, float]) -> List[str]:
        """Build list of formatted lines for open positions below a given 24h volume threshold.
        """
        lines: List[str] = []
        for asset_position in user_state.get("assetPositions", []):
            pos = asset_position.get("position", {})
            raw_coin = str(pos.get("coin", ""))
            coin_symbol = raw_coin.lstrip('k')
            vol = float(coin_volume_map.get(coin_symbol, 0.0))
            if vol < self.COIN_MIN_VOLUME:
                lines.append(
                    f"<b>{coin_symbol}</b>\n"
                    f"  • 24h vol: {fmt(vol)} USDC\n"
                )
        return lines

    @staticmethod
    def _build_portfolio_summary_message(user_state: Dict) -> str:
        """Return HTML-formatted summary of long/short portfolio state."""
        # Categorize positions as long or short
        long_positions = 0
        short_positions = 0
        total_long_margin = 0.0
        total_short_margin = 0.0
        total_long_position_value = 0.0
        total_short_position_value = 0.0

        for asset_position in user_state.get("assetPositions", []):
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
        return '\n'.join(message_lines)

    @staticmethod
    def _build_reversal_lines(reversals: List[ReversalSignal]) -> List[str]:
        lines = [
            "<b>🔄 Reversal Signals</b>",
            ""
        ]
        for reversal in sorted(
            reversals,
            key=lambda x: (abs(x.full_candles_change_pct), abs(x.current_change_pct)),
            reverse=True,
        ):
            initial_signal = "Surge" if reversal.movement_type == 'surge' else "Crash"
            icon = "🚀" if reversal.movement_type == 'surge' else "📉"
            line = (
                f"{icon} <b>{telegram_utils.get_link(reversal.symbol, f'TA_{reversal.symbol}')}</b> ({reversal.name})\n"
                f" • {initial_signal} ({fmt(reversal.full_candles_change_pct)}%)\n"
                f" • Daily change: {fmt(reversal.current_change_pct)}%\n"
            )
            if reversal.confirmed:
                line += f" • ✅ Confirmed reversal\n"
            lines.append(line)
        return lines

    async def detect_price_movements(
        self,
        coins: List[Dict],
        lookback_days: int,
        threshold_pct: float
    ) -> List[ReversalSignal]:
        """
        Detect coins with potential reversal signals.
        
        Args:
            coins: List of coin dictionaries with 'symbol' and 'name' keys
            lookback_days: Number of full daily candles to analyze (N)
            threshold_pct: Absolute price change percentage threshold
            
        Returns:
            List of ReversalSignal objects
        """
        now = int(time.time() * 1000)
        current_candle_start = now - (now % (Timeframe.DAY_1.minutes * 60 * 1000))
        
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
                    lookback_days + 1, 
                    hyperliquid_utils.info.candles_snapshot,
                    True
                )
                
                full_candles, partial_candle = self._extract_recent_candles(
                    candles, current_candle_start, coin, lookback_days
                )
                if full_candles is None:
                    continue

                # First, try to detect a confirmed reversal pattern:
                # N candles trending, then 1 full candle reversing, and current partial confirming
                confirmed = self._detect_confirmed_reversal(
                    candles=candles,
                    current_candle_start=current_candle_start,
                    coin_entry=entry,
                    lookback_days=lookback_days,
                    threshold_pct=threshold_pct,
                )
                if confirmed:
                    reversals.append(confirmed)
                    logger.info(f"Analysis for {coin} done in {(time.time() - start_time):.2f}s")
                    continue

                # Classify the movement
                movement = self._classify_movement(full_candles, entry, threshold_pct)
                if movement:
                    movement_type, price_change_pct = movement
                    
                    # Check for reversal signal
                    reversal = self._detect_partial_reversal(
                        movement_type, partial_candle, entry, price_change_pct
                    )
                    if reversal:
                        reversals.append(reversal)

                logger.info(f"Analysis for {coin} done in {(time.time() - start_time):.2f}s")
                    
            except Exception as e:
                logger.error(f"Error analyzing {coin}: {str(e)}", exc_info=True)
                continue
        
        return reversals

    @staticmethod
    def _detect_confirmed_reversal(
        candles: List[Dict],
        current_candle_start: int,
        coin_entry: Dict,
        lookback_days: int,
        threshold_pct: float,
    ) -> Optional[ReversalSignal]:
        """Detect a confirmed reversal pattern over N+1 full candles plus current partial.

        Pattern definition:
        - The oldest N of the last N+1 full daily candles exhibit a surge/crash exceeding threshold.
        - The most recent full candle (yesterday) moves in the opposite direction (reversal).
        - The current partial daily candle continues in the direction of the reversal (confirmation).
        """
        if not candles:
            return None

        candles_sorted = sorted(candles, key=lambda x: x['T'])
        complete = [c for c in candles_sorted if c['T'] < current_candle_start]
        partial = next((c for c in candles_sorted if c['T'] >= current_candle_start), None)

        if len(complete) < lookback_days + 1 or not partial:
            return None

        # Trend on first N of the last N+1 full candles
        trend_candles = complete[-(lookback_days + 1):-1]
        reversal_full = complete[-1]

        # Classify initial trend
        first_open = float(trend_candles[0]['o'])
        last_close = float(trend_candles[-1]['c'])
        if first_open == 0:
            return None
        trend_change_pct = ((last_close - first_open) / first_open) * 100

        movement_type: Optional[str] = None
        if trend_change_pct > threshold_pct:
            movement_type = 'surge'
        elif trend_change_pct < -threshold_pct:
            movement_type = 'crash'
        else:
            return None

        # Reversal full candle must be opposite to the movement_type
        rev_open = float(reversal_full['o'])
        rev_close = float(reversal_full['c'])
        rev_dir = rev_close - rev_open
        if rev_dir == 0:
            return None
        if movement_type == 'surge' and rev_dir >= 0:
            return None
        if movement_type == 'crash' and rev_dir <= 0:
            return None

        # Current partial must confirm the reversal (same direction as reversal full candle)
        cur_open = float(partial['o'])
        cur_close = float(partial['c'])
        cur_dir = cur_close - cur_open
        if cur_dir == 0:
            return None
        if (rev_dir > 0 and cur_dir <= 0) or (rev_dir < 0 and cur_dir >= 0):
            return None

        current_change_pct = ((cur_close - cur_open) / cur_open) * 100 if cur_open else 0.0

        logger.info(
            f"✅ Confirmed reversal in {coin_entry['symbol']}: "
            f"trend {trend_change_pct:.2f}% ({movement_type}), "
            f"yesterday reversed, current confirms"
        )

        return ReversalSignal(
            symbol=coin_entry['symbol'],
            name=coin_entry['name'],
            movement_type=movement_type,
            full_candles_change_pct=trend_change_pct,
            current_change_pct=current_change_pct,
            current_price=cur_close,
            confirmed=True,
        )

    @staticmethod
    def _extract_recent_candles(
        candles: List[Dict],
        current_candle_start: int,
        coin_symbol: str,
        lookback_days: int,
    ) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        if not candles:
            logger.warning(f"No candle data returned for {coin_symbol}")
            return None, None

        candles = sorted(candles, key=lambda x: x['T'])
        complete = [c for c in candles if c['T'] < current_candle_start]
        partial = next((c for c in candles if c['T'] >= current_candle_start), None)

        if len(complete) < lookback_days:
            logger.warning(
                f"Insufficient complete candles for {coin_symbol}: {len(complete)} full candles available, need {lookback_days}"
            )
            return None, partial

        return complete[-lookback_days:], partial

    @staticmethod
    def _classify_movement(
        full_candles: List[Dict],
        coin_entry: Dict,
        threshold_pct: float,
    ) -> Optional[Tuple[str, float]]:
        """Classify price movement as surge or crash if threshold is exceeded."""
        first_open = float(full_candles[0]['o'])
        last_close = float(full_candles[-1]['c'])
        
        if first_open == 0:
            return None
            
        price_change_pct = ((last_close - first_open) / first_open) * 100

        if price_change_pct > threshold_pct:
            logger.info(f"🚀 Surge detected in {coin_entry['symbol']}: {price_change_pct:.2f}%")
            return 'surge', price_change_pct

        if price_change_pct < -threshold_pct:
            logger.info(f"📉 Crash detected in {coin_entry['symbol']}: {price_change_pct:.2f}%")
            return 'crash', price_change_pct

        return None

    @staticmethod
    def _detect_partial_reversal(
        movement_type: str,
        partial_candle: Optional[Dict],
        coin_entry: Dict,
        full_candles_change_pct: float,
    ) -> Optional[ReversalSignal]:
        """Detect if the partial candle shows a potential reversal signal.

        A reversal is defined as the current partial daily candle moving in the
        opposite direction of the aggregate move across the last N full daily
        candles (movement_type = 'surge' for up, 'crash' for down).
        """
        if not partial_candle:
            return None

        current_open = float(partial_candle['o'])
        current_close = float(partial_candle['c'])
        current_direction = current_close - current_open

        if current_direction == 0:
            return None

        # Reversal relative to the multi-candle trend
        if movement_type == 'surge' and current_direction >= 0:
            return None
        if movement_type == 'crash' and current_direction <= 0:
            return None

        current_change_pct = ((current_close - current_open) / current_open) * 100 if current_open else 0.0

        logger.info(
            f"🔄 Reversal signal in {coin_entry['symbol']}: "
            f"full candles {full_candles_change_pct:.2f}%, "
            f"current partial {current_change_pct:.2f}% (opposite to {movement_type})"
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

            # Send portfolio summary
            summary_message = self._build_portfolio_summary_message(user_state)
            await telegram_utils.reply(update, summary_message, parse_mode=ParseMode.HTML)

            message = await telegram_utils.send(f"Analyzing coins...") # type: ignore

            # Fetch meta and mids once and reuse
            meta = hyperliquid_utils.info.meta_and_asset_ctxs()
            all_mids = hyperliquid_utils.info.all_mids()

            filtered_coins = self.filter_top_coins(meta, all_mids)
            
            coins_to_analyze = filtered_coins[2:80]
            reversals = await self.detect_price_movements(
                coins_to_analyze,
                self._lookback_days,
                self._threshold_pct,
            )
            
            await message.delete() # type: ignore
            
            # Pre-compute low-liquidity open positions once
            coin_volume_map = self._coin_volume_map_from_meta(meta)
            low_liquidity_positions = self._compute_low_liquidity_positions(
                user_state, coin_volume_map
            )

            if reversals:
                reversal_message_lines = self._build_reversal_lines(reversals)
                await telegram_utils.reply(update, '\n'.join(reversal_message_lines), parse_mode=ParseMode.HTML)

            if low_liquidity_positions:
                low_liq_lines = [
                    f"<b>⚠️ Open positions with low 24h volume</b>",
                    ""
                ]
                low_liq_lines.extend(low_liquidity_positions)
                await telegram_utils.reply(update, '\n'.join(low_liq_lines), parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}", exc_info=True)
            await telegram_utils.reply(update, f"Error analyzing portfolio: {str(e)}")

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE):
        """Initialize strategy - implementation to be added."""

        analyze_button_text = "analyze"
        telegram_utils.add_buttons([f"/{analyze_button_text}"], 2)
        telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))

        logger.info("AlphaG strategy initialized")
