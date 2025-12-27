from typing import Dict, List, Optional, Tuple
import base64
import os

from logging_utils import logger
from telegram_utils import telegram_utils
from utils import exchange_enabled, fmt_price

from ..wyckoff_types import SignificantLevelsData, Timeframe, WyckoffState
from .wyckoff_multi_timeframe_types import MultiTimeframeDirection

MIN_RR_DEFAULT = 1.4

# Entry timing filter constants
PIVOT_PROXIMITY_THRESHOLD = 0.015  # 1.5% from pivot point
VWAP_CONFIRMATION_REQUIRED = True  # Require VWAP alignment for entry


def check_entry_timing(
    coin: str,
    direction: MultiTimeframeDirection,
    mid: float,
    states: Dict[Timeframe, WyckoffState],
    significant_levels: Dict[Timeframe, SignificantLevelsData],
) -> Tuple[bool, str, Optional[float]]:
    """
    Check if current price action offers good entry timing based on:
    1. Proximity to pivot points (support for longs, resistance for shorts)
    2. VWAP confirmation (price above VWAP for longs, below for shorts)
    3. RSI not overbought/oversold against direction
    
    Returns:
        tuple: (is_good_entry, reason, optimal_entry_price)
    """
    if direction == MultiTimeframeDirection.NEUTRAL:
        return True, "", None
    
    reasons = []
    entry_price = mid
    
    # Get 30m and 1h states for timing analysis
    state_30m = states.get(Timeframe.MINUTES_30)
    state_1h = states.get(Timeframe.HOUR_1)
    
    # 1. Check VWAP alignment if available
    # vwap_bias is an int: 1 = above VWAP, -1 = below VWAP, 0 = neutral
    vwap_aligned = True
    if state_30m and state_30m.vwap_bias != 0:
        if direction == MultiTimeframeDirection.BULLISH:
            # For longs, we want price at or above VWAP (1 or 0)
            vwap_aligned = state_30m.vwap_bias >= 0
        else:  # BEARISH
            # For shorts, we want price at or below VWAP (-1 or 0)
            vwap_aligned = state_30m.vwap_bias <= 0
        
        if not vwap_aligned and VWAP_CONFIRMATION_REQUIRED:
            vwap_desc = "above" if state_30m.vwap_bias > 0 else "below"
            reasons.append(f"VWAP={vwap_desc} contradicts {direction.value} entry")
    
    # 2. Check RSI for overbought/oversold conditions
    # Use higher thresholds for crypto - momentum often runs to extremes
    rsi_ok = True
    for state in [state_30m, state_1h]:
        if state and state.rsi_value > 0:
            # Only block entries at extreme RSI levels (82+ for longs, 18- for shorts)
            # Also require both timeframes to confirm extreme, not just one
            if direction == MultiTimeframeDirection.BULLISH and state.rsi_value > 82:
                # Check if this is a sustained overbought (both TFs agree)
                other_rsi = state_1h.rsi_value if state == state_30m and state_1h else (
                    state_30m.rsi_value if state == state_1h and state_30m else 50)
                if other_rsi > 75:  # Both timeframes showing overbought
                    rsi_ok = False
                    reasons.append(f"RSI overbought ({state.rsi_value:.1f}) for long entry")
            elif direction == MultiTimeframeDirection.BEARISH and state.rsi_value < 18:
                other_rsi = state_1h.rsi_value if state == state_30m and state_1h else (
                    state_30m.rsi_value if state == state_1h and state_30m else 50)
                if other_rsi < 25:  # Both timeframes showing oversold
                    rsi_ok = False
                    reasons.append(f"RSI oversold ({state.rsi_value:.1f}) for short entry")
    
    # 3. Check proximity to significant levels for optimal entry
    optimal_entry = None
    for tf in [Timeframe.MINUTES_30, Timeframe.HOUR_1]:
        if tf not in significant_levels:
            continue
        
        levels = significant_levels[tf]
        
        if direction == MultiTimeframeDirection.BULLISH:
            # For longs, look for nearby support levels
            nearby_supports = [
                s for s in levels.get("support", [])
                if abs(s - mid) / mid <= PIVOT_PROXIMITY_THRESHOLD and s <= mid
            ]
            if nearby_supports:
                # Entry near support is good
                optimal_entry = max(nearby_supports)  # Closest to price from below
                logger.info(f"{coin}: Good long entry near support at {optimal_entry:.4f}")
        else:  # BEARISH
            # For shorts, look for nearby resistance levels
            nearby_resistances = [
                r for r in levels.get("resistance", [])
                if abs(r - mid) / mid <= PIVOT_PROXIMITY_THRESHOLD and r >= mid
            ]
            if nearby_resistances:
                # Entry near resistance is good
                optimal_entry = min(nearby_resistances)  # Closest to price from above
                logger.info(f"{coin}: Good short entry near resistance at {optimal_entry:.4f}")
    
    # Entry timing is good if VWAP is aligned and RSI is not extreme
    is_good = vwap_aligned and rsi_ok
    reason = "; ".join(reasons) if reasons else "Good entry timing"
    
    return is_good, reason, optimal_entry


def get_trade_suggestion(
    coin: str,
    direction: MultiTimeframeDirection,
    mid: float,
    significant_levels: Dict[Timeframe, SignificantLevelsData],
    confidence: float,
    states: Optional[Dict[Timeframe, WyckoffState]] = None,
) -> Optional[str]:
    """Generate trade suggestion with stop loss and take profit based on nearby levels.

    This function is extracted from wyckoff_multi_timeframe_description to keep that file small.
    It examines significant support/resistance levels across timeframes and, if confidence and
    risk:reward constraints are satisfied, returns a formatted suggestion string.
    
    Args:
        coin: The coin symbol
        direction: Bullish or bearish direction
        mid: Current mid price
        significant_levels: Dict of timeframe to support/resistance levels
        confidence: Confidence level (0-1)
        states: Optional dict of timeframe to WyckoffState for entry timing checks
    """

    min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
    if confidence < min_confidence:
        logger.info(
            f"Skipping trade suggestion for {coin}: confidence {confidence:.2f} below {min_confidence:.2f}"
        )
        return None

    # Check entry timing if states are available
    if states:
        entry_ok, entry_reason, _ = check_entry_timing(
            coin, direction, mid, states, significant_levels
        )
        if not entry_ok:
            logger.info(f"Skipping trade suggestion for {coin}: poor entry timing - {entry_reason}")
            return None

    min_rr = float(os.getenv("HTB_TRADE_MIN_RR", str(MIN_RR_DEFAULT)))

    def get_valid_levels(
        coin: str,
        timeframe: Timeframe,
        min_dist_sl: float,
        max_dist_sl: float,
        min_dist_tp: float,
        max_dist_tp: float,
    ) -> tuple[List[float], List[float], List[float], List[float]]:
        """Get valid support and resistance levels for a specific timeframe with separate ranges for SL and TP."""
        if timeframe not in significant_levels:
            logger.info(
                f"Skipping trade suggestion for {coin} {timeframe.name}: timeframe not in significant levels"
            )
            return ([], [], [], [])

        # Separate levels for take profit and stop loss with different distance constraints
        tp_resistances = [
            r
            for r in significant_levels[timeframe]["resistance"]
            if min_dist_tp <= abs(r - mid) <= max_dist_tp
        ]
        tp_supports = [
            s
            for s in significant_levels[timeframe]["support"]
            if min_dist_tp <= abs(s - mid) <= max_dist_tp
        ]
        sl_resistances = [
            r
            for r in significant_levels[timeframe]["resistance"]
            if min_dist_sl <= abs(r - mid) <= max_dist_sl
        ]
        sl_supports = [
            s
            for s in significant_levels[timeframe]["support"]
            if min_dist_sl <= abs(s - mid) <= max_dist_sl
        ]

        if not (tp_resistances or tp_supports or sl_resistances or sl_supports):
            logger.info(
                f"Skipping trade suggestion for {coin} {timeframe.name}: no levels within TP/SL ranges"
            )

        return (tp_resistances, tp_supports, sl_resistances, sl_supports)

    def get_trade_levels(
        direction: MultiTimeframeDirection,
        tp_resistances: List[float],
        tp_supports: List[float],
        sl_resistances: List[float],
        sl_supports: List[float],
    ) -> tuple[str, float, float]:
        """Compute trade side, TP and SL levels based on direction."""

        def _buffers(distance_pct: float) -> tuple[float, float]:
            """Dynamic buffers: tighter when distance is small, slightly wider if further away."""
            # distance_pct is in [0, 1] fraction (e.g. 0.02 for 2%)
            # Base buffers ~0.15%, tighten to 0.10% if close; allow up to 0.20% when far
            if distance_pct <= 0.01:
                adj = 0.0005
            elif distance_pct <= 0.02:
                adj = 0.0008  # Slightly tighter than before (was 0.0010)
            elif distance_pct >= 0.035:
                adj = 0.0015  # Slightly tighter than before (was 0.0020)
            else:
                # linear interpolate between 0.08% and 0.15%
                # 0.02 -> 0.0008, 0.035 -> 0.0015
                slope = (0.0015 - 0.0008) / (0.035 - 0.02)
                adj = 0.0008 + slope * (distance_pct - 0.02)
            # Use same buffer for SL/TP to keep simple and stable
            return adj, adj

        candidates: List[tuple[float, str, float, float, float]] = []

        if direction == MultiTimeframeDirection.BULLISH:
            side = "Long"
            raw_tp = [r for r in tp_resistances if r > mid]
            raw_sl = [s for s in sl_supports if s < mid]
            # Transformations to place orders inside the level
            make_tp = lambda lvl, buf: lvl * (1 - buf)
            make_sl = lambda lvl, buf: lvl * (1 - buf)
            valid_pair = lambda tp, sl: tp > mid and sl < mid
            # Distance and RR components
            tp_dist_unbuffered = lambda lvl: abs((lvl - mid) / mid)
            tp_pct = lambda tp: abs((tp - mid) / mid)
            sl_pct = lambda sl: abs((mid - sl) / mid)
        else:
            side = "Short"
            raw_tp = [s for s in tp_supports if s < mid]
            raw_sl = [r for r in sl_resistances if r > mid]
            make_tp = lambda lvl, buf: lvl * (1 + buf)
            make_sl = lambda lvl, buf: lvl * (1 + buf)
            valid_pair = lambda tp, sl: tp < mid and sl > mid
            tp_dist_unbuffered = lambda lvl: abs((mid - lvl) / mid)
            tp_pct = lambda tp: abs((mid - tp) / mid)
            sl_pct = lambda sl: abs((sl - mid) / mid)

        for tp_level in raw_tp:
            dist_tp_pct = tp_dist_unbuffered(tp_level)
            sl_buf, tp_buf = _buffers(dist_tp_pct)
            tp = make_tp(tp_level, tp_buf)
            for sl_level in raw_sl:
                sl = make_sl(sl_level, sl_buf)
                if not valid_pair(tp, sl):
                    continue
                tp_p = tp_pct(tp)
                sl_p = sl_pct(sl)
                if sl_p == 0:
                    continue
                rr = tp_p / sl_p
                candidates.append((rr, side, tp, sl, tp_p))

        if not candidates:
            logger.info(
                f"Skipping trade suggestion for {coin}: no candidate pairs for {direction.name} direction"
            )
            raise ValueError("no_valid_pair")

        windowed = [c for c in candidates if c[0] >= min_rr]
        if not windowed:
            logger.info(
                f"Skipping trade suggestion for {coin}: all {len(candidates)} candidates below RR {min_rr:.2f}"
            )
            raise ValueError("no_valid_pair")

        selected = min(windowed, key=lambda c: c[4])
        _, side_out, tp_out, sl_out, _ = selected
        return side_out, tp_out, sl_out

    def _validate_and_format_trade(
        coin: str, side: str, entry: float, tp: float, sl: float, timeframe: Timeframe
    ) -> Optional[str]:
        """Validate a computed trade and return formatted message, logging all rejection reasons."""

        if side == "Long" and (tp <= entry or sl >= entry):
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: Invalid long levels (entry={entry:.4f}, tp={tp:.4f}, sl={sl:.4f})"
            )
            return None
        if side == "Short" and (tp >= entry or sl <= entry):
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: Invalid short levels (entry={entry:.4f}, tp={tp:.4f}, sl={sl:.4f})"
            )
            return None

        if entry == 0:
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: Entry price is zero"
            )
            return None

        tp_pct = abs((tp - entry) / entry) * 100
        sl_pct = abs((sl - entry) / entry) * 100
        if sl_pct == 0:
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: Stop loss distance is zero (entry={entry:.4f}, sl={sl:.4f})"
            )
            return None

        rr = tp_pct / sl_pct
        if rr < min_rr:
            logger.info(
                f"Skipping trade suggestion for {coin} {timeframe.name}: R:R too low (RR={rr:.2f} < {min_rr:.2f}, tp%={tp_pct:.2f}, sl%={sl_pct:.2f})"
            )
            return None

        logger.info(
            f"Accepted trade suggestion for {coin} {timeframe.name}: "
            f"tp%={tp_pct:.2f}, sl%={sl_pct:.2f}, RR={rr:.2f}"
        )

        enc_side = "L" if side == "Long" else "S"
        enc_trade = base64.b64encode(
            f"{enc_side}_{coin}_{fmt_price(sl)}_{fmt_price(tp)}".encode("utf-8")
        ).decode("utf-8")
        trade_link = (
            f" ({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})"
            if exchange_enabled
            else ""
        )

        return (
            f"<b>💰 {side} trade for {coin}</b>{trade_link}<b>:</b>\n"
            f"Market price: {fmt_price(mid)} USDC\n"
            f"Stop loss: {fmt_price(sl)} USDC (-{sl_pct:.1f}%)\n"
            f"Take profit: {fmt_price(tp)} USDC (+{tp_pct:.1f}%)"
        )

    if direction == MultiTimeframeDirection.NEUTRAL:
        logger.info(f"Skipping trade suggestion for {coin}: direction neutral")
        return None

    # Prevent negative or zero values in calculations
    if mid <= 0:
        logger.info(f"Skipping trade suggestion for {coin}: invalid mid price {mid}")
        return None

    # Distance bands 
    min_distance_sl = mid * 0.0125
    max_distance_sl = mid * 0.03

    min_distance_tp = mid * 0.0125
    max_distance_tp = mid * 0.035

    # Evaluate across timeframes starting from the shortest and return the first valid suggestion
    timeframes_order = [
        Timeframe.MINUTES_15,
        Timeframe.MINUTES_30,
        Timeframe.HOUR_1,
        Timeframe.HOURS_4
    ]

    def _search_levels(
        sl_min: float, sl_max: float, tp_min: float, tp_max: float
    ) -> Optional[str]:
        for timeframe in timeframes_order:
            tp_resistances, tp_supports, sl_resistances, sl_supports = get_valid_levels(
                coin, timeframe, sl_min, sl_max, tp_min, tp_max
            )

            try:
                side, tp, sl = get_trade_levels(
                    direction, tp_resistances, tp_supports, sl_resistances, sl_supports
                )
            except ValueError:
                logger.info(
                    f"Skipping trade suggestion for {coin} {timeframe.name}: no valid level pair"
                )
                continue

            message = _validate_and_format_trade(coin, side, mid, tp, sl, timeframe)
            if message:
                return message
        return None

    best_trade = _search_levels(
        min_distance_sl, max_distance_sl, min_distance_tp, max_distance_tp
    )
    if best_trade:
        return best_trade

    logger.info(
        f"No trade suggestion produced for {coin}: exhausted timeframes without valid TP/SL pairs"
    )
    return None
