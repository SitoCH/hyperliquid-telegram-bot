from typing import Dict, List, Optional, TypeVar
from utils import fmt_price
from telegram_utils import telegram_utils
import base64
from utils import exchange_enabled
import os
from logging_utils import logger

from ..wyckoff_types import (
    WyckoffPhase, WyckoffSign, SignificantLevelsData,
    CompositeAction, Timeframe, VolatilityState
)

from .wyckoff_multi_timeframe_types import (
    AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM, MODERATE_VOLUME_THRESHOLD, STRONG_VOLUME_THRESHOLD,
    MIXED_MOMENTUM, LOW_MOMENTUM
)

# Constants for common thresholds
MODERATE_CONFIDENCE_THRESHOLD = 0.5
MODERATE_ALIGNMENT_THRESHOLD = 0.6

# Minimum acceptable Risk:Reward for proposed trades
MIN_RR = 1.1

T = TypeVar('T')

def _weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average of values with corresponding weights."""
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight

def _get_timeframe_weights(analysis: AllTimeframesAnalysis) -> List[float]:
    """Return standard timeframe weights for consistency."""
    return [
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]

def _get_volume_description(volume_strength: float) -> str:
    """Get standardized volume description based on strength."""
    if volume_strength > STRONG_VOLUME_THRESHOLD:
        return "strong volume"
    elif volume_strength > MODERATE_VOLUME_THRESHOLD:
        return "moderate volume"
    return "light volume"

def _get_sign_description(sign: WyckoffSign) -> str:
    """
    Get a proper readable description for a Wyckoff Sign instead of the raw enum value.
    """
    if sign == WyckoffSign.NONE:
        return ""

    descriptions = {
        WyckoffSign.SELLING_CLIMAX: "selling climax",
        WyckoffSign.AUTOMATIC_RALLY: "automatic rally",
        WyckoffSign.SECONDARY_TEST: "secondary test",
        WyckoffSign.LAST_POINT_OF_SUPPORT: "last point of support",
        WyckoffSign.SIGN_OF_STRENGTH: "sign of strength",
        WyckoffSign.BUYING_CLIMAX: "buying climax",
        WyckoffSign.UPTHRUST: "upthrust",
        WyckoffSign.SECONDARY_TEST_RESISTANCE: "secondary test resistance",
        WyckoffSign.LAST_POINT_OF_RESISTANCE: "last point of resistance",
        WyckoffSign.SIGN_OF_WEAKNESS: "sign of weakness"
    }
    
    return f" | {descriptions.get(sign, sign.value.lower())}"

def _calculate_weighted_volume_strength(analysis: AllTimeframesAnalysis) -> float:
    """Calculate weighted volume strength across timeframes."""
    weights = _get_timeframe_weights(analysis)
    values = [
        analysis.short_term.volume_strength,
        analysis.intermediate.volume_strength,
        analysis.long_term.volume_strength
    ]
    return _weighted_average(values, weights)

def generate_all_timeframes_description(coin: str, analysis: AllTimeframesAnalysis, mid: float, significant_levels: Dict[Timeframe, SignificantLevelsData], interactive_analysis: bool) -> str:
    """Generate comprehensive description including four timeframe groups."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"
    
    # Get phase, volume, sign and action info for short and intermediate timeframes
    short_phase = analysis.short_term.dominant_phase.value
    if analysis.short_term.uncertain_phase and analysis.short_term.dominant_phase != WyckoffPhase.UNKNOWN:
        short_phase = f"~{short_phase}"
        
    interm_phase = analysis.intermediate.dominant_phase.value
    if analysis.intermediate.uncertain_phase and analysis.intermediate.dominant_phase != WyckoffPhase.UNKNOWN:
        interm_phase = f"~{interm_phase}"

    short_volume = _get_volume_description(analysis.short_term.volume_strength)
    interm_volume = _get_volume_description(analysis.intermediate.volume_strength)

    short_sign = _get_sign_description(analysis.short_term.dominant_sign)
    interm_sign = _get_sign_description(analysis.intermediate.dominant_sign)
    
    short_action = "" if analysis.short_term.dominant_action == CompositeAction.UNKNOWN else f" | {analysis.short_term.dominant_action.value}"
    interm_action = "" if analysis.intermediate.dominant_action == CompositeAction.UNKNOWN else f" | {analysis.intermediate.dominant_action.value}"

    emoji = _get_trend_emoji_all_timeframes(analysis)

    if not interactive_analysis:
        return (
            f"{emoji} <b>Market Analysis:</b>\n"
            f"Confidence: {confidence_pct}\n"
            f"Intraday Trend (30m-1h): {interm_phase} ({interm_volume}{interm_sign}{interm_action})\n"
            f"Immediate Signals (15m): {short_phase} ({short_volume}{short_sign}{short_action})\n"      )

    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    long_term_desc = _get_timeframe_trend_description(analysis.long_term)
    context_desc = _get_timeframe_trend_description(analysis.context)

    insight = _generate_actionable_insight_all_timeframes(analysis)

    full_description = (
        f"{emoji} <b>Market Analysis:</b>\n"
        f"Confidence: {confidence_pct}\n"
        f"Funding: {analysis.intermediate.funding_state.value}\n\n"
        f"<b>🔍 Timeframes:</b>\n"
        f"Timeframe alignment: {alignment_pct}\n"
        f"Market Context (4h-8h):\n{context_desc}\n"
        f"Daily Bias (2h):\n{long_term_desc}\n"
        f"Intraday Trend (30m-1h):\n{intermediate_desc}\n"
        f"Immediate Signals (15m):\n{short_term_desc}\n\n"
        f"{insight}"
    )

    trade_suggestion = _get_trade_suggestion(coin, analysis.overall_direction, mid, significant_levels, analysis.confidence_level)
    if trade_suggestion:
        full_description += f"\n\n{trade_suggestion}"

    return full_description

def _get_trend_emoji_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """Get appropriate trend emoji based on overall analysis state."""
    # First check if we have enough confidence
    if analysis.confidence_level < 0.4:
        return "📊"  # Low confidence
        
    # Get the overall trend strength using both alignment and momentum
    trend_strength = (
        analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD and 
        analysis.confidence_level > MODERATE_CONFIDENCE_THRESHOLD and 
        analysis.momentum_intensity > MODERATE_MOMENTUM
    )
    
    direction = analysis.overall_direction
    
    # Use simplified dictionary lookup for emoji selection
    emoji_map = {
        (MultiTimeframeDirection.BULLISH, True): "⬆️",   # Strong bullish
        (MultiTimeframeDirection.BULLISH, False): "↗️" if analysis.momentum_intensity > WEAK_MOMENTUM else "➡️⬆️",
        (MultiTimeframeDirection.BEARISH, True): "⬇️",   # Strong bearish
        (MultiTimeframeDirection.BEARISH, False): "↘️" if analysis.momentum_intensity > WEAK_MOMENTUM else "➡️⬇️",
    }
    
    if direction == MultiTimeframeDirection.NEUTRAL:
        # Check if we're in consolidation or in conflict
        if analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD:
            return "↔️"  # Clear consolidation
        elif analysis.momentum_intensity < LOW_MOMENTUM:
            return "🔀"  # Very low momentum, ranging market
        return "🔄"  # Mixed signals
    
    return emoji_map.get((direction, trend_strength), "📊")

def _generate_actionable_insight_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Generate intraday-focused actionable insights focused on crypto trading timeframes.
    """
    # Graded confidence messaging for clearer risk posture
    if analysis.confidence_level < 0.4:
        return (
            "<b>Analysis:</b>\nVery low confidence across timeframes.\n"
            "<b>Recommendation:</b>\nObserve only; avoid adding new risk until signals strengthen."
        )
    elif analysis.confidence_level < 0.5:
        return (
            "<b>Analysis:</b>\nLow confidence signals across timeframes.\n"
            "<b>Recommendation:</b>\nBe cautious; use reduced size and wait for alignment and volume confirmation."
        )

    def get_intraday_signal() -> str:
        """Get primary trading signal focused on intraday timeframes."""
        # Focus on intermediate timeframe for intraday trading bias (30m-1h)
        intraday_bias = analysis.intermediate.momentum_bias
        
        # Use short term for immediate direction (15m)
        immediate_bias = analysis.short_term.momentum_bias
        
        # Calculate volume context for signal strength
        avg_volume = (
            analysis.short_term.volume_strength * 0.6 +  # Higher weight for short timeframe volume
            analysis.intermediate.volume_strength * 0.4  # Lower weight for intermediate timeframe
        )

        # Alignment label to reflect how coherent the intraday picture is
        if analysis.alignment_score > 0.75:
            align_label = "Strong"
        elif analysis.alignment_score >= 0.6:
            align_label = "Moderate"
        else:
            align_label = "Mixed"

        if analysis.momentum_intensity > STRONG_MOMENTUM:
            momentum_desc = "strong"
            position_advice = "Consider aggressive intraday positions with proper risk management."
        elif analysis.momentum_intensity > MODERATE_MOMENTUM:
            momentum_desc = "steady"
            position_advice = "Favorable environment for swing positions with moderate risk exposure."
        elif analysis.momentum_intensity > WEAK_MOMENTUM:
            momentum_desc = "moderate"
            position_advice = "Use scaled entries and definitive technical triggers for entries."
        elif analysis.momentum_intensity > MIXED_MOMENTUM:
            momentum_desc = "mixed"
            position_advice = "Focus on shorter timeframe setups and reduced position sizes."
        else:
            momentum_desc = "weak"
            position_advice = "Consider range trading strategies or reduce exposure until clearer signals emerge."

        # Volatility-aware adjustments to execution guidance
        if analysis.intermediate.volatility_state == VolatilityState.HIGH:
            position_advice += " Widen stops and stagger entries; take partial profits faster."
        elif analysis.intermediate.volatility_state == VolatilityState.NORMAL:
            position_advice += " Require breakout confirmation; avoid chasing early moves."
        
        # Volume qualifier for more precise signal description
        volume_qualifier = ""
        if avg_volume > STRONG_VOLUME_THRESHOLD:
            volume_qualifier = "high volume "
        elif avg_volume < MODERATE_VOLUME_THRESHOLD:
            volume_qualifier = "light volume "
        
        # Check intraday alignment - focus on short and intermediate timeframes
        # Soft alignment: treat immediate NEUTRAL as aligned with intraday bias
        intraday_aligned = (
            immediate_bias == intraday_bias or immediate_bias == MultiTimeframeDirection.NEUTRAL
        )
        
        # Generate improved timeframe-specific momentum prefix
        signal_prefix = f"{align_label} alignment, {momentum_desc} {volume_qualifier}momentum "
        
        # Improved signal direction with more specific crypto trading language
        if intraday_aligned:
            if intraday_bias == MultiTimeframeDirection.BULLISH:
                signal_direction = (
                    f"bullish alignment across 15m-1h timeframes suggesting continuation. "
                    f"Price action shows {momentum_desc} buying interest"
                )
                if avg_volume > STRONG_VOLUME_THRESHOLD:
                    signal_direction += " with above-average volume support"
                signal_direction += ". "
                
            elif intraday_bias == MultiTimeframeDirection.BEARISH:
                signal_direction = (
                    f"bearish alignment across 15m-1h timeframes indicating continued selling pressure. "
                    f"Price structure shows {momentum_desc} selling"
                )
                if avg_volume > STRONG_VOLUME_THRESHOLD:
                    signal_direction += " with above-average volume"
                signal_direction += ". "
                
            else:
                signal_direction = (
                    f"neutral price action across key timeframes suggesting range-bound conditions. "
                    f"Trade range edges or wait for break of range high/low with rising volume. "
                )
                
        else:  # Timeframes not aligned
            # Check for potential reversal setups - common in crypto
            potential_reversal = (
                immediate_bias != MultiTimeframeDirection.NEUTRAL and
                intraday_bias != immediate_bias and 
                analysis.short_term.volume_strength > MODERATE_VOLUME_THRESHOLD
            )
            
            if potential_reversal:
                if immediate_bias == MultiTimeframeDirection.BULLISH:
                    signal_direction = (
                        f"potential bullish reversal forming on 15m against "
                        f"{'bullish' if analysis.long_term.momentum_bias == MultiTimeframeDirection.BULLISH else 'bearish'} "
                        f"larger trend. Look for confirmation on the 30m-1h timeframes before adding size. "
                    )
                else:  # BEARISH
                    signal_direction = (
                        f"potential bearish reversal forming on 15m against "
                        f"{'bullish' if analysis.long_term.momentum_bias == MultiTimeframeDirection.BULLISH else 'bearish'} "
                        f"larger trend. Wait for confirmation on the 30m-1h timeframes before committing. "
                    )
            else:
                if immediate_bias == MultiTimeframeDirection.NEUTRAL:
                    # Short-term consolidation
                    signal_direction = (
                        f"15m consolidation within the {intraday_bias.value} intermediate trend. "
                        f"Trade range edges or wait for break of range high/low with rising volume. "
                    )
                else:
                    # General timeframe conflict
                    signal_direction = (
                        f"{immediate_bias.value} short-term momentum not yet confirmed on higher timeframes. "
                        f"{'Consider quick scalps only until alignment improves.' if avg_volume > MODERATE_VOLUME_THRESHOLD else 'Avoid chasing moves until alignment improves.'} "
                    )

        # Concise Wyckoff context tail (dominant sign/action)
        context_tail = ""
        short_sign = analysis.short_term.dominant_sign
        interm_sign = analysis.intermediate.dominant_sign
        chosen_sign = short_sign if short_sign != WyckoffSign.NONE else interm_sign
        if chosen_sign != WyckoffSign.NONE:
            sign_text = _get_sign_description(chosen_sign).replace(" | ", "")
            if sign_text:
                context_tail += f" after {sign_text}"

        # Add dominant action note once if present
        if not context_tail:
            # If no sign tail, try action context; otherwise, append action after sign
            pass
        action_text = None
        if analysis.short_term.dominant_action != CompositeAction.UNKNOWN:
            action_text = analysis.short_term.dominant_action.value
        elif analysis.intermediate.dominant_action != CompositeAction.UNKNOWN:
            action_text = analysis.intermediate.dominant_action.value
        if action_text:
            if context_tail:
                context_tail += f"; setup aligns with {action_text}"
            else:
                context_tail += f" setup aligns with {action_text}"

        # Funding-aware cautioning
        # Coerce enum or string funding_state into a normalized lowercase string
        _funding_raw = getattr(analysis.intermediate.funding_state, "value", analysis.intermediate.funding_state)
        funding_val = str(_funding_raw).lower()
        if "positive" in funding_val:
            position_advice += " Crowded longs; prefer pullback entries over breakouts."
        elif "negative" in funding_val:
            position_advice += " Short-cover squeeze risk; breakout entries acceptable with tight risk."

        return (
            f"{signal_prefix}with {signal_direction.strip()}"  # ensure clean spacing
            f"{context_tail}. "
            f"{position_advice}"
        )

    # Format the complete insight
    intraday_signal = get_intraday_signal()

    return (
        f"<b>💡 Trading Insight:</b>\n{intraday_signal}"
    )

def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate enhanced trend description for a timeframe group."""
    # Add uncertainty marker if needed
    phase_desc = analysis.dominant_phase.value
    if analysis.uncertain_phase and analysis.dominant_phase != WyckoffPhase.UNKNOWN:
        phase_desc = f"~ {phase_desc}"
    
    # Format action description with validation
    action_desc = (
        analysis.dominant_action.value 
        if analysis.dominant_action != CompositeAction.UNKNOWN 
        else "no clear action"
    )
    
    volume_desc = _get_volume_description(analysis.volume_strength)
        
    volatility = ""
    if analysis.volatility_state == VolatilityState.HIGH:
        volatility = " | high volatility"
        
    return (
        f"• {phase_desc} phase {action_desc}\n"
        f"  └─ {volume_desc}{volatility}"
    )

def _get_trade_suggestion(coin: str, direction: MultiTimeframeDirection, mid: float, significant_levels: Dict[Timeframe, SignificantLevelsData], confidence: float) -> Optional[str]:
    """Generate trade suggestion with stop loss and take profit based on nearby levels."""

    min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
    if confidence < min_confidence:
        return None

    def get_valid_levels(timeframe: Timeframe, min_dist_sl: float, max_dist_sl: float, 
                         min_dist_tp: float, max_dist_tp: float) -> tuple[List[float], List[float], List[float], List[float]]:
        """Get valid support and resistance levels for a specific timeframe with separate ranges for SL and TP."""
        if timeframe not in significant_levels:
            return ([], [], [], [])
        
        # Separate levels for take profit and stop loss with different distance constraints
        tp_resistances = [r for r in significant_levels[timeframe]['resistance'] if min_dist_tp < abs(r - mid) < max_dist_tp]
        tp_supports = [s for s in significant_levels[timeframe]['support'] if min_dist_tp < abs(s - mid) < max_dist_tp]
        sl_resistances = [r for r in significant_levels[timeframe]['resistance'] if min_dist_sl < abs(r - mid) < max_dist_sl]
        sl_supports = [s for s in significant_levels[timeframe]['support'] if min_dist_sl < abs(s - mid) < max_dist_sl]
        
        return (tp_resistances, tp_supports, sl_resistances, sl_supports)

    def get_trade_levels(direction: MultiTimeframeDirection, tp_resistances: List[float], 
                         tp_supports: List[float], sl_resistances: List[float], 
                         sl_supports: List[float]) -> tuple[str, float, float]:
        """Compute trade side, TP and SL levels based on direction."""

        def _buffers(distance_pct: float) -> tuple[float, float]:
            """Dynamic buffers: tighter when distance is small, slightly wider if further away."""
            # distance_pct is in [0, 1] fraction (e.g. 0.02 for 2%)
            # Base buffers ~0.15%, tighten to 0.10% if close; allow up to 0.20% when far
            if distance_pct <= 0.02:
                adj = 0.0010
            elif distance_pct >= 0.035:
                adj = 0.0020
            else:
                # linear interpolate between 0.10% and 0.20%
                # 0.02 -> 0.0010, 0.035 -> 0.0020
                slope = (0.0020 - 0.0010) / (0.035 - 0.02)
                adj = 0.0010 + slope * (distance_pct - 0.02)
            # Use same buffer for SL/TP to keep simple and stable
            return adj, adj

        best = None  # (rr, side, tp, sl)

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
                if best is None or rr > best[0]:
                    best = (rr, side, tp, sl)

        if best is None:
            # No valid directional pair found
            raise ValueError("no_valid_pair")

        _, side, tp, sl = best
        return side, tp, sl

    def _validate_and_format_trade(coin: str, side: str, entry: float, tp: float, sl: float, timeframe: Timeframe) -> Optional[str]:
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
            logger.info(f"Skipping trade suggestion {coin} {timeframe.name}: Entry price is zero")
            return None

        tp_pct = abs((tp - entry) / entry) * 100
        sl_pct = abs((sl - entry) / entry) * 100
        if sl_pct == 0:
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: Stop loss distance is zero (entry={entry:.4f}, sl={sl:.4f})"
            )
            return None

        # Minimum R:R enforced via constant
        min_rr = MIN_RR
        rr = tp_pct / sl_pct
        if rr < min_rr:
            logger.info(
                f"Skipping trade suggestion {coin} {timeframe.name}: R:R too low (RR={rr:.2f} < {min_rr:.2f}, tp%={tp_pct:.2f}, sl%={sl_pct:.2f})"
            )
            return None

        enc_side = "L" if side == "Long" else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(sl)}_{fmt_price(tp)}".encode('utf-8')).decode('utf-8')
        trade_link = f" ({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

        return (
            f"<b>💰 {side} Trade Setup</b>{trade_link}<b>:</b>\n"
            f"Market price: {fmt_price(mid)} USDC\n"
            f"Stop Loss: {fmt_price(sl)} USDC (-{sl_pct:.1f}%)\n"
            f"Take Profit: {fmt_price(tp)} USDC (+{tp_pct:.1f}%)"
        )

    if direction == MultiTimeframeDirection.NEUTRAL:
        return None

    # Prevent negative or zero values in calculations
    if mid <= 0:
        return None

    # Distance bands tuned for healthier baseline R:R
    # SL: 1.75%–3.5%, TP: 1.75%–5%
    min_distance_sl = mid * 0.0175
    max_distance_sl = mid * 0.035
    
    min_distance_tp = mid * 0.0175
    max_distance_tp = mid * 0.05

    # Evaluate across all timeframes and pick the best R:R that passes validation
    timeframes_order = [
        Timeframe.HOUR_1, Timeframe.MINUTES_30, Timeframe.MINUTES_15, Timeframe.HOURS_4
    ]

    def _search_levels(sl_min: float, sl_max: float, tp_min: float, tp_max: float) -> Optional[str]:
        best_msg = None
        best_rr = -1.0
        for timeframe in timeframes_order:
            tp_resistances, tp_supports, sl_resistances, sl_supports = get_valid_levels(
                timeframe, sl_min, sl_max, tp_min, tp_max
            )

            # Need at least one directional candidate on each side depending on direction
            directional_possible = (
                (direction == MultiTimeframeDirection.BULLISH and any(r > mid for r in tp_resistances) and any(s < mid for s in sl_supports)) or
                (direction == MultiTimeframeDirection.BEARISH and any(s < mid for s in tp_supports) and any(r > mid for r in sl_resistances))
            )
            if not directional_possible:
                continue
            try:
                side, tp, sl = get_trade_levels(direction, tp_resistances, tp_supports, sl_resistances, sl_supports)
            except ValueError:
                continue
            # Use validator to enforce final constraints and format
            message = _validate_and_format_trade(coin, side, mid, tp, sl, timeframe)
            if not message:
                continue

            # Extract RR again for ranking
            tp_pct = abs((tp - mid) / mid) * 100
            sl_pct = abs((sl - mid) / mid) * 100
            if sl_pct == 0:
                continue
            rr = tp_pct / sl_pct
            if rr > best_rr:
                best_rr = rr
                best_msg = message
        return best_msg

    # First pass with conservative bands
    best_trade = _search_levels(min_distance_sl, max_distance_sl, min_distance_tp, max_distance_tp)
    if best_trade:
        return best_trade

    # Second pass: adaptive expansion when R:R is hard to satisfy
    expanded_min_sl = mid * 0.0125   # 1.25%
    expanded_max_sl = mid * 0.04     # 4.0%
    expanded_min_tp = mid * 0.02     # 2.0%
    expanded_max_tp = mid * 0.055    # 5.5%

    best_trade = _search_levels(expanded_min_sl, expanded_max_sl, expanded_min_tp, expanded_max_tp)
    if best_trade:
        return best_trade

    return None
