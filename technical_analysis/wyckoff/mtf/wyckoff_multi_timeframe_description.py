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
        (MultiTimeframeDirection.BULLISH, True): "📈",   # Strong bullish
        (MultiTimeframeDirection.BULLISH, False): "↗️" if analysis.momentum_intensity > WEAK_MOMENTUM else "➡️⬆️",
        (MultiTimeframeDirection.BEARISH, True): "📉",   # Strong bearish
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
    if analysis.confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals across timeframes.\n<b>Recommendation:</b>\nReduce exposure and wait for clearer setups."

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
        
        # Volume qualifier for more precise signal description
        volume_qualifier = ""
        if avg_volume > STRONG_VOLUME_THRESHOLD:
            volume_qualifier = "high-volume "
        elif avg_volume < MODERATE_VOLUME_THRESHOLD:
            volume_qualifier = "thin-volume "
        
        # Check intraday alignment - focus on short and intermediate timeframes
        intraday_aligned = immediate_bias == intraday_bias
        
        # Generate improved timeframe-specific momentum prefix
        signal_prefix = f"{momentum_desc.capitalize()} {volume_qualifier}momentum "
        
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
                    signal_direction += " with significant volume"
                signal_direction += ". "
                
            else:
                signal_direction = (
                    f"neutral price action across key timeframes suggesting range-bound conditions. "
                    f"Watch for breakout triggers with increasing volume. "
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
                        f"Watch for breakout direction with increasing volume. "
                    )
                else:
                    # General timeframe conflict
                    signal_direction = (
                        f"{immediate_bias.value} short-term momentum not yet confirmed on higher timeframes. "
                        f"{'Consider quick scalps only until alignment improves.' if avg_volume > MODERATE_VOLUME_THRESHOLD else 'Avoid chasing moves until alignment improves.'} "
                    )

        return (
            f"{signal_prefix}with {signal_direction}"
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
        sl_buffer_pct = 0.0015
        tp_buffer_pct = 0.0015
        
        if direction == MultiTimeframeDirection.BULLISH:
            # R:R-optimal: TP farthest resistance, SL nearest support
            farthest_resistance = max(tp_resistances, key=lambda x: abs(x - mid))
            nearest_support = min(sl_supports, key=lambda x: abs(x - mid))
            tp = farthest_resistance * (1 - tp_buffer_pct)   # Slightly inside resistance
            sl = nearest_support * (1 - sl_buffer_pct)       # Slightly below support
            return "Long", tp, sl
        
        # For shorts: R:R-optimal - TP farthest support, SL nearest resistance
        farthest_support = max(tp_supports, key=lambda x: abs(x - mid))
        nearest_resistance = min(sl_resistances, key=lambda x: abs(x - mid))
        tp = farthest_support * (1 + tp_buffer_pct)     # Slightly inside support
        sl = nearest_resistance * (1 + sl_buffer_pct)   # Slightly above resistance
        return "Short", tp, sl

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

        min_rr = 1.1
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
    # SL: 1.5%–3.25%, TP: 1.75%–4%
    min_distance_sl = mid * 0.015
    max_distance_sl = mid * 0.0325
    
    min_distance_tp = mid * 0.0175
    max_distance_tp = mid * 0.04

    for timeframe in [Timeframe.HOUR_1, Timeframe.MINUTES_30, Timeframe.MINUTES_15, Timeframe.HOURS_4, Timeframe.HOURS_8]:
        tp_resistances, tp_supports, sl_resistances, sl_supports = get_valid_levels(
            timeframe, min_distance_sl, max_distance_sl, min_distance_tp, max_distance_tp
        )
        
        if ((direction == MultiTimeframeDirection.BULLISH and tp_resistances and sl_supports) or
            (direction == MultiTimeframeDirection.BEARISH and tp_supports and sl_resistances)):
            side, tp, sl = get_trade_levels(direction, tp_resistances, tp_supports, sl_resistances, sl_supports)
            formatted_trade = _validate_and_format_trade(coin, side, mid, tp, sl, timeframe)
            if formatted_trade:
                return formatted_trade

    return None
