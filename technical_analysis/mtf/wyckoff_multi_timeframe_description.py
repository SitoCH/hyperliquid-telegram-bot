from enum import Enum
from typing import Dict, List, Optional, Tuple, TypeVar, Any, Callable
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]
from utils import fmt_price
from telegram_utils import telegram_utils
import base64
from utils import exchange_enabled

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, SignificantLevelsData,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
    AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM, MODERATE_VOLUME_THRESHOLD, STRONG_VOLUME_THRESHOLD,
    MIXED_MOMENTUM, LOW_MOMENTUM
)

# Constants for common thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.7
MODERATE_CONFIDENCE_THRESHOLD = 0.5

HIGH_ALIGNMENT_THRESHOLD = 0.75
MODERATE_ALIGNMENT_THRESHOLD = 0.6

HIGH_FUNDING_THRESHOLD = 0.5
EXTREME_FUNDING_THRESHOLD = 0.7

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
    
    momentum = _calculate_momentum_strength(analysis)
    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    long_term_desc = _get_timeframe_trend_description(analysis.long_term)
    context_desc = _get_timeframe_trend_description(analysis.context)

    structure = _get_full_market_structure(analysis)
    trend = _generate_trend_description(analysis)
    sentiment = _analyze_market_sentiment(analysis)
    
    emoji = _get_trend_emoji_all_timeframes(analysis)
    insight = _generate_actionable_insight_all_timeframes(analysis)

    base_description = (
        f"{emoji} <b>Market Analysis:</b>\n"
        f"Trend: {trend}\n"
        f"Market structure: {structure}\n"
        f"Momentum: {momentum}\n"
        f"Sentiment: {sentiment}\n"
        f"Confidence: {confidence_pct}\n"
    )

    if not interactive_analysis:
        return base_description

    full_description = (
        f"{base_description}\n"
        f"<b>🔍 Timeframes:</b>\n"
        f"Timeframe alignment: {alignment_pct}\n"
        f"Market Context (4h-8h):\n{context_desc}\n"
        f"Daily Bias (2h):\n{long_term_desc}\n"
        f"Intraday Trend (30m-1h):\n{intermediate_desc}\n"
        f"Immediate Signals (5m-15m):\n{short_term_desc}\n\n"
        f"{insight}"
    )

    trade_suggestion = _get_trade_suggestion(coin, analysis.overall_direction, mid, significant_levels)
    if trade_suggestion:
        full_description += f"\n\n{trade_suggestion}"

    return full_description

def _calculate_momentum_strength(analysis: AllTimeframesAnalysis) -> str:
    """Calculate and describe momentum in plain English."""
    direction = analysis.overall_direction
    
    # Use match statement for cleaner momentum description
    if analysis.momentum_intensity > STRONG_MOMENTUM:
        return ("strong buying pressure across all timeframes" if direction == MultiTimeframeDirection.BULLISH 
                else "strong selling pressure across all timeframes")
    elif analysis.momentum_intensity > MODERATE_MOMENTUM:
        return ("steady accumulation with increasing volume" if direction == MultiTimeframeDirection.BULLISH 
                else "sustained distribution with good volume")
    elif analysis.momentum_intensity > WEAK_MOMENTUM:
        return ("moderate upward pressure" if direction == MultiTimeframeDirection.BULLISH 
                else "moderate downward pressure")
    elif analysis.momentum_intensity > MIXED_MOMENTUM:
        return "mixed momentum with no clear direction"
    elif analysis.momentum_intensity > LOW_MOMENTUM:
        return "low momentum, possible consolidation phase"
    return "very low momentum, market likely ranging"

def _analyze_market_sentiment(analysis: AllTimeframesAnalysis) -> str:
    """Analyze overall market sentiment in clear terms."""
    signals = []
    
    # Analyze funding rates
    avg_funding = (
        analysis.short_term.funding_sentiment +
        analysis.intermediate.funding_sentiment +
        analysis.long_term.funding_sentiment
    ) / 3
    
    # Funding rate analysis
    if abs(avg_funding) > 0.7:
        if avg_funding > 0:
            signals.append("extremely high funding rates suggesting potential long squeeze")
        else:
            signals.append("very negative funding rates suggesting potential short squeeze")
    elif abs(avg_funding) > 0.4:
        if avg_funding > 0:
            signals.append("longs paying high premiums")
        else:
            signals.append("shorts paying high premiums")
    
    # Volatility analysis with timeframe context
    if all(group.volatility_state == VolatilityState.HIGH 
           for group in [analysis.short_term, analysis.intermediate]):
        if analysis.long_term.volatility_state == VolatilityState.HIGH:
            signals.append("extreme volatility across all timeframes")
        else:
            signals.append("high short-term volatility, choppy conditions")
    
    # Volume context
    volume_signals = []
    for group in [analysis.short_term, analysis.intermediate, analysis.long_term]:
        if group.volume_strength > 0.8:
            volume_signals.append(1)
        elif group.volume_strength < 0.3:
            volume_signals.append(-1)
        else:
            volume_signals.append(0)
    
    avg_volume = sum(volume_signals) / len(volume_signals)
    if abs(avg_volume) > 0.5:
        if avg_volume > 0:
            signals.append("strong volume confirming moves")
        else:
            signals.append("low volume suggesting weak conviction")
    
    # Combine all signals into a readable format
    if not signals:
        return "neutral market conditions"
    
    return ", ".join(signals)

def _get_full_market_structure(analysis: AllTimeframesAnalysis) -> str:
    """Get comprehensive market structure description across four timeframes."""
    # Extract timeframes and weights
    phases = [
        analysis.context.dominant_phase,      
        analysis.long_term.dominant_phase,    
        analysis.intermediate.dominant_phase, 
        analysis.short_term.dominant_phase    
    ]
    
    weights = [
        analysis.context.group_weight,
        analysis.long_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.short_term.group_weight
    ]
    
    confidences = [
        not analysis.context.uncertain_phase,
        not analysis.long_term.uncertain_phase,
        not analysis.intermediate.uncertain_phase,
        not analysis.short_term.uncertain_phase
    ]
    
    # Helper function to calculate weighted phase/bias contributions
    def calculate_weighted_contributions(items, confidence_factor=0.6):
        results = {}
        for item, weight, is_confident in zip(items, weights, confidences):
            factor = 1.0 if is_confident else confidence_factor
            if item not in results:
                results[item] = 0.0
            results[item] += weight * factor
        return results
    
    # Calculate phase and bias weights with the helper function
    phase_weights = calculate_weighted_contributions(phases)
    
    biases = [
        analysis.context.momentum_bias,      
        analysis.long_term.momentum_bias,    
        analysis.intermediate.momentum_bias, 
        analysis.short_term.momentum_bias    
    ]
    
    # Use slightly higher confidence factor for biases
    bias_weights = calculate_weighted_contributions(biases, 0.7)
    
    # Find dominant values
    total_weight = sum(weights)
    dominant_phase = max(phase_weights.keys(), key=lambda p: phase_weights[p]) if phase_weights else WyckoffPhase.UNKNOWN
    dominant_bias = max(bias_weights.keys(), key=lambda p: bias_weights[p]) if bias_weights else MultiTimeframeDirection.NEUTRAL
    
    # Calculate alignments
    phase_alignment = phase_weights.get(dominant_phase, 0) / total_weight if total_weight > 0 else 0
    bias_alignment = bias_weights.get(dominant_bias, 0) / total_weight if total_weight > 0 else 0
    
    # Calculate timeframe conflicts - streamline to focus only on what's needed
    short_term_conflict = analysis.short_term.momentum_bias != analysis.intermediate.momentum_bias
    trend_conflict = analysis.long_term.momentum_bias != analysis.intermediate.momentum_bias
    structural_conflict = analysis.context.momentum_bias != analysis.long_term.momentum_bias
    
    # Calculate volume context - use weighted approach focused on shorter timeframes
    avg_volume_strength = (
        analysis.short_term.volume_strength * 0.5 +
        analysis.intermediate.volume_strength * 0.3 +
        analysis.long_term.volume_strength * 0.2
    )
    
    volume_context = ""
    if avg_volume_strength > STRONG_VOLUME_THRESHOLD:
        volume_context = " with strong volume"
    elif avg_volume_strength < MODERATE_VOLUME_THRESHOLD:
        volume_context = " on thin volume"
    
    # Dynamically adjust alignment threshold based on volatility
    volatility_adjustment = 0.1 if analysis.short_term.volatility_state == VolatilityState.HIGH else 0
    alignment_threshold = 0.7 - volatility_adjustment
    
    # Check conditions in order of priority for crypto trading
    
    # 1. High volatility conditions - most important for intraday
    if (analysis.short_term.volatility_state == VolatilityState.HIGH and 
        analysis.intermediate.volatility_state == VolatilityState.HIGH and
        analysis.short_term.volume_strength > STRONG_VOLUME_THRESHOLD):
        return f"volatile {analysis.short_term.dominant_phase.value} structure with {analysis.short_term.momentum_bias.value} momentum{volume_context}"
    
    # 2. Potential breakout - critical for trading opportunities
    short_term_breakout = (
        analysis.short_term.momentum_bias != MultiTimeframeDirection.NEUTRAL and
        analysis.short_term.volume_strength > STRONG_VOLUME_THRESHOLD and
        not analysis.short_term.uncertain_phase
    )
    
    if short_term_breakout and short_term_conflict:
        return f"{analysis.intermediate.dominant_phase.value} structure with potential {analysis.short_term.momentum_bias.value} breakout{volume_context}"
    
    # 3. Strong alignment - clear market structure
    if phase_alignment > alignment_threshold and bias_alignment > alignment_threshold:
        confidence = "strong" if phase_alignment > 0.8 else "clear"
        return f"{confidence} {dominant_phase.value} structure with {dominant_bias.value} momentum{volume_context}"
    
    # 4. Multiple timeframe conflicts - transition phase
    if short_term_conflict and trend_conflict:
        if analysis.short_term.volatility_state == VolatilityState.HIGH:
            return f"choppy market with mixed signals across timeframes{volume_context}"
        return f"mixed signals across timeframes - transition phase{volume_context}"
    
    # 5. Major structural conflict - potential reversal 
    if structural_conflict:
        confidence = "" if analysis.short_term.uncertain_phase else "confirmed "
        return f"possible trend reversal - {confidence}{analysis.short_term.dominant_phase.value} forming{volume_context}"
    
    # 6. Short term deviation - important for intraday 
    if short_term_conflict:
        st_bias = analysis.short_term.momentum_bias.value
        lt_phase = analysis.long_term.dominant_phase.value
        return f"{lt_phase} structure with {st_bias} short-term move{volume_context}"
    
    # 7. Default case - balanced description
    conf_level = "developing" if phase_alignment < 0.6 else "established"
    momentum_desc = dominant_bias.value if bias_alignment > 0.6 else "mixed"
    return f"{conf_level} {dominant_phase.value} structure with {momentum_desc} momentum{volume_context}"

def _generate_trend_description(analysis: AllTimeframesAnalysis) -> str:
    """Generate a coherent trend description combining strength and context."""
    # Get the trend strength
    strength = _determine_trend_strength(analysis)
    
    # Handle neutral direction
    if analysis.overall_direction == MultiTimeframeDirection.NEUTRAL:
        if analysis.momentum_intensity < LOW_MOMENTUM:
            return f"{strength} ranging market with minimal directional movement"
        else:
            return f"{strength} consolidation phase"
    
    # Get direction string for bullish/bearish
    direction = analysis.overall_direction.value + " "
    
    # Calculate volume and conviction
    volume_strength = _calculate_weighted_volume_strength(analysis)
    
    # Combine everything into a coherent description
    volume_desc = ""
    if volume_strength > STRONG_VOLUME_THRESHOLD:
        volume_desc = "with high volume"
    elif volume_strength < MODERATE_VOLUME_THRESHOLD:
        volume_desc = "with low volume"
    
    conviction = ""
    if analysis.confidence_level > HIGH_CONFIDENCE_THRESHOLD:
        conviction = "high-conviction "
    elif analysis.confidence_level > MODERATE_CONFIDENCE_THRESHOLD:
        conviction = "established "
    else:
        conviction = "developing "
    
    return f"{strength} {conviction}{direction}trend {volume_desc}".strip()

def _determine_market_context(analysis: AllTimeframesAnalysis) -> str:
    """Determine overall market context considering three timeframes."""
    volume_strength = _calculate_weighted_volume_strength(analysis)
    
    if analysis.overall_direction == MultiTimeframeDirection.NEUTRAL:
        if volume_strength > STRONG_VOLUME_THRESHOLD:
            return "high volume ranging market"
        return "low volume consolidation"

    context = analysis.overall_direction.value
    if volume_strength > STRONG_VOLUME_THRESHOLD and analysis.confidence_level > HIGH_CONFIDENCE_THRESHOLD:
        return f"high-conviction {context} trend"
    elif volume_strength > MODERATE_VOLUME_THRESHOLD and analysis.confidence_level > MODERATE_CONFIDENCE_THRESHOLD:
        return f"established {context} trend"
    
    return f"developing {context} bias"

def _determine_trend_strength(analysis: AllTimeframesAnalysis) -> str:
    """Determine overall trend strength considering three timeframes."""
    # Calculate weighted alignment
    values = [
        analysis.short_term.internal_alignment,
        analysis.intermediate.internal_alignment,
        analysis.long_term.internal_alignment
    ]
    weights = _get_timeframe_weights(analysis)
    
    weighted_alignment = _weighted_average(values, weights)
        
    if weighted_alignment > 0.85:
        return "extremely strong"
    elif weighted_alignment > 0.7:
        return "very strong"
    elif weighted_alignment > 0.5:
        return "strong"
    elif weighted_alignment > 0.3:
        return "moderate"
    
    return "weak"

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
        
        # Determine momentum level using the same thresholds as _calculate_momentum_strength
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

def _get_trade_suggestion(coin: str, direction: MultiTimeframeDirection, mid: float, significant_levels: Dict[Timeframe, SignificantLevelsData]) -> Optional[str]:
    """Generate trade suggestion with stop loss and take profit based on nearby levels."""
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
                         sl_supports: List[float]) -> tuple[Optional[str], Optional[float], Optional[float]]:
        """Get trade type, take profit and stop loss levels based on direction."""
        # Safety check for empty lists
        if ((direction == MultiTimeframeDirection.BULLISH and (not tp_resistances or not sl_supports)) or
            (direction == MultiTimeframeDirection.BEARISH and (not tp_supports or not sl_resistances))):
            return None, None, None
            
        buffer_pct = 0.0015
        
        if direction == MultiTimeframeDirection.BULLISH:
            # For longs: TP slightly above resistance, SL slightly below support
            closest_resistance = min(tp_resistances, key=lambda x: abs(x - mid))
            closest_support = max(sl_supports, key=lambda x: abs(x - mid))
            tp = closest_resistance
            sl = closest_support * (1 - buffer_pct)     # SL slightly below support
            return "Long", tp, sl
        
        # For shorts: TP slightly below support, SL slightly above resistance
        closest_support = max(tp_supports, key=lambda x: abs(x - mid))
        closest_resistance = min(sl_resistances, key=lambda x: abs(x - mid))
        tp = closest_support
        sl = closest_resistance * (1 + buffer_pct)  # SL slightly above resistance
        return "Short", tp, sl

    def format_trade(coin: str, side: str, entry: float, tp: float, sl: float) -> Optional[str]:
        """Format trade suggestion with consistent calculations and layout."""
        if (side == "Long" and (tp <= entry or sl >= entry)) or \
           (side == "Short" and (tp >= entry or sl <= entry)):
            return None
            
        # Add safety checks to prevent division by zero
        if entry == 0:
            return None
            
        tp_pct = abs((tp - entry) / entry) * 100
        sl_pct = abs((sl - entry) / entry) * 100
        
        enc_side = "L" if side == "Long" else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(sl)}_{fmt_price(tp)}".encode('utf-8')).decode('utf-8')
        trade_link = f"({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

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

    min_distance_sl = mid * 0.0175
    max_distance_sl = mid * 0.06
    
    min_distance_tp = mid * 0.015
    max_distance_tp = mid * 0.05

    for timeframe in [Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.MINUTES_15, Timeframe.HOURS_4]:
        tp_resistances, tp_supports, sl_resistances, sl_supports = get_valid_levels(
            timeframe, min_distance_sl, max_distance_sl, min_distance_tp, max_distance_tp
        )
        
        if ((direction == MultiTimeframeDirection.BULLISH and tp_resistances and sl_supports) or
            (direction == MultiTimeframeDirection.BEARISH and tp_supports and sl_resistances)):
            side, tp, sl = get_trade_levels(direction, tp_resistances, tp_supports, sl_resistances, sl_supports)
            if side and tp and sl:  # Add safety check for None values
                formatted_trade = format_trade(coin, side, mid, tp, sl)
                if formatted_trade:
                    return formatted_trade

    return None
