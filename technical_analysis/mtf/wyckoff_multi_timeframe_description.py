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
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
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
    )

    if not interactive_analysis:
        return base_description

    full_description = (
        f"{base_description}\n"
        f"<b>🔍 Timeframes:</b>\n"
        f"Market Context (4h-8h):\n{context_desc}\n"
        f"Daily Bias (2h):\n{long_term_desc}\n"
        f"Intraday Trend (30m-1h):\n{intermediate_desc}\n"
        f"Immediate Signals (15m):\n{short_term_desc}\n\n"
        f"<b>🎯 Signal Quality:</b>\n"
        f"• Timeframe Alignment: {alignment_pct}\n"
        f"• Confidence Level: {confidence_pct}\n\n"
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
    phases = [
        analysis.context.dominant_phase,      # Structural context
        analysis.long_term.dominant_phase,    # Main trend
        analysis.intermediate.dominant_phase, # Intraday swings
        analysis.short_term.dominant_phase    # Quick signals
    ]
    dominant_phase = max(set(phases), key=phases.count)
    phase_alignment = phases.count(dominant_phase) / len(phases)

    biases = [
        analysis.context.momentum_bias,      # Structural bias
        analysis.long_term.momentum_bias,    # Main trend bias
        analysis.intermediate.momentum_bias, # Intraday bias
        analysis.short_term.momentum_bias    # Quick signals bias
    ]
    dominant_bias = max(set(biases), key=biases.count)
    bias_alignment = biases.count(dominant_bias) / len(biases)

    # Check for time-horizon conflicts
    short_term_conflict = (analysis.short_term.momentum_bias != 
                          analysis.intermediate.momentum_bias)
    trend_conflict = (analysis.long_term.momentum_bias != 
                     analysis.intermediate.momentum_bias)
    structural_conflict = (analysis.context.momentum_bias != 
                          analysis.long_term.momentum_bias)

    if phase_alignment > 0.75 and bias_alignment > 0.75:
        # Strong alignment across all timeframes
        return f"strong {dominant_phase.value} structure with {dominant_bias.value} momentum"
    elif short_term_conflict and trend_conflict:
        # Significant divergence across timeframes
        return "mixed signals across multiple timeframes - transition likely"
    elif structural_conflict:
        # Potential major trend change
        return f"possible trend reversal - {analysis.short_term.dominant_phase.value} forming"
    elif short_term_conflict:
        # Short-term deviation
        return f"{analysis.long_term.dominant_phase.value} with short-term {analysis.short_term.momentum_bias.value} move"
    else:
        # Default case
        return f"developing {dominant_phase.value} structure with mixed momentum"

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
        
        # Use the same market structure determination as the top analysis
        market_structure = _get_full_market_structure(analysis)
        
        # Determine momentum level using the same thresholds as _calculate_momentum_strength
        momentum_desc = ""
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
        
        # Check intraday alignment - focus on short and intermediate timeframes
        intraday_aligned = immediate_bias == intraday_bias
        
        # Generate specific intraday trading signal
        signal_prefix = f"{momentum_desc.capitalize()} momentum "
        
        if intraday_aligned:
            if intraday_bias == MultiTimeframeDirection.BULLISH:
                signal_direction = "bullish intraday alignment offering favorable long opportunities. "
            elif intraday_bias == MultiTimeframeDirection.BEARISH:
                signal_direction = "bearish intraday alignment favoring short positions. "
            else:
                signal_direction = "neutral price action across trading timeframes. "
        else:
            if immediate_bias == MultiTimeframeDirection.BULLISH:
                signal_direction = "bullish short-term momentum against " + \
                                 f"{'bullish' if analysis.long_term.momentum_bias == MultiTimeframeDirection.BULLISH else 'bearish'} " + \
                                 "larger trend. "
            elif immediate_bias == MultiTimeframeDirection.BEARISH:
                signal_direction = "bearish short-term momentum against " + \
                                 f"{'bullish' if analysis.long_term.momentum_bias == MultiTimeframeDirection.BULLISH else 'bearish'} " + \
                                 "larger trend. "
            else:
                signal_direction = "consolidating short-term action within the broader trend. "
        
        return (
            f"{signal_prefix}with {signal_direction}"
            f"Market structure is {market_structure}. "
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
    def get_valid_levels(timeframe: Timeframe, min_dist: float, max_dist: float) -> tuple[List[float], List[float]]:
        """Get valid support and resistance levels for a specific timeframe."""
        if timeframe not in significant_levels:
            return ([], [])
        return (
            [r for r in significant_levels[timeframe]['resistance'] if min_dist < abs(r - mid) < max_dist],
            [s for s in significant_levels[timeframe]['support'] if min_dist < abs(s - mid) < max_dist]
        )

    def get_trade_levels(direction: MultiTimeframeDirection, resistances: List[float], supports: List[float]) -> tuple[Optional[str], Optional[float], Optional[float]]:
        """Get trade type, take profit and stop loss levels based on direction."""
        # Safety check for empty lists
        if not resistances or not supports:
            return None, None, None
            
        buffer_pct = 0.0015
        
        if direction == MultiTimeframeDirection.BULLISH:
            # For longs: TP slightly above resistance, SL slightly below support
            closest_resistance = min(resistances, key=lambda x: abs(x - mid))
            closest_support = max(supports, key=lambda x: abs(x - mid))
            tp = closest_resistance
            sl = closest_support * (1 - buffer_pct)     # SL slightly below support
            return "Long", tp, sl
        
        # For shorts: TP slightly below support, SL slightly above resistance
        closest_support = max(supports, key=lambda x: abs(x - mid))
        closest_resistance = min(resistances, key=lambda x: abs(x - mid))
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
            f"Stop Loss: {fmt_price(sl)} USDC (-{sl_pct:.1f}%)\n"
            f"Take Profit: {fmt_price(tp)} USDC (+{tp_pct:.1f}%)"
        )

    if direction == MultiTimeframeDirection.NEUTRAL:
        return None

    # Prevent negative or zero values in calculations
    if mid <= 0:
        return None

    min_distance = mid * 0.015
    max_distance = mid * 0.05

    # Try with 30min timeframe first, then 1h if needed
    for timeframe in [Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]:
        resistances, supports = get_valid_levels(timeframe, min_distance, max_distance)
        if resistances and supports:
            side, tp, sl = get_trade_levels(direction, resistances, supports)
            if side and tp and sl:  # Add safety check for None values
                formatted_trade = format_trade(coin, side, mid, tp, sl)
                if formatted_trade:
                    return formatted_trade

    return None
