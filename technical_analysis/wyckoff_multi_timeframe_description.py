from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]
from utils import fmt_price
from telegram_utils import telegram_utils
import base64
from utils import exchange_enabled

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, SignificantLevelsData,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

from .wyckoff_multi_timeframe_types import (
    AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM,
    MIXED_MOMENTUM, LOW_MOMENTUM
)

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
    context = _determine_market_context(analysis)
    sentiment = _analyze_market_sentiment(analysis)
    
    emoji = _get_trend_emoji_all_timeframes(analysis)
    insight = _generate_actionable_insight_all_timeframes(analysis)

    base_description = (
        f"{emoji} <b>Market Analysis:</b>\n"
        f"Trend: {_determine_trend_strength(analysis)} {context}\n"
        f"Market Structure: {structure}\n"
        f"Momentum: {momentum}\n"
        f"Sentiment: {sentiment}\n"
    )

    if not interactive_analysis:
        return base_description

    full_description = (
        f"{base_description}\n"
        f"<b>🔍 Timeframes:</b>\n"
        f"Market Context (>4h):\n{context_desc}\n"
        f"Main Trend (1-4h):\n{long_term_desc}\n"
        f"Intraday (30m):\n{intermediate_desc}\n"
        f"Near-Term (15m):\n{short_term_desc}\n\n"
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
    
    # Replace basic strength descriptions with more detailed explanations
    if analysis.momentum_intensity > STRONG_MOMENTUM:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Strong buying pressure across all timeframes"
        return "Strong selling pressure across all timeframes"
    elif analysis.momentum_intensity > MODERATE_MOMENTUM:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Steady accumulation with increasing volume"
        return "Sustained distribution with good volume"
    elif analysis.momentum_intensity > WEAK_MOMENTUM:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Moderate upward pressure"
        return "Moderate downward pressure"
    elif analysis.momentum_intensity > MIXED_MOMENTUM:
        return "Mixed momentum with no clear direction"
    elif analysis.momentum_intensity > LOW_MOMENTUM:
        return "Low momentum, possible consolidation phase"
    return "Very low momentum, market likely ranging"


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
            signals.append("Extremely high funding rates suggesting potential long squeeze")
        else:
            signals.append("Very negative funding rates suggesting potential short squeeze")
    elif abs(avg_funding) > 0.4:
        if avg_funding > 0:
            signals.append("Longs paying high premiums")
        else:
            signals.append("Shorts paying high premiums")
    
    # Liquidation risk analysis
    high_risk_count = sum(
        1 for group in [analysis.short_term, analysis.intermediate, analysis.long_term]
        if group.liquidation_risk == LiquidationRisk.HIGH
    )
    
    if high_risk_count >= 2:
        signals.append("High liquidation risk, cascading liquidations possible")
    
    # Volatility analysis with timeframe context
    if all(group.volatility_state == VolatilityState.HIGH 
           for group in [analysis.short_term, analysis.intermediate]):
        if analysis.long_term.volatility_state == VolatilityState.HIGH:
            signals.append("Extreme volatility across all timeframes")
        else:
            signals.append("High short-term volatility, choppy conditions")
    
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
            signals.append("Strong volume confirming moves")
        else:
            signals.append("Low volume suggesting weak conviction")
    
    # Combine all signals into a readable format
    if not signals:
        return "Neutral market conditions"
    
    return " • ".join(signals)


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
        return f"Strong {dominant_phase.value} structure with {dominant_bias.value} momentum"
    elif short_term_conflict and trend_conflict:
        # Significant divergence across timeframes
        return "Mixed signals across multiple timeframes - transition likely"
    elif structural_conflict:
        # Potential major trend change
        return f"Possible trend reversal - {analysis.short_term.dominant_phase.value} forming"
    elif short_term_conflict:
        # Short-term deviation
        return f"{analysis.long_term.dominant_phase.value} with short-term {analysis.short_term.momentum_bias.value} move"
    else:
        # Default case
        return f"Developing {dominant_phase.value} structure with mixed momentum"

def _determine_market_context(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall market context considering three timeframes.
    """
    # Weight by timeframe importance
    weights = [
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]
    total_weight = sum(weights)
    if total_weight == 0:
        return "undefined context"

    # Calculate weighted volume strength
    volume_strength = (
        analysis.short_term.volume_strength * weights[0] +
        analysis.intermediate.volume_strength * weights[1] +
        analysis.long_term.volume_strength * weights[2]
    ) / total_weight

    if analysis.overall_direction == MultiTimeframeDirection.NEUTRAL:
        if volume_strength > 0.7:
            return "high volume ranging market"
        return "low volume consolidation"

    context = analysis.overall_direction.value
    if volume_strength > 0.7 and analysis.confidence_level > 0.7:
        return f"high-conviction {context} trend"
    elif volume_strength > 0.5 and analysis.confidence_level > 0.6:
        return f"established {context} trend"
    
    return f"developing {context} bias"

def _determine_trend_strength(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall trend strength considering three timeframes.
    """
    # Calculate weighted alignment
    alignments = [
        analysis.short_term.internal_alignment,
        analysis.intermediate.internal_alignment,
        analysis.long_term.internal_alignment
    ]
    weights = [
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]
    
    total_weight = sum(weights)
    if total_weight == 0:
        return "Undefined"
        
    weighted_alignment = sum(a * w for a, w in zip(alignments, weights)) / total_weight
    
    if weighted_alignment > 0.85:
        return "Extremely strong"
    elif weighted_alignment > 0.7:
        return "Very strong"
    elif weighted_alignment > 0.5:
        return "Strong"
    elif weighted_alignment > 0.3:
        return "Moderate"
    
    return "Weak"

def _get_trend_emoji_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Get appropriate trend emoji based on overall analysis state.
    """
    # First check if we have enough confidence
    if analysis.confidence_level < 0.4:
        return "📊"  # Low confidence
        
    # Get the overall trend strength
    trend_strength = analysis.alignment_score > 0.6 and analysis.confidence_level > 0.6
    
    match analysis.overall_direction:
        case MultiTimeframeDirection.BULLISH:
            if trend_strength:
                return "📈"  # Strong bullish
            return "↗️"  # Weak bullish
            
        case MultiTimeframeDirection.BEARISH:
            if trend_strength:
                return "📉"  # Strong bearish
            return "↘️"  # Weak bearish
            
        case MultiTimeframeDirection.NEUTRAL:
            # Check if we're in consolidation or in conflict
            if analysis.alignment_score > 0.6:
                return "↔️"  # Clear consolidation
            return "🔄"  # Mixed signals
            
    return "📊"  # Fallback for unknown states

def _generate_actionable_insight_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Generate comprehensive actionable insights considering all timeframes.
    """
    if analysis.confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals across timeframes.\n<b>Recommendation:</b>\nReduce exposure and wait for clearer setups."

    def get_main_signal() -> str:
        """Get primary trading signal based on timeframe hierarchy."""
        # Start with context timeframe for overall bias
        context_bias = analysis.context.momentum_bias
        context_phase = analysis.context.dominant_phase
        
        # Check main trend alignment
        trend_aligned = (
            analysis.long_term.momentum_bias == analysis.context.momentum_bias and
            analysis.intermediate.momentum_bias == analysis.context.momentum_bias
        )
        
        # Check for strong short-term momentum
        strong_momentum = (
            analysis.short_term.volume_strength > 0.7 and
            analysis.intermediate.volume_strength > 0.6 and
            analysis.momentum_intensity > MODERATE_MOMENTUM
        )

        # Generate main signal
        if trend_aligned and strong_momentum:
            return (
                f"Strong trend alignment across all timeframes ({context_bias.value}). "
                f"Market structure is {context_phase.value.lower()}. "
                "Aggressive positions can be considered with proper risk management."
            )
        elif trend_aligned:
            return (
                f"Trend alignment detected ({context_bias.value}) but momentum is moderate. "
                "Consider scaled entries with tight risk control."
            )
        else:
            return (
                "Mixed signals across timeframes. "
                "Focus on shorter timeframe opportunities with reduced position sizes."
            )

    # Format the complete insight
    main_signal = get_main_signal()

    return (
        f"<b>💡 Trading Insight:</b>\n{main_signal}"
    )

def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate enhanced trend description for a timeframe group."""
    volume_desc = (
        "strong volume" if analysis.volume_strength > 0.7 else
        "moderate volume" if analysis.volume_strength > 0.4 else
        "light volume"
    )
    
    risk_warning = ""
    if analysis.liquidation_risk == LiquidationRisk.HIGH:
        risk_warning = " ⚠️ High liquidation risk"
    
    volatility = ""
    if analysis.volatility_state == VolatilityState.HIGH:
        volatility = " | High volatility"
    
    funding = ""
    if abs(analysis.funding_sentiment) > 0.5:
        funding = f" | {'Bullish' if analysis.funding_sentiment > 0 else 'Bearish'} funding"
    
    return (
        f"• {analysis.dominant_phase.value} phase {analysis.dominant_action.value}\n"
        f"  └─ {volume_desc}{volatility}{funding}{risk_warning}"
    )

def _get_trade_suggestion(coin: str, direction: MultiTimeframeDirection, mid: float, significant_levels: Dict[Timeframe, SignificantLevelsData]) -> Optional[str]:
    """Generate trade suggestion with stop loss and take profit based on nearby levels."""
    def get_valid_levels(timeframe: Timeframe, min_dist: float, max_dist: float) -> tuple[List[float], List[float]]:
        """Get valid support and resistance levels for a specific timeframe."""
        return (
            [r for r in significant_levels[timeframe]['resistance'] if min_dist < abs(r - mid) < max_dist],
            [s for s in significant_levels[timeframe]['support'] if min_dist < abs(s - mid) < max_dist]
        )

    def get_trade_levels(direction: MultiTimeframeDirection, resistances: List[float], supports: List[float]) -> tuple[str, float, float]:
        """Get trade type, take profit and stop loss levels based on direction."""
        buffer_pct = 0.001  # 0.1% buffer for trigger levels
        
        if direction == MultiTimeframeDirection.BULLISH:
            # For longs: TP slightly above resistance, SL slightly below support
            closest_resistance = min(resistances, key=lambda x: abs(x - mid))
            closest_support = max(supports, key=lambda x: abs(x - mid))
            tp = closest_resistance * (1 + buffer_pct)  # TP slightly above resistance
            sl = closest_support * (1 - buffer_pct)     # SL slightly below support
            return "Long", tp, sl
        
        # For shorts: TP slightly below support, SL slightly above resistance
        closest_support = max(supports, key=lambda x: abs(x - mid))
        closest_resistance = min(resistances, key=lambda x: abs(x - mid))
        tp = closest_support * (1 - buffer_pct)     # TP slightly below support
        sl = closest_resistance * (1 + buffer_pct)  # SL slightly above resistance
        return "Short", tp, sl

    def format_trade(coin: str, side: str, entry: float, tp: float, sl: float) -> Optional[str]:
        """Format trade suggestion with consistent calculations and layout."""
        if (side == "Long" and (tp <= entry or sl >= entry)) or \
           (side == "Short" and (tp >= entry or sl <= entry)):
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

    min_distance = mid * 0.015
    max_distance = mid * 0.05

    # Try with 30min timeframe first, then 1h if needed
    for timeframe in [Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]:
        resistances, supports = get_valid_levels(timeframe, min_distance, max_distance)
        if resistances and supports:
            side, tp, sl = get_trade_levels(direction, resistances, supports)
            return format_trade(coin, side, mid, tp, sl)

    return None
