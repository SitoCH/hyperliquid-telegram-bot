from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]
from utils import fmt_price

from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)


def generate_all_timeframes_description(analysis: AllTimeframesAnalysis, mid: float, resistance_levels: List[float], support_levels: List[float], interactive_analysis: bool) -> str:
    """Generate comprehensive description including three timeframe groups."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"
    
    momentum = _calculate_momentum_strength(analysis)
    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    long_term_desc = _get_timeframe_trend_description(analysis.long_term)

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
        f"<b>🔍 Timeframe Analysis:</b>\n"
        f"Long-Term View (8h-1d):\n{long_term_desc}\n"
        f"Mid-Term View (1h-4h):\n{intermediate_desc}\n"
        f"Near-Term View (15m-30m):\n{short_term_desc}\n\n"
        f"<b>🎯 Signal Quality:</b>\n"
        f"• Timeframe Alignment: {alignment_pct}\n"
        f"• Confidence Level: {confidence_pct}\n\n"
        f"{insight}"
    )

    trade_suggestion = _get_trade_suggestion(analysis.overall_direction, mid, resistance_levels, support_levels)
    if trade_suggestion:
        full_description += f"\n\n{trade_suggestion}"

    return full_description

def _calculate_momentum_strength(analysis: AllTimeframesAnalysis) -> str:
    """Calculate and describe momentum in plain English."""
    # Define timeframe weights (sum = 1.0)
    weights = {
        "long": 0.45,    # Long-term carries most weight
        "mid": 0.35,     # Mid-term secondary importance
        "short": 0.20    # Short-term least weight but still significant
    }
    
    # Calculate directional alignment with overall trend
    timeframe_scores = {
        "long": (
            1.0 if analysis.long_term.momentum_bias == analysis.overall_direction
            else 0.5 if analysis.long_term.momentum_bias == MultiTimeframeDirection.NEUTRAL
            else 0.0
        ),
        "mid": (
            1.0 if analysis.intermediate.momentum_bias == analysis.overall_direction
            else 0.5 if analysis.intermediate.momentum_bias == MultiTimeframeDirection.NEUTRAL
            else 0.0
        ),
        "short": (
            1.0 if analysis.short_term.momentum_bias == analysis.overall_direction
            else 0.5 if analysis.short_term.momentum_bias == MultiTimeframeDirection.NEUTRAL
            else 0.0
        )
    }
    
    # Calculate volume-weighted momentum
    volume_scores = {
        "long": analysis.long_term.volume_strength,
        "mid": analysis.intermediate.volume_strength,
        "short": analysis.short_term.volume_strength
    }
    
    # Calculate phase confirmation bonus
    phase_bonus = {
        "long": 0.2 if _is_phase_confirming_momentum(analysis.long_term) else 0.0,
        "mid": 0.2 if _is_phase_confirming_momentum(analysis.intermediate) else 0.0,
        "short": 0.2 if _is_phase_confirming_momentum(analysis.short_term) else 0.0
    }
    
    # Calculate final weighted score
    final_score = sum(
        (timeframe_scores[tf] * 0.5 +      # 50% weight to directional alignment
         volume_scores[tf] * 0.3 +         # 30% weight to volume confirmation
         phase_bonus[tf] * 0.2) *          # 20% weight to phase confirmation
        weights[tf]                        # Apply timeframe weight
        for tf in weights.keys()
    )
    
    # Replace basic strength descriptions with more detailed explanations
    if final_score > 0.85:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Strong buying pressure across all timeframes"
        return "Strong selling pressure across all timeframes"
    elif final_score > 0.70:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Steady accumulation with increasing volume"
        return "Sustained distribution with good volume"
    elif final_score > 0.55:
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            return "Moderate upward pressure"
        return "Moderate downward pressure"
    elif final_score > 0.40:
        return "Mixed momentum with no clear direction"
    elif final_score > 0.25:
        return "Low momentum, possible consolidation phase"
    return "Very low momentum, market likely ranging"

def _is_phase_confirming_momentum(analysis: TimeframeGroupAnalysis) -> bool:
    """Check if the Wyckoff phase confirms the momentum bias."""
    bullish_phases = {WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION}
    bearish_phases = {WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION}
    
    if analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
        return analysis.dominant_phase in bullish_phases
    elif analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
        return analysis.dominant_phase in bearish_phases
    return False

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
    """Get comprehensive market structure description across three timeframes."""
    phases = [
        analysis.long_term.dominant_phase,
        analysis.intermediate.dominant_phase,
        analysis.short_term.dominant_phase
    ]
    dominant_phase = max(set(phases), key=phases.count)
    phase_alignment = phases.count(dominant_phase) / len(phases)

    biases = [
        analysis.long_term.momentum_bias,
        analysis.intermediate.momentum_bias,
        analysis.short_term.momentum_bias,
    ]
    dominant_bias = max(set(biases), key=biases.count)
    bias_alignment = biases.count(dominant_bias) / len(biases)

    # New logic for handling conflicting signals
    if phase_alignment > 0.75 and bias_alignment > 0.75:
        # Check for conflicting signals
        is_conflict = (
            (dominant_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.POSSIBLE_MARKDOWN] and 
             dominant_bias == MultiTimeframeDirection.BULLISH) or
            (dominant_phase in [WyckoffPhase.MARKUP, WyckoffPhase.POSSIBLE_MARKUP] and 
             dominant_bias == MultiTimeframeDirection.BEARISH)
        )
        
        if is_conflict:
            if dominant_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.POSSIBLE_MARKDOWN]:
                return f"Potential reversal, {dominant_phase.value} showing bullish momentum"
            else:
                return f"Potential reversal, {dominant_phase.value} showing bearish momentum"
        else:
            return f"Strong {dominant_phase.value} structure with {dominant_bias.value} momentum"
            
    elif bias_alignment > 0.75:
        return f"Mixed structure with dominant {dominant_bias.value} momentum"
    elif phase_alignment > 0.75:
        return f"Clear {dominant_phase.value} structure with mixed momentum"
    
    return "Complex structure with mixed signals across timeframes"

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

    def get_action_plan() -> str:
        """Get base signal and action plan based on all timeframes."""
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            if analysis.confidence_level > 0.7:
                return (
                    "Longs: Prioritize entries on dips with tight stop-losses below key support levels. "
                    "Consider adding to positions as the trend strengthens.\n"
                    "Shorts: Avoid, high risk of bull traps. If shorting, use extremely tight stop-losses."
                )

            return (
                "Longs: Scaled entries near support zones with careful risk management. "
                "Use smaller position sizes due to mixed signals.\n"
                "Shorts: Only consider at significant resistance with strong bearish signals. "
                "Confirm with price action before entering."
            )

        elif analysis.overall_direction == MultiTimeframeDirection.BEARISH:
            if analysis.confidence_level > 0.7:
                return (
                    "Shorts: Focus on entries during rallies with tight stop-losses above key resistance levels. "
                    "Add to positions as the trend accelerates.\n"
                    "Longs: Avoid, high risk of bear traps. If longing, use extremely tight stop-losses."
                )

            return (
                "Shorts: Scaled entries near resistance zones with strict risk control. "
                "Confirm bearish signals with price action and volume.\n"
                "Longs: Only attempt at major support with clear bullish reversal patterns. "
                "Be cautious of potential bear traps."
            )

        action_plan = (
            "Both Directions: Trade range extremes with confirmation. "
            "Use smaller position sizes and tighter stop-losses.\n"
            "Avoid large positions until a clear trend emerges. "
            "Focus on short-term trades."
        )
        return action_plan

    action_plan = get_action_plan()

    # Add timeframe-specific insights
    timeframe_insights = []
    if analysis.short_term.momentum_bias != analysis.long_term.momentum_bias:
        timeframe_insights.append(
            f"Timeframe divergence: Long-term bias is {analysis.long_term.momentum_bias.value}, while short-term bias is {analysis.short_term.momentum_bias.value}. "
            f"Potential for trend reversal or continuation based on breakout direction. "
            f"Watch for a break of key levels to confirm the direction."
        )
    if analysis.short_term.dominant_phase != analysis.intermediate.dominant_phase:
        timeframe_insights.append(
            f"Phase mismatch: Short-term in {analysis.short_term.dominant_phase.value}, but mid-term in {analysis.intermediate.dominant_phase.value}. "
            f"Expect volatility as market seeks equilibrium. "
            f"Be prepared for rapid price swings and adjust stop-losses accordingly."
        )

    # Add risk warnings
    risk_warnings = []
    high_liq_risks = [tf.dominant_phase.value for tf in [analysis.short_term, analysis.intermediate, analysis.long_term] if tf.liquidation_risk == LiquidationRisk.HIGH]
    if high_liq_risks:
        risk_warnings.append(
            f"High liquidation risk on {', '.join(high_liq_risks)}. "
            f"Reduce leverage significantly to avoid forced liquidations. "
            f"Consider using isolated margin."
        )

    if analysis.short_term.volatility_state == VolatilityState.HIGH:
        risk_warnings.append(
            "High short-term volatility. "
            "Use smaller position sizes and wider stop-losses to account for rapid price swings. "
            "Avoid over-leveraging."
        )

    # Combine risk warnings if both high liquidation risk and high volatility are present
    if high_liq_risks and analysis.short_term.volatility_state == VolatilityState.HIGH:
        risk_warnings.append(
            "Combined high liquidation risk and high short-term volatility. "
            "Extreme caution is advised. Consider staying out of the market until conditions stabilize."
        )

    # Format the complete insight
    insights = []
    if timeframe_insights:
        insights.append("<b>⚡ Timeframe Analysis:</b>\n" + "\n".join(f"- {i}" for i in timeframe_insights))
    insights.append(f"\n<b>📝 Trading Strategy:</b>\n{action_plan}")
    if risk_warnings:
        insights.append("\n<b>⚠️ Risk Management:</b>\n" + "\n".join(f"- {w}" for w in risk_warnings))

    return "\n".join(insights)

 
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

def _get_trade_suggestion(direction: MultiTimeframeDirection, mid: float, resistance_levels: List[float], support_levels: List[float]) -> Optional[str]:
    """Generate trade suggestion with stop loss and take profit based on nearby levels."""
    if direction == MultiTimeframeDirection.NEUTRAL:
        return None

    # Filter levels within 1.5-6% range
    min_distance = mid * 0.015
    max_distance = mid * 0.06
    
    valid_resistances = [r for r in resistance_levels if min_distance < abs(r - mid) < max_distance]
    valid_supports = [s for s in support_levels if min_distance < abs(s - mid) < max_distance]
    
    if not valid_resistances or not valid_supports:
        return None

    def format_trade(side: str, entry: float, tp: float, sl: float) -> Optional[str]:
        """Format trade suggestion with consistent calculations and layout."""
        if (side == "Long" and (tp <= entry or sl >= entry)) or \
           (side == "Short" and (tp >= entry or sl <= entry)):
            return None
            
        tp_pct = abs((tp - entry) / entry) * 100
        sl_pct = abs((sl - entry) / entry) * 100
        
        return (
            f"<b>💰 {side} Trade Setup:</b>\n"
            f"Take Profit: {fmt_price(tp)} USDC (+{tp_pct:.1f}%)\n"
            f"Stop Loss: {fmt_price(sl)} USDC (-{sl_pct:.1f}%)"
        )

    if direction == MultiTimeframeDirection.BULLISH:
        tp = min(valid_resistances, key=lambda x: abs(x - mid))
        sl = max(valid_supports, key=lambda x: abs(x - mid))
        return format_trade("Long", mid, tp, sl)
    else:  # BEARISH
        tp = max(valid_supports, key=lambda x: abs(x - mid))
        sl = min(valid_resistances, key=lambda x: abs(x - mid))
        return format_trade("Short", mid, tp, sl)
