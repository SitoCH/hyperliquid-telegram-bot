from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)


def generate_all_timeframes_description(analysis: AllTimeframesAnalysis) -> str:
    """Generate comprehensive description including all timeframe groups."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"

    # Get descriptions for all timeframe groups
    scalping_desc = _get_timeframe_trend_description(analysis.scalping)
    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    long_term_desc = _get_timeframe_trend_description(analysis.long_term)

    # Get market structure and context
    structure = _get_full_market_structure(analysis)
    context = _determine_market_context(analysis)
    
    # Get appropriate emoji
    emoji = _get_trend_emoji_all_timeframes(analysis)

    # Generate action plan
    insight = _generate_actionable_insight_all_timeframes(analysis)

    description = (
        f"{emoji} Market Structure Analysis:\n"
        f"Trend: {_determine_trend_strength(analysis)} {context}\n"
        f"Market Structure: {structure}\n\n"
        f"Long Term (8h-1d):\n{long_term_desc}\n"
        f"Intermediate (4h):\n{intermediate_desc}\n"
        f"Short Term (30m-1h):\n{short_term_desc}\n"
        f"Scalping (5m-15m):\n{scalping_desc}\n\n"
        f"Timeframe Alignment: {alignment_pct}\n"
        f"Signal Confidence: {confidence_pct}\n\n"
        f"{insight}"
    )

    return description

def _get_full_market_structure(analysis: AllTimeframesAnalysis) -> str:
    """
    Get comprehensive market structure description across all timeframes.
    """
    # Count aligned phases
    phases = [
        analysis.long_term.dominant_phase,
        analysis.intermediate.dominant_phase,
        analysis.short_term.dominant_phase,
        analysis.scalping.dominant_phase
    ]
    dominant_phase = max(set(phases), key=phases.count)
    phase_alignment = phases.count(dominant_phase) / len(phases)

    # Count aligned biases
    biases = [
        analysis.long_term.momentum_bias,
        analysis.intermediate.momentum_bias,
        analysis.short_term.momentum_bias,
        analysis.scalping.momentum_bias
    ]
    dominant_bias = max(set(biases), key=biases.count)
    bias_alignment = biases.count(dominant_bias) / len(biases)

    if phase_alignment > 0.75 and bias_alignment > 0.75:
        return f"Strong {dominant_phase.value} structure with {dominant_bias.value} momentum"
    elif bias_alignment > 0.75:
        return f"Mixed phase structure with aligned {dominant_bias.value} momentum"
    elif phase_alignment > 0.75:
        return f"Aligned {dominant_phase.value} structure with mixed momentum"
    
    return "Complex structure with mixed signals across timeframes"

def _determine_market_context(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall market context considering all timeframes.
    """
    # Weight by timeframe importance
    weights = [
        analysis.scalping.group_weight,
        analysis.short_term.group_weight,
        analysis.intermediate.group_weight,
        analysis.long_term.group_weight
    ]
    total_weight = sum(weights)
    if total_weight == 0:
        return "undefined context"

    # Calculate weighted volume strength
    volume_strength = (
        analysis.scalping.volume_strength * weights[0] +
        analysis.short_term.volume_strength * weights[1] +
        analysis.intermediate.volume_strength * weights[2] +
        analysis.long_term.volume_strength * weights[3]
    ) / total_weight

    if analysis.overall_direction == MultiTimeframeDirection.NEUTRAL:
        if volume_strength > 0.7:
            return "high-volume ranging market"
        return "low-volume consolidation"

    context = analysis.overall_direction.value
    if volume_strength > 0.7 and analysis.confidence_level > 0.7:
        return f"high-conviction {context} trend"
    elif volume_strength > 0.5 and analysis.confidence_level > 0.6:
        return f"established {context} trend"
    
    return f"developing {context} bias"

def _determine_trend_strength(analysis: AllTimeframesAnalysis) -> str:
    """
    Determine overall trend strength considering all timeframes.
    """
    # Calculate weighted alignment
    alignments = [
        analysis.scalping.internal_alignment,
        analysis.short_term.internal_alignment,
        analysis.intermediate.internal_alignment,
        analysis.long_term.internal_alignment
    ]
    weights = [
        analysis.scalping.group_weight,
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
    Get appropriate trend emoji based on all timeframe analysis.
    """
    if analysis.confidence_level < 0.5:
        return "📊"

    # Count directional biases
    biases = [
        analysis.long_term.momentum_bias,
        analysis.intermediate.momentum_bias,
        analysis.short_term.momentum_bias,
        analysis.scalping.momentum_bias
    ]
    bullish_count = sum(1 for b in biases if b == MultiTimeframeDirection.BULLISH)
    bearish_count = sum(1 for b in biases if b == MultiTimeframeDirection.BEARISH)

    if bullish_count >= 3:  # Strong bullish alignment
        return "📈" if analysis.confidence_level > 0.7 else "↗️"
    elif bearish_count >= 3:  # Strong bearish alignment
        return "📉" if analysis.confidence_level > 0.7 else "↘️"
    
    return "↔️"

def _generate_actionable_insight_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Generate comprehensive actionable insights considering all timeframes.
    """
    if analysis.confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals across timeframes.\n<b>Recommendation:</b>\nReduce exposure and wait for clearer setups."

    def get_full_context() -> tuple[str, str]:
        """Get base signal and action plan based on all timeframes."""
        if analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            if analysis.confidence_level > 0.7:
                return (
                    "Strong bullish alignment across multiple timeframes with good volume support.",
                    "Longs: Add on dips with defined risk.\nShorts: Avoid counter-trend positions."
                )
            return (
                "Developing bullish structure with mixed timeframe signals.",
                "Longs: Scaled entries with tight risk management.\nShorts: Only at key resistance levels."
                )
        elif analysis.overall_direction == MultiTimeframeDirection.BEARISH:
            if analysis.confidence_level > 0.7:
                return (
                    "Strong bearish alignment across multiple timeframes with sustained selling.",
                    "Shorts: Add on rallies with defined risk.\nLongs: Avoid counter-trend positions."
                )
            return (
                "Developing bearish structure with mixed timeframe signals.",
                "Shorts: Scaled entries with tight risk management.\nLongs: Only at key support levels."
                )
        
        return (
            "Mixed signals across timeframes indicating a transitional or ranging market.",
            "Both Directions: Focus on range extremes and wait for clearer directional signals."
        )

    base_signal, action_plan = get_full_context()
    
    # Add timeframe-specific insights
    timeframe_insights = []
    if analysis.scalping.momentum_bias != analysis.long_term.momentum_bias:
        timeframe_insights.append(
            f"Timeframe divergence: {analysis.long_term.momentum_bias.value} on higher timeframes "
            f"vs {analysis.scalping.momentum_bias.value} on lower timeframes"
        )

    # Add risk warnings
    risk_warnings = []
    if any(tf.liquidation_risk == LiquidationRisk.HIGH for tf in 
           [analysis.scalping, analysis.short_term, analysis.intermediate, analysis.long_term]):
        risk_warnings.append("Multiple timeframes showing high liquidation risk")

    if any(tf.volatility_state == VolatilityState.HIGH for tf in 
           [analysis.scalping, analysis.short_term]):
        risk_warnings.append("High short-term volatility, adjust position sizes accordingly")

    # Format the complete insight
    insights = [f"<b>Analysis:</b>\n{base_signal}"]
    if timeframe_insights:
        insights.append("\n<b>Timeframe Notes:</b>\n" + "\n".join(f"- {i}" for i in timeframe_insights))
    insights.append(f"\n<b>Strategy:</b>\n{action_plan}")
    if risk_warnings:
        insights.append("\n<b>Risk Warnings:</b>\n" + "\n".join(f"- {w}" for w in risk_warnings))

    return "\n".join(insights)

 
def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate trend description for a timeframe group."""
    return f"• {analysis.dominant_phase.value} phase {analysis.dominant_action.value} with " + (
        "strong volume" if analysis.volume_strength > 0.7 else
        "moderate volume" if analysis.volume_strength > 0.4 else
        "light volume"
    )
