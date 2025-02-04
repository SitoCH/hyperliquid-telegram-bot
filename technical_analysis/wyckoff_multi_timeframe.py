from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

class MultiTimeframeDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class TimeframeGroupAnalysis:
    dominant_phase: WyckoffPhase
    dominant_action: CompositeAction
    internal_alignment: float
    volume_strength: float
    momentum_bias: MultiTimeframeDirection
    group_weight: float
    funding_sentiment: float  # -1 to 1, negative means bearish funding
    liquidity_state: MarketLiquidity
    liquidation_risk: LiquidationRisk
    volatility_state: VolatilityState

@dataclass
class MultiTimeframeContext:
    alignment_score: float  # 0 to 1, indicating how well timeframes align
    confidence_level: float  # 0 to 1, indicating strength of signals
    description: str
    direction: MultiTimeframeDirection

@dataclass
class AllTimeframesAnalysis:
    scalping: TimeframeGroupAnalysis
    short_term: TimeframeGroupAnalysis
    intermediate: TimeframeGroupAnalysis
    long_term: TimeframeGroupAnalysis
    overall_direction: MultiTimeframeDirection
    confidence_level: float
    alignment_score: float

def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return timeframe.settings.phase_weight

def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState]
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across all timeframe groups.
    """
    # Input validation
    if not states:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="No timeframe data available for analysis",
            direction=MultiTimeframeDirection.NEUTRAL
        )

    # Group all timeframes with proper type handling
    scalping = {tf: state for tf, state in states.items() if tf in {Timeframe.MINUTES_5, Timeframe.MINUTES_15}}
    short_term = {tf: state for tf, state in states.items() if tf in {Timeframe.MINUTES_30, Timeframe.HOUR_1}}
    intermediate = {tf: state for tf, state in states.items() if tf in {Timeframe.HOURS_4}}
    long_term = {tf: state for tf, state in states.items() if tf in {Timeframe.HOURS_8, Timeframe.DAY_1}}

    try:
        # Analyze all groups
        scalping_analysis = _analyze_timeframe_group(scalping)
        short_term_analysis = _analyze_timeframe_group(short_term)
        intermediate_analysis = _analyze_timeframe_group(intermediate)
        long_term_analysis = _analyze_timeframe_group(long_term)

        # Update weights based on phase_weight
        scalping_analysis.group_weight = _calculate_group_weight(scalping)
        short_term_analysis.group_weight = _calculate_group_weight(short_term)
        intermediate_analysis.group_weight = _calculate_group_weight(intermediate)
        long_term_analysis.group_weight = _calculate_group_weight(long_term)

        # Calculate overall alignment across all groups
        all_analysis = AllTimeframesAnalysis(
            scalping=scalping_analysis,
            short_term=short_term_analysis,
            intermediate=intermediate_analysis,
            long_term=long_term_analysis,
            overall_direction=_determine_overall_direction([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ]),
            confidence_level=_calculate_overall_confidence([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ]),
            alignment_score=_calculate_overall_alignment([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ])
        )

        # Generate comprehensive description including all timeframes
        description = _generate_all_timeframes_description(all_analysis)

        return MultiTimeframeContext(
            alignment_score=all_analysis.alignment_score,
            confidence_level=all_analysis.confidence_level,
            description=description,
            direction=all_analysis.overall_direction
        )
        
    except Exception as e:
        # Fallback in case of analysis errors
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description=f"Error analyzing timeframes: {str(e)}",
            direction=MultiTimeframeDirection.NEUTRAL
        )


def _analyze_timeframe_group(
    group: Dict[Timeframe, WyckoffState]
) -> TimeframeGroupAnalysis:
    """Analyze a group with enhanced crypto-specific signals."""
    if not group:
        return TimeframeGroupAnalysis(
            dominant_phase=WyckoffPhase.UNKNOWN,
            dominant_action=CompositeAction.UNKNOWN,
            internal_alignment=0.0,
            volume_strength=0.0,
            momentum_bias=MultiTimeframeDirection.NEUTRAL,
            group_weight=0.0,
            funding_sentiment=0.0,
            liquidity_state=MarketLiquidity.UNKNOWN,
            liquidation_risk=LiquidationRisk.UNKNOWN,
            volatility_state=VolatilityState.UNKNOWN
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    action_weights: Dict[CompositeAction, float] = {}
    total_weight = 0.0
    
    # Track exhaustion signals
    upside_exhaustion = 0
    downside_exhaustion = 0

    # Track rapid movement signals
    rapid_bullish_moves = 0
    rapid_bearish_moves = 0
    
    for tf, state in group.items():
        weight = get_phase_weight(tf)
            
        # Detect potential exhaustion based on phase combinations
        if (state.phase == WyckoffPhase.MARKUP and 
            state.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]):
            upside_exhaustion += 1
            
        if (state.phase == WyckoffPhase.MARKDOWN and 
            state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.CONSOLIDATING]):
            downside_exhaustion += 1
            
        # Adjust weights based on exhaustion signals
        if state.is_upthrust:
            upside_exhaustion += 1
            weight *= 1.2  # Increase weight for potential reversal signals
        elif state.is_spring:
            downside_exhaustion += 1
            weight *= 1.2
            
        # Reduce weight if we see contrary volume signals
        if (state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN] and 
            state.volume == VolumeState.LOW):
            weight *= 0.8
            
        # Factor in effort vs result analysis
        if state.effort_vs_result == EffortResult.WEAK:
            if state.phase == WyckoffPhase.MARKUP:
                upside_exhaustion += 1
            elif state.phase == WyckoffPhase.MARKDOWN:
                downside_exhaustion += 1

        # Detect rapid price movements
        if state.phase == WyckoffPhase.MARKUP and state.volume == VolumeState.HIGH:
            rapid_bullish_moves += 1
        elif state.phase == WyckoffPhase.MARKDOWN and state.volume == VolumeState.HIGH:
            rapid_bearish_moves += 1
            
        # Increase weight for strong directional moves
        if state.pattern == MarketPattern.TRENDING:
            if state.volume == VolumeState.HIGH:
                weight *= 1.3  # 30% boost for high volume trends
        
        total_weight += weight

        if not state.uncertain_phase:
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + weight
        action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Determine dominant characteristics
    dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0] if phase_weights else WyckoffPhase.UNKNOWN
    dominant_action = max(action_weights.items(), key=lambda x: x[1])[0] if action_weights else CompositeAction.UNKNOWN

    # Calculate internal alignment
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Calculate volume strength with exhaustion consideration
    volume_strength = sum(1 for s in group.values() if s.volume == VolumeState.HIGH) / len(group)
    
    # Adjust volume strength based on exhaustion signals
    if upside_exhaustion >= len(group) // 2:
        volume_strength *= 0.7  # Reduce volume significance if exhaustion detected
    elif downside_exhaustion >= len(group) // 2:
        volume_strength *= 0.7

    # Calculate funding sentiment (-1 to 1)
    funding_signals = []
    for state in group.values():
        if state.funding_state == FundingState.HIGHLY_POSITIVE:
            funding_signals.append(1.0)
        elif state.funding_state == FundingState.POSITIVE:
            funding_signals.append(0.5)
        elif state.funding_state == FundingState.SLIGHTLY_POSITIVE:
            funding_signals.append(0.25)
        elif state.funding_state == FundingState.HIGHLY_NEGATIVE:
            funding_signals.append(-1.0)
        elif state.funding_state == FundingState.NEGATIVE:
            funding_signals.append(-0.5)
        elif state.funding_state == FundingState.SLIGHTLY_NEGATIVE:
            funding_signals.append(-0.25)
    
    funding_sentiment = sum(funding_signals) / len(funding_signals) if funding_signals else 0

    # Analyze market states
    liquidity_counts = {state.liquidity: 0 for state in group.values()}
    risk_counts = {state.liquidation_risk: 0 for state in group.values()}
    volatility_counts = {state.volatility: 0 for state in group.values()}
    
    for state in group.values():
        liquidity_counts[state.liquidity] += 1
        risk_counts[state.liquidation_risk] += 1
        volatility_counts[state.volatility] += 1
        
    liquidity_state = max(liquidity_counts.items(), key=lambda x: x[1])[0]
    liquidation_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
    volatility_state = max(volatility_counts.items(), key=lambda x: x[1])[0]

    # Enhanced momentum bias calculation with exhaustion consideration
    bullish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] and upside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.MARKING_UP] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKDOWN) or
        (s.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKDOWN)
    ))
    
    bearish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] and downside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.MARKING_DOWN] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKUP) or
        (s.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKUP)
    ))

    # Adjust momentum bias based on exhaustion signals
    consolidation_count = sum(1 for s in group.values() if s.composite_action == CompositeAction.CONSOLIDATING)
    total_signals = len(group)

    # Modified momentum bias calculation
    if consolidation_count / total_signals > 0.5:
        momentum_bias = MultiTimeframeDirection.NEUTRAL
    else:
        # Factor in exhaustion signals
        if upside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bullish_signals > bearish_signals else MultiTimeframeDirection.BEARISH
        elif downside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bearish_signals > bullish_signals else MultiTimeframeDirection.BULLISH
        else:
            momentum_bias = (
                MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals else
                MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals else
                MultiTimeframeDirection.NEUTRAL
            )

    # Modify momentum bias calculation
    if rapid_bullish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BULLISH
    elif rapid_bearish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BEARISH

    return TimeframeGroupAnalysis(
        dominant_phase=dominant_phase,
        dominant_action=dominant_action,
        internal_alignment=internal_alignment,
        volume_strength=volume_strength,
        momentum_bias=momentum_bias,
        group_weight=total_weight,
        funding_sentiment=funding_sentiment,
        liquidity_state=liquidity_state,
        liquidation_risk=liquidation_risk,
        volatility_state=volatility_state
    )

def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group based on phase weights."""
    return sum(get_phase_weight(tf) for tf in timeframes.keys())

def _calculate_overall_alignment(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate alignment across all timeframe groups."""
    # Filter out None values and empty groups
    valid_analyses = [a for a in analyses if a is not None]
    if len(valid_analyses) < 2:
        return 0.0
        
    total_weight = sum(analysis.group_weight for analysis in valid_analyses)
    if total_weight == 0:
        return 0.0
        
    # Calculate weighted phase and action alignments
    weighted_alignments = []
    for i, analysis1 in enumerate(valid_analyses):
        for j, analysis2 in enumerate(valid_analyses[i+1:], i+1):
            weight = (analysis1.group_weight + analysis2.group_weight) / total_weight
            phase_aligned = 1.0 if analysis1.dominant_phase == analysis2.dominant_phase else 0.0
            bias_aligned = 1.0 if analysis1.momentum_bias == analysis2.momentum_bias else 0.0
            weighted_alignments.append(weight * (phase_aligned * 0.6 + bias_aligned * 0.4))
    
    return sum(weighted_alignments) / len(weighted_alignments) if weighted_alignments else 0.0

def _calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate overall confidence across all timeframe groups with improved weighting."""
    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return 0.0

    # Updated weight factors with more emphasis on alignment and volume
    alignment_weight = 0.45    # Increased from 0.40 - more importance to alignment
    volume_weight = 0.35       # Unchanged
    consistency_weight = 0.20  # Reduced from 0.25 - less weight on pure consistency

    # Enhanced volume confirmation with timeframe importance
    volume_confirmation = sum(
        analysis.volume_strength * (analysis.group_weight / total_weight)
        for analysis in analyses
    )

    # Improved trend consistency check
    bias_counts: Dict[MultiTimeframeDirection, int] = {}
    weighted_biases: Dict[MultiTimeframeDirection, float] = {}
    for analysis in analyses:
        bias = analysis.momentum_bias
        bias_counts[bias] = bias_counts.get(bias, 0) + 1
        weighted_biases[bias] = weighted_biases.get(bias, 0) + analysis.group_weight
    
    # Calculate trend consistency score
    max_bias_count = max(bias_counts.values()) if bias_counts else 0
    max_weighted_bias = max(weighted_biases.values()) if weighted_biases else 0
    
    # Combine count-based and weight-based consistency
    trend_consistency = (
        (max_bias_count / len(analyses)) * 0.4 +  # How many timeframes agree
        (max_weighted_bias / total_weight) * 0.6   # How important are the agreeing timeframes
    )

    return min(
        (volume_confirmation * volume_weight +
         trend_consistency * consistency_weight +
         _calculate_overall_alignment(analyses) * alignment_weight),
        1.0
    )

def _determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall direction considering all timeframe groups."""
    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return MultiTimeframeDirection.NEUTRAL

    weighted_signals = {
        direction: sum(
            analysis.group_weight / total_weight
            for analysis in analyses
            if analysis.momentum_bias == direction
        )
        for direction in MultiTimeframeDirection
    }

    # Require stronger consensus for directional bias
    strongest_direction = max(weighted_signals.items(), key=lambda x: x[1])
    if strongest_direction[1] > 0.6:  # Require 60% weight agreement
        return strongest_direction[0]
    
    return MultiTimeframeDirection.NEUTRAL

def _generate_all_timeframes_description(analysis: AllTimeframesAnalysis) -> str:
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
