from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, Timeframe, VolumeState, FundingState
)

@dataclass
class MultiTimeframeContext:
    alignment_score: float  # 0 to 1, indicating how well timeframes align
    confidence_level: float  # 0 to 1, indicating strength of signals
    description: str

def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return timeframe.settings.phase_weight

def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState]
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across multiple timeframes to determine market structure.
    
    Evaluates alignment between timeframes, confidence of signals, and generates
    trading context with higher weighting on longer timeframes.
    
    Args:
        states: Dictionary mapping timeframes to their Wyckoff states
        
    Returns:
        MultiTimeframeContext with alignment score, confidence level and description
    """
    # Input validation
    if not states:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="No timeframe data available for analysis"
        )

    # Ensure minimum required timeframes
    required_higher = {Timeframe.HOURS_4, Timeframe.HOURS_8, Timeframe.DAY_1}
    required_lower = {Timeframe.HOUR_1, Timeframe.MINUTES_30, Timeframe.MINUTES_15}
    
    available_higher = set(states.keys()) & required_higher
    available_lower = set(states.keys()) & required_lower
    
    if not available_higher or not available_lower:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="Insufficient timeframe coverage for reliable analysis"
        )

    # Group timeframes
    higher_tf = {tf: state for tf, state in states.items() if tf in available_higher}
    lower_tf = {tf: state for tf, state in states.items() if tf in available_lower}

    try:
        # Analyze each group
        higher_analysis = _analyze_timeframe_group(higher_tf)
        lower_analysis = _analyze_timeframe_group(lower_tf)

        # Calculate alignment and confidence
        groups_alignment = _calculate_dual_groups_alignment(higher_analysis, lower_analysis)
        confidence_level = _calculate_dual_confidence(higher_analysis, lower_analysis, groups_alignment)

        # Generate comprehensive description
        description = _generate_dual_group_description(
            higher_analysis,
            lower_analysis,
            groups_alignment,
            confidence_level,
            higher_tf  # Pass higher_tf to the description generator
        )

        return MultiTimeframeContext(
            alignment_score=groups_alignment,
            confidence_level=confidence_level,
            description=description
        )
        
    except Exception as e:
        # Fallback in case of analysis errors
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description=f"Error analyzing timeframes: {str(e)}"
        )

@dataclass
class TimeframeGroupAnalysis:
    dominant_phase: WyckoffPhase
    dominant_action: CompositeAction
    internal_alignment: float
    volume_strength: float
    momentum_bias: str
    group_weight: float

def _analyze_timeframe_group(
    group: Dict[Timeframe, WyckoffState]
) -> TimeframeGroupAnalysis:
    """Analyze a group of related timeframes."""
    if not group:
        return TimeframeGroupAnalysis(
            dominant_phase=WyckoffPhase.UNKNOWN,
            dominant_action=CompositeAction.UNKNOWN,
            internal_alignment=0.0,
            volume_strength=0.0,
            momentum_bias="neutral",
            group_weight=0.0
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    action_weights: Dict[CompositeAction, float] = {}
    total_weight = 0.0

    for tf, state in group.items():
        weight = get_phase_weight(tf)
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

    # Calculate volume strength
    volume_strength = sum(1 for s in group.values() if s.volume == VolumeState.HIGH) / len(group)

    # Determine momentum bias
    bullish_signals = sum(1 for s in group.values() if 
        s.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] or 
        s.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.MARKING_UP])
    bearish_signals = sum(1 for s in group.values() if 
        s.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] or 
        s.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.MARKING_DOWN])
    
    momentum_bias = (
        "bullish" if bullish_signals > bearish_signals else
        "bearish" if bearish_signals > bullish_signals else
        "neutral"
    )

    return TimeframeGroupAnalysis(
        dominant_phase=dominant_phase,
        dominant_action=dominant_action,
        internal_alignment=internal_alignment,
        volume_strength=volume_strength,
        momentum_bias=momentum_bias,
        group_weight=total_weight
    )

def _calculate_dual_groups_alignment(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> float:
    """Calculate alignment between higher and lower timeframe groups."""
    # Calculate phase alignment
    phase_alignment = 1.0 if higher.dominant_phase == lower.dominant_phase else 0.0

    # Calculate bias alignment
    bias_alignment = 1.0 if higher.momentum_bias == lower.momentum_bias else 0.0

    # Calculate weighted alignment score
    alignment = (phase_alignment * 0.6) + (bias_alignment * 0.4)

    # Apply internal alignment weights
    weighted_alignment = (
        alignment * 0.6 +
        (higher.internal_alignment * 0.6 +
         lower.internal_alignment * 0.4) * 0.4
    )

    return weighted_alignment

def _calculate_dual_confidence(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float
) -> float:
    """Calculate confidence level for dual group analysis."""
    # Weight factors
    alignment_weight = 0.4
    volume_weight = 0.3
    consistency_weight = 0.3

    # Volume confirmation (weighted towards higher timeframes)
    volume_confirmation = (
        higher.volume_strength * 0.6 +
        lower.volume_strength * 0.4
    )

    # Trend consistency
    trend_consistency = 1.0 if higher.momentum_bias == lower.momentum_bias else 0.0

    return (
        groups_alignment * alignment_weight +
        volume_confirmation * volume_weight +
        trend_consistency * consistency_weight
    )

def _generate_dual_group_description(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float,
    confidence_level: float,
    higher_tf: Dict[Timeframe, WyckoffState]
) -> str:
    """Generate a narrative description for dual group analysis."""
    alignment_pct = f"{groups_alignment * 100:.0f}%"
    confidence_pct = f"{confidence_level * 100:.0f}%"

    market_context = _determine_dual_market_context(higher, lower)
    trend_strength = _determine_dual_trend_strength(higher, lower)
    actionable_insight = _generate_dual_actionable_insight(higher, lower, confidence_level)
    
    # Get funding state (same for all timeframes, so we can take from either group)
    funding_state = next(iter(higher_tf.values())).funding_state if higher_tf else FundingState.UNKNOWN

    # Create narrative description
    description = (
        f"Market analysis shows a {trend_strength.lower()} {market_context.lower()} with {funding_state.value} funding. "
        f"Higher timeframes indicate {higher.dominant_phase.value} phase with {higher.dominant_action.value}, "
        f"while lower timeframes show {lower.dominant_phase.value} with {lower.dominant_action.value}. "
        f"Overall alignment between timeframes is {alignment_pct} with {confidence_pct} confidence.\n"
        f"{actionable_insight}"
    )

    return description

def _generate_dual_actionable_insight(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    confidence_level: float
) -> str:
    """Generate actionable insights from dual timeframe analysis."""
    if confidence_level < 0.5:
        return "<b>Interpretation:</b> Insufficient confidence for clear directional bias\n<b>Suggestion:</b> Avoid new positions, reduce existing exposure"

    # Get base market condition
    if higher.momentum_bias == lower.momentum_bias:
        if higher.momentum_bias == "bullish":
            base_signal = "Strong bullish continuation likely"
            action_plan = (
                "Longs: Hold existing positions, add on pullbacks to higher timeframe support\n"
                "Shorts: Avoid counter-trend positions, only scalp clear rejections"
            )
        elif higher.momentum_bias == "bearish":
            base_signal = "Strong bearish continuation likely"
            action_plan = (
                "Shorts: Hold existing positions, add on rallies to higher timeframe resistance\n"
                "Longs: Avoid counter-trend positions, only scalp clear bounces"
            )
        else:
            base_signal = "Neutral market conditions"
            action_plan = (
                "Both Sides: Focus on range-bound strategies\n"
                "Wait for clear breakout confirmation before directional trades"
            )
    else:
        base_signal = f"Potential {lower.momentum_bias} reversal forming against {higher.momentum_bias} trend"
        if lower.momentum_bias == "bullish":
            action_plan = (
                "Longs: Build positions gradually with tight stops below lower timeframe support\n"
                "Shorts: Take profit on existing positions, avoid adding to shorts"
            )
        else:
            action_plan = (
                "Shorts: Build positions gradually with tight stops above lower timeframe resistance\n"
                "Longs: Take profit on existing positions, avoid adding to longs"
            )

    return f"<b>Interpretation:</b> {base_signal}\n<b>Suggestions:</b>\n{action_plan}"

def _determine_dual_market_context(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> str:
    """Determine market context from two timeframe groups."""
    if higher.momentum_bias == lower.momentum_bias:
        return f"Strong {higher.momentum_bias} bias across timeframes"
    
    return f"Higher timeframe {higher.momentum_bias} with lower timeframe {lower.momentum_bias}"

def _determine_dual_trend_strength(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> str:
    """Determine trend strength from two timeframe groups."""
    avg_alignment = (higher.internal_alignment * 0.6 + lower.internal_alignment * 0.4)
    
    if avg_alignment > 0.8:
        return "Very Strong"
    elif avg_alignment > 0.6:
        return "Strong"
    elif avg_alignment > 0.4:
        return "Moderate"
    else:
        return "Weak"
