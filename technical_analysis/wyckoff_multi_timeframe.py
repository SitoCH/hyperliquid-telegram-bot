from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, Timeframe, VolumeState
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
    """Analyze Wyckoff states across multiple timeframes."""
    if not states:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="Insufficient timeframe data for analysis"
        )

    # Calculate weighted phase frequency
    phase_weights: Dict[WyckoffPhase, float] = {}
    action_weights: Dict[CompositeAction, float] = {}
    total_weight = 0.0

    for timeframe, state in states.items():
        weight = get_phase_weight(timeframe)
        total_weight += weight

        # Add weighted votes for phase and action
        if not state.uncertain_phase:
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + weight
        action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Determine dominant phase and action
    dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0] if phase_weights else WyckoffPhase.UNKNOWN
    dominant_action = max(action_weights.items(), key=lambda x: x[1])[0] if action_weights else CompositeAction.UNKNOWN

    # Calculate alignment score
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    alignment_score = (phase_alignment + action_alignment) / 2

    # Calculate confidence level based on multiple factors
    volume_confidence = sum(1 for s in states.values() if s.volume == VolumeState.HIGH) / len(states)
    pattern_agreement = sum(1 for s in states.values() if s.pattern == states[Timeframe.HOURS_4].pattern) / len(states)
    
    confidence_level = (alignment_score * 0.5 + 
                       volume_confidence * 0.3 + 
                       pattern_agreement * 0.2)

    # Generate description
    description = _generate_mtf_description(
        states, dominant_phase, dominant_action, 
        alignment_score, confidence_level
    )

    return MultiTimeframeContext(
        alignment_score=alignment_score,
        confidence_level=confidence_level,
        description=description
    )

def _generate_mtf_description(
    states: Dict[Timeframe, WyckoffState],
    dominant_phase: WyckoffPhase,
    dominant_action: CompositeAction,
    alignment_score: float,
    confidence_level: float
) -> str:
    """Generate a detailed multi-timeframe analysis description."""
    # Determine dominant bias based on phase and action
    dominant_bias = (
        "bullish" if dominant_action == CompositeAction.ACCUMULATING
        else "bearish" if dominant_action == CompositeAction.DISTRIBUTING
        else "neutral"
    )

    # Format alignment and confidence as percentages
    alignment_pct = f"{alignment_score * 100:.0f}%"
    confidence_pct = f"{confidence_level * 100:.0f}%"

    # Analyze timeframe relationships
    higher_tf_agreement = (
        states[Timeframe.DAY_1].phase == states[Timeframe.HOURS_8].phase and
        states[Timeframe.HOURS_8].phase == states[Timeframe.HOURS_4].phase
    )
    
    medium_tf_agreement = (
        states[Timeframe.HOURS_8].phase == states[Timeframe.HOURS_4].phase and
        states[Timeframe.HOURS_4].phase == states[Timeframe.HOUR_1].phase
    )
    
    lower_tf_agreement = (
        states[Timeframe.HOURS_4].phase == states[Timeframe.HOUR_1].phase and
        states[Timeframe.HOUR_1].phase == states[Timeframe.MINUTES_15].phase
    )

    # Enhanced trend context with 8h perspective
    trend_context = (
        "Strong trending market across all timeframes" if higher_tf_agreement and medium_tf_agreement and lower_tf_agreement else
        "Clear trend in higher timeframes" if higher_tf_agreement and medium_tf_agreement else
        "Developing trend in medium timeframes" if medium_tf_agreement else
        "Mixed signals across timeframes" if alignment_score < 0.6 else
        "Transitioning market structure"
    )

    description = (
        f"• Bias: {dominant_bias.title()}\n"
        f"• Market Structure: {trend_context}\n"
        f"• Alignment: {alignment_pct} agreement across timeframes\n"
        f"• Signal Confidence: {confidence_pct}\n"
        "\nTimeframe Analysis:\n"
    )

    # Add specific timeframe details with hierarchical importance
    for tf in [Timeframe.DAY_1, Timeframe.HOURS_8, Timeframe.HOURS_4, Timeframe.HOUR_1, Timeframe.MINUTES_30, Timeframe.MINUTES_15]:
        state = states[tf]
        timeframe_desc = _get_timeframe_description(tf, state)
        description += timeframe_desc

    return description

def _get_timeframe_description(tf: Timeframe, state: WyckoffState) -> str:
    """Generate a descriptive line for each timeframe's state."""
    phase_desc = f"{state.phase.value}"
    if state.uncertain_phase:
        phase_desc += " (uncertain)"
    
    action_desc = ""
    if state.composite_action != CompositeAction.NEUTRAL:
        action_desc = f" - {state.composite_action.value}"
    
    volume_desc = " with high volume" if state.volume == VolumeState.HIGH else ""
    
    return f"• {tf.settings.description}: {phase_desc}{action_desc}{volume_desc}\n"
