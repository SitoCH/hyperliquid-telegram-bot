from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, Timeframe, VolumeState
)

@dataclass
class MultiTimeframeContext:
    alignment_score: float  # 0-1 score indicating how well timeframes align
    dominant_phase: WyckoffPhase  # Most significant phase across timeframes
    dominant_action: CompositeAction  # Most significant composite action
    confidence_level: float  # 0-1 score for overall confidence
    description: str  # Human-readable analysis

def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return {
        Timeframe.MINUTES_15: 0.1,
        Timeframe.HOUR_1: 0.2,
        Timeframe.HOURS_4: 0.3,
        Timeframe.DAY_1: 0.4
    }[timeframe]

def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState]
) -> MultiTimeframeContext:
    """Analyze Wyckoff states across multiple timeframes."""
    if not states:
        return MultiTimeframeContext(0.0, WyckoffPhase.UNKNOWN, 
                                   CompositeAction.UNKNOWN, 0.0, 
                                   "Insufficient data for analysis")

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
        dominant_phase=dominant_phase,
        dominant_action=dominant_action,
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
    """Generate a human-readable multi-timeframe analysis description."""
    
    # Early return for low confidence
    if confidence_level < 0.4:
        return "Insufficient confidence in multi-timeframe analysis. Market structure unclear."

    # Format alignment quality
    alignment_desc = (
        "strong" if alignment_score > 0.8
        else "moderate" if alignment_score > 0.6
        else "weak"
    )

    # Analyze timeframe relationships
    higher_tf_state = states.get(Timeframe.DAY_1)
    lower_tf_state = states.get(Timeframe.HOUR_1)
    
    if not (higher_tf_state and lower_tf_state):
        return "Incomplete timeframe data for analysis."

    # Build description
    lines = [
        f"Multi-timeframe analysis shows {alignment_desc} alignment "
        f"({alignment_score:.0%} agreement between timeframes).",
        
        f"Dominant market phase is {dominant_phase.value} with "
        f"{dominant_action.value} on multiple timeframes.",
    ]

    # Add trend structure analysis
    if higher_tf_state.phase == lower_tf_state.phase:
        lines.append("Trend structure is aligned across timeframes, "
                    "increasing probability of continuation.")
    else:
        lines.append("Divergence between higher and lower timeframes "
                    "suggests potential trend transition.")

    # Add volume analysis
    volume_agreement = all(s.volume == VolumeState.HIGH for s in states.values())
    if volume_agreement:
        lines.append("Strong volume confirmation across all timeframes.")
    
    # Add confidence statement
    lines.append(f"Overall analysis confidence: {confidence_level:.0%}")

    return "\n".join(lines)
