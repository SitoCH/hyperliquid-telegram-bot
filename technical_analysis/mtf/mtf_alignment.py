import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from logging_utils import logger

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, SignificantLevelsData, _TIMEFRAME_SETTINGS,
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
     AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT
)

# Thresholds for confidence calculation
BASE_CONFIDENCE_THRESHOLD = 0.20
STRONG_SIGNAL_MULTIPLIER = 1.3

DIRECTIONAL_WEIGHT: Final[float] = 0.35
VOLUME_WEIGHT: Final[float] = 0.35
ALIGNMENT_WEIGHT: Final[float] = 0.10
MOMENTUM_WEIGHT: Final[float] = 0.20

def _calculate_timeframe_importance() -> Tuple[float, float, float, float]:
    """
    Calculate the importance of each timeframe group based on the phase_weight
    values from timeframe settings.
    """
    # Extract phase weights from timeframe settings
    short_term_weights = [_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES]
    intermediate_weights = [_TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES]
    long_term_weights = [_TIMEFRAME_SETTINGS[tf].phase_weight for tf in LONG_TERM_TIMEFRAMES]
    context_weights = [_TIMEFRAME_SETTINGS[tf].phase_weight for tf in CONTEXT_TIMEFRAMES]
    
    # Sum weights by timeframe group
    st_weight = sum(short_term_weights)
    int_weight = sum(intermediate_weights)
    lt_weight = sum(long_term_weights)
    ctx_weight = sum(context_weights)
    
    total_weight = st_weight + int_weight + lt_weight + ctx_weight
    
    return (
        st_weight / total_weight,
        int_weight / total_weight,
        lt_weight / total_weight,
        ctx_weight / total_weight
    )

def calculate_overall_alignment(short_term_analysis: TimeframeGroupAnalysis, intermediate_analysis: TimeframeGroupAnalysis) -> float:
    """Calculate alignment across all timeframe groups with more conservative weighting for crypto intraday trading."""

    total_weight = short_term_analysis.group_weight + intermediate_analysis.group_weight

    alignment_scores = []
    # Compare short-term and intermediate analyses
    weight = (short_term_analysis.group_weight + intermediate_analysis.group_weight) / total_weight

    # Phase Alignment Scoring
    phase_aligned = 0.0
    if short_term_analysis.dominant_phase == intermediate_analysis.dominant_phase:
        phase_aligned = 1.0
    elif short_term_analysis.dominant_phase.value.replace('~', '') == intermediate_analysis.dominant_phase.value.replace('~', ''):
        phase_aligned = 0.70
    elif (is_bullish_phase(short_term_analysis.dominant_phase) and is_bullish_phase(intermediate_analysis.dominant_phase)) or \
         (is_bearish_phase(short_term_analysis.dominant_phase) and is_bearish_phase(intermediate_analysis.dominant_phase)):
        phase_aligned = 0.45

    # Action Alignment Scoring: More conservative for crypto intraday
    action_aligned = 0.0
    if short_term_analysis.dominant_action == intermediate_analysis.dominant_action:
        action_aligned = 1.0
    elif (is_bullish_action(short_term_analysis.dominant_action) and is_bullish_action(intermediate_analysis.dominant_action)) or \
         (is_bearish_action(short_term_analysis.dominant_action) and is_bearish_action(intermediate_analysis.dominant_action)):
        action_aligned = 0.6

    # Momentum Bias Alignment: More nuanced for crypto
    bias_aligned = 0.0
    if short_term_analysis.momentum_bias == intermediate_analysis.momentum_bias:
        bias_aligned = 1.0
    elif short_term_analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL and intermediate_analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
        bias_aligned = 0.50 if short_term_analysis.momentum_bias != intermediate_analysis.momentum_bias else 1.0
    else:
        bias_aligned = 0.40

    # Volume Agreement: More direct comparison
    volume_agreement = 1 - abs(short_term_analysis.volume_strength - intermediate_analysis.volume_strength)

    # Composite Score: Adjusted Weights
    alignment_score = (
        phase_aligned * 0.35 +
        action_aligned * 0.35 +
        bias_aligned * 0.20 +
        volume_agreement * 0.10
    )

    alignment_scores.append(alignment_score * weight)

    if not alignment_scores:
        return 0.0

    return sum(alignment_scores)

def calculate_overall_confidence(short_term: TimeframeGroupAnalysis, intermediate: TimeframeGroupAnalysis, long_term: TimeframeGroupAnalysis, context: TimeframeGroupAnalysis) -> float:
    """
    Calculate overall confidence using a simplified algorithm optimized for crypto.
    """
    
    # Calculate directional agreement score directly
    directional_score = _calculate_directional_score_direct(short_term, intermediate, long_term, context)
    
    # Calculate volume confirmation score with direct parameters
    volume_score = _calculate_volume_score_direct(short_term, intermediate, long_term, context)
    
    # Calculate alignment between timeframes (if both exist)
    alignment_score = 0.0
    if short_term is not None and intermediate is not None:
        alignment_score = calculate_overall_alignment(short_term, intermediate)
    
    # Calculate momentum intensity directly
    momentum_score = _calculate_momentum_score_direct(short_term, intermediate)

    # Detect early strong signals in short timeframes that deserve higher confidence
    early_signal_boost = 0.0
    if short_term is not None and intermediate is not None:
        # Boost for strong alignment in shorter timeframes
        if short_term.internal_alignment >= 0.7 and short_term.volume_strength >= 0.65:
            early_signal_boost += 0.1
        
        # Boost for clear directional bias with volume
        if short_term.momentum_bias != MultiTimeframeDirection.NEUTRAL and short_term.volume_strength >= 0.6:
            early_signal_boost += 0.05
        
        # Boost for strong alignment between short and intermediate
        if (short_term.momentum_bias == intermediate.momentum_bias and 
            short_term.momentum_bias != MultiTimeframeDirection.NEUTRAL and
            short_term.volume_strength >= 0.6):
            early_signal_boost += 0.12

        if (intermediate.internal_alignment >= 0.7 and 
            intermediate.momentum_bias != MultiTimeframeDirection.NEUTRAL):
            early_signal_boost += 0.06
    
    # Base confidence calculation
    confidence = (
        directional_score * DIRECTIONAL_WEIGHT +
        volume_score * VOLUME_WEIGHT +
        alignment_score * ALIGNMENT_WEIGHT +
        momentum_score * MOMENTUM_WEIGHT
    )
    
    # Apply early signal boost (capped)
    confidence = min(1.0, confidence + early_signal_boost)

    return max(min(confidence, 1.0), BASE_CONFIDENCE_THRESHOLD)


def _calculate_directional_score_direct(short_term: Optional[TimeframeGroupAnalysis],
                                       intermediate: Optional[TimeframeGroupAnalysis],
                                       long_term: Optional[TimeframeGroupAnalysis],
                                       context: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate directional agreement score with emphasis on short and intermediate timeframes.
    In crypto, these timeframes often provide the most actionable signals.
    """
    scores = []
    weights = []
    
    # Get the dominant bias from each timeframe
    group_biases = {}

    # Calculate importance dynamically from phase weights
    short_term_importance, intermediate_importance, long_term_importance, context_importance = _calculate_timeframe_importance()
    
    # Process short term
    if short_term is not None:
        bias = 1 if short_term.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if short_term.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['short'] = bias
        scores.append(abs(bias) * short_term.internal_alignment)
        weights.append(short_term_importance)
    
    # Process intermediate term
    if intermediate is not None:
        bias = 1 if intermediate.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if intermediate.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['intermediate'] = bias
        scores.append(abs(bias) * intermediate.internal_alignment)
        weights.append(intermediate_importance)
    
    # Process long term
    if long_term is not None:
        bias = 1 if long_term.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if long_term.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['long'] = bias
        scores.append(abs(bias) * long_term.internal_alignment)
        weights.append(long_term_importance)
    
    # Process context
    if context is not None:
        bias = 1 if context.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if context.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['context'] = bias
        scores.append(abs(bias) * context.internal_alignment)
        weights.append(context_importance)
    
    # Calculate cross-timeframe agreement
    agreement_bonus = 0.0
    if 'short' in group_biases and 'intermediate' in group_biases:
        short_sign = 1 if group_biases['short'] > 0 else -1 if group_biases['short'] < 0 else 0
        int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
        if short_sign != 0 and int_sign != 0 and short_sign == int_sign:
            agreement_bonus += 0.15  # Increased from 0.10 - most important agreement for intraday
    
    # Cross-agreement between other timeframes is less important for intraday
    if 'intermediate' in group_biases and 'long' in group_biases:
        int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
        long_sign = 1 if group_biases['long'] > 0 else -1 if group_biases['long'] < 0 else 0
        if int_sign != 0 and long_sign != 0 and int_sign == int_sign:
            agreement_bonus += 0.05  # Reduced from 0.07
    
    if not scores:
        return 0.0
        
    # Calculate weighted average
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Less dampening for intraday signals
    base_score *= 0.95  # Apply 5% reduction instead of 10%
    
    # Apply agreement bonus and cap at 1.0
    return min(1.0, base_score + agreement_bonus)

def _calculate_momentum_score_direct(short_term: Optional[TimeframeGroupAnalysis],
                                     intermediate: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate momentum score focusing on short and intermediate timeframes.
    """
    # Focus primarily on short and intermediate timeframes
    relevant_analyses = []
    if short_term is not None:
        relevant_analyses.append(short_term)
    if intermediate is not None:
        relevant_analyses.append(intermediate)
        
    if not relevant_analyses:
        return 0.0
    
    # Calculate average momentum
    momentum_values = []
    for analysis in relevant_analyses:
        # Convert momentum bias to a numeric value
        if analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
            momentum = 1.0
        elif analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
            momentum = -1.0
        else:
            momentum = 0.0
            
        # Adjust by internal alignment - more aligned = more reliable momentum
        adjusted_momentum = abs(momentum) * analysis.internal_alignment
        
        # Further adjust by volume - volume confirms momentum
        adjusted_momentum *= (0.6 + 0.4 * analysis.volume_strength)  # Increased base factor
        
        # Reduced dampening factor
        adjusted_momentum *= 0.95  # Only reduce momentum by 5% instead of 15%
        
        momentum_values.append(adjusted_momentum)
    
    avg_momentum = sum(momentum_values) / len(momentum_values) if momentum_values else 0.0
    
    # Apply a more responsive curve for crypto momentum
    if avg_momentum > 0.5:  # Lowered threshold from 0.6 to 0.5
        return min(1.0, avg_momentum * 1.15)  # Increased boost
    else:
        return avg_momentum * 0.95  # Reduced penalty

def _calculate_volume_score_direct(short_term: Optional[TimeframeGroupAnalysis], 
                                  intermediate: Optional[TimeframeGroupAnalysis],
                                  long_term: Optional[TimeframeGroupAnalysis],
                                  context: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate volume confirmation score with improved early signal detection.
    """
    weighted_scores = []
    
    for analysis in [short_term, intermediate, long_term, context]:
        if analysis is None:
            continue

        score = analysis.volume_strength
        
        # Apply multiplier based on timeframe group - increased short-term importance
        if analysis == short_term:
            multiplier = 1.25  # Increased from 1.1
        elif analysis == intermediate:
            multiplier = 1.05  # Increased from 1.0
        else:
            multiplier = 0.9

        # Apply action-based adjustments - increased weights
        if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
            multiplier *= 1.25  # Increased from 1.15
        elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
            multiplier *= 1.15  # Increased from 1.05

        weighted_scores.append(score * multiplier * analysis.internal_alignment)
    
    avg_volume_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    
    # Reduced dampening factor
    avg_volume_score *= 0.95  # Reduced penalty from 0.9
    
    # Apply a more aggressive scaling for crypto markets
    if avg_volume_score > 0.6:  # Lowered threshold from 0.7
        avg_volume_score *= 1.15  # Increased boost from 1.1
    
    return min(1.0, avg_volume_score)
