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

SHORT_TERM_IMPORTANCE = 0.40
INTERMEDIATE_IMPORTANCE = 0.40
LONG_TERM_IMPORTANCE = 0.12
CONTEXT_IMPORTANCE = 0.08

# Thresholds for confidence calculation
BASE_CONFIDENCE_THRESHOLD = 0.20  # Reduced from 0.25
STRONG_SIGNAL_MULTIPLIER = 1.3   # Reduced from 1.5
VOLATILITY_BOOST = 1.1          # Reduced from 1.2

DIRECTIONAL_WEIGHT: Final[float] = 0.35
VOLUME_WEIGHT: Final[float] = 0.30 
ALIGNMENT_WEIGHT: Final[float] = 0.15
MOMENTUM_WEIGHT: Final[float] = 0.20

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
    
    This new implementation focuses on:
    1. Directional agreement between timeframes
    2. Volume confirmation
    3. Momentum intensity
    4. Adaptability to crypto market volatility
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
    
    # Calculate volatility adjustment based on volatility state
    volatility_adjustment = _calculate_volatility_adjustment_direct(short_term, intermediate, long_term, context)
    
    # Use weights directly since they now sum to 1.0
    base_confidence = (
        directional_score * DIRECTIONAL_WEIGHT +
        volume_score * VOLUME_WEIGHT +
        alignment_score * ALIGNMENT_WEIGHT +
        momentum_score * MOMENTUM_WEIGHT
    )
    
    # Apply volatility adjustment - higher volatility can mean clearer signals in crypto
    adjusted_confidence = base_confidence * volatility_adjustment

    # Ensure the confidence is within bounds
    return max(min(adjusted_confidence, 1.0), BASE_CONFIDENCE_THRESHOLD)


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
    
    # Process short term
    if short_term is not None:
        bias = 1 if short_term.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if short_term.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['short'] = bias
        scores.append(abs(bias) * short_term.internal_alignment)
        weights.append(SHORT_TERM_IMPORTANCE)
    
    # Process intermediate term
    if intermediate is not None:
        bias = 1 if intermediate.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if intermediate.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['intermediate'] = bias
        scores.append(abs(bias) * intermediate.internal_alignment)
        weights.append(INTERMEDIATE_IMPORTANCE)
    
    # Process long term
    if long_term is not None:
        bias = 1 if long_term.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if long_term.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['long'] = bias
        scores.append(abs(bias) * long_term.internal_alignment)
        weights.append(LONG_TERM_IMPORTANCE)
    
    # Process context
    if context is not None:
        bias = 1 if context.momentum_bias == MultiTimeframeDirection.BULLISH else \
              -1 if context.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        
        group_biases['context'] = bias
        scores.append(abs(bias) * context.internal_alignment)
        weights.append(CONTEXT_IMPORTANCE)
    
    # Calculate cross-timeframe agreement
    agreement_bonus = 0.0
    if len(group_biases) >= 2:
        # Check if short and intermediate agree
        if 'short' in group_biases and 'intermediate' in group_biases:
            short_sign = 1 if group_biases['short'] > 0 else -1 if group_biases['short'] < 0 else 0
            int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
            if short_sign != 0 and int_sign != 0 and short_sign == int_sign:
                agreement_bonus += 0.10  # Reduced from 0.15
        
        # Check if intermediate and long agree
        if 'intermediate' in group_biases and 'long' in group_biases:
            int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
            long_sign = 1 if group_biases['long'] > 0 else -1 if group_biases['long'] < 0 else 0
            if int_sign != 0 and long_sign != 0 and int_sign == long_sign:
                agreement_bonus += 0.07  # Reduced from 0.10
    
    if not scores:
        return 0.0
        
    # Calculate weighted average
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Apply a dampening factor to the base score to reduce overall directional confidence
    base_score *= 0.9  # Apply 10% reduction to all directional scores
    
    # Apply agreement bonus and cap at 1.0
    return min(1.0, base_score + agreement_bonus)

def _calculate_momentum_score_direct(short_term: Optional[TimeframeGroupAnalysis],
                                     intermediate: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate momentum score focusing on short and intermediate timeframes.
    For crypto trading, momentum is particularly important.
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
        adjusted_momentum *= (0.5 + 0.5 * analysis.volume_strength)
        
        # Add a dampening factor to reduce overall momentum scores
        adjusted_momentum *= 0.85  # Reduce momentum by 15%
        
        momentum_values.append(adjusted_momentum)
    
    avg_momentum = sum(momentum_values) / len(momentum_values) if momentum_values else 0.0
    
    # Apply a less steep curve for crypto momentum
    if avg_momentum > 0.6:
        return min(1.0, avg_momentum * 1.1)  # Reduced boost from 1.3 to 1.1
    else:
        return avg_momentum * 0.9  # Further reduction of 10% for lower momentum scores

def _calculate_volume_score_direct(short_term: Optional[TimeframeGroupAnalysis], 
                                  intermediate: Optional[TimeframeGroupAnalysis],
                                  long_term: Optional[TimeframeGroupAnalysis],
                                  context: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate volume confirmation score with direct timeframe parameters.
    """
    weighted_scores = []
    
    for analysis in [short_term, intermediate, long_term, context]:
        if analysis is None:
            continue
            
        # Basic score is volume strength
        score = analysis.volume_strength
        
        # Apply multiplier based on timeframe
        if analysis.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES}:
            multiplier = 1.1  # Reduced from 1.2
        elif analysis.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES}:
            multiplier = 1.0  # Reduced from 1.1
        else:
            multiplier = 0.8  # Reduced from 0.9
            
        # Apply action-based adjustments
        if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
            multiplier *= 1.15  # Reduced from 1.25
        elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
            multiplier *= 1.05  # Reduced from 1.15
            
        weighted_scores.append(score * multiplier * analysis.internal_alignment)
    
    avg_volume_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    
    # Add a dampening factor to all volume scores
    avg_volume_score *= 0.9
    
    # Apply a more controlled scaling for crypto markets
    if avg_volume_score > 0.7:
        avg_volume_score *= 1.1  # Reduced from 1.2
    
    return min(1.0, avg_volume_score)

def _calculate_volatility_adjustment_direct(short_term: Optional[TimeframeGroupAnalysis],
                                           intermediate: Optional[TimeframeGroupAnalysis],
                                           long_term: Optional[TimeframeGroupAnalysis],
                                           context: Optional[TimeframeGroupAnalysis]) -> float:
    """
    Calculate volatility adjustment factor with direct timeframe parameters.
    More conservative approach to volatility adjustment.
    """
    total_count = 0
    high_volatility_count = 0
    
    for analysis in [short_term, intermediate, long_term, context]:
        if analysis is None:
            continue
            
        total_count += 1
        if analysis.volatility_state == VolatilityState.HIGH:
            high_volatility_count += 1
    
    # More conservative approach to volatility adjustment
    # If most timeframes show high volatility, apply a smaller boost
    if total_count > 0 and high_volatility_count >= max(2, total_count * 0.6):  # Increased threshold
        return VOLATILITY_BOOST
    # Add a slight reduction for normal/low volatility
    elif total_count > 0:
        return 0.95  # 5% reduction in confidence for normal/low volatility
    return 1.0
