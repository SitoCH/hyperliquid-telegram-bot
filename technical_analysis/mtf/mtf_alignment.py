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
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT,
    DIRECTIONAL_WEIGHT, VOLUME_WEIGHT, PHASE_WEIGHT
)


def calculate_overall_alignment(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate alignment across all timeframe groups with improved weighting."""
    valid_analyses = [a for a in analyses if a is not None]
    if len(valid_analyses) < 2:
        return 0.0

    total_weight = sum(analysis.group_weight for analysis in valid_analyses)
    if total_weight == 0:
        return 0.0

    alignment_scores = []
    comparison_count = 0

    for i, analysis1 in enumerate(valid_analyses):
        for j, analysis2 in enumerate(valid_analyses[i + 1:], i + 1):
            comparison_count += 1
            weight = (analysis1.group_weight + analysis2.group_weight) / total_weight

            # Phase Alignment Scoring: More nuanced approach
            phase_aligned = 0.0
            if analysis1.dominant_phase == analysis2.dominant_phase:
                phase_aligned = 1.0  # Perfect alignment
            elif analysis1.dominant_phase.value.replace('~', '') == analysis2.dominant_phase.value.replace('~', ''):
                phase_aligned = 0.75  # Possible phase alignment
            elif (is_bullish_phase(analysis1.dominant_phase) and is_bullish_phase(analysis2.dominant_phase)) or \
                 (is_bearish_phase(analysis1.dominant_phase) and is_bearish_phase(analysis2.dominant_phase)):
                phase_aligned = 0.5  # General agreement on bullish/bearish phase

            # Action Alignment Scoring: Refined logic
            action_aligned = 0.0
            if analysis1.dominant_action == analysis2.dominant_action:
                action_aligned = 1.0
            elif (is_bullish_action(analysis1.dominant_action) and is_bullish_action(analysis2.dominant_action)) or \
                 (is_bearish_action(analysis1.dominant_action) and is_bearish_action(analysis2.dominant_action)):
                action_aligned = 0.6  # Agreement on bullish/bearish action

            # Momentum Bias Alignment: Simplified
            bias_aligned = 1.0 if analysis1.momentum_bias == analysis2.momentum_bias else 0.0
            if analysis1.momentum_bias != MultiTimeframeDirection.NEUTRAL and analysis2.momentum_bias != MultiTimeframeDirection.NEUTRAL:
                bias_aligned = 0.5 if analysis1.momentum_bias != analysis2.momentum_bias else 1.0

            # Volume Agreement: More direct comparison
            volume_agreement = 1 - abs(analysis1.volume_strength - analysis2.volume_strength)

            # Composite Score: Adjusted Weights
            alignment_score = (
                phase_aligned * 0.40 +  # Phase alignment (40%)
                action_aligned * 0.30 +  # Action alignment (30%)
                bias_aligned * 0.20 +  # Bias alignment (20%)
                volume_agreement * 0.10  # Volume agreement (10%)
            )

            alignment_scores.append(alignment_score * weight)

    if not alignment_scores:
        return 0.0

    return sum(alignment_scores) / comparison_count

# Timeframe importance weights for crypto
SHORT_TERM_IMPORTANCE = 0.35
INTERMEDIATE_IMPORTANCE = 0.40
LONG_TERM_IMPORTANCE = 0.15
CONTEXT_IMPORTANCE = 0.10

# Thresholds for confidence calculation
BASE_CONFIDENCE_THRESHOLD = 0.25
STRONG_SIGNAL_MULTIPLIER = 1.5
VOLATILITY_BOOST = 1.2

# Define proportion adjustments for crypto markets
# Rather than redefining constants, we'll adjust the proportions within the function
CRYPTO_DIRECTIONAL_RATIO = 0.9   # Slightly lower than imported DIRECTIONAL_WEIGHT
CRYPTO_VOLUME_RATIO = 0.85       # Slightly lower than imported VOLUME_WEIGHT
ALIGNMENT_PROPORTION = 0.15      # New proportion for alignment
MOMENTUM_PROPORTION = 0.10       # New proportion for momentum

def calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
    """
    Calculate overall confidence using a simplified algorithm optimized for crypto.
    
    This new implementation focuses on:
    1. Directional agreement between timeframes
    2. Volume confirmation
    3. Momentum intensity
    4. Adaptability to crypto market volatility
    """
    if not analyses or len(analyses) < 2:
        return 0.0
    
    # Group analyses by timeframe
    timeframe_groups = _group_analyses_by_timeframe(analyses)
    
    # Calculate directional agreement score
    directional_score = _calculate_directional_score(timeframe_groups)
    
    # Calculate volume confirmation score
    volume_score = _calculate_volume_score(analyses)
    
    # Calculate alignment between timeframes
    alignment_score = calculate_overall_alignment(analyses)
    
    # Calculate momentum intensity
    momentum_score = _calculate_momentum_score(timeframe_groups)
    
    # Calculate volatility adjustment based on volatility state
    volatility_adjustment = _calculate_volatility_adjustment(analyses)
    
    # Calculate the total weight to ensure proportions sum to 1.0
    total_weight = (DIRECTIONAL_WEIGHT * CRYPTO_DIRECTIONAL_RATIO) + \
                  (VOLUME_WEIGHT * CRYPTO_VOLUME_RATIO) + \
                  ALIGNMENT_PROPORTION + MOMENTUM_PROPORTION
    
    # Combine scores with appropriate weights for crypto
    base_confidence = (
        directional_score * (DIRECTIONAL_WEIGHT * CRYPTO_DIRECTIONAL_RATIO / total_weight) +
        volume_score * (VOLUME_WEIGHT * CRYPTO_VOLUME_RATIO / total_weight) +
        alignment_score * (ALIGNMENT_PROPORTION / total_weight) +
        momentum_score * (MOMENTUM_PROPORTION / total_weight)
    )
    
    # Apply volatility adjustment - higher volatility can mean clearer signals in crypto
    adjusted_confidence = base_confidence * volatility_adjustment
    
    # Apply a more aggressive scaling function for crypto
    # This allows strong signals to reach higher confidence values
    final_confidence = _crypto_confidence_scaling(adjusted_confidence)
    
    # Ensure the confidence is within bounds
    return max(min(final_confidence, 1.0), BASE_CONFIDENCE_THRESHOLD)

def _crypto_confidence_scaling(x: float) -> float:
    """
    More aggressive scaling function for crypto that allows strong signals
    to reach higher confidence values.
    """
    # For crypto, we want a steeper curve that allows values above 0.6 to scale more aggressively
    if x < 0.4:
        return x  # Keep lower values as is
    elif x < 0.7:
        return x * 1.15  # Slight boost for medium signals
    else:
        # More aggressive boost for strong signals, capped at 1.0
        return min(1.0, x * 1.3)

def _calculate_directional_score(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> float:
    """
    Calculate directional agreement score with emphasis on short and intermediate timeframes.
    In crypto, these timeframes often provide the most actionable signals.
    """
    scores = []
    weights = []
    
    # Get the dominant bias from each timeframe group
    group_biases = {}
    for group_name, group in timeframe_groups.items():
        if not group:
            continue
            
        # Average the momentum bias for the group
        biases = [
            1 if a.momentum_bias == MultiTimeframeDirection.BULLISH else
            -1 if a.momentum_bias == MultiTimeframeDirection.BEARISH else 0
            for a in group
        ]
        if not biases:
            continue
            
        avg_bias = sum(biases) / len(biases)
        group_biases[group_name] = avg_bias
        
        # Calculate internal alignment within the group
        internal_alignment = sum(a.internal_alignment for a in group) / len(group)
        
        # Calculate score based on strength of bias
        score = abs(avg_bias) * internal_alignment
        scores.append(score)
        
        # Assign weight based on timeframe importance
        if group_name == 'short':
            weights.append(SHORT_TERM_IMPORTANCE)
        elif group_name == 'intermediate':
            weights.append(INTERMEDIATE_IMPORTANCE)
        elif group_name == 'long':
            weights.append(LONG_TERM_IMPORTANCE)
        else:  # context
            weights.append(CONTEXT_IMPORTANCE)
    
    # Calculate cross-timeframe agreement
    agreement_bonus = 0.0
    if len(group_biases) >= 2:
        # Check if short and intermediate agree
        if 'short' in group_biases and 'intermediate' in group_biases:
            short_sign = 1 if group_biases['short'] > 0 else -1 if group_biases['short'] < 0 else 0
            int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
            if short_sign != 0 and int_sign != 0 and short_sign == int_sign:
                agreement_bonus += 0.15  # Significant bonus for short-intermediate agreement
        
        # Check if intermediate and long agree
        if 'intermediate' in group_biases and 'long' in group_biases:
            int_sign = 1 if group_biases['intermediate'] > 0 else -1 if group_biases['intermediate'] < 0 else 0
            long_sign = 1 if group_biases['long'] > 0 else -1 if group_biases['long'] < 0 else 0
            if int_sign != 0 and long_sign != 0 and int_sign == long_sign:
                agreement_bonus += 0.10  # Bonus for intermediate-long agreement
    
    if not scores:
        return 0.0
        
    # Calculate weighted average
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Apply agreement bonus and cap at 1.0
    return min(1.0, base_score + agreement_bonus)

def _calculate_volume_score(analyses: List[TimeframeGroupAnalysis]) -> float:
    """
    Calculate volume confirmation score with emphasis on recent volume activity.
    In crypto, volume spikes often precede significant moves.
    """
    if not analyses:
        return 0.0
    
    weighted_scores = []
    
    for analysis in analyses:
        # Basic score is volume strength
        score = analysis.volume_strength
        
        # Apply multiplier based on timeframe
        if analysis.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES}:
            multiplier = 1.2  # Higher weight for short-term volume in crypto
        elif analysis.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES}:
            multiplier = 1.1  # Good weight for intermediate timeframes
        else:
            multiplier = 0.9  # Less weight for long-term volume
            
        # Apply action-based adjustments
        if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
            multiplier *= 1.25  # Volume in trending markets is more significant
        elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
            multiplier *= 1.15  # Volume in accumulation/distribution is also important
            
        weighted_scores.append(score * multiplier * analysis.internal_alignment)
    
    avg_volume_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    
    # Apply a more aggressive scaling for crypto markets
    if avg_volume_score > 0.7:
        avg_volume_score *= 1.2  # Boost high volume signals
    
    return min(1.0, avg_volume_score)

def _calculate_momentum_score(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> float:
    """
    Calculate momentum score focusing on short and intermediate timeframes.
    For crypto trading, momentum is particularly important.
    """
    # Focus primarily on short and intermediate timeframes
    relevant_groups = []
    if timeframe_groups['short']:
        relevant_groups.extend(timeframe_groups['short'])
    if timeframe_groups['intermediate']:
        relevant_groups.extend(timeframe_groups['intermediate'])
        
    if not relevant_groups:
        return 0.0
    
    # Calculate average momentum
    momentum_values = []
    for analysis in relevant_groups:
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
        
        momentum_values.append(adjusted_momentum)
    
    avg_momentum = sum(momentum_values) / len(momentum_values) if momentum_values else 0.0
    
    # Apply a steeper curve for crypto momentum
    if avg_momentum > 0.6:
        return min(1.0, avg_momentum * 1.3)  # Boost strong momentum
    else:
        return avg_momentum

def _calculate_volatility_adjustment(analyses: List[TimeframeGroupAnalysis]) -> float:
    """
    Calculate volatility adjustment factor. In crypto, higher volatility 
    often leads to clearer signals and better trading opportunities.
    """
    # Count analyses with high volatility
    high_volatility_count = sum(1 for a in analyses if a.volatility_state == VolatilityState.HIGH)
    
    # If most timeframes show high volatility, apply a boost
    if high_volatility_count >= max(1, len(analyses) * 0.5):
        return VOLATILITY_BOOST
    return 1.0

def _group_analyses_by_timeframe(analyses: List[TimeframeGroupAnalysis]) -> Dict[str, List[TimeframeGroupAnalysis]]:
    """Group analyses by timeframe category."""
    return {
        'short': [a for a in analyses if a.group_weight in {
            _TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES
        }],
        'intermediate': [a for a in analyses if a.group_weight in {
            _TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES
        }],
        'long': [a for a in analyses if a.group_weight in {
            _TIMEFRAME_SETTINGS[tf].phase_weight for tf in LONG_TERM_TIMEFRAMES
        }],
        'context': [a for a in analyses if a.group_weight in {
            _TIMEFRAME_SETTINGS[tf].phase_weight for tf in CONTEXT_TIMEFRAMES
        }]
    }
