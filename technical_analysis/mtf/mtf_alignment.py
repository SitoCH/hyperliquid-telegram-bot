import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
from technical_analysis.wyckoff_types import SignificantLevelsData
from logging_utils import logger


from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, _TIMEFRAME_SETTINGS,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
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

# Define constants for enhanced readability
# Weight constants
ALIGNMENT_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.25
INTRADAY_WEIGHT = 0.15

# Threshold constants
VOLUME_THRESHOLD = 0.7
ALIGNMENT_THRESHOLD = 0.6
INTERNAL_ALIGNMENT_THRESHOLD = 0.65
MIN_CONFIDENCE_THRESHOLD = 0.35

# Activation function parameters
VOLUME_ACTIVATION = (7.0, 0.45)  # (steepness, threshold)
DIRECTIONAL_ACTIVATION = (6.5, 0.5)
ALIGNMENT_ACTIVATION = (6.0, 0.4)
INTRADAY_ACTIVATION = (8.0, 0.55)
FINAL_ACTIVATION = (6.0, 0.45)  # For sigmoid scaling at the end

# Boost multipliers
MARKUP_MARKDOWN_BOOST = 1.3
ACCUMULATION_DISTRIBUTION_BOOST = 1.2
INTRADAY_ALIGNMENT_BOOST = 1.25
HIGHER_TF_ALIGNMENT_BOOST = 1.15
VOLUME_PHASE_BOOST = 1.2
INTRADAY_VOLUME_BOOST = 1.2
INTRADAY_PHASE_BOOST = 1.15

RECENCY_BOOST = 1.4       # Weight recent signals more heavily

def nonlinear_activation(x, steepness=6.0, threshold=0.5):
    """Apply non-linear activation to emphasize strong signals and suppress weak ones"""
    return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))

def calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate overall confidence with enhanced intraday sensitivity."""
    if not analyses:
        return 0.0

    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return 0.0

    timeframe_groups = _group_analyses_by_timeframe(analyses)
    
    volume_confirmation = _calculate_volume_confirmation(analyses, timeframe_groups, total_weight)
    
    directional_agreement = _calculate_directional_agreement(timeframe_groups, total_weight)
    
    alignment_score = calculate_overall_alignment(analyses)
    
    intraday_confidence = _calculate_intraday_confidence(timeframe_groups)
    
    market_regime = _detect_market_regime(analyses, timeframe_groups)
    
    # Apply non-linear scaling to emphasize strong signals
    volume_confirmation = nonlinear_activation(volume_confirmation, *VOLUME_ACTIVATION)
    directional_agreement = nonlinear_activation(directional_agreement, *DIRECTIONAL_ACTIVATION)
    alignment_score = nonlinear_activation(alignment_score, *ALIGNMENT_ACTIVATION)
    
    if intraday_confidence > 0:
        intraday_confidence = nonlinear_activation(intraday_confidence, *INTRADAY_ACTIVATION)
        
    # Adjust weights based on market regime
    adjusted_weights = _adjust_weights_for_market_regime(market_regime)
    
    # Combine scores with dynamic weights
    raw_confidence = (
        volume_confirmation * adjusted_weights["volume"] +
        directional_agreement * adjusted_weights["directional"] +
        alignment_score * adjusted_weights["alignment"] +
        intraday_confidence * adjusted_weights["intraday"]
    )

    # Dynamic minimum confidence threshold
    min_confidence = _calculate_minimum_confidence(analyses, timeframe_groups)
    
    # Apply sigmoid-like scaling for final confidence score
    scaled_confidence = 1 / (1 + pow(2.0, -FINAL_ACTIVATION[0] * (raw_confidence - FINAL_ACTIVATION[1])))

    return max(min(scaled_confidence, 1.0), min_confidence)


def _detect_market_regime(analyses: List[TimeframeGroupAnalysis], 
                          timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> str:
    """
    Detect if market is in trending or ranging mode based on momentum thresholds.
    Returns: "trending" or "ranging"
    """
    # Focus on short-term analyses for intraday
    short_term_analyses = timeframe_groups.get('short', []) + timeframe_groups.get('intermediate', [])
    
    if not short_term_analyses:
        return "unknown"
    
    # Use existing momentum constants to determine market regime
    momentum_strengths = []
    for analysis in short_term_analyses:
        if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
            # Add momentum strength based on internal alignment and volume
            if analysis.internal_alignment > 0.7 and analysis.volume_strength > 0.65:
                momentum_strengths.append(STRONG_MOMENTUM)
            elif analysis.internal_alignment > 0.5 and analysis.volume_strength > 0.5:
                momentum_strengths.append(MODERATE_MOMENTUM)
            else:
                momentum_strengths.append(WEAK_MOMENTUM)
        else:
            momentum_strengths.append(LOW_MOMENTUM)
    
    # If no valid momentum strengths, default to unknown
    if not momentum_strengths:
        return "unknown"
    
    # Calculate average momentum strength
    avg_momentum = sum(momentum_strengths) / len(momentum_strengths)
    
    # Determine market regime based on momentum thresholds
    if avg_momentum >= MODERATE_MOMENTUM:
        return "trending"
    else:
        return "ranging"
    
def _adjust_weights_for_market_regime(regime: str) -> Dict[str, float]:
    """
    Adjust component weights based on market regime.
    """
    if regime == "trending":
        return {
            "volume": VOLUME_WEIGHT * 1.1,
            "directional": CONSISTENCY_WEIGHT * 1.2,
            "alignment": ALIGNMENT_WEIGHT * 0.9,
            "intraday": INTRADAY_WEIGHT * 1.1
        }
    elif regime == "ranging":
        return {
            "volume": VOLUME_WEIGHT * 0.9,
            "directional": CONSISTENCY_WEIGHT * 0.8,
            "alignment": ALIGNMENT_WEIGHT * 1.2,
            "intraday": INTRADAY_WEIGHT * 1.2
        }
    else:
        return {
            "volume": VOLUME_WEIGHT,
            "directional": CONSISTENCY_WEIGHT,
            "alignment": ALIGNMENT_WEIGHT,
            "intraday": INTRADAY_WEIGHT
        }

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

def _calculate_volume_confirmation(analyses: List[TimeframeGroupAnalysis], 
                                  timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]],
                                  total_weight: float) -> float:
    """Calculate volume confirmation score with intraday focus and recency bias."""
    volume_scores = []
    
    # Sort analyses by timeframe priority (shorter timeframes first)
    analyses_by_recency = sorted(
        analyses, 
        key=lambda a: 0 if a in timeframe_groups['short'] else 
                      1 if a in timeframe_groups['intermediate'] else 
                      2 if a in timeframe_groups['long'] else 3
    )
    
    # Apply recency bias - more recent (short-term) analyses get stronger weight
    for i, analysis in enumerate(analyses_by_recency):
        base_score = analysis.volume_strength
        recency_factor = RECENCY_BOOST if i < len(analyses_by_recency) // 2 else 1.0
        
        if analysis in timeframe_groups['short'] or analysis in timeframe_groups['intermediate']:
            if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_score *= MARKUP_MARKDOWN_BOOST
            elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
                base_score *= ACCUMULATION_DISTRIBUTION_BOOST
                
        volume_scores.append(base_score * (analysis.group_weight / total_weight) * recency_factor)

    return sum(volume_scores)

def _calculate_directional_agreement(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]], 
                                    total_weight: float) -> float:
    """Calculate directional agreement score with timeframe hierarchy."""
    directional_scores = []
    group_order = ['intermediate', 'short', 'long', 'context']
    prev_bias = None

    for group_name in group_order:
        group = timeframe_groups[group_name]
        for analysis in group:
            score = 1.0 if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL else 0.5

            if prev_bias and analysis.momentum_bias == prev_bias:
                if group_name in ['short', 'intermediate']:
                    score *= INTRADAY_ALIGNMENT_BOOST
                else:
                    score *= HIGHER_TF_ALIGNMENT_BOOST

            if analysis.volume_strength > VOLUME_THRESHOLD and analysis.internal_alignment > ALIGNMENT_THRESHOLD:
                score *= VOLUME_PHASE_BOOST
            
            directional_scores.append(score * (analysis.group_weight / total_weight))
            prev_bias = analysis.momentum_bias

    return sum(directional_scores)

def _calculate_intraday_confidence(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> float:
    """Calculate confidence based on short-term and intermediate timeframe alignment."""
    if not (timeframe_groups['short'] and timeframe_groups['intermediate']):
        return 0.0
        
    try:
        short_analysis = timeframe_groups['short'][0]
        intermediate_analysis = timeframe_groups['intermediate'][0]
        
        if short_analysis.momentum_bias == intermediate_analysis.momentum_bias:
            intraday_score = 1.0
            
            if (short_analysis.volume_strength > VOLUME_THRESHOLD and 
                intermediate_analysis.volume_strength > VOLUME_THRESHOLD):
                intraday_score *= INTRADAY_VOLUME_BOOST
                
            if short_analysis.dominant_phase == intermediate_analysis.dominant_phase:
                intraday_score *= INTRADAY_PHASE_BOOST
                
            return min(1.0, intraday_score)
    except (IndexError, AttributeError):
        logger.warning("Error calculating intraday confidence - check timeframe group data")
        
    return 0.0

def _calculate_minimum_confidence(analyses: List[TimeframeGroupAnalysis],
                                 timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> float:
    """Calculate dynamic minimum confidence threshold."""
    if len(analyses) < 2:
        return 0.0
        
    return MIN_CONFIDENCE_THRESHOLD if all(
        a.volume_strength > VOLUME_THRESHOLD and 
        a.internal_alignment > INTERNAL_ALIGNMENT_THRESHOLD and 
        (a in timeframe_groups['short'] or a in timeframe_groups['intermediate'])
        for a in analyses[:2]
    ) else 0.0
