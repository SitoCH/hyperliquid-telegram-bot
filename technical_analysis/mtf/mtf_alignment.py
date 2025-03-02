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
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
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

def calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate overall confidence with enhanced intraday sensitivity."""
    if not analyses:
        return 0.0

    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return 0.0

    # Adjusted weights for intraday focus
    alignment_weight = 0.25     # Reduced from 0.30
    volume_weight = 0.35       # Unchanged
    consistency_weight = 0.25  # Reduced from 0.35
    intraday_weight = 0.15    # New component

    # Group analyses by timeframe
    timeframe_groups = {
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

    # Enhanced volume confirmation with intraday focus
    volume_scores = []
    for analysis in analyses:
        # Base volume score
        base_score = analysis.volume_strength
        
        # Stronger boost for intraday volume confirmation
        if analysis in timeframe_groups['short'] or analysis in timeframe_groups['intermediate']:
            if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_score *= 1.3  # Increased from 1.2
            elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
                base_score *= 1.2  # Increased from 1.1
                
        volume_scores.append(base_score * (analysis.group_weight / total_weight))

    volume_confirmation = sum(volume_scores)

    # Enhanced trend consistency with group hierarchy
    directional_scores = []

    # Process groups in order of importance for crypto intraday trading
    group_order = ['intermediate', 'short', 'long', 'context']
    prev_bias = None

    for group_name in group_order:
        group = timeframe_groups[group_name]
        for analysis in group:
            # Base directional score
            score = 1.0 if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL else 0.5

            # Alignment bonus with previous timeframe
            if prev_bias and analysis.momentum_bias == prev_bias:
                if group_name in ['short', 'intermediate']:
                    score *= 1.25  # Stronger bonus for intraday alignment
                else:
                    score *= 1.15

            # Extra weight for strong momentum with volume
            if analysis.volume_strength > 0.7 and analysis.internal_alignment > 0.6:
                score *= 1.2
            
            directional_scores.append(score * (analysis.group_weight / total_weight))
            prev_bias = analysis.momentum_bias

    directional_agreement = sum(directional_scores)

    # Calculate alignment score
    alignment_score = calculate_overall_alignment(analyses)

    # New: Calculate intraday confidence
    intraday_confidence = 0.0
    if timeframe_groups['short'] and timeframe_groups['intermediate']:
        short_analysis = timeframe_groups['short'][0]
        intermediate_analysis = timeframe_groups['intermediate'][0]
        
        # Check for strong intraday alignment
        if short_analysis.momentum_bias == intermediate_analysis.momentum_bias:
            intraday_score = 1.0
            # Boost for volume confirmation
            if short_analysis.volume_strength > 0.7 and intermediate_analysis.volume_strength > 0.7:
                intraday_score *= 1.2
            # Boost for phase alignment
            if short_analysis.dominant_phase == intermediate_analysis.dominant_phase:
                intraday_score *= 1.15
            intraday_confidence = min(1.0, intraday_score)
        
    # Add non-linear signal weighting for superior hourly quality
    def nonlinear_activation(x, steepness=6.0, threshold=0.5):
        """Apply non-linear activation to emphasize strong signals and suppress weak ones"""
        return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))

    # Apply non-linear scaling to individual components
    volume_confirmation = nonlinear_activation(volume_confirmation, steepness=7.0, threshold=0.45)
    directional_agreement = nonlinear_activation(directional_agreement, steepness=6.5, threshold=0.5)
    alignment_score = nonlinear_activation(alignment_score, steepness=6.0, threshold=0.4)

    # For intraday signals, use more aggressive thresholds to reduce noise
    if intraday_confidence > 0:
        intraday_confidence = nonlinear_activation(intraday_confidence, steepness=8.0, threshold=0.55)

    # Combine scores with dynamic minimum threshold
    raw_confidence = (
        volume_confirmation * volume_weight +
        directional_agreement * consistency_weight +
        alignment_score * alignment_weight +
        intraday_confidence * intraday_weight
    )

    # Dynamic minimum confidence based on volume and alignment
    min_confidence = 0.35 if all(
        a.volume_strength > 0.7 and 
        a.internal_alignment > 0.65 and 
        (a in timeframe_groups['short'] or a in timeframe_groups['intermediate'])
        for a in analyses[:2]  # Check first two analyses
    ) else 0.0
    
    # Apply sigmoid-like scaling with adjusted steepness for faster response
    scaled_confidence = 1 / (1 + pow(2.0, -6 * (raw_confidence - 0.45)))

    return max(min(scaled_confidence, 1.0), min_confidence)
