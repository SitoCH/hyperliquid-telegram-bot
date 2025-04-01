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


def calculate_overall_alignment(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> float:
    """Calculate alignment across all timeframe groups with improved weighting."""
    short_term = timeframe_groups.get('short', [])
    intermediate = timeframe_groups.get('intermediate', [])

    valid_analyses = short_term + intermediate

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

SHORT_TERM_IMPORTANCE = 0.40
INTERMEDIATE_IMPORTANCE = 0.40
LONG_TERM_IMPORTANCE = 0.12
CONTEXT_IMPORTANCE = 0.08

# Thresholds for confidence calculation
BASE_CONFIDENCE_THRESHOLD = 0.25
STRONG_SIGNAL_MULTIPLIER = 1.5
VOLATILITY_BOOST = 1.2

DIRECTIONAL_WEIGHT: Final[float] = 0.35
VOLUME_WEIGHT: Final[float] = 0.30 
ALIGNMENT_WEIGHT: Final[float] = 0.15
MOMENTUM_WEIGHT: Final[float] = 0.20

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
    alignment_score = calculate_overall_alignment(timeframe_groups)
    
    # Calculate momentum intensity
    momentum_score = _calculate_momentum_score(timeframe_groups)
    
    # Calculate volatility adjustment based on volatility state
    volatility_adjustment = _calculate_volatility_adjustment(analyses)
    
    # Use weights directly since they now sum to 1.0
    base_confidence = (
        directional_score * DIRECTIONAL_WEIGHT +
        volume_score * VOLUME_WEIGHT +
        alignment_score * ALIGNMENT_WEIGHT +
        momentum_score * MOMENTUM_WEIGHT
    )
    
    # Apply volatility adjustment - higher volatility can mean clearer signals in crypto
    adjusted_confidence = base_confidence * volatility_adjustment
    
    # Apply a more aggressive scaling function for crypto
    # This allows strong signals to reach higher confidence values
    final_confidence = _crypto_confidence_scaling(adjusted_confidence)
    
    # Ensure the confidence is within bounds
    return max(min(final_confidence, 1.0), BASE_CONFIDENCE_THRESHOLD)

# Update the scaling parameters to be more aggressive
def _crypto_confidence_scaling(x: float) -> float:
    """
    More aggressive scaling function for crypto that allows strong signals
    to reach higher confidence values.
    """
    # Even more aggressive curve to boost confidence values
    if x < 0.35:  # Lowered from 0.4
        return x * 1.05  # Slight boost even to lower values
    elif x < 0.6:  # Lowered from 0.7
        return x * 1.25  # Higher boost for medium signals
    else:
        # Much more aggressive boost for strong signals
        return min(1.0, x * 1.4)  # Increased from 1.3

# Add an opportunity detection function that identifies trading setups
def detect_trading_opportunity(analyses: List[TimeframeGroupAnalysis], 
                               confidence: float, 
                               alignment: float) -> Tuple[bool, str, float]:
    """
    Detect if current market conditions represent a trading opportunity.
    
    Args:
        analyses: List of timeframe group analyses
        confidence: Overall confidence score
        alignment: Overall alignment score
        
    Returns:
        Tuple of (is_opportunity, reason, strength)
    """
    if not analyses or len(analyses) < 2:
        return False, "Insufficient data", 0.0
    
    # Group analyses by timeframe
    timeframe_groups = _group_analyses_by_timeframe(analyses)
    
    # 1. Check for strong trending signals 
    trending_opportunity = _check_trending_opportunity(timeframe_groups, confidence)
    if trending_opportunity[0]:
        return trending_opportunity
    
    # 2. Check for reversal patterns
    reversal_opportunity = _check_reversal_opportunity(timeframe_groups, confidence, alignment)
    if reversal_opportunity[0]:
        return reversal_opportunity
        
    # 3. Check for breakout patterns
    breakout_opportunity = _check_breakout_opportunity(timeframe_groups, confidence)
    if breakout_opportunity[0]:
        return breakout_opportunity
        
    # 4. Check for accumulation/distribution completion
    completion_opportunity = _check_completion_opportunity(timeframe_groups)
    if completion_opportunity[0]:
        return completion_opportunity
    
    # No opportunity detected
    return False, "No clear opportunity", 0.0

def _check_trending_opportunity(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]], 
                               confidence: float) -> Tuple[bool, str, float]:
    """Check for strong trending market opportunity."""
    # Focus on short-term and intermediate timeframes
    short_term = timeframe_groups.get('short', [])
    intermediate = timeframe_groups.get('intermediate', [])
    
    if not short_term or not intermediate:
        return False, "Missing timeframe data", 0.0
    
    # Get the dominant bias direction from short and intermediate timeframes
    short_term_bullish = sum(1 for a in short_term if a.momentum_bias == MultiTimeframeDirection.BULLISH)
    short_term_bearish = sum(1 for a in short_term if a.momentum_bias == MultiTimeframeDirection.BEARISH)
    
    int_term_bullish = sum(1 for a in intermediate if a.momentum_bias == MultiTimeframeDirection.BULLISH)
    int_term_bearish = sum(1 for a in intermediate if a.momentum_bias == MultiTimeframeDirection.BEARISH)
    
    # Check for strong volume confirmation
    high_volume_count = sum(1 for a in short_term + intermediate if a.volume_strength > 0.6)
    
    # Check if there's strong trending action in the same direction
    trending_actions = [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]
    trending_action_count = sum(1 for a in short_term + intermediate if a.dominant_action in trending_actions)
    
    # Calculate trend strength and direction
    short_term_dir = "bullish" if short_term_bullish > short_term_bearish else "bearish" if short_term_bearish > short_term_bullish else "neutral"
    int_term_dir = "bullish" if int_term_bullish > int_term_bearish else "bearish" if int_term_bearish > int_term_bullish else "neutral"
    
    # If we have alignment and strong trend signals
    if short_term_dir == int_term_dir and short_term_dir != "neutral":
        trend_strength = (short_term_bullish + int_term_bullish) / len(short_term + intermediate) if short_term_dir == "bullish" else \
                         (short_term_bearish + int_term_bearish) / len(short_term + intermediate)
        
        # Strong trend with good volume and trending action
        if (trend_strength > 0.6 and 
            high_volume_count >= max(1, len(short_term + intermediate) * 0.4) and
            trending_action_count >= max(1, len(short_term + intermediate) * 0.3) and
            confidence > 0.5):
            
            direction = "bullish" if short_term_dir == "bullish" else "bearish"
            strength = min(1.0, trend_strength * confidence * (1 + high_volume_count / len(short_term + intermediate)))
            
            return True, f"Strong {direction} trend with volume confirmation", strength
    
    return False, "No trending opportunity", 0.0

def _check_reversal_opportunity(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]],
                               confidence: float,
                               alignment: float) -> Tuple[bool, str, float]:
    """Check for potential market reversal opportunity."""
    # Short-term timeframes are most important for reversals
    short_term = timeframe_groups.get('short', [])
    intermediate = timeframe_groups.get('intermediate', [])
    
    if not short_term:
        return False, "Missing short-term timeframe data", 0.0
    
    # Look for changes in action/momentum in short timeframes vs intermediate
    if short_term and intermediate:
        # Get dominant bias
        short_term_bias = _get_dominant_direction(short_term)
        int_term_bias = _get_dominant_direction(intermediate)
        
        # Check for potential reversal: short timeframes showing opposite direction from intermediate
        if (short_term_bias != MultiTimeframeDirection.NEUTRAL and 
            int_term_bias != MultiTimeframeDirection.NEUTRAL and
            short_term_bias != int_term_bias):
            
            # Calculate strength based on short-term alignment
            short_term_alignment = sum(a.internal_alignment for a in short_term) / len(short_term)
            reversal_strength = short_term_alignment * confidence * 0.8
            
            # Check for volume confirmation
            high_volume_short = sum(1 for a in short_term if a.volume_strength > 0.65)
            if high_volume_short >= len(short_term) * 0.5:
                reversal_strength *= 1.2
            
            direction = "bullish" if short_term_bias == MultiTimeframeDirection.BULLISH else "bearish"
            
            if reversal_strength > 0.6 and confidence > 0.45:
                return True, f"Potential {direction} reversal forming", reversal_strength
    
    return False, "No reversal opportunity", 0.0

def _check_breakout_opportunity(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]],
                               confidence: float) -> Tuple[bool, str, float]:
    """Check for potential breakout/breakdown opportunity."""
    # For breakouts, look at phase transitions and recent volume surges
    
    # Combine short-term and intermediate for breakout detection
    recent_tfs = timeframe_groups.get('short', []) + timeframe_groups.get('intermediate', [])
    if not recent_tfs:
        return False, "Missing timeframe data", 0.0
    
    # Look for consolidation ending phases
    consolidation_actions = [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]
    accumulation_phases = [WyckoffPhase.ACCUMULATION, WyckoffPhase.RANGING]
    distribution_phases = [WyckoffPhase.DISTRIBUTION, WyckoffPhase.RANGING]
    
    # Count phases and actions
    consolidation_action_count = sum(1 for a in recent_tfs if a.dominant_action in consolidation_actions)
    accumulation_count = sum(1 for a in recent_tfs if a.dominant_phase in accumulation_phases)
    distribution_count = sum(1 for a in recent_tfs if a.dominant_phase in distribution_phases)
    
    # Get direction bias
    bias = _get_dominant_direction(recent_tfs)
    
    # High volume is crucial for breakouts
    high_volume_count = sum(1 for a in recent_tfs if a.volume_strength > 0.7)
    
    # A good breakout setup typically shows a shift from consolidation actions to a directional bias
    consolidation_ratio = consolidation_action_count / len(recent_tfs) if recent_tfs else 0
    recent_consolidation = consolidation_ratio >= 0.3  # At least 30% showing consolidation
    
    # Check for accumulation leading to bullish bias (potential breakout)
    if (accumulation_count >= len(recent_tfs) * 0.4 and 
        bias == MultiTimeframeDirection.BULLISH and
        high_volume_count >= max(1, len(recent_tfs) * 0.3) and
        recent_consolidation and  # Added consolidation condition
        confidence > 0.55):
        
        # Adjust strength based on consolidation - stronger breakouts often follow extensive consolidation
        consolidation_bonus = min(0.2, consolidation_ratio * 0.4)  # Max 20% bonus from consolidation
        breakout_strength = min(1.0, (confidence + consolidation_bonus) * (high_volume_count / len(recent_tfs)) * 1.3)
        return True, "Potential bullish breakout from accumulation", breakout_strength
    
    # Check for distribution leading to bearish bias (potential breakdown)
    elif (distribution_count >= len(recent_tfs) * 0.4 and 
          bias == MultiTimeframeDirection.BEARISH and
          high_volume_count >= max(1, len(recent_tfs) * 0.3) and
          recent_consolidation and  # Added consolidation condition
          confidence > 0.55):
        
        # Adjust strength based on consolidation - stronger breakdowns often follow extensive consolidation
        consolidation_bonus = min(0.2, consolidation_ratio * 0.4)  # Max 20% bonus from consolidation
        breakdown_strength = min(1.0, (confidence + consolidation_bonus) * (high_volume_count / len(recent_tfs)) * 1.3)
        return True, "Potential bearish breakdown from distribution", breakdown_strength
    
    return False, "No breakout opportunity", 0.0

def _check_completion_opportunity(timeframe_groups: Dict[str, List[TimeframeGroupAnalysis]]) -> Tuple[bool, str, float]:
    """Check for completion of accumulation/distribution phases."""
    # This captures later-stage setups where a pattern is completing
    
    short_term = timeframe_groups.get('short', [])
    intermediate = timeframe_groups.get('intermediate', [])
    
    if not short_term or not intermediate:
        return False, "Missing timeframe data", 0.0
    
    # Check for phase transitions
    # Short-term showing markup/markdown while intermediate still shows accumulation/distribution
    short_term_phases = [a.dominant_phase for a in short_term]
    short_term_actions = [a.dominant_action for a in short_term]
    int_term_phases = [a.dominant_phase for a in intermediate]
    
    # Completion of accumulation into markup
    markup_count = sum(1 for p in short_term_phases if p == WyckoffPhase.MARKUP)
    acc_count = sum(1 for p in int_term_phases if p == WyckoffPhase.ACCUMULATION)
    
    if markup_count >= len(short_term) * 0.5 and acc_count >= len(intermediate) * 0.5:
        # Look for high volume and markup action
        high_vol_markup = sum(1 for i, a in enumerate(short_term) 
                             if a.volume_strength > 0.65 and 
                                short_term_actions[i] == CompositeAction.MARKING_UP)
        
        if high_vol_markup >= 1:
            completion_strength = min(1.0, 0.7 + (high_vol_markup / len(short_term)) * 0.3)
            return True, "Accumulation phase completing, potential markup beginning", completion_strength
    
    # Completion of distribution into markdown
    markdown_count = sum(1 for p in short_term_phases if p == WyckoffPhase.MARKDOWN)
    dist_count = sum(1 for p in int_term_phases if p == WyckoffPhase.DISTRIBUTION)
    
    if markdown_count >= len(short_term) * 0.5 and dist_count >= len(intermediate) * 0.5:
        # Look for high volume and markdown action
        high_vol_markdown = sum(1 for i, a in enumerate(short_term) 
                               if a.volume_strength > 0.65 and 
                                  short_term_actions[i] == CompositeAction.MARKING_DOWN)
        
        if high_vol_markdown >= 1:
            completion_strength = min(1.0, 0.7 + (high_vol_markdown / len(short_term)) * 0.3)
            return True, "Distribution phase completing, potential markdown beginning", completion_strength
    
    return False, "No phase completion opportunity", 0.0

def _get_dominant_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Get the dominant direction from a list of analyses."""
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL
    
    bullish_count = sum(1 for a in analyses if a.momentum_bias == MultiTimeframeDirection.BULLISH)
    bearish_count = sum(1 for a in analyses if a.momentum_bias == MultiTimeframeDirection.BEARISH)
    
    if bullish_count > bearish_count:
        return MultiTimeframeDirection.BULLISH
    elif bearish_count > bullish_count:
        return MultiTimeframeDirection.BEARISH
    else:
        return MultiTimeframeDirection.NEUTRAL

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
