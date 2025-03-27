import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
from technical_analysis.wyckoff_types import SignificantLevelsData
from logging_utils import logger

from .mtf_direction import determine_overall_direction
from .mtf_alignment import calculate_overall_alignment, calculate_overall_confidence

from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, _TIMEFRAME_SETTINGS,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM, MODERATE_VOLUME_THRESHOLD,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT,
    DIRECTIONAL_WEIGHT, VOLUME_WEIGHT, PHASE_WEIGHT
)


def _is_phase_confirming_momentum(analysis: TimeframeGroupAnalysis) -> bool:
    """Check if the Wyckoff phase confirms the momentum bias."""
    if analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
        return is_bullish_phase(analysis.dominant_phase)
    elif analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
        return is_bearish_phase(analysis.dominant_phase)
    return False


def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState], coin: str, mid: float, significant_levels: Dict[Timeframe, SignificantLevelsData], interactive_analysis: bool
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across three timeframe groups.
    """
    if not states:
        return MultiTimeframeContext(
            description="No timeframe data available for analysis",
            should_notify=False
        )

    # Group timeframes into four categories
    short_term = {tf: state for tf, state in states.items() if tf in SHORT_TERM_TIMEFRAMES}
    intermediate = {tf: state for tf, state in states.items() if tf in INTERMEDIATE_TIMEFRAMES}
    long_term = {tf: state for tf, state in states.items() if tf in LONG_TERM_TIMEFRAMES}
    context = {tf: state for tf, state in states.items() if tf in CONTEXT_TIMEFRAMES}

    try:
        # Analyze all groups
        short_term_analysis = _analyze_timeframe_group(short_term)
        intermediate_analysis = _analyze_timeframe_group(intermediate)
        long_term_analysis = _analyze_timeframe_group(long_term)
        context_analysis = _analyze_timeframe_group(context)

        # Update weights
        short_term_analysis.group_weight = _calculate_group_weight(short_term)
        intermediate_analysis.group_weight = _calculate_group_weight(intermediate)
        long_term_analysis.group_weight = _calculate_group_weight(long_term)
        context_analysis.group_weight = _calculate_group_weight(context)

        # Calculate overall direction with all timeframes
        overall_direction = determine_overall_direction([
            short_term_analysis, 
            intermediate_analysis, 
            long_term_analysis,
            context_analysis
        ])

        # Calculate momentum intensity with all timeframes
        momentum_intensity = _calculate_momentum_intensity([
            short_term_analysis,
            intermediate_analysis,
            long_term_analysis,
            context_analysis
        ], overall_direction)

        # Calculate overall alignment across all groups
        all_analysis = AllTimeframesAnalysis(
            short_term=short_term_analysis,
            intermediate=intermediate_analysis,
            long_term=long_term_analysis,
            context=context_analysis,
            overall_direction=overall_direction,
            momentum_intensity=momentum_intensity,
            confidence_level= calculate_overall_confidence([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis,
                context_analysis
            ]),
            alignment_score= calculate_overall_alignment([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis,
                context_analysis
            ])
        )

        # Generate comprehensive description
        description = generate_all_timeframes_description(coin, all_analysis, mid, significant_levels, interactive_analysis)

        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.75"))

        choppy_conditions = all(group.volatility_state == VolatilityState.HIGH for group in [all_analysis.short_term, all_analysis.intermediate])
        
        # Enhanced notification criteria
        should_notify = (
            all_analysis.confidence_level >= min_confidence and 
            momentum_intensity >= WEAK_MOMENTUM and 
            not choppy_conditions and
            all_analysis.overall_direction != MultiTimeframeDirection.NEUTRAL and
            # Avoid uncertain phases in key timeframes
            not all_analysis.short_term.uncertain_phase and
            not all_analysis.intermediate.uncertain_phase and
            all_analysis.short_term.volume_strength >= MODERATE_VOLUME_THRESHOLD and
            all_analysis.intermediate.volume_strength >= MODERATE_VOLUME_THRESHOLD and
            all_analysis.short_term.dominant_phase != WyckoffPhase.RANGING and
            all_analysis.intermediate.dominant_phase != WyckoffPhase.RANGING
        )
        
        # Add direction-specific criteria
        if all_analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            # Bullish criteria - require stronger signals
            should_notify = should_notify and (
                # Ensure internal alignment is strong enough
                all_analysis.short_term.internal_alignment >= 0.50 and
                all_analysis.intermediate.internal_alignment >= 0.45 and
                # At least one of these bullish confirmation signals
                (all_analysis.short_term.dominant_phase == WyckoffPhase.MARKUP or 
                 all_analysis.short_term.dominant_phase == WyckoffPhase.ACCUMULATION or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.MARKUP or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.ACCUMULATION or
                 all_analysis.short_term.momentum_bias == MultiTimeframeDirection.BULLISH or
                 all_analysis.intermediate.momentum_bias == MultiTimeframeDirection.BULLISH)
            )
        else:  # BEARISH
            # Bearish criteria - adjusted thresholds for typical bearish behavior
            should_notify = should_notify and (
                # Still require reasonable alignment but less strict
                all_analysis.short_term.internal_alignment >= 0.50 and
                all_analysis.intermediate.internal_alignment >= 0.45 and
                # At least one of these bearish confirmation signals
                (all_analysis.short_term.dominant_phase == WyckoffPhase.MARKDOWN or 
                 all_analysis.short_term.dominant_phase == WyckoffPhase.DISTRIBUTION or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.MARKDOWN or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.DISTRIBUTION or
                 all_analysis.short_term.momentum_bias == MultiTimeframeDirection.BEARISH or
                 all_analysis.intermediate.momentum_bias == MultiTimeframeDirection.BEARISH)
            )

        return MultiTimeframeContext(
            description=description,
            should_notify=should_notify
        )
        
    except Exception as e:
        logger.error(e, exc_info=True)
        return MultiTimeframeContext(
            description=f"Error analyzing timeframes: {str(e)}",
            should_notify=False
        )

# Create a helper function to calculate volume-based weight adjustment
def get_volume_weight_adjustment(volume_state: VolumeState) -> float:
    """Return weight adjustment multiplier based on volume state"""
    if volume_state == VolumeState.VERY_HIGH:
        return 1.3
    elif volume_state == VolumeState.HIGH:
        return 1.1
    elif volume_state == VolumeState.LOW:
        return 0.9
    elif volume_state == VolumeState.VERY_LOW:
        return 0.8
    else:  # NEUTRAL or UNKNOWN
        return 1.0
    
# Create another helper for volume strength mapping
def get_base_volume_strength(volume_state: VolumeState) -> float:
    """Return base volume strength value based on volume state"""
    if volume_state == VolumeState.VERY_HIGH:
        return 1.5
    elif volume_state == VolumeState.HIGH:
        return 1.0
    elif volume_state == VolumeState.NEUTRAL:
        return 0.7
    elif volume_state == VolumeState.LOW:
        return 0.4
    elif volume_state == VolumeState.VERY_LOW:
        return 0.25
    else:  # UNKNOWN
        return 0.5

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
            volatility_state=VolatilityState.UNKNOWN
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    confident_phase_weights: Dict[WyckoffPhase, float] = {}  # Track confident phases separately
    uncertain_phase_weights: Dict[WyckoffPhase, float] = {}  # Track uncertain phases separately
    action_weights: Dict[CompositeAction, float] = {}
    uncertain_action_weights: Dict[CompositeAction, float] = {}  # Track uncertain actions
    total_weight = 0.0

    # Track exhaustion signals - symmetric treatment
    upside_exhaustion = 0
    downside_exhaustion = 0

    # Track rapid movement signals - symmetric treatment
    rapid_bullish_moves = 0.0
    rapid_bearish_moves = 0.0

    for tf, state in group.items():
        weight = tf.settings.phase_weight
            
        # Detect potential exhaustion based on phase combinations - symmetric detection
        if (state.phase == WyckoffPhase.MARKUP and 
            state.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]):
            upside_exhaustion += 1

        if (state.phase == WyckoffPhase.MARKDOWN and 
            state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.CONSOLIDATING]):
            downside_exhaustion += 1

        # Adjust weights based on exhaustion signals - symmetric treatment
        if state.is_upthrust:
            upside_exhaustion += 1
            weight *= 1.2
        elif state.is_spring:
            downside_exhaustion += 1
            weight *= 1.2

        # Optimize volume-based weight adjustment
        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            weight *= get_volume_weight_adjustment(state.volume)

        # Factor in effort vs result analysis - symmetric treatment
        if state.effort_vs_result == EffortResult.WEAK:
            if state.phase == WyckoffPhase.MARKUP:
                upside_exhaustion += 1
            elif state.phase == WyckoffPhase.MARKDOWN:
                downside_exhaustion += 1

        # Detect rapid price movements - optimized for new volume states
        # Use VERY_HIGH and HIGH volume states for detecting rapid moves
        if state.phase == WyckoffPhase.MARKUP and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            # Give stronger weight to VERY_HIGH volume
            rapid_bullish_moves += 1.5 if state.volume == VolumeState.VERY_HIGH else 1.0
        elif state.phase == WyckoffPhase.MARKDOWN and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            # Give stronger weight to VERY_HIGH volume
            rapid_bearish_moves += 1.5 if state.volume == VolumeState.VERY_HIGH else 1.0
        
        # Optimize trending weight boost with the same helper
        if state.pattern == MarketPattern.TRENDING and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            weight *= get_volume_weight_adjustment(state.volume)
        
        total_weight += weight

        # Improved phase weight handling with certain/uncertain separation
        if state.phase != WyckoffPhase.UNKNOWN:
            if state.uncertain_phase:
                # Store uncertain phases separately with reduced weight
                uncertain_phase_weights[state.phase] = uncertain_phase_weights.get(state.phase, 0) + (weight * 0.7)
            else:
                # Confident phases get full weight
                confident_phase_weights[state.phase] = confident_phase_weights.get(state.phase, 0) + weight
            
            # Track overall phase weights for determining dominant phase
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + (
                weight * 0.7 if state.uncertain_phase else weight
            )

        # Improved action weight handling
        if state.composite_action != CompositeAction.UNKNOWN:
            if state.uncertain_phase:  # Use phase uncertainty to affect action weight
                uncertain_action_weights[state.composite_action] = uncertain_action_weights.get(state.composite_action, 0) + (weight * 0.7)
            else:
                action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Determine dominant phase using combined weights
    if phase_weights:
        dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0]
        # Check if this phase is mostly from uncertain signals
        uncertain_weight = uncertain_phase_weights.get(dominant_phase, 0)
        confident_weight = confident_phase_weights.get(dominant_phase, 0)
        dominant_phase_is_uncertain = uncertain_weight > confident_weight
    else:
        dominant_phase = WyckoffPhase.UNKNOWN
        dominant_phase_is_uncertain = True

    # Determine dominant action with uncertainty handling
    if action_weights or uncertain_action_weights:
        # Combine both certain and uncertain action weights
        combined_action_weights = action_weights.copy()
        for action, weight in uncertain_action_weights.items():
            combined_action_weights[action] = combined_action_weights.get(action, 0) + weight
            
        dominant_action = max(combined_action_weights.items(), key=lambda x: x[1])[0]
    else:
        dominant_action = CompositeAction.UNKNOWN

    # Calculate internal alignment
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Enhanced volume strength calculation with improved volume state handling
    volume_factors = []
    total_volume_weight = 0.0
    
    for tf, state in group.items():
        # Use timeframe-specific settings for weighting
        tf_weight = tf.settings.phase_weight
        total_volume_weight += tf_weight
        
        # Use the helper for base strength determination
        base_strength = get_base_volume_strength(state.volume)

        # Adjust based on effort vs result - symmetric treatment
        if state.effort_vs_result == EffortResult.STRONG:
            base_strength *= 1.2
        elif state.effort_vs_result == EffortResult.WEAK:
            base_strength *= 0.8
            
        # Adjust for phase confirmation - symmetric treatment for markup/markdown
        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            if state.composite_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_strength *= 1.3
        
        # Consider timeframe-specific volume characteristics
        # Higher importance for longer timeframes and more extreme settings
        volume_importance = tf.settings.volume_ma_window / 20.0  # Normalize based on typical window size
        
        # Add the weighted volume factor to our list
        volume_factors.append((base_strength * volume_importance, tf_weight))
    
    # Calculate weighted average of volume factors
    volume_strength = (
        sum(factor * weight for factor, weight in volume_factors) / total_volume_weight 
        if total_volume_weight > 0 else 0.0
    )
    
    # Clamp volume strength between 0 and 1
    volume_strength = max(0.0, min(1.0, volume_strength))

    # Adjust volume strength based on exhaustion signals - symmetric treatment
    exhaustion_threshold = int(len(group) * 0.6)  # Increased from len(group) // 2 to be more conservative
    if upside_exhaustion >= exhaustion_threshold:
        volume_strength *= 0.7
    elif downside_exhaustion >= exhaustion_threshold:
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
    volatility_counts = {state.volatility: 0 for state in group.values()}

    for state in group.values():
        liquidity_counts[state.liquidity] += 1
        volatility_counts[state.volatility] += 1

    liquidity_state = max(liquidity_counts.items(), key=lambda x: x[1])[0]
    volatility_state = max(volatility_counts.items(), key=lambda x: x[1])[0]

    # Enhanced momentum bias calculation with clear signal weighting and transition handling
    bullish_signals = 0.0
    bearish_signals = 0.0
        
    for s in group.values():
        # Primary phase signals - core market structure signals
        if is_bullish_phase(s.phase) and upside_exhaustion < len(group) // 2:
            # Non-exhausted bullish phase
            bullish_signals += 1.0
        elif is_bearish_phase(s.phase) and downside_exhaustion < len(group) // 2:
            # Non-exhausted bearish phase
            bearish_signals += 1.0
        
        # Action signals - immediate behavior signals
        if is_bullish_action(s.composite_action):
            bullish_signals += 0.8  # Slightly less weight than phase
        elif is_bearish_action(s.composite_action):
            bearish_signals += 0.8
        
        # Transition signals - early reversal indicators
        if s.composite_action == CompositeAction.REVERSING:
            if s.phase == WyckoffPhase.MARKDOWN:
                bullish_signals += 0.7  # Reversal in downtrend - bullish
            elif s.phase == WyckoffPhase.MARKUP:
                bearish_signals += 0.7  # Reversal in uptrend - bearish
        
        # Sentiment signals - contra-indicators often suggest reversals
        if s.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE]:
            # Amplify signal based on volume state
            if s.volume == VolumeState.VERY_HIGH:
                bullish_signals += 0.8    # Stronger signal with very high volume
            elif s.volume == VolumeState.HIGH:
                bullish_signals += 0.6    # Regular signal with high volume
            elif s.volume != VolumeState.VERY_LOW:  # Skip very low volume
                bullish_signals += 0.4    # Weaker signal with normal/low volume
        elif s.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE]:
            # Amplify signal based on volume state
            if s.volume == VolumeState.VERY_HIGH:
                bearish_signals += 0.8    # Stronger signal with very high volume
            elif s.volume == VolumeState.HIGH:
                bearish_signals += 0.6    # Regular signal with high volume
            elif s.volume != VolumeState.VERY_LOW:  # Skip very low volume
                bearish_signals += 0.4    # Weaker signal with normal/low volume
        
        # Effort-result signals - efficiency and exhaustion indicators
        if s.effort_vs_result == EffortResult.WEAK:
            if is_bearish_phase(s.phase):
                bullish_signals += 0.4  # Weak selling effort can signal bullish potential
            elif is_bullish_phase(s.phase):
                bearish_signals += 0.4  # Weak buying effort can signal bearish potential
        
        # Volatility signals - often indicate potential change of character
        if s.volatility == VolatilityState.HIGH:
            if is_bearish_phase(s.phase):
                bullish_signals += 0.3  # High volatility in bearish phase can signal capitulation
            elif is_bullish_phase(s.phase):
                bearish_signals += 0.3  # High volatility in bullish phase can signal blow-off top
                
        # Ranging phases get partial credit to both sides - acknowledging uncertainty
        if s.phase == WyckoffPhase.RANGING:
            # Split signal between both directions but with lower weight
            bullish_signals += 0.2
            bearish_signals += 0.2

    # Normalize signals by count of timeframes for consistency
    if len(group) > 0:
        bullish_signals /= len(group)
        bearish_signals /= len(group)

    # Adjust momentum bias based on exhaustion signals - symmetric treatment
    consolidation_count = sum(1 for s in group.values() if s.composite_action == CompositeAction.CONSOLIDATING)
    total_signals = len(group)

    # Phase dominance check - ensure momentum bias respects dominant phase
    # Balanced treatment for bullish and bearish phases
    phase_momentum_override = None
    if is_bearish_phase(dominant_phase):
        # Strong bearish phase should limit bullish bias
        if bullish_signals > bearish_signals and not dominant_phase_is_uncertain:
            bearish_signals = max(bearish_signals, bullish_signals * 0.8)
            phase_momentum_override = MultiTimeframeDirection.BEARISH
    elif is_bullish_phase(dominant_phase):
        # Strong bullish phase should limit bearish bias  
        if bearish_signals > bullish_signals and not dominant_phase_is_uncertain:
            bullish_signals = max(bullish_signals, bearish_signals * 0.8)
            phase_momentum_override = MultiTimeframeDirection.BULLISH

    # Modified momentum bias calculation - ensure bearish gets equal treatment
    if consolidation_count / total_signals > 0.5:
        momentum_bias = MultiTimeframeDirection.NEUTRAL
    else:
        # Equal threshold for both directions
        threshold_diff = 0.1
        momentum_bias = (
            MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals + threshold_diff else
            MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals + threshold_diff else
            MultiTimeframeDirection.NEUTRAL
        )

    # Modify momentum bias calculation for rapid moves - symmetric treatment
    if rapid_bullish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BULLISH
    elif rapid_bearish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BEARISH
        
    # Apply phase override after all other checks
    if phase_momentum_override is not None:
        momentum_bias = phase_momentum_override

    return TimeframeGroupAnalysis(
        dominant_phase=dominant_phase,
        uncertain_phase=dominant_phase_is_uncertain,
        dominant_action=dominant_action,
        internal_alignment=internal_alignment,
        volume_strength=volume_strength,
        momentum_bias=momentum_bias,
        group_weight=total_weight,
        funding_sentiment=funding_sentiment,
        liquidity_state=liquidity_state,
        volatility_state=volatility_state
    )


def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group based on phase weights."""
    return sum(tf.settings.phase_weight for tf in timeframes.keys())

def _calculate_momentum_intensity(analyses: List[TimeframeGroupAnalysis], overall_direction: MultiTimeframeDirection) -> float:
    """Calculate momentum intensity with optimized scoring for faster intraday response."""
    if not analyses or overall_direction == MultiTimeframeDirection.NEUTRAL:
        return 0.0

    # Check for phase inconsistency with improved efficiency
    phase_direction_conflicts = sum(
        1 for analysis in analyses 
        if ((overall_direction == MultiTimeframeDirection.BULLISH and is_bearish_phase(analysis.dominant_phase)) or
            (overall_direction == MultiTimeframeDirection.BEARISH and is_bullish_phase(analysis.dominant_phase)))
    )
    
    # Reduced penalty for conflicts to allow faster direction changes
    conflict_ratio = phase_direction_conflicts / len(analyses) if analyses else 0
    conflict_penalty = max(0.5, 1.0 - conflict_ratio * 0.5)

    # Optimize directional score calculation
    directional_scores = []
    short_term_volume = 0.0
    short_term_count = 0
    
    # Score calculation with simplified logic - map calculation to a function
    for analysis in analyses:
        # Base score
        if analysis.momentum_bias == overall_direction:
            base_score = 1.0
        elif analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL:
            base_score = 0.5
        else:
            base_score = 0.0
        
        # Phase alignment
        phase_aligned = (
            (overall_direction == MultiTimeframeDirection.BULLISH and is_bullish_phase(analysis.dominant_phase)) or
            (overall_direction == MultiTimeframeDirection.BEARISH and is_bearish_phase(analysis.dominant_phase))
        )
        
        # Calculate all boosts at once
        phase_consistency = 0.7 if not phase_aligned else 1.0
        volume_boost = 1.0 + (analysis.volume_strength * 0.35)
        phase_boost = 1.2 if phase_aligned else 1.0
        
        # Funding impact
        funding_aligned = (
            (analysis.funding_sentiment > 0 and overall_direction == MultiTimeframeDirection.BULLISH) or
            (analysis.funding_sentiment < 0 and overall_direction == MultiTimeframeDirection.BEARISH)
        )
        funding_impact = abs(analysis.funding_sentiment) * 0.2
        funding_boost = (1.0 + funding_impact) if funding_aligned else (1.0 - funding_impact)
        
        # Combined score calculation
        score = base_score * phase_consistency * volume_boost * phase_boost * funding_boost
        directional_scores.append((score, analysis.group_weight))
        
        # Track short timeframe volume for boost
        if analysis.group_weight in {tf.settings.phase_weight for tf in SHORT_TERM_TIMEFRAMES}:
            short_term_volume += analysis.volume_strength
            short_term_count += 1
    
    # More aggressive volume boost calculation
    hourly_volume_boost = 1.0
    if short_term_count > 0:
        avg_short_term_volume = short_term_volume / short_term_count
        hourly_volume_boost = 1.0 + max(0, min(0.2, avg_short_term_volume - 0.4))  # Increased from 0.15 and lowered threshold
    
    # Final calculation with error handling
    total_weight = sum(weight for _, weight in directional_scores)
    if total_weight == 0:
        return 0.0
        
    weighted_momentum = sum(score * weight for score, weight in directional_scores) / total_weight
    
    # Improved final value calculation with smoother scaling
    final_momentum = weighted_momentum * hourly_volume_boost * conflict_penalty
    
    # Apply less steep curve for more responsive results
    final_momentum = 1.0 / (1.0 + np.exp(-4 * (final_momentum - 0.45))) if final_momentum > 0 else 0.0
    
    return min(1.0, final_momentum)
