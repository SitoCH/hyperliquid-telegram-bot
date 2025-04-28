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
    WyckoffState, WyckoffPhase, WyckoffSign, MarketPattern, _TIMEFRAME_SETTINGS,
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity
)

from .wyckoff_multi_timeframe_types import (
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM, MODERATE_VOLUME_THRESHOLD,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT
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
            confidence_level= calculate_overall_confidence(
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis,
                context_analysis
            ),
            alignment_score= calculate_overall_alignment(short_term_analysis, intermediate_analysis)
        )

        # Generate comprehensive description
        description = generate_all_timeframes_description(coin, all_analysis, mid, significant_levels, interactive_analysis)

        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.75"))
        
        notification_checks = [
            (all_analysis.confidence_level >= min_confidence, f"Low confidence: {all_analysis.confidence_level:.2f} < {min_confidence:.2f}"),
            (momentum_intensity >= WEAK_MOMENTUM, f"Weak momentum: {momentum_intensity:.2f} < {WEAK_MOMENTUM:.2f}"),
            (all_analysis.intermediate.volatility_state != VolatilityState.HIGH, "Choppy intermediate market conditions detected"),
            (all_analysis.overall_direction != MultiTimeframeDirection.NEUTRAL, "Neutral market direction"),
            (all_analysis.short_term.dominant_sign != WyckoffSign.SECONDARY_TEST_RESISTANCE, "Secondary Test Resistance (STR) on short-term"),
            (all_analysis.short_term.dominant_sign != WyckoffSign.SECONDARY_TEST, "Secondary Test (ST) on short-term"),
            # Be more lenient with short-term uncertainty if intermediate is certain
            ((not all_analysis.short_term.uncertain_phase or not all_analysis.intermediate.uncertain_phase), 
             "Both short and intermediate timeframes show uncertain phases"),
            (all_analysis.intermediate.volume_strength >= MODERATE_VOLUME_THRESHOLD, 
             f"Low intermediate volume strength: {all_analysis.intermediate.volume_strength:.2f} < {MODERATE_VOLUME_THRESHOLD:.2f}"),
            (all_analysis.short_term.volume_strength >= MODERATE_VOLUME_THRESHOLD * 0.9, 
             f"Low short-term volume strength: {all_analysis.short_term.volume_strength:.2f} < {MODERATE_VOLUME_THRESHOLD * 0.9:.2f}"),
            # Instead of checking only the dominant phase, check if any phase is not ranging
            ((all_analysis.short_term.dominant_phase != WyckoffPhase.RANGING or 
              all_analysis.intermediate.dominant_phase != WyckoffPhase.RANGING), 
             "Both short and intermediate timeframes are ranging"),
        ]
        
        should_notify = all(check[0] for check in notification_checks)
        
        # Log reasons for not notifying
        if not should_notify:
            reasons = [reason for condition, reason in notification_checks if not condition]
            logger.info(f"Notification suppressed for {coin} due to following reasons:")
            for reason in reasons:
                logger.info(f"- {reason}")
        
        # Add direction-specific criteria
        if should_notify and all_analysis.overall_direction == MultiTimeframeDirection.BULLISH:
            bullish_checks = [
                (all_analysis.short_term.internal_alignment >= 0.50, f"Low short-term alignment: {all_analysis.short_term.internal_alignment:.2f} < 0.50"),
                (all_analysis.intermediate.internal_alignment >= 0.45, f"Low intermediate alignment: {all_analysis.intermediate.internal_alignment:.2f} < 0.45"),
                (all_analysis.short_term.dominant_phase == WyckoffPhase.MARKUP or 
                 all_analysis.short_term.dominant_phase == WyckoffPhase.ACCUMULATION or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.MARKUP or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.ACCUMULATION or
                 all_analysis.short_term.momentum_bias == MultiTimeframeDirection.BULLISH or
                 all_analysis.intermediate.momentum_bias == MultiTimeframeDirection.BULLISH, 
                 "Missing bullish confirmation signals")
            ]
            
            should_notify = all(check[0] for check in bullish_checks)
            
            if not should_notify:
                reasons = [reason for condition, reason in bullish_checks if not condition]
                logger.info(f"Bullish notification suppressed for {coin} due to following reasons:")
                for reason in reasons:
                    logger.info(f"- {reason}")
                
        elif should_notify:  # BEARISH
            bearish_checks = [
                (all_analysis.short_term.internal_alignment >= 0.50, f"Low short-term alignment: {all_analysis.short_term.internal_alignment:.2f} < 0.50"),
                (all_analysis.intermediate.internal_alignment >= 0.45, f"Low intermediate alignment: {all_analysis.intermediate.internal_alignment:.2f} < 0.45"),
                (all_analysis.short_term.dominant_phase == WyckoffPhase.MARKDOWN or 
                 all_analysis.short_term.dominant_phase == WyckoffPhase.DISTRIBUTION or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.MARKDOWN or
                 all_analysis.intermediate.dominant_phase == WyckoffPhase.DISTRIBUTION or
                 all_analysis.short_term.momentum_bias == MultiTimeframeDirection.BEARISH or
                 all_analysis.intermediate.momentum_bias == MultiTimeframeDirection.BEARISH,
                 "Missing bearish confirmation signals")
            ]
            
            should_notify = all(check[0] for check in bearish_checks)
            
            if not should_notify:
                reasons = [reason for condition, reason in bearish_checks if not condition]
                logger.info(f"Bearish notification suppressed for {coin} due to following reasons:")
                for reason in reasons:
                    logger.info(f"- {reason}")

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
            volatility_state=VolatilityState.UNKNOWN,
            dominant_sign=WyckoffSign.NONE
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    confident_phase_weights: Dict[WyckoffPhase, float] = {}  # Track confident phases separately
    uncertain_phase_weights: Dict[WyckoffPhase, float] = {}  # Track uncertain phases separately
    action_weights: Dict[CompositeAction, float] = {}
    uncertain_action_weights: Dict[CompositeAction, float] = {}  # Track uncertain actions separately
    
    # Track Wyckoff signs
    sign_weights: Dict[WyckoffSign, float] = {}
    
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

        # Detect rapid price movements - optimized for new volume state
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
                
        # Track Wyckoff signs with weights
        if state.wyckoff_sign != WyckoffSign.NONE:
            # Give additional weight to significant signs
            sign_weight = weight
            
            # Boost weight for critical signs in certain market conditions
            if state.wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.BUYING_CLIMAX]:
                sign_weight *= 1.3  # Climactic events are very important
            elif state.wyckoff_sign in [WyckoffSign.SIGN_OF_STRENGTH, WyckoffSign.SIGN_OF_WEAKNESS]:
                sign_weight *= 1.2  # SOS/SOW are key directional signals
            elif state.wyckoff_sign in [WyckoffSign.UPTHRUST, WyckoffSign.SECONDARY_TEST]:
                sign_weight *= 1.1  # Tests are important but less decisive
                
            # Further boost for signs confirmed by volume
            if state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
                sign_weight *= 1.2
                
            # Add to sign weights dictionary
            sign_weights[state.wyckoff_sign] = sign_weights.get(state.wyckoff_sign, 0) + sign_weight

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
        
    # Determine dominant Wyckoff sign
    dominant_sign = WyckoffSign.NONE
    if sign_weights:
        # Get most heavily weighted sign
        dominant_sign = max(sign_weights.items(), key=lambda x: x[1])[0]
        
        # Calculate sign confidence as a percentage of total weight
        sign_weight = sign_weights[dominant_sign]
        sign_confidence = sign_weight / total_weight if total_weight > 0 else 0
        
        # Only accept as dominant if it represents a significant portion of signals
        if sign_confidence < 0.25:  # If the sign represents less than 25% of all signals
            dominant_sign = WyckoffSign.NONE

    # Calculate internal alignment
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Enhanced volume strength calculation with improved volume state handling
    volume_factors = []
    total_volume_weight = 0.0
    has_very_high_volume = any(state.volume == VolumeState.VERY_HIGH for state in group.values())

    for tf, state in group.items():
        # Use timeframe-specific settings for weighting
        tf_weight = tf.settings.phase_weight
        total_volume_weight += tf_weight

        # Use the helper for base strength determination
        base_strength = get_base_volume_strength(state.volume)
        
        # Enhanced phase-specific volume interpretation for crypto markets
        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            base_strength *= 1.2
            # Higher multiplier for active trend phases - crypto moves fast
            if state.composite_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_strength *= 1.3
                
        # Special handling for accumulation/distribution with high volume
        if state.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.DISTRIBUTION] and state.volume in [VolumeState.HIGH, VolumeState.VERY_HIGH]:
            # High volume in accumulation/distribution often signals impending breakout
            base_strength *= 1.15
        
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

    # Better normalization for crypto's extreme volume spikes
    if has_very_high_volume:
        # Apply sigmoid-like normalization to prevent overvaluing extreme spikes
        # but maintain their significance
        volume_strength = 0.7 + (0.3 * (1.0 / (1.0 + np.exp(-5 * (volume_strength - 0.6)))))

    # Clamp volume strength between 0 and 1
    volume_strength = max(0.0, min(1.0, volume_strength))

    # Improved exhaustion detection for crypto markets
    exhaustion_threshold = int(len(group) * 0.5)  # More sensitive than before (was 0.6)
    severe_exhaustion_threshold = int(len(group) * 0.75)

    # Different penalties based on severity
    if upside_exhaustion >= severe_exhaustion_threshold or downside_exhaustion >= severe_exhaustion_threshold:
        volume_strength *= 0.6  # Severe exhaustion (stronger penalty)
    elif upside_exhaustion >= exhaustion_threshold or downside_exhaustion >= exhaustion_threshold:
        volume_strength *= 0.75  # Moderate exhaustion

    # Check for divergence between volume and price - common in crypto turning points
    divergence_count = sum(1 for s in group.values() if
        (s.effort_vs_result == EffortResult.WEAK and s.volume in [VolumeState.HIGH, VolumeState.VERY_HIGH]))

    # Apply penalty for significant divergence
    if divergence_count >= len(group) // 3:
        volume_strength *= 0.85

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
            
        # Consider Wyckoff signs for directional signals
        if s.wyckoff_sign == WyckoffSign.SELLING_CLIMAX or s.wyckoff_sign == WyckoffSign.AUTOMATIC_RALLY:
            bullish_signals += 0.6  # These typically indicate potential bottoming
        elif s.wyckoff_sign == WyckoffSign.SIGN_OF_STRENGTH or s.wyckoff_sign == WyckoffSign.LAST_POINT_OF_SUPPORT:
            bullish_signals += 0.8  # Strong bullish signals
        elif s.wyckoff_sign == WyckoffSign.BUYING_CLIMAX or s.wyckoff_sign == WyckoffSign.UPTHRUST:
            bearish_signals += 0.6  # These typically indicate potential topping
        elif s.wyckoff_sign == WyckoffSign.SIGN_OF_WEAKNESS or s.wyckoff_sign == WyckoffSign.LAST_POINT_OF_RESISTANCE:
            bearish_signals += 0.8  # Strong bearish signals

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
        volatility_state=volatility_state,
        dominant_sign=dominant_sign
    )


def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group based on phase weights."""
    # Simply use the configured weights directly from settings
    return sum(tf.settings.phase_weight for tf in timeframes.keys())


def _calculate_momentum_intensity(analyses: List[TimeframeGroupAnalysis], overall_direction: MultiTimeframeDirection) -> float:
    """Calculate momentum intensity with optimized scoring for faster intraday response."""
    if not analyses or overall_direction == MultiTimeframeDirection.NEUTRAL:
        return 0.0

    # Check for phase inconsistency with improved efficiency
    phase_direction_conflicts = sum(
        1 for analysis in analyses if 
        ((overall_direction == MultiTimeframeDirection.BULLISH and is_bearish_phase(analysis.dominant_phase)) or
         (overall_direction == MultiTimeframeDirection.BEARISH and is_bullish_phase(analysis.dominant_phase)))
    )

    # Reduced penalty for conflicts to allow faster direction changes
    conflict_ratio = phase_direction_conflicts / len(analyses) if analyses else 0
    conflict_penalty = max(0.5, 1.0 - conflict_ratio * 0.5)
            
    # Score calculation with simplified logic - map calculation to a function
    directional_scores = []
    short_term_volume = 0.0
    short_term_count = 0
    
    for analysis in analyses:
        # Base score
        if analysis.momentum_bias == overall_direction:
            base_score = 1.0
        elif analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL:
            base_score = 0.5
        else:
            base_score = 0.0
        
        # Apply uncertainty adjustment uniformly across all timeframes 
        if analysis.uncertain_phase and analysis.momentum_bias != overall_direction:
            base_score *= 0.6  # Reduce impact of uncertain contradicting signals
        
        # Phase alignment
        phase_aligned = _is_phase_confirming_momentum(analysis)
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
    
    # Improved sigmoid function for more responsive hourly updates
    # Increased steepness from 4 to 5 and adjusted center point from 0.45 to 0.42
    # This makes the function more sensitive in the mid-range while maintaining balanced extremes
    final_momentum = 1.0 / (1.0 + np.exp(-5 * (final_momentum - 0.42))) if final_momentum > 0 else 0.0
    return min(1.0, final_momentum)