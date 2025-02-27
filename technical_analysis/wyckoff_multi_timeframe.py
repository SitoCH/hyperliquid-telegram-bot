import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
from .wyckoff_types import SignificantLevelsData
from logging_utils import logger


from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, _TIMEFRAME_SETTINGS,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

from .wyckoff_multi_timeframe_types import (
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT,
    DIRECTIONAL_WEIGHT, VOLUME_WEIGHT, PHASE_WEIGHT
)

def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return timeframe.settings.phase_weight


def _is_phase_confirming_momentum(analysis: TimeframeGroupAnalysis) -> bool:
    """Check if the Wyckoff phase confirms the momentum bias."""
    bullish_phases = {WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION}
    bearish_phases = {WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION}
    
    if analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
        return analysis.dominant_phase in bullish_phases
    elif analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
        return analysis.dominant_phase in bearish_phases
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
        overall_direction = _determine_overall_direction([
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
            confidence_level=_calculate_overall_confidence([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis,
                context_analysis
            ]),
            alignment_score=_calculate_overall_alignment([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis,
                context_analysis
            ])
        )

        # Generate comprehensive description
        description = generate_all_timeframes_description(coin, all_analysis, mid, significant_levels, interactive_analysis)

        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.75"))
        should_notify = (all_analysis.confidence_level >= min_confidence and 
            momentum_intensity > MODERATE_MOMENTUM and 
            (all_analysis.short_term.volatility_state != VolatilityState.HIGH or all_analysis.intermediate.volatility_state != VolatilityState.HIGH) and
            all_analysis.overall_direction != MultiTimeframeDirection.NEUTRAL)

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
            liquidation_risk=LiquidationRisk.UNKNOWN,
            volatility_state=VolatilityState.UNKNOWN
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    confident_phase_weights: Dict[WyckoffPhase, float] = {}  # Track confident phases separately
    uncertain_phase_weights: Dict[WyckoffPhase, float] = {}  # Track uncertain phases separately
    action_weights: Dict[CompositeAction, float] = {}
    uncertain_action_weights: Dict[CompositeAction, float] = {}  # Track uncertain actions
    total_weight = 0.0

    # Track exhaustion signals
    upside_exhaustion = 0
    downside_exhaustion = 0

    # Track rapid movement signals
    rapid_bullish_moves = 0
    rapid_bearish_moves = 0

    for tf, state in group.items():
        weight = get_phase_weight(tf)
            
        # Detect potential exhaustion based on phase combinations
        if (state.phase == WyckoffPhase.MARKUP and 
            state.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]):
            upside_exhaustion += 1

        if (state.phase == WyckoffPhase.MARKDOWN and 
            state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.CONSOLIDATING]):
            downside_exhaustion += 1

        # Adjust weights based on exhaustion signals
        if state.is_upthrust:
            upside_exhaustion += 1
            weight *= 1.2  # Increase weight for potential reversal signals
        elif state.is_spring:
            downside_exhaustion += 1
            weight *= 1.2

        # Reduce weight if we see contrary volume signals
        if (state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN] and 
            state.volume == VolumeState.LOW):
            weight *= 0.8

        # Factor in effort vs result analysis
        if state.effort_vs_result == EffortResult.WEAK:
            if state.phase == WyckoffPhase.MARKUP:
                upside_exhaustion += 1
            elif state.phase == WyckoffPhase.MARKDOWN:
                downside_exhaustion += 1

        # Detect rapid price movements
        if state.phase == WyckoffPhase.MARKUP and state.volume == VolumeState.HIGH:
            rapid_bullish_moves += 1
        elif state.phase == WyckoffPhase.MARKDOWN and state.volume == VolumeState.HIGH:
            rapid_bearish_moves += 1

        # Increase weight for strong directional moves
        if state.pattern == MarketPattern.TRENDING:
            if state.volume == VolumeState.HIGH:
                weight *= 1.3  # 30% boost for high volume trends
        
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

    # Enhanced volume strength calculation
    volume_factors = []
    for state in group.values():
        base_strength = 1.0 if state.volume == VolumeState.HIGH else 0.5

        # Adjust based on effort vs result
        if state.effort_vs_result == EffortResult.STRONG:
            base_strength *= 1.2
        elif state.effort_vs_result == EffortResult.WEAK:
            base_strength *= 0.8
            
        # Adjust for phase confirmation
        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            if state.composite_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_strength *= 1.3
                
        volume_factors.append(base_strength)

    volume_strength = sum(volume_factors) / len(volume_factors)

    # Clamp volume strength between 0 and 1
    volume_strength = max(0.0, min(1.0, volume_strength))

    # Adjust volume strength based on exhaustion signals
    if upside_exhaustion >= len(group) // 2:
        volume_strength *= 0.7  # Reduce volume significance if exhaustion detected
    elif downside_exhaustion >= len(group) // 2:
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
    risk_counts = {state.liquidation_risk: 0 for state in group.values()}
    volatility_counts = {state.volatility: 0 for state in group.values()}

    for state in group.values():
        liquidity_counts[state.liquidity] += 1
        risk_counts[state.liquidation_risk] += 1
        volatility_counts[state.volatility] += 1

    liquidity_state = max(liquidity_counts.items(), key=lambda x: x[1])[0]
    liquidation_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
    volatility_state = max(volatility_counts.items(), key=lambda x: x[1])[0]

    # Enhanced momentum bias calculation with exhaustion consideration
    bullish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] and upside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.MARKING_UP] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKDOWN) or
        (s.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKDOWN)
    ))

    bearish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] and downside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.MARKING_DOWN] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKUP) or
        (s.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKUP)
    ))

    # Adjust momentum bias based on exhaustion signals
    consolidation_count = sum(1 for s in group.values() if s.composite_action == CompositeAction.CONSOLIDATING)
    total_signals = len(group)

    # Modified momentum bias calculation
    if consolidation_count / total_signals > 0.5:
        momentum_bias = MultiTimeframeDirection.NEUTRAL
    else:
        # Factor in exhaustion signals
        if upside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bullish_signals > bearish_signals else MultiTimeframeDirection.BEARISH
        elif downside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bearish_signals > bullish_signals else MultiTimeframeDirection.BULLISH
        else:
            momentum_bias = (
                MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals else
                MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals else
                MultiTimeframeDirection.NEUTRAL
            )

    # Modify momentum bias calculation
    if rapid_bullish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BULLISH
    elif rapid_bearish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BEARISH

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
        liquidation_risk=liquidation_risk,
        volatility_state=volatility_state
    )

def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group based on phase weights."""
    return sum(get_phase_weight(tf) for tf in timeframes.keys())

def _calculate_overall_alignment(analyses: List[TimeframeGroupAnalysis]) -> float:
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
            elif (analysis1.dominant_phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] and
                  analysis2.dominant_phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP]) or \
                 (analysis1.dominant_phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] and
                  analysis2.dominant_phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN]):
                phase_aligned = 0.5  # General agreement on bullish/bearish phase

            # Action Alignment Scoring: Refined logic
            action_aligned = 0.0
            if analysis1.dominant_action == analysis2.dominant_action:
                action_aligned = 1.0
            elif (analysis1.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.ACCUMULATING] and
                  analysis2.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.ACCUMULATING]) or \
                 (analysis1.dominant_action in [CompositeAction.MARKING_DOWN, CompositeAction.DISTRIBUTING] and
                  analysis2.dominant_action in [CompositeAction.MARKING_DOWN, CompositeAction.DISTRIBUTING]):
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

def _calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
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
    alignment_score = _calculate_overall_alignment(analyses)

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

def _determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall direction considering Wyckoff phase weights for each timeframe group."""
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL

    # Group timeframes by their actual settings in _TIMEFRAME_SETTINGS
    timeframe_groups = {
        'short': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                           for tf in SHORT_TERM_TIMEFRAMES}],
        'mid': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                         for tf in INTERMEDIATE_TIMEFRAMES}],
        'long': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                          for tf in LONG_TERM_TIMEFRAMES}]
    }

    def get_weighted_direction(group: List[TimeframeGroupAnalysis]) -> Tuple[MultiTimeframeDirection, float, float]:
        if not group:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0

        group_total_weight = sum(a.group_weight for a in group)
        if group_total_weight == 0:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0

        weighted_signals = {
            direction: sum(
                (a.group_weight / group_total_weight) * 
                (1 + a.volume_strength * 0.3) *  # Volume boost
                (1.2 if a.volatility_state == VolatilityState.HIGH else 1.0)  # Volatility adjustment
                for a in group
                if a.momentum_bias == direction
            )
            for direction in MultiTimeframeDirection
        }
        
        strongest = max(weighted_signals.items(), key=lambda x: x[1])
        avg_volume = sum(a.volume_strength for a in group) / len(group)

        return strongest[0], strongest[1], avg_volume

    # Get weighted directions with volume context
    st_dir, st_weight, st_vol = get_weighted_direction(timeframe_groups['short'])
    mid_dir, mid_weight, mid_vol = get_weighted_direction(timeframe_groups['mid'])
    lt_dir, lt_weight, _ = get_weighted_direction(timeframe_groups['long'])

    # Check for high-conviction intraday moves first
    if st_dir != MultiTimeframeDirection.NEUTRAL:
        if st_weight > 0.8 and st_vol > 0.7:  # Strong short-term move with volume
            if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_dir != st_dir:
                return MultiTimeframeDirection.NEUTRAL  # Conflict with intermediate trend
            return st_dir

    # Check for strong intermediate trend
    if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_weight > 0.7:
        # Allow counter-trend short-term moves if volume is low
        if st_dir != MultiTimeframeDirection.NEUTRAL and st_dir != mid_dir:
            if st_vol < 0.5:  # Low volume counter-trend move
                return mid_dir
            return MultiTimeframeDirection.NEUTRAL  # High volume conflict
        return mid_dir

    # Consider longer-term trend with confirmation
    if lt_dir != MultiTimeframeDirection.NEUTRAL and lt_weight > 0.6:
        # Need confirmation from either shorter timeframe
        if mid_dir == lt_dir or st_dir == lt_dir:
            return lt_dir
        # Check if counter-trend moves are weak
        if mid_vol < 0.5 and st_vol < 0.5:
            return lt_dir

    # Check for aligned moves even with lower weights
    if st_dir == mid_dir and mid_dir == lt_dir and st_dir != MultiTimeframeDirection.NEUTRAL:
        if (st_weight + mid_weight + lt_weight) / 3 > 0.5:  # Average weight threshold
            return st_dir

    return MultiTimeframeDirection.NEUTRAL

def _calculate_momentum_intensity(analyses: List[TimeframeGroupAnalysis], overall_direction: MultiTimeframeDirection) -> float:
    """
    Calculate momentum intensity across all timeframe groups with hourly optimization.
    
    Args:
        analyses: List of timeframe group analyses
        overall_direction: Previously determined overall market direction
    
    Returns:
        float: Momentum intensity score (0.0 to 1.0)
    """
    if not analyses:
        return 0.0

    # Get directional scores for each timeframe group
    directional_scores = []
    hourly_volume_boost = 1.0  # Will be adjusted based on short-term volume

    for analysis in analyses:
        # Base directional alignment with overall trend
        if analysis.momentum_bias == overall_direction:
            base_score = 1.0
        elif analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL:
            base_score = 0.5
        else:
            base_score = 0.0

        # Volume confirmation boost
        volume_boost = 1.0 + (analysis.volume_strength * 0.3)

        # Phase confirmation bonus
        phase_aligned = False
        if overall_direction == MultiTimeframeDirection.BULLISH:
            phase_aligned = analysis.dominant_phase in [WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION]
        elif overall_direction == MultiTimeframeDirection.BEARISH:
            phase_aligned = analysis.dominant_phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION]
        phase_boost = 1.2 if phase_aligned else 1.0

        # Funding rate impact (more weight for shorter timeframes)
        funding_impact = abs(analysis.funding_sentiment) * 0.2
        if (analysis.funding_sentiment > 0 and overall_direction == MultiTimeframeDirection.BULLISH) or \
           (analysis.funding_sentiment < 0 and overall_direction == MultiTimeframeDirection.BEARISH):
            funding_boost = 1.0 + funding_impact
        else:
            funding_boost = 1.0 - funding_impact

        # Calculate final score with all factors
        score = base_score * volume_boost * phase_boost * funding_boost
        directional_scores.append((score, analysis.group_weight))

        # Track hourly volume for potential boost
        if analysis.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES}:
            if analysis.volume_strength > 0.7:
                hourly_volume_boost = 1.2

    total_weight = sum(weight for _, weight in directional_scores)
    if total_weight == 0:
        return 0.0

    weighted_momentum = sum(score * weight for score, weight in directional_scores) / total_weight
    
    # Apply hourly volume boost and clamp final value
    final_momentum = min(1.0, weighted_momentum * hourly_volume_boost)
    return max(0.0, final_momentum)
