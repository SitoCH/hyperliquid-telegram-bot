from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
import pandas as pd  # type: ignore[import]

from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)


def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return timeframe.settings.phase_weight

def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState]
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across all timeframe groups.
    """
    # Input validation
    if not states:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="No timeframe data available for analysis",
            direction=MultiTimeframeDirection.NEUTRAL
        )

    # Group all timeframes with proper type handling
    scalping = {tf: state for tf, state in states.items() if tf in {Timeframe.MINUTES_5, Timeframe.MINUTES_15}}
    short_term = {tf: state for tf, state in states.items() if tf in {Timeframe.MINUTES_30, Timeframe.HOUR_1}}
    intermediate = {tf: state for tf, state in states.items() if tf in {Timeframe.HOURS_4}}
    long_term = {tf: state for tf, state in states.items() if tf in {Timeframe.HOURS_8, Timeframe.DAY_1}}

    try:
        # Analyze all groups
        scalping_analysis = _analyze_timeframe_group(scalping)
        short_term_analysis = _analyze_timeframe_group(short_term)
        intermediate_analysis = _analyze_timeframe_group(intermediate)
        long_term_analysis = _analyze_timeframe_group(long_term)

        # Update weights based on phase_weight
        scalping_analysis.group_weight = _calculate_group_weight(scalping)
        short_term_analysis.group_weight = _calculate_group_weight(short_term)
        intermediate_analysis.group_weight = _calculate_group_weight(intermediate)
        long_term_analysis.group_weight = _calculate_group_weight(long_term)

        # Calculate overall alignment across all groups
        all_analysis = AllTimeframesAnalysis(
            scalping=scalping_analysis,
            short_term=short_term_analysis,
            intermediate=intermediate_analysis,
            long_term=long_term_analysis,
            overall_direction=_determine_overall_direction([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ]),
            confidence_level=_calculate_overall_confidence([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ]),
            alignment_score=_calculate_overall_alignment([
                scalping_analysis, short_term_analysis, 
                intermediate_analysis, long_term_analysis
            ])
        )

        # Generate comprehensive description including all timeframes
        description = generate_all_timeframes_description(all_analysis)

        return MultiTimeframeContext(
            alignment_score=all_analysis.alignment_score,
            confidence_level=all_analysis.confidence_level,
            description=description,
            direction=all_analysis.overall_direction
        )
        
    except Exception as e:
        # Fallback in case of analysis errors
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description=f"Error analyzing timeframes: {str(e)}",
            direction=MultiTimeframeDirection.NEUTRAL
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
    action_weights: Dict[CompositeAction, float] = {}
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

        if not state.uncertain_phase:
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + weight
        action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Determine dominant characteristics
    dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0] if phase_weights else WyckoffPhase.UNKNOWN
    dominant_action = max(action_weights.items(), key=lambda x: x[1])[0] if action_weights else CompositeAction.UNKNOWN

    # Calculate internal alignment
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Calculate volume strength with exhaustion consideration
    volume_strength = sum(1 for s in group.values() if s.volume == VolumeState.HIGH) / len(group)
    
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
    """Calculate alignment across all timeframe groups."""
    # Filter out None values and empty groups
    valid_analyses = [a for a in analyses if a is not None]
    if len(valid_analyses) < 2:
        return 0.0
        
    total_weight = sum(analysis.group_weight for analysis in valid_analyses)
    if total_weight == 0:
        return 0.0
        
    # Calculate weighted phase and action alignments
    weighted_alignments = []
    for i, analysis1 in enumerate(valid_analyses):
        for j, analysis2 in enumerate(valid_analyses[i+1:], i+1):
            weight = (analysis1.group_weight + analysis2.group_weight) / total_weight
            phase_aligned = 1.0 if analysis1.dominant_phase == analysis2.dominant_phase else 0.0
            bias_aligned = 1.0 if analysis1.momentum_bias == analysis2.momentum_bias else 0.0
            weighted_alignments.append(weight * (phase_aligned * 0.6 + bias_aligned * 0.4))
    
    return sum(weighted_alignments) / len(weighted_alignments) if weighted_alignments else 0.0

def _calculate_overall_confidence(analyses: List[TimeframeGroupAnalysis]) -> float:
    """Calculate overall confidence across all timeframe groups with improved weighting."""
    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return 0.0

    # Updated weight factors with more emphasis on alignment and volume
    alignment_weight = 0.45    # Increased from 0.40 - more importance to alignment
    volume_weight = 0.35       # Unchanged
    consistency_weight = 0.20  # Reduced from 0.25 - less weight on pure consistency

    # Enhanced volume confirmation with timeframe importance
    volume_confirmation = sum(
        analysis.volume_strength * (analysis.group_weight / total_weight)
        for analysis in analyses
    )

    # Improved trend consistency check
    bias_counts: Dict[MultiTimeframeDirection, int] = {}
    weighted_biases: Dict[MultiTimeframeDirection, float] = {}
    for analysis in analyses:
        bias = analysis.momentum_bias
        bias_counts[bias] = bias_counts.get(bias, 0) + 1
        weighted_biases[bias] = weighted_biases.get(bias, 0) + analysis.group_weight
    
    # Calculate trend consistency score
    max_bias_count = max(bias_counts.values()) if bias_counts else 0
    max_weighted_bias = max(weighted_biases.values()) if weighted_biases else 0
    
    # Combine count-based and weight-based consistency
    trend_consistency = (
        (max_bias_count / len(analyses)) * 0.4 +  # How many timeframes agree
        (max_weighted_bias / total_weight) * 0.6   # How important are the agreeing timeframes
    )

    return min(
        (volume_confirmation * volume_weight +
         trend_consistency * consistency_weight +
         _calculate_overall_alignment(analyses) * alignment_weight),
        1.0
    )

def _determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall direction considering all timeframe groups."""
    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return MultiTimeframeDirection.NEUTRAL

    weighted_signals = {
        direction: sum(
            analysis.group_weight / total_weight
            for analysis in analyses
            if analysis.momentum_bias == direction
        )
        for direction in MultiTimeframeDirection
    }

    # Require stronger consensus for directional bias
    strongest_direction = max(weighted_signals.items(), key=lambda x: x[1])
    if strongest_direction[1] > 0.6:  # Require 60% weight agreement
        return strongest_direction[0]
    
    return MultiTimeframeDirection.NEUTRAL
