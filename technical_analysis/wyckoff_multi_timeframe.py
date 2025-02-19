import pandas as pd  # type: ignore[import]

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
from .wyckoff_types import SignificantLevelsData
from logging_utils import logger


from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

from .wyckoff_multi_timeframe_types import (
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
            alignment_score=0.0,
            confidence_level=0.0,
            description="No timeframe data available for analysis",
            direction=MultiTimeframeDirection.NEUTRAL,
            momentum_intensity=0.0
        )

    # Group timeframes into three categories
    short_term = {tf: state for tf, state in states.items() if tf in {Timeframe.MINUTES_15, Timeframe.MINUTES_30}}
    intermediate = {tf: state for tf, state in states.items() if tf in {Timeframe.HOUR_1, Timeframe.HOURS_2}}
    long_term = {tf: state for tf, state in states.items() if tf in {Timeframe.HOURS_4, Timeframe.HOURS_8, Timeframe.DAY_1}}

    try:
        # Analyze all groups
        short_term_analysis = _analyze_timeframe_group(short_term)
        intermediate_analysis = _analyze_timeframe_group(intermediate)
        long_term_analysis = _analyze_timeframe_group(long_term)

        # Update weights
        short_term_analysis.group_weight = _calculate_group_weight(short_term)
        intermediate_analysis.group_weight = _calculate_group_weight(intermediate)
        long_term_analysis.group_weight = _calculate_group_weight(long_term)

        # Calculate momentum intensity with aligned weights
        timeframe_weights = {
            "short": SHORT_TERM_WEIGHT,
            "mid": INTERMEDIATE_WEIGHT, 
            "long": LONG_TERM_WEIGHT
        }
        
        # First determine overall direction for reference
        overall_direction = _determine_overall_direction([
            short_term_analysis, 
            intermediate_analysis, 
            long_term_analysis
        ])
        
        # Now calculate momentum intensity using the overall direction as reference
        timeframe_scores = {
            "short": (
                1.0 if short_term_analysis.momentum_bias == overall_direction
                else 0.5 if short_term_analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL
                else 0.0
            ),
            "mid": (
                1.0 if intermediate_analysis.momentum_bias == overall_direction
                else 0.5 if intermediate_analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL
                else 0.0
            ),
            "long": (
                1.0 if long_term_analysis.momentum_bias == overall_direction
                else 0.5 if long_term_analysis.momentum_bias == MultiTimeframeDirection.NEUTRAL
                else 0.0
            )
        }
        
        volume_scores = {
            "short": short_term_analysis.volume_strength,
            "mid": intermediate_analysis.volume_strength,
            "long": long_term_analysis.volume_strength
        }
        
        phase_bonus = {
            "short": 0.2 if _is_phase_confirming_momentum(short_term_analysis) else 0.0,
            "mid": 0.2 if _is_phase_confirming_momentum(intermediate_analysis) else 0.0,
            "long": 0.2 if _is_phase_confirming_momentum(long_term_analysis) else 0.0
        }
        
        momentum_intensity = sum(
            (timeframe_scores[tf] * DIRECTIONAL_WEIGHT +
             volume_scores[tf] * VOLUME_WEIGHT +
             phase_bonus[tf] * PHASE_WEIGHT) *
            timeframe_weights[tf]
            for tf in timeframe_weights.keys()
        )

        # Calculate overall alignment across all groups
        all_analysis = AllTimeframesAnalysis(
            short_term=short_term_analysis,
            intermediate=intermediate_analysis,
            long_term=long_term_analysis,
            overall_direction=overall_direction,  # Use the same overall_direction
            momentum_intensity=momentum_intensity,
            confidence_level=_calculate_overall_confidence([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis
            ]),
            alignment_score=_calculate_overall_alignment([
                short_term_analysis, 
                intermediate_analysis, 
                long_term_analysis
            ])
        )

        # Generate comprehensive description
        description = generate_all_timeframes_description(coin, all_analysis, mid, significant_levels, interactive_analysis)

        return MultiTimeframeContext(
            alignment_score=all_analysis.alignment_score,
            confidence_level=all_analysis.confidence_level,
            description=description,
            direction=all_analysis.overall_direction,
            momentum_intensity=momentum_intensity
        )
        
    except Exception as e:
        logger.error(e, exc_info=True)
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description=f"Error analyzing timeframes: {str(e)}",
            direction=MultiTimeframeDirection.NEUTRAL,
            momentum_intensity=0.0
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
    possible_phase_weights: Dict[WyckoffPhase, float] = {}  # Track possible phases separately
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

        # Improved phase weight handling
        if state.phase != WyckoffPhase.UNKNOWN:
            if state.uncertain_phase:
                # Store possible phases separately with reduced weight
                base_phase = WyckoffPhase(state.phase.value.replace('~', '').strip())
                possible_phase_weights[base_phase] = possible_phase_weights.get(base_phase, 0) + (weight * 0.7)
            else:
                # Direct phases get full weight
                phase_weights[state.phase] = phase_weights.get(state.phase, 0) + weight

        # Improved action weight handling
        if state.composite_action != CompositeAction.UNKNOWN:
            if state.uncertain_phase:  # Use phase uncertainty to affect action weight
                uncertain_action_weights[state.composite_action] = uncertain_action_weights.get(state.composite_action, 0) + (weight * 0.7)
            else:
                action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Merge possible phases with confirmed phases
    for phase, weight in possible_phase_weights.items():
        phase_weights[phase] = phase_weights.get(phase, 0) + weight

    # Merge uncertain actions with confirmed actions
    for action, weight in uncertain_action_weights.items():
        action_weights[action] = action_weights.get(action, 0) + weight

    # Determine dominant characteristics with better handling of unknowns
    if phase_weights:
        dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0]
        # Add uncertainty marker if mostly from possible phases
        if (possible_phase_weights.get(dominant_phase, 0) > 
            phase_weights[dominant_phase] * 0.7):  # 70% threshold
            possible_name = f"~ {dominant_phase.value}"
            dominant_phase = next(p for p in WyckoffPhase 
                               if p.value == possible_name)
    else:
        dominant_phase = WyckoffPhase.UNKNOWN

    # Determine dominant action with uncertainty handling
    if action_weights:
        dominant_action = max(action_weights.items(), key=lambda x: x[1])[0]
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
    """Calculate overall confidence with improved volume handling."""
    if not analyses:
        return 0.0

    total_weight = sum(analysis.group_weight for analysis in analyses)
    if total_weight == 0:
        return 0.0

    # Adjusted weights for better balance
    alignment_weight = 0.30
    volume_weight = 0.35
    consistency_weight = 0.35

    # Enhanced volume confirmation
    volume_scores = []
    for analysis in analyses:
        # Base volume score
        base_score = analysis.volume_strength
        
        # Boost score if volume aligns with momentum
        if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
            if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
                base_score *= 1.2
            elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
                base_score *= 1.1
                
        volume_scores.append(base_score * (analysis.group_weight / total_weight))
    
    volume_confirmation = sum(volume_scores)

    # Enhanced trend consistency calculation
    directional_scores = []
    prev_bias = None
    for analysis in sorted(analyses, key=lambda x: x.group_weight, reverse=True):
        score = 1.0 if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL else 0.5
        
        # Check for bias alignment with higher timeframes
        if prev_bias and analysis.momentum_bias == prev_bias:
            score *= 1.2
        
        directional_scores.append(score * (analysis.group_weight / total_weight))
        prev_bias = analysis.momentum_bias
    
    directional_agreement = sum(directional_scores)

    # Calculate alignment score
    alignment_score = _calculate_overall_alignment(analyses)

    # Combine scores with dynamic minimum threshold
    raw_confidence = (
        volume_confirmation * volume_weight +
        directional_agreement * consistency_weight +
        alignment_score * alignment_weight
    )

    # Dynamic minimum confidence based on volume and alignment
    min_confidence = 0.3 if all(a.volume_strength > 0.7 and a.internal_alignment > 0.6 for a in analyses) else 0.0
    
    # Apply sigmoid-like scaling to emphasize strong signals
    scaled_confidence = 1 / (1 + pow(2.0, -5 * (raw_confidence - 0.5)))
    
    return max(min(scaled_confidence, 1.0), min_confidence)

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
