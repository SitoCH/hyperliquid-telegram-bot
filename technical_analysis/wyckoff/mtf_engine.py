import os
from typing import Dict
from logging_utils import logger

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, WyckoffSign, MarketPattern,
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState,
    VolatilityState, MarketLiquidity, SignificantLevelsData,
    MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext,
    AllTimeframesAnalysis
)

from .signals import (
    determine_overall_direction,
    calculate_overall_alignment,
    calculate_overall_confidence,
    calculate_momentum_intensity,
    get_volume_weight_adjustment,
    get_base_volume_strength
)

from .formatter import generate_all_timeframes_description


def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState],
    coin: str,
    mid: float,
    significant_levels: Dict[Timeframe, SignificantLevelsData],
    interactive_analysis: bool
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across three timeframe groups.
    Unified engine implementation.
    """
    if not states:
        return MultiTimeframeContext(
            description="No timeframe data available for analysis",
            should_notify=False
        )

    # Group timeframes
    short_term = {tf: state for tf, state in states.items() if tf in SHORT_TERM_TIMEFRAMES}
    intermediate = {tf: state for tf, state in states.items() if tf in INTERMEDIATE_TIMEFRAMES}
    context = {tf: state for tf, state in states.items() if tf in CONTEXT_TIMEFRAMES}

    try:
        # Analyze each group
        short_term_analysis = _analyze_timeframe_group(short_term)
        intermediate_analysis = _analyze_timeframe_group(intermediate)
        context_analysis = _analyze_timeframe_group(context)

        # Update weights (can be done inside _analyze_timeframe_group, but keeping it for compatibility)
        short_term_analysis.group_weight = _calculate_group_weight(short_term)
        intermediate_analysis.group_weight = _calculate_group_weight(intermediate)
        context_analysis.group_weight = _calculate_group_weight(context)

        # Calculate overall direction
        overall_direction = determine_overall_direction([
            short_term_analysis,
            intermediate_analysis,
            context_analysis
        ])

        # Calculate momentum intensity
        momentum_intensity = calculate_momentum_intensity([
            short_term_analysis,
            intermediate_analysis,
            context_analysis
        ], overall_direction)

        # Create AllTimeframesAnalysis object
        all_analysis = AllTimeframesAnalysis(
            short_term=short_term_analysis,
            intermediate=intermediate_analysis,
            context=context_analysis,
            overall_direction=overall_direction,
            momentum_intensity=momentum_intensity,
            confidence_level=calculate_overall_confidence(
                short_term_analysis,
                intermediate_analysis,
                context_analysis
            ),
            alignment_score=calculate_overall_alignment(short_term_analysis, intermediate_analysis)
        )

        # Generate description
        description = generate_all_timeframes_description(coin, all_analysis, mid, significant_levels, interactive_analysis)

        # Notification logic
        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))

        should_notify = (
            all_analysis.confidence_level >= min_confidence
            and all_analysis.overall_direction != MultiTimeframeDirection.NEUTRAL
            and all_analysis.alignment_score >= 0.28
        )

        # Additional direction-specific checks
        if should_notify:
            MIN_ALIGNMENT = 0.45
            if not (all_analysis.short_term.internal_alignment >= MIN_ALIGNMENT
                    or all_analysis.intermediate.internal_alignment >= MIN_ALIGNMENT):
                should_notify = False

        return MultiTimeframeContext(
            description=description,
            should_notify=should_notify
        )

    except Exception as e:
        logger.error(f"Error in unified MTF engine: {e}", exc_info=True)
        return MultiTimeframeContext(
            description=f"Error analyzing timeframes: {str(e)}",
            should_notify=False
        )


def _analyze_timeframe_group(group: Dict[Timeframe, WyckoffState]) -> TimeframeGroupAnalysis:
    """Analyze a timeframe group to determine dominant phase, action, and alignment."""
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

    phase_weights: Dict[WyckoffPhase, float] = {}
    confident_phase_weights: Dict[WyckoffPhase, float] = {}
    uncertain_phase_weights: Dict[WyckoffPhase, float] = {}
    action_weights: Dict[CompositeAction, float] = {}
    uncertain_action_weights: Dict[CompositeAction, float] = {}
    sign_weights: Dict[WyckoffSign, float] = {}

    total_weight = 0.0
    upside_exhaustion = 0
    downside_exhaustion = 0
    rapid_bullish_moves = 0.0
    rapid_bearish_moves = 0.0

    for tf, state in group.items():
        weight = tf.settings.phase_weight

        if state.phase == WyckoffPhase.MARKUP and state.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]:
            upside_exhaustion += 1
        if state.phase == WyckoffPhase.MARKDOWN and state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.CONSOLIDATING]:
            downside_exhaustion += 1

        if state.is_upthrust:
            upside_exhaustion += 1
            weight *= 1.2
        elif state.is_spring:
            downside_exhaustion += 1
            weight *= 1.2

        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            weight *= get_volume_weight_adjustment(state.volume)

        if state.effort_vs_result == EffortResult.WEAK:
            if state.phase == WyckoffPhase.MARKUP:
                upside_exhaustion += 1
            elif state.phase == WyckoffPhase.MARKDOWN:
                downside_exhaustion += 1

        if state.phase == WyckoffPhase.MARKUP and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            rapid_bullish_moves += 1.5 if state.volume == VolumeState.VERY_HIGH else 1.0
        elif state.phase == WyckoffPhase.MARKDOWN and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            rapid_bearish_moves += 1.5 if state.volume == VolumeState.VERY_HIGH else 1.0

        if state.pattern == MarketPattern.TRENDING and state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
            weight *= get_volume_weight_adjustment(state.volume)

        total_weight += weight

        if state.phase != WyckoffPhase.UNKNOWN:
            if state.uncertain_phase:
                uncertain_phase_weights[state.phase] = uncertain_phase_weights.get(state.phase, 0) + (weight * 0.7)
            else:
                confident_phase_weights[state.phase] = confident_phase_weights.get(state.phase, 0) + weight
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + (weight * 0.7 if state.uncertain_phase else weight)

        if state.composite_action != CompositeAction.UNKNOWN:
            if state.uncertain_phase:
                uncertain_action_weights[state.composite_action] = uncertain_action_weights.get(state.composite_action, 0) + (weight * 0.7)
            else:
                action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

        if state.wyckoff_sign != WyckoffSign.NONE:
            sign_weight = weight
            if state.wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.BUYING_CLIMAX]:
                sign_weight *= 1.3
            elif state.wyckoff_sign in [WyckoffSign.SIGN_OF_STRENGTH, WyckoffSign.SIGN_OF_WEAKNESS]:
                sign_weight *= 1.2
            if state.volume in [VolumeState.VERY_HIGH, VolumeState.HIGH]:
                sign_weight *= 1.2
            sign_weights[state.wyckoff_sign] = sign_weights.get(state.wyckoff_sign, 0) + sign_weight

    if phase_weights:
        dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0]
        dominant_phase_is_uncertain = uncertain_phase_weights.get(dominant_phase, 0) > confident_phase_weights.get(dominant_phase, 0)
    else:
        dominant_phase = WyckoffPhase.UNKNOWN
        dominant_phase_is_uncertain = True

    combined_action_weights = action_weights.copy()
    for action, w in uncertain_action_weights.items():
        combined_action_weights[action] = combined_action_weights.get(action, 0) + w
    dominant_action = max(combined_action_weights.items(), key=lambda x: x[1])[0] if combined_action_weights else CompositeAction.UNKNOWN

    dominant_sign = WyckoffSign.NONE
    if sign_weights:
        sum_sign_weight = sum(sign_weights.values())
        top_sign, top_weight = max(sign_weights.items(), key=lambda x: x[1])
        if (top_weight / total_weight >= 0.20) and (top_weight / sum_sign_weight >= 0.55):
            dominant_sign = top_sign

    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(combined_action_weights.values()) / total_weight if combined_action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Volume strength calculation
    volume_factors = []
    for tf, state in group.items():
        base_strength = get_base_volume_strength(state.volume)
        if state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN]:
            base_strength *= 1.2
        volume_factors.append((base_strength, tf.settings.phase_weight))

    volume_strength = sum(f * w for f, w in volume_factors) / total_weight if total_weight > 0 else 0.0
    volume_strength = max(0.0, min(1.0, volume_strength))

    # Momentum bias calculation
    bullish_signals = 0.0
    bearish_signals = 0.0
    for s in group.values():
        if is_bullish_phase(s.phase):
            bullish_signals += 1.0
        elif is_bearish_phase(s.phase):
            bearish_signals += 1.0
        if is_bullish_action(s.composite_action):
            bullish_signals += 0.8
        elif is_bearish_action(s.composite_action):
            bearish_signals += 0.8

    if len(group) > 0:
        bullish_signals /= len(group)
        bearish_signals /= len(group)

    if rapid_bullish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BULLISH
    elif rapid_bearish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BEARISH
    else:
        momentum_bias = (
            MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals + 0.1 else
            MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals + 0.1 else
            MultiTimeframeDirection.NEUTRAL
        )

    # Final result
    return TimeframeGroupAnalysis(
        dominant_phase=dominant_phase,
        uncertain_phase=dominant_phase_is_uncertain,
        dominant_action=dominant_action,
        internal_alignment=internal_alignment,
        volume_strength=volume_strength,
        momentum_bias=momentum_bias,
        group_weight=total_weight,
        funding_sentiment=0.0,  # Placeholder
        liquidity_state=MarketLiquidity.UNKNOWN,
        volatility_state=VolatilityState.UNKNOWN,
        dominant_sign=dominant_sign
    )


def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group."""
    return sum(tf.settings.phase_weight for tf in timeframes.keys())
