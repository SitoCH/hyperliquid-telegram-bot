import numpy as np
from typing import List, Tuple, Final

from .wyckoff_types import (
    WyckoffPhase, CompositeAction,
    WyckoffSign, VolumeState,
    MultiTimeframeDirection, TimeframeGroupAnalysis,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, CONTEXT_WEIGHT,
    MODERATE_VOLUME_THRESHOLD, LOW_VOLUME_THRESHOLD
)

# Thresholds for confidence calculation
BASE_CONFIDENCE_THRESHOLD = 0.20
STRONG_SIGNAL_MULTIPLIER = 1.3

DIRECTIONAL_WEIGHT: Final[float] = 0.35
VOLUME_WEIGHT: Final[float] = 0.35
ALIGNMENT_WEIGHT: Final[float] = 0.08
MOMENTUM_WEIGHT: Final[float] = 0.22


def _calculate_timeframe_importance() -> Tuple[float, float, float]:
    """
    Calculate the importance of each timeframe group based on the phase_weight
    values from timeframe settings.
    """
    total_weight = SHORT_TERM_WEIGHT + INTERMEDIATE_WEIGHT + CONTEXT_WEIGHT
    if total_weight == 0:
        return 0.33, 0.33, 0.34

    return (
        SHORT_TERM_WEIGHT / total_weight,
        INTERMEDIATE_WEIGHT / total_weight,
        CONTEXT_WEIGHT / total_weight
    )


def determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall market direction using a weighted voting system based on timeframe phases and momentum."""
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL

    bullish_weight = 0.0
    bearish_weight = 0.0
    total_weight = 0.0

    intermediate_momentum = None

    for _idx, analysis in enumerate(analyses):
        if _idx == 1:
            intermediate_momentum = analysis.momentum_bias

        weight = analysis.group_weight
        certainty_factor = 0.5 if analysis.uncertain_phase else 1.0

        volume_factor = 1.0
        if analysis.volume_strength > MODERATE_VOLUME_THRESHOLD:
            volume_factor = 1.2
        elif analysis.volume_strength < LOW_VOLUME_THRESHOLD:
            volume_factor = 0.8

        intermediate_confirmation_factor = 1.0
        if _idx == 0 and intermediate_momentum is not None:
            if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
                if analysis.momentum_bias != intermediate_momentum and intermediate_momentum != MultiTimeframeDirection.NEUTRAL:
                    intermediate_confirmation_factor = 0.80

        adjusted_weight = weight * certainty_factor * volume_factor * intermediate_confirmation_factor
        total_weight += adjusted_weight

        if is_bullish_phase(analysis.dominant_phase) or analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
            if is_bullish_phase(analysis.dominant_phase) and analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
                bullish_weight += adjusted_weight
            else:
                bullish_weight += adjusted_weight * 0.6

        if is_bearish_phase(analysis.dominant_phase) or analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
            if is_bearish_phase(analysis.dominant_phase) and analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
                bearish_weight += adjusted_weight
            else:
                bearish_weight += adjusted_weight * 0.6

    if total_weight < 0.01:
        return MultiTimeframeDirection.NEUTRAL

    bullish_strength = bullish_weight / total_weight
    bearish_strength = bearish_weight / total_weight
    direction_difference = abs(bullish_strength - bearish_strength)

    avg_volume = sum(a.volume_strength for a in analyses) / len(analyses)
    certain_phases = sum(1 for a in analyses if not a.uncertain_phase)
    certain_ratio = certain_phases / len(analyses)

    decision_threshold = 0.20
    if certain_ratio > 0.7 or avg_volume > MODERATE_VOLUME_THRESHOLD:
        decision_threshold = 0.12

    if direction_difference < decision_threshold:
        return MultiTimeframeDirection.NEUTRAL

    return MultiTimeframeDirection.BULLISH if bullish_strength > bearish_strength else MultiTimeframeDirection.BEARISH


def calculate_overall_alignment(short_term_analysis: TimeframeGroupAnalysis, intermediate_analysis: TimeframeGroupAnalysis) -> float:
    """Calculate alignment across timeframe groups."""
    total_weight = short_term_analysis.group_weight + intermediate_analysis.group_weight
    if total_weight == 0:
        return 0.0

    weight = (short_term_analysis.group_weight + intermediate_analysis.group_weight) / total_weight

    phase_aligned = 0.0
    if short_term_analysis.dominant_phase == intermediate_analysis.dominant_phase:
        phase_aligned = 1.0
    elif short_term_analysis.dominant_phase.value.replace('~', '') == intermediate_analysis.dominant_phase.value.replace('~', ''):
        phase_aligned = 0.70
    elif (is_bullish_phase(short_term_analysis.dominant_phase) and is_bullish_phase(intermediate_analysis.dominant_phase)) or \
         (is_bearish_phase(short_term_analysis.dominant_phase) and is_bearish_phase(intermediate_analysis.dominant_phase)):
        phase_aligned = 0.55

    action_aligned = 0.0
    if short_term_analysis.dominant_action == intermediate_analysis.dominant_action:
        action_aligned = 1.0
    elif (is_bullish_action(short_term_analysis.dominant_action) and is_bullish_action(intermediate_analysis.dominant_action)) or \
         (is_bearish_action(short_term_analysis.dominant_action) and is_bearish_action(intermediate_analysis.dominant_action)):
        action_aligned = 0.6

    bias_aligned = 0.0
    if short_term_analysis.momentum_bias == intermediate_analysis.momentum_bias:
        bias_aligned = 1.0
    elif short_term_analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL and intermediate_analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
        bias_aligned = 0.50
    else:
        bias_aligned = 0.30

    volume_agreement = 1 - abs(short_term_analysis.volume_strength - intermediate_analysis.volume_strength)

    alignment_score = (
        phase_aligned * 0.30
        + action_aligned * 0.30
        + bias_aligned * 0.30
        + volume_agreement * 0.10
    )

    return alignment_score * weight


def calculate_overall_confidence(short_term: TimeframeGroupAnalysis, intermediate: TimeframeGroupAnalysis, context: TimeframeGroupAnalysis) -> float:
    """Calculate overall confidence score."""
    directional_score = _calculate_directional_score_direct(short_term, intermediate, context)
    volume_score = _calculate_volume_score_direct(short_term, intermediate, context)
    alignment_score = calculate_overall_alignment(short_term, intermediate)
    momentum_score = _calculate_momentum_score_direct(short_term, intermediate)

    early_signal_boost = 0.0
    if short_term.internal_alignment >= 0.7 and short_term.volume_strength >= 0.65:
        early_signal_boost += 0.1
    if short_term.momentum_bias != MultiTimeframeDirection.NEUTRAL and short_term.volume_strength >= 0.6:
        early_signal_boost += 0.05
    if (short_term.momentum_bias == intermediate.momentum_bias
        and short_term.momentum_bias != MultiTimeframeDirection.NEUTRAL
            and short_term.volume_strength >= 0.6):
        early_signal_boost += 0.12
    if (intermediate.internal_alignment >= 0.7 and intermediate.volume_strength >= 0.6
            and intermediate.momentum_bias != MultiTimeframeDirection.NEUTRAL):
        early_signal_boost += 0.06

    confidence = (
        directional_score * DIRECTIONAL_WEIGHT
        + volume_score * VOLUME_WEIGHT
        + alignment_score * ALIGNMENT_WEIGHT
        + momentum_score * MOMENTUM_WEIGHT
    )

    confidence = min(1.0, confidence + early_signal_boost)
    return max(min(confidence, 1.0), BASE_CONFIDENCE_THRESHOLD)


def _calculate_directional_score_direct(short_term: TimeframeGroupAnalysis,
                                        intermediate: TimeframeGroupAnalysis,
                                        context: TimeframeGroupAnalysis) -> float:
    """Calculate directional agreement score."""
    scores = []
    weights = []
    group_biases = {}

    st_imp, int_imp, ctx_imp = _calculate_timeframe_importance()

    for group_name, analysis, importance in [('short', short_term, st_imp), ('intermediate', intermediate, int_imp), ('context', context, ctx_imp)]:
        bias = 1 if analysis.momentum_bias == MultiTimeframeDirection.BULLISH else \
            -1 if analysis.momentum_bias == MultiTimeframeDirection.BEARISH else 0
        group_biases[group_name] = bias
        scores.append(abs(bias) * analysis.internal_alignment)
        weights.append(importance)

    agreement_bonus = 0.0
    if group_biases.get('short', 0) != 0 and group_biases.get('intermediate', 0) != 0 and group_biases['short'] == group_biases['intermediate']:
        agreement_bonus += 0.15
    if group_biases.get('intermediate', 0) != 0 and group_biases.get('context', 0) != 0 and group_biases['intermediate'] == group_biases['context']:
        agreement_bonus += 0.05

    if not scores:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    base_score = (weighted_sum / total_weight if total_weight > 0 else 0.0) * 0.95

    return min(1.0, base_score + agreement_bonus)


def _calculate_momentum_score_direct(short_term: TimeframeGroupAnalysis,
                                     intermediate: TimeframeGroupAnalysis) -> float:
    """Calculate momentum score."""
    momentum_values = []
    for analysis in [short_term, intermediate]:
        momentum = 1.0 if analysis.momentum_bias == MultiTimeframeDirection.BULLISH else \
            -1.0 if analysis.momentum_bias == MultiTimeframeDirection.BEARISH else 0.0
        adjusted = abs(momentum) * analysis.internal_alignment * (0.6 + 0.4 * analysis.volume_strength) * 0.95
        momentum_values.append(adjusted)

    avg_momentum = sum(momentum_values) / len(momentum_values) if momentum_values else 0.0

    if avg_momentum > 0.45:
        return min(1.0, avg_momentum * 1.20)
    elif avg_momentum > 0.25:
        return min(1.0, avg_momentum * 1.10)
    return avg_momentum


def _calculate_volume_score_direct(short_term: TimeframeGroupAnalysis,
                                   intermediate: TimeframeGroupAnalysis,
                                   context: TimeframeGroupAnalysis) -> float:
    """Calculate volume confirmation score."""
    weighted_scores = []
    for analysis, multiplier in [(short_term, 1.25), (intermediate, 1.05), (context, 0.9)]:
        if analysis.dominant_action in [CompositeAction.MARKING_UP, CompositeAction.MARKING_DOWN]:
            multiplier *= 1.25
        elif analysis.dominant_action in [CompositeAction.ACCUMULATING, CompositeAction.DISTRIBUTING]:
            multiplier *= 1.15
        weighted_scores.append(analysis.volume_strength * multiplier * analysis.internal_alignment)

    avg_volume_score = (sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0) * 0.95
    if avg_volume_score > 0.6:
        avg_volume_score *= 1.15
    return min(1.0, avg_volume_score)


def calculate_momentum_intensity(analyses: List[TimeframeGroupAnalysis], overall_direction: MultiTimeframeDirection) -> float:
    """Calculate momentum intensity."""
    if not analyses or overall_direction == MultiTimeframeDirection.NEUTRAL:
        return 0.0

    rapid_moves = sum(1 for a in analyses if (
        (a.momentum_bias == overall_direction and a.volume_strength > 0.7)
        or (a.dominant_sign in [WyckoffSign.SIGN_OF_STRENGTH, WyckoffSign.SIGN_OF_WEAKNESS] and a.volume_strength > 0.6)
    ))
    if rapid_moves >= len(analyses) // 2:
        return 0.80

    is_bearish = overall_direction == MultiTimeframeDirection.BEARISH
    base_momentum = 0.42

    aligned_count = sum(1 for a in analyses if a.momentum_bias == overall_direction)
    alignment_score = aligned_count / len(analyses)

    phase_confirmations = sum(1 for a in analyses if (
        (is_bearish and (is_bearish_phase(a.dominant_phase) or a.dominant_phase == WyckoffPhase.DISTRIBUTION))
        or (not is_bearish and (is_bullish_phase(a.dominant_phase) or a.dominant_phase == WyckoffPhase.ACCUMULATION))
    ))
    phase_score = phase_confirmations / len(analyses)

    total_weight = sum(a.group_weight for a in analyses)
    volume_score = sum(a.volume_strength * a.group_weight for a in analyses) / total_weight if total_weight > 0 else 0
    volume_factor = 1.10 if is_bearish else 1.0
    adjusted_volume = min(1.0, volume_score * volume_factor)

    # Simplified timeframe agreement check
    timeframe_bonus = 0.0
    if aligned_count >= 2:
        timeframe_bonus = 0.1

    raw_momentum = base_momentum + (alignment_score * 0.25) + (phase_score * 0.20) + (adjusted_volume * 0.15) + timeframe_bonus
    sigmoid_threshold = 0.38
    final_momentum = 1.0 / (1.0 + np.exp(-6 * (raw_momentum - sigmoid_threshold)))

    return float(max(0.01, min(1.0, final_momentum)))


def is_phase_confirming_momentum(analysis: TimeframeGroupAnalysis) -> bool:
    """Check if the Wyckoff phase confirms the momentum bias."""
    if analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
        return is_bullish_phase(analysis.dominant_phase)
    elif analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
        return is_bearish_phase(analysis.dominant_phase)
    return False


def get_volume_weight_adjustment(volume_state: VolumeState) -> float:
    """Return weight adjustment multiplier based on volume state"""
    adjustments = {
        VolumeState.VERY_HIGH: 1.3,
        VolumeState.HIGH: 1.1,
        VolumeState.LOW: 0.9,
        VolumeState.VERY_LOW: 0.8
    }
    return adjustments.get(volume_state, 1.0)


def get_base_volume_strength(volume_state: VolumeState) -> float:
    """Return base volume strength value based on volume state"""
    strengths = {
        VolumeState.VERY_HIGH: 1.5,
        VolumeState.HIGH: 1.0,
        VolumeState.NEUTRAL: 0.7,
        VolumeState.LOW: 0.4,
        VolumeState.VERY_LOW: 0.25,
        VolumeState.UNKNOWN: 0.5
    }
    return strengths.get(volume_state, 0.5)
