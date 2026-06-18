from typing import Dict, Optional, TypeVar
from .wyckoff_types import (
    WyckoffPhase, WyckoffSign, WyckoffState, SignificantLevelsData,
    CompositeAction, Timeframe, VolatilityState, MultiTimeframeDirection,
    TimeframeGroupAnalysis, AllTimeframesAnalysis, VolumeState, EffortResult,
    MODERATE_VOLUME_THRESHOLD, STRONG_VOLUME_THRESHOLD
)
from .trade_suggestion import get_trade_suggestion

T = TypeVar("T")

# Constants for common thresholds
MODERATE_CONFIDENCE_THRESHOLD = 0.5
MODERATE_ALIGNMENT_THRESHOLD = 0.6


def generate_wyckoff_description(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    volume_state: VolumeState,
    is_spring: bool,
    is_upthrust: bool,
    effort_result: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign
) -> str:
    """Generate a descriptive summary of the current Wyckoff state for a single timeframe."""
    if phase == WyckoffPhase.UNKNOWN:
        return "Insufficient data for Wyckoff analysis"

    phase_name = {
        WyckoffPhase.ACCUMULATION: "Accumulation",
        WyckoffPhase.DISTRIBUTION: "Distribution",
        WyckoffPhase.MARKUP: "Markup",
        WyckoffPhase.MARKDOWN: "Markdown",
        WyckoffPhase.RANGING: "Ranging"
    }.get(phase, "Unknown")

    phase_desc = f"{'Possible ' if uncertain_phase else ''}{phase_name}"
    components = []
    components.append(f"{phase_desc} phase with {volume_state.value} volume")

    if is_spring:
        components.append("Spring pattern detected (potential reversal from lows)")
    if is_upthrust:
        components.append("Upthrust pattern detected (potential reversal from highs)")

    if composite_action != CompositeAction.UNKNOWN and composite_action != CompositeAction.NEUTRAL:
        components.append(f"Composite operators {composite_action.value}")

    if wyckoff_sign != WyckoffSign.NONE:
        components.append(_get_sign_explanation(wyckoff_sign))

    if effort_result == EffortResult.STRONG:
        components.append("Volume effectively translating to price movement")
    elif effort_result == EffortResult.WEAK:
        components.append("Volume not effectively translating to price movement")

    return "\n".join(components)


def _get_sign_explanation(sign: WyckoffSign) -> str:
    """Get detailed explanation for a Wyckoff sign."""
    sign_explanations = {
        WyckoffSign.SELLING_CLIMAX: "Selling Climax (SC) - excessive selling near a potential bottom",
        WyckoffSign.AUTOMATIC_RALLY: "Automatic Rally (AR) - technical bounce after a selling climax",
        WyckoffSign.SECONDARY_TEST: "Secondary Test (ST) - retesting previous low with lower volume",
        WyckoffSign.LAST_POINT_OF_SUPPORT: "Last Point of Support (LPS) - final test of support before markup",
        WyckoffSign.SIGN_OF_STRENGTH: "Sign of Strength (SOS) - significant buying pressure after accumulation",
        WyckoffSign.BUYING_CLIMAX: "Buying Climax (BC) - excessive buying near a potential top",
        WyckoffSign.UPTHRUST: "Upthrust (UT) - failed move above resistance with rejection",
        WyckoffSign.SECONDARY_TEST_RESISTANCE: "Secondary Test Resistance (STR) - test of previous high with lower volume",
        WyckoffSign.LAST_POINT_OF_RESISTANCE: "Last Point of Supply (LPSY) - final supply test before markdown",
        WyckoffSign.SIGN_OF_WEAKNESS: "Sign of Weakness (SOW) - significant selling pressure after distribution"
    }
    return sign_explanations.get(sign, sign.value)


def generate_all_timeframes_description(
    coin: str,
    analysis: AllTimeframesAnalysis,
    mid: float,
    significant_levels: Dict[Timeframe, SignificantLevelsData],
    interactive_analysis: bool,
    states: Optional[Dict[Timeframe, WyckoffState]] = None,
) -> str:
    """Generate comprehensive description including multi-timeframe analysis."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"

    short_phase = analysis.short_term.dominant_phase.value
    if analysis.short_term.uncertain_phase and analysis.short_term.dominant_phase != WyckoffPhase.UNKNOWN:
        short_phase = f"~{short_phase}"

    interm_phase = analysis.intermediate.dominant_phase.value
    if analysis.intermediate.uncertain_phase and analysis.intermediate.dominant_phase != WyckoffPhase.UNKNOWN:
        interm_phase = f"~{interm_phase}"

    short_volume = _get_volume_description(analysis.short_term.volume_strength)
    interm_volume = _get_volume_description(analysis.intermediate.volume_strength)

    short_sign = _get_sign_short_description(analysis.short_term.dominant_sign)
    interm_sign = _get_sign_short_description(analysis.intermediate.dominant_sign)

    short_action = "" if analysis.short_term.dominant_action == CompositeAction.UNKNOWN else f" | {analysis.short_term.dominant_action.value}"
    interm_action = "" if analysis.intermediate.dominant_action == CompositeAction.UNKNOWN else f" | {analysis.intermediate.dominant_action.value}"

    emoji = _get_trend_emoji(analysis)

    if not interactive_analysis:
        return (
            f"{emoji} <b>Market Analysis:</b>\n"
            f"Confidence: {confidence_pct}\n"
            f"Intraday Trend (30m-1h): {interm_phase} ({interm_volume}{interm_sign}{interm_action})\n"
            f"Immediate Signals (15m): {short_phase} ({short_volume}{short_sign}{short_action})\n"
        )

    short_term_desc = _get_timeframe_group_description(analysis.short_term)
    intermediate_desc = _get_timeframe_group_description(analysis.intermediate)
    context_desc = _get_timeframe_group_description(analysis.context)

    insight = _generate_actionable_insight(analysis)

    full_description = (
        f"{emoji} <b>Market Analysis:</b>\n"
        f"Confidence: {confidence_pct}\n"
        f"Funding: {analysis.intermediate.funding_state.value}\n\n"
        f"<b>🔍 Timeframes:</b>\n"
        f"Timeframe alignment: {alignment_pct}\n"
        f"Market Context (4h-8h):\n{context_desc}\n"
        f"Intraday Trend (30m-1h):\n{intermediate_desc}\n"
        f"Immediate Signals (15m):\n{short_term_desc}\n\n"
        f"{insight}"
    )

    suggestion = get_trade_suggestion(
        coin,
        analysis.overall_direction,
        mid,
        significant_levels,
        analysis.confidence_level,
        states,
    )
    if suggestion:
        full_description += f"\n\n{suggestion}"

    return full_description


def _get_volume_description(volume_strength: float) -> str:
    """Get standardized volume description."""
    if volume_strength > STRONG_VOLUME_THRESHOLD:
        return "strong volume"
    elif volume_strength > MODERATE_VOLUME_THRESHOLD:
        return "moderate volume"
    return "light volume"


def _get_sign_short_description(sign: WyckoffSign) -> str:
    """Get short readable description for a sign."""
    if sign == WyckoffSign.NONE:
        return ""
    descriptions = {
        WyckoffSign.SELLING_CLIMAX: "selling climax",
        WyckoffSign.AUTOMATIC_RALLY: "automatic rally",
        WyckoffSign.SECONDARY_TEST: "secondary test",
        WyckoffSign.LAST_POINT_OF_SUPPORT: "last point of support",
        WyckoffSign.SIGN_OF_STRENGTH: "sign of strength",
        WyckoffSign.BUYING_CLIMAX: "buying climax",
        WyckoffSign.UPTHRUST: "upthrust",
        WyckoffSign.SECONDARY_TEST_RESISTANCE: "secondary test resistance",
        WyckoffSign.LAST_POINT_OF_RESISTANCE: "last point of resistance",
        WyckoffSign.SIGN_OF_WEAKNESS: "sign of weakness",
    }
    return f" | {descriptions.get(sign, sign.value.lower())}"


def _get_trend_emoji(analysis: AllTimeframesAnalysis) -> str:
    """Get trend emoji."""
    if analysis.confidence_level < 0.4:
        return "📊"

    trend_strength = (
        analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD
        and analysis.confidence_level > MODERATE_CONFIDENCE_THRESHOLD
        and analysis.momentum_intensity > 0.62  # MODERATE_MOMENTUM
    )

    direction = analysis.overall_direction
    emoji_map = {
        (MultiTimeframeDirection.BULLISH, True): "⬆️",
        (MultiTimeframeDirection.BULLISH, False): "↗️" if analysis.momentum_intensity > 0.38 else "➡️⬆️",
        (MultiTimeframeDirection.BEARISH, True): "⬇️",
        (MultiTimeframeDirection.BEARISH, False): "↘️" if analysis.momentum_intensity > 0.38 else "➡️⬇️",
    }

    if direction == MultiTimeframeDirection.NEUTRAL:
        if analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD:
            return "↔️"
        return "🔄"

    return emoji_map.get((direction, trend_strength), "📊")


def _get_timeframe_group_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate description for a timeframe group."""
    phase_desc = analysis.dominant_phase.value
    if analysis.uncertain_phase and analysis.dominant_phase != WyckoffPhase.UNKNOWN:
        phase_desc = f"~ {phase_desc}"

    action_desc = analysis.dominant_action.value if analysis.dominant_action != CompositeAction.UNKNOWN else "no clear action"
    volume_desc = _get_volume_description(analysis.volume_strength)
    volatility = " | high volatility" if analysis.volatility_state == VolatilityState.HIGH else ""

    return f"• {phase_desc} phase {action_desc}\n  └─ {volume_desc}{volatility}"


def _generate_actionable_insight(analysis: AllTimeframesAnalysis) -> str:
    """Generate trading insights."""
    if analysis.confidence_level < 0.4:
        return "<b>Analysis:</b>\nVery low confidence across timeframes.\n<b>Recommendation:</b>\nObserve only."
    elif analysis.confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals.\n<b>Recommendation:</b>\nBe cautious."

    # More complex logic could be moved from mtf_description if needed,
    # but for brevity let's use a simpler version or port the full one.
    # Porting the full logic for quality:

    intraday_bias = analysis.intermediate.momentum_bias
    avg_volume = analysis.short_term.volume_strength * 0.6 + analysis.intermediate.volume_strength * 0.4

    align_label = "Strong" if analysis.alignment_score > 0.75 else "Moderate" if analysis.alignment_score >= 0.6 else "Mixed"

    if analysis.momentum_intensity > 0.88:
        momentum_desc = "strong"
    elif analysis.momentum_intensity > 0.62:
        momentum_desc = "steady"
    else:
        momentum_desc = "moderate"

    volume_qualifier = "high volume " if avg_volume > STRONG_VOLUME_THRESHOLD else "light volume " if avg_volume < MODERATE_VOLUME_THRESHOLD else ""
    signal_prefix = f"{align_label} alignment, {momentum_desc} {volume_qualifier}momentum "

    direction_text = f"bias is {intraday_bias.value}" if intraday_bias != MultiTimeframeDirection.NEUTRAL else "is range-bound"

    return f"<b>💡 Trading Insight:</b>\n{signal_prefix}with {direction_text}."
