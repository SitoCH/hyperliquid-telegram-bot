from typing import Dict, List, Optional, TypeVar

from ..wyckoff_types import (
    WyckoffPhase,
    WyckoffSign,
    WyckoffState,
    SignificantLevelsData,
    CompositeAction,
    Timeframe,
    VolatilityState,
)

from .wyckoff_multi_timeframe_types import (
    AllTimeframesAnalysis,
    MultiTimeframeDirection,
    TimeframeGroupAnalysis,
    STRONG_MOMENTUM,
    MODERATE_MOMENTUM,
    WEAK_MOMENTUM,
    MODERATE_VOLUME_THRESHOLD,
    STRONG_VOLUME_THRESHOLD,
    MIXED_MOMENTUM,
    LOW_MOMENTUM,
)

from .trade_suggestion import get_trade_suggestion

# Constants for common thresholds
MODERATE_CONFIDENCE_THRESHOLD = 0.5
MODERATE_ALIGNMENT_THRESHOLD = 0.6


def _get_phase_context(phase: WyckoffPhase, direction: MultiTimeframeDirection) -> str:
    """Get phase-specific context that varies the language."""
    phase_contexts = {
        WyckoffPhase.ACCUMULATION: "building a base" if direction != MultiTimeframeDirection.BEARISH else "testing support",
        WyckoffPhase.MARKUP: "in an uptrend" if direction == MultiTimeframeDirection.BULLISH else "attempting recovery",
        WyckoffPhase.DISTRIBUTION: "topping out" if direction != MultiTimeframeDirection.BULLISH else "meeting resistance",
        WyckoffPhase.MARKDOWN: "in a downtrend" if direction == MultiTimeframeDirection.BEARISH else "under pressure",
        WyckoffPhase.RANGING: "consolidating",
        WyckoffPhase.UNKNOWN: "unclear structure",
    }
    return phase_contexts.get(phase, "evolving")


def _get_tf_divergence_note(
    context_bias: MultiTimeframeDirection,
    intraday_bias: MultiTimeframeDirection,
    context_phase: WyckoffPhase,
) -> str:
    """Generate note when higher TF diverges from intraday."""
    if context_bias == intraday_bias or context_bias == MultiTimeframeDirection.NEUTRAL:
        return ""
    
    if context_bias == MultiTimeframeDirection.BULLISH:
        return f" Higher TFs remain bullish ({context_phase.value}); dips may find support."
    elif context_bias == MultiTimeframeDirection.BEARISH:
        return f" Higher TFs bearish ({context_phase.value}); rallies may face resistance."
    return ""



T = TypeVar("T")


def _weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average of values with corresponding weights."""
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def _get_timeframe_weights(analysis: AllTimeframesAnalysis) -> List[float]:
    """Return standard timeframe weights for consistency."""
    return [analysis.short_term.group_weight, analysis.intermediate.group_weight]


def _get_volume_description(volume_strength: float) -> str:
    """Get standardized volume description based on strength."""
    if volume_strength > STRONG_VOLUME_THRESHOLD:
        return "strong volume"
    elif volume_strength > MODERATE_VOLUME_THRESHOLD:
        return "moderate volume"
    return "light volume"


def _get_sign_description(sign: WyckoffSign) -> str:
    """
    Get a proper readable description for a Wyckoff Sign instead of the raw enum value.
    """
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


def _calculate_weighted_volume_strength(analysis: AllTimeframesAnalysis) -> float:
    """Calculate weighted volume strength across timeframes."""
    weights = _get_timeframe_weights(analysis)
    values = [
        analysis.short_term.volume_strength,
        analysis.intermediate.volume_strength,
    ]
    return _weighted_average(values, weights)


def generate_all_timeframes_description(
    coin: str,
    analysis: AllTimeframesAnalysis,
    mid: float,
    significant_levels: Dict[Timeframe, SignificantLevelsData],
    interactive_analysis: bool,
    states: Optional[Dict[Timeframe, WyckoffState]] = None,
) -> str:
    """Generate comprehensive description including four timeframe groups."""
    alignment_pct = f"{analysis.alignment_score * 100:.0f}%"
    confidence_pct = f"{analysis.confidence_level * 100:.0f}%"

    # Get phase, volume, sign and action info for short and intermediate timeframes
    short_phase = analysis.short_term.dominant_phase.value
    if (
        analysis.short_term.uncertain_phase
        and analysis.short_term.dominant_phase != WyckoffPhase.UNKNOWN
    ):
        short_phase = f"~{short_phase}"

    interm_phase = analysis.intermediate.dominant_phase.value
    if (
        analysis.intermediate.uncertain_phase
        and analysis.intermediate.dominant_phase != WyckoffPhase.UNKNOWN
    ):
        interm_phase = f"~{interm_phase}"

    short_volume = _get_volume_description(analysis.short_term.volume_strength)
    interm_volume = _get_volume_description(analysis.intermediate.volume_strength)

    short_sign = _get_sign_description(analysis.short_term.dominant_sign)
    interm_sign = _get_sign_description(analysis.intermediate.dominant_sign)

    short_action = (
        ""
        if analysis.short_term.dominant_action == CompositeAction.UNKNOWN
        else f" | {analysis.short_term.dominant_action.value}"
    )
    interm_action = (
        ""
        if analysis.intermediate.dominant_action == CompositeAction.UNKNOWN
        else f" | {analysis.intermediate.dominant_action.value}"
    )

    emoji = _get_trend_emoji_all_timeframes(analysis)

    if not interactive_analysis:
        return (
            f"{emoji} <b>Market Analysis:</b>\n"
            f"Confidence: {confidence_pct}\n"
            f"Intraday Trend (30m-1h): {interm_phase} ({interm_volume}{interm_sign}{interm_action})\n"
            f"Immediate Signals (15m): {short_phase} ({short_volume}{short_sign}{short_action})\n"
        )

    short_term_desc = _get_timeframe_trend_description(analysis.short_term)
    intermediate_desc = _get_timeframe_trend_description(analysis.intermediate)
    context_desc = _get_timeframe_trend_description(analysis.context)

    insight = _generate_actionable_insight_all_timeframes(analysis)

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

    trade_suggestion = get_trade_suggestion(
        coin,
        analysis.overall_direction,
        mid,
        significant_levels,
        analysis.confidence_level,
        states,
    )
    if trade_suggestion:
        full_description += f"\n\n{trade_suggestion}"

    return full_description


def _get_trend_emoji_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """Get appropriate trend emoji based on overall analysis state."""
    # First check if we have enough confidence
    if analysis.confidence_level < 0.4:
        return "📊"  # Low confidence

    # Get the overall trend strength using both alignment and momentum
    trend_strength = (
        analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD
        and analysis.confidence_level > MODERATE_CONFIDENCE_THRESHOLD
        and analysis.momentum_intensity > MODERATE_MOMENTUM
    )

    direction = analysis.overall_direction

    # Use simplified dictionary lookup for emoji selection
    emoji_map = {
        (MultiTimeframeDirection.BULLISH, True): "⬆️",  # Strong bullish
        (MultiTimeframeDirection.BULLISH, False): (
            "↗️" if analysis.momentum_intensity > WEAK_MOMENTUM else "➡️⬆️"
        ),
        (MultiTimeframeDirection.BEARISH, True): "⬇️",  # Strong bearish
        (MultiTimeframeDirection.BEARISH, False): (
            "↘️" if analysis.momentum_intensity > WEAK_MOMENTUM else "➡️⬇️"
        ),
    }

    if direction == MultiTimeframeDirection.NEUTRAL:
        # Check if we're in consolidation or in conflict
        if analysis.alignment_score > MODERATE_ALIGNMENT_THRESHOLD:
            return "↔️"  # Clear consolidation
        elif analysis.momentum_intensity < LOW_MOMENTUM:
            return "🔀"  # Very low momentum, ranging market
        return "🔄"  # Mixed signals

    return emoji_map.get((direction, trend_strength), "📊")


def _generate_actionable_insight_all_timeframes(analysis: AllTimeframesAnalysis) -> str:
    """
    Generate intraday-focused actionable insights focused on crypto trading timeframes.
    """
    # Graded confidence messaging for clearer risk posture
    if analysis.confidence_level < 0.4:
        return (
            "<b>Analysis:</b>\nVery low confidence across timeframes.\n"
            "<b>Recommendation:</b>\nObserve only; avoid adding new risk until signals strengthen."
        )
    elif analysis.confidence_level < 0.5:
        return (
            "<b>Analysis:</b>\nLow confidence signals across timeframes.\n"
            "<b>Recommendation:</b>\nBe cautious; use reduced size and wait for alignment and volume confirmation."
        )

    def get_intraday_signal() -> str:
        """Get primary trading signal focused on intraday timeframes."""
        # Focus on intermediate timeframe for intraday trading bias (30m-1h)
        intraday_bias = analysis.intermediate.momentum_bias

        # Use short term for immediate direction (15m)
        immediate_bias = analysis.short_term.momentum_bias

        # Calculate volume context for signal strength
        avg_volume = (
            analysis.short_term.volume_strength
            * 0.6  # Higher weight for short timeframe volume
            + analysis.intermediate.volume_strength
            * 0.4  # Lower weight for intermediate timeframe
        )

        # Alignment label to reflect how coherent the intraday picture is
        if analysis.alignment_score > 0.75:
            align_label = "Strong"
        elif analysis.alignment_score >= 0.6:
            align_label = "Moderate"
        else:
            align_label = "Mixed"

        if analysis.momentum_intensity > STRONG_MOMENTUM:
            momentum_desc = "strong"
            position_advice = (
                "Consider aggressive intraday positions with proper risk management."
            )
        elif analysis.momentum_intensity > MODERATE_MOMENTUM:
            momentum_desc = "steady"
            position_advice = (
                "Favorable environment for swing positions with moderate risk exposure."
            )
        elif analysis.momentum_intensity > WEAK_MOMENTUM:
            momentum_desc = "moderate"
            position_advice = (
                "Use scaled entries and definitive technical triggers for entries."
            )
        elif analysis.momentum_intensity > MIXED_MOMENTUM:
            momentum_desc = "mixed"
            position_advice = (
                "Focus on shorter timeframe setups and reduced position sizes."
            )
        else:
            momentum_desc = "weak"
            position_advice = "Consider range trading strategies or reduce exposure until clearer signals emerge."

        # Volatility-aware adjustments to execution guidance
        if analysis.intermediate.volatility_state == VolatilityState.HIGH:
            position_advice += (
                " Widen stops and stagger entries; take partial profits faster."
            )
        elif analysis.intermediate.volatility_state == VolatilityState.NORMAL:
            position_advice += (
                " Require breakout confirmation; avoid chasing early moves."
            )

        # Volume qualifier for more precise signal description
        volume_qualifier = ""
        if avg_volume > STRONG_VOLUME_THRESHOLD:
            volume_qualifier = "high volume "
        elif avg_volume < MODERATE_VOLUME_THRESHOLD:
            volume_qualifier = "light volume "

        # Check intraday alignment - focus on short and intermediate timeframes
        # Soft alignment: treat immediate NEUTRAL as aligned with intraday bias
        intraday_aligned = (
            immediate_bias == intraday_bias
            or immediate_bias == MultiTimeframeDirection.NEUTRAL
        )

        # Generate improved timeframe-specific momentum prefix
        signal_prefix = (
            f"{align_label} alignment, {momentum_desc} {volume_qualifier}momentum "
        )

        # Improved signal direction with phase-aware, varied language
        interm_phase = analysis.intermediate.dominant_phase
        context_phase = analysis.context.dominant_phase
        context_bias = analysis.context.momentum_bias
        
        # Get phase-specific backdrop from context TF
        backdrop = _get_phase_context(context_phase, context_bias)
        
        if intraday_aligned:
            if intraday_bias == MultiTimeframeDirection.BULLISH:
                # Vary based on whether we're early (accumulation) or extended (markup)
                if interm_phase == WyckoffPhase.ACCUMULATION:
                    signal_direction = f"bullish setup forming; price {backdrop} on higher TFs"
                elif interm_phase == WyckoffPhase.MARKUP:
                    signal_direction = f"uptrend continuation; momentum {momentum_desc}"
                else:
                    signal_direction = f"bullish bias emerging from {interm_phase.value} phase"
                if avg_volume > STRONG_VOLUME_THRESHOLD:
                    signal_direction += ", volume confirms"
                signal_direction += ". "

            elif intraday_bias == MultiTimeframeDirection.BEARISH:
                if interm_phase == WyckoffPhase.DISTRIBUTION:
                    signal_direction = f"bearish setup developing; price {backdrop} on higher TFs"
                elif interm_phase == WyckoffPhase.MARKDOWN:
                    signal_direction = f"downtrend continuation; selling pressure {momentum_desc}"
                else:
                    signal_direction = f"bearish bias emerging from {interm_phase.value} phase"
                if avg_volume > STRONG_VOLUME_THRESHOLD:
                    signal_direction += ", volume confirms"
                signal_direction += ". "

            else:
                signal_direction = (
                    f"range-bound; price {backdrop}. "
                    f"Trade edges or wait for breakout with volume. "
                )

        else:  # Timeframes not aligned
            # Check for potential reversal setups - common in crypto
            potential_reversal = (
                immediate_bias != MultiTimeframeDirection.NEUTRAL
                and intraday_bias != immediate_bias
                and analysis.short_term.volume_strength > MODERATE_VOLUME_THRESHOLD
            )
            
            # Get divergence context from higher TFs
            tf_divergence = _get_tf_divergence_note(context_bias, intraday_bias, context_phase)

            if potential_reversal:
                if immediate_bias == MultiTimeframeDirection.BULLISH:
                    signal_direction = (
                        f"early bullish reversal on 15m; await 30m-1h confirmation.{tf_divergence} "
                    )
                else:  # BEARISH
                    signal_direction = (
                        f"early bearish reversal on 15m; await 30m-1h confirmation.{tf_divergence} "
                    )
            else:
                if immediate_bias == MultiTimeframeDirection.NEUTRAL:
                    signal_direction = (
                        f"15m consolidating within {intraday_bias.value} intraday trend.{tf_divergence} "
                    )
                else:
                    # TF conflict - be specific about which TFs disagree
                    signal_direction = (
                        f"15m {immediate_bias.value} vs 30m-1h {intraday_bias.value}; conflicting signals.{tf_divergence} "
                    )

        # Concise Wyckoff context with sign-specific insights
        context_tail = ""
        short_sign = analysis.short_term.dominant_sign
        interm_sign = analysis.intermediate.dominant_sign
        chosen_sign = short_sign if short_sign != WyckoffSign.NONE else interm_sign
        
        # Map signs to actionable context
        sign_insights = {
            WyckoffSign.SELLING_CLIMAX: "capitulation may signal bottom",
            WyckoffSign.AUTOMATIC_RALLY: "relief bounce in progress",
            WyckoffSign.SECONDARY_TEST: "retesting prior level",
            WyckoffSign.LAST_POINT_OF_SUPPORT: "final support test before markup",
            WyckoffSign.SIGN_OF_STRENGTH: "demand overcoming supply",
            WyckoffSign.BUYING_CLIMAX: "exhaustion may signal top",
            WyckoffSign.UPTHRUST: "failed breakout, watch for reversal",
            WyckoffSign.SECONDARY_TEST_RESISTANCE: "retesting resistance",
            WyckoffSign.LAST_POINT_OF_RESISTANCE: "final resistance before markdown",
            WyckoffSign.SIGN_OF_WEAKNESS: "supply overcoming demand",
        }
        
        if chosen_sign != WyckoffSign.NONE and chosen_sign in sign_insights:
            context_tail = sign_insights[chosen_sign]

        # Add action context if different from sign implication
        action_text = None
        if analysis.short_term.dominant_action != CompositeAction.UNKNOWN:
            action_text = analysis.short_term.dominant_action.value
        elif analysis.intermediate.dominant_action != CompositeAction.UNKNOWN:
            action_text = analysis.intermediate.dominant_action.value
        
        if action_text and context_tail:
            # Only add if it provides new info
            if action_text.lower() not in context_tail.lower():
                context_tail = f"{context_tail} ({action_text})"
        elif action_text and not context_tail:
            context_tail = f"Composite action: {action_text}"

        # Build the complete insight with context tail integrated naturally
        wyckoff_note = f" Wyckoff: {context_tail}." if context_tail else ""
        
        return (
            f"{signal_prefix}with {signal_direction.strip()}"
            f"{wyckoff_note} "
            f"{position_advice}"
        )

    # Format the complete insight
    intraday_signal = get_intraday_signal()

    return f"<b>💡 Trading Insight:</b>\n{intraday_signal}"


def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate enhanced trend description for a timeframe group."""
    # Add uncertainty marker if needed
    phase_desc = analysis.dominant_phase.value
    if analysis.uncertain_phase and analysis.dominant_phase != WyckoffPhase.UNKNOWN:
        phase_desc = f"~ {phase_desc}"

    # Format action description with validation
    action_desc = (
        analysis.dominant_action.value
        if analysis.dominant_action != CompositeAction.UNKNOWN
        else "no clear action"
    )

    volume_desc = _get_volume_description(analysis.volume_strength)

    volatility = ""
    if analysis.volatility_state == VolatilityState.HIGH:
        volatility = " | high volatility"

    return f"• {phase_desc} phase {action_desc}\n" f"  └─ {volume_desc}{volatility}"
