from .wyckoff_types import MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, CompositeAction, WyckoffSign, FundingState, VolumeState


def generate_wyckoff_description(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    is_high_volume: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort_vs_result: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign,
    funding_state: FundingState
) -> str:
    """Generate concise Wyckoff analysis description with funding context."""
    base_phase = phase.name.replace("_", " ").lower()
    
    # Core market structure analysis
    description_parts = [
        f"Market Phase: {base_phase}",
        f"Institutional Activity: {composite_action.value}"
    ]
    
    # Technical formation analysis
    if wyckoff_sign != WyckoffSign.NONE:
        description_parts.append(f"Key Formation: {wyckoff_sign.value}")
    
    # Spring/Upthrust analysis
    if is_spring:
        description_parts.append(
            "Spring Pattern: Support test followed by rapid recovery indicating institutional buying"
        )
    elif is_upthrust:
        description_parts.append(
            "Upthrust Pattern: Resistance test followed by swift rejection showing distribution"
        )
    
    # Volume and effort analysis
    volume_effort = (
        "Strong institutional participation" if is_high_volume and effort_vs_result == EffortResult.STRONG
        else "Efficient price movement on low volume" if not is_high_volume and effort_vs_result == EffortResult.STRONG
        else "Weak participation and price action"
    )
    description_parts.append(f"Volume Analysis: {volume_effort}")
    
    # Funding rate analysis
    if funding_state != FundingState.UNKNOWN:
        funding_context = {
            FundingState.HIGHLY_POSITIVE: "Extreme long bias in funding suggests overleveraged market",
            FundingState.POSITIVE: "Notable long bias in funding rates",
            FundingState.SLIGHTLY_POSITIVE: "Mild long bias in funding",
            FundingState.NEUTRAL: "Balanced funding rates",
            FundingState.SLIGHTLY_NEGATIVE: "Mild short bias in funding",
            FundingState.NEGATIVE: "Notable short bias in funding rates",
            FundingState.HIGHLY_NEGATIVE: "Extreme short bias in funding suggests overleveraged market"
        }
        description_parts.append(f"Funding Context: {funding_context[funding_state]}")

    analysis = "\n".join(f"• {part}" for part in description_parts)
    
    suggestion = generate_trading_suggestion(
        phase, uncertain_phase, momentum_strength, is_spring, is_upthrust,
        effort_vs_result, composite_action, wyckoff_sign, funding_state
    )
    
    return f"<b>Wyckoff Analysis:</b>\n{analysis}\n\n<b>Trading Context:</b>\n{suggestion}"

def generate_trading_suggestion(
    phase: WyckoffPhase,
    uncertain_phase: bool,
    momentum_strength: float,
    is_spring: bool,
    is_upthrust: bool,
    effort: EffortResult,
    composite_action: CompositeAction,
    wyckoff_sign: WyckoffSign,
    funding_state: FundingState
) -> str:
    """Generate clear, actionable trading context for all experience levels."""
    
    # Risk assessment
    risk_level = "extreme" if uncertain_phase or funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.HIGHLY_NEGATIVE] else "high" if effort == EffortResult.WEAK else "normal"
    
    # Convert momentum to easy-to-understand terms
    momentum_desc = (
        "strongly bullish" if momentum_strength > 75 else
        "bullish" if momentum_strength > 25 else
        "strongly bearish" if momentum_strength < -75 else
        "bearish" if momentum_strength < -25 else
        "neutral"
    )

    # Critical market conditions
    if risk_level == "extreme":
        return (
            "High-Risk Market Conditions:\n"
            "• Market appears unstable or showing extreme sentiment\n"
            "• Recommended Action: Stay mostly in cash or stablecoins\n"
            "• Existing Positions: Consider reducing size significantly\n"
            "• Next Steps: Wait for clearer market direction and normal funding rates"
        )

    # Strong technical setups
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return (
            f"Potential Bottom Formation - {momentum_desc.title()} Momentum:\n"
            "• Market Structure: Strong bounce from support level\n"
            "• What This Means: Big players likely accumulating\n"
            "• Possible Strategy: Consider building long positions gradually\n"
            "• Key Level to Watch: Recent low point\n"
            "• Warning Signs: Heavy selling volume on bounces"
        )

    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return (
            f"Potential Top Formation - {momentum_desc.title()} Momentum:\n"
            "• Market Structure: Failed breakout above resistance\n"
            "• What This Means: Big players likely distributing\n"
            "• Possible Strategy: Consider reducing longs or building shorts\n"
            "• Key Level to Watch: Recent high point\n"
            "• Warning Signs: Strong buying volume on dips"
        )

    # Incorporate composite action into phase contexts
    composite_context = {
        CompositeAction.ACCUMULATING: "Large players appear to be buying",
        CompositeAction.DISTRIBUTING: "Large players appear to be selling",
        CompositeAction.MARKING_UP: "Large players actively pushing prices up",
        CompositeAction.MARKING_DOWN: "Large players actively pushing prices down",
        CompositeAction.NEUTRAL: "No clear big player activity"
    }[composite_action]

    # Phase-specific actionable contexts with momentum and composite action
    phase_contexts = {
        WyckoffPhase.ACCUMULATION: (
            f"Bottom Formation Process - {composite_context}:\n"
            "• Current Stage: Market building energy for potential rise\n"
            "• Key Signs: Watch for decreased selling pressure\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Possible Strategy: Consider starting to buy gradually\n"
            "• Risk Management: Keep stops below recent lows"
        ),
        WyckoffPhase.DISTRIBUTION: (
            f"Top Formation Process - {composite_context}:\n"
            "• Current Stage: Market showing signs of selling pressure\n"
            "• Key Signs: Watch for weakening rallies\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Possible Strategy: Consider taking profits or short positions\n"
            "• Risk Management: Keep stops above recent highs"
        ),
        WyckoffPhase.MARKUP: (
            f"Upward Trend - {composite_context}:\n"
            "• Current Stage: Market in rising phase\n"
            "• Key Signs: Previous resistance becomes support\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Possible Strategy: Buy dips while trend continues\n"
            "• Risk Management: Trail stops below support levels"
        ),
        WyckoffPhase.MARKDOWN: (
            f"Downward Trend - {composite_context}:\n"
            "• Current Stage: Market in falling phase\n"
            "• Key Signs: Previous support becomes resistance\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Possible Strategy: Sell rallies while trend continues\n"
            "• Risk Management: Trail stops above resistance levels"
        ),
        WyckoffPhase.RANGING: (
            f"Sideways Market - {composite_context}:\n"
            "• Current Stage: Market moving sideways\n"
            "• Key Signs: Price bouncing between clear levels\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Possible Strategy: Trade between support and resistance\n"
            "• Risk Management: Exit if price breaks range"
        )
    }

    return phase_contexts.get(phase, (
        f"Unclear Market Conditions - {composite_context}:\n"
        "• Current Stage: Market direction developing\n"
        "• Key Signs: Watch volume and price action\n"
        f"• Momentum: {momentum_desc.title()}\n"
        "• Possible Strategy: Wait for clearer setup\n"
        "• Risk Management: Keep higher cash position"
    ))
