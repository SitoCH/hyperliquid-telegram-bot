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
        wyckoff_descriptions = {
            WyckoffSign.SELLING_CLIMAX: "Selling Climax (SC), rapid price drop with high volume, potential bottom",
            WyckoffSign.AUTOMATIC_RALLY: "Automatic Rally (AR), initial bounce after selling climax, sets range",
            WyckoffSign.SECONDARY_TEST: "Secondary Test (ST), retest of selling climax low, confirms support",
            WyckoffSign.SIGN_OF_STRENGTH: "Sign of Strength (SOS), strong rally with high volume, indicates demand",
            WyckoffSign.LAST_POINT_OF_SUPPORT: "Last Point of Support (LPS), final low before markup, confirms demand",
            WyckoffSign.BUYING_CLIMAX: "Buying Climax (BC), rapid price rise with high volume, potential top",
            WyckoffSign.UPTHRUST: "Upthrust (UT), false breakout above resistance, indicates supply",
            WyckoffSign.SECONDARY_TEST_RESISTANCE: "Secondary Test Resistance (STR), retest of buying climax high, confirms resistance",
            WyckoffSign.LAST_POINT_OF_RESISTANCE: "Last Point of Resistance (LPSY), final high before markdown, confirms supply",
            WyckoffSign.SIGN_OF_WEAKNESS: "Sign of Weakness (SOW), strong decline with high volume, indicates supply"
        }
        description_parts.append(f"Key Formation: {wyckoff_descriptions[wyckoff_sign]}")
    
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
    """Generate clear, actionable trading context for both long and short positions."""
    
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
            "• Recommended Action: Close or reduce both long and short positions\n"
            "• Current Status: Better to stay neutral\n"
            "• Next Steps: Wait for clearer market direction and normal funding rates\n"
            f"• Warning: Funding rates show {funding_state.value} bias - increased liquidation risk"
        )

    # Strong technical setups
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return (
            f"Potential Bottom Formation - {momentum_desc.title()} Momentum:\n"
            "• Market Structure: Strong bounce from support with spring pattern\n"
            "• Long Strategy: Consider scaling into longs above spring level\n"
            "• Short Warning: Cover shorts, high risk of squeeze\n"
            "• Key Levels: Support at spring low, resistance at recent high\n"
            "• Risk Management: Stop loss under spring low for longs\n"
            f"• Funding Context: {funding_state.value} - adjust position size accordingly"
        )

    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return (
            f"Potential Top Formation - {momentum_desc.title()} Momentum:\n"
            "• Market Structure: Failed breakout with upthrust pattern\n"
            "• Short Strategy: Consider scaling into shorts below upthrust\n"
            "• Long Warning: Exit longs, high risk of reversal\n"
            "• Key Levels: Resistance at upthrust high, support at recent low\n"
            "• Risk Management: Stop loss above upthrust high for shorts\n"
            f"• Funding Context: {funding_state.value} - adjust position size accordingly"
        )

    # Incorporate composite action into phase contexts
    composite_context = {
        CompositeAction.ACCUMULATING: "Large players actively buying - caution with shorts",
        CompositeAction.DISTRIBUTING: "Large players actively selling - caution with longs",
        CompositeAction.MARKING_UP: "Large players pushing prices up - potential short squeeze",
        CompositeAction.MARKING_DOWN: "Large players pushing prices down - potential long squeeze",
        CompositeAction.NEUTRAL: "No clear institutional activity - trade smaller size"
    }[composite_action]

    # Phase-specific actionable contexts with momentum and composite action
    phase_contexts = {
        WyckoffPhase.ACCUMULATION: (
            f"Bottom Formation Process - {composite_context}:\n"
            "• Long Strategy: Scale into longs near support levels\n"
            "• Short Strategy: Only quick countertrend trades\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Risk Management: Tight stops below support for longs\n"
            f"• Funding Context: {funding_state.value} affects position sizing"
        ),
        WyckoffPhase.DISTRIBUTION: (
            f"Top Formation Process - {composite_context}:\n"
            "• Short Strategy: Scale into shorts near resistance\n"
            "• Long Strategy: Only quick countertrend trades\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Risk Management: Tight stops above resistance for shorts\n"
            f"• Funding Context: {funding_state.value} affects position sizing"
        ),
        WyckoffPhase.MARKUP: (
            f"Upward Trend - {composite_context}:\n"
            "• Long Strategy: Buy dips with trend following\n"
            "• Short Warning: Avoid fighting the trend\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Risk Management: Trail stops, add to winners\n"
            f"• Funding Context: {funding_state.value} affects leverage risk"
        ),
        WyckoffPhase.MARKDOWN: (
            f"Downward Trend - {composite_context}:\n"
            "• Short Strategy: Sell rallies with trend following\n"
            "• Long Warning: Avoid fighting the trend\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Risk Management: Trail stops, add to winners\n"
            f"• Funding Context: {funding_state.value} affects leverage risk"
        ),
        WyckoffPhase.RANGING: (
            f"Sideways Market - {composite_context}:\n"
            "• Long Strategy: Buy near support, sell resistance\n"
            "• Short Strategy: Sell near resistance, cover support\n"
            f"• Momentum: {momentum_desc.title()}\n"
            "• Risk Management: Tight stops outside range\n"
            f"• Funding Context: {funding_state.value} affects carry cost"
        )
    }

    return phase_contexts.get(phase, (
        f"Unclear Market Conditions - {composite_context}:\n"
        "• Trading Style: Reduce position sizes both ways\n"
        "• Long Strategy: Only very clear support levels\n"
        "• Short Strategy: Only very clear resistance levels\n"
        f"• Momentum: {momentum_desc.title()}\n"
        f"• Funding Context: {funding_state.value} - consider carry cost"
    ))
