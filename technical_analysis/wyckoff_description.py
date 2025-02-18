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
            FundingState.HIGHLY_POSITIVE: "High long bias in funding suggests overleveraged market",
            FundingState.POSITIVE: "Long bias in funding rates",
            FundingState.SLIGHTLY_POSITIVE: "Slight long bias in funding",
            FundingState.NEUTRAL: "Balanced funding rates",
            FundingState.SLIGHTLY_NEGATIVE: "Slight short bias in funding",
            FundingState.NEGATIVE: "Short bias in funding rates",
            FundingState.HIGHLY_NEGATIVE: "High short bias in funding suggests overleveraged market"
        }
        description_parts.append(f"Funding Context: {funding_context[funding_state]}")

    analysis = "\n".join(f"• {part}" for part in description_parts)
    
    return f"<b>Wyckoff Analysis:</b>\n{analysis}"
