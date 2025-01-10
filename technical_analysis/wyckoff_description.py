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
    """Generate enhanced Wyckoff analysis description including funding rates."""
    base_phase = phase.name.replace("_", " ").lower()
    
    # Start with clear market structure
    description_parts = [
        f"Market is in {base_phase} phase",
        f"{composite_action.value.lower()}"
    ]
    
    # Add technical signs with more context
    if wyckoff_sign != WyckoffSign.NONE:
        description_parts.append(f"showing {wyckoff_sign.value} formation")
    
    # Enhanced spring/upthrust descriptions
    if is_spring:
        description_parts.append(
            "with bullish Spring pattern (test of support with rapid recovery) "
            "suggesting accumulation by institutions"
        )
    elif is_upthrust:
        description_parts.append(
            "with bearish Upthrust pattern (test of resistance with rapid rejection) "
            "suggesting distribution by institutions"
        )
    
    # Volume and effort analysis with more context
    if is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append(
            "backed by significant institutional volume showing clear directional intent"
        )
    elif not is_high_volume and effort_vs_result == EffortResult.STRONG:
        description_parts.append(
            "showing efficient price movement despite low volume, suggesting smart money activity"
        )
    
    # Enhanced funding rate context
    if funding_state != FundingState.UNKNOWN:
        funding_advice = {
            FundingState.HIGHLY_POSITIVE: (
                "Extreme positive funding indicates severely overleveraged longs. "
                "High risk of long liquidations. Consider waiting for funding reset or "
                "look for short opportunities with strong institutional signals"
            ),
            FundingState.POSITIVE: (
                "Positive funding shows aggressive long positioning. "
                "Consider reduced leverage for longs and tighter stops. "
                "Watch for potential long squeezes on strong distribution signals"
            ),
            FundingState.SLIGHTLY_POSITIVE: (
                "Slightly positive funding suggests mild long bias. "
                "Normal position sizing acceptable but monitor funding trend"
            ),
            FundingState.NEUTRAL: (
                "Neutral funding indicates balanced positioning. "
                "Focus on technical signals for trade direction"
            ),
            FundingState.SLIGHTLY_NEGATIVE: (
                "Slightly negative funding suggests mild short bias. "
                "Normal position sizing acceptable but monitor funding trend"
            ),
            FundingState.NEGATIVE: (
                "Negative funding shows aggressive short positioning. "
                "Consider reduced leverage for shorts and tighter stops. "
                "Watch for potential short squeezes on strong accumulation signals"
            ),
            FundingState.HIGHLY_NEGATIVE: (
                "Extreme negative funding indicates severely overleveraged shorts. "
                "High risk of short liquidations. Consider waiting for funding reset or "
                "look for long opportunities with strong institutional signals"
            ),
            FundingState.UNKNOWN: ""
        }
        description_parts.append(funding_advice.get(funding_state, ""))

    # Join description with better flow
    main_description = ", ".join(description_parts)
    
    # Add trading suggestion
    suggestion = generate_trading_suggestion(
        phase, uncertain_phase, momentum_strength, is_spring, is_upthrust,
        effort_vs_result, composite_action, wyckoff_sign, funding_state
    )
    
    return f"{main_description}.\n\n<b>Trading Perspective:</b>\n{suggestion}"

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
    """Generate detailed crypto-specific trading suggestions with risk management."""
    
    # First, assess overall market risk level
    risk_level = "high" if uncertain_phase or effort == EffortResult.WEAK else "normal"
    if funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.HIGHLY_NEGATIVE]:
        risk_level = "extreme"
        
    # Determine position sizing based on risk
    position_guidance = {
        "extreme": "Consider very small positions (0.25x normal size) or staying flat",
        "high": "Reduce position sizes to 0.5x normal size",
        "normal": "Standard position sizing acceptable"
    }[risk_level]

    # Handle high-risk scenarios first
    if risk_level == "extreme":
        return (
            f"High-risk market conditions detected:\n"
            f"• Extreme funding rates suggesting potential liquidation cascade\n"
            f"• {position_guidance}\n"
            f"• Wait for funding reset or very strong institutional signals\n"
            f"• Use wider stops to account for increased volatility"
        )

    if uncertain_phase:
        return (
            f"Unclear market structure - Exercise caution:\n"
            f"• {position_guidance}\n"
            f"• Wait for clear institutional footprints\n"
            f"• Look for high volume with clear price direction\n"
            f"• Keep larger portion of portfolio in stable coins"
        )

    # Handle strong institutional patterns
    if wyckoff_sign in [WyckoffSign.SELLING_CLIMAX, WyckoffSign.LAST_POINT_OF_SUPPORT] and is_spring:
        return (
            f"High-probability Accumulation setup detected:\n"
            f"• {position_guidance}\n"
            f"• Scale into longs gradually (3-4 entries)\n"
            f"• Initial stop under Spring low ({'-1.5' if risk_level == 'normal' else '-2.0'}x ATR)\n"
            f"• Look for volume confirmation on bounces\n"
            f"• Trail stops as price develops new support levels"
        )

    if wyckoff_sign in [WyckoffSign.BUYING_CLIMAX, WyckoffSign.LAST_POINT_OF_RESISTANCE] and is_upthrust:
        return (
            f"High-probability Distribution setup detected:\n"
            f"• {position_guidance}\n"
            f"• Scale into shorts gradually (3-4 entries)\n"
            f"• Initial stop above Upthrust high ({'+1.5' if risk_level == 'normal' else '+2.0'}x ATR)\n"
            f"• Look for volume confirmation on drops\n"
            f"• Trail stops as price develops new resistance levels"
        )

    # Handle strong composite operator actions
    if composite_action == CompositeAction.ACCUMULATING:
        direction = "bullish" if momentum_strength > 0 else "neutral to bullish"
        return (
            f"Institutional Accumulation detected - {direction} bias:\n"
            f"• {position_guidance}\n"
            f"• Enter longs on successful support tests\n"
            f"• Look for decreasing volume on pullbacks\n"
            f"• Watch for spring patterns near support\n"
            f"• Use shallow retracements for entries"
        )

    if composite_action == CompositeAction.DISTRIBUTING:
        direction = "bearish" if momentum_strength < 0 else "neutral to bearish"
        return (
            f"Institutional Distribution detected - {direction} bias:\n"
            f"• {position_guidance}\n"
            f"• Enter shorts on failed resistance tests\n"
            f"• Look for decreasing volume on rallies\n"
            f"• Watch for upthrust patterns near resistance\n"
            f"• Use shallow bounces for entries"
        )

    # Phase-specific suggestions with funding rate context
    phase_suggestions = {
        WyckoffPhase.ACCUMULATION: (
            f"Accumulation Phase - Building long positions:\n"
            f"• {position_guidance}\n"
            f"• Watch for absorption patterns at support\n"
            f"• Look for declining volume on dips\n"
            f"• Wait for spring patterns or stopping action\n"
            f"• Consider partial profits if funding turns highly positive"
        ),
        WyckoffPhase.DISTRIBUTION: (
            f"Distribution Phase - Building short positions:\n"
            f"• {position_guidance}\n"
            f"• Watch for supply testing on rallies\n"
            f"• Look for declining volume on bounces\n"
            f"• Wait for upthrust patterns or stopping action\n"
            f"• Consider partial profits if funding turns highly negative"
        ),
        WyckoffPhase.MARKUP: (
            f"Markup Phase - Long bias with protection:\n"
            f"• {position_guidance}\n"
            f"• Trail stops under developing support\n"
            f"• Add on high-volume pullbacks\n"
            f"• Watch for distribution signs at resistance\n"
            f"• Tighten stops if funding becomes extremely positive"
        ),
        WyckoffPhase.MARKDOWN: (
            f"Markdown Phase - Short bias with protection:\n"
            f"• {position_guidance}\n"
            f"• Trail stops above developing resistance\n"
            f"• Add on high-volume bounces\n"
            f"• Watch for accumulation signs at support\n"
            f"• Tighten stops if funding becomes extremely negative"
        ),
        WyckoffPhase.RANGING: (
            f"Ranging Phase - Two-way trading opportunities:\n"
            f"• {position_guidance}\n"
            f"• Trade range extremes with volume confirmation\n"
            f"• Use smaller position sizes for range trades\n"
            f"• Watch for range expansion signals\n"
            f"• Consider range breakout setups with funding confirmation"
        )
    }

    return phase_suggestions.get(phase, (
        f"Market structure developing:\n"
        f"• {position_guidance}\n"
        f"• Wait for clear institutional patterns\n"
        f"• Monitor volume and price action at key levels\n"
        f"• Keep larger cash position until clarity emerges"
    ))
