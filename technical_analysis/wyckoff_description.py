from .wyckoff_types import (
    WyckoffPhase, CompositeAction, WyckoffSign, FundingState,
    EffortResult, VolumeState
)

def generate_wyckoff_description(
    phase: WyckoffPhase, 
    uncertain_phase: bool, 
    is_high_volume: bool, 
    momentum_strength: float,
    is_spring: bool, 
    is_upthrust: bool, 
    effort_result: EffortResult,
    composite_action: CompositeAction, 
    wyckoff_sign: WyckoffSign,
    funding_state: FundingState
) -> str:
    """Generate a descriptive summary of the current Wyckoff state."""
    
    if phase == WyckoffPhase.UNKNOWN:
        return "Insufficient data for Wyckoff analysis"
    
    # Phase description with uncertainty marker
    phase_name = {
        WyckoffPhase.ACCUMULATION: "Accumulation",
        WyckoffPhase.DISTRIBUTION: "Distribution",
        WyckoffPhase.MARKUP: "Markup",
        WyckoffPhase.MARKDOWN: "Markdown",
        WyckoffPhase.RANGING: "Ranging"
    }.get(phase, "Unknown")
    
    phase_desc = f"{'Possible ' if uncertain_phase else ''}{phase_name}"
    
    # Build description components
    components = []
    
    # Add phase with volume context
    volume_desc = "high volume" if is_high_volume else "low volume"
    components.append(f"{phase_desc} phase with {volume_desc}")
    
    # Add spring or upthrust if present
    if is_spring:
        components.append("Spring pattern detected (potential reversal from lows)")
    if is_upthrust:
        components.append("Upthrust pattern detected (potential reversal from highs)")
    
    # Add composite operator action if significant
    if composite_action != CompositeAction.UNKNOWN and composite_action != CompositeAction.NEUTRAL:
        components.append(f"Composite operators {composite_action.value}")
    
    # Add Wyckoff sign with expanded description
    if wyckoff_sign != WyckoffSign.NONE:
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
        
        explanation = sign_explanations.get(wyckoff_sign, wyckoff_sign.value)
        components.append(explanation)
    
    # Add effort vs result analysis
    if effort_result == EffortResult.STRONG:
        components.append("Strong volume effectiveness")
    elif effort_result == EffortResult.WEAK:
        components.append("Weak volume effectiveness (potential exhaustion)")
    
    # Add funding rate info if available
    if funding_state not in [FundingState.NEUTRAL, FundingState.UNKNOWN]:
        components.append(f"Funding rates are {funding_state.value}")
    
    # Combine all components
    return "\n".join(components)
