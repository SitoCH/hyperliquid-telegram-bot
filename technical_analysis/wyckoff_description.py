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
    
    # Add Wyckoff sign if present
    if wyckoff_sign != WyckoffSign.NONE:
        components.append(f"Wyckoff sign: {wyckoff_sign.value}")
    
    # Add effort vs result analysis
    if effort_result == EffortResult.STRONG:
        components.append("Strong volume effectiveness")
    elif effort_result == EffortResult.WEAK:
        components.append("Weak volume effectiveness (potential exhaustion)")
    
    # Add funding rate info if available
    if funding_state not in [FundingState.NEUTRAL, FundingState.UNKNOWN]:
        components.append(f"Funding rates are {funding_state.value}")
    
    # Combine all components
    return " • ".join(components)
