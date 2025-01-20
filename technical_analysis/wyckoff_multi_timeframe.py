from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

class MultiTimeframeDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

@dataclass
class MultiTimeframeContext:
    alignment_score: float  # 0 to 1, indicating how well timeframes align
    confidence_level: float  # 0 to 1, indicating strength of signals
    description: str
    direction: MultiTimeframeDirection

def get_phase_weight(timeframe: Timeframe) -> float:
    """Get the weight for each timeframe's contribution to analysis."""
    return timeframe.settings.phase_weight

def analyze_multi_timeframe(
    states: Dict[Timeframe, WyckoffState]
) -> MultiTimeframeContext:
    """
    Analyze Wyckoff states across multiple timeframes to determine market structure.
    
    Evaluates alignment between timeframes, confidence of signals, and generates
    trading context with higher weighting on longer timeframes.
    
    Args:
        states: Dictionary mapping timeframes to their Wyckoff states
        
    Returns:
        MultiTimeframeContext with alignment score, confidence level and description
    """
    # Input validation
    if not states:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="No timeframe data available for analysis",
            direction=MultiTimeframeDirection.NEUTRAL
        )

    # Ensure minimum required timeframes
    required_higher = {Timeframe.HOURS_4, Timeframe.HOURS_8, Timeframe.DAY_1}
    required_lower = {Timeframe.HOUR_1, Timeframe.MINUTES_30, Timeframe.MINUTES_15}
    
    available_higher = set(states.keys()) & required_higher
    available_lower = set(states.keys()) & required_lower
    
    if not available_higher or not available_lower:
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description="Insufficient timeframe coverage for reliable analysis",
            direction=MultiTimeframeDirection.NEUTRAL
        )

    # Group timeframes
    higher_tf = {tf: state for tf, state in states.items() if tf in available_higher}
    lower_tf = {tf: state for tf, state in states.items() if tf in available_lower}

    try:
        # Analyze each group
        higher_analysis = _analyze_timeframe_group(higher_tf)
        lower_analysis = _analyze_timeframe_group(lower_tf)

        # Calculate alignment and confidence
        groups_alignment = _calculate_dual_groups_alignment(higher_analysis, lower_analysis)
        confidence_level = _calculate_dual_confidence(higher_analysis, lower_analysis, groups_alignment)

        # Generate comprehensive description
        description = _generate_dual_group_description(
            higher_analysis,
            lower_analysis,
            groups_alignment,
            confidence_level,
            higher_tf  # Pass higher_tf to the description generator
        )

        # Determine direction based on momentum bias and confidence
        direction = _determine_direction(higher_analysis, lower_analysis, confidence_level)

        return MultiTimeframeContext(
            alignment_score=groups_alignment,
            confidence_level=confidence_level,
            description=description,
            direction=direction
        )
        
    except Exception as e:
        # Fallback in case of analysis errors
        return MultiTimeframeContext(
            alignment_score=0.0,
            confidence_level=0.0,
            description=f"Error analyzing timeframes: {str(e)}",
            direction=MultiTimeframeDirection.NEUTRAL
        )

@dataclass
class TimeframeGroupAnalysis:
    dominant_phase: WyckoffPhase
    dominant_action: CompositeAction
    internal_alignment: float
    volume_strength: float
    momentum_bias: MultiTimeframeDirection
    group_weight: float
    funding_sentiment: float  # -1 to 1, negative means bearish funding
    liquidity_state: MarketLiquidity
    liquidation_risk: LiquidationRisk
    volatility_state: VolatilityState

def _analyze_timeframe_group(
    group: Dict[Timeframe, WyckoffState]
) -> TimeframeGroupAnalysis:
    """Analyze a group with enhanced crypto-specific signals."""
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
            liquidation_risk=LiquidationRisk.UNKNOWN,
            volatility_state=VolatilityState.UNKNOWN
        )

    # Calculate weighted votes for phases and actions
    phase_weights: Dict[WyckoffPhase, float] = {}
    action_weights: Dict[CompositeAction, float] = {}
    total_weight = 0.0

    # Add volatility-based weight adjustment
    for tf, state in group.items():
        weight = get_phase_weight(tf)
        
        # Increase weight for high volatility and volume conditions
        if state.volatility == VolatilityState.HIGH and state.volume == VolumeState.HIGH:
            weight *= 1.2  # 20% boost for high volatility + volume
        
        # Boost weight for spring/upthrust patterns
        if state.is_spring or state.is_upthrust:
            weight *= 1.15  # 15% boost for significant price action patterns
            
        total_weight += weight

        if not state.uncertain_phase:
            phase_weights[state.phase] = phase_weights.get(state.phase, 0) + weight
        action_weights[state.composite_action] = action_weights.get(state.composite_action, 0) + weight

    # Determine dominant characteristics
    dominant_phase = max(phase_weights.items(), key=lambda x: x[1])[0] if phase_weights else WyckoffPhase.UNKNOWN
    dominant_action = max(action_weights.items(), key=lambda x: x[1])[0] if action_weights else CompositeAction.UNKNOWN

    # Calculate internal alignment
    phase_alignment = max(phase_weights.values()) / total_weight if phase_weights else 0
    action_alignment = max(action_weights.values()) / total_weight if action_weights else 0
    internal_alignment = (phase_alignment + action_alignment) / 2

    # Calculate volume strength
    volume_strength = sum(1 for s in group.values() if s.volume == VolumeState.HIGH) / len(group)

    # Calculate funding sentiment (-1 to 1)
    funding_signals = []
    for state in group.values():
        if state.funding_state == FundingState.HIGHLY_POSITIVE:
            funding_signals.append(1.0)
        elif state.funding_state == FundingState.POSITIVE:
            funding_signals.append(0.5)
        elif state.funding_state == FundingState.SLIGHTLY_POSITIVE:
            funding_signals.append(0.25)
        elif state.funding_state == FundingState.HIGHLY_NEGATIVE:
            funding_signals.append(-1.0)
        elif state.funding_state == FundingState.NEGATIVE:
            funding_signals.append(-0.5)
        elif state.funding_state == FundingState.SLIGHTLY_NEGATIVE:
            funding_signals.append(-0.25)
    
    funding_sentiment = sum(funding_signals) / len(funding_signals) if funding_signals else 0

    # Analyze liquidity state
    liquidity_counts = {state.liquidity: 0 for state in group.values()}
    for state in group.values():
        liquidity_counts[state.liquidity] += 1
    liquidity_state = max(liquidity_counts.items(), key=lambda x: x[1])[0]

    # Analyze liquidation risk
    risk_counts = {state.liquidation_risk: 0 for state in group.values()}
    for state in group.values():
        risk_counts[state.liquidation_risk] += 1
    liquidation_risk = max(risk_counts.items(), key=lambda x: x[1])[0]

    # Analyze volatility state
    volatility_counts = {state.volatility: 0 for state in group.values()}
    for state in group.values():
        volatility_counts[state.volatility] += 1
    volatility_state = max(volatility_counts.items(), key=lambda x: x[1])[0]

    # Enhance momentum bias calculation with funding and liquidation data
    bullish_signals = sum(1 for s in group.values() if (
        s.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] or 
        s.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.MARKING_UP] or
        (s.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKDOWN)  # Potential short squeeze
    ))
    
    bearish_signals = sum(1 for s in group.values() if (
        s.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] or 
        s.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.MARKING_DOWN] or
        (s.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKUP)  # Potential long liquidation
    ))

    momentum_bias = (
        MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals else
        MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals else
        MultiTimeframeDirection.NEUTRAL
    )

    return TimeframeGroupAnalysis(
        dominant_phase=dominant_phase,
        dominant_action=dominant_action,
        internal_alignment=internal_alignment,
        volume_strength=volume_strength,
        momentum_bias=momentum_bias,
        group_weight=total_weight,
        funding_sentiment=funding_sentiment,
        liquidity_state=liquidity_state,
        liquidation_risk=liquidation_risk,
        volatility_state=volatility_state
    )

def _calculate_dual_groups_alignment(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> float:
    """Calculate alignment between higher and lower timeframe groups."""
    # Calculate phase alignment
    phase_alignment = 1.0 if higher.dominant_phase == lower.dominant_phase else 0.0

    # Calculate bias alignment
    bias_alignment = 1.0 if higher.momentum_bias == lower.momentum_bias else 0.0

    # Calculate weighted alignment score
    alignment = (phase_alignment * 0.6) + (bias_alignment * 0.4)

    # Apply internal alignment weights
    weighted_alignment = (
        alignment * 0.6 +
        (higher.internal_alignment * 0.6 +
         lower.internal_alignment * 0.4) * 0.4
    )

    return weighted_alignment

def _calculate_dual_confidence(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float
) -> float:
    """Calculate confidence level with enhanced lower timeframe reactivity."""
    # Adjusted weight factors to be more reactive
    alignment_weight = 0.35  # Reduced from 0.4
    volume_weight = 0.35    # Increased from 0.3
    consistency_weight = 0.3

    # Volume confirmation with dynamic weighting
    # Increase lower timeframe influence when volume is significantly higher
    lower_volume_factor = min(0.6, lower.volume_strength * 1.2)  # Can go up to 60% weight
    higher_volume_factor = 1.0 - lower_volume_factor
    
    volume_confirmation = (
        higher.volume_strength * higher_volume_factor +
        lower.volume_strength * lower_volume_factor
    )

    # Enhanced trend consistency check
    # Give more weight to lower timeframes during strong moves
    trend_consistency = (
        1.0 if higher.momentum_bias == lower.momentum_bias else
        0.7 if lower.volume_strength > 0.8 else  # Strong lower timeframe moves get 70% credit
        0.3  # Minimal consistency during disagreement
    )

    return (
        groups_alignment * alignment_weight +
        volume_confirmation * volume_weight +
        trend_consistency * consistency_weight
    )

def _generate_dual_group_description(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float,
    confidence_level: float,
    higher_tf: Dict[Timeframe, WyckoffState]
) -> str:
    """Generate a narrative description for dual group analysis."""
    alignment_pct = f"{groups_alignment * 100:.0f}%"
    confidence_pct = f"{confidence_level * 100:.0f}%"

    market_context = _determine_dual_market_context(higher, lower)
    trend_strength = _determine_dual_trend_strength(higher, lower)
    actionable_insight = _generate_dual_actionable_insight(higher, lower, confidence_level)
    
    # Get funding state (same for all timeframes, so we can take from either group)
    funding_state = next(iter(higher_tf.values())).funding_state if higher_tf else FundingState.UNKNOWN

    # Determine emoji based on market bias
    if higher.momentum_bias == lower.momentum_bias:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            emoji = "📈"  # Chart increasing
        elif higher.momentum_bias == MultiTimeframeDirection.BEARISH:
            emoji = "📉"  # Chart decreasing
        else:
            emoji = "📊"  # Bar chart
    else:
        emoji = "↔️"  # Left-right arrow for mixed signals

    # Create narrative description with emoji
    description = (
        f"{emoji} Market analysis shows a {trend_strength.lower()} {market_context.lower()} with {funding_state.value} funding.\n"
        f"Higher timeframes indicate {higher.dominant_phase.value} phase with {higher.dominant_action.value}, "
        f"while lower timeframes show {lower.dominant_phase.value} with {lower.dominant_action.value}.\n"
        f"Overall alignment between timeframes is {alignment_pct} with {confidence_pct} confidence.\n"
        f"{actionable_insight}"
    )

    return description

def _generate_dual_actionable_insight(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    confidence_level: float
) -> str:
    """Generate crypto-specific actionable insights."""
    if confidence_level < 0.5:
        return "<b>Analysis:</b>\nLow confidence signals across timeframes.\n<b>Recommendation:</b>\nReduce exposure and wait for clearer setups."

    def get_trend_intensity(momentum: MultiTimeframeDirection, volume: float) -> str:
        if volume > 0.8:
            return "dominant" if momentum == MultiTimeframeDirection.BULLISH else "heavy"
        elif volume > 0.6:
            return "strong" if momentum == MultiTimeframeDirection.BULLISH else "significant"
        elif volume > 0.4:
            return "moderate"
        else:
            return "mild"

    higher_intensity = get_trend_intensity(higher.momentum_bias, higher.volume_strength)
    lower_intensity = get_trend_intensity(lower.momentum_bias, lower.volume_strength)

    # Get base market condition with more nuanced descriptions
    if higher.momentum_bias == lower.momentum_bias:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            base_signal = f"Market showing {higher_intensity} buying pressure with {lower_intensity} momentum on lower timeframes."
            action_plan = (
                "Longs: Maintain positions, add during dips to major support levels.\n"
                "Shorts: Exercise caution, limit to quick reversals at key resistance."
            )
        elif higher.momentum_bias == MultiTimeframeDirection.BEARISH:
            base_signal = f"Market exhibiting {higher_intensity} selling pressure with {lower_intensity} downside momentum."
            action_plan = (
                "Shorts: Maintain positions, add during relief rallies to resistance.\n"
                "Longs: Limited to quick scalps at major support levels."
            )
        else:
            base_signal = f"Market in equilibrium with {lower_intensity} two-sided action."
            action_plan = (
                "Both Directions: Focus on range extremes.\n"
                "Monitor for range expansion and breakout opportunities."
            )
    else:
        base_signal = (
            f"Potential trend shift: {higher_intensity} {higher.momentum_bias.value} on higher timeframes "
            f"versus {lower_intensity} {lower.momentum_bias.value} on lower timeframes."
        )
        if lower.momentum_bias == MultiTimeframeDirection.BULLISH:
            action_plan = (
                "Longs: Scale in carefully with defined risk below key support.\n"
                "Shorts: Consider profit taking, avoid adding to positions."
            )
        else:
            action_plan = (
                "Shorts: Scale in carefully with defined risk above key resistance.\n"
                "Longs: Consider profit taking, avoid adding to positions."
            )

    # Add crypto-specific warnings and opportunities
    risk_warnings = []
    opportunities = []

    # Liquidation cascade risks
    if higher.liquidation_risk == LiquidationRisk.HIGH:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            risk_warnings.append("High risk of short liquidations, potential for violent upside moves")
        else:
            risk_warnings.append("High risk of long liquidations, protect positions with strict stops")

    # Funding rate opportunities
    if abs(higher.funding_sentiment) > 0.7:
        if higher.funding_sentiment > 0:
            opportunities.append("High positive funding offers counter-trend short opportunities")
        else:
            opportunities.append("High negative funding offers counter-trend long opportunities")

    # Liquidity-based insights
    if higher.liquidity_state == MarketLiquidity.LOW:
        risk_warnings.append("Low liquidity environment, expect higher slippage and volatile moves")
        if higher.volatility_state == VolatilityState.HIGH:
            risk_warnings.append("High volatility with low liquidity, reduce position sizes")

    warnings = "\n<b>Risk Warnings:</b>\n" + "\n".join(f"- {w}" for w in risk_warnings) if risk_warnings else ""
    opps = "\n<b>Opportunities:</b>\n" + "\n".join(f"- {o}" for o in opportunities) if opportunities else ""

    return f"<b>Analysis:</b>\n{base_signal}\n<b>Strategy:</b>\n{action_plan}{warnings}{opps}"

def _determine_dual_market_context(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> str:
    """Determine market context from two timeframe groups."""
    if higher.momentum_bias == lower.momentum_bias:
        context = higher.momentum_bias.value
        if higher.volume_strength > 0.7 and lower.volume_strength > 0.6:
            return f"high-conviction {context} trend"
        elif higher.volume_strength > 0.5:
            return f"established {context} trend"
        else:
            return f"developing {context} bias"
    
    return f"{higher.momentum_bias.value} structure with {lower.momentum_bias.value} short-term momentum"

def _determine_dual_trend_strength(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> str:
    """Determine trend strength from two timeframe groups."""
    avg_alignment = (higher.internal_alignment * 0.6 + lower.internal_alignment * 0.4)
    
    if avg_alignment > 0.85:
        return "Extremely Strong"
    elif avg_alignment > 0.7:
        return "Very Strong"
    elif avg_alignment > 0.5:
        return "Strong"
    elif avg_alignment > 0.3:
        return "Moderate"
    else:
        return "Weak"

def _determine_direction(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    confidence_level: float
) -> MultiTimeframeDirection:
    """
    Determine the trading direction with enhanced alignment to trend strength 
    and market context analysis.
    """
    # Calculate composite strength similar to trend strength calculation
    avg_alignment = (higher.internal_alignment * 0.6 + lower.internal_alignment * 0.4)
    
    # Use volume confirmation like market context
    strong_volume = (higher.volume_strength > 0.7 and lower.volume_strength > 0.6)
    moderate_volume = (higher.volume_strength > 0.5)
    
    # If confidence is too low, return neutral (increased threshold for better quality)
    if confidence_level < 0.65:  # Slightly higher threshold
        return MultiTimeframeDirection.NEUTRAL
        
    # Strong conviction setup
    if avg_alignment > 0.7 and strong_volume:  # Aligned with "Very Strong" trend
        if higher.momentum_bias == lower.momentum_bias:
            return higher.momentum_bias
    
    # Moderate conviction setup
    elif avg_alignment > 0.5 and moderate_volume:  # Aligned with "Strong" trend
        if higher.momentum_bias == lower.momentum_bias:
            return higher.momentum_bias
    
    # Higher timeframe dominance with strong signals
    elif higher.internal_alignment > 0.7 and higher.volume_strength > 0.6:
        return higher.momentum_bias
    
    return MultiTimeframeDirection.NEUTRAL
