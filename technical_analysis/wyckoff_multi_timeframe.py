from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd  # type: ignore[import]

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
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

        # Update group weights based on timeframe composition
        higher_analysis.group_weight = _calculate_group_weight(higher_tf)
        lower_analysis.group_weight = _calculate_group_weight(lower_tf)

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
    
    # Track exhaustion signals
    upside_exhaustion = 0
    downside_exhaustion = 0
    
    # Track distribution/accumulation signals
    distribution_signals = 0
    accumulation_signals = 0

    # Track rapid movement signals
    rapid_bullish_moves = 0
    rapid_bearish_moves = 0
    
    for tf, state in group.items():
        weight = get_phase_weight(tf)
        
        # Check for potential exhaustion signals
        if state.phase == WyckoffPhase.DISTRIBUTION:
            distribution_signals += 1
        elif state.phase == WyckoffPhase.ACCUMULATION:
            accumulation_signals += 1
            
        # Detect potential exhaustion based on phase combinations
        if (state.phase == WyckoffPhase.MARKUP and 
            state.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.CONSOLIDATING]):
            upside_exhaustion += 1
            
        if (state.phase == WyckoffPhase.MARKDOWN and 
            state.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.CONSOLIDATING]):
            downside_exhaustion += 1
            
        # Adjust weights based on exhaustion signals
        if state.is_upthrust:
            upside_exhaustion += 1
            weight *= 1.2  # Increase weight for potential reversal signals
        elif state.is_spring:
            downside_exhaustion += 1
            weight *= 1.2
            
        # Reduce weight if we see contrary volume signals
        if (state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.MARKDOWN] and 
            state.volume == VolumeState.LOW):
            weight *= 0.8
            
        # Factor in effort vs result analysis
        if state.effort_vs_result == EffortResult.WEAK:
            if state.phase == WyckoffPhase.MARKUP:
                upside_exhaustion += 1
            elif state.phase == WyckoffPhase.MARKDOWN:
                downside_exhaustion += 1

        # Detect rapid price movements
        if state.phase == WyckoffPhase.MARKUP and state.volume == VolumeState.HIGH:
            rapid_bullish_moves += 1
        elif state.phase == WyckoffPhase.MARKDOWN and state.volume == VolumeState.HIGH:
            rapid_bearish_moves += 1
            
        # Increase weight for strong directional moves
        if state.pattern == MarketPattern.TRENDING:
            if state.volume == VolumeState.HIGH:
                weight *= 1.3  # 30% boost for high volume trends
        
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

    # Calculate volume strength with exhaustion consideration
    volume_strength = sum(1 for s in group.values() if s.volume == VolumeState.HIGH) / len(group)
    
    # Adjust volume strength based on exhaustion signals
    if upside_exhaustion >= len(group) // 2:
        volume_strength *= 0.7  # Reduce volume significance if exhaustion detected
    elif downside_exhaustion >= len(group) // 2:
        volume_strength *= 0.7

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

    # Analyze market states
    liquidity_counts = {state.liquidity: 0 for state in group.values()}
    risk_counts = {state.liquidation_risk: 0 for state in group.values()}
    volatility_counts = {state.volatility: 0 for state in group.values()}
    
    for state in group.values():
        liquidity_counts[state.liquidity] += 1
        risk_counts[state.liquidation_risk] += 1
        volatility_counts[state.volatility] += 1
        
    liquidity_state = max(liquidity_counts.items(), key=lambda x: x[1])[0]
    liquidation_risk = max(risk_counts.items(), key=lambda x: x[1])[0]
    volatility_state = max(volatility_counts.items(), key=lambda x: x[1])[0]

    # Enhanced momentum bias calculation with exhaustion consideration
    bullish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP] and upside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.ACCUMULATING, CompositeAction.MARKING_UP] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKDOWN) or
        (s.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKDOWN)
    ))
    
    bearish_signals = sum(1 for s in group.values() if (
        (s.phase in [WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN] and downside_exhaustion < len(group) // 2) or 
        s.composite_action in [CompositeAction.DISTRIBUTING, CompositeAction.MARKING_DOWN] or
        (s.composite_action == CompositeAction.REVERSING and s.phase == WyckoffPhase.MARKUP) or
        (s.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE] and s.volume == VolumeState.HIGH) or
        (s.liquidation_risk == LiquidationRisk.HIGH and s.phase == WyckoffPhase.MARKUP)
    ))

    # Adjust momentum bias based on exhaustion signals
    consolidation_count = sum(1 for s in group.values() if s.composite_action == CompositeAction.CONSOLIDATING)
    total_signals = len(group)

    # Modified momentum bias calculation
    if consolidation_count / total_signals > 0.5:
        momentum_bias = MultiTimeframeDirection.NEUTRAL
    else:
        # Factor in exhaustion signals
        if upside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bullish_signals > bearish_signals else MultiTimeframeDirection.BEARISH
        elif downside_exhaustion >= len(group) // 2:
            momentum_bias = MultiTimeframeDirection.NEUTRAL if bearish_signals > bullish_signals else MultiTimeframeDirection.BULLISH
        else:
            momentum_bias = (
                MultiTimeframeDirection.BULLISH if bullish_signals > bearish_signals else
                MultiTimeframeDirection.BEARISH if bearish_signals > bullish_signals else
                MultiTimeframeDirection.NEUTRAL
            )

    # Modify momentum bias calculation
    if rapid_bullish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BULLISH
    elif rapid_bearish_moves >= len(group) // 2:
        momentum_bias = MultiTimeframeDirection.BEARISH

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

def _calculate_group_weight(timeframes: Dict[Timeframe, WyckoffState]) -> float:
    """Calculate total weight for a timeframe group based on phase weights."""
    return sum(get_phase_weight(tf) for tf in timeframes.keys())

def _calculate_dual_groups_alignment(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis
) -> float:
    """Calculate alignment between higher and lower timeframe groups."""
    # Calculate phase alignment
    phase_alignment = 1.0 if higher.dominant_phase == lower.dominant_phase else 0.0

    # Calculate bias alignment
    bias_alignment = 1.0 if higher.momentum_bias == lower.momentum_bias else 0.0

    # Calculate relative weights
    total_weight = higher.group_weight + lower.group_weight
    higher_weight = higher.group_weight / total_weight if total_weight > 0 else 0.5
    lower_weight = lower.group_weight / total_weight if total_weight > 0 else 0.5

    # Calculate weighted alignment score
    alignment = (phase_alignment * 0.6) + (bias_alignment * 0.4)

    # Apply internal alignment weights based on timeframe weights
    weighted_alignment = (
        alignment * 0.6 +
        (higher.internal_alignment * higher_weight +
         lower.internal_alignment * lower_weight) * 0.4
    )

    return weighted_alignment

def _calculate_dual_confidence(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float
) -> float:
    """Calculate confidence level with enhanced intermediate timeframe focus."""
    # Adjusted weight factors for more balanced analysis
    alignment_weight = 0.40    # Increased from 0.35
    volume_weight = 0.35       # Unchanged
    consistency_weight = 0.25  # Reduced from 0.30

    # Volume confirmation with balanced weighting
    # Give more weight to immediate timeframe volume signals
    lower_volume_factor = min(0.55, lower.volume_strength * 1.1)  # Reduced max influence to 55%
    higher_volume_factor = 1.0 - lower_volume_factor
    
    volume_confirmation = (
        higher.volume_strength * higher_volume_factor +
        lower.volume_strength * lower_volume_factor
    )

    # Enhanced trend consistency check with focus on intermediate timeframes
    trend_consistency = (
        1.0 if higher.momentum_bias == lower.momentum_bias else
        0.8 if lower.volume_strength > 0.7 else  # Strong lower timeframe moves get 80% credit
        0.4  # Increased base consistency during disagreement
    )

    # Adjusted rapid movement bonus
    rapid_movement_bonus = 0.0
    if higher.momentum_bias == lower.momentum_bias:
        if higher.volume_strength > 0.65 and lower.volume_strength > 0.65:  # Reduced thresholds
            rapid_movement_bonus = 0.12  # Reduced from 0.15

    confidence = (
        groups_alignment * alignment_weight +
        volume_confirmation * volume_weight +
        trend_consistency * consistency_weight +
        rapid_movement_bonus
    )

    return min(confidence, 1.0)

def _generate_dual_group_description(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    groups_alignment: float,
    confidence_level: float,
    higher_tf: Dict[Timeframe, WyckoffState]
) -> str:
    """Generate a narrative description focused on timeframe relationships."""
    alignment_pct = f"{groups_alignment * 100:.0f}%"
    confidence_pct = f"{confidence_level * 100:.0f}%"

    # Analyze timeframe-specific trends
    higher_trend = _get_timeframe_trend_description(higher)
    lower_trend = _get_timeframe_trend_description(lower)

    # Determine market structure based on timeframe alignment
    structure = _get_market_structure(higher, lower)

    # Get dominant market conditions
    market_context = _determine_dual_market_context(higher, lower)
    trend_strength = _determine_dual_trend_strength(higher, lower)
    actionable_insight = _generate_dual_actionable_insight(higher, lower, confidence_level)
    
    # Get funding state (same for all timeframes)
    funding_state = next(iter(higher_tf.values())).funding_state if higher_tf else FundingState.UNKNOWN

    # Determine trend emoji based on timeframe alignment
    emoji = _get_trend_emoji(higher, lower, confidence_level)

    description = (
        f"{emoji} Market Structure Analysis:\n"
        f"Trend: {trend_strength} {market_context}\n"
        f"Market Structure: {structure}\n\n"
        f"Higher Timeframes:\n{higher_trend}\n"
        f"Lower Timeframes:\n{lower_trend}\n\n"
        f"Funding Rate: {funding_state.value}\n"
        f"Timeframe Alignment: {alignment_pct}\n"
        f"Signal Confidence: {confidence_pct}\n\n"
        f"{actionable_insight}"
    )

    return description

def _get_timeframe_trend_description(analysis: TimeframeGroupAnalysis) -> str:
    """Generate detailed trend description for a timeframe group."""
    phase_desc = f"{analysis.dominant_phase.value}"
    action_desc = analysis.dominant_action.value
    
    volume_desc = (
        "strong volume" if analysis.volume_strength > 0.7 else
        "moderate volume" if analysis.volume_strength > 0.4 else
        "light volume"
    )
    
    return f"• {phase_desc} phase {action_desc} with {volume_desc}"

def _get_market_structure(higher: TimeframeGroupAnalysis, lower: TimeframeGroupAnalysis) -> str:
    """Determine overall market structure based on timeframe relationships."""
    if higher.dominant_phase == lower.dominant_phase:
        return f"Aligned {higher.dominant_phase.value} structure across timeframes"
        
    if higher.momentum_bias == lower.momentum_bias:
        return f"Mixed structure with aligned {higher.momentum_bias.value} momentum"
        
    higher_state = "bullish" if higher.momentum_bias == MultiTimeframeDirection.BULLISH else "bearish"
    lower_state = "bullish" if lower.momentum_bias == MultiTimeframeDirection.BULLISH else "bearish"
    return f"Transitional structure ({higher_state} → {lower_state})"

def _get_trend_emoji(
    higher: TimeframeGroupAnalysis,
    lower: TimeframeGroupAnalysis,
    confidence_level: float
) -> str:
    """Get appropriate trend emoji based on timeframe analysis."""
    if confidence_level < 0.5:
        return "📊"
        
    if higher.momentum_bias == lower.momentum_bias:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            return "📈" if confidence_level > 0.7 else "↗️"
        if higher.momentum_bias == MultiTimeframeDirection.BEARISH:
            return "📉" if confidence_level > 0.7 else "↘️"
    
    return "↔️"

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

    # Get base market condition with aligned terminology
    if higher.momentum_bias == lower.momentum_bias:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            base_signal = (
                f"Higher timeframes {higher.dominant_action.value} with {higher_intensity} strength, "
                f"lower timeframes confirming with {lower_intensity} momentum."
            )
            action_plan = (
                "Longs: Maintain positions, add during dips to major support levels.\n"
                "Shorts: Exercise caution, limit to quick reversals at key resistance."
            )
        elif higher.momentum_bias == MultiTimeframeDirection.BEARISH:
            base_signal = (
                f"Higher timeframes {higher.dominant_action.value} with {higher_intensity} pressure, "
                f"lower timeframes showing {lower_intensity} continuation."
            )
            action_plan = (
                "Shorts: Maintain positions, add during relief rallies to resistance.\n"
                "Longs: Limited to quick scalps at major support levels."
            )
        else:
            base_signal = (
                f"Market in equilibrium phase, higher timeframes {higher.dominant_action.value}, "
                f"lower timeframes showing {lower_intensity} two-sided action."
            )
            action_plan = (
                "Both Directions: Focus on range extremes.\n"
                "Monitor for range expansion and breakout opportunities."
            )
    else:
        base_signal = (
            f"Timeframe divergence: higher timeframes {higher.dominant_action.value} ({higher_intensity}), "
            f"while lower timeframes {lower.dominant_action.value} ({lower_intensity})."
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

    # Add risk warnings and opportunities
    risk_warnings = []
    opportunities = []

    if higher.liquidation_risk == LiquidationRisk.HIGH:
        if higher.momentum_bias == MultiTimeframeDirection.BULLISH:
            risk_warnings.append("High risk of short liquidations, potential for violent upside moves")
        else:
            risk_warnings.append("High risk of long liquidations, protect positions with strict stops")

    if abs(higher.funding_sentiment) > 0.7:
        if higher.funding_sentiment > 0:
            opportunities.append("High positive funding offers counter-trend short opportunities")
        else:
            opportunities.append("High negative funding offers counter-trend long opportunities")

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
    Determine market direction with enhanced timeframe analysis.
    Emphasizes alignment between timeframes and validates with volume.
    """
    # Immediate return on very low confidence
    if confidence_level < 0.45:
        return MultiTimeframeDirection.NEUTRAL

    # Calculate relative weights from group weights
    total_weight = higher.group_weight + lower.group_weight
    higher_weight = higher.group_weight / total_weight if total_weight > 0 else 0.5
    lower_weight = lower.group_weight / total_weight if total_weight > 0 else 0.5
    
    weighted_alignment = (
        higher.internal_alignment * higher_weight +
        lower.internal_alignment * lower_weight
    )

    # Volume confirmation thresholds
    strong_higher_volume = higher.volume_strength > 0.65
    strong_lower_volume = lower.volume_strength > 0.55
    
    # Perfect alignment scenario
    if higher.momentum_bias == lower.momentum_bias:
        # Strong conviction setup with weighted alignment
        if weighted_alignment > 0.65 and strong_higher_volume and strong_lower_volume:
            return higher.momentum_bias
            
        # Moderate conviction setup
        if weighted_alignment > 0.55 and (strong_higher_volume or strong_lower_volume):
            return higher.momentum_bias
            
        # Developing trend setup
        if weighted_alignment > 0.45 and confidence_level > 0.60:
            return higher.momentum_bias
    
    # Higher timeframe dominance - weighted by timeframe importance
    if higher.internal_alignment > 0.70 and strong_higher_volume:
        if confidence_level > 0.65 and higher_weight > lower_weight:
            return higher.momentum_bias
    
    # Lower timeframe momentum shift - requires stronger confirmation when weight is lower
    if lower.internal_alignment > 0.75 and strong_lower_volume:
        required_confidence = 0.70 + (higher_weight - lower_weight) * 0.2  # Increase required confidence if higher weight is much larger
        if confidence_level > required_confidence and lower.momentum_bias != higher.momentum_bias:
            # Potential reversal signal
            return lower.momentum_bias
    
    # Transitional market
    if weighted_alignment > 0.50 and confidence_level > 0.60:
        # Favor higher timeframe unless lower shows very strong signals
        if strong_lower_volume and lower.internal_alignment > 0.80 and lower_weight > 0.35:
            return lower.momentum_bias
        return higher.momentum_bias

    return MultiTimeframeDirection.NEUTRAL
