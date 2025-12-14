from typing import List


from .wyckoff_multi_timeframe_types import MultiTimeframeDirection, TimeframeGroupAnalysis

from ..wyckoff_types import (
    is_bearish_phase, is_bullish_phase
)

from .wyckoff_multi_timeframe_types import (
    MODERATE_VOLUME_THRESHOLD, LOW_VOLUME_THRESHOLD
)

def determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall market direction using a weighted voting system based on timeframe phases and momentum.
    
    Improved to require intermediate trend confirmation before generating signals - reduces counter-trend entries.
    """
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL

    # Calculate weights for each direction based on timeframe importance and phase certainty
    bullish_weight = 0.0
    bearish_weight = 0.0
    total_weight = 0.0
    
    # Track short-term and intermediate momentum separately for confirmation
    short_term_momentum = None
    intermediate_momentum = None
    
    for idx, analysis in enumerate(analyses):
        # Track short-term (first) and intermediate (second) momentum
        if idx == 0:
            short_term_momentum = analysis.momentum_bias
        elif idx == 1:
            intermediate_momentum = analysis.momentum_bias
        
        # Base weight from the timeframe group
        weight = analysis.group_weight
        
        # Apply certainty modifier
        certainty_factor = 0.5 if analysis.uncertain_phase else 1.0
        
        # Volume confirmation factor
        volume_factor = 1.0
        if analysis.volume_strength > MODERATE_VOLUME_THRESHOLD:
            volume_factor = 1.2
        elif analysis.volume_strength < LOW_VOLUME_THRESHOLD:
            volume_factor = 0.8

        # Reduce short-term weight if intermediate disagrees
        intermediate_confirmation_factor = 1.0
        if idx == 0 and intermediate_momentum is not None:  # Short-term analysis
            if analysis.momentum_bias != MultiTimeframeDirection.NEUTRAL:
                if analysis.momentum_bias != intermediate_momentum and intermediate_momentum != MultiTimeframeDirection.NEUTRAL:
                    # Short-term disagrees with intermediate - reduce its influence by 40%
                    intermediate_confirmation_factor = 0.6
        
        # Calculate adjusted weight for this timeframe
        adjusted_weight = weight * certainty_factor * volume_factor * intermediate_confirmation_factor
        total_weight += adjusted_weight
        
        # Assign weight based on phase and momentum bias
        if is_bullish_phase(analysis.dominant_phase) or analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
            # Add full weight if both phase and momentum align, otherwise partial weight
            if is_bullish_phase(analysis.dominant_phase) and analysis.momentum_bias == MultiTimeframeDirection.BULLISH:
                bullish_weight += adjusted_weight
            else:
                # Partial contribution when only phase or only momentum is bullish
                bullish_weight += adjusted_weight * 0.6
                
        if is_bearish_phase(analysis.dominant_phase) or analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
            # Add full weight if both phase and momentum align, otherwise partial weight
            if is_bearish_phase(analysis.dominant_phase) and analysis.momentum_bias == MultiTimeframeDirection.BEARISH:
                bearish_weight += adjusted_weight
            else:
                # Partial contribution when only phase or only momentum is bearish
                bearish_weight += adjusted_weight * 0.6

    # Check if we have enough total weight to make a decision
    if total_weight < 0.01:
        return MultiTimeframeDirection.NEUTRAL
        
    # Calculate relative strength of each direction
    bullish_strength = bullish_weight / total_weight
    bearish_strength = bearish_weight / total_weight
    
    # Calculate the difference to determine if there's a clear winner
    direction_difference = abs(bullish_strength - bearish_strength)
    
    # Determine thresholds based on volume and consistency
    avg_volume = sum(a.volume_strength for a in analyses) / len(analyses)
    
    # Dynamic threshold based on volume and number of certain phases
    certain_phases = sum(1 for a in analyses if not a.uncertain_phase)
    certain_ratio = certain_phases / len(analyses)
    
    # Lower threshold when we have high certainty or strong volume
    decision_threshold = 0.20
    if certain_ratio > 0.7 or avg_volume > MODERATE_VOLUME_THRESHOLD:
        decision_threshold = 0.12
    
    # Make final decision
    if direction_difference < decision_threshold:
        return MultiTimeframeDirection.NEUTRAL
    
    # Return the stronger direction
    if bullish_strength > bearish_strength:
        return MultiTimeframeDirection.BULLISH
    else:
        return MultiTimeframeDirection.BEARISH
