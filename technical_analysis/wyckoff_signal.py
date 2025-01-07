import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import WyckoffState, WyckoffPhase, MarketPattern, CompositeAction, VolumeState, VolatilityState, EffortResult
from .funding_rates_cache import FundingRateEntry
from statistics import mean
from typing import List, Optional


def detect_actionable_wyckoff_signal(
    df: pd.DataFrame,
    funding_rates: Optional[List[FundingRateEntry]] = None,
    extreme_funding_threshold: float = 0.01,  # 1% threshold for extreme funding
    min_volume_factor: float = 1.5,  # Minimum volume compared to average
    min_confirmation_bars: int = 3  # Minimum bars to confirm pattern
) -> bool:
    """
    Detect high-probability Wyckoff setups focusing on specific events.
    """
    if len(df) < min_confirmation_bars or 'wyckoff' not in df.columns:
        return False
        
    # Ensure we have valid WyckoffState objects
    if not isinstance(df['wyckoff'].iloc[-1], WyckoffState):
        return False
        
    current_state = df['wyckoff'].iloc[-1]
    
    # Check volume confirmation
    recent_volume = df['v'].iloc[-min_confirmation_bars:]
    volume_ma = df['v'].rolling(24).mean().iloc[-1]  # 24-period volume MA
    volume_confirmed = recent_volume.mean() > volume_ma * min_volume_factor
    
    # Check for repeated patterns (avoid false signals)
    recent_states = df['wyckoff'].iloc[-min_confirmation_bars:]
    # Filter out any non-WyckoffState values
    valid_states = [state for state in recent_states if isinstance(state, WyckoffState)]
    
    if not valid_states or len(valid_states) < min_confirmation_bars:
        return False
        
    consistent_phase = all(state.phase == current_state.phase for state in valid_states)
    
    # Check for spring or upthrust with stricter conditions
    event_signal = (
        ((current_state.is_spring and current_state.phase == WyckoffPhase.ACCUMULATION) or
         (current_state.is_upthrust and current_state.phase == WyckoffPhase.DISTRIBUTION)) and
        volume_confirmed and
        consistent_phase and
        current_state.volatility != VolatilityState.HIGH  # Avoid extremely volatile periods
    )
    
    # Calculate weighted average funding rate
    funding_signal = False
    if funding_rates and len(funding_rates) > 0:
        now = max(rate['time'] for rate in funding_rates)
        weighted_sum = 0
        total_weight = 0
        decay_factor = 0.85  # Higher weight to recent rates
        
        for rate in funding_rates:
            time_diff = (now - rate['time']) / (1000 * 3600)  # Hours difference
            weight = decay_factor ** time_diff
            weighted_sum += rate['fundingRate'] * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_funding = weighted_sum / total_weight
            funding_signal = (
                (avg_funding < -extreme_funding_threshold and 
                 current_state.phase == WyckoffPhase.ACCUMULATION and
                 current_state.volume == VolumeState.HIGH) or
                (avg_funding > extreme_funding_threshold and 
                 current_state.phase == WyckoffPhase.DISTRIBUTION and
                 current_state.volume == VolumeState.HIGH)
            )
    
    return (
        (event_signal or funding_signal) and
        not current_state.uncertain_phase and
        current_state.effort_vs_result == EffortResult.STRONG and
        volume_confirmed
    )
