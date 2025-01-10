import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import WyckoffState, WyckoffPhase, MarketPattern, CompositeAction, VolumeState, VolatilityState, EffortResult, FundingState
from .funding_rates_cache import FundingRateEntry
from statistics import mean
from typing import List, Optional


def detect_actionable_wyckoff_signal(
    df: pd.DataFrame,
    funding_rates: Optional[List[FundingRateEntry]] = None,
    min_confirmation_bars: int = 3  # Minimum bars to confirm pattern
) -> bool:
    """
    Detect high-probability Wyckoff setups focusing on specific events and composite actions.
    """
    if len(df) < min_confirmation_bars or 'wyckoff' not in df.columns:
        return False
        
    # Ensure we have valid WyckoffState objects
    if not isinstance(df['wyckoff'].iloc[-1], WyckoffState):
        return False
        
    current_state = df['wyckoff'].iloc[-1]
    
    # Check for repeated patterns (avoid false signals)
    recent_states = df['wyckoff'].iloc[-min_confirmation_bars:]
    valid_states = [state for state in recent_states if isinstance(state, WyckoffState)]
    
    if not valid_states or len(valid_states) < min_confirmation_bars:
        return False
        
    consistent_phase = all(state.phase == current_state.phase for state in valid_states)
    consistent_action = all(state.composite_action == current_state.composite_action for state in valid_states)
    consistent_volume = all(state.volume == VolumeState.HIGH for state in valid_states)
    
    # Enhanced event signal detection including composite actions
    event_signal = (
        # Spring with accumulation signals
        (current_state.is_spring and 
         current_state.phase == WyckoffPhase.ACCUMULATION and
         current_state.composite_action == CompositeAction.ACCUMULATING) or
        # Upthrust with distribution signals
        (current_state.is_upthrust and 
         current_state.phase == WyckoffPhase.DISTRIBUTION and
         current_state.composite_action == CompositeAction.DISTRIBUTING) or
        # Strong markup signals
        (current_state.phase == WyckoffPhase.MARKUP and
         current_state.composite_action == CompositeAction.MARKING_UP and
         current_state.pattern == MarketPattern.TRENDING) or
        # Strong markdown signals
        (current_state.phase == WyckoffPhase.MARKDOWN and
         current_state.composite_action == CompositeAction.MARKING_DOWN and
         current_state.pattern == MarketPattern.TRENDING)
    )
    
    # Combine with consistency checks
    action_signal = (
        event_signal and
        consistent_phase and
        consistent_action and
        consistent_volume and
        current_state.volatility != VolatilityState.HIGH
    )
    
    # Check funding state alignment with market structure
    funding_signal = (
        # Accumulation with negative funding
        (current_state.funding_state in [FundingState.HIGHLY_NEGATIVE, FundingState.NEGATIVE] and 
         current_state.phase == WyckoffPhase.ACCUMULATION and
         current_state.composite_action == CompositeAction.ACCUMULATING and
         current_state.volume == VolumeState.HIGH) or
        # Distribution with positive funding
        (current_state.funding_state in [FundingState.HIGHLY_POSITIVE, FundingState.POSITIVE] and 
         current_state.phase == WyckoffPhase.DISTRIBUTION and
         current_state.composite_action == CompositeAction.DISTRIBUTING and
         current_state.volume == VolumeState.HIGH)
    )
    
    return (
        (action_signal or funding_signal) and
        not current_state.uncertain_phase and
        current_state.effort_vs_result == EffortResult.STRONG and
        current_state.volume == VolumeState.HIGH
    )
