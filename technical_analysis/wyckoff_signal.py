import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import WyckoffState, WyckoffPhase, MarketPattern, CompositeAction, VolumeState, VolatilityState, EffortResult
from statistics import mean
from typing import List, Optional


def detect_wyckoff_flip(current_wyckoff: WyckoffState, prev_wyckoff: WyckoffState, older_wyckoff: WyckoffState) -> bool:
    """
    Detect significant Wyckoff phase transitions in crypto markets.
    Enhanced for crypto's higher volatility and manipulation risks.
    """
    valid_transitions = {
        (WyckoffPhase.ACCUMULATION, WyckoffPhase.MARKUP, 1),
        (WyckoffPhase.RANGING, WyckoffPhase.ACCUMULATION, 1),
        (WyckoffPhase.MARKDOWN, WyckoffPhase.ACCUMULATION, 1),
        (WyckoffPhase.DISTRIBUTION, WyckoffPhase.MARKDOWN, -1),
        (WyckoffPhase.RANGING, WyckoffPhase.DISTRIBUTION, -1),
        (WyckoffPhase.MARKUP, WyckoffPhase.DISTRIBUTION, -1)
    }
    
    transition = (prev_wyckoff.phase, current_wyckoff.phase)
    
    for from_phase, to_phase, expected_momentum in valid_transitions:
        if transition == (from_phase, to_phase):
            momentum_aligned = (
                (expected_momentum > 0 and 
                 current_wyckoff.pattern == MarketPattern.TRENDING and
                 current_wyckoff.composite_action in [CompositeAction.MARKING_UP, CompositeAction.ACCUMULATING]) or
                (expected_momentum < 0 and 
                 current_wyckoff.pattern == MarketPattern.TRENDING and
                 current_wyckoff.composite_action in [CompositeAction.MARKING_DOWN, CompositeAction.DISTRIBUTING])
            )
            
            # Detect potential manipulation
            potential_manipulation = (
                current_wyckoff.volume == VolumeState.HIGH and 
                current_wyckoff.volatility == VolatilityState.HIGH and
                current_wyckoff.volume_spread == VolumeState.HIGH
            )
            
            # Validate phase transition
            return (
                momentum_aligned and
                not current_wyckoff.uncertain_phase and
                prev_wyckoff.phase == older_wyckoff.phase and
                current_wyckoff.volume == VolumeState.HIGH and
                current_wyckoff.volatility in [VolatilityState.HIGH, VolatilityState.NORMAL] and
                not potential_manipulation and
                current_wyckoff.effort_vs_result == EffortResult.STRONG and
                not (current_wyckoff.is_spring and current_wyckoff.phase == WyckoffPhase.DISTRIBUTION) and
                not (current_wyckoff.is_upthrust and current_wyckoff.phase == WyckoffPhase.ACCUMULATION)
            )
    
    return False

def detect_actionable_wyckoff_signal(
    df: pd.DataFrame,
    min_confirmation_periods: int = 3,
    volume_threshold: float = 2.0,
    momentum_threshold: float = 0.03
) -> bool:
    """Enhanced detection of high-probability Wyckoff trading opportunities for crypto markets."""
    if len(df) < min_confirmation_periods:
        return False
        
    current_state = df['wyckoff'].iloc[-1]
    
    phase_changed = detect_wyckoff_flip(
        df['wyckoff'].iloc[-1],
        df['wyckoff'].iloc[-2],
        df['wyckoff'].iloc[-3]
    )
    
    if not phase_changed:
        return False
    
    recent_volume = df['v'].iloc[-5:]
    volume_ma = df['v'].rolling(30).mean().iloc[-1]
    
    # Core validation checks
    volume_valid = (
        (recent_volume > volume_ma).sum() >= 3 and
        recent_volume.iloc[-1] > volume_ma * volume_threshold and
        not (recent_volume.iloc[-1] > recent_volume.iloc[-2:].mean() * 3)
    )
    
    price_valid = (
        (current_state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION] and
         df['l'].iloc[-1] > df['l'].iloc[-5:].min() * 0.985) or
        (current_state.phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION] and
         df['h'].iloc[-1] < df['h'].iloc[-5:].max() * 1.015)
    )
    
    volatility = df['c'].pct_change().std() * np.sqrt(len(df))
    momentum = (df['c'].iloc[-1] - df['c'].iloc[-2]) / df['c'].iloc[-2]
    momentum_valid = (
        (momentum > momentum_threshold * (1 + volatility) and 
         current_state.phase in [WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION]) or
        (momentum < -momentum_threshold * (1 + volatility) and 
         current_state.phase in [WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION])
    )
    
    # Stability checks
    stability_valid = (
        (recent_volume.std() / recent_volume.mean() < 2.0) and  # Volume stability
        ((df['h'].iloc[-3:] - df['l'].iloc[-3:]).std() / df['c'].iloc[-1] < 0.05)  # Price stability
    )
 
    return (
        phase_changed and
        volume_valid and
        price_valid and
        momentum_valid and
        stability_valid and
        current_state.effort_vs_result == EffortResult.STRONG and
        not current_state.uncertain_phase and
        (volume_valid and momentum_valid)
    )
