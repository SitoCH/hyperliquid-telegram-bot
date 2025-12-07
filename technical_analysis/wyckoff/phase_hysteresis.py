"""
Phase hysteresis module to reduce phase flip-flopping in Wyckoff analysis.

This module implements a confirmation mechanism that requires N consecutive
phase readings before changing the phase, reducing noise in short-term timeframes.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from .wyckoff_types import WyckoffPhase, Timeframe

# Number of consecutive confirmations required before changing phase
PHASE_CONFIRMATION_BARS: Dict[Timeframe, int] = {
    Timeframe.MINUTES_15: 2,  # Require 2 consecutive readings (30 min of confirmation)
    Timeframe.MINUTES_30: 2,  # Require 2 consecutive readings (1 hour of confirmation)
    Timeframe.HOUR_1: 1,      # Less hysteresis for longer timeframes
    Timeframe.HOURS_4: 1,
    Timeframe.HOURS_8: 1,
}


@dataclass
class PhaseTracker:
    """Tracks phase history for a single coin/timeframe combination."""
    current_phase: WyckoffPhase = WyckoffPhase.UNKNOWN
    pending_phase: Optional[WyckoffPhase] = None
    confirmation_count: int = 0
    
    def update(self, new_phase: WyckoffPhase, required_confirmations: int) -> Tuple[WyckoffPhase, bool]:
        """
        Update the phase tracker with a new phase reading.
        
        Returns:
            Tuple of (confirmed_phase, is_transition_pending)
        """
        # First reading - accept immediately
        if self.current_phase == WyckoffPhase.UNKNOWN:
            self.current_phase = new_phase
            self.pending_phase = None
            self.confirmation_count = 0
            return new_phase, False
        
        # Same as current phase - reset any pending transition
        if new_phase == self.current_phase:
            self.pending_phase = None
            self.confirmation_count = 0
            return self.current_phase, False
        
        # New phase different from current
        if self.pending_phase == new_phase:
            # Continue counting confirmations for this pending phase
            self.confirmation_count += 1
            if self.confirmation_count >= required_confirmations:
                # Confirmed - switch to new phase
                self.current_phase = new_phase
                self.pending_phase = None
                self.confirmation_count = 0
                return self.current_phase, False
            else:
                # Still waiting for confirmation - return current phase
                return self.current_phase, True
        else:
            # Different pending phase - reset counter
            self.pending_phase = new_phase
            self.confirmation_count = 1
            if required_confirmations <= 1:
                # No hysteresis needed - switch immediately
                self.current_phase = new_phase
                self.pending_phase = None
                self.confirmation_count = 0
                return self.current_phase, False
            else:
                # Waiting for confirmation - return current phase
                return self.current_phase, True


class PhaseHysteresisManager:
    """
    Global manager for phase hysteresis across all coin/timeframe combinations.
    
    This uses a simple in-memory cache. Phase history is lost on restart,
    which is acceptable as it will quickly re-establish after a few candles.
    """
    
    _instance: Optional['PhaseHysteresisManager'] = None
    _trackers: Dict[str, PhaseTracker]
    
    def __new__(cls) -> 'PhaseHysteresisManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._trackers = defaultdict(PhaseTracker)
        return cls._instance
    
    def get_confirmed_phase(
        self, 
        coin: str, 
        timeframe: Timeframe, 
        detected_phase: WyckoffPhase,
        uncertain: bool
    ) -> Tuple[WyckoffPhase, bool]:
        """
        Get the confirmed phase after applying hysteresis.
        
        Args:
            coin: The coin symbol
            timeframe: The timeframe being analyzed
            detected_phase: The newly detected phase
            uncertain: Whether the detection was uncertain
            
        Returns:
            Tuple of (confirmed_phase, is_uncertain)
            The is_uncertain flag is set to True if we're still waiting for phase confirmation
        """
        key = f"{coin}_{timeframe.name}"
        tracker = self._trackers[key]
        
        # Get required confirmations for this timeframe
        required = PHASE_CONFIRMATION_BARS.get(timeframe, 1)
        
        # If the detection was uncertain, require more confirmations
        if uncertain:
            required = max(required, 2)
        
        confirmed_phase, is_pending = tracker.update(detected_phase, required)
        
        # If a transition is pending, mark as uncertain
        final_uncertain = uncertain or is_pending
        
        return confirmed_phase, final_uncertain
    
    def reset(self, coin: Optional[str] = None, timeframe: Optional[Timeframe] = None) -> None:
        """Reset phase history for a specific coin/timeframe or all."""
        if coin is None and timeframe is None:
            self._trackers.clear()
        elif coin is not None and timeframe is not None:
            key = f"{coin}_{timeframe.name}"
            if key in self._trackers:
                del self._trackers[key]
        elif coin is not None:
            # Reset all timeframes for this coin
            keys_to_remove = [k for k in self._trackers if k.startswith(f"{coin}_")]
            for key in keys_to_remove:
                del self._trackers[key]


# Global instance
phase_hysteresis = PhaseHysteresisManager()
