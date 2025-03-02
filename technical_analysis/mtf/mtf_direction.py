import pandas as pd  # type: ignore[import]
import numpy as np
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Final
from dataclasses import dataclass
from .wyckoff_multi_timeframe_description import generate_all_timeframes_description
from technical_analysis.wyckoff_types import SignificantLevelsData
from logging_utils import logger


from .wyckoff_multi_timeframe_types import AllTimeframesAnalysis, MultiTimeframeDirection, TimeframeGroupAnalysis, MultiTimeframeContext

from technical_analysis.wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, _TIMEFRAME_SETTINGS,
    is_bearish_action, is_bullish_action, is_bearish_phase, is_bullish_phase,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

from .wyckoff_multi_timeframe_types import (
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    STRONG_MOMENTUM, MODERATE_MOMENTUM, WEAK_MOMENTUM,
    MIXED_MOMENTUM, LOW_MOMENTUM,
    SHORT_TERM_WEIGHT, INTERMEDIATE_WEIGHT, LONG_TERM_WEIGHT,
    DIRECTIONAL_WEIGHT, VOLUME_WEIGHT, PHASE_WEIGHT
)


def determine_overall_direction(analyses: List[TimeframeGroupAnalysis]) -> MultiTimeframeDirection:
    """Determine overall direction considering Wyckoff phase weights for each timeframe group."""
    if not analyses:
        return MultiTimeframeDirection.NEUTRAL

    # Group timeframes by their actual settings in _TIMEFRAME_SETTINGS
    timeframe_groups = {
        'short': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                           for tf in SHORT_TERM_TIMEFRAMES}],
        'mid': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                         for tf in INTERMEDIATE_TIMEFRAMES}],
        'long': [a for a in analyses if a.group_weight in {_TIMEFRAME_SETTINGS[tf].phase_weight 
                                                          for tf in LONG_TERM_TIMEFRAMES}]
    }

    # Count dominant phases and ensure overall direction respects market structure
    bullish_phases = sum(1 for a in analyses if is_bullish_phase(a.dominant_phase))
    bearish_phases = sum(1 for a in analyses if is_bearish_phase(a.dominant_phase))
    
    # Market structure consistency check
    market_structure_bias = None
    if bearish_phases > bullish_phases and bearish_phases >= len(analyses) / 3:  # Make threshold more lenient
        market_structure_bias = MultiTimeframeDirection.BEARISH
    elif bullish_phases > bearish_phases and bullish_phases >= len(analyses) / 3:
        market_structure_bias = MultiTimeframeDirection.BULLISH

    def get_weighted_direction(group: List[TimeframeGroupAnalysis]) -> Tuple[MultiTimeframeDirection, float, float]:
        if not group:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0

        group_total_weight = sum(a.group_weight for a in group)
        if group_total_weight == 0:
            return MultiTimeframeDirection.NEUTRAL, 0.0, 0.0

        weighted_signals = {
            direction: sum(
                (a.group_weight / group_total_weight) * 
                (1 + a.volume_strength * 0.3) *  # Volume boost
                (1.2 if a.volatility_state == VolatilityState.HIGH else 1.0) *  # Volatility adjustment
                # Phase consistency factor
                (0.7 if direction == MultiTimeframeDirection.BULLISH and 
                      is_bearish_phase(a.dominant_phase) else
                 0.7 if direction == MultiTimeframeDirection.BEARISH and
                      is_bullish_phase(a.dominant_phase) else 1.0)
                for a in group
                if a.momentum_bias == direction
            )
            for direction in MultiTimeframeDirection
        }
        
        strongest = max(weighted_signals.items(), key=lambda x: x[1])
        avg_volume = sum(a.volume_strength for a in group) / len(group)

        return strongest[0], strongest[1], avg_volume

    # Get weighted directions with volume context
    st_dir, st_weight, st_vol = get_weighted_direction(timeframe_groups['short'])
    mid_dir, mid_weight, mid_vol = get_weighted_direction(timeframe_groups['mid'])
    lt_dir, lt_weight, _ = get_weighted_direction(timeframe_groups['long'])

    # Check for high-conviction intraday moves first
    if st_dir != MultiTimeframeDirection.NEUTRAL:
        if st_weight > 0.8 and st_vol > 0.7:  # Strong short-term move with volume
            if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_dir != st_dir:
                return MultiTimeframeDirection.NEUTRAL  # Conflict with intermediate trend
            # Market structure consistency check
            if market_structure_bias and market_structure_bias != st_dir:
                return MultiTimeframeDirection.NEUTRAL  # Conflict with market structure
            return st_dir

    # Check for strong intermediate trend
    if mid_dir != MultiTimeframeDirection.NEUTRAL and mid_weight > 0.7:
        # Allow counter-trend short-term moves if volume is low
        if st_dir != MultiTimeframeDirection.NEUTRAL and st_dir != mid_dir:
            if st_vol < 0.5:  # Low volume counter-trend move
                # Market structure consistency check
                if market_structure_bias and market_structure_bias != mid_dir:
                    return MultiTimeframeDirection.NEUTRAL
                return mid_dir
            return MultiTimeframeDirection.NEUTRAL  # High volume conflict
        # Market structure consistency check
        if market_structure_bias and market_structure_bias != mid_dir:
            return MultiTimeframeDirection.NEUTRAL
        return mid_dir

    # Consider longer-term trend with confirmation
    if lt_dir != MultiTimeframeDirection.NEUTRAL and lt_weight > 0.6:
        # Need confirmation from either shorter timeframe
        if mid_dir == lt_dir or st_dir == lt_dir:
            # Market structure consistency check
            if market_structure_bias and market_structure_bias != lt_dir:
                return MultiTimeframeDirection.NEUTRAL
            return lt_dir
        # Check if counter-trend moves are weak
        if mid_vol < 0.5 and st_vol < 0.5:
            # Market structure consistency check
            if market_structure_bias and market_structure_bias != lt_dir:
                return MultiTimeframeDirection.NEUTRAL
            return lt_dir

    # Check for aligned moves even with lower weights
    if st_dir == mid_dir and mid_dir == lt_dir and st_dir != MultiTimeframeDirection.NEUTRAL:
        if (st_weight + mid_weight + lt_weight) / 3 > 0.5:  # Average weight threshold
            # Market structure consistency check
            if market_structure_bias and market_structure_bias != st_dir:
                return MultiTimeframeDirection.NEUTRAL
            return st_dir

    # When in doubt, respect market structure
    if market_structure_bias:
        return market_structure_bias

    return MultiTimeframeDirection.NEUTRAL
