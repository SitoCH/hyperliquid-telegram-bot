from enum import Enum
from dataclasses import dataclass
from typing import Final

from .wyckoff_types import (
    WyckoffPhase, CompositeAction, MarketLiquidity,
    LiquidationRisk, VolatilityState, Timeframe, _TIMEFRAME_SETTINGS
)

# Calculate timeframe group weights from settings
SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15, Timeframe.MINUTES_30}
INTERMEDIATE_TIMEFRAMES = {Timeframe.HOUR_1, Timeframe.HOURS_2}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8, Timeframe.DAY_1}

# Dynamic calculation of group weights
SHORT_TERM_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES)
INTERMEDIATE_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES)
LONG_TERM_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in LONG_TERM_TIMEFRAMES)

# Momentum thresholds
STRONG_MOMENTUM: Final[float] = 0.75  # Reduced from 0.8 to account for crypto volatility
MODERATE_MOMENTUM: Final[float] = 0.5  # Reduced from 0.6 for more sensitive signals
WEAK_MOMENTUM: Final[float] = 0.3     # Reduced from 0.4 to catch early moves
MIXED_MOMENTUM: Final[float] = 0.15   # Reduced from 0.2 for noise filtering
LOW_MOMENTUM: Final[float] = 0.05     # Reduced from 0.1 to better identify ranging periods

# Analysis weight factors
DIRECTIONAL_WEIGHT: Final[float] = 0.50
VOLUME_WEIGHT: Final[float] = 0.35
PHASE_WEIGHT: Final[float] = 0.15

class MultiTimeframeDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

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

@dataclass
class MultiTimeframeContext:
    description: str
    shoud_notify: bool

@dataclass
class AllTimeframesAnalysis:
    short_term: TimeframeGroupAnalysis
    intermediate: TimeframeGroupAnalysis
    long_term: TimeframeGroupAnalysis
    overall_direction: MultiTimeframeDirection
    confidence_level: float
    alignment_score: float
    momentum_intensity: float = 0.0