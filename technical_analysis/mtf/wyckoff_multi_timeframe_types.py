from enum import Enum
from dataclasses import dataclass
from typing import Final

from technical_analysis.wyckoff_types import (
    WyckoffPhase, CompositeAction, MarketLiquidity,
    VolatilityState, Timeframe, _TIMEFRAME_SETTINGS,
    SHORT_TERM_TIMEFRAMES, INTERMEDIATE_TIMEFRAMES, LONG_TERM_TIMEFRAMES, CONTEXT_TIMEFRAMES,
    WyckoffSign
)

# Recalculate group weights
SHORT_TERM_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in SHORT_TERM_TIMEFRAMES)
INTERMEDIATE_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in INTERMEDIATE_TIMEFRAMES)
LONG_TERM_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in LONG_TERM_TIMEFRAMES)
CONTEXT_WEIGHT = sum(_TIMEFRAME_SETTINGS[tf].phase_weight for tf in CONTEXT_TIMEFRAMES)

STRONG_VOLUME_THRESHOLD: Final[float] = 0.78
MODERATE_VOLUME_THRESHOLD: Final[float] = 0.42
LOW_VOLUME_THRESHOLD: Final[float] = MODERATE_VOLUME_THRESHOLD * 0.65

STRONG_MOMENTUM: Final[float] = 0.85
MODERATE_MOMENTUM: Final[float] = 0.58
WEAK_MOMENTUM: Final[float] = 0.35
MIXED_MOMENTUM: Final[float] = 0.22
LOW_MOMENTUM: Final[float] = 0.10


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
    volatility_state: VolatilityState
    dominant_sign: WyckoffSign = WyckoffSign.NONE  # Add dominant Wyckoff sign
    uncertain_phase: bool = True  # Add this field with default value True

@dataclass
class MultiTimeframeContext:
    description: str
    should_notify: bool

@dataclass
class AllTimeframesAnalysis:
    short_term: TimeframeGroupAnalysis
    intermediate: TimeframeGroupAnalysis
    long_term: TimeframeGroupAnalysis
    context: TimeframeGroupAnalysis
    overall_direction: MultiTimeframeDirection
    confidence_level: float
    alignment_score: float
    momentum_intensity: float = 0.0