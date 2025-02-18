from enum import Enum
from dataclasses import dataclass
from typing import Final

from .wyckoff_types import (
    WyckoffState, WyckoffPhase, MarketPattern, 
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState, VolatilityState, MarketLiquidity, LiquidationRisk
)

# Timeframe group weights aligned with existing settings
SHORT_TERM_WEIGHT: Final[float] = 0.35  # 15m + 30m combined weight
INTERMEDIATE_WEIGHT: Final[float] = 0.43  # 1h + 2h combined weight
LONG_TERM_WEIGHT: Final[float] = 0.22  # 4h + 8h + 1d combined weight

# Momentum scoring components weights
DIRECTIONAL_WEIGHT: Final[float] = 0.50  # Weight for directional alignment
VOLUME_WEIGHT: Final[float] = 0.30      # Weight for volume confirmation
PHASE_WEIGHT: Final[float] = 0.20       # Weight for phase confirmation

# Momentum thresholds for description
STRONG_MOMENTUM: Final[float] = 0.85
MODERATE_MOMENTUM: Final[float] = 0.70
WEAK_MOMENTUM: Final[float] = 0.55
MIXED_MOMENTUM: Final[float] = 0.40
LOW_MOMENTUM: Final[float] = 0.25


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
    alignment_score: float  # 0 to 1, indicating how well timeframes align
    confidence_level: float  # 0 to 1, indicating strength of signals
    description: str
    direction: MultiTimeframeDirection
    momentum_intensity: float

@dataclass
class AllTimeframesAnalysis:
    short_term: TimeframeGroupAnalysis
    intermediate: TimeframeGroupAnalysis
    long_term: TimeframeGroupAnalysis
    overall_direction: MultiTimeframeDirection
    confidence_level: float
    alignment_score: float
    momentum_intensity: float = 0.0