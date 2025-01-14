from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict

class WyckoffPhase(Enum):
    """
    Represents the current market phase in Wyckoff analysis.
    Phases are longer-term market conditions that describe the overall structure
    and stage of the market. A phase may last for an extended period and can
    be uncertain (denoted by POSSIBLE_ prefix).
    """
    ACCUMULATION = "acc."
    DISTRIBUTION = "dist."
    MARKUP = "markup"
    MARKDOWN = "markdown"
    RANGING = "rang."
    POSSIBLE_ACCUMULATION = "~ acc."
    POSSIBLE_DISTRIBUTION = "~ dist."
    POSSIBLE_MARKUP = "~ markup"
    POSSIBLE_MARKDOWN = "~ markdown"
    POSSIBLE_RANGING = "~ rang."
    UNKNOWN = "unknown"

class VolumeState(Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"

class MarketPattern(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    UNKNOWN = "unknown"

class EffortResult(Enum):
    STRONG = "strong"
    WEAK = "weak"
    UNKNOWN = "unknown"

class VolatilityState(Enum):
    HIGH = "high"
    NORMAL = "normal"
    UNKNOWN = "unknown"

class WyckoffSign(Enum):
    SELLING_CLIMAX = "SC"
    AUTOMATIC_RALLY = "AR"
    SECONDARY_TEST = "ST"
    LAST_POINT_OF_SUPPORT = "LPS"
    SIGN_OF_STRENGTH = "SOS"
    BUYING_CLIMAX = "BC"
    UPTHRUST = "UT"
    SECONDARY_TEST_RESISTANCE = "STR"
    LAST_POINT_OF_RESISTANCE = "LPSY"
    SIGN_OF_WEAKNESS = "SOW"
    NONE = "NONE"

class CompositeAction(Enum):
    """
    Represents the current action of composite operators (institutions).
    Unlike phases, composite actions represent immediate institutional behavior
    and are more tactical in nature. They indicate what large operators are
    doing right now, providing actionable insights for trading decisions.
    """
    ACCUMULATING = "absorbing supply"
    DISTRIBUTING = "distributing supply"
    MARKING_UP = "actively pushing prices higher"
    MARKING_DOWN = "actively pushing prices lower"
    NEUTRAL = "showing no clear directional activity"
    UNKNOWN = "unknown"

class FundingState(Enum):
    HIGHLY_POSITIVE = "highly positive"
    POSITIVE = "positive"
    SLIGHTLY_POSITIVE = "slightly positive"
    NEUTRAL = "neutral"
    SLIGHTLY_NEGATIVE = "slightly negative"
    NEGATIVE = "negative"
    HIGHLY_NEGATIVE = "highly negative"
    UNKNOWN = "unknown"

@dataclass
class WyckoffState:
    phase: WyckoffPhase
    uncertain_phase: bool
    volume: VolumeState
    pattern: MarketPattern
    volatility: VolatilityState
    is_spring: bool
    is_upthrust: bool
    volume_spread: VolumeState
    effort_vs_result: EffortResult
    composite_action: CompositeAction
    wyckoff_sign: WyckoffSign
    funding_state: FundingState
    description: str

    def to_dict(self):
        return {
            'phase': self.phase.value,
            'uncertain_phase': self.uncertain_phase,
            'volume': self.volume.value,
            'pattern': self.pattern.value,
            'volatility': self.volatility.value,
            'is_spring': self.is_spring,
            'is_upthrust': self.is_upthrust,
            'volume_spread': self.volume_spread.value,
            'effort_vs_result': self.effort_vs_result.value,
            'composite_action': self.composite_action.value,
            'wyckoff_sign': self.wyckoff_sign.value,
            'funding_state': self.funding_state.value,
            'description': self.description
        }

    @staticmethod
    def unknown():
        return WyckoffState(
            phase=WyckoffPhase.UNKNOWN,
            uncertain_phase=True,
            volume=VolumeState.UNKNOWN,
            pattern=MarketPattern.UNKNOWN,
            volatility=VolatilityState.UNKNOWN,
            is_spring=False,
            is_upthrust=False,
            volume_spread=VolumeState.UNKNOWN,
            effort_vs_result=EffortResult.UNKNOWN,
            composite_action=CompositeAction.UNKNOWN,
            wyckoff_sign=WyckoffSign.NONE,
            funding_state=FundingState.UNKNOWN,
            description="Unknown market state"
        )

@dataclass
class TimeframeSettings:
    phase_weight: float
    max_lookback: int
    ema_length: int
    atr_settings: tuple[int, int, int, int, int]  # atr_length, macd_fast, macd_slow, macd_signal, st_length
    supertrend_multiplier: float
    base_multiplier: float
    momentum_multiplier: float
    description: str
    
    @property
    def thresholds(self) -> tuple[float, float, float, float, float, float]:
        """Returns (volume_threshold, strong_dev_threshold, neutral_zone_threshold, 
                   momentum_threshold, effort_threshold, volume_surge_threshold)"""
        return (
            1.5 * self.base_multiplier,      # volume_threshold
            1.8 * self.base_multiplier,      # strong_dev_threshold
            0.8 * self.base_multiplier,      # neutral_zone_threshold
            0.5 * self.momentum_multiplier,  # momentum_threshold
            0.65 * self.base_multiplier,     # effort_threshold
            2.0 * self.base_multiplier       # volume_surge_threshold
        )

class Timeframe(Enum):
    MINUTES_15 = ("15m", 15)
    MINUTES_30 = ("30m", 30)
    HOUR_1 = ("1h", 60)
    HOURS_4 = ("4h", 240)
    HOURS_8 = ("8h", 480)
    DAY_1 = ("1d", 1440)
    
    def __init__(self, name: str, minutes: int):
        self._name = name
        self._minutes = minutes
        self._settings = None  # Will be initialized later
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def minutes(self) -> int:
        return self._minutes

    @property
    def settings(self) -> TimeframeSettings:
        if self._settings is None:
            self._settings = _TIMEFRAME_SETTINGS[self] # type: ignore
        return self._settings # type: ignore
        
    def __str__(self) -> str:
        return self._name

# Define settings after Timeframe class is fully defined
_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.05,
        max_lookback=200,
        ema_length=21,
        atr_settings=(14, 8, 21, 5, 8),
        supertrend_multiplier=2.8,
        base_multiplier=0.8,
        momentum_multiplier=0.7,
        description="15 min trend"
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.10,
        max_lookback=175,
        ema_length=22,
        atr_settings=(18, 10, 24, 7, 9),
        supertrend_multiplier=2.9,
        base_multiplier=0.9,
        momentum_multiplier=0.85,
        description="30 min trend"
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.15,
        max_lookback=150,
        ema_length=24,
        atr_settings=(21, 12, 26, 9, 10),
        supertrend_multiplier=3.0,
        base_multiplier=1.0,
        momentum_multiplier=1.0,
        description="Hourly trend"
    ),
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.175,
        max_lookback=100,
        ema_length=28,
        atr_settings=(28, 12, 32, 9, 12),
        supertrend_multiplier=3.2,
        base_multiplier=1.2,
        momentum_multiplier=1.4,
        description="4h trend"
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.175,
        max_lookback=80,
        ema_length=31,
        atr_settings=(30, 12, 36, 9, 13),
        supertrend_multiplier=3.35,
        base_multiplier=1.35,
        momentum_multiplier=1.6,
        description="8h trend"
    ),
    Timeframe.DAY_1: TimeframeSettings(
        phase_weight=0.175,
        max_lookback=60,
        ema_length=34,
        atr_settings=(34, 12, 40, 9, 14),
        supertrend_multiplier=3.5,
        base_multiplier=1.5,
        momentum_multiplier=1.8,
        description="Daily trend"
    ),
}
