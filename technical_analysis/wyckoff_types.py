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
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def minutes(self) -> int:
        return self._minutes
        
    def __str__(self) -> str:
        return self._name

@dataclass
class ThresholdConfig:
    volume_threshold: float
    strong_dev_threshold: float
    neutral_zone_threshold: float
    momentum_threshold: float
    effort_threshold: float
    volume_surge_threshold: float
    
    @staticmethod
    def for_timeframe(timeframe: Timeframe) -> 'ThresholdConfig':
        base_multiplier = {
            Timeframe.MINUTES_15: 0.8,  # More sensitive for quick trades
            Timeframe.MINUTES_30: 0.9,  # Between 15m and 1h sensitivity
            Timeframe.HOUR_1: 1.0,      # Base reference
            Timeframe.HOURS_4: 1.2,     # More conservative
            Timeframe.HOURS_8: 1.35,    # Even more conservative
            Timeframe.DAY_1: 1.5        # Most conservative
        }[timeframe]
        
        momentum_multiplier = {
            Timeframe.MINUTES_15: 0.7,  # Faster momentum changes
            Timeframe.MINUTES_30: 0.85, # Between 15m and 1h momentum
            Timeframe.HOUR_1: 1.0,
            Timeframe.HOURS_4: 1.4,
            Timeframe.HOURS_8: 1.6,     # More emphasis on longer trends
            Timeframe.DAY_1: 1.8
        }[timeframe]
        
        return ThresholdConfig(
            volume_threshold=1.5 * base_multiplier,
            strong_dev_threshold=1.8 * base_multiplier,
            neutral_zone_threshold=0.8 * base_multiplier,
            momentum_threshold=0.5 * momentum_multiplier,
            effort_threshold=0.65 * base_multiplier,
            volume_surge_threshold=2.0 * base_multiplier
        )
