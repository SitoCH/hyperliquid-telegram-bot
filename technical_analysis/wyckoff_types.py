from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

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
    AUTOMATIC_REACTION = "AR"
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
