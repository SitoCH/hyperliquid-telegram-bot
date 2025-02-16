from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, TypedDict, List
import pandas as pd  # type: ignore[import]

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
    CONSOLIDATING = "consolidating in range"  # New
    REVERSING = "showing reversal signals"    # New
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

class MarketLiquidity(Enum):
    HIGH = "high liquidity"
    MODERATE = "moderate liquidity"
    LOW = "low liquidity"
    UNKNOWN = "unknown liquidity"

class LiquidationRisk(Enum):
    HIGH = "high liquidation risk"
    MODERATE = "moderate liquidation risk"
    LOW = "low liquidation risk"
    UNKNOWN = "unknown risk"

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
    liquidity: MarketLiquidity = MarketLiquidity.UNKNOWN
    liquidation_risk: LiquidationRisk = LiquidationRisk.UNKNOWN

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
            'description': self.description,
            'liquidity': self.liquidity.value,
            'liquidation_risk': self.liquidation_risk.value,
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
            description="Unknown market state",
            liquidity=MarketLiquidity.UNKNOWN,
            liquidation_risk=LiquidationRisk.UNKNOWN,
        )

@dataclass
class TimeframeSettings:
    """Settings for technical analysis parameters per timeframe"""
    phase_weight: float              # Used in get_phase_weight()
    ema_length: int                  # Used in apply_indicators() for EMA calculation
    atr_settings: tuple[int, int, int, int, int]  # Used in get_indicator_settings()
    supertrend_multiplier: float     # Used in apply_indicators() for Supertrend
    base_multiplier: float          # Used in thresholds property
    momentum_multiplier: float      # Used in thresholds property
    description: str                # Used for logging/display purposes
    # Add new settings
    volume_ma_window: int                 # From wyckoff.py calculate_volume_metrics()
    volume_short_ma_window: int           # From wyckoff.py calculate_volume_metrics()
    volume_long_ma_window: int            # From wyckoff.py calculate_volume_metrics()
    spring_upthrust_window: int           # From wyckoff.py detect_spring_upthrust()
    volume_trend_window: int              # From wyckoff.py calculate_volume_metrics()
    price_strength_ma_window: int         # From wyckoff.py determine_phase_by_price_strength()
    price_change_window: int              # From wyckoff.py determine_phase_by_price_strength()
    support_resistance_lookback: int      # From significant_levels.py find_significant_levels()
    chart_image_time_delta: pd.Timedelta  # From wyckoff_chart.py save_to_buffer()

    @property
    def thresholds(self) -> tuple[float, float, float, float, float, float]:
        """Returns standardized thresholds using multipliers"""
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
    HOURS_2 = ("2h", 120)  # New timeframe
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

# Rebalanced weights for crypto focus
_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.12,
        ema_length=8,
        atr_settings=(8, 5, 13, 3, 5),
        supertrend_multiplier=2.0,
        base_multiplier=0.95,  # Decreased from 1.05 for higher sensitivity
        momentum_multiplier=1.3,  # Increased from 1.2 for faster reaction
        description="15 min trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.16,
        ema_length=13,
        atr_settings=(10, 6, 18, 4, 6),
        supertrend_multiplier=2.3,
        base_multiplier=1.05,  # Decreased from 1.15
        momentum_multiplier=1.35,  # Increased from 1.25
        description="30 min trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.24,
        ema_length=21,
        atr_settings=(14, 9, 21, 7, 8),
        supertrend_multiplier=2.6,
        base_multiplier=1.15,  # Decreased from 1.25
        momentum_multiplier=1.45,  # Increased from 1.35
        description="Hourly trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.HOURS_2: TimeframeSettings(
        phase_weight=0.16,
        ema_length=28,
        atr_settings=(20, 10, 25, 8, 10),
        supertrend_multiplier=2.8,
        base_multiplier=1.25,  # Decreased from 1.35
        momentum_multiplier=1.55,  # Increased from 1.45
        description="2h trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.18,
        ema_length=34,
        atr_settings=(34, 12, 34, 9, 14),
        supertrend_multiplier=3.0,
        base_multiplier=1.35,  # Decreased from 1.45
        momentum_multiplier=1.75,  # Increased from 1.65
        description="4h trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.08,
        ema_length=41,
        atr_settings=(38, 12, 40, 9, 16),
        supertrend_multiplier=3.2,
        base_multiplier=1.45,  # Decreased from 1.55
        momentum_multiplier=1.85,  # Increased from 1.75
        description="8h trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    Timeframe.DAY_1: TimeframeSettings(
        phase_weight=0.06,
        ema_length=55,
        atr_settings=(41, 12, 48, 9, 21),
        supertrend_multiplier=3.5,
        base_multiplier=1.55,  # Decreased from 1.65
        momentum_multiplier=2.05,  # Increased from 1.95
        description="Daily trend",
        # Add new settings
        volume_ma_window=20,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=8,
        price_change_window=3,
        support_resistance_lookback=40,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
