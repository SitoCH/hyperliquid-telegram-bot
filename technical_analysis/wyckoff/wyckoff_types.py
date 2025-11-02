from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, TypedDict, List, Set
import pandas as pd  # type: ignore[import]


class WyckoffPhase(Enum):
    """
    Represents the current market phase in Wyckoff analysis.
    Phases are longer-term market conditions that describe the overall structure
    and stage of the market. A phase may last for an extended period.
    Use with the uncertain_phase flag to indicate confidence level.
    """

    ACCUMULATION = "acc."
    DISTRIBUTION = "dist."
    MARKUP = "markup"
    MARKDOWN = "markdown"
    RANGING = "rang."
    UNKNOWN = "unknown"


class VolumeState(Enum):
    VERY_HIGH = "very high"
    HIGH = "high"
    NEUTRAL = "neutral"
    LOW = "low"
    VERY_LOW = "very low"
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
    CONSOLIDATING = "consolidating in range"
    REVERSING = "showing reversal signals"
    NEUTRAL = "without clear action"
    UNKNOWN = "unknown"


# Constant phase categorizations to avoid duplication and ensure consistency
BULLISH_PHASES: Set[WyckoffPhase] = {WyckoffPhase.MARKUP, WyckoffPhase.ACCUMULATION}
BEARISH_PHASES: Set[WyckoffPhase] = {WyckoffPhase.MARKDOWN, WyckoffPhase.DISTRIBUTION}
BULLISH_ACTIONS: Set[CompositeAction] = {
    CompositeAction.MARKING_UP,
    CompositeAction.ACCUMULATING,
}
BEARISH_ACTIONS: Set[CompositeAction] = {
    CompositeAction.MARKING_DOWN,
    CompositeAction.DISTRIBUTING,
}


def is_bullish_phase(phase: WyckoffPhase) -> bool:
    """Check if the given phase is bullish."""
    return phase in BULLISH_PHASES


def is_bearish_phase(phase: WyckoffPhase) -> bool:
    """Check if the given phase is bearish."""
    return phase in BEARISH_PHASES


def is_bullish_action(action: CompositeAction) -> bool:
    """Determine if a Wyckoff action is bullish, with more conservative classification for crypto markets."""
    return action in {
        CompositeAction.ACCUMULATING,
        CompositeAction.MARKING_UP,
    }


def is_bearish_action(action: CompositeAction) -> bool:
    """Determine if a Wyckoff action is bearish, with more conservative classification for crypto markets."""
    return action in {
        CompositeAction.DISTRIBUTING,
        CompositeAction.MARKING_DOWN,
    }


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


@dataclass
class WyckoffState:
    phase: WyckoffPhase
    uncertain_phase: bool
    volume: VolumeState
    pattern: MarketPattern
    volatility: VolatilityState
    is_spring: bool
    is_upthrust: bool
    effort_vs_result: EffortResult
    composite_action: CompositeAction
    wyckoff_sign: WyckoffSign
    funding_state: FundingState
    description: str
    liquidity: MarketLiquidity = MarketLiquidity.UNKNOWN

    def to_dict(self):
        return {
            "phase": self.phase.value,
            "uncertain_phase": self.uncertain_phase,
            "volume": self.volume.value,
            "pattern": self.pattern.value,
            "volatility": self.volatility.value,
            "is_spring": self.is_spring,
            "is_upthrust": self.is_upthrust,
            "effort_vs_result": self.effort_vs_result.value,
            "composite_action": self.composite_action.value,
            "wyckoff_sign": self.wyckoff_sign.value,
            "funding_state": self.funding_state.value,
            "description": self.description,
            "liquidity": self.liquidity.value,
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
            effort_vs_result=EffortResult.UNKNOWN,
            composite_action=CompositeAction.UNKNOWN,
            wyckoff_sign=WyckoffSign.NONE,
            funding_state=FundingState.UNKNOWN,
            description="Unknown market state",
            liquidity=MarketLiquidity.UNKNOWN,
        )


@dataclass
class VolumeMetrics:
    """Container for volume-related metrics"""

    strength: float  # Normalized volume (z-score)
    ratio: float  # Current volume / SMA ratio
    trend: float  # Short-term trend direction
    impulse: float  # Rate of change
    sma: float  # Simple moving average
    consistency: float  # Recent volume consistency
    short_ma: float  # Short-term moving average
    long_ma: float  # Long-term moving average
    trend_strength: float  # Trend strength indicator
    state: VolumeState  # Categorized volume state (VERY_HIGH, HIGH, NEUTRAL, LOW, VERY_LOW)


@dataclass
class TimeframeSettings:
    """Settings for technical analysis parameters per timeframe"""

    phase_weight: float
    description: str
    chart_image_time_delta: pd.Timedelta

    ema_length: int
    atr_settings: tuple[int, int, int, int, int]
    supertrend_multiplier: float
    base_multiplier: float
    momentum_multiplier: float

    volume_ma_window: int

    spring_upthrust_window: int
    support_resistance_lookback: int
    swing_lookback: int = 5
    effort_lookback: int = 5
    min_move_multiplier: float = 1.0

    spring_factor: float = 1.0
    liquidation_factor: float = 1.0
    breakout_factor: float = 1.0
    significant_levels_factor: float = 1.0
    atr_multiplier: float = 0.25
    volume_weighted_efficiency: float = 0.2
    high_threshold: float = 1.0
    low_threshold: float = 1.0

    wyckoff_volatility_factor: float = 1.0
    wyckoff_trend_lookback: int = 5
    wyckoff_lps_volume_threshold: float = 0.3
    wyckoff_lps_price_multiplier: float = 0.7
    wyckoff_sos_multiplier: float = 1.2
    wyckoff_ut_multiplier: float = 0.4
    wyckoff_sc_multiplier: float = 1.2
    wyckoff_ar_multiplier: float = 1.1
    wyckoff_confirmation_threshold: float = 0.35

    @property
    def volume_long_ma_window(self) -> int:
        """Long-term volume MA is double the main window"""
        return self.volume_ma_window * 2

    @property
    def volume_short_ma_window(self) -> int:
        """Derive short MA from main window size for backward compatibility"""
        return max(5, self.volume_ma_window // 3)

    @property
    def volume_trend_window(self) -> int:
        """Derive trend window from long window for backward compatibility"""
        return max(7, self.volume_long_ma_window // 3)

    @property
    def thresholds(self) -> tuple[float, float, float, float]:
        """Returns standardized thresholds using multipliers.
        Order: (volume_threshold, strong_dev_threshold, neutral_zone_threshold, momentum_threshold)
        """
        return (
            1.5 * self.base_multiplier,  # volume_threshold
            1.8 * self.base_multiplier,  # strong_dev_threshold
            0.8 * self.base_multiplier,  # neutral_zone_threshold
            0.5 * self.momentum_multiplier,  # momentum_threshold
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
            self._settings = _TIMEFRAME_SETTINGS[self]  # type: ignore
        return self._settings  # type: ignore

    def __str__(self) -> str:
        return self._name


# Optimized Timeframe Settings for Intraday Crypto Trading with Balanced Multi-Timeframe Analysis
# Weights rebalanced for day trading: higher weight on 15m/30m for better entry/exit timing
# Total weights: 0.25 + 0.32 + 0.30 + 0.10 + 0.03 = 1.00
_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.25,  # 25% - Increased from 0.18 for better scalping signals
        description="15min scalping entries",
        chart_image_time_delta=pd.Timedelta(hours=8),
        ema_length=8,
        atr_settings=(10, 6, 14, 4, 6),
        supertrend_multiplier=1.8,
        base_multiplier=0.85,
        momentum_multiplier=1.6,
        volume_ma_window=16,
        spring_upthrust_window=3,
        support_resistance_lookback=24,
        swing_lookback=3,
        effort_lookback=3,
        min_move_multiplier=0.70,  # Increased from 0.65 to reduce noise on fastest timeframe
        spring_factor=0.75,
        liquidation_factor=0.85,
        breakout_factor=0.85,
        significant_levels_factor=0.80,
        atr_multiplier=0.22,
        volume_weighted_efficiency=0.35,
        high_threshold=0.80,
        low_threshold=0.80,
        wyckoff_volatility_factor=0.70,
        wyckoff_trend_lookback=3,
        wyckoff_lps_volume_threshold=0.18,
        wyckoff_lps_price_multiplier=0.55,
        wyckoff_sos_multiplier=1.05,
        wyckoff_ut_multiplier=0.30,
        wyckoff_sc_multiplier=1.05,
        wyckoff_ar_multiplier=0.95,
        wyckoff_confirmation_threshold=0.25,
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.32,  # 32% - Increased from 0.28 for better swing trade signals
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=16),
        ema_length=10,
        atr_settings=(14, 8, 18, 6, 8),
        supertrend_multiplier=2.0,
        base_multiplier=0.88,  # Increased from 0.85 to match higher weight and importance
        momentum_multiplier=1.5,  # Increased from 1.4 to better capture intraday swings
        volume_ma_window=16,
        spring_upthrust_window=4,
        support_resistance_lookback=36,
        swing_lookback=3,
        effort_lookback=4,
        min_move_multiplier=0.80,
        spring_factor=0.82,
        liquidation_factor=0.90,
        breakout_factor=0.88,  # Increased from 0.85 to better catch swing breakouts
        significant_levels_factor=0.92,  # Increased from 0.90 for better S/R detection
        atr_multiplier=0.24,
        volume_weighted_efficiency=0.30,
        high_threshold=0.85,
        low_threshold=0.80,
        wyckoff_volatility_factor=0.80,
        wyckoff_trend_lookback=3,
        wyckoff_lps_volume_threshold=0.22,  # Increased from 0.24 - now more important timeframe
        wyckoff_lps_price_multiplier=0.62,
        wyckoff_sos_multiplier=1.10,  # Increased from 1.08 for stronger bullish signals
        wyckoff_ut_multiplier=0.36,  # Increased from 0.34 for better upthrust detection
        wyckoff_sc_multiplier=1.10,
        wyckoff_ar_multiplier=1.00,  # Increased from 0.98 for better rally detection
        wyckoff_confirmation_threshold=0.28,
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.30,  # 30% - Reduced from 0.42 to balance with shorter timeframes for day trading
        description="1h trend confirmation",
        chart_image_time_delta=pd.Timedelta(hours=48),
        ema_length=14,
        atr_settings=(18, 12, 24, 8, 10),
        supertrend_multiplier=2.2,
        base_multiplier=0.90,
        momentum_multiplier=1.5,
        volume_ma_window=20,
        spring_upthrust_window=5,
        support_resistance_lookback=48,
        swing_lookback=5,
        effort_lookback=6,
        min_move_multiplier=0.90,
        spring_factor=0.90,
        liquidation_factor=0.88,
        breakout_factor=0.92,
        significant_levels_factor=1.0,
        atr_multiplier=0.26,
        volume_weighted_efficiency=0.24,
        high_threshold=0.90,
        low_threshold=0.88,
        wyckoff_volatility_factor=0.90,
        wyckoff_trend_lookback=5,
        wyckoff_lps_volume_threshold=0.30,
        wyckoff_lps_price_multiplier=0.68,
        wyckoff_sos_multiplier=1.18,
        wyckoff_ut_multiplier=0.40,
        wyckoff_sc_multiplier=1.20,
        wyckoff_ar_multiplier=1.08,
        wyckoff_confirmation_threshold=0.32,
    ),
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.10,  # 10% - Unchanged, provides structural context
        description="4h structural context",
        chart_image_time_delta=pd.Timedelta(
            days=5
        ),
        ema_length=21,
        atr_settings=(24, 16, 30, 12, 16),
        supertrend_multiplier=2.4,
        base_multiplier=1.0,
        momentum_multiplier=1.2,
        volume_ma_window=24,
        spring_upthrust_window=8,
        support_resistance_lookback=84,
        swing_lookback=8,
        effort_lookback=10,
        min_move_multiplier=1.2,
        spring_factor=1.0,
        liquidation_factor=1.0,
        breakout_factor=1.0,
        significant_levels_factor=1.3,
        atr_multiplier=0.28,
        volume_weighted_efficiency=0.18,
        high_threshold=1.0,
        low_threshold=0.9,
        wyckoff_volatility_factor=1.0,
        wyckoff_trend_lookback=7,
        wyckoff_lps_volume_threshold=0.32,
        wyckoff_lps_price_multiplier=0.75,
        wyckoff_sos_multiplier=1.25,
        wyckoff_ut_multiplier=0.45,
        wyckoff_sc_multiplier=1.25,
        wyckoff_ar_multiplier=1.15,
        wyckoff_confirmation_threshold=0.35,
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.03,  # 3% - Slightly increased from 0.02 for minimal regime awareness
        description="8h market regime context",
        chart_image_time_delta=pd.Timedelta(days=10),
        ema_length=30,
        atr_settings=(32, 20, 40, 14, 18),
        supertrend_multiplier=2.6,
        base_multiplier=1.1,
        momentum_multiplier=1.0,
        volume_ma_window=30,
        spring_upthrust_window=10,
        support_resistance_lookback=120,
        swing_lookback=10,
        effort_lookback=12,
        min_move_multiplier=1.5,
        spring_factor=1.2,
        liquidation_factor=1.1,
        breakout_factor=1.2,
        significant_levels_factor=1.5,
        atr_multiplier=0.30,
        volume_weighted_efficiency=0.15,
        high_threshold=1.2,
        low_threshold=0.8,
        wyckoff_volatility_factor=1.2,
        wyckoff_trend_lookback=8,
        wyckoff_lps_volume_threshold=0.35,
        wyckoff_lps_price_multiplier=0.80,
        wyckoff_sos_multiplier=1.30,
        wyckoff_ut_multiplier=0.50,
        wyckoff_sc_multiplier=1.30,
        wyckoff_ar_multiplier=1.20,
        wyckoff_confirmation_threshold=0.40,
    ),
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}


class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
