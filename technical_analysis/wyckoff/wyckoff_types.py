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

    # Indicator periods for intraday trading
    bb_period: int = 20  # Bollinger Bands period
    rsi_period: int = 14  # RSI period
    fib_lookback: int = 32  # Fibonacci lookback period

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


_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.40,  # Increased from 0.35 to emphasize execution timeframe
        description="15min entries",
        chart_image_time_delta=pd.Timedelta(hours=8),
        ema_length=10,
        atr_settings=(10, 8, 17, 6, 7),  # Slower MACD (8,17,6) and ST length (7) to reduce noise
        supertrend_multiplier=2.0,  # Increased from 1.8 to reduce whipsaws
        base_multiplier=0.95,
        momentum_multiplier=1.15,
        volume_ma_window=14,
        spring_upthrust_window=3,
        support_resistance_lookback=48,  # Increased from 18 to 48 (12h) for better levels
        swing_lookback=3,
        effort_lookback=3,
        min_move_multiplier=0.90,
        bb_period=20,
        rsi_period=14,
        fib_lookback=32,
        spring_factor=1.05,
        liquidation_factor=1.05,
        breakout_factor=1.10,
        significant_levels_factor=1.05,
        atr_multiplier=0.15,
        volume_weighted_efficiency=0.25,
        high_threshold=0.95,
        low_threshold=0.85,
        wyckoff_volatility_factor=0.8,
        wyckoff_trend_lookback=4,
        wyckoff_lps_volume_threshold=0.20,
        wyckoff_lps_price_multiplier=0.55,
        wyckoff_sos_multiplier=1.05,
        wyckoff_ut_multiplier=0.30,
        wyckoff_sc_multiplier=1.05,
        wyckoff_ar_multiplier=0.95,
        wyckoff_confirmation_threshold=0.32,
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.25,  # Adjusted from 0.27
        description="30min swings",
        chart_image_time_delta=pd.Timedelta(hours=16),
        ema_length=16,
        atr_settings=(14, 10, 20, 7, 9),  # Slightly slower MACD/ST
        supertrend_multiplier=2.1,  # Increased from 1.9
        base_multiplier=1.00,
        momentum_multiplier=1.12,
        volume_ma_window=16,
        spring_upthrust_window=4,
        support_resistance_lookback=48,  # Increased from 30 to 48 (24h)
        swing_lookback=4,
        effort_lookback=4,
        min_move_multiplier=1.00,
        bb_period=24,
        rsi_period=14,
        fib_lookback=40,
        spring_factor=1.05,
        liquidation_factor=1.05,
        breakout_factor=1.10,
        significant_levels_factor=1.15,
        atr_multiplier=0.18,
        volume_weighted_efficiency=0.24,
        high_threshold=0.95,
        low_threshold=0.85,
        wyckoff_volatility_factor=0.85,
        wyckoff_trend_lookback=4,
        wyckoff_lps_volume_threshold=0.22,
        wyckoff_lps_price_multiplier=0.60,
        wyckoff_sos_multiplier=1.08,
        wyckoff_ut_multiplier=0.32,
        wyckoff_sc_multiplier=1.08,
        wyckoff_ar_multiplier=0.98,
        wyckoff_confirmation_threshold=0.33,
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.20,  # Adjusted from 0.23
        description="1h confirmation",
        chart_image_time_delta=pd.Timedelta(hours=48),
        ema_length=20,
        atr_settings=(18, 12, 26, 9, 10),  # Standard MACD (12,26,9) for 1h
        supertrend_multiplier=2.2,  # Increased from 2.0
        base_multiplier=0.95,
        momentum_multiplier=1.30,
        volume_ma_window=20,
        spring_upthrust_window=5,
        support_resistance_lookback=72,  # Increased from 60 to 72 (3 days)
        swing_lookback=4,
        effort_lookback=5,
        min_move_multiplier=1.00,
        bb_period=26,
        rsi_period=14,
        fib_lookback=48,
        spring_factor=0.95,
        liquidation_factor=0.95,
        breakout_factor=1.0,
        significant_levels_factor=1.25,
        atr_multiplier=0.25,
        volume_weighted_efficiency=0.26,
        high_threshold=0.9,
        low_threshold=0.8,
        wyckoff_volatility_factor=0.9,
        wyckoff_trend_lookback=5,
        wyckoff_lps_volume_threshold=0.26,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.12,
        wyckoff_ut_multiplier=0.36,
        wyckoff_sc_multiplier=1.14,
        wyckoff_ar_multiplier=1.02,
        wyckoff_confirmation_threshold=0.33,
    ),
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.10,  # Increased from 0.05 to give more weight to longer-term context
        description="4h context",
        chart_image_time_delta=pd.Timedelta(days=5),
        ema_length=21,
        atr_settings=(24, 16, 30, 12, 16),
        supertrend_multiplier=2.4,
        base_multiplier=1.00,
        momentum_multiplier=1.15,
        volume_ma_window=24,
        spring_upthrust_window=8,
        support_resistance_lookback=84,
        swing_lookback=8,
        effort_lookback=10,
        min_move_multiplier=1.20,
        bb_period=20,
        rsi_period=14,
        fib_lookback=30,
        spring_factor=1.05,
        liquidation_factor=1.0,
        breakout_factor=1.05,
        significant_levels_factor=1.35,
        atr_multiplier=0.28,
        volume_weighted_efficiency=0.20,
        high_threshold=1.0,
        low_threshold=0.85,
        wyckoff_volatility_factor=1.0,
        wyckoff_trend_lookback=7,
        wyckoff_lps_volume_threshold=0.30,
        wyckoff_lps_price_multiplier=0.72,
        wyckoff_sos_multiplier=1.22,
        wyckoff_ut_multiplier=0.42,
        wyckoff_sc_multiplier=1.22,
        wyckoff_ar_multiplier=1.10,
        wyckoff_confirmation_threshold=0.36,
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.05,  # Increased from 0.02 to give more weight to regime context
        description="8h regime context",
        chart_image_time_delta=pd.Timedelta(days=10),
        ema_length=30,
        atr_settings=(32, 20, 40, 14, 18),
        supertrend_multiplier=2.7,
        base_multiplier=1.10,
        momentum_multiplier=1.05,
        volume_ma_window=30,
        spring_upthrust_window=10,
        support_resistance_lookback=120,
        swing_lookback=10,
        effort_lookback=12,
        min_move_multiplier=1.50,
        bb_period=20,
        rsi_period=14,
        fib_lookback=24,
        spring_factor=1.15,
        liquidation_factor=1.1,
        breakout_factor=1.15,
        significant_levels_factor=1.5,
        atr_multiplier=0.30,
        volume_weighted_efficiency=0.18,
        high_threshold=1.1,
        low_threshold=0.9,
        wyckoff_volatility_factor=1.1,
        wyckoff_trend_lookback=8,
        wyckoff_lps_volume_threshold=0.34,
        wyckoff_lps_price_multiplier=0.78,
        wyckoff_sos_multiplier=1.28,
        wyckoff_ut_multiplier=0.48,
        wyckoff_sc_multiplier=1.28,
        wyckoff_ar_multiplier=1.16,
        wyckoff_confirmation_threshold=0.40,
    ),
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}


class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
