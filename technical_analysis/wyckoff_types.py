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
BULLISH_ACTIONS: Set[CompositeAction] = {CompositeAction.MARKING_UP, CompositeAction.ACCUMULATING}
BEARISH_ACTIONS: Set[CompositeAction] = {CompositeAction.MARKING_DOWN, CompositeAction.DISTRIBUTING}


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
            'phase': self.phase.value,
            'uncertain_phase': self.uncertain_phase,
            'volume': self.volume.value,
            'pattern': self.pattern.value,
            'volatility': self.volatility.value,
            'is_spring': self.is_spring,
            'is_upthrust': self.is_upthrust,
            'effort_vs_result': self.effort_vs_result.value,
            'composite_action': self.composite_action.value,
            'wyckoff_sign': self.wyckoff_sign.value,
            'funding_state': self.funding_state.value,
            'description': self.description,
            'liquidity': self.liquidity.value
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
            liquidity=MarketLiquidity.UNKNOWN
        )

@dataclass
class VolumeMetrics:
    """Container for volume-related metrics"""
    strength: float      # Normalized volume (z-score)
    ratio: float        # Current volume / SMA ratio
    trend: float        # Short-term trend direction
    impulse: float      # Rate of change
    sma: float         # Simple moving average
    consistency: float  # Recent volume consistency
    short_ma: float    # Short-term moving average
    long_ma: float     # Long-term moving average
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
    def thresholds(self) -> tuple[float, float, float, float, float, float]:
        """Returns standardized thresholds using multipliers"""
        return (
            1.5 * self.base_multiplier,
            1.8 * self.base_multiplier,
            0.8 * self.base_multiplier,
            0.5 * self.momentum_multiplier,
            0.65 * self.base_multiplier,
            2.0 * self.base_multiplier
        )

class Timeframe(Enum):
    MINUTES_15 = ("15m", 15)
    MINUTES_30 = ("30m", 30)
    HOUR_1 = ("1h", 60)
    HOURS_2 = ("2h", 120)
    HOURS_4 = ("4h", 240)
    HOURS_8 = ("8h", 480)
    
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

# Optimized Timeframe Settings for Intraday Crypto Trading with Hourly Analysis
_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.18,
        description="15min tactical entries",
        chart_image_time_delta=pd.Timedelta(hours=12),
        
        ema_length=9,
        atr_settings=(12, 7, 16, 5, 7),
        supertrend_multiplier=1.7,
        base_multiplier=0.75,
        momentum_multiplier=1.4,
        
        volume_ma_window=12,
        
        spring_upthrust_window=3,
        support_resistance_lookback=30,
        swing_lookback=3,
        effort_lookback=3,
        min_move_multiplier=0.65,
        
        spring_factor=0.80,
        liquidation_factor=0.82,
        breakout_factor=0.82,
        significant_levels_factor=0.82,
        atr_multiplier=0.22,
        volume_weighted_efficiency=0.35,
        high_threshold=0.80,
        low_threshold=0.80,

        wyckoff_volatility_factor=0.68,
        wyckoff_trend_lookback=3,
        wyckoff_lps_volume_threshold=0.20,
        wyckoff_lps_price_multiplier=0.58,
        wyckoff_sos_multiplier=1.02,
        wyckoff_ut_multiplier=0.30,
        wyckoff_sc_multiplier=1.02,
        wyckoff_ar_multiplier=0.92,
        wyckoff_confirmation_threshold=0.25
    ),
    
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.22,
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=24),
        
        ema_length=12,
        atr_settings=(16, 10, 20, 7, 9),
        supertrend_multiplier=2.0,
        base_multiplier=0.8,
        momentum_multiplier=1.4,
        
        volume_ma_window=16,
        
        spring_upthrust_window=5,
        support_resistance_lookback=42,
        swing_lookback=4,
        effort_lookback=5,
        min_move_multiplier=0.75,
        
        spring_factor=0.85,
        liquidation_factor=0.9,
        breakout_factor=0.85,
        significant_levels_factor=0.88,
        atr_multiplier=0.24,
        volume_weighted_efficiency=0.35,
        high_threshold=0.85,
        low_threshold=0.85,

        wyckoff_volatility_factor=0.82,
        wyckoff_trend_lookback=4,
        wyckoff_lps_volume_threshold=0.25,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.12,
        wyckoff_ut_multiplier=0.35,
        wyckoff_sc_multiplier=1.1,
        wyckoff_ar_multiplier=1.0,
        wyckoff_confirmation_threshold=0.28
    ),
    
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.35,
        description="1h primary daily trend",
        chart_image_time_delta=pd.Timedelta(hours=48),
        
        ema_length=14,
        atr_settings=(18, 12, 24, 8, 10),
        supertrend_multiplier=1.95,
        base_multiplier=0.85,
        momentum_multiplier=1.5,
        
        volume_ma_window=18,
        
        spring_upthrust_window=5,
        support_resistance_lookback=48,
        swing_lookback=5,
        effort_lookback=6,
        min_move_multiplier=0.82,
        
        spring_factor=0.87,
        liquidation_factor=0.88,
        breakout_factor=0.88,
        significant_levels_factor=0.9,
        atr_multiplier=0.25,
        volume_weighted_efficiency=0.38,
        high_threshold=0.88,
        low_threshold=0.88,

        wyckoff_volatility_factor=0.88,
        wyckoff_trend_lookback=5,
        wyckoff_lps_volume_threshold=0.28,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.1,
        wyckoff_ut_multiplier=0.37,
        wyckoff_sc_multiplier=1.15,
        wyckoff_ar_multiplier=1.05,
        wyckoff_confirmation_threshold=0.28
    ),
    
    Timeframe.HOURS_2: TimeframeSettings(
        phase_weight=0.10,
        description="2h daily bias",
        chart_image_time_delta=pd.Timedelta(hours=72),
        
        ema_length=16,
        atr_settings=(20, 12, 24, 8, 10),
        supertrend_multiplier=1.9,
        base_multiplier=0.92,
        momentum_multiplier=1.4,
        
        volume_ma_window=16,
        
        spring_upthrust_window=5,
        support_resistance_lookback=48,
        swing_lookback=6,
        effort_lookback=6,
        min_move_multiplier=0.9,
        
        spring_factor=0.9,
        liquidation_factor=0.9,
        breakout_factor=0.92,
        significant_levels_factor=0.95,
        atr_multiplier=0.26,
        volume_weighted_efficiency=0.28,
        high_threshold=0.92,
        low_threshold=0.88,

        wyckoff_volatility_factor=0.92,
        wyckoff_trend_lookback=4,
        wyckoff_lps_volume_threshold=0.28,
        wyckoff_lps_price_multiplier=0.68,
        wyckoff_sos_multiplier=1.15,
        wyckoff_ut_multiplier=0.4,
        wyckoff_sc_multiplier=1.15,
        wyckoff_ar_multiplier=1.05,
        wyckoff_confirmation_threshold=0.30
    ),
    
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.10,
        description="4h daily context",
        chart_image_time_delta=pd.Timedelta(days=4),
        
        ema_length=20,
        atr_settings=(22, 14, 28, 10, 14),
        supertrend_multiplier=2.1,
        base_multiplier=0.95,
        momentum_multiplier=1.5,
        
        volume_ma_window=20,
        
        spring_upthrust_window=6,
        support_resistance_lookback=60,
        swing_lookback=7,
        effort_lookback=7,
        min_move_multiplier=1.1,
        
        spring_factor=0.95,
        liquidation_factor=0.95,
        breakout_factor=0.95,
        significant_levels_factor=1.2,
        atr_multiplier=0.28,
        volume_weighted_efficiency=0.22,
        high_threshold=1.0,
        low_threshold=0.9,

        wyckoff_volatility_factor=0.95,
        wyckoff_trend_lookback=5,
        wyckoff_lps_volume_threshold=0.30,
        wyckoff_lps_price_multiplier=0.7,
        wyckoff_sos_multiplier=1.15,
        wyckoff_ut_multiplier=0.42,
        wyckoff_sc_multiplier=1.2,
        wyckoff_ar_multiplier=1.1,
        wyckoff_confirmation_threshold=0.32
    ),
    
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.05,
        description="8h market regime",
        chart_image_time_delta=pd.Timedelta(days=6),
        
        ema_length=28,
        atr_settings=(30, 18, 36, 12, 16),
        supertrend_multiplier=2.3,
        base_multiplier=1.05,
        momentum_multiplier=1.7,
        
        volume_ma_window=24,
        
        spring_upthrust_window=7,
        support_resistance_lookback=75,
        swing_lookback=8,
        effort_lookback=10,
        min_move_multiplier=1.5,
        
        spring_factor=1.1,
        liquidation_factor=1.0,
        breakout_factor=1.1,
        significant_levels_factor=1.5,
        atr_multiplier=0.3,
        volume_weighted_efficiency=0.15,
        high_threshold=1.1,
        low_threshold=0.85,

        wyckoff_volatility_factor=1.1,
        wyckoff_trend_lookback=6,
        wyckoff_lps_volume_threshold=0.32,
        wyckoff_lps_price_multiplier=0.75,
        wyckoff_sos_multiplier=1.25,
        wyckoff_ut_multiplier=0.45,
        wyckoff_sc_multiplier=1.25,
        wyckoff_ar_multiplier=1.15,
        wyckoff_confirmation_threshold=0.35
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
