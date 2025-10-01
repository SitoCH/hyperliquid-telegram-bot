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

# Optimized Timeframe Settings for Intraday Crypto Trading with Hourly Analysis
_TIMEFRAME_SETTINGS = {
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.10,  # Reduced for noise filtering
        description="15min scalping entries",
        chart_image_time_delta=pd.Timedelta(hours=8),  # Shorter for faster reaction
        
        ema_length=8,  # Faster response for scalping
        atr_settings=(10, 6, 14, 4, 6),  # More responsive ATR
        supertrend_multiplier=1.8,  # Tighter for scalping
        base_multiplier=0.7,  # More sensitive for quick moves
        momentum_multiplier=1.6,  # Higher for crypto volatility
        
        volume_ma_window=12,  # Shorter for faster volume detection
        
        spring_upthrust_window=3,  # Tighter window for scalping
        support_resistance_lookback=24,  # Shorter lookback
        swing_lookback=2,  # Very short for scalping
        effort_lookback=2,
        min_move_multiplier=0.6,  # Lower threshold for micro moves
        
        spring_factor=0.70,  # More sensitive
        liquidation_factor=0.75,
        breakout_factor=0.75,
        significant_levels_factor=0.75,
        atr_multiplier=0.20,
        volume_weighted_efficiency=0.40,  # Higher for volume importance
        high_threshold=0.75,
        low_threshold=0.75,

        wyckoff_volatility_factor=0.60,  # Lower for noise filtering
        wyckoff_trend_lookback=2,
        wyckoff_lps_volume_threshold=0.15,  # More sensitive
        wyckoff_lps_price_multiplier=0.50,
        wyckoff_sos_multiplier=0.95,
        wyckoff_ut_multiplier=0.25,
        wyckoff_sc_multiplier=0.95,
        wyckoff_ar_multiplier=0.85,
        wyckoff_confirmation_threshold=0.20  # Lower for faster signals
    ),    
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.20,  # Balanced importance
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=16),  # Better context
        
        ema_length=10,  # Balanced responsiveness
        atr_settings=(14, 8, 18, 6, 8),  # Optimized for 30min
        supertrend_multiplier=2.0,
        base_multiplier=0.80,  # Balanced sensitivity
        momentum_multiplier=1.4,  # Crypto-appropriate
        
        volume_ma_window=14,  # Balanced volume detection
        
        spring_upthrust_window=4,  # Adequate window
        support_resistance_lookback=36,  # Good lookback
        swing_lookback=3,  # Balanced for swing detection
        effort_lookback=4,
        min_move_multiplier=0.75,  # Reasonable threshold
        
        spring_factor=0.80,  # Balanced sensitivity
        liquidation_factor=0.85,
        breakout_factor=0.80,
        significant_levels_factor=0.85,
        atr_multiplier=0.22,
        volume_weighted_efficiency=0.32,  # Balanced volume importance
        high_threshold=0.82,
        low_threshold=0.78,

        wyckoff_volatility_factor=0.75,  # Balanced for noise vs signal
        wyckoff_trend_lookback=3,
        wyckoff_lps_volume_threshold=0.22,
        wyckoff_lps_price_multiplier=0.60,
        wyckoff_sos_multiplier=1.05,
        wyckoff_ut_multiplier=0.32,
        wyckoff_sc_multiplier=1.08,
        wyckoff_ar_multiplier=0.95,
        wyckoff_confirmation_threshold=0.25  # Balanced confirmation
    ),
      Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.40,  # Primary timeframe for trend confirmation
        description="1h primary trend analysis",
        chart_image_time_delta=pd.Timedelta(hours=48),  # Good context window
        
        ema_length=14,  # Standard for hourly
        atr_settings=(18, 12, 24, 8, 10),
        supertrend_multiplier=2.2,  # Slightly higher for crypto
        base_multiplier=0.85,
        momentum_multiplier=1.5,
        
        volume_ma_window=18,  # Adequate for hourly volume patterns
        
        spring_upthrust_window=5,
        support_resistance_lookback=48,  # 2 days lookback
        swing_lookback=5,
        effort_lookback=6,
        min_move_multiplier=0.85,  # Higher threshold for meaningful moves
        
        spring_factor=0.85,
        liquidation_factor=0.88,
        breakout_factor=0.88,
        significant_levels_factor=0.9,
        atr_multiplier=0.25,
        volume_weighted_efficiency=0.25,  # Balanced volume weighting
        high_threshold=0.88,
        low_threshold=0.85,

        wyckoff_volatility_factor=0.85,  # Standard for primary timeframe
        wyckoff_trend_lookback=5,
        wyckoff_lps_volume_threshold=0.28,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.15,
        wyckoff_ut_multiplier=0.37,
        wyckoff_sc_multiplier=1.18,
        wyckoff_ar_multiplier=1.05,
        wyckoff_confirmation_threshold=0.30
    ),
      Timeframe.HOURS_2: TimeframeSettings(
        phase_weight=0.20,  # Increased for better intermediate trend analysis
        description="2h intermediate trend context",
        chart_image_time_delta=pd.Timedelta(hours=96),  # Extended for better context
        
        ema_length=16,
        atr_settings=(20, 14, 26, 10, 12),  # Better balance for 2h
        supertrend_multiplier=2.3,  # Higher for crypto volatility
        base_multiplier=0.90,  # Less sensitive but still responsive
        momentum_multiplier=1.3,  # Moderate for intermediate term
        
        volume_ma_window=20,  # Longer for better volume smoothing
        
        spring_upthrust_window=6,  # Extended for better pattern recognition
        support_resistance_lookback=60,  # 5 days for 2h timeframe
        swing_lookback=6,
        effort_lookback=8,  # Extended for better effort analysis
        min_move_multiplier=1.0,  # Higher threshold for significant moves
        
        spring_factor=0.90,  # Less sensitive for stronger signals
        liquidation_factor=0.92,
        breakout_factor=0.90,
        significant_levels_factor=1.0,
        atr_multiplier=0.26,
        volume_weighted_efficiency=0.22,  # Lower for intermediate term
        high_threshold=0.90,
        low_threshold=0.85,

        wyckoff_volatility_factor=0.90,  # Balanced for intermediate term
        wyckoff_trend_lookback=6,  # Extended lookback
        wyckoff_lps_volume_threshold=0.30,
        wyckoff_lps_price_multiplier=0.70,
        wyckoff_sos_multiplier=1.20,
        wyckoff_ut_multiplier=0.40,
        wyckoff_sc_multiplier=1.20,
        wyckoff_ar_multiplier=1.10,
        wyckoff_confirmation_threshold=0.32
    ),    
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.08,  # Reduced for context only
        description="4h structural context",
        chart_image_time_delta=pd.Timedelta(days=5),  # Extended for better structure view
        
        ema_length=21,  # Standard daily-like period
        atr_settings=(24, 16, 30, 12, 16),  # Longer periods for structure
        supertrend_multiplier=2.4,  # Higher for less noise
        base_multiplier=1.0,  # Standard sensitivity
        momentum_multiplier=1.2,  # Lower for structure
        
        volume_ma_window=24,  # Longer for structural volume
        
        spring_upthrust_window=8,  # Extended for structural patterns
        support_resistance_lookback=84,  # Full week of 4h candles
        swing_lookback=8,  # Extended for structural swings
        effort_lookback=10,
        min_move_multiplier=1.2,  # Higher for significant structural moves
        
        spring_factor=1.0,  # Standard for structural signals
        liquidation_factor=1.0,
        breakout_factor=1.0,
        significant_levels_factor=1.3,  # Higher for key structural levels
        atr_multiplier=0.28,
        volume_weighted_efficiency=0.18,  # Lower for structural analysis
        high_threshold=1.0,
        low_threshold=0.9,

        wyckoff_volatility_factor=1.0,  # Standard for structural
        wyckoff_trend_lookback=7,  # Extended for trend context
        wyckoff_lps_volume_threshold=0.32,
        wyckoff_lps_price_multiplier=0.75,
        wyckoff_sos_multiplier=1.25,
        wyckoff_ut_multiplier=0.45,
        wyckoff_sc_multiplier=1.25,
        wyckoff_ar_multiplier=1.15,
        wyckoff_confirmation_threshold=0.35  # Higher for structural confirmation
    ),    
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.02,  # Minimal weight for long-term context only
        description="8h market regime context",
        chart_image_time_delta=pd.Timedelta(days=10),  # Extended for regime analysis
        
        ema_length=30,  # Longer for regime identification
        atr_settings=(32, 20, 40, 14, 18),  # Extended periods for regime context
        supertrend_multiplier=2.6,  # Higher to filter noise in regime analysis
        base_multiplier=1.1,  # Less sensitive for regime changes
        momentum_multiplier=1.0,  # Lower for long-term regime
        
        volume_ma_window=30,  # Extended for regime volume analysis
        
        spring_upthrust_window=10,  # Extended for regime-level patterns
        support_resistance_lookback=120,  # Extended for major structural levels
        swing_lookback=10,  # Extended for regime swings
        effort_lookback=12,
        min_move_multiplier=1.5,  # Much higher for regime-significant moves
        
        spring_factor=1.2,  # Higher for regime-level signals
        liquidation_factor=1.1,
        breakout_factor=1.2,
        significant_levels_factor=1.5,  # Much higher for major levels
        atr_multiplier=0.30,
        volume_weighted_efficiency=0.15,  # Lower for regime analysis
        high_threshold=1.2,
        low_threshold=0.8,

        wyckoff_volatility_factor=1.2,  # Higher for regime filtering
        wyckoff_trend_lookback=8,  # Extended for regime context
        wyckoff_lps_volume_threshold=0.35,  # Higher for regime significance
        wyckoff_lps_price_multiplier=0.80,
        wyckoff_sos_multiplier=1.30,  # Higher for regime-level strength
        wyckoff_ut_multiplier=0.50,
        wyckoff_sc_multiplier=1.30,
        wyckoff_ar_multiplier=1.20,
        wyckoff_confirmation_threshold=0.40  # Much higher for regime confirmation
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
