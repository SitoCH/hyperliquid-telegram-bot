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
    
    # --- Core Analysis Settings ---
    phase_weight: float              # Used in get_phase_weight()
    description: str                # Used for logging/display purposes
    chart_image_time_delta: pd.Timedelta  # For chart rendering
    
    # --- Indicator Settings ---
    ema_length: int                  # Used in apply_indicators() for EMA calculation
    atr_settings: tuple[int, int, int, int, int]  # Used in get_indicator_settings()
    supertrend_multiplier: float     # Used in apply_indicators() for Supertrend
    base_multiplier: float          # Used in thresholds property
    momentum_multiplier: float      # Used in thresholds property
    
    # --- Volume Analysis Settings ---
    volume_ma_window: int           # Primary volume moving average window
    
    # --- Support/Resistance & Pattern Detection ---
    spring_upthrust_window: int     # For reversal pattern detection
    support_resistance_lookback: int # For S/R level identification
    swing_lookback: int = 5         # Number of candles to analyze for recent swings
    effort_lookback: int = 5        # Periods to look back for effort-result analysis
    min_move_multiplier: float = 1.0  # Minimum price move multiplier for effort-result analysis
    
    # --- Adaptive Threshold Factors ---
    spring_factor: float = 1.0      # Factor for spring/upthrust detection
    liquidation_factor: float = 1.0 # Factor for liquidation cascade detection
    breakout_factor: float = 1.0    # Factor for breakout detection
    significant_levels_factor: float = 1.0  # Factor for price level detection
    atr_multiplier: float = 0.25    # ATR multiplier for support/resistance clustering
    volume_weighted_efficiency: float = 0.2  # Efficiency multiplier for volume impact on price movement
    high_threshold: float = 1.0     # Upper threshold for efficiency calculation
    low_threshold: float = 1.0      # Lower threshold for efficiency calculation
    
    # --- Wyckoff Sign Detection Parameters ---
    wyckoff_volatility_factor: float = 1.0  # Volatility factor for sign detection
    wyckoff_trend_lookback: int = 5         # Lookback period for trend reversal detection
    wyckoff_st_tolerance_low: float = 0.996  # Lower tolerance for secondary test detection
    wyckoff_st_tolerance_high: float = 1.012 # Upper tolerance for secondary test detection
    wyckoff_lps_volume_threshold: float = 0.3 # Volume threshold for LPS/LPSY detection
    wyckoff_lps_price_multiplier: float = 0.7 # Price multiplier for LPS/LPSY detection
    wyckoff_sos_multiplier: float = 1.2      # Multiplier for SOS/SOW detection
    wyckoff_ut_multiplier: float = 0.4       # Multiplier for upthrust detection
    wyckoff_sc_multiplier: float = 1.2       # Multiplier for selling climax detection
    wyckoff_ar_multiplier: float = 1.1       # Multiplier for automatic rally detection
    wyckoff_confirmation_threshold: float = 0.35 # Base threshold for trend/volume confirmation

    # Properties to provide derived values and backward compatibility
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
        # --- Core Analysis Settings ---
        phase_weight=0.20,  # Balanced for responsive tactical signals
        description="15min tactical entries",
        chart_image_time_delta=pd.Timedelta(hours=12),
        
        # --- Indicator Settings ---
        ema_length=10,  # Reduced for faster signal generation
        atr_settings=(14, 8, 18, 6, 8),  # Reduced for quicker sensitivity to volatility
        supertrend_multiplier=1.8,  # Decreased for earlier trend identification
        base_multiplier=0.75,  # Decreased for faster signal sensitivity
        momentum_multiplier=1.5,  # Decreased for quicker momentum detection
        
        # --- Volume Analysis Settings ---
        volume_ma_window=14,  # Reduced for more responsive volume analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=4,  # Reduced for earlier pattern detection
        support_resistance_lookback=35,  # Reduced for more immediate levels
        swing_lookback=3,         # Reduced for faster swing detection
        effort_lookback=3,        # Reduced for tactical short-term effort-result
        min_move_multiplier=0.7,  # Reduced to capture smaller moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.85,  # Reduced for earlier spring detection
        liquidation_factor=0.9,  # Reduced to catch liquidation moves earlier
        breakout_factor=0.9,  # Reduced for earlier breakout signals
        significant_levels_factor=0.85,  # Reduced for more immediate level detection
        atr_multiplier=0.25,  # Reduced for tighter zones
        volume_weighted_efficiency=0.3,  # Increased for stronger volume impact
        high_threshold=0.85,  # Reduced for earlier threshold triggers
        low_threshold=0.85,  # Symmetric reduced threshold

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=0.7,  # Reduced for earlier sign detection
        wyckoff_trend_lookback=3,        # Reduced for faster signal detection
        wyckoff_st_tolerance_low=0.990,  # Wider tolerance for detecting tests earlier
        wyckoff_st_tolerance_high=1.020, # Wider tolerance for detecting tests earlier
        wyckoff_lps_volume_threshold=0.22, # Lower threshold for faster signals
        wyckoff_lps_price_multiplier=0.6, # More sensitive
        wyckoff_sos_multiplier=1.05,     # More sensitive for SOS/SOW
        wyckoff_ut_multiplier=0.32,      # More sensitive for upthrusts
        wyckoff_sc_multiplier=1.05,      # More sensitive for climax
        wyckoff_ar_multiplier=0.95,      # More sensitive for auto rally
        wyckoff_confirmation_threshold=0.28 # Lower threshold for earlier signals
    ),
    
    Timeframe.MINUTES_30: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.25,  # Balanced for intermediate signals
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=24),
        
        # --- Indicator Settings ---
        ema_length=12,  # Reduced for faster signals
        atr_settings=(16, 10, 20, 7, 9),  # Reduced for quicker adaptation
        supertrend_multiplier=2.0,  # Reduced for earlier trend identification 
        base_multiplier=0.8,  # Reduced for faster signal triggering
        momentum_multiplier=1.4,  # Reduced for quicker momentum signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=16,  # Reduced for more responsive analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,  # Slightly reduced for earlier detection
        support_resistance_lookback=42,  # Reduced for more relevant levels
        swing_lookback=4,         # Reduced for more responsive swing detection
        effort_lookback=5,        # Slightly reduced for better effort-result
        min_move_multiplier=0.75,  # Reduced to capture smaller meaningful moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.9,  # Reduced for earlier spring detection
        liquidation_factor=0.95,  # Reduced for earlier liquidation signals
        breakout_factor=0.9,  # Reduced for earlier breakout detection
        significant_levels_factor=0.9,  # Reduced for more immediate levels
        atr_multiplier=0.24,  # Reduced for tighter zones
        volume_weighted_efficiency=0.32,  # Increased for stronger volume impact
        high_threshold=0.9,  # Reduced for earlier threshold triggers
        low_threshold=0.9,  # Symmetric threshold

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=0.85,  # Reduced for earlier detection
        wyckoff_trend_lookback=4,      
        wyckoff_st_tolerance_low=0.992,  # Wider tolerance
        wyckoff_st_tolerance_high=1.018,
        wyckoff_lps_volume_threshold=0.25,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.12,
        wyckoff_ut_multiplier=0.35,
        wyckoff_sc_multiplier=1.1,
        wyckoff_ar_multiplier=1.0,
        wyckoff_confirmation_threshold=0.30
    ),
    
    Timeframe.HOUR_1: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.30,  # Primary decision timeframe
        description="1h primary daily trend",
        chart_image_time_delta=pd.Timedelta(hours=48),
        
        # --- Indicator Settings ---
        ema_length=14,  # Reduced for quicker trend identification
        atr_settings=(18, 12, 24, 8, 10),  # Reduced for faster adaptation
        supertrend_multiplier=2.0,  # Reduced for earlier trend changes
        base_multiplier=0.85,  # Reduced for more sensitive base thresholds
        momentum_multiplier=1.5,  # Reduced for quicker momentum detection
        
        # --- Volume Analysis Settings ---
        volume_ma_window=18,  # Reduced for more responsive volume signals
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,  # Reduced for earlier pattern detection
        support_resistance_lookback=48,  # Reduced for more relevant levels
        swing_lookback=5,         # Reduced for quicker swing identification
        effort_lookback=6,        # Reduced for better effort-result
        min_move_multiplier=0.85,  # Reduced to capture meaningful moves earlier
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.9,  # Reduced for earlier detection
        liquidation_factor=0.9,  # Reduced for earlier detection
        breakout_factor=0.9,  # Reduced for earlier breakout detection
        significant_levels_factor=0.9,  # Reduced for faster level adaptation
        atr_multiplier=0.25,  # Reduced for tigher zones
        volume_weighted_efficiency=0.35,  # Increased for stronger volume impact
        high_threshold=0.9,  # Reduced for earlier threshold triggers
        low_threshold=0.9,  # Symmetric threshold

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=0.9,   # Reduced for earlier sign detection
        wyckoff_trend_lookback=4,        # Reduced for faster trend assessment
        wyckoff_st_tolerance_low=0.994,  # Slightly wider tolerance
        wyckoff_st_tolerance_high=1.015,
        wyckoff_lps_volume_threshold=0.28,
        wyckoff_lps_price_multiplier=0.65,
        wyckoff_sos_multiplier=1.1,     
        wyckoff_ut_multiplier=0.37,
        wyckoff_sc_multiplier=1.15,
        wyckoff_ar_multiplier=1.05,
        wyckoff_confirmation_threshold=0.30 
    ),
    
    Timeframe.HOURS_2: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.10,  # Daily bias context
        description="2h daily bias",
        chart_image_time_delta=pd.Timedelta(hours=72),
        
        # --- Indicator Settings ---
        ema_length=18,  # Reduced for faster trend identification
        atr_settings=(22, 14, 26, 9, 12),  # Reduced for faster adaptation
        supertrend_multiplier=2.0,  # Reduced for earlier trend identification 
        base_multiplier=0.95,  # Reduced for more sensitive signals
        momentum_multiplier=1.5,  # Reduced for better momentum detection
        
        # --- Volume Analysis Settings ---
        volume_ma_window=18,  # Reduced for more responsive analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6,  # Reduced for earlier pattern detection
        support_resistance_lookback=52,  # Reduced for more relevant levels
        swing_lookback=7,         # Reduced for faster swing structure recognition
        effort_lookback=7,        # Reduced for better trend-based effort-result analysis
        min_move_multiplier=1.0,  # Reduced for more meaningful moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.95,  # Reduced for earlier detection
        liquidation_factor=0.95,  # Reduced for earlier detection
        breakout_factor=1.0,  # Maintained for balance with shorter timeframes
        significant_levels_factor=1.0,  # Maintained for balance
        atr_multiplier=0.27,  # Reduced for tighter zones
        volume_weighted_efficiency=0.25,  # Increased for better efficiency
        high_threshold=1.0,  # Maintained as baseline
        low_threshold=0.9,  # Slightly reduced for better sensitivity

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=1.0,   # Maintained as baseline
        wyckoff_trend_lookback=5,       
        wyckoff_st_tolerance_low=0.995,  # Slightly wider for better detection
        wyckoff_st_tolerance_high=1.012,
        wyckoff_lps_volume_threshold=0.30,
        wyckoff_lps_price_multiplier=0.7,
        wyckoff_sos_multiplier=1.2,
        wyckoff_ut_multiplier=0.42,
        wyckoff_sc_multiplier=1.2,
        wyckoff_ar_multiplier=1.1,
        wyckoff_confirmation_threshold=0.33
    ),
    
    Timeframe.HOURS_4: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.10,  # Context timeframe
        description="4h daily context",
        chart_image_time_delta=pd.Timedelta(days=4),
        
        # --- Indicator Settings ---
        ema_length=24,  # Reduced while maintaining stability
        atr_settings=(26, 16, 32, 12, 16),  # Reduced for better adaptation
        supertrend_multiplier=2.2,  # Reduced while maintaining reliability
        base_multiplier=1.0,  # Reduced for better balance
        momentum_multiplier=1.6,  # Reduced for better momentum signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=22,  # Reduced for more responsive analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=7,  # Reduced while maintaining reliability
        support_resistance_lookback=65,  # Reduced for more relevant levels
        swing_lookback=8,        # Reduced for better market structure points
        effort_lookback=8,       # Reduced for better effort-result patterns
        min_move_multiplier=1.2,  # Maintained for significant moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.1,  # Maintained for reliability
        liquidation_factor=1.0,  # Reduced for better detection
        breakout_factor=1.1,  # Reduced for earlier breakouts
        significant_levels_factor=1.3,  # Reduced but still significant
        atr_multiplier=0.29,  # Slightly reduced
        volume_weighted_efficiency=0.18,  # Slightly increased
        high_threshold=1.1,  # Maintained for reliable signals
        low_threshold=0.85,  # Increased for better sensitivity

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=1.1,   # Maintained for noise filtering
        wyckoff_trend_lookback=6,        # Reduced slightly
        wyckoff_st_tolerance_low=0.997,  # Maintained for reliability
        wyckoff_st_tolerance_high=1.010,
        wyckoff_lps_volume_threshold=0.32,
        wyckoff_lps_price_multiplier=0.75,
        wyckoff_sos_multiplier=1.25,
        wyckoff_ut_multiplier=0.45,
        wyckoff_sc_multiplier=1.25,
        wyckoff_ar_multiplier=1.15,
        wyckoff_confirmation_threshold=0.35
    ),
    
    Timeframe.HOURS_8: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.05,  # Long-term market regime
        description="8h market regime",
        chart_image_time_delta=pd.Timedelta(days=6),
        
        # --- Indicator Settings ---
        ema_length=32,  # Reduced while maintaining stability
        atr_settings=(34, 20, 40, 14, 18),  # Reduced while maintaining stability
        supertrend_multiplier=2.5,  # Reduced slightly
        base_multiplier=1.1,  # Reduced for better balance
        momentum_multiplier=1.8,  # Reduced for better signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=28,  # Reduced for more responsive analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=8,  # Reduced while maintaining reliability
        support_resistance_lookback=85,  # Reduced while keeping strong levels
        swing_lookback=10,        # Reduced for better major swing points detection
        effort_lookback=12,       # Reduced for better effort-result analysis
        min_move_multiplier=1.7,  # Reduced while keeping significance
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.25,  # Reduced while maintaining reliability
        liquidation_factor=1.1,  # Reduced for better detection
        breakout_factor=1.3,  # Reduced for better detection
        significant_levels_factor=1.8,  # Reduced while maintaining significance
        atr_multiplier=0.32,  # Reduced slightly
        volume_weighted_efficiency=0.12,  # Increased slightly
        high_threshold=1.2,  # Reduced for better signals
        low_threshold=0.75,  # Increased for better thresholds

        # --- Wyckoff Sign Detection Parameters ---
        wyckoff_volatility_factor=1.2,   # Reduced while filtering noise
        wyckoff_trend_lookback=7,        # Reduced slightly
        wyckoff_st_tolerance_low=0.998,  # Maintained tight tolerance
        wyckoff_st_tolerance_high=1.006,
        wyckoff_lps_volume_threshold=0.35,
        wyckoff_lps_price_multiplier=0.8,
        wyckoff_sos_multiplier=1.35,
        wyckoff_ut_multiplier=0.5,
        wyckoff_sc_multiplier=1.35,
        wyckoff_ar_multiplier=1.2,
        wyckoff_confirmation_threshold=0.38
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
