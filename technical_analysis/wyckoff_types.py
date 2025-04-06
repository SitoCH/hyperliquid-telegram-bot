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
    CONSOLIDATING = "consolidating in range"  # New
    REVERSING = "showing reversal signals"    # New
    NEUTRAL = "showing no clear directional activity"
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
    # SHORT_TERM_TIMEFRAMES (25% total) - Increased from 22% to account for missing 5m
    Timeframe.MINUTES_15: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.25,  # Increased to account for removed 5m timeframe
        description="15min tactical entries",
        chart_image_time_delta=pd.Timedelta(hours=12),
        
        # --- Indicator Settings ---
        ema_length=14,  # Increased for more reliable signal 
        atr_settings=(18, 10, 20, 8, 10),  # Increased for stability
        supertrend_multiplier=2.2,  # Increased for better reliability
        base_multiplier=0.85,  # Increased for better robustness
        momentum_multiplier=1.8,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=20,  # Increased for smoother volume analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6,  # Increased for more reliable pattern detection
        support_resistance_lookback=42,  # Increased for more significant levels
        swing_lookback=4,         # Shorter lookback for more responsive swing detection
        effort_lookback=4,        # Shorter lookback for tactical effort-result analysis
        min_move_multiplier=0.85,  # Increased for more meaningful moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.0,  # Increased for fewer false springs
        liquidation_factor=1.1,  # Increased to filter minor liquidations
        breakout_factor=1.05,  # Increased for more significant breakouts
        significant_levels_factor=1.0,  # Increased for stronger level detection
        atr_multiplier=0.28,  # Increased for wider support/resistance zones
        volume_weighted_efficiency=0.25,  # Increased for stronger volume impact analysis
        high_threshold=1.0,  # Balanced threshold
        low_threshold=1.0  # Symmetric threshold
    ),
    
    # INTERMEDIATE_TIMEFRAMES (55% total) - Increased from 52% to emphasize hourly decision making
    Timeframe.MINUTES_30: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.25,  # Increased slightly
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=24),
        
        # --- Indicator Settings ---
        ema_length=16,  # Increased for more reliable signals
        atr_settings=(20, 12, 24, 8, 10),  # Increased for stability
        supertrend_multiplier=2.2,  # Increased for better reliability
        base_multiplier=0.9,  # Increased for robustness
        momentum_multiplier=1.6,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=20,  # Increased for smoother analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6, 
        support_resistance_lookback=50,  # Increased for more significant levels
        swing_lookback=5,         # Balanced lookback for swing detection
        effort_lookback=5,        # Standard lookback for effort-result analysis
        min_move_multiplier=0.85,  # Increased for more meaningful moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.0,  # Increased for fewer false signals
        liquidation_factor=1.05,  # Increased for better detection
        breakout_factor=1.0,  # Increased for balanced breakout detection
        significant_levels_factor=1.0,  # Increased for stronger levels
        atr_multiplier=0.26,  # Increased slightly
        volume_weighted_efficiency=0.28,  # Increased for stronger volume impact
        high_threshold=0.95,  # Slightly reduced for balance
        low_threshold=0.95  # Symmetric for balance
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.30,  # Increased to emphasize hourly decision making
        description="1h primary daily trend",
        chart_image_time_delta=pd.Timedelta(hours=48),
        
        # --- Indicator Settings ---
        ema_length=20,  # Increased for more reliable signals
        atr_settings=(24, 14, 28, 10, 12),  # Increased for stability
        supertrend_multiplier=2.2,  # Increased for better reliability
        base_multiplier=1.0,  # Increased for robustness
        momentum_multiplier=1.7,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=20,  # Standardized for better consistency
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6,  # Increased for reliability
        support_resistance_lookback=55,  # Increased for stronger level detection
        swing_lookback=6,         # Slightly longer lookback to identify more significant swings
        effort_lookback=7,        # Longer lookback for consistent effort-result analysis
        min_move_multiplier=1.0,  # Standardized for baseline
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.0,  # Standardized baseline
        liquidation_factor=1.0,  # Standardized baseline
        breakout_factor=1.0,  # Standardized baseline
        significant_levels_factor=1.0,  # Standardized baseline
        atr_multiplier=0.28,  # Increased for wider zones
        volume_weighted_efficiency=0.26,  # Increased for stronger volume impact
        high_threshold=1.0,  # Standard baseline
        low_threshold=1.0  # Symmetric with high
    ),
    
    # LONG_TERM_TIMEFRAMES (10% total) - Slightly reduced from 12%
    Timeframe.HOURS_2: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.10,  # Adjusted
        description="2h daily bias",
        chart_image_time_delta=pd.Timedelta(hours=72),
        
        # --- Indicator Settings ---
        ema_length=24,  # Increased for stability
        atr_settings=(26, 16, 30, 10, 14),  # Increased for stability
        supertrend_multiplier=2.2,  # Adjusted for consistency
        base_multiplier=1.1,  # Increased for robustness
        momentum_multiplier=1.7,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=22,  # Increased for smoother analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=7,  # Increased for reliability
        support_resistance_lookback=60,  # Increased for stronger level detection
        swing_lookback=8,         # Longer lookback to identify meaningful swing structures
        effort_lookback=8,        # Longer lookback for trend-based effort-result analysis
        min_move_multiplier=1.2,  # Increased for more significant moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.1,  # Increased for fewer false signals
        liquidation_factor=1.05,  # Increased for better detection
        breakout_factor=1.15,  # Increased for stronger breakouts
        significant_levels_factor=1.2,  # Increased for stronger levels
        atr_multiplier=0.3,  # Increased for wider zones
        volume_weighted_efficiency=0.2,  # Increased for stronger volume impact
        high_threshold=1.1,  # Increased for clearer thresholds
        low_threshold=0.9  # Decreased for clearer thresholds
    ),
    
    # CONTEXT_TIMEFRAMES (10% total) - Reduced from 12% to balance the weightings
    Timeframe.HOURS_4: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.07,  # Adjusted
        description="4h daily context",
        chart_image_time_delta=pd.Timedelta(days=4),
        
        # --- Indicator Settings ---
        ema_length=32,  # Increased for stability
        atr_settings=(32, 18, 38, 14, 18),  # Increased for stability
        supertrend_multiplier=2.5,  # Adjusted for consistency
        base_multiplier=1.2,  # Increased for robustness
        momentum_multiplier=1.8,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=26,  # Increased for smoother analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=8,  # Increased for reliability
        support_resistance_lookback=75,  # Increased for stronger levels
        swing_lookback=10,        # Extended lookback for significant market structure points
        effort_lookback=10,       # Extended lookback for broader effort-result patterns
        min_move_multiplier=1.5,  # Increased for more significant moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.25,  # Increased for fewer false signals
        liquidation_factor=1.1,  # Increased for better detection
        breakout_factor=1.3,  # Increased for stronger breakouts
        significant_levels_factor=1.5,  # Increased for stronger levels
        atr_multiplier=0.32,  # Increased for wider zones
        volume_weighted_efficiency=0.15,  # Adjusted for balance
        high_threshold=1.2,  # Increased for clearer thresholds
        low_threshold=0.8  # Decreased for clearer thresholds
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.03,  # Adjusted
        description="8h market regime",
        chart_image_time_delta=pd.Timedelta(days=6),
        
        # --- Indicator Settings ---
        ema_length=40,  # Increased for stability
        atr_settings=(40, 22, 46, 16, 22),  # Increased for stability
        supertrend_multiplier=2.8,  # Increased for reliability
        base_multiplier=1.3,  # Increased for robustness
        momentum_multiplier=2.0,  # Increased for fewer false signals
        
        # --- Volume Analysis Settings ---
        volume_ma_window=32,  # Increased for smoother analysis
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=9,  # Increased for reliability
        support_resistance_lookback=100,  # Increased for stronger levels
        swing_lookback=12,        # Long lookback for major swing points detection
        effort_lookback=14,       # Long lookback for macro effort-result analysis
        min_move_multiplier=2.0,  # Increased for more significant moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.4,  # Increased for fewer false signals
        liquidation_factor=1.2,  # Increased for better detection
        breakout_factor=1.5,  # Increased for stronger breakouts
        significant_levels_factor=2.0,  # Increased for stronger levels
        atr_multiplier=0.35,  # Increased for wider zones
        volume_weighted_efficiency=0.1,  # Adjusted for balance
        high_threshold=1.3,  # Increased for clearer thresholds
        low_threshold=0.7  # Decreased for clearer thresholds
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
