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
    """Check if the given action is bullish."""
    return action in BULLISH_ACTIONS


def is_bearish_action(action: CompositeAction) -> bool:
    """Check if the given action is bearish."""
    return action in BEARISH_ACTIONS

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
    MINUTES_5 = ("5m", 5)      # New 5-minute timeframe
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
    # SHORT_TERM_TIMEFRAMES (24% total) - Reduced from 26% for less noise
    Timeframe.MINUTES_5: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.02,  # Further reduced to minimize noise impact
        description="5min noise-filtered signals",
        chart_image_time_delta=pd.Timedelta(hours=6),
        
        # --- Indicator Settings ---
        ema_length=12,  # Further increased for stronger smoothing
        atr_settings=(16, 9, 18, 6, 8),  # Increased for more stable ATR
        supertrend_multiplier=3.0,  # Increased to filter out small moves
        base_multiplier=0.9,  # Increased for more conservative signals
        momentum_multiplier=1.8,  # Increased to filter momentum noise
        
        # --- Volume Analysis Settings ---
        volume_ma_window=18,  # Increased for smoother volume profile
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=7,  # Increased to reduce false springs/upthrusts
        support_resistance_lookback=42,  # Increased for more significant levels
        effort_lookback=5,  # Back to standard value for consistency
        min_move_multiplier=0.8,  # Balanced value between 0.6 and 1.0
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.1,  # Further increased to only detect significant springs
        liquidation_factor=1.3,  # Increased to filter out non-liquidation events
        breakout_factor=1.2,  # Increased for clearer breakouts
        significant_levels_factor=1.15,  # Increased for more meaningful levels
        atr_multiplier=0.28,  # Slightly increased for wider zones
        volume_weighted_efficiency=0.18,  # More balanced value
        high_threshold=1.0,  # Reset to neutral baseline
        low_threshold=1.0  # Reset to neutral baseline for symmetry
    ),
    Timeframe.MINUTES_15: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.22,  # Slightly reduced
        description="15min tactical entries",
        chart_image_time_delta=pd.Timedelta(hours=12),
        
        # --- Indicator Settings ---
        ema_length=11,  # Increased from 9
        atr_settings=(15, 8, 18, 6, 8),  # Slightly increased
        supertrend_multiplier=2.0,  # Increased from 1.8
        base_multiplier=0.75,  # Slightly increased
        momentum_multiplier=1.6,  # Slightly increased
        
        # --- Volume Analysis Settings ---
        volume_ma_window=16,  # Increased from 14
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,  # Increased from 4
        support_resistance_lookback=36,  # Increased from 32
        effort_lookback=5,  # Standardized
        min_move_multiplier=0.7,  # More balanced
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.85,  # Increased from 0.75
        liquidation_factor=1.0,  # Increased from 0.9
        breakout_factor=0.92,  # Increased from 0.85
        significant_levels_factor=0.85,  # Increased from 0.75
        atr_multiplier=0.24,  # Increased from 0.22
        volume_weighted_efficiency=0.22,  # More balanced
        high_threshold=0.95,  # More neutral
        low_threshold=0.95  # Symmetric with high threshold
    ),
    
    # INTERMEDIATE_TIMEFRAMES (52% total) - Increased from 50% as the core decision timeframes
    Timeframe.MINUTES_30: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.24,  # Slightly increased from 0.23
        description="30min intraday swings",
        chart_image_time_delta=pd.Timedelta(hours=24),
        
        # --- Indicator Settings ---
        ema_length=13,  # Slightly increased from 12
        atr_settings=(18, 10, 22, 7, 9),  # Increased for stability
        supertrend_multiplier=2.0,  # Increased from 1.9
        base_multiplier=0.82,  # Slightly increased
        momentum_multiplier=1.4,  # Slightly increased
        
        # --- Volume Analysis Settings ---
        volume_ma_window=17,  # Slightly increased
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5, 
        support_resistance_lookback=45,  # Slightly increased
        effort_lookback=5,  # Standard value
        min_move_multiplier=0.75,  # Balanced value
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.9,  # Slightly increased
        liquidation_factor=1.0,  # Slightly increased
        breakout_factor=0.95,  # Slightly increased
        significant_levels_factor=0.85,  # Slightly increased
        atr_multiplier=0.24,  # Slightly increased
        volume_weighted_efficiency=0.25,  # More balanced value
        high_threshold=0.90,  # More neutral
        low_threshold=0.90  # Symmetric for balance
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.28,  # Slightly increased from 0.27
        description="1h primary intraday trend",
        chart_image_time_delta=pd.Timedelta(hours=48),
        
        # --- Indicator Settings ---
        ema_length=18,  # Kept the same
        atr_settings=(22, 13, 26, 9, 11),  # Slightly increased
        supertrend_multiplier=2.0,  # Reduced from 2.1 for better reactivity
        base_multiplier=0.92,  # Slightly decreased for better reactivity
        momentum_multiplier=1.5,  # Slightly decreased for better reactivity
        
        # --- Volume Analysis Settings ---
        volume_ma_window=18,  # Slightly decreased from 19
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,
        support_resistance_lookback=50,  # Slightly decreased from 54
        effort_lookback=5,  # Standard value
        min_move_multiplier=0.9,  # More balanced
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.95,  # Slightly decreased
        liquidation_factor=0.95,  # Slightly decreased
        breakout_factor=0.95,  # Slightly decreased
        significant_levels_factor=0.95,  # Slightly decreased
        atr_multiplier=0.25,
        volume_weighted_efficiency=0.24,  # More balanced
        high_threshold=1.0,  # Reset to neutral
        low_threshold=1.0  # Symmetric with high
    ),
    
    # LONG_TERM_TIMEFRAMES (12% total) - Kept the same
    Timeframe.HOURS_2: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.12,
        description="2h intraday bias",
        chart_image_time_delta=pd.Timedelta(hours=72),
        
        # --- Indicator Settings ---
        ema_length=22,  # Slightly decreased from 24 for better reactivity
        atr_settings=(24, 14, 28, 9, 12),  # Slightly adjusted
        supertrend_multiplier=2.0,  # Reduced from 2.2 for better reactivity
        base_multiplier=1.0,  # Slightly decreased from 1.05
        momentum_multiplier=1.5,  # Slightly decreased from 1.6
        
        # --- Volume Analysis Settings ---
        volume_ma_window=20,  # Slightly decreased from 22
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6,
        support_resistance_lookback=56,  # Slightly decreased from 60
        effort_lookback=6,
        min_move_multiplier=1.1,  # Slightly decreased from 1.2
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.05,  # Slightly decreased
        liquidation_factor=1.0,  # Slightly decreased
        breakout_factor=1.1,  # Slightly decreased
        significant_levels_factor=1.15,  # Slightly decreased
        atr_multiplier=0.26,  # Slightly decreased
        volume_weighted_efficiency=0.18,  # Slightly increased
        high_threshold=1.05,  # Slightly decreased
        low_threshold=0.95  # Slightly increased
    ),
    
    # CONTEXT_TIMEFRAMES (12% total) - Kept the same
    Timeframe.HOURS_4: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.08,
        description="4h daily context",
        chart_image_time_delta=pd.Timedelta(days=4),
        
        # --- Indicator Settings ---
        ema_length=30,  # Slightly decreased from 32
        atr_settings=(30, 16, 36, 12, 16),  # Slightly decreased
        supertrend_multiplier=2.4,  # Slightly decreased from 2.6
        base_multiplier=1.15,  # Slightly decreased from 1.2
        momentum_multiplier=1.7,  # Slightly decreased from 1.8
        
        # --- Volume Analysis Settings ---
        volume_ma_window=24,  # Slightly decreased from 26
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=7,
        support_resistance_lookback=72,  # Slightly decreased from 78
        effort_lookback=8,
        min_move_multiplier=1.4,  # Slightly decreased from 1.5
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.2,  # Slightly decreased from 1.25
        liquidation_factor=1.05,  # Slightly decreased from 1.1
        breakout_factor=1.25,  # Slightly decreased from 1.3
        significant_levels_factor=1.4,  # Slightly decreased from 1.5
        atr_multiplier=0.28,  # Slightly decreased from 0.3
        volume_weighted_efficiency=0.12,  # Slightly increased from 0.1
        high_threshold=1.15,  # Slightly decreased from 1.2
        low_threshold=0.85  # Slightly increased from 0.8
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.04,
        description="8h market regime",
        chart_image_time_delta=pd.Timedelta(days=6),
        
        # --- Indicator Settings ---
        ema_length=36,  # Decreased from 40
        atr_settings=(38, 20, 44, 14, 20),  # Slightly decreased
        supertrend_multiplier=2.6,  # Decreased from 2.8
        base_multiplier=1.25,  # Slightly decreased from 1.3
        momentum_multiplier=1.8,  # Slightly decreased from 1.9
        
        # --- Volume Analysis Settings ---
        volume_ma_window=30,  # Slightly decreased from 32
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=8,
        support_resistance_lookback=96,  # Slightly decreased from 104
        effort_lookback=10,
        min_move_multiplier=1.8,  # Slightly decreased from 2.0
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.35,  # Slightly decreased from 1.4
        liquidation_factor=1.15,  # Slightly decreased from 1.2
        breakout_factor=1.4,  # Slightly decreased from 1.5
        significant_levels_factor=1.8,  # Slightly decreased from 2.0
        atr_multiplier=0.32,  # Slightly decreased from 0.35
        volume_weighted_efficiency=0.08,  # Slightly increased from 0.05
        high_threshold=1.25,  # Slightly decreased from 1.3
        low_threshold=0.75  # Slightly increased from 0.7
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_5, Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
