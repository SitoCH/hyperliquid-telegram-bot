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

# Rebalanced weights for crypto focus
_TIMEFRAME_SETTINGS = {
    # SHORT_TERM_TIMEFRAMES (29% total) - For scalping and quick tactical decisions
    Timeframe.MINUTES_5: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.08,  # Reduced to 0.08 (from 0.084)
        description="5min scalping",
        chart_image_time_delta=pd.Timedelta(hours=6),
        
        # --- Indicator Settings ---
        ema_length=5,
        atr_settings=(8, 5, 12, 3, 5),
        supertrend_multiplier=1.5,
        base_multiplier=0.55,
        momentum_multiplier=1.7,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=10,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=3,
        support_resistance_lookback=24,
        effort_lookback=2,            # Minimal lookback for ultra-short timeframe
        min_move_multiplier=0.4,      # Very sensitive to small moves
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.6,            # Even more sensitive on ultra-short timeframes
        liquidation_factor=0.8,
        breakout_factor=0.7,
        significant_levels_factor=0.6,
        atr_multiplier=0.18,          # Tightest setting for quick reactions
        volume_weighted_efficiency=0.35,  # Most reactive
        high_threshold=0.75,          # Easiest to achieve
        low_threshold=1.3             # Highest floor for noise filtering
    ),
    Timeframe.MINUTES_15: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.21,  # Increased to 0.21 (from 0.205)
        description="15min scalping",
        chart_image_time_delta=pd.Timedelta(hours=12),
        
        # --- Indicator Settings ---
        ema_length=8,
        atr_settings=(12, 7, 16, 5, 7),
        supertrend_multiplier=1.7,
        base_multiplier=0.65,
        momentum_multiplier=1.5,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=13,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=4,
        support_resistance_lookback=30,
        effort_lookback=3,
        min_move_multiplier=0.5,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.7,
        liquidation_factor=0.9,
        breakout_factor=0.8,
        significant_levels_factor=0.7,
        atr_multiplier=0.2,
        volume_weighted_efficiency=0.3,
        high_threshold=0.8,
        low_threshold=1.2
    ),
    
    # INTERMEDIATE_TIMEFRAMES (44% total) - Primary trading decision timeframes
    Timeframe.MINUTES_30: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.21,  # Increased to 0.21 (from 0.205)
        description="30min swings",
        chart_image_time_delta=pd.Timedelta(hours=24),
        
        # --- Indicator Settings ---
        ema_length=12,
        atr_settings=(16, 9, 20, 6, 8),
        supertrend_multiplier=1.9,
        base_multiplier=0.74,
        momentum_multiplier=1.3,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=14,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,
        support_resistance_lookback=42,
        effort_lookback=4,
        min_move_multiplier=0.75,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=0.85,
        liquidation_factor=0.95,
        breakout_factor=0.9,
        significant_levels_factor=0.8,
        atr_multiplier=0.22,
        volume_weighted_efficiency=0.25,
        high_threshold=0.85,
        low_threshold=1.1
    ),
    Timeframe.HOUR_1: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.23,  # Set to 0.23 (from 0.231)
        description="1h trend",
        chart_image_time_delta=pd.Timedelta(hours=48),
        
        # --- Indicator Settings ---
        ema_length=18,
        atr_settings=(20, 12, 26, 8, 10),
        supertrend_multiplier=2.1,
        base_multiplier=0.98,
        momentum_multiplier=1.6,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=19,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=5,
        support_resistance_lookback=52,
        effort_lookback=5,
        min_move_multiplier=1.0,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.0,
        liquidation_factor=1.0,
        breakout_factor=1.0,
        significant_levels_factor=1.0,
        atr_multiplier=0.25,
        volume_weighted_efficiency=0.2,
        high_threshold=1.0,
        low_threshold=1.0
    ),
    
    # LONG_TERM_TIMEFRAMES (12% total) - Daily bias and trend direction
    Timeframe.HOURS_2: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.12,  # Reduced to 0.12 (from 0.125)
        description="2h trend",
        chart_image_time_delta=pd.Timedelta(hours=72),
        
        # --- Indicator Settings ---
        ema_length=26,
        atr_settings=(24, 14, 30, 9, 12),
        supertrend_multiplier=2.3,
        base_multiplier=1.05,
        momentum_multiplier=1.6,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=22,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=6,
        support_resistance_lookback=60,
        effort_lookback=6,
        min_move_multiplier=1.25,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.1,
        liquidation_factor=1.05,
        breakout_factor=1.15,
        significant_levels_factor=1.2,
        atr_multiplier=0.27,
        volume_weighted_efficiency=0.15,
        high_threshold=1.1,
        low_threshold=0.9
    ),
    
    # CONTEXT_TIMEFRAMES (15% total) - Market structure and bigger picture (capped)
    Timeframe.HOURS_4: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.10,  # Adjusted to 0.10 (from 0.095)
        description="4h trend",
        chart_image_time_delta=pd.Timedelta(days=4),
        
        # --- Indicator Settings ---
        ema_length=32,
        atr_settings=(32, 16, 38, 12, 16),
        supertrend_multiplier=2.6,
        base_multiplier=1.2,
        momentum_multiplier=1.8,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=26,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=7,
        support_resistance_lookback=78,
        effort_lookback=8,
        min_move_multiplier=1.5,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.25,
        liquidation_factor=1.1,
        breakout_factor=1.3,
        significant_levels_factor=1.5,
        atr_multiplier=0.3,
        volume_weighted_efficiency=0.1,
        high_threshold=1.2,
        low_threshold=0.8
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        # --- Core Analysis Settings ---
        phase_weight=0.05,  # Reduced to 0.05 (from 0.055)
        description="8h trend",
        chart_image_time_delta=pd.Timedelta(days=6),
        
        # --- Indicator Settings ---
        ema_length=40,
        atr_settings=(42, 20, 48, 14, 20),
        supertrend_multiplier=2.8,
        base_multiplier=1.3,
        momentum_multiplier=1.9,
        
        # --- Volume Analysis Settings ---
        volume_ma_window=32,
        
        # --- Support/Resistance & Pattern Detection ---
        spring_upthrust_window=8,
        support_resistance_lookback=104,
        effort_lookback=10,
        min_move_multiplier=2.0,
        
        # --- Adaptive Threshold Factors ---
        spring_factor=1.4,
        liquidation_factor=1.2,
        breakout_factor=1.5,
        significant_levels_factor=2.0,
        atr_multiplier=0.35,
        volume_weighted_efficiency=0.05,
        high_threshold=1.3,
        low_threshold=0.7
    )
}

SHORT_TERM_TIMEFRAMES = {Timeframe.MINUTES_5, Timeframe.MINUTES_15}
INTERMEDIATE_TIMEFRAMES = {Timeframe.MINUTES_30, Timeframe.HOUR_1}
LONG_TERM_TIMEFRAMES = {Timeframe.HOURS_2}
CONTEXT_TIMEFRAMES = {Timeframe.HOURS_4, Timeframe.HOURS_8}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
