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
    phase_weight: float              # Used in get_phase_weight()
    ema_length: int                  # Used in apply_indicators() for EMA calculation
    atr_settings: tuple[int, int, int, int, int]  # Used in get_indicator_settings()
    supertrend_multiplier: float     # Used in apply_indicators() for Supertrend
    base_multiplier: float          # Used in thresholds property
    momentum_multiplier: float      # Used in thresholds property
    description: str                # Used for logging/display purposes
    # Simplified volume settings - further reduced
    volume_ma_window: int           # Primary volume moving average window
    spring_upthrust_window: int     # For reversal pattern detection
    support_resistance_lookback: int # For S/R level identification
    chart_image_time_delta: pd.Timedelta  # For chart rendering
    effort_lookback: int = 5        # Periods to look back for effort-result analysis
    min_move_multiplier: float = 1.0  # Minimum price move multiplier for effort-result analysis
    # Adaptive threshold factors
    spring_factor: float = 1.0      # Factor for spring/upthrust detection
    liquidation_factor: float = 1.0 # Factor for liquidation cascade detection
    breakout_factor: float = 1.0    # Factor for breakout detection
    significant_levels_factor: float = 1.0  # Factor for price level detection
    atr_multiplier: float = 0.25    # ATR multiplier for support/resistance clustering

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
    HOURS_2 = ("2h", 120)  # New timeframe
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
    # Short-term group (40% total) - Increased for intraday crypto responsiveness
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.20,  # Increased for faster intraday signals
        ema_length=8,
        atr_settings=(12, 7, 16, 5, 7),
        supertrend_multiplier=1.7,
        base_multiplier=0.65,
        momentum_multiplier=1.5,
        description="15min scalping",
        volume_ma_window=13,
        spring_upthrust_window=4,
        support_resistance_lookback=30,
        chart_image_time_delta=pd.Timedelta(hours=12),
        effort_lookback=3,            # Faster response for scalping
        min_move_multiplier=0.5,      # More sensitive to smaller moves
        spring_factor=0.7,           # More sensitive on shorter timeframes
        liquidation_factor=0.9,
        breakout_factor=0.8,
        significant_levels_factor=0.7,
        atr_multiplier=0.2  # Tighter for short timeframes
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.20,  # Increased for tactical intraday decisions
        ema_length=12,
        atr_settings=(16, 9, 20, 6, 8),
        supertrend_multiplier=1.9,
        base_multiplier=0.74,
        momentum_multiplier=1.3,
        description="30min swings",
        volume_ma_window=14,
        spring_upthrust_window=5,
        support_resistance_lookback=42,
        chart_image_time_delta=pd.Timedelta(hours=24),
        effort_lookback=4,            # Quick response for swing trades
        min_move_multiplier=0.75,     # Still sensitive to smaller moves
        spring_factor=0.85,
        liquidation_factor=0.95,
        breakout_factor=0.9,
        significant_levels_factor=0.8,
        atr_multiplier=0.22
    ),
    
    # Intermediate group (35% total) - Critical for intraday context
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.22,  # Maintained high weight for primary intraday trend
        ema_length=18,
        atr_settings=(20, 12, 26, 8, 10),
        supertrend_multiplier=2.1,
        base_multiplier=0.98,
        momentum_multiplier=1.6,
        description="1h trend",
        volume_ma_window=19,
        spring_upthrust_window=5,
        support_resistance_lookback=52,
        chart_image_time_delta=pd.Timedelta(hours=48),
        effort_lookback=5,            # Standard lookback for main trend
        min_move_multiplier=1.0,      # Base threshold for moves
        spring_factor=1.0,           # Base reference
        liquidation_factor=1.0,
        breakout_factor=1.0,
        significant_levels_factor=1.0,
        atr_multiplier=0.25
    ),
    Timeframe.HOURS_2: TimeframeSettings(
        phase_weight=0.13,  # Still important for intraday structure
        ema_length=26,
        atr_settings=(24, 14, 30, 9, 12),
        supertrend_multiplier=2.3,
        base_multiplier=1.05,
        momentum_multiplier=1.6,
        description="2h trend",
        volume_ma_window=22,
        spring_upthrust_window=6,
        support_resistance_lookback=60,
        chart_image_time_delta=pd.Timedelta(hours=72),
        effort_lookback=6,            # Longer lookback for trend structure
        min_move_multiplier=1.25,     # Less sensitive to small moves
        spring_factor=1.1,
        liquidation_factor=1.05,
        breakout_factor=1.15,
        significant_levels_factor=1.2,
        atr_multiplier=0.27
    ),
    
    # Long-term group (25% total) - Reduced but still vital for context
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.15,  # Reduced but still important for larger context
        ema_length=32,
        atr_settings=(32, 16, 38, 12, 16),
        supertrend_multiplier=2.6,
        base_multiplier=1.2,
        momentum_multiplier=1.8,
        description="4h trend",
        volume_ma_window=26,
        spring_upthrust_window=7,
        support_resistance_lookback=78,
        chart_image_time_delta=pd.Timedelta(days=4),
        effort_lookback=8,            # Significant lookback for larger trends
        min_move_multiplier=1.5,      # Only considering meaningful moves
        spring_factor=1.25,
        liquidation_factor=1.1,
        breakout_factor=1.3,
        significant_levels_factor=1.5,
        atr_multiplier=0.3
    ),
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.10,  # Reduced but maintained for campaign insight
        ema_length=40,
        atr_settings=(42, 20, 48, 14, 20),
        supertrend_multiplier=2.8,
        base_multiplier=1.3,
        momentum_multiplier=1.9,
        description="8h trend",
        volume_ma_window=32,
        spring_upthrust_window=8,
        support_resistance_lookback=104,
        chart_image_time_delta=pd.Timedelta(days=6),
        effort_lookback=10,           # Maximum lookback for market context
        min_move_multiplier=2.0,      # Only considering significant moves
        spring_factor=1.4,
        liquidation_factor=1.2,
        breakout_factor=1.5,
        significant_levels_factor=2.0,
        atr_multiplier=0.35  # Wider for longer timeframes
    )
}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
