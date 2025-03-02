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
    # Short-term group (35% total) - Last hour's action
    Timeframe.MINUTES_15: TimeframeSettings(
        phase_weight=0.15,          # Quick scalping signals
        ema_length=6,               # Faster EMA
        atr_settings=(6, 4, 10, 3, 4),  # More responsive ATR
        supertrend_multiplier=1.6,  # More sensitive
        base_multiplier=0.80,       # More sensitive to moves
        momentum_multiplier=1.5,    # More reactive
        description="15min scalping",
        volume_ma_window=10,        # Shorter for faster signals
        volume_short_ma_window=3,
        volume_long_ma_window=6,    # Shorter for faster trend detection
        spring_upthrust_window=3,   # Quick reversal detection
        volume_trend_window=4,
        price_strength_ma_window=5,
        price_change_window=3,
        support_resistance_lookback=24,  # Less noise in shorter lookback
        chart_image_time_delta=pd.Timedelta(hours=12)  # Shorter view for day trading
    ),
    Timeframe.MINUTES_30: TimeframeSettings(
        phase_weight=0.20,          # Recent momentum
        ema_length=10,              # Faster EMA
        atr_settings=(8, 5, 14, 4, 5),
        supertrend_multiplier=1.8,  # More sensitive
        base_multiplier=0.90,       # More reactive
        momentum_multiplier=1.4,    # Faster momentum detection
        description="30min swings",
        volume_ma_window=12,
        volume_short_ma_window=3,
        volume_long_ma_window=8,
        spring_upthrust_window=4,
        volume_trend_window=5,
        price_strength_ma_window=7,
        price_change_window=3,
        support_resistance_lookback=36,
        chart_image_time_delta=pd.Timedelta(hours=24)
    ),
    
    # Intermediate group (30% total) - Recent trend
    Timeframe.HOUR_1: TimeframeSettings(
        phase_weight=0.30,          # Primary decision timeframe
        ema_length=16,              # Faster for crypto
        atr_settings=(12, 8, 18, 6, 7),
        supertrend_multiplier=2.1,  # More sensitive
        base_multiplier=1.0,        # Balanced
        momentum_multiplier=1.5,    # More reactive
        description="1h trend",
        volume_ma_window=16,
        volume_short_ma_window=3,
        volume_long_ma_window=10,
        spring_upthrust_window=4,
        volume_trend_window=6,
        price_strength_ma_window=8,
        price_change_window=4,
        support_resistance_lookback=48,
        chart_image_time_delta=pd.Timedelta(hours=48)
    ),
    
    # Long-term group (25% total) - Established trend
    Timeframe.HOURS_2: TimeframeSettings(
        phase_weight=0.15,          # Trend confirmation
        ema_length=24,              # Faster
        atr_settings=(16, 8, 21, 7, 8),
        supertrend_multiplier=2.3,
        base_multiplier=1.1,
        momentum_multiplier=1.6,
        description="2h trend",
        volume_ma_window=20,
        volume_short_ma_window=4,
        volume_long_ma_window=12,
        spring_upthrust_window=5,
        volume_trend_window=8,
        price_strength_ma_window=10,
        price_change_window=4,
        support_resistance_lookback=56,
        chart_image_time_delta=pd.Timedelta(hours=72)
    ),
    Timeframe.HOURS_4: TimeframeSettings(
        phase_weight=0.10,          # Trend context
        ema_length=30,              # Faster
        atr_settings=(24, 10, 28, 8, 12),
        supertrend_multiplier=2.6,
        base_multiplier=1.2,
        momentum_multiplier=1.8,
        description="4h trend",
        volume_ma_window=24,
        volume_short_ma_window=4,
        volume_long_ma_window=14,
        spring_upthrust_window=5,
        volume_trend_window=10,
        price_strength_ma_window=12,
        price_change_window=5,
        support_resistance_lookback=72,
        chart_image_time_delta=pd.Timedelta(days=4)
    ),
    
    # Context group (10% total) - Market structure
    Timeframe.HOURS_8: TimeframeSettings(
        phase_weight=0.10,          # Keep for structure
        ema_length=38,
        atr_settings=(32, 12, 36, 9, 14),
        supertrend_multiplier=2.8,
        base_multiplier=1.3,
        momentum_multiplier=1.9,
        description="8h trend",
        volume_ma_window=32,
        volume_short_ma_window=5,
        volume_long_ma_window=18,
        spring_upthrust_window=6,
        volume_trend_window=12,
        price_strength_ma_window=16,
        price_change_window=6,
        support_resistance_lookback=96,
        chart_image_time_delta=pd.Timedelta(days=6)
    )
}

class SignificantLevelsData(TypedDict):
    resistance: List[float]
    support: List[float]
