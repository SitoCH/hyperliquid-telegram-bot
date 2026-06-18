import pytest
from unittest.mock import MagicMock
from technical_analysis.wyckoff.mtf_engine import analyze_multi_timeframe
from technical_analysis.wyckoff.wyckoff_types import (
    WyckoffState, WyckoffPhase, WyckoffSign, MarketPattern,
    CompositeAction, EffortResult, Timeframe, VolumeState, FundingState,
    VolatilityState, MarketLiquidity, MultiTimeframeDirection
)


def test_analyze_multi_timeframe_basic():
    # Mock states for different timeframes
    states = {
        Timeframe.MINUTES_15: WyckoffState(
            phase=WyckoffPhase.MARKUP,
            uncertain_phase=False,
            volume=VolumeState.HIGH,
            pattern=MarketPattern.TRENDING,
            volatility=VolatilityState.NORMAL,
            is_spring=False,
            is_upthrust=False,
            effort_vs_result=EffortResult.STRONG,
            composite_action=CompositeAction.MARKING_UP,
            wyckoff_sign=WyckoffSign.SIGN_OF_STRENGTH,
            funding_state=FundingState.NEUTRAL,
            description="Bullish 15m"
        ),
        Timeframe.HOUR_1: WyckoffState(
            phase=WyckoffPhase.ACCUMULATION,
            uncertain_phase=False,
            volume=VolumeState.NEUTRAL,
            pattern=MarketPattern.RANGING,
            volatility=VolatilityState.NORMAL,
            is_spring=True,
            is_upthrust=False,
            effort_vs_result=EffortResult.STRONG,
            composite_action=CompositeAction.ACCUMULATING,
            wyckoff_sign=WyckoffSign.LAST_POINT_OF_SUPPORT,
            funding_state=FundingState.NEUTRAL,
            description="Bullish 1h"
        ),
        Timeframe.HOURS_4: WyckoffState(
            phase=WyckoffPhase.ACCUMULATION,
            uncertain_phase=True,
            volume=VolumeState.LOW,
            pattern=MarketPattern.RANGING,
            volatility=VolatilityState.NORMAL,
            is_spring=False,
            is_upthrust=False,
            effort_vs_result=EffortResult.NEUTRAL,
            composite_action=CompositeAction.NEUTRAL,
            wyckoff_sign=WyckoffSign.NONE,
            funding_state=FundingState.NEUTRAL,
            description="Bullish 4h"
        )
    }

    significant_levels = {
        tf: {'resistance': [105.0], 'support': [95.0]}
        for tf in states.keys()
    }

    context = analyze_multi_timeframe(
        states=states,
        coin="BTC",
        mid=100.0,
        significant_levels=significant_levels,  # type: ignore[arg-type]
        interactive_analysis=True
    )

    assert context.description != ""
    assert "Market Analysis" in context.description
    # Given the bullish states, it should likely be bullish or at least non-error
