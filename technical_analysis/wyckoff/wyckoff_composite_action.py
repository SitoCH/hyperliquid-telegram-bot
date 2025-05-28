import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final, Dict, List, Optional, Any, Tuple
from .wyckoff_types import (
    MarketPattern, VolatilityState, WyckoffState, WyckoffPhase, EffortResult, 
    CompositeAction, WyckoffSign, FundingState, VolumeState, Timeframe, VolumeMetrics
)
from ..funding_rates_cache import FundingRateEntry
from .wyckoff_description import generate_wyckoff_description
from dataclasses import dataclass
from logging_utils import logger
from .adaptive_thresholds import AdaptiveThresholdManager

EFFORT_THRESHOLD: Final[float] = 0.75  # Increased from 0.7 for cleaner effort signals

def detect_composite_action(
    df: pd.DataFrame,
    price_strength: float,
    vol_metrics: VolumeMetrics,
    effort_vs_result: float
) -> CompositeAction:
    """Enhanced composite action detection for crypto markets with improved pattern recognition."""
    try:
        # Safety check for minimum required data
        if len(df) < 5:
            return CompositeAction.UNKNOWN
            
        # Calculate adaptive thresholds based on recent volatility
        price_volatility = df['c'].pct_change().rolling(5).std().iloc[-1]
        min_price_move = max(0.01, price_volatility * 2.0)  # More adaptive threshold
        
        # Recent price action metrics
        recent_change = df['c'].pct_change().iloc[-1] 
        body_to_range_ratio = abs(df['c'].iloc[-1] - df['o'].iloc[-1]) / max(df['h'].iloc[-1] - df['l'].iloc[-1], 1e-8)
        
        # Calculate volume trend - consistent definition to use throughout the function
        volume_trend = (df['v'].iloc[-1] / df['v'].iloc[-5:].mean())
        
        # Improved liquidation cascade detection with volatility-adjusted thresholds
        liquidation_cascade = (
            abs(recent_change) > max(0.05, price_volatility * 4) and  # Adaptive to market conditions
            abs(vol_metrics.strength) > 2.5 and
            vol_metrics.ratio > 3.0 and
            body_to_range_ratio > 0.8  # Strong directional candle
        )
        
        if liquidation_cascade:
            return (CompositeAction.MARKING_UP if recent_change > 0 
                    else CompositeAction.MARKING_DOWN)

        # Enhanced absorption pattern detection using min_price_move
        absorption_volume = (
            vol_metrics.ratio > 2.0 and                      # Double normal volume
            abs(vol_metrics.strength) > 1.5 and              # Significant volume deviation
            body_to_range_ratio < 0.3 and                    # Small body relative to range
            volume_trend > 1.2                               # Rising volume pattern - using volume_trend
        )
        
        if absorption_volume:
            # More nuanced decision based on context
            if price_strength < -0.5:  # More clearly bearish context
                return CompositeAction.ACCUMULATING
            elif price_strength > 0.5:  # More clearly bullish context
                return CompositeAction.DISTRIBUTING
            # If price strength is ambiguous, check where within the range the price closed
            else:
                position_in_range = (df['c'].iloc[-1] - df['l'].iloc[-1]) / max(df['h'].iloc[-1] - df['l'].iloc[-1], 1e-8)
                return CompositeAction.ACCUMULATING if position_in_range < 0.35 else CompositeAction.DISTRIBUTING
        
        # Improved whale manipulation pattern detection with min_price_move
        sudden_volume_spike = volume_trend > 3.0  # Using volume_trend consistently
        price_rejection = (max(
            abs(df['h'].iloc[-1] - df['c'].iloc[-1]),  # Upper wick
            abs(df['l'].iloc[-1] - df['c'].iloc[-1])   # Lower wick
        ) > abs(df['c'].iloc[-1] - df['o'].iloc[-1]) * 2)  # Significant wick compared to body
        
        # Use min_price_move to check if the price move is significant enough
        significant_move = abs(recent_change) > min_price_move * 0.8
        
        if sudden_volume_spike and price_rejection and significant_move:
            # Check where the rejection happened
            if df['h'].iloc[-1] - df['c'].iloc[-1] > df['c'].iloc[-1] - df['l'].iloc[-1]:  # Rejection from top
                return CompositeAction.DISTRIBUTING
            return CompositeAction.ACCUMULATING

        # Improved absorption of supply/demand detection with min_price_move
        price_range = df['h'] - df['l']
        price_close = df['c'] - df['o']
        relative_range = price_range.iloc[-1] / price_range.iloc[-10:].mean()
        
        absorption = (
            relative_range > 1.2 and  # Wider than normal range
            abs(price_close.iloc[-1]) < min_price_move * 0.7 and  # Small body relative to volatility
            vol_metrics.consistency > 0.6  # Consistent volume pattern
        )
        
        if absorption and volume_trend > 1.3:  # Using volume_trend instead of VOLUME_THRESHOLD
            # Check for a potential reversal setup using min_price_move for thresholds
            if (vol_metrics.trend_strength > 0.5 and price_strength < -0.5) or \
               (vol_metrics.trend_strength < -0.5 and price_strength > 0.5):
                return CompositeAction.REVERSING
                
            # Normal absorption pattern
            if price_strength < 0:
                return CompositeAction.ACCUMULATING
            return CompositeAction.DISTRIBUTING

        # Strong effort vs result detection using min_price_move
        strong_effort = abs(effort_vs_result) > EFFORT_THRESHOLD
        price_move_significant = abs(recent_change) > min_price_move
        
        if strong_effort and price_move_significant:
            if effort_vs_result > 0 and vol_metrics.short_ma > vol_metrics.long_ma:
                return CompositeAction.MARKING_UP
            elif effort_vs_result < 0 and vol_metrics.short_ma < vol_metrics.long_ma:
                return CompositeAction.MARKING_DOWN

        # Better consolidation detection using volume_trend
        is_narrowing_range = price_range.iloc[-3:].std() < price_range.iloc[-8:-3].std() * 0.8
        is_declining_volume = volume_trend < 0.85  # Using volume_trend
        
        if is_narrowing_range and is_declining_volume and abs(price_strength) < 1.0:
            return CompositeAction.CONSOLIDATING

        # Improved divergence detection using min_price_move and volume_trend
        price_highs = df['h'].rolling(5).max()
        price_lows = df['l'].rolling(5).min()
        
        # Bullish divergence with consistent use of min_price_move and volume_trend
        bullish_divergence = (
            price_lows.iloc[-1] < price_lows.iloc[-5] * 0.995 and  # Lower low (with small threshold)
            volume_trend > 1.5 and                                # Higher volume - using volume_trend
            effort_vs_result > 0.2 and                             # Positive effort vs result
            df['c'].iloc[-1] > df['o'].iloc[-1] and                # Bullish candle
            abs(df['c'].iloc[-1] - df['o'].iloc[-1]) > min_price_move  # Significant price move
        )
        
        # Bearish divergence with consistent use of min_price_move and volume_trend
        bearish_divergence = (
            price_highs.iloc[-1] > price_highs.iloc[-5] * 1.005 and  # Higher high (with small threshold)
            volume_trend > 1.5 and                                  # Higher volume - using volume_trend
            effort_vs_result < -0.2 and                              # Negative effort vs result
            df['c'].iloc[-1] < df['o'].iloc[-1] and                  # Bearish candle
            abs(df['c'].iloc[-1] - df['o'].iloc[-1]) > min_price_move  # Significant price move
        )
        
        if bullish_divergence:
            return CompositeAction.ACCUMULATING

        if bearish_divergence:
            return CompositeAction.DISTRIBUTING
        
        # Default case - more conditions for neutral determination
        return CompositeAction.NEUTRAL
    except Exception as e:
        logger.error(f"Error in composite action detection: {e}")
        return CompositeAction.UNKNOWN
