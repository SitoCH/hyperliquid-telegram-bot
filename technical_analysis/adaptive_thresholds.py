from typing import Dict, Final, Optional
import numpy as np
import pandas as pd
from logging_utils import logger
from .wyckoff_types import Timeframe

class AdaptiveThresholdManager:
    """Manages dynamic thresholds for Wyckoff analysis based on market conditions"""
    
    @staticmethod
    def get_spring_upthrust_thresholds(df: pd.DataFrame, timeframe: Timeframe) -> Dict[str, float]:
        """Calculate dynamic thresholds for spring/upthrust detection"""
        if df.empty or len(df) < 5:
            # Default fallback values
            return {"spring": 0.001, "upthrust": 0.001}
        
        try:
            # Calculate volatility using ATR
            if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
                atr = df['ATR'].iloc[-1]
                price = df['c'].iloc[-1]
                volatility = atr / price
            else:
                # Fallback to standard deviation if ATR isn't available
                volatility = df['c'].pct_change().std()
                
            # Scale threshold by timeframe and volatility
            timeframe_factor = {
                Timeframe.MINUTES_15: 0.7,   # More sensitive on shorter timeframes
                Timeframe.MINUTES_30: 0.85,
                Timeframe.HOUR_1: 1.0,       # Base reference
                Timeframe.HOURS_2: 1.1,
                Timeframe.HOURS_4: 1.25,
                Timeframe.HOURS_8: 1.4
            }.get(timeframe, 1.0)
            
            # Calculate adaptive thresholds
            base_threshold = 0.001
            volatility_multiplier = 10 * np.clip(volatility * 100, 0.25, 4.0)
            spring_threshold = base_threshold * (1 + volatility_multiplier) * timeframe_factor
            upthrust_threshold = spring_threshold * 1.05  # Slightly higher for upthrush
            
            return {
                "spring": spring_threshold,
                "upthrust": upthrust_threshold
            }
        except Exception as e:
            logger.warning(f"Error calculating spring/upthrust thresholds: {e}")
            return {"spring": 0.001, "upthrust": 0.001}
    
    @staticmethod
    def get_liquidation_thresholds(df: pd.DataFrame, timeframe: Timeframe) -> Dict[str, float]:
        """Calculate dynamic thresholds for liquidation cascade detection"""
        if df.empty or len(df) < 10:
            # Default fallback values
            return {
                "vol_threshold": 2.5, 
                "price_threshold": 0.04,
                "velocity_threshold": 2.0,
                "effort_threshold": 0.7
            }
            
        try:
            # Calculate average volatility from recent data
            vol_std = df['v'].pct_change().std()
            price_std = df['c'].pct_change().std()
            
            # Scale thresholds by timeframe
            timeframe_factor = {
                Timeframe.MINUTES_15: 0.9,   # More sensitive on shorter timeframes
                Timeframe.MINUTES_30: 0.95,
                Timeframe.HOUR_1: 1.0,       # Base reference
                Timeframe.HOURS_2: 1.05,
                Timeframe.HOURS_4: 1.1,
                Timeframe.HOURS_8: 1.2
            }.get(timeframe, 1.0)
            
            # Calculate adaptive liquidation thresholds
            vol_threshold = 2.5 * np.clip(1.0 / (vol_std * 10 + 0.5), 0.8, 1.4)
            price_threshold = max(0.02, min(0.06, price_std * 3.0)) * timeframe_factor
            velocity_threshold = 2.0 * timeframe_factor
            effort_threshold = 0.7 * timeframe_factor
            
            return {
                "vol_threshold": vol_threshold,
                "price_threshold": price_threshold,
                "velocity_threshold": velocity_threshold,
                "effort_threshold": effort_threshold
            }
        except Exception as e:
            logger.warning(f"Error calculating liquidation thresholds: {e}", exc_info=True)
            return {
                "vol_threshold": 2.5, 
                "price_threshold": 0.04,
                "velocity_threshold": 2.0,
                "effort_threshold": 0.7
            }
    
    @staticmethod
    def get_breakout_threshold(df: pd.DataFrame, timeframe: Timeframe) -> float:
        """Calculate dynamic threshold for breakout detection"""
        if df.empty or len(df) < 10:
            return 0.015  # Default fallback
        
        try:
            # Use ATR for volatility-aware threshold
            if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
                atr = df['ATR'].iloc[-1]
                price = df['c'].iloc[-1]
                volatility_factor = atr / price
            else:
                # Fallback to price standard deviation
                volatility_factor = df['c'].pct_change().rolling(10).std().mean()
            
            # Scale by timeframe
            timeframe_factor = {
                Timeframe.MINUTES_15: 0.8,
                Timeframe.MINUTES_30: 0.9,
                Timeframe.HOUR_1: 1.0,
                Timeframe.HOURS_2: 1.15,
                Timeframe.HOURS_4: 1.3,
                Timeframe.HOURS_8: 1.5
            }.get(timeframe, 1.0)
            
            # Calculate adaptive breakout threshold
            base_threshold = 0.015
            return max(0.01, base_threshold * (1 + volatility_factor * 5) * timeframe_factor)
        except Exception as e:
            logger.warning(f"Error calculating breakout threshold: {e}", exc_info=True)
            return 0.015
