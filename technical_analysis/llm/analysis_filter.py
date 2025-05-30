import pandas as pd
import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from logging_utils import logger
from ..wyckoff.wyckoff_types import Timeframe
from .openrouter_client import OpenRouterClient          

class AnalysisFilter:
    """Filter logic to determine when expensive AI analysis should be triggered."""

    def should_run_llm_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, interactive: bool) -> Tuple[bool, str]:
        """Use a cheap LLM model to determine if expensive analysis is warranted."""

        if interactive:
            return True, f"LLM filter triggered analysis for {coin}: interactive mode"

        market_summary = self._create_market_summary(dataframes)
        filter_prompt = self._create_filter_prompt(coin, market_summary)

        try:
            filter_client = OpenRouterClient()
            model = os.getenv("HTB_OPENROUTER_FAST_MODEL", "meta-llama/llama-4-maverick:free")
            response = filter_client.call_api(model, filter_prompt)
            
            should_analyze, reason = self._parse_filter_response(response)

            action = "triggered" if should_analyze else "skipped"
            logger.info(f"LLM filter {action} analysis for {coin}: {reason}")
            
            return should_analyze, reason
            
        except Exception as e:
            logger.error(f"LLM filter failed for {coin}: {str(e)}", exc_info=True)
            return False, "Fallback: LLM filter failed"


    def _create_market_summary(self, dataframes: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """Create a comprehensive market summary for cheap LLM filtering."""
        summary: Dict[str, Dict[str, Any]] = {"timeframes": {}}
        
        for tf, df in dataframes.items():
            if df.empty or len(df) < 10:
                continue
                
            current_price = df['c'].iloc[-1]
            timeframe_data = {
                "price_changes": self._calculate_price_changes(df, current_price),
                "volatility": self._calculate_volatility(df, current_price),
                "volume": self._analyze_volume_data(df),
                "indicators": self._analyze_all_indicators(df),
                "levels": self._analyze_support_resistance(df),
                "candles_analyzed": len(df)
            }
            
            summary["timeframes"][str(tf)] = timeframe_data
        
        return summary
    
    def _calculate_price_changes(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate price changes over multiple periods."""
        lookback_periods = [5, 10, 20, 50] if len(df) >= 50 else [5, 10, min(20, len(df))]
        price_changes = {}
        
        for period in lookback_periods:
            if len(df) >= period:
                open_price = df['o'].iloc[-period]
                price_changes[f"{period}p"] = round((current_price - open_price) / open_price * 100, 2)
        
        return price_changes
    
    def _calculate_volatility(self, df: pd.DataFrame, current_price: float) -> float:
        """Calculate volatility over the analysis period."""
        high_20 = df['h'].iloc[-20:].max() if len(df) >= 20 else df['h'].max()
        low_20 = df['l'].iloc[-20:].min() if len(df) >= 20 else df['l'].min()
        return round((high_20 - low_20) / current_price * 100, 2)
    
    def _analyze_volume_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume patterns and trends."""
        volume_data = {}
        
        if 'v_ratio' in df.columns and not df['v_ratio'].empty:
            volume_data['current_ratio'] = round(df['v_ratio'].iloc[-1], 2)
            volume_data['avg_5p'] = round(df['v_ratio'].iloc[-5:].mean(), 2)
            volume_data['max_20p'] = round(df['v_ratio'].iloc[-20:].max(), 2) if len(df) >= 20 else round(df['v_ratio'].max(), 2)
        
        if 'v_trend' in df.columns and not df['v_trend'].empty:
            volume_data['trend'] = round(df['v_trend'].iloc[-1], 2)
        
        return volume_data
    
    def _analyze_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pass through all available technical indicators."""
        indicators = {}
        last_idx = -1
        
        # Create list of all indicators to extract
        indicator_names = [
            'SuperTrend', 'EMA', 'VWAP',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ROC',
            'STOCH_K', 'STOCH_D',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'ATR',
            'TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B', 'CHIKOU'
        ]
        
        # Extract all indicator values
        for indicator in indicator_names:
            indicators[indicator] = df[indicator].iloc[last_idx] if indicator in df.columns else None
        
        return indicators    
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Pass through support and resistance levels."""
        levels: Dict[str, float] = {}
        last_idx = -1
        
        # Support/resistance levels
        level_names = ['VWAP', 'PIVOT', 'R1', 'R2', 'S1', 'S2', 
                      'FIB_23', 'FIB_38', 'FIB_50', 'FIB_61', 'FIB_78']
        
        for level in level_names:
            levels[level.lower()] = df[level].iloc[last_idx]
        
        return levels

    def _create_filter_prompt(self, coin: str, market_summary: Dict[str, Any]) -> str:
        """Create prompt for cheap LLM model to determine if expensive analysis is needed."""
        return f"""Your role is to act as a balanced filter for {coin} to determine if market conditions warrant expensive detailed technical analysis. Use these guidelines to make informed decisions.

Current Market Data:
{json.dumps(market_summary, indent=2)}

ANALYSIS CRITERIA:
TRIGGER ANALYSIS when you see:
• Significant price moves: >2.5% in short timeframes (5m-1h) OR >4% in longer timeframes (4h+) 
• Volume confirmation: Volume ratio >1.4x average during price moves
• Clear directional indicators: RSI <35 or >65, MACD divergence, trend changes
• Support/resistance tests: Price approaching key levels with momentum
• Cross-timeframe alignment: Similar signals across multiple timeframes

SKIP ANALYSIS when you see:
• Minimal activity: Price moves <1.5% across all recent periods
• Low volume: All volume ratios <1.2x average
• Neutral indicators: RSI 40-60, flat MACD, mixed signals
• No clear direction: Conflicting signals across timeframes
• Sideways action: Very low volatility with no breakout potential

FOCUS ON ACTIONABLE SITUATIONS:
Look for developing patterns, momentum shifts, or technical setups that could provide trading opportunities. Balance between catching opportunities and avoiding noise.

BE SPECIFIC in your reasoning - avoid generic phrases like "multiple timeframes show notable moves". Instead, cite specific price percentages, volume ratios, or indicator readings.

Provide your analysis in JSON format:
{{
  "should_analyze": true/false,
  "reason": "A specific sentence explaining the key condition that triggered or prevented analysis with concrete numbers",
  "confidence": 0.0-1.0,
}}"""

    def _parse_filter_response(self, response: str) -> Tuple[bool, str]:
        """Parse the cheap LLM response to determine if analysis should proceed."""
        try:
            import json
            data = json.loads(response)
            should_analyze = data.get("should_analyze", False)
            reason = data.get("reason", "LLM filter decision")
            confidence = data.get("confidence", 0.5)

            if confidence < 0.7:
                should_analyze = False
                reason = f"Insufficient confidence for analysis. {reason}"

            detailed_reason = f"{reason} (confidence: {confidence:.0%})"
            
            return should_analyze, detailed_reason
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM filter response: {str(e)}", exc_info=True)
            return False, "LLM filter parsing failed"
