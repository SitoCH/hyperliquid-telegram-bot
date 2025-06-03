import pandas as pd
import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from logging_utils import logger
from ..wyckoff.wyckoff_types import Timeframe
from .litellm_client import LiteLLMClient          

class AnalysisFilter:
    """Filter logic to determine when expensive AI analysis should be triggered."""
      # Lookback period for trend analysis
    TREND_LOOKBACK_PERIODS = 20
    
    async def should_run_llm_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, interactive: bool, funding_rates: List) -> Tuple[bool, str, float]:
        """Use a cheap LLM model to determine if expensive analysis is warranted."""

        always_run_filter = os.getenv("HTB_ALWAYS_RUN_LLM_FILTER", "False").lower() == "true"
        
        if interactive and not always_run_filter:
            return True, f"LLM filter triggered analysis for {coin}: interactive mode", 1.0

        market_summary = self._create_market_summary(dataframes, funding_rates)
        filter_prompt = self._create_filter_prompt(coin, market_summary)

        try:
            filter_client = LiteLLMClient()
            model = os.getenv("HTB_LLM_FAST_MODEL", "unknown")
            response = await filter_client.call_api(model, filter_prompt)
            
            should_analyze, reason, confidence = self._parse_filter_response(response)

            action = "triggered" if should_analyze else "skipped"
            logger.info(f"LLM filter {action} analysis for {coin}: {reason} (confidence: {confidence:.0%})")
            
            return should_analyze, reason, confidence
            
        except Exception as e:
            logger.error(f"LLM filter failed for {coin}: {str(e)}", exc_info=True)
            return False, "Fallback: LLM filter failed", 0.0


    def _create_market_summary(self, dataframes: Dict[Timeframe, pd.DataFrame], funding_rates: List) -> Dict[str, Any]:
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
          # Add funding rates summary
        if funding_rates:
            latest_funding = funding_rates[-1] if funding_rates else None
            avg_1h = sum(r.funding_rate for r in funding_rates[-1:]) / len(funding_rates[-1:]) if len(funding_rates) >= 1 else (latest_funding.funding_rate if latest_funding else 0)
            avg_4h = sum(r.funding_rate for r in funding_rates[-4:]) / len(funding_rates[-4:]) if len(funding_rates) >= 4 else (latest_funding.funding_rate if latest_funding else 0)
            avg_24h = sum(r.funding_rate for r in funding_rates[-24:]) / len(funding_rates[-24:]) if len(funding_rates) >= 24 else (latest_funding.funding_rate if latest_funding else 0)
            summary["funding"] = {
                "current_rate": latest_funding.funding_rate if latest_funding else 0,
                "1h_avg": avg_1h,
                "4h_avg": avg_4h,
                "24h_avg": avg_24h,
                "data_points": len(funding_rates)
            }
        else:
            summary["funding"] = {"current_rate": 0, "1h_avg": 0, "4h_avg": 0, "24h_avg": 0, "data_points": 0}
        
        return summary
    
    def _calculate_price_changes(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate price changes over multiple periods."""
        lookback_periods = [5, 10, self.TREND_LOOKBACK_PERIODS, 50] if len(df) >= 50 else [5, 10, min(self.TREND_LOOKBACK_PERIODS, len(df))]
        price_changes = {}
        
        for period in lookback_periods:
            if len(df) >= period:
                open_price = df['o'].iloc[-period]
                price_changes[f"{period}p"] = round((current_price - open_price) / open_price * 100, 2)
        
        return price_changes
    
    def _calculate_volatility(self, df: pd.DataFrame, current_price: float) -> float:
        """Calculate volatility over the analysis period."""
        high_period = df['h'].iloc[-self.TREND_LOOKBACK_PERIODS:].max() if len(df) >= self.TREND_LOOKBACK_PERIODS else df['h'].max()
        low_period = df['l'].iloc[-self.TREND_LOOKBACK_PERIODS:].min() if len(df) >= self.TREND_LOOKBACK_PERIODS else df['l'].min()
        return round((high_period - low_period) / current_price * 100, 2)
    
    def _analyze_volume_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume patterns and trends."""
        volume_data = {}
        
        if 'v_ratio' in df.columns and not df['v_ratio'].empty:
            volume_data['current_ratio'] = round(df['v_ratio'].iloc[-1], 2)
            volume_data['avg_5p'] = round(df['v_ratio'].iloc[-5:].mean(), 2)
            volume_data['max_trend_period'] = round(df['v_ratio'].iloc[-self.TREND_LOOKBACK_PERIODS:].max(), 2) if len(df) >= self.TREND_LOOKBACK_PERIODS else round(df['v_ratio'].max(), 2)
        
        if 'v_trend' in df.columns and not df['v_trend'].empty:
            volume_data['trend'] = round(df['v_trend'].iloc[-1], 2)
        
        return volume_data
    
    def _analyze_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pass through all available technical indicators with trend analysis values."""
        indicators = {}
        lookback = min(self.TREND_LOOKBACK_PERIODS, len(df))
        
        # Create list of all indicators to extract
        indicator_names = [
            'SuperTrend', 'EMA', 'VWAP',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ROC',
            'STOCH_K', 'STOCH_D',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'ATR',
            'TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B', 'CHIKOU'
        ]
        
        # Extract recent values for each indicator to show trends
        for indicator in indicator_names:
            if indicator in df.columns:
                values = df[indicator].iloc[-lookback:].tolist()
                indicators[indicator] = {
                    'current': values[-1] if values else None,
                    'values': values
                }
            else:
                indicators[indicator] = {
                    'current': None,
                    'values': []
                }
        
        return indicators    
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pass through support and resistance levels with recent values for trend analysis."""
        levels: Dict[str, Any] = {}
        lookback = min(self.TREND_LOOKBACK_PERIODS, len(df))
        
        # Support/resistance levels
        level_names = ['VWAP', 'PIVOT', 'R1', 'R2', 'S1', 'S2', 
                      'FIB_23', 'FIB_38', 'FIB_50', 'FIB_61', 'FIB_78']
        
        for level in level_names:
            if level in df.columns:
                values = df[level].iloc[-lookback:].tolist()
                levels[level.lower()] = {
                    'current': values[-1] if values else None,
                    'values': values
                }
            else:
                levels[level.lower()] = {
                    'current': None,
                    'values': []
                }
        
        return levels

    def _create_filter_prompt(self, coin: str, market_summary: Dict[str, Any]) -> str:
        """Create prompt for cheap LLM model to determine if expensive analysis is needed."""
        return f"""Your role is to act as a balanced filter for {coin} to determine if market conditions warrant expensive detailed technical analysis.

Current Market Data:
{json.dumps(market_summary, indent=2)}

ANALYSIS CRITERIA:
TRIGGER ANALYSIS when you see MANY of these actionable conditions, an analysis is expensive and should only be performed when warranted:
• Moderate price moves: >0.5% in short timeframes (5m-15m), >1.0% in medium timeframes (1h-4h), OR >2% in longer timeframes (12h+) 
• Volume confirmation: Volume ratio >1.2x average during price moves
• Technical indicators showing directional bias:
  - RSI: <35 (oversold) or >65 (overbought)
  - MACD: Histogram divergence, signal line crossovers, or momentum shifts
  - ROC: Rate of change indicating strong momentum
  - Stochastic: %K or %D in extreme zones (<20 or >80)
  - SuperTrend: Price breaking above/below SuperTrend line
  - Bollinger Bands: Price touching bands, squeeze/expansion patterns
  - ATR: Volatility spikes or compression patterns
• Moving averages and trend indicators:
  - EMA: Price breaking above/below key EMA levels
  - VWAP: Price deviating significantly from VWAP
  - Ichimoku: Cloud breakouts, Tenkan/Kijun crosses, or Chikou span signals
• Support/resistance proximity: Price within 2% of key levels (VWAP, Pivot points, Fibonacci levels)
• Cross-timeframe signals: Similar directional bias across 2+ timeframes
• Notable funding rates: Current rate >0.0003 or <-0.0003, OR significant divergence from 24h average
• Technical setups: Clear breakout patterns, reversal signals, or momentum shifts

REQUIRE AT LEAST 5-6 OF THESE CONDITIONS for higher confidence analysis.

SKIP ANALYSIS only when MULTIPLE weak conditions are present:
• Minimal activity: Price moves <0.8% across all recent periods
• Very low volume: All volume ratios <1.1x average  
• Completely neutral indicators: RSI 45-55, flat MACD, no clear direction
• No proximity to levels: Price >3% away from all key support/resistance
• Extreme low volatility: ATR suggesting range-bound action

Focus on catching developing opportunities rather than perfect setups.

CRITICAL: Response must be pure JSON only - no explanations or markdown.
Provide your analysis in the following JSON format:
{{
  "should_analyze": true/false,
  "reason": "Specific condition that triggered or prevented analysis with concrete numbers",
  "confidence": 0.0-1.0,
}}"""

    def _parse_filter_response(self, response: str) -> Tuple[bool, str, float]:
        """Parse the cheap LLM response to determine if analysis should proceed."""
        try:
            data = json.loads(response)
            should_analyze = data.get("should_analyze", False)
            reason = data.get("reason", "LLM filter decision")
            confidence = data.get("confidence", 0.5)

            if confidence < 0.7:
                should_analyze = False
                reason = f"Insufficient confidence for analysis. {reason}"
            
            return should_analyze, reason, confidence
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM filter response: {str(e)}\nResponse:\n{response}", exc_info=True)
            return False, "LLM filter parsing failed", 0.0
