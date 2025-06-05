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

        # Pre-filter: Quick noise detection before expensive LLM call
        if not self._passes_pre_filter(dataframes, funding_rates):
            return False, f"Pre-filter rejected {coin}: insufficient activity or conflicting signals", 0.0

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

    def _passes_pre_filter(self, dataframes: Dict[Timeframe, pd.DataFrame], funding_rates: List) -> bool:
        """Quick pre-filter to catch obvious noise before LLM analysis."""
        
        # Check if we have any meaningful data
        if not dataframes or all(df.empty or len(df) < 10 for df in dataframes.values()):
            return False
        
        # Aggregate checks across all timeframes
        price_changes = []
        volume_ratios = []
        
        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
            
            # Calculate recent price changes
            current_price = df['c'].iloc[-1]
            price_5p = df['o'].iloc[-5] if len(df) >= 5 else df['o'].iloc[0]
            price_change = abs((current_price - price_5p) / price_5p * 100)
            price_changes.append(price_change)
            
            # Volume ratio check
            if 'v_ratio' in df.columns and not df['v_ratio'].empty:
                volume_ratios.append(df['v_ratio'].iloc[-1])
          # Noise filters - fail if ANY are triggered
        if price_changes:
            max_price_change = max(price_changes)
            avg_price_change = sum(price_changes) / len(price_changes)
            
            # Dead market: all price changes are tiny (increased threshold)
            if max_price_change < 0.35:
                return False
            
            # Low activity: average price change is minimal (increased threshold)
            if avg_price_change < 0.25:
                return False
        
        if volume_ratios:
            max_volume = max(volume_ratios)
            avg_volume = sum(volume_ratios) / len(volume_ratios)
            
            # Volume drought: no timeframe has decent volume (increased threshold)
            if max_volume < 1.15:
                return False
            
            # Weak volume across the board (increased threshold)
            if avg_volume < 1.08:
                return False
        
        # Check funding rates for extreme signals
        if funding_rates:
            latest_funding = funding_rates[-1].funding_rate if funding_rates else 0
            # Only bypass other filters if funding is extremely high
            if abs(latest_funding) > 0.0008:
                return True
        
        return True

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
        return f"""You are a balanced signal filter for {coin}. Detect developing opportunities while avoiding noise and false signals.

Current Market Data:
{json.dumps(market_summary, indent=2)}

SIGNAL DETECTION CRITERIA (require MULTIPLE confirmations):

🔥 HIGH PRIORITY SIGNALS (require 2+ conditions):
• Extreme price moves: >1.2% in any timeframe with volume ratio >1.5x
• RSI extremes with confirmation: RSI <20 or >80 AND price momentum in same direction
• MACD strong momentum: Histogram acceleration >2 periods AND volume >1.4x
• Major level breaks: Price breaking key levels with >1.0% move AND volume >1.5x

⚡ MEDIUM PRIORITY SIGNALS (require 2+ conditions):
• Strong moves: >0.8% price change AND volume ratio >1.3x
• Technical convergence: RSI, MACD, and price all aligned with volume >1.2x
• Key level interaction: Price within 0.5% of levels AND volume spike >1.4x
• Volatility breakout: BB width expanding >20% AND price move >0.6%
• Funding extremes: Rate >0.0004 AND price movement confirming direction

📈 LOW PRIORITY SIGNALS (require ALL 3 conditions):
• Momentum building: ROC acceleration AND Stochastic signal AND volume >1.15x
• Multi-timeframe alignment: Same signal across 3+ timeframes AND volume confirmation
• Ichimoku convergence: Multiple component signals AND price momentum AND volume

STRICT NOISE FILTERS - SKIP when ANY are present:
• Micro moves: ALL price changes <0.3% across all timeframes
• Volume drought: Average volume ratio <1.1x across ALL timeframes
• Choppy action: Price reversals >3 times in 10 periods
• Weak conviction: Price change >0.8% but volume <1.2x (fake moves)
• Range prison: Price within 0.3% range for 8+ periods
• Signal conflict: RSI overbought but MACD bullish (or vice versa) without strong volume
• Dead zone: ALL indicators flat (change <5% in 10 periods) AND volume <1.1x

ANALYSIS DECISION LOGIC:
• ANALYZE: High priority (2+ conditions) OR Medium priority (2+ conditions) OR Low priority (ALL 3 conditions)
• SKIP: Any noise filter triggered (no exceptions for weak signals)
• Force analyze ONLY if: funding rate >0.0005 AND price change >1.5% AND volume >1.8x

Confidence: Based on signal strength and confirmation quality

Prefer quality signals with multiple confirmations over weak single indicators.

Response must be pure JSON - no markdown, no explanations:
{{
  "should_analyze": true/false,
  "reason": "Specific conditions met with concrete data points",
  "confidence": 0.0-1.0
}}"""

    def _parse_filter_response(self, response: str) -> Tuple[bool, str, float]:
        """Parse the cheap LLM response to determine if analysis should proceed."""
        try:
            data = json.loads(response)
            should_analyze = data.get("should_analyze", False)
            reason = data.get("reason", "LLM filter decision")
            confidence = data.get("confidence", 0.5)
            
            if confidence < 0.85:
                should_analyze = False
                reason = f"Insufficient confidence ({confidence:.0%}) for analysis. {reason}"
            
            return should_analyze, reason, confidence
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM filter response: {str(e)}\nResponse:\n{response}", exc_info=True)
            return False, "LLM filter parsing failed", 0.0
