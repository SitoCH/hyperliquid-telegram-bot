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
            return True, f"Pre-filter approved analysis for {coin}: interactive mode", 1.0
        
        passes_filter, filter_reason, filter_confidence = self._passes_pre_filter(dataframes, funding_rates)
        if not passes_filter:
            pre_filter_message = f"Pre-filter rejected analysis for {coin}: {filter_reason}"
            logger.info(pre_filter_message)
            return False, pre_filter_message, filter_confidence

        market_summary = self._create_market_summary(dataframes, funding_rates)
        filter_prompt = self._create_filter_prompt(coin, market_summary)

        try:
            filter_client = LiteLLMClient()
            model = os.getenv("HTB_LLM_FAST_MODEL", "unknown")
            response = await filter_client.call_api(model, filter_prompt)
            should_analyze, reason, confidence = self._parse_filter_response(response)

            action = "approved" if should_analyze else "rejected"
            logger.info(f"LLM filter {action} analysis for {coin}: {reason} (confidence: {confidence:.0%})")
            return should_analyze, reason, confidence
            
        except Exception as e:
            logger.error(f"LLM filter failed for {coin}: {str(e)}", exc_info=True)
            return False, "Fallback: LLM filter failed", 0.0

    def _passes_pre_filter(self, dataframes: Dict[Timeframe, pd.DataFrame], funding_rates: List) -> Tuple[bool, str, float]:
        """Quick pre-filter to catch obvious noise before LLM analysis, including market change and funding checks."""
        # Check if we have any meaningful data
        if not dataframes or all(df.empty or len(df) < 10 for df in dataframes.values()):
            return False, "Insufficient data - no meaningful dataframes available", 0.0

        # Check for extreme funding conditions that warrant immediate analysis
        if funding_rates:
            latest_funding = funding_rates[-1] if funding_rates else None
            if latest_funding and abs(latest_funding.funding_rate) > 0.0008:  # 0.08% threshold
                return True, f"Emergency bypass - extreme funding rate detected: {latest_funding.funding_rate:.6f}", 1.0

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

        # Emergency bypass for extreme price movements (likely liquidation events)
        if price_changes and max(price_changes) > 3.0:
            return True, f"Emergency bypass - extreme price movement detected: {max(price_changes):.2f}%", 1.0

        # Noise filters - fail if ANY are triggered
        if price_changes:
            max_price_change = max(price_changes)
            avg_price_change = sum(price_changes) / len(price_changes)
            # Dead market: all price changes are tiny - more lenient for catching early moves
            if max_price_change < 0.25:
                return False, f"Dead market - max price change {max_price_change:.2f}% < 0.25%", 0.0
            # Low activity: average price change is minimal - reduced for early signal detection
            if avg_price_change < 0.15:
                return False, f"Low activity - avg price change {avg_price_change:.2f}% < 0.15%", 0.0
        if volume_ratios:
            max_volume = max(volume_ratios)
            avg_volume = sum(volume_ratios) / len(volume_ratios)
            significant_move = price_changes and max(price_changes) > 1.0
            moderate_move = price_changes and max(price_changes) > 0.5
            if significant_move:
                if max_volume < 0.45:
                    return False, f"Extreme volume drought during significant move - max volume {max_volume:.2f} < 0.45", 0.0
            elif moderate_move:
                if max_volume < 0.65:
                    return False, f"Low volume during moderate move - max volume {max_volume:.2f} < 0.65", 0.0
            else:
                if max_volume < 0.85:
                    return False, f"Volume drought - max volume {max_volume:.2f} < 0.85", 0.0
                if avg_volume < 0.70:
                    return False, f"Weak volume - avg volume {avg_volume:.2f} < 0.70", 0.0

        # Check if market conditions have changed significantly since last few periods
        if not self._has_significant_market_change(dataframes):
            return False, "No significant market change detected", 0.0

        return True, "Pre-filter passed", 1.0

    def _has_significant_market_change(self, dataframes: Dict[Timeframe, pd.DataFrame]) -> bool:
        """Check if market conditions have changed significantly using dataframe history."""
        
        for tf, df in dataframes.items():
            if df.empty or len(df) < 10:
                continue
            
            # Compare current state with 5 periods ago to detect meaningful changes
            current_idx = -1
            previous_idx = -6 if len(df) >= 6 else -len(df)
            
            # Price change analysis
            current_price = df['c'].iloc[current_idx]
            previous_price = df['c'].iloc[previous_idx]
            price_change = abs((current_price - previous_price) / previous_price * 100)
            
            # Volume change analysis
            if 'v_ratio' in df.columns and not df['v_ratio'].empty:
                current_volume = df['v_ratio'].iloc[current_idx]
                previous_volume = df['v_ratio'].iloc[previous_idx]
                volume_change = abs(current_volume - previous_volume)
                
                # Significant change if volume ratio changed by >0.5 AND price moved >0.5%
                if volume_change > 0.5 and price_change > 0.5:
                    return True
            
            # RSI momentum shift detection
            if 'RSI' in df.columns and not df['RSI'].empty and len(df) >= 6:
                current_rsi = df['RSI'].iloc[current_idx]
                previous_rsi = df['RSI'].iloc[previous_idx]
                if not (pd.isna(current_rsi) or pd.isna(previous_rsi)):
                    rsi_change = abs(current_rsi - previous_rsi)
                    
                    # Significant RSI shift (>10 points) suggests regime change
                    if rsi_change > 10:
                        return True
                    
                    # RSI crossing key levels (30/70) with price movement
                    rsi_crossing_oversold = previous_rsi < 30 and current_rsi > 30
                    rsi_crossing_overbought = previous_rsi > 70 and current_rsi < 70
                    if (rsi_crossing_oversold or rsi_crossing_overbought) and price_change > 0.8:
                        return True
            
            # MACD momentum change detection
            if 'MACD_Hist' in df.columns and not df['MACD_Hist'].empty and len(df) >= 6:
                current_hist = df['MACD_Hist'].iloc[current_idx]
                previous_hist = df['MACD_Hist'].iloc[previous_idx]
                if not (pd.isna(current_hist) or pd.isna(previous_hist)):
                    # MACD histogram sign change (momentum reversal)
                    if (current_hist > 0 and previous_hist < 0) or (current_hist < 0 and previous_hist > 0):
                        return True
            
            # SuperTrend change (trend direction shift)
            if 'SuperTrend' in df.columns and not df['SuperTrend'].empty and len(df) >= 6:
                current_st = df['SuperTrend'].iloc[current_idx]
                previous_st = df['SuperTrend'].iloc[previous_idx]
                if not (pd.isna(current_st) or pd.isna(previous_st)):
                    # Price relationship to SuperTrend changed
                    was_above = df['c'].iloc[previous_idx] > previous_st
                    is_above = current_price > current_st
                    if was_above != is_above:  # Trend direction changed
                        return True
            
            # Volatility expansion detection
            current_volatility = (df['h'].iloc[current_idx] - df['l'].iloc[current_idx]) / current_price
            previous_volatility = (df['h'].iloc[previous_idx] - df['l'].iloc[previous_idx]) / df['c'].iloc[previous_idx]
            volatility_expansion = current_volatility / previous_volatility if previous_volatility > 0 else 1
            
            # Significant volatility expansion (>50% increase) suggests new market activity
            if volatility_expansion > 1.5 and price_change > 0.5:
                return True
        
        # No significant changes detected across any timeframe
        return False

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
        return f"""You are a balanced but highly selective signal filter for {coin}. Detect developing opportunities while avoiding noise and false signals.

Current Market Data:
{json.dumps(market_summary, indent=2)}

TIMEFRAME BALANCE APPROACH:
• Higher timeframes (4h, 1d): Provide trend context and filter direction - prevent counter-trend trades
• Medium timeframes (1h, 15m): Primary signal detection - catch momentum shifts early
• Lower timeframes (5m, 1m): Entry timing and momentum confirmation - validate breakouts

SIGNAL DETECTION CRITERIA (balanced approach for early but quality signals):

🔥 HIGH PRIORITY SIGNALS (require 2+ conditions, early momentum focus):
• Strong momentum: >1.5% in 15m/1h OR >0.8% in 4h with volume ratio >1.5x
• RSI extremes with momentum: RSI <25 or >75 in any timeframe AND price momentum alignment
• MACD acceleration: Histogram growing >2 periods in 15m+ AND volume >1.4x
• Level breaks with volume: Price breaking key levels >0.8% in any timeframe AND volume >1.5x
• Volatility expansion: BB width expanding >15% in 15m+ AND price move >0.5%

⚡ MEDIUM PRIORITY SIGNALS (require 2+ conditions, momentum + confirmation):
• Decent moves: >1.0% in 15m/1h OR >0.6% in 4h with volume ratio >1.3x
• Technical alignment: RSI and MACD aligned in same direction in 15m+ with volume >1.2x
• Key level approach: Price within 0.8% of levels in any timeframe AND volume spike >1.3x
• Trend acceleration: ROC increasing in 15m+ AND Stochastic signal AND volume >1.2x
• Funding pressure: Rate >0.0003 AND price movement confirming direction

📈 LOW PRIORITY SIGNALS (require ALL 3 conditions, early detection):
• Building momentum: ROC acceleration AND Stochastic cross in 15m+ AND volume >1.15x
• Multi-timeframe sync: Same signal across 2+ timeframes (don't require 4h+ signal)
• Ichimoku setup: Multiple component alignment in 15m+ AND price momentum AND volume

STRICT NOISE FILTERS - SKIP when ANY are present:
• Micro moves: Price changes <0.3% in ALL 4h+ timeframes (ignore lower timeframe noise)
• Volume drought: Average volume ratio <1.1x across 1h+ timeframes
• Choppy action: Price reversals >3 times in 10 periods in main timeframes (1h+)
• Weak conviction: Price change >0.8% in 4h+ but volume <1.2x (fake moves)
• Range prison: Price within 0.3% range for 8+ periods in 4h+ timeframes
• Signal conflict: RSI overbought but MACD bullish in same timeframe without strong volume
• Dead zone: ALL indicators flat in 1h+ timeframes AND volume <1.1x

ANALYSIS DECISION LOGIC:
• ANALYZE: High priority (2+ conditions) OR Medium priority (2+ conditions) OR Low priority (ALL 3 conditions)
• SKIP: Any noise filter triggered (no exceptions for weak signals). It is better to SKIP than to approve a low-probability or ambiguous setup.
• Force analyze ONLY if: funding rate >0.0005 AND price change >1.5% in 4h+ AND volume >1.8x
• Early signals: Allow 15m/1h signals to trigger analysis if volume and momentum are strong
• Trend filter: Use 4h/1d only to avoid obvious counter-trend trades, not to block all signals

Confidence: Based on signal strength, momentum quality, and volume confirmation

Balance early signal detection with trend awareness - catch moves early but avoid fighting major trends.

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
            
            min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
            if confidence < min_confidence:
                should_analyze = False
                reason = f"Insufficient confidence ({confidence:.0%}) for analysis. {reason}"
            
            return should_analyze, reason, confidence
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM filter response: {str(e)}\nResponse:\n{response}", exc_info=True)
            return False, "LLM filter parsing failed", 0.0
