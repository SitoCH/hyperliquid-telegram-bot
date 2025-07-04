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
        # Available timeframe categorization (15m, 30m, 1h, 4h only)
        medium_tf_price_moves = []  # 1h, 4h - for trend direction and major levels
        medium_tf_volume_ratios = []
        signal_tf_moves = []  # 15m, 30m - primary signal detection
        signal_tf_volume_ratios = []
        choppy_reversals = {}
        range_prison_detected = False
        range_prison_tf = None
        bb_expansion_info = None
        
        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
            # Calculate recent price changes
            current_price = df['c'].iloc[-1]
            price_5p = df['o'].iloc[-5] if len(df) >= 5 else df['o'].iloc[0]
            price_change = abs((current_price - price_5p) / price_5p * 100)
            price_changes.append(price_change)
            # Volume ratio check
            v_ratio = df['v_ratio'].iloc[-1] if 'v_ratio' in df.columns and not df['v_ratio'].empty else None
            if v_ratio is not None:
                volume_ratios.append(v_ratio)
            tf_str = str(tf).lower()
            
            # Timeframe classification with available timeframes only
            is_medium_tf = '1h' in tf_str or '4h' in tf_str  # Medium timeframes for direction
            is_signal_tf = '15m' in tf_str or '30m' in tf_str  # Primary signal timeframes
            
            # Medium timeframes (1h, 4h) - for trend direction and volume context
            if is_medium_tf:
                medium_tf_price_moves.append(price_change)
                if v_ratio is not None:
                    medium_tf_volume_ratios.append(v_ratio)
                # Choppy reversals in medium timeframes
                if len(df) >= 11:
                    reversals = 0
                    last_dir = None
                    for i in range(-10, 0):
                        diff = df['c'].iloc[i] - df['c'].iloc[i-1]
                        dir = 1 if diff > 0 else -1 if diff < 0 else 0
                        if last_dir is not None and dir != 0 and dir != last_dir:
                            reversals += 1
                        if dir != 0:
                            last_dir = dir
                    choppy_reversals[tf_str] = reversals
                # Range prison: check 4h for consolidation (since no 1d available)
                if '4h' in tf_str and len(df) >= 8:
                    min_p = df['c'].iloc[-8:].min()
                    max_p = df['c'].iloc[-8:].max()
                    if (max_p - min_p) / min_p * 100 < 0.4:  # 4h consolidation threshold
                        range_prison_detected = True
                        range_prison_tf = tf_str
            
            # Signal timeframes (15m, 30m) - primary signal detection
            if is_signal_tf:
                signal_tf_moves.append(price_change)
                if v_ratio is not None:
                    signal_tf_volume_ratios.append(v_ratio)
                # Choppy reversals in signal timeframes (more sensitive)
                if len(df) >= 11:
                    reversals = 0
                    last_dir = None
                    for i in range(-10, 0):
                        diff = df['c'].iloc[i] - df['c'].iloc[i-1]
                        dir = 1 if diff > 0 else -1 if diff < 0 else 0
                        if last_dir is not None and dir != 0 and dir != last_dir:
                            reversals += 1
                        if dir != 0:
                            last_dir = dir
                    choppy_reversals[tf_str] = max(choppy_reversals.get(tf_str, 0), reversals)
            
            # BB width expansion (for context-rich rejection) - check all timeframes
            if 'BB_width' in df.columns and len(df) >= 3:
                bb_width_now = df['BB_width'].iloc[-1]
                bb_width_prev = df['BB_width'].iloc[-3]
                if bb_width_prev > 0 and (bb_width_now - bb_width_prev) / bb_width_prev > 0.15 and price_change > 0.5:
                    bb_expansion_info = {
                        'tf': tf_str,
                        'bb_width_now': bb_width_now,
                        'bb_width_prev': bb_width_prev,
                        'price_move': price_change,
                        'v_ratio': v_ratio
                    }
        # Noise filters adapted to available timeframes (15m, 30m, 1h, 4h)
        
        # 4h consolidation: check for major range-bound conditions
        if range_prison_detected:
            return False, f"4h timeframe consolidation: Price within 0.4% range for 8+ periods in {range_prison_tf}. Major consolidation detected.", 0.3
        
        # Signal timeframe micro moves: if signal timeframes (15m, 30m) show no movement, likely noise
        if signal_tf_moves and all(x < 0.2 for x in signal_tf_moves):
            return False, f"Signal timeframe micro moves: Price changes <0.2% in ALL signal timeframes (15m, 30m). High risk of false intraday signal.", 0.3
        
        # Medium timeframe volume drought: 1h+ volume ratios too low
        medium_tf_vols = [v for v in medium_tf_volume_ratios if v is not None]
        if medium_tf_vols and all(x < 1.0 for x in medium_tf_vols):
            return False, f"Medium timeframe volume drought: Volume ratio <1.0x across 1h/4h timeframes. Insufficient momentum for sustained moves.", 0.3
        
        # Signal timeframe volume drought: 15m/30m need decent volume for reliable signals
        signal_tf_vols = [v for v in signal_tf_volume_ratios if v is not None]
        if signal_tf_vols and all(x < 0.8 for x in signal_tf_vols):
            return False, f"Signal timeframe volume drought: Volume ratio <0.8x in signal timeframes (15m, 30m). Insufficient volume for reliable signals.", 0.3
        
        # Choppy action: price reversals in key timeframes
        choppy_medium_tfs = [tf for tf, rev in choppy_reversals.items() if rev > 4 and ('1h' in tf or '4h' in tf)]
        choppy_signal_tfs = [tf for tf, rev in choppy_reversals.items() if rev > 3 and ('15m' in tf or '30m' in tf)]
        
        if choppy_medium_tfs:
            return False, f"Medium timeframe chop: Price reversals >4 times in 10 periods in timeframes: {', '.join(choppy_medium_tfs)}. Trend direction unclear.", 0.3
        
        if choppy_signal_tfs:
            return False, f"Signal timeframe chop: Price reversals >3 times in 10 periods in timeframes: {', '.join(choppy_signal_tfs)}. High noise in signal detection timeframes.", 0.4
        # Volatility expansion with volume drought (context-rich rejection)
        if bb_expansion_info and bb_expansion_info['v_ratio'] is not None and bb_expansion_info['v_ratio'] < 1.1:
            msg = (
                f"Volatility expansion with BB width {bb_expansion_info['bb_width_now']:.2f} (>15% from {bb_expansion_info['bb_width_prev']:.2f}), "
                f"price move {bb_expansion_info['price_move']:.2f}%, and volume drought (v_ratio {bb_expansion_info['v_ratio']:.2f}) detected in {bb_expansion_info['tf']} timeframe. "
                f"Noise filter triggered due to weak volume alignment, indicating high risk of false signal."
            )
            return False, msg, 0.4

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

        # Momentum detection adapted to available timeframes (15m, 30m, 1h, 4h)
        available_tf_momentum = any(
            # Signal timeframe momentum (15m, 30m) - primary for intraday trading
            (('15m' in str(tf).lower() or '30m' in str(tf).lower()) and 
             abs((df['c'].iloc[-1] - df['o'].iloc[-5]) / df['o'].iloc[-5] * 100) > 1.0 and
             'v_ratio' in df.columns and df['v_ratio'].iloc[-1] > 1.2)
            # Medium timeframe confirmation (1h, 4h) - for trend alignment
            or (('1h' in str(tf).lower()) and 
                abs((df['c'].iloc[-1] - df['o'].iloc[-5]) / df['o'].iloc[-5] * 100) > 0.8 and
                'v_ratio' in df.columns and df['v_ratio'].iloc[-1] > 1.3)
            # 4h timeframe for stronger moves (less frequent but more reliable)
            or (('4h' in str(tf).lower()) and 
                abs((df['c'].iloc[-1] - df['o'].iloc[-5]) / df['o'].iloc[-5] * 100) > 0.6 and
                'v_ratio' in df.columns and df['v_ratio'].iloc[-1] > 1.4)
            for tf, df in dataframes.items() if not df.empty and len(df) >= 5
        )
        
        if not available_tf_momentum:
            return False, (
                "No momentum detected in available timeframes. Signals require: "
                ">1.0% move in 15m/30m with volume >1.2x OR >0.8% in 1h with volume >1.3x OR "
                ">0.6% in 4h with volume >1.4x. Current data shows insufficient momentum for reliable opportunities."
            ), 0.2
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
        return f"""You are a balanced but selective signal filter for {coin}. Detect developing opportunities while avoiding noise and false signals.

Current Market Data:
{json.dumps(market_summary, indent=2)}

AVAILABLE TIMEFRAME APPROACH:
• Higher timeframes (4h): Trend direction and major support/resistance levels
• Medium timeframe (1h): Trend confirmation and momentum validation
• Signal timeframes (15m, 30m): PRIMARY signal detection and opportunity identification

SIGNAL DETECTION CRITERIA (balanced approach for early but quality signals):

🔥 HIGH PRIORITY SIGNALS (require 2+ conditions):
• Strong momentum: >1.0% in 15m/30m with volume >1.2x OR >0.8% in 1h with volume >1.3x OR >0.6% in 4h with volume >1.4x
• RSI extremes with momentum: RSI <30 or >70 in any timeframe AND price momentum alignment
• MACD acceleration: Histogram growing >2 periods in any timeframe AND volume >1.3x
• Level breaks: Price breaking key levels >0.6% in signal timeframes AND volume >1.4x
• Volatility expansion: BB width expanding >12% in any timeframe AND price move >0.4%

⚡ MEDIUM PRIORITY SIGNALS (require 2+ conditions):
• Decent moves: >0.8% in 15m/30m OR >0.6% in 1h OR >0.5% in 4h with volume >1.1x
• Technical alignment: RSI and MACD aligned in same direction with volume >1.1x
• Level approach: Price within 0.6% of key levels in any timeframe AND volume >1.2x
• Momentum build: ROC acceleration in 15m/30m/1h AND Stochastic signal AND volume >1.1x
• Funding + price: Rate >0.0002 AND price movement confirming direction

📈 LOW PRIORITY SIGNALS (require ALL 3 conditions):
• Building momentum: ROC acceleration AND Stochastic cross in 15m/30m AND volume >1.0x
• Multi-timeframe sync: Same signal across 2+ timeframes (15m, 30m, 1h)
• Technical setup: Multiple indicator alignment AND price momentum

NOISE FILTERS - SKIP when ANY present:
• Signal timeframe micro moves: Price changes <0.2% in ALL 15m/30m timeframes
• Medium timeframe volume drought: Volume ratio <1.0x in 1h/4h timeframes
• Signal timeframe volume drought: Volume ratio <0.8x in 15m/30m timeframes  
• Signal timeframe chop: Price reversals >3 times in 10 periods in 15m/30m
• Medium timeframe chop: Price reversals >4 times in 10 periods in 1h/4h
• 4h consolidation: Price within 0.4% range for 8+ periods in 4h timeframe
• Weak conviction: Price move without adequate volume for the timeframe

ANALYSIS DECISION LOGIC:
• ANALYZE: High priority (2+ conditions) OR Medium priority (2+ conditions) OR Low priority (ALL 3)
• SKIP: Any noise filter triggered. Better to miss than take low-probability setups
• Force analyze: Funding rate >0.0004 AND >1.2% price change in signal timeframes AND volume >1.6x
• Primary focus: 15m/30m signals with 1h confirmation - use 4h for trend context only
• Trend awareness: Use 4h for major trend direction but don't let it block good shorter-term signals

Confidence: Based on signal strength, volume confirmation, and timeframe alignment

Focus on capturing moves efficiently - catch momentum early with proper confirmation.

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
