import pandas as pd
import json
import os
from typing import Dict, Any, List, Tuple, Optional
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
        action = "approved" if passes_filter else "rejected"
        pre_filter_message = f"Pre-filter {action} analysis for {coin}: {filter_reason}"
        logger.info(pre_filter_message)

        if not passes_filter:
            return False, pre_filter_message, filter_confidence

        market_summary = self._create_market_summary(dataframes, funding_rates)
        filter_prompt = self._create_filter_prompt(coin, market_summary)

        try:
            filter_client = LiteLLMClient()
            model = os.getenv("HTB_LLM_FAST_MODEL", "unknown")
            response = await filter_client.call_api(model, filter_prompt, 'minimal')
            should_analyze, reason, confidence = self._parse_filter_response(response)

            action = "approved" if should_analyze else "rejected"
            logger.info(f"LLM filter {action} analysis for {coin}: {reason} (confidence: {confidence:.0%})")
            return should_analyze, reason, confidence
            
        except Exception as e:
            logger.error(f"LLM filter failed for {coin}: {str(e)}", exc_info=True)
            return False, "Fallback: LLM filter failed", 0.0

    def _passes_pre_filter(self, dataframes: Dict[Timeframe, pd.DataFrame], funding_rates: List) -> Tuple[bool, str, float]:
        """Quick pre-filter to catch obvious noise before LLM analysis.

        Simplified structure: early guards, metric collection, then a linear set of
        well-named checks. Thresholds and messages preserved, but code paths are
        easier to follow and duplicate logic is reduced.
        """

        # -----------------------------
        # Tiny local helpers
        # -----------------------------
        def tf_str(tf: Any) -> str:
            return str(tf).lower()

        def is_medium(tf_s: str) -> bool:
            return ('1h' in tf_s) or ('4h' in tf_s)

        def is_signal(tf_s: str) -> bool:
            return ('15m' in tf_s) or ('30m' in tf_s)

        def last_v_ratio(df: pd.DataFrame) -> Optional[float]:
            return float(df['v_ratio'].iloc[-1]) if 'v_ratio' in df.columns and not df['v_ratio'].empty else None

        def count_reversals(df: pd.DataFrame, lookback: int = 10) -> int:
            if len(df) < lookback + 1:
                return 0
            rev = 0
            last_dir = None
            for i in range(-lookback, 0):
                diff = df['c'].iloc[i] - df['c'].iloc[i - 1]
                ddir = 1 if diff > 0 else -1 if diff < 0 else 0
                if last_dir is not None and ddir != 0 and ddir != last_dir:
                    rev += 1
                if ddir != 0:
                    last_dir = ddir
            return rev

        # 1) Basic data sufficiency
        if not dataframes or all(df.empty or len(df) < 10 for df in dataframes.values()):
            return False, "Insufficient data - no meaningful dataframes available", 0.0

        # 2) Extreme funding emergency bypass
        latest_funding = funding_rates[-1] if funding_rates else None
        if latest_funding and abs(latest_funding.funding_rate) > 0.0008:  # 0.08%
            return True, f"Emergency bypass - extreme funding rate detected: {latest_funding.funding_rate:.6f}", 1.0

        # 3) Collect per-timeframe metrics
        price_changes: List[float] = []            # absolute
        signed_price_changes: List[float] = []     # signed
        all_v_ratios: List[float] = []
        medium_v_ratios: List[float] = []
        signal_abs_moves: List[float] = []
        signal_v_ratios: List[float] = []
        choppy_reversals: Dict[str, int] = {}
        chop_allowance: Dict[str, int] = {}
        range_prison_detected = False
        range_prison_tf: Optional[str] = None
        bb_expansion_info: Optional[Dict[str, Any]] = None
        signed_moves_by_tf: Dict[str, float] = {}

        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
            tf_s = tf_str(tf)

            # Price move over 5 periods
            o_5 = df['o'].iloc[-5] if len(df) >= 5 else df['o'].iloc[0]
            c_now = df['c'].iloc[-1]
            signed_move = (c_now - o_5) / o_5 * 100
            abs_move = abs(signed_move)
            price_changes.append(abs_move)
            signed_price_changes.append(signed_move)
            signed_moves_by_tf[tf_s] = signed_move

            # Volume ratio
            v_ratio = last_v_ratio(df)
            if v_ratio is not None:
                all_v_ratios.append(v_ratio)

            # Medium TF specifics
            if is_medium(tf_s):
                if v_ratio is not None:
                    medium_v_ratios.append(v_ratio)
                choppy_reversals[tf_s] = count_reversals(df, 10)
                chop_allowance[tf_s] = self._dynamic_chop_allowance(df, base_allowable=5)
                if '4h' in tf_s and len(df) >= 8:
                    seg = df['c'].iloc[-8:]
                    if (seg.max() - seg.min()) / seg.min() * 100 < 0.4:
                        range_prison_detected = True
                        range_prison_tf = tf_s

            # Signal TF specifics
            if is_signal(tf_s):
                signal_abs_moves.append(abs_move)
                if v_ratio is not None:
                    signal_v_ratios.append(v_ratio)
                # Merge with any prior (if also counted above)
                choppy_reversals[tf_s] = max(choppy_reversals.get(tf_s, 0), count_reversals(df, 10))
                chop_allowance[tf_s] = self._dynamic_chop_allowance(df, base_allowable=4)

            # BB width expansion
            if 'BB_width' in df.columns and len(df) >= 3:
                bb_now = df['BB_width'].iloc[-1]
                bb_prev = df['BB_width'].iloc[-3]
                if bb_prev > 0 and (bb_now - bb_prev) / bb_prev > 0.15 and abs_move > 0.5:
                    bb_expansion_info = {
                        'tf': tf_s,
                        'bb_width_now': bb_now,
                        'bb_width_prev': bb_prev,
                        'price_move': abs_move,
                        'v_ratio': v_ratio,
                    }

        # 4) Directional context
        largest_down = min(signed_price_changes) if signed_price_changes else 0.0

        # Primary timeframe bearish bias (30m / 1h) for threshold easing
        p30 = signed_moves_by_tf.get('30m', 0.0)
        p1h = signed_moves_by_tf.get('1h', 0.0)
        bearish_primary_fast = (p30 <= -0.9) or (p1h <= -0.9)
        bearish_primary_moderate = (p30 <= -0.6) or (p1h <= -0.6)

        # Mean-reversion carve-out
        mean_rev_reason = self._detect_mean_reversion_opportunity(dataframes)
        if mean_rev_reason:
            return True, mean_rev_reason, 0.8

        # 5) Structural / noise filters
        if range_prison_detected:
            # Only hard-reject if signal TFs also show weak momentum and low volume
            weak_signal_moves = (signal_abs_moves and all(x < 0.4 for x in signal_abs_moves))
            weak_signal_volume = (signal_v_ratios and all(x < 0.9 for x in signal_v_ratios))
            if weak_signal_moves and weak_signal_volume:
                return False, (
                    f"4h consolidation + weak signals: Tight 4h range in {range_prison_tf} and muted 15m/30m momentum+volume."
                ), 0.3

        if signal_abs_moves and all(x < 0.2 for x in signal_abs_moves):
            return False, (
                "Signal timeframe micro moves: Price changes <0.2% in ALL signal timeframes (15m, 30m). High risk of false intraday signal."
            ), 0.3

        if medium_v_ratios:
            medium_drought_th = 0.80 if (bearish_primary_fast or bearish_primary_moderate) else 0.95
            if all(x < medium_drought_th for x in medium_v_ratios):
                vals = ", ".join([f"{x:.2f}x" for x in medium_v_ratios])
                return False, (
                    f"Medium timeframe volume drought: Volume ratio <{medium_drought_th:.2f}x across 1h/4h timeframes (current: {vals}). Insufficient momentum for sustained moves."
                ), 0.3

        if signal_v_ratios:
            signal_drought_th = 0.70 if (bearish_primary_fast or bearish_primary_moderate) else 0.85
            if all(x < signal_drought_th for x in signal_v_ratios):
                svals = ", ".join([f"{x:.2f}x" for x in signal_v_ratios])
                return False, (
                    f"Signal timeframe volume drought: Volume ratio <{signal_drought_th:.2f}x in 15m/30m (current: {svals}). Insufficient volume for reliable signals."
                ), 0.3

        choppy_medium_tfs = [tf for tf, rev in choppy_reversals.items() if (('1h' in tf) or ('4h' in tf)) and rev > chop_allowance.get(tf, 5)]
        if choppy_medium_tfs:
            details = ", ".join([f"{tf} (> {chop_allowance.get(tf, 5)})" for tf in choppy_medium_tfs])
            return False, f"Medium timeframe chop: Price reversals exceed dynamic allowance in: {details}. Trend direction unclear.", 0.3

        choppy_signal_tfs = [tf for tf, rev in choppy_reversals.items() if (('15m' in tf) or ('30m' in tf)) and rev > chop_allowance.get(tf, 4)]
        if choppy_signal_tfs:
            details = ", ".join([f"{tf} (> {chop_allowance.get(tf, 4)})" for tf in choppy_signal_tfs])
            return False, f"Signal timeframe chop: Price reversals exceed dynamic allowance in: {details}. High noise in signal detection timeframes.", 0.4

        if bb_expansion_info and bb_expansion_info['v_ratio'] is not None and bb_expansion_info['v_ratio'] < 1.1:
            # Only block if signal momentum is weak; otherwise allow potential breakdowns to proceed
            weak_signal = (signal_abs_moves and max(signal_abs_moves) < 0.5)
            if weak_signal:
                return False, (
                    f"Volatility expansion with low volume and weak signal momentum (BB width {bb_expansion_info['bb_width_now']:.2f}, v_ratio {bb_expansion_info['v_ratio']:.2f})."
                ), 0.4

        # Activity checks
        if price_changes:
            max_move = max(price_changes)
            avg_move = sum(price_changes) / len(price_changes)
            if max_move < 0.25:
                return False, f"Dead market - max price change {max_move:.2f}% < 0.25%", 0.0
            if avg_move < 0.15:
                return False, f"Low activity - avg price change {avg_move:.2f}% < 0.15%", 0.0

        # Volume quality relative to move magnitude and direction
        if all_v_ratios:
            max_volume = max(all_v_ratios)
            avg_volume = sum(all_v_ratios) / len(all_v_ratios)
            significant_abs = price_changes and max(price_changes) > 1.0
            moderate_abs = price_changes and max(price_changes) > 0.5
            significant_bearish = largest_down <= -1.0
            moderate_bearish = largest_down <= -0.5

            if significant_abs:
                drought_th = 0.35 if significant_bearish else 0.45
                if max_volume < drought_th:
                    return False, f"Extreme volume drought during significant move - max volume {max_volume:.2f} < {drought_th:.2f}", 0.0
            elif moderate_abs:
                drought_th = 0.50 if moderate_bearish else 0.65
                if max_volume < drought_th:
                    return False, f"Low volume during moderate move - max volume {max_volume:.2f} < {drought_th:.2f}", 0.0
            else:
                base_th = 0.70 if (bearish_primary_fast or bearish_primary_moderate) else 0.85
                if max_volume < base_th:
                    return False, f"Volume drought - max volume {max_volume:.2f} < {base_th:.2f}", 0.0
                avg_min = 0.60 if (bearish_primary_fast or bearish_primary_moderate) else 0.70
                if avg_volume < avg_min:
                    return False, f"Weak volume - avg volume {avg_volume:.2f} below minimum {avg_min:.2f}", 0.0

        if not self._has_significant_market_change(dataframes):
            return False, "No significant market change detected", 0.0

        # Core momentum check aligned with main framework
        has_tier1_momentum = False
        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
            label = tf_str(tf)
            move_pct = (df['c'].iloc[-1] - df['o'].iloc[-5]) / df['o'].iloc[-5] * 100
            v_opt = last_v_ratio(df)
            v_val = v_opt if v_opt is not None else 1.0  # Treat missing volume as neutral instead of zero
            sig = is_signal(label)
            on1h = '1h' in label
            on4h = '4h' in label

            # Tier 1 momentum requirements (aligned with main framework)
            # Primary timeframes: 1H (40%) and 30m (35%) priority
            if on1h and abs(move_pct) > 0.8 and v_val > 1.3:
                has_tier1_momentum = True
                break
            elif '30m' in label and abs(move_pct) > 1.0 and v_val > 1.2:
                has_tier1_momentum = True  
                break
            elif sig and abs(move_pct) > 1.2 and v_val > 1.5:  # Higher bar for 15m
                has_tier1_momentum = True
                break
            elif on4h and abs(move_pct) > 0.6 and v_val > 1.4:  # Context only
                has_tier1_momentum = True
                break

        if not has_tier1_momentum:
            return False, (
                "No Tier 1 momentum detected. Required: 1H >0.8%+vol>1.3x | 30m >1.0%+vol>1.2x | 15m >1.2%+vol>1.5x | 4H >0.6%+vol>1.4x (context only)."
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

            tf_s = str(tf).lower()
            if ('30m' in tf_s or '1h' in tf_s) and len(df) >= 10:
                fast_prev_idx = -10 if len(df) >= 10 else -len(df)
                fast_prev_price = df['c'].iloc[fast_prev_idx]
                fast_change = (current_price - fast_prev_price) / fast_prev_price * 100
                if fast_change <= -0.6:  # >= 0.6% decline over last ~10 bars
                    return True
        
        # No significant changes detected across any timeframe
        return False

    def _dynamic_chop_allowance(self, df: pd.DataFrame, base_allowable: int = 5) -> int:
        """Adapt chop reversal allowance using recent realized volatility.

        - Lower vol -> lower allowance (tighter filter)
        - Higher vol -> higher allowance (more forgiveness)
        """
        n = min(20, len(df))
        if n < 5:
            return base_allowable
        tail = df.iloc[-n:]
        c = tail['c']
        h = tail['h']
        l = tail['l']
        vol_pct = (((h - l) / c) * 100).mean()
        allowance = base_allowable
        if vol_pct < 0.4:
            allowance -= 1
        elif vol_pct > 1.2:
            allowance += 1
        return max(2, min(8, allowance))

    def _detect_mean_reversion_opportunity(self, dataframes: Dict[Timeframe, pd.DataFrame]) -> Optional[str]:
        """Detect mean-reversion setups using core indicators (RSI, BB, volume) when trend is weak.

        Long MR: RSI <= 25, close near/below BB_lower (<= +0.2%), v_ratio <= 0.9
        Short MR: RSI >= 75, close near/above BB_upper (>= -0.2%), v_ratio <= 0.9
        """
        reasons: List[str] = []
        
        # Check for mean reversion on primary signal timeframes (15m, 30m)
        for tf, df in dataframes.items():
            tf_str = str(tf).lower()
            if not (('15m' in tf_str) or ('30m' in tf_str)):
                continue
            if df.empty or len(df) < 5:
                continue

            cols = df.columns
            close = float(df['c'].iloc[-1])
            v_ratio = float(df['v_ratio'].iloc[-1]) if 'v_ratio' in cols and not df['v_ratio'].empty else 1.0
            rsi = float(df['RSI'].iloc[-1]) if 'RSI' in cols and not df['RSI'].empty else None

            bb_lower = float(df['BB_lower'].iloc[-1]) if 'BB_lower' in cols and not df['BB_lower'].empty else None
            bb_upper = float(df['BB_upper'].iloc[-1]) if 'BB_upper' in cols and not df['BB_upper'].empty else None

            if rsi is None or (bb_lower is None and bb_upper is None):
                continue

            near = 0.002  # 0.2%
            
            # Long mean-reversion (aligned with core framework - RSI extremes)
            if rsi is not None and bb_lower is not None:
                if rsi <= 25 and v_ratio <= 0.9 and close <= bb_lower * (1 + near):
                    reasons.append(f"Mean-reversion LONG candidate on {tf_str}: RSI {rsi:.1f} (extreme oversold), close near BB_lower, light volume {v_ratio:.2f}x")

            # Short mean-reversion (aligned with core framework - RSI extremes) 
            if rsi is not None and bb_upper is not None:
                if rsi >= 75 and v_ratio <= 0.9 and close >= bb_upper * (1 - near):
                    reasons.append(f"Mean-reversion SHORT candidate on {tf_str}: RSI {rsi:.1f} (extreme overbought), close near BB_upper, light volume {v_ratio:.2f}x")

        if reasons:
            return "; ".join(reasons)
        return None

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
        """Focus on 6 core crypto indicators aligned with main LLMPromptGenerator approach."""
        indicators = {}
        lookback = min(self.TREND_LOOKBACK_PERIODS, len(df))

        # Core 6 crypto indicators (aligned with main analysis framework)
        core_indicators = [
            'RSI',           # Primary momentum
            'v_ratio',       # Volume confirmation (critical for crypto) 
            'ATR',           # Volatility context
            'MACD_Hist',     # Momentum confirmation
            'SuperTrend',    # Primary trend filter
            'BB_width',      # Bollinger Band squeeze/breakout
            'BB_upper', 'BB_middle', 'BB_lower',  # BB levels
            'EMA', 'VWAP'    # Key levels for confluence
        ]

        # Extract recent values for core indicators
        for indicator in core_indicators:
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

        # Add derived indicator insights for better filtering
        if 'RSI' in df.columns and len(df) >= 5:
            rsi_current = indicators.get('RSI', {}).get('current')
            if rsi_current is not None and isinstance(rsi_current, (int, float)):
                if 55 <= rsi_current <= 65:
                    rsi_signal = 'bullish'
                elif 45 < rsi_current < 55:
                    rsi_signal = 'neutral'
                elif 35 <= rsi_current <= 45:
                    rsi_signal = 'bearish'
                else:
                    rsi_signal = 'extreme'
                indicators['RSI_signal'] = {
                    'current': rsi_signal,
                    'values': []
                }

        if 'MACD_Hist' in df.columns and len(df) >= 3:
            hist_values = indicators.get('MACD_Hist', {}).get('values', [])
            if isinstance(hist_values, list) and len(hist_values) >= 3:
                try:
                    hist_trend = 'positive' if hist_values[-1] > hist_values[-3] else 'negative'
                    indicators['MACD_momentum'] = {
                        'current': hist_trend,
                        'values': []
                    }
                except (IndexError, TypeError):
                    pass

        if 'SuperTrend' in df.columns and 'c' in df.columns:
            st_current = indicators.get('SuperTrend', {}).get('current')
            price_current = df['c'].iloc[-1] if not df.empty else None
            if st_current is not None and price_current is not None and isinstance(st_current, (int, float)) and isinstance(price_current, (int, float)):
                indicators['SuperTrend_signal'] = {
                    'current': 'bullish' if price_current > st_current else 'bearish',
                    'values': []
                }

        return indicators
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Focus on core support/resistance levels aligned with main framework."""
        levels: Dict[str, Any] = {}
        lookback = min(self.TREND_LOOKBACK_PERIODS, len(df))
        
        # Core key levels for confluence (aligned with main framework)
        core_levels = [
            'EMA',      # Dynamic support/resistance
            'VWAP',     # Volume-weighted levels
            'BB_upper', 'BB_middle', 'BB_lower',  # Bollinger Band levels
        ]
        
        # Add pivot levels if available (major S/R)
        pivot_levels = ['PIVOT', 'R1', 'R2', 'S1', 'S2']
        
        # Add key Fibonacci levels if available
        fib_levels = ['FIB_38', 'FIB_50', 'FIB_61']
        
        all_levels = core_levels + pivot_levels + fib_levels
        
        for level in all_levels:
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
        
        # Calculate key level confluence (within 0.5% of current price)
        current_price = df['c'].iloc[-1] if 'c' in df.columns and not df.empty else None
        if current_price:
            confluence_levels = []
            tolerance = 0.005  # 0.5%
            
            for level_name, level_data in levels.items():
                level_value = level_data.get('current')
                if level_value and abs(level_value - current_price) / current_price <= tolerance:
                    confluence_levels.append(level_name)
            
            levels['confluence_count'] = {
                'current': len(confluence_levels),
                'levels': confluence_levels
            }
        
        return levels
    
    def _create_filter_prompt(self, coin: str, market_summary: Dict[str, Any]) -> str:
        """Create prompt for cheap LLM model to determine if expensive analysis is needed, aligned with main LLMPromptGenerator approach."""
        return f"""You are a selective signal filter for {coin}. Focus on high-probability setups using 6 core crypto indicators. Aligned with main analysis framework.

Current Market Data:
{json.dumps(market_summary, indent=2)}

🚨 TIMEFRAME PRIORITY (aligned with main analysis):
- PRIMARY: 1H (40%) and 30m (35%) timeframes are decision drivers
- SUPPORT: 15m (15%) for entry timing confirmation  
- CONTEXT ONLY: 4H (10%) for broad trend awareness - DO NOT base decisions on 4H data alone
- IGNORE 4H conflicts: If 1H+30m align strongly, 4H opposition should NOT prevent signals

CORE CRYPTO INDICATOR FRAMEWORK (6 essential indicators only):

1. RSI (Primary Momentum):
   - LONG bias: RSI 45-65 with positive divergence or breaking above 45 from oversold
   - SHORT bias: RSI 35-55 with negative divergence or breaking below 55 from overbought
   - Filter out: RSI >75 or <25 without clear reversal patterns

2. Volume Confirmation (Critical for crypto):
   - Strong: V_Ratio >1.5 confirms breakout/breakdown
   - Weak: V_Ratio <0.8 weakens signal strength
   - Volume must support price direction for valid signals

3. ATR Volatility Context:
   - High ATR (>3% of price): Requires wider stops, shorter time horizon
   - Low ATR (<1.5% of price): Allows tighter stops, longer time horizon

4. MACD Histogram (Momentum Confirmation):
   - LONG signal: MACD histogram turning positive with volume
   - SHORT signal: MACD histogram turning negative with volume
   - Divergence: Histogram direction opposing price movement

5. SuperTrend (Primary Trend Filter):
   - LONG trend: Price above SuperTrend line (bullish bias)
   - SHORT trend: Price below SuperTrend line (bearish bias)
   - Flip signals: SuperTrend color change with volume confirmation

6. Bollinger Band Squeeze (Breakout Anticipation):
   - Squeeze setup: BB_width contracting = potential breakout imminent
   - LONG breakout: BB_width expanding with price above BB_middle + volume
   - SHORT breakdown: BB_width expanding with price below BB_middle + volume

TIER 1 SIGNAL REQUIREMENTS (Need 2+ for analysis approval):
• Multi-timeframe alignment: 1H and 30m agree OR strong momentum on 30m+15m with 1H neutral
• Volume confirmation: V_Ratio >1.5 during setup formation
• Key level confluence: Price at major S/R (EMA/VWAP/Pivot levels) +/- 0.5%
• Momentum confirmation: RSI 45-65 range with positive divergence OR MACD histogram alignment
• Trend alignment: SuperTrend supports direction OR Bollinger Band breakout setup

TIER 2 CONFIRMATIONS (Need 1+ additional):
• MACD histogram momentum shift (positive for LONG, negative for SHORT)
• Bollinger Band squeeze release with volume (BB_width expanding + V_Ratio >1.5)
• SuperTrend alignment with price direction
• EMA/VWAP reclaim or breakdown with sustained follow-through
• Funding rate mean reversion setup (extreme readings)

AUTOMATIC DISQUALIFIERS (Force rejection):
• RSI extreme without reversal: RSI >75 or <25 without clear reversal patterns
• Conflicting primary timeframes: 1H vs 30m disagree on direction
• Low volume confirmation: V_Ratio <0.8 during setup formation
• Bollinger squeeze without direction: BB_width contracting without clear bias
• MACD and RSI divergence: Indicators showing opposing momentum signals
• Signal timeframe micro moves: Price changes <0.2% in ALL 15m/30m timeframes
• Volume drought: V_Ratio <0.8 in signal timeframes or <1.0x in 1h/4h
• Choppy price action: Excessive reversals in recent periods

SIMPLIFIED ANALYSIS DECISION:
• ANALYZE: 2+ Tier 1 factors + 1+ Tier 2 confirmation
• SKIP: Any disqualifier present OR insufficient signal strength
• Emergency bypass: Funding rate >0.0008 AND >1.2% price change AND volume >1.6x
• Focus: Quality over quantity - better to miss than analyze low-probability setups

Confidence based on:
- Signal strength (number of Tier 1/2 factors met)
- Volume confirmation quality
- Multi-timeframe alignment
- Absence of disqualifiers

Response must be pure JSON - no markdown, no explanations. Example:
{{
    "should_analyze": true,
    "reason": "1H+30m alignment, MACD histogram positive, SuperTrend bullish, volume >1.5x. Core crypto setup confirmed.",
    "confidence": 0.82
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
