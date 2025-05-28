import time
import re
from typing import Dict, Any, List
import pandas as pd
from tzlocal import get_localzone
import json
import os
import base64
import requests
from datetime import datetime
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from utils import fmt_price
from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from ..candles_cache import get_candles_with_cache
from ..wyckoff.wyckoff_types import Timeframe
from ..data_processor import prepare_dataframe, apply_indicators
from ..funding_rates_cache import get_funding_with_cache, FundingRateEntry
from utils import exchange_enabled
from .prompt_generator import LLMPromptGenerator

class LLMAnalysisResult:
    """Container for LLM analysis results."""
    
    def __init__(
        self,
        signal: str = "hold",
        confidence: float = 0.5,
        prediction: str = "sideways",
        risk_level: str = "medium",
        should_notify: bool = False,
        description: str = "",
        timeframe_signals: Dict[str, Any] | None = None,
        analysis_cost: float = 0.0,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        key_drivers: List[str] | None = None,
        recap_heading: str = "",
        trading_insight: str = "",
        time_horizon_hours: int = 4,
        trading_setup: Dict[str, Any] | None = None
    ):
        self.signal = signal
        self.confidence = confidence
        self.prediction = prediction
        self.risk_level = risk_level
        self.should_notify = should_notify
        self.description = description
        self.timeframe_signals = timeframe_signals or {}
        self.analysis_cost = analysis_cost
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target_price = target_price
        self.key_drivers = key_drivers or []
        self.recap_heading = recap_heading
        self.trading_insight = trading_insight
        self.time_horizon_hours = time_horizon_hours
        self.trading_setup = trading_setup or {}


class LLMAnalyzer:
    """AI-based technical analysis implementation."""

    def __init__(self):
        # Enhanced lookback periods for each timeframe (in days)
        self.timeframe_lookback_days = {
            Timeframe.MINUTES_15: 5.0,
            Timeframe.MINUTES_30: 5.0,
            Timeframe.HOUR_1: 10.0,
            Timeframe.HOURS_4: 14.0,
        }        
        # Initialize prompt generator
        self.prompt_generator = LLMPromptGenerator(self.timeframe_lookback_days)
        
        # Enhanced thresholds for triggering AI analysis - optimized to reduce false triggers
        self.ai_trigger_thresholds = {
            # Basic triggers - increased thresholds for stricter filtering
            'price_movement': 0.025,     # 2.5% price movement (more conservative)
            'volume_spike': 1.8,         # 1.8x volume increase (less sensitive)
            'volatility': 0.018,         # 1.8% volatility (less sensitive)
            
            # Intraday specific triggers - more selective
            'momentum_breakout': 0.015,   # 1.5% breakout from range
            'structure_break': 0.012,     # 1.2% structure level break
            'macd_signal': True,          # MACD signal line crosses
            'supertrend_flip': True,      # SuperTrend direction change
            'bb_squeeze': 0.015,          # Bollinger Band squeeze release
            'funding_extreme': 0.0005,    # 0.05% funding rate extreme
            
            # Multi-indicator confluence requirements
            'rsi_extreme': 80,            # RSI overbought/oversold levels
            'rsi_oversold': 20,
            'stoch_extreme': 80,          # Stochastic extreme levels
            'fibonacci_proximity': 0.003, # Within 0.3% of key Fib levels
            'pivot_proximity': 0.005,     # Within 0.5% of pivot levels
            'ichimoku_signal': True,      # Ichimoku cloud signals
            
            # Session-based thresholds
            'asian_session_multiplier': 0.6,    # Lower thresholds during Asian session
            'overlap_session_multiplier': 1.1,   # Slight increase during overlaps
            'weekend_multiplier': 0.7,          # Reduced for weekend trading': 0.7,          # Reduced for weekend trading
            
            # Combined scoring - stricter requirements
            'combined_score': 0.6,        # Higher threshold for trigger
            'confluence_bonus': 0.3,      # Higher bonus for multiple indicators
            'timeframe_agreement': 0.25,  # Higher bonus for multi-timeframe agreement
            'minimum_indicators': 3       # Minimum number of indicators needed
        }
    
    def _should_run_ai_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, interactive: bool) -> tuple[bool, str]:
        """Pre-filter to determine if expensive AI analysis is warranted with strict filtering."""
        if interactive:
            return True, "Interactive analysis requested"
        
        # Get current hour for session-based adjustments
        current_hour = datetime.now().hour
        session_multiplier = self._get_session_multiplier(current_hour)
        
        trigger_scores = []
        reasons = []
        timeframe_signals = {}
        significant_indicators = 0
        
        # Check multiple timeframes for significant activity
        for tf, df in dataframes.items():
            if df.empty or len(df) < 10:
                continue
            
            scores, tf_reasons, signal_data = self._analyze_timeframe_triggers(df, tf, session_multiplier)
            
            # Only count significant scores (filter out noise)
            significant_scores = [s for s in scores if s >= 0.15]
            if significant_scores:
                trigger_scores.extend(significant_scores)
                reasons.extend(tf_reasons)
                timeframe_signals[tf] = signal_data
                significant_indicators += len(significant_scores)
        
        # Require minimum number of significant indicators
        if significant_indicators < self.ai_trigger_thresholds['minimum_indicators']:
            reason = f"Insufficient significant indicators ({significant_indicators}/{self.ai_trigger_thresholds['minimum_indicators']})"
            logger.debug(f"AI analysis skipped for {coin}: {reason}")
            return False, reason
        
        # Enhanced confluence check with stricter requirements
        confluence_score = self._calculate_confluence_score(timeframe_signals)
        timeframe_agreement = self._check_timeframe_agreement(timeframe_signals)
        
        # Bonus only for strong confluence across multiple timeframes
        if confluence_score > 0.8 and timeframe_agreement >= 0.7:
            trigger_scores.append(self.ai_trigger_thresholds['confluence_bonus'])
            reasons.append("Strong multi-timeframe confluence")
        elif timeframe_agreement >= 0.6:
            trigger_scores.append(self.ai_trigger_thresholds['timeframe_agreement'])
            reasons.append("Timeframe agreement detected")
        
        # Calculate combined trigger score with weighted average
        base_score = sum(trigger_scores) / max(len(trigger_scores), 1) if trigger_scores else 0
        adjusted_score = base_score * session_multiplier
        
        # Additional filtering: require strong signals in primary timeframes
        primary_tf_strength = self._check_primary_timeframe_strength(timeframe_signals)
        if primary_tf_strength < 0.3:
            reason = f"Weak primary timeframe signals (strength: {primary_tf_strength:.2f})"
            logger.debug(f"AI analysis skipped for {coin}: {reason}")
            return False, reason
        
        should_analyze = (adjusted_score >= self.ai_trigger_thresholds['combined_score'] and 
                         primary_tf_strength >= 0.3)
        
        if should_analyze:
            reason = f"Score: {adjusted_score:.2f} (session: {session_multiplier:.1f}x, indicators: {significant_indicators}) - " + "; ".join(reasons[:3])
            logger.info(f"AI analysis triggered for {coin}: {reason}")
        else:
            reason = f"Low activity (score: {adjusted_score:.2f}, threshold: {self.ai_trigger_thresholds['combined_score']}, strength: {primary_tf_strength:.2f})"
            logger.debug(f"AI analysis skipped for {coin}: {reason}")
        
        return should_analyze, "; ".join(reasons) if reasons else "No significant activity detected"
    
    def _analyze_timeframe_triggers(self, df: pd.DataFrame, tf: Timeframe, session_multiplier: float) -> tuple[list, list, dict]:
        """Analyze individual timeframe for trigger conditions."""
        scores = []
        reasons = []
        signal_data = {}
        
        # Price movement analysis with session adjustment
        recent_price_change = abs((df['c'].iloc[-1] - df['c'].iloc[-5]) / df['c'].iloc[-5])
        threshold = self.ai_trigger_thresholds['price_movement'] * (1 / session_multiplier)
        if recent_price_change > threshold:
            score = recent_price_change * 2 * session_multiplier
            scores.append(score)
            reasons.append(f"{tf}: {recent_price_change:.1%} price movement")
        
        # Volume spike detection
        if 'v_ratio' in df.columns and not df['v_ratio'].empty:
            volume_ratio = df['v_ratio'].iloc[-1]
            if volume_ratio > self.ai_trigger_thresholds['volume_spike']:
                score = (volume_ratio - 1) * 0.5 * session_multiplier
                scores.append(score)
                reasons.append(f"{tf}: {volume_ratio:.1f}x volume spike")
        
        # Volatility analysis with session adjustment
        if len(df) >= 20:
            volatility = df['c'].pct_change().rolling(20).std().iloc[-1]
            if volatility > self.ai_trigger_thresholds['volatility'] * (1 / session_multiplier):
                score = volatility * 10 * session_multiplier
                scores.append(score)
                reasons.append(f"{tf}: High volatility {volatility:.1%}")
        
        # Enhanced indicator analysis with stricter thresholds
        
        # RSI extremes - require more extreme levels
        if 'RSI' in df.columns and not df['RSI'].empty:
            rsi = df['RSI'].iloc[-1]
            if rsi > self.ai_trigger_thresholds['rsi_extreme'] or rsi < self.ai_trigger_thresholds['rsi_oversold']:
                score = 0.3 if rsi > 85 or rsi < 15 else 0.15  # Higher score for extreme levels
                scores.append(score)
                reasons.append(f"{tf}: RSI extreme {rsi:.1f}")
        
        # Stochastic extremes - new indicator check
        if 'STOCH_K' in df.columns and not df['STOCH_K'].empty:
            stoch_k = df['STOCH_K'].iloc[-1]
            if stoch_k > self.ai_trigger_thresholds['stoch_extreme'] or stoch_k < (100 - self.ai_trigger_thresholds['stoch_extreme']):
                scores.append(0.2)
                reasons.append(f"{tf}: Stochastic extreme {stoch_k:.1f}")
        
        # Fibonacci proximity check - new feature
        fib_score = self._check_fibonacci_proximity(df)
        if fib_score > 0:
            scores.append(fib_score)
            reasons.append(f"{tf}: Near Fibonacci level")
        
        # Pivot point proximity - new feature
        pivot_score = self._check_pivot_proximity(df)
        if pivot_score > 0:
            scores.append(pivot_score)
            reasons.append(f"{tf}: Near pivot level")
        
        # Ichimoku cloud signals - new indicator
        ichimoku_score = self._check_ichimoku_signals(df)
        if ichimoku_score > 0:
            scores.append(ichimoku_score)
            reasons.append(f"{tf}: Ichimoku signal")
        
        # MACD signal line crosses
        if self._check_macd_signal(df):
            scores.append(0.25)
            reasons.append(f"{tf}: MACD signal cross")
        
        # SuperTrend flips
        if self._check_supertrend_flip(df):
            scores.append(0.3)
            reasons.append(f"{tf}: SuperTrend flip")
        
        # Bollinger Band squeeze detection
        bb_squeeze_score = self._check_bb_squeeze(df)
        if bb_squeeze_score > 0:
            scores.append(bb_squeeze_score)
            reasons.append(f"{tf}: BB squeeze/expansion")
        
        # Structure breaks (high/low breaks)
        structure_score = self._check_structure_breaks(df, tf)
        if structure_score > 0:
            scores.append(structure_score)
            reasons.append(f"{tf}: Structure break detected")
        
        # Momentum breakouts
        momentum_score = self._check_momentum_breakout(df, tf)
        if momentum_score > 0:
            scores.append(momentum_score)
            reasons.append(f"{tf}: Momentum breakout")
        
        # Build signal data for confluence calculation
        signal_data = {
            'price_change': recent_price_change,
            'volume_ratio': df.get('v_ratio', pd.Series([1.0])).iloc[-1] if 'v_ratio' in df.columns else 1.0,
            'volatility': volatility if len(df) >= 20 else 0.0,
            'rsi': df.get('rsi', pd.Series([50.0])).iloc[-1] if 'rsi' in df.columns else 50.0,
            'macd_signal': self._check_macd_signal(df),
            'supertrend_bullish': self._is_supertrend_bullish(df),
            'bb_position': self._get_bb_position(df),
            'structure_trend': self._get_structure_trend(df),
            'momentum': recent_price_change,
            'strength': sum(scores) if scores else 0.0
        }
        
        return scores, reasons, signal_data
    
    def _get_session_multiplier(self, current_hour: int) -> float:
        """Calculate session-based multiplier for trigger sensitivity."""
        # UTC hours for major trading sessions
        if 0 <= current_hour < 8:  # Asian session
            return self.ai_trigger_thresholds['asian_session_multiplier']
        elif 8 <= current_hour < 16:  # European session
            return 1.0  # Base multiplier
        elif 16 <= current_hour < 22:  # US session
            return 1.1  # Slightly higher sensitivity
        else:  # Overlap/quiet hours
            return self.ai_trigger_thresholds['overlap_session_multiplier']
    
    def _calculate_confluence_score(self, timeframe_signals: Dict) -> float:
        """Calculate confluence score across multiple timeframes."""
        if not timeframe_signals:
            return 0.0
        
        total_score = 0
        signal_count = 0
        
        for tf_data in timeframe_signals.values():
            if tf_data.get('strength', 0) > 0:
                total_score += tf_data['strength']
                signal_count += 1
        
        return total_score / max(signal_count, 1) if signal_count > 0 else 0.0
    
    def _check_macd_signal(self, df: pd.DataFrame) -> bool:
        """Check for MACD signal line crosses."""
        if 'MACD' not in df.columns or 'MACD_signal' not in df.columns or len(df) < 3:
            return False
        
        macd = df['MACD'].iloc[-3:]
        signal = df['MACD_signal'].iloc[-3:]
        
        # Check for bullish or bearish cross in last 2 periods
        return ((macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]) or
                (macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]))
    
    def _check_supertrend_flip(self, df: pd.DataFrame) -> bool:
        """Check for SuperTrend direction changes."""
        if 'SuperTrend' not in df.columns or len(df) < 3:
            return False
        
        current_price = df['c'].iloc[-1]
        prev_price = df['c'].iloc[-2]
        current_st = df['SuperTrend'].iloc[-1]
        prev_st = df['SuperTrend'].iloc[-2]
        
        # Check for trend flip
        prev_bullish = prev_price > prev_st
        current_bullish = current_price > current_st
        
        return prev_bullish != current_bullish
    
    def _check_bb_squeeze(self, df: pd.DataFrame) -> float:
        """Check for Bollinger Band squeeze and expansion."""
        if not all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']) or len(df) < 20:
            return 0.0
        
        # Calculate BB width
        bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        current_width = bb_width.iloc[-1]
        avg_width = bb_width.rolling(20).mean().iloc[-1]
        
        # Squeeze: current width significantly below average
        if current_width < avg_width * 0.7:
            return 0.2
        # Expansion: current width significantly above average
        elif current_width > avg_width * 1.3:
            return 0.25
        
        return 0.0
    
    def _check_structure_breaks(self, df: pd.DataFrame, tf: Timeframe) -> float:
        """Check for market structure breaks (support/resistance)."""
        if len(df) < 20:
            return 0.0
        
        # Look for recent highs and lows
        high_window = 10 if tf in [Timeframe.MINUTES_15, Timeframe.MINUTES_30] else 5
        recent_high = df['h'].rolling(high_window).max().iloc[-1]
        recent_low = df['l'].rolling(high_window).min().iloc[-1]
        current_price = df['c'].iloc[-1]
        
        # Check for breaks
        if current_price > recent_high * 1.002:  # 0.2% above recent high
            return 0.3
        elif current_price < recent_low * 0.998:  # 0.2% below recent low
            return 0.25  # Slightly different value to avoid duplication
        
        return 0.0
    
    def _check_momentum_breakout(self, df: pd.DataFrame, tf: Timeframe) -> float:
        """Check for momentum breakouts from consolidation."""
        if len(df) < 20:
            return 0.0
        
        # Calculate price range and breakout
        lookback = 15 if tf in [Timeframe.MINUTES_15, Timeframe.MINUTES_30] else 10
        price_range = df['c'].rolling(lookback)
        range_high = price_range.max().iloc[-1]
        range_low = price_range.min().iloc[-1]
        current_price = df['c'].iloc[-1]
        
        # Check for range breakout
        range_size = (range_high - range_low) / range_low
        if range_size < 0.02:  # Tight consolidation (2% range)
            if current_price > range_high * 1.001:  # Upward breakout
                return 0.25
            elif current_price < range_low * 0.999:  # Downward breakout
                return 0.22  # Slightly different value to avoid duplication
        
        return 0.0
    
    def _is_supertrend_bullish(self, df: pd.DataFrame) -> bool:
        """Check if SuperTrend indicates bullish trend."""
        if 'SuperTrend' not in df.columns or len(df) < 1:
            return False
        return df['c'].iloc[-1] > df['SuperTrend'].iloc[-1]
    
    def _get_bb_position(self, df: pd.DataFrame) -> str:
        """Get current position relative to Bollinger Bands."""
        if not all(col in df.columns for col in ['BB_upper', 'BB_lower', 'BB_middle']) or len(df) < 1:
            return "neutral"
        
        price = df['c'].iloc[-1]
        upper = df['BB_upper'].iloc[-1]
        lower = df['BB_lower'].iloc[-1]
        middle = df['BB_middle'].iloc[-1]
        
        if price > upper:
            return "above_upper"
        elif price < lower:
            return "below_lower"
        elif price > middle:
            return "above_middle"
        else:
            return "below_middle"
    
    def _get_structure_trend(self, df: pd.DataFrame) -> str:
        """Determine current market structure trend."""
        if len(df) < 10:
            return "neutral"
        
        # Simple trend based on recent price action
        recent_prices = df['c'].iloc[-10:]
        if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.01:
            return "bullish"
        elif recent_prices.iloc[-1] < recent_prices.iloc[-0] * 0.99:
            return "bearish"
        else:
            return "neutral"

    async def analyze(self, context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
        """Main AI analysis entry point."""
        
        now = int(time.time() * 1000)
        local_tz = get_localzone()

        # Get candles for analysis - use lookback days directly
        candles_data = {}
        for tf, lookback_days in self.timeframe_lookback_days.items():
            candles_data[tf] = get_candles_with_cache(coin, tf, now, lookback_days, hyperliquid_utils.info.candles_snapshot)

        # Check if we have enough data for basic analysis
        if len(candles_data[Timeframe.MINUTES_15]) < 10:
            logger.warning(f"Insufficient candles for AI analysis on {coin}")
            return

        # Prepare dataframes
        dataframes = {
            tf: prepare_dataframe(candles, local_tz) 
            for tf, candles in candles_data.items()
        }

        # Apply basic indicators for AI analysis
        for tf, df in dataframes.items():
            if not df.empty:
                apply_indicators(df, tf)

        # Pre-filter: Check if AI analysis is needed
        should_analyze, filter_reason = self._should_run_ai_analysis(dataframes, coin, interactive_analysis)
        
        if not should_analyze:
            # Send simple message for non-interactive requests when no analysis is needed
            if not interactive_analysis:
                logger.debug(f"Skipping AI analysis for {coin}: {filter_reason}")
                return
            else:
                # For interactive requests, still provide basic analysis
                await self._send_basic_analysis_message(context, coin, filter_reason)
                return

        # Perform expensive AI analysis only when triggered
        ai_result = await self._perform_ai_analysis(dataframes, coin)

        should_notify = interactive_analysis or ai_result.should_notify

        if should_notify:
            await self._send_ai_analysis_message(context, coin, ai_result)

    async def _send_basic_analysis_message(self, context: ContextTypes.DEFAULT_TYPE, coin: str, reason: str) -> None:
        """Send basic analysis when AI analysis is not triggered."""
        message = (
            f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n"
            f"📊 <b>Status:</b> No significant activity detected\n"
            f"💬 <b>Reason:</b> {reason}\n\n"
            f"ℹ️ <i>AI analysis is triggered only when significant market movements are detected to optimize costs.</i>"
        )
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)
    async def _perform_ai_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str) -> LLMAnalysisResult:
        """Core AI analysis logic using OpenRouter.ai."""

        try:
            # Get current market data
            mid_price = dataframes[Timeframe.MINUTES_15]['c'].iloc[-1] if not dataframes[Timeframe.MINUTES_15].empty else 0.0
            now = int(time.time() * 1000)
            funding_rates = get_funding_with_cache(coin, now, 5)
              # Generate prompt for LLM
            prompt = self.prompt_generator.generate_prediction_prompt(coin, dataframes, funding_rates, mid_price)
            
            # Call OpenRouter.ai API and track cost
            ai_response, analysis_cost = self._call_openrouter_api(prompt)
            
            # Parse AI response into structured result
            result = self._parse_ai_response(ai_response, coin)
            result.analysis_cost = analysis_cost
            
            # Add timeframe analysis for additional context
            timeframe_signals = {}
            for tf, df in dataframes.items():
                if not df.empty and len(df) >= 5:
                    signal_data = self._get_simple_momentum(df)
                    timeframe_signals[str(tf)] = signal_data
            
            result.timeframe_signals = timeframe_signals
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed for {coin}: {str(e)}", exc_info=True)
            return LLMAnalysisResult(
                description=f"AI analysis for {coin}: Analysis failed due to technical error. Using fallback analysis.",
                signal="hold",
                confidence=0.5,
                analysis_cost=0.0
            )
    
    def _get_simple_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple momentum calculation for AI analysis."""
        recent_change = (df['c'].iloc[-1] - df['c'].iloc[-5]) / df['c'].iloc[-5]
        
        # Volume confirmation
        volume_ratio = 1.0
        if 'v_ratio' in df.columns and not df['v_ratio'].empty:
            volume_ratio = df['v_ratio'].iloc[-1]
        
        # Adjust strength based on volume
        strength = abs(recent_change)
        if volume_ratio > 1.5:
            strength *= 1.2
        elif volume_ratio < 0.5:
            strength *= 0.8
        
        return {
            "momentum": "positive" if recent_change > 0 else "negative",
            "strength": strength,
            "price_change": recent_change,
            "volume_ratio": volume_ratio
        }
    
    async def _send_ai_analysis_message(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        coin: str, 
        ai_result: LLMAnalysisResult
    ) -> None:
        """Send AI analysis results to Telegram."""
        
        # Build analysis message
        message = self._build_analysis_message(coin, ai_result)
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)
    
    def _build_analysis_message(self, coin: str, ai_result: LLMAnalysisResult) -> str:
        """Build the analysis message text."""

        message = f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n\n"
        message += f"<b>📊 Market Analysis:</b> {ai_result.recap_heading}\n\n"
        
        # Add signal information
        message += (
            f"📊 <b>Signal:</b> {ai_result.signal.lower()}\n"
            f"🎯 <b>Confidence:</b> {ai_result.confidence:.0%}\n"
            f"📈 <b>Prediction:</b> {ai_result.prediction.lower()}\n"
            f"⚠️ <b>Risk Level:</b> {ai_result.risk_level.lower()}"
        )

        # Add trading insight if available
        trading_insight = getattr(ai_result, 'trading_insight', '')
        if trading_insight:
            message += f"\n\n💡 <b>Trading Insight:</b>\n{trading_insight}"
        
        
        trade_setup = self._build_trade_setup_format(coin, ai_result)
        if trade_setup:
            message += trade_setup
           
        return message
    

    def _build_trade_setup_format(self, coin: str, ai_result: LLMAnalysisResult) -> str:
        """Build formatted trade setup."""
        if not (ai_result.entry_price > 0 and (ai_result.stop_loss > 0 or ai_result.target_price > 0)):
            return ""

        enc_side = "L" if ai_result.signal == "buy" else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(ai_result.stop_loss)}_{fmt_price(ai_result.target_price)}".encode('utf-8')).decode('utf-8')
        trade_link = f"({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

        side = "Long" if ai_result.signal == "buy" else "Short"

        setup = f"\n\n<b>💰 {side} Trade Setup</b>{trade_link}<b>:</b>"
        
        # Market price (entry price)
        if ai_result.entry_price > 0:
            setup += f"\nMarket price: {fmt_price(ai_result.entry_price)} USDC"
        
        # Stop Loss with percentage
        if ai_result.stop_loss > 0 and ai_result.entry_price > 0:
            if ai_result.signal == "buy":
                sl_percentage = ((ai_result.stop_loss - ai_result.entry_price) / ai_result.entry_price) * 100
            else:  # short
                sl_percentage = ((ai_result.entry_price - ai_result.stop_loss) / ai_result.entry_price) * 100
            
            setup += f"\nStop Loss: {fmt_price(ai_result.stop_loss)} USDC ({sl_percentage:+.1f}%)"
        
        # Take Profit with percentage
        if ai_result.target_price > 0 and ai_result.entry_price > 0:
            if ai_result.signal == "buy":
                tp_percentage = ((ai_result.target_price - ai_result.entry_price) / ai_result.entry_price) * 100
            else:  # short
                tp_percentage = ((ai_result.entry_price - ai_result.target_price) / ai_result.entry_price) * 100
            
            setup += f"\nTake Profit: {fmt_price(ai_result.target_price)} USDC ({tp_percentage:+.1f}%)"
        
        return setup


    def _build_price_levels(self, ai_result: LLMAnalysisResult) -> str:
        """Build price levels section."""
        if not (ai_result.entry_price > 0 or ai_result.stop_loss > 0 or ai_result.target_price > 0):
            return ""
        
        levels = "\n📋 <b>Price Levels:</b>"
        if ai_result.entry_price > 0:
            levels += f"\n   Entry: ${ai_result.entry_price:.4f}"
        if ai_result.stop_loss > 0:
            levels += f"\n   Stop Loss: ${ai_result.stop_loss:.4f}"
        if ai_result.target_price > 0:
            levels += f"\n   Target: ${ai_result.target_price:.4f}"
        return levels
    
    def _format_timeframe_signals(self, timeframe_signals: Dict[str, Any], include_details: bool) -> str:
        """Format timeframe signals for display."""
        formatted = ""
        for tf, signal in timeframe_signals.items():
            emoji = "📈" if signal['momentum'] == "positive" else "📉"
            formatted += (
                f"\n{emoji} <b>{tf}:</b> {signal['momentum'].title()} "
                f"({signal['strength']:.1%})"
            )
            
            if include_details:
                formatted += f" | Vol: {signal['volume_ratio']:.1f}x"
        
        return formatted

    def _call_openrouter_api(self, prompt: str) -> tuple[str, float]:
        """Call OpenRouter.ai API for AI analysis and return response with cost."""
        api_key = os.getenv("HTB_OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("HTB_OPENROUTER_API_KEY environment variable not set")
        
        model = os.getenv("HTB_OPENROUTER_MODEL", "google/gemini-2.5-flash-preview-05-20")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency technical analyst. Respond ONLY with valid JSON format. Do not include any text outside the JSON structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "reasoning": {
                "max_tokens": 5000
            },
            "response_format": {
                "type": "json_object"
            },
            "usage": {
                "include": "true"
            },
            "max_tokens": 10000
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValueError(f"OpenRouter API error {response.status_code}: {response.text}")
            
            data = response.json()
            
            # Extract actual cost from OpenRouter response
            usage = data.get("usage", {})
            total_cost = usage.get("cost", 0.0)
            
            return data["choices"][0]["message"]["content"], total_cost
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OpenRouter API request failed: {str(e)}")
    
    def _parse_ai_response(self, ai_response: str, coin: str) -> LLMAnalysisResult:
        """Parse AI response into structured analysis result."""
        try:
            # Try to parse as JSON first
            response_data = json.loads(ai_response)
            
            # Extract values from JSON response
            signal = response_data.get("signal", "hold").lower()
            confidence = float(response_data.get("confidence", 0.5))
            prediction = response_data.get("prediction", "sideways").lower()
            risk_level = response_data.get("risk_level", "medium").lower()
            entry_price = float(response_data.get("entry_price") or 0.0)
            stop_loss = float(response_data.get("stop_loss") or 0.0)
            target_price = float(response_data.get("target_price") or 0.0)
            
            # Validate risk management rules - ensure SL/TP are at least 1.25% away
            if entry_price > 0:
                min_sl_distance = entry_price * 0.0125  # 1.25%
                min_tp_distance = entry_price * 0.0125  # 1.25%
                
                if signal == "buy":
                    # For long positions: SL below entry, TP above entry
                    if stop_loss > 0 and stop_loss > entry_price - min_sl_distance:
                        stop_loss = 0.0  # Invalid SL, clear it
                    if target_price > 0 and target_price < entry_price + min_tp_distance:
                        target_price = 0.0  # Invalid TP, clear it
                elif signal == "sell":
                    # For short positions: SL above entry, TP below entry
                    if stop_loss > 0 and stop_loss < entry_price + min_sl_distance:
                        stop_loss = 0.0  # Invalid SL, clear it
                    if target_price > 0 and target_price > entry_price - min_tp_distance:
                        target_price = 0.0  # Invalid TP, clear it
            
            # Extract new fields
            recap_heading = response_data.get("recap_heading", "")
            trading_insight = response_data.get("trading_insight", "")
            key_drivers = response_data.get("key_drivers", [])
            
            # Simple description from response or fallback
            description = response_data.get("analysis", "AI-powered intraday analysis completed")
            
            # Determine if we should notify based on signal strength
            min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
            should_notify = signal in ["buy", "sell"] and confidence >= min_confidence
            
            return LLMAnalysisResult(
                signal=signal,
                confidence=confidence,
                prediction=prediction,
                risk_level=risk_level,
                should_notify=should_notify,
                description=description,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                key_drivers=key_drivers,
                recap_heading=recap_heading,
                trading_insight=trading_insight
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"JSON parsing failed for {coin}: {str(e)}\n{ai_response}", exc_info=True)
            return LLMAnalysisResult(
                description=f"AI analysis for {coin}: Analysis failed due to technical error. Using fallback analysis.",
                signal="hold",
                confidence=0.5,
                analysis_cost=0.0
            )

    def _check_fibonacci_proximity(self, df: pd.DataFrame) -> float:
        """Check if price is near key Fibonacci levels."""
        if not all(col in df.columns for col in ['FIB_23', 'FIB_38', 'FIB_50', 'FIB_61', 'FIB_78']) or df.empty:
            return 0.0
        
        current_price = df['c'].iloc[-1]
        fib_levels = [
            df['FIB_23'].iloc[-1],
            df['FIB_38'].iloc[-1],
            df['FIB_50'].iloc[-1],
            df['FIB_61'].iloc[-1],
            df['FIB_78'].iloc[-1]
        ]
        
        # Check proximity to any Fibonacci level
        proximity_threshold = self.ai_trigger_thresholds['fibonacci_proximity']
        for level in fib_levels:
            if level > 0:  # Valid level
                distance = abs(current_price - level) / current_price
                if distance <= proximity_threshold:
                    return 0.25  # High score for Fib proximity
        
        return 0.0
    
    def _check_pivot_proximity(self, df: pd.DataFrame) -> float:
        """Check if price is near pivot points."""
        if not all(col in df.columns for col in ['PIVOT', 'R1', 'R2', 'S1', 'S2']) or df.empty:
            return 0.0
        
        current_price = df['c'].iloc[-1]
        pivot_levels = [
            df['PIVOT'].iloc[-1],
            df['R1'].iloc[-1],
            df['R2'].iloc[-1],
            df['S1'].iloc[-1],
            df['S2'].iloc[-1]
        ]
        
        # Check proximity to any pivot level
        proximity_threshold = self.ai_trigger_thresholds['pivot_proximity']
        for level in pivot_levels:
            if level > 0:  # Valid level
                distance = abs(current_price - level) / current_price
                if distance <= proximity_threshold:
                    return 0.2  # Score for pivot proximity
        
        return 0.0
    
    def _check_ichimoku_signals(self, df: pd.DataFrame) -> float:
        """Check for Ichimoku cloud signals."""
        if not all(col in df.columns for col in ['TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B']) or len(df) < 3:
            return 0.0
        
        current_price = df['c'].iloc[-1]
        tenkan = df['TENKAN'].iloc[-1]
        kijun = df['KIJUN'].iloc[-1]
        senkou_a = df['SENKOU_A'].iloc[-1]
        senkou_b = df['SENKOU_B'].iloc[-1]
        
        score = 0.0
        
        # Tenkan-Kijun cross
        if len(df) >= 2:
            prev_tenkan = df['TENKAN'].iloc[-2]
            prev_kijun = df['KIJUN'].iloc[-2]
            
            # Bullish cross
            if prev_tenkan <= prev_kijun and tenkan > kijun:
                score += 0.3
            # Bearish cross
            elif prev_tenkan >= prev_kijun and tenkan < kijun:
                score += 0.3
        
        # Cloud breakout
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if len(df) >= 2:
            prev_price = df['c'].iloc[-2]
            
            # Breakout above cloud
            if prev_price <= cloud_top and current_price > cloud_top:
                score += 0.25
            # Breakout below cloud
            elif prev_price >= cloud_bottom and current_price < cloud_bottom:
                score += 0.25
        
        return score
    
    def _check_timeframe_agreement(self, timeframe_signals: Dict) -> float:
        """Check agreement between different timeframes."""
        if len(timeframe_signals) < 2:
            return 0.0
        
        signals = []
        for tf_data in timeframe_signals.values():
            if tf_data.get('strength', 0) > 0.1:  # Only consider significant signals
                # Determine signal direction
                if tf_data.get('momentum', 'neutral') == 'positive':
                    signals.append(1)
                elif tf_data.get('momentum', 'neutral') == 'negative':
                    signals.append(-1)
                else:
                    signals.append(0)
        
        if not signals:
            return 0.0
        
        # Calculate agreement percentage
        positive_signals = sum(1 for s in signals if s > 0)
        negative_signals = sum(1 for s in signals if s < 0)
        total_signals = len(signals)
        
        # Agreement is the percentage of signals pointing in the same direction
        agreement = max(positive_signals, negative_signals) / total_signals
        return agreement
    
    def _check_primary_timeframe_strength(self, timeframe_signals: Dict) -> float:
        """Check strength of signals in primary timeframes (15m, 1h)."""
        primary_timeframes = [Timeframe.MINUTES_15, Timeframe.HOUR_1]
        primary_strength = 0.0
        primary_count = 0
        
        for tf, tf_data in timeframe_signals.items():
            if tf in primary_timeframes:
                strength = tf_data.get('strength', 0.0)
                primary_strength += strength
                primary_count += 1
        
        return primary_strength / max(primary_count, 1) if primary_count > 0 else 0.0
