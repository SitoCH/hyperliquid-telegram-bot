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
from technical_analysis.candles_cache import get_candles_with_cache
from technical_analysis.wyckoff_types import Timeframe
from technical_analysis.data_processor import prepare_dataframe, apply_indicators
from technical_analysis.funding_rates_cache import get_funding_with_cache, FundingRateEntry
from utils import exchange_enabled

class AIAnalysisResult:
    """Container for AI analysis results."""
    
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
        intraday_signal: str = "hold",
        intraday_confidence: float = 0.5,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        target_price: float = 0.0,
        key_drivers: List[str] | None = None,
        price_predictions: Dict[str, Any] | None = None,
        market_analysis: str = "",
        technical_analysis: str = "",
        risk_assessment: str = ""
    ):
        self.signal = signal
        self.confidence = confidence
        self.prediction = prediction
        self.risk_level = risk_level
        self.should_notify = should_notify
        self.description = description
        self.timeframe_signals = timeframe_signals or {}
        self.analysis_cost = analysis_cost
        self.intraday_signal = intraday_signal
        self.intraday_confidence = intraday_confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target_price = target_price
        self.key_drivers = key_drivers or []
        self.price_predictions = price_predictions or {}
        self.market_analysis = market_analysis
        self.technical_analysis = technical_analysis
        self.risk_assessment = risk_assessment


class AIAnalyzer:
    """AI-based technical analysis implementation."""
    
    def __init__(self):
        self.ai_timeframes = {
            Timeframe.MINUTES_15: 28,
            Timeframe.MINUTES_30: 42,
            Timeframe.HOUR_1: 60,
            Timeframe.HOURS_2: 75,
            Timeframe.HOURS_4: 90,
            Timeframe.HOURS_8: 120
        }
        
        # Thresholds for triggering AI analysis
        self.ai_trigger_thresholds = {
            'price_movement': 0.02,  # 2% price movement
            'volume_spike': 1.5,     # 1.5x volume increase
            'volatility': 0.015,     # 1.5% volatility
            'combined_score': 0.5    # Combined threshold score
        }
    
    def _should_run_ai_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, interactive: bool) -> tuple[bool, str]:
        """Pre-filter to determine if expensive AI analysis is warranted."""
        if interactive:
            return True, "Interactive analysis requested"
        
        trigger_scores = []
        reasons = []
        
        # Check multiple timeframes for significant activity
        for tf, df in dataframes.items():
            if df.empty or len(df) < 10:
                continue
            
            scores, tf_reasons = self._analyze_timeframe_triggers(df, tf)
            trigger_scores.extend(scores)
            reasons.extend(tf_reasons)
        
        # Calculate combined trigger score
        combined_score = sum(trigger_scores) / len(self.ai_timeframes) if trigger_scores else 0
        should_analyze = combined_score >= self.ai_trigger_thresholds['combined_score']
        
        if should_analyze:
            reason = f"Trigger score: {combined_score:.2f} - " + "; ".join(reasons[:3])
            logger.info(f"AI analysis triggered for {coin}: {reason}")
        else:
            logger.debug(f"AI analysis skipped for {coin}: Low activity (score: {combined_score:.2f})")
        
        return should_analyze, "; ".join(reasons) if reasons else "No significant activity detected"
    
    def _analyze_timeframe_triggers(self, df: pd.DataFrame, tf: Timeframe) -> tuple[list, list]:
        """Analyze individual timeframe for trigger conditions."""
        scores = []
        reasons = []
        
        # Price movement analysis
        recent_price_change = abs((df['c'].iloc[-1] - df['c'].iloc[-5]) / df['c'].iloc[-5])
        if recent_price_change > self.ai_trigger_thresholds['price_movement']:
            scores.append(recent_price_change * 2)
            reasons.append(f"{tf}: {recent_price_change:.1%} price movement")
        
        # Volume spike detection
        if 'v_ratio' in df.columns and not df['v_ratio'].empty:
            volume_ratio = df['v_ratio'].iloc[-1]
            if volume_ratio > self.ai_trigger_thresholds['volume_spike']:
                scores.append((volume_ratio - 1) * 0.5)
                reasons.append(f"{tf}: {volume_ratio:.1f}x volume spike")
        
        # Volatility and RSI analysis
        if len(df) >= 20:
            volatility = df['c'].pct_change().rolling(20).std().iloc[-1]
            if volatility > self.ai_trigger_thresholds['volatility']:
                scores.append(volatility * 10)
                reasons.append(f"{tf}: High volatility {volatility:.1%}")
        
        if 'rsi' in df.columns and not df['rsi'].empty:
            rsi = df['rsi'].iloc[-1]
            if rsi > 70 or rsi < 30:
                scores.append(0.3)
                reasons.append(f"{tf}: RSI extreme {rsi:.1f}")
        
        return scores, reasons
    
    async def analyze(self, context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
        """Main AI analysis entry point."""
        
        now = int(time.time() * 1000)
        local_tz = get_localzone()

        # Get candles for analysis
        candles_data = {
            tf: get_candles_with_cache(coin, tf, now, lookback, hyperliquid_utils.info.candles_snapshot)
            for tf, lookback in self.ai_timeframes.items()
        }

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


    def _generate_llm_prediction_prompt(
        self,
        coin: str, 
        dataframes: Dict[Timeframe, pd.DataFrame], 
        funding_rate: FundingRateEntry | None, 
        mid: float
    ) -> str:
        """
        Generate a prompt for LLM to predict price movements based on candles and funding rates.
        
        Args:
            coin: The cryptocurrency symbol
            dataframes: Dictionary of timeframes and their candle data
            funding_rate: Funding rate
            mid: Current mid price
            
        Returns:
            Formatted prompt string for LLM
        """
        prompt_parts = [
            f"Analyze the following cryptocurrency data for {coin} and predict the price in 1, 2, and 4 hours.",
            f"Current price: ${mid:.4f}",
            "",
            "=== MARKET CONTEXT ===",
            f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"24h price change: {self._calculate_24h_change(dataframes.get(Timeframe.HOUR_1), mid):.2f}%",
            "",
            "=== CANDLE DATA ===",
        ]    # Enhanced candle data with more candles and additional indicators
        timeframe_candle_counts = {
            Timeframe.MINUTES_15: 288,  # Last 72 hours (288 * 15min = 72h)
            Timeframe.MINUTES_30: 240,  # Last 120 hours (240 * 30min = 5 days)
            Timeframe.HOUR_1: 504,      # Last 21 days (504 * 1h = 21 days)
            Timeframe.HOURS_4: 252      # Last 42 days (252 * 4h = 42 days)
        }
        
        for timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]:
            if timeframe in dataframes:
                df = dataframes[timeframe]
                if not df.empty:
                    candle_count = timeframe_candle_counts.get(timeframe, 20)
                    recent_candles = df.tail(candle_count)
                    
                    prompt_parts.append(f"\n{timeframe.name} Timeframe (last {len(recent_candles)} candles):")
                    prompt_parts.append("Time | O | H | L | C | Vol | ATR | MACD | ST | BB_Up | BB_Low | Vol_Ratio")
                    
                    for idx, row in recent_candles.iterrows():
                        timestamp = idx.strftime("%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
                        prompt_parts.append(
                            f"{timestamp} | {row.get('o', 0):.4f} | {row.get('h', 0):.4f} | "
                            f"{row.get('l', 0):.4f} | {row.get('c', 0):.4f} | {row.get('v', 0):.0f} | "
                            f"{row.get('ATR', 0):.4f} | {row.get('MACD', 0):.4f} | {row.get('SuperTrend', 0):.4f} | "
                            f"{row.get('BB_upper', 0):.4f} | {row.get('BB_lower', 0):.4f} | {row.get('v_ratio', 1):.2f}"
                        )
                    
                    # Add timeframe summary
                    prompt_parts.extend([
                        f"Summary for {timeframe.name}:",
                        f"- Price range: ${recent_candles['l'].min():.4f} - ${recent_candles['h'].max():.4f}",
                        f"- Average volume: {recent_candles['v'].mean():.0f}",
                        f"- Current trend: {'Bullish' if recent_candles['c'].iloc[-1] > recent_candles['SuperTrend'].iloc[-1] else 'Bearish'}",
                        f"- Volatility (ATR): {recent_candles.get('ATR', pd.Series([0])).iloc[-1]:.4f}",
                        ""
                    ])
        
        # Enhanced funding rates data
        prompt_parts.extend([
            "=== FUNDING RATES ===",
            "Time | Rate | Premium | 8h_Avg | Trend"
        ])
            
        # Add market sentiment indicators
        funding_trend = "Bullish bias" if funding_rate and funding_rate.funding_rate > 0.0001 else "Bearish bias" if funding_rate and funding_rate.funding_rate < -0.0001 else "Neutral"
        
        prompt_parts.extend([
            "",
            "=== MARKET SENTIMENT ===",
            f"Current funding sentiment: {funding_trend}",
            f"Funding rate magnitude: {'High' if funding_rate and abs(funding_rate.funding_rate) > 0.0005 else 'Moderate' if funding_rate and abs(funding_rate.funding_rate) > 0.0001 else 'Low'}",
            ""
        ])
        # Enhanced prediction request with intraday-focused analysis
        prompt_parts.extend([
            "=== INTRADAY TRADING ANALYSIS FRAMEWORK ===",
            "Perform a comprehensive intraday-focused analysis optimized for Hyperliquid perpetual futures trading:",
            "",
            "1. INTRADAY MOMENTUM ANALYSIS:",
            "   - Analyze 15m and 30m momentum shifts and breakouts",
            "   - Identify short-term trend reversals and continuation patterns",
            "   - Evaluate volume surge patterns for scalping opportunities",
            "   - Assess price action around psychological levels and round numbers",
            "",
            "2. SCALPING OPPORTUNITIES:",
            "   - Identify 1-4 hour trading windows with high probability setups",
            "   - Analyze micro-structure breaks and pullback entries",
            "   - Evaluate range-bound trading opportunities and breakout plays",
            "   - Assess quick momentum trades with tight risk management",
            "",
            "3. SESSION-BASED ANALYSIS:",
            "   - Consider Asian, European, and US trading session characteristics",
            "   - Analyze opening range breakouts and session highs/lows",
            "   - Evaluate liquidity patterns during different market hours",
            "   - Factor in weekend positioning and session transitions",
            "",
            "4. WYCKOFF INTRADAY PHASES:",
            "   - Identify short-term accumulation/distribution within trends",
            "   - Analyze volume-price relationships for immediate entries",
            "   - Look for springs, upthrusts, and stopping volume on lower timeframes",
            "   - Determine smart money activity in current session",
            "",
            "5. ELLIOTT WAVE MICRO-COUNTS:",
            "   - Focus on 5-wave impulse moves within 1-4 hour windows",
            "   - Identify corrective patterns for counter-trend entries",
            "   - Use fibonacci levels for precise entry and exit timing",
            "",
            "6. MARKET STRUCTURE FOR SCALPING:",
            "   - Identify recent higher highs, higher lows patterns on 15m-1h",
            "   - Analyze immediate support/resistance and order blocks",
            "   - Look for liquidity sweeps and false breakouts to fade",
            "   - Evaluate supply/demand imbalances for quick trades",
            "",
            "7. FUNDING RATE ARBITRAGE:",
            "   - Assess funding rate implications for directional bias",
            "   - Identify funding-driven price movements near 8h intervals",
            "   - Consider perpetual futures premium/discount opportunities",
            "",
            "8. INTRADAY SCALPING STRATEGY FOCUS:",
            "   - Prioritize setups with 30-240 minute holding periods",
            "   - Focus on 1:2 or better risk-reward ratios for quick trades",
            "   - Identify momentum breakouts on 15m-1h timeframes",
            "   - Analyze mean reversion opportunities in ranging markets",
            "   - Look for volume confirmation on micro-structure breaks",
            "",
            "9. SESSION-SPECIFIC INTRADAY PATTERNS:",
            "   - Asian session: Range-bound trading, liquidity building",
            "   - European session: Initial breakouts, volatility increase",
            "   - US session: High volume moves, trend continuation/reversal",
            "   - Overlap periods: Maximum liquidity and volatility windows",
            "   - Weekend/holiday positioning effects on crypto markets",
            "",
            "10. HYPERLIQUID-SPECIFIC CONSIDERATIONS:",
            "   - Perpetual futures dynamics and basis trading opportunities",
            "   - Funding rate timing effects (every 1 hour)",
            "   - Order book depth analysis for slippage assessment",
            "   - Cross-margin vs isolated margin strategy implications",
            "   - Leverage considerations based on volatility regime",
            "",
            "=== PREDICTION REQUEST ===",
            "Based on your comprehensive analysis, provide your response in the following JSON format:",
            "",
            "{",
            '  "signal": "buy|sell|hold",',
            '  "confidence": 0.7,',
            '  "prediction": "bullish|bearish|sideways",',
            '  "risk_level": "low|medium|high",',
            '  "intraday_signal": "buy|sell|hold",',
            '  "intraday_confidence": 0.8,',
            '  "entry_price": 1234.56,',
            '  "stop_loss": 1200.00,',
            '  "target_price": 1300.00,',
            '  "time_horizon_hours": 4,',
            '  "wyckoff_analysis": "Current phase identification, volume analysis, and composite operator behavior insights...",',
            '  "elliott_wave_analysis": "Wave count, current position, and projected targets based on wave theory...",',
            '  "market_structure_analysis": "BOS/CHoCH analysis, supply/demand zones, and liquidity analysis...",',
            '  "market_analysis": "Brief market context, sentiment, and broader market conditions...",',
            '  "technical_analysis": "Multi-timeframe technical confluence, indicators, and pattern analysis...",',
            '  "risk_assessment": "Risk factors, position sizing, and risk management considerations...",',
            '  "key_drivers": ["wyckoff_phase", "elliott_wave_target", "structure_break", "volume_confirmation"],',
            '  "price_predictions": {',
            '    "1_hour": {"price": 1235.00, "change_percent": 0.5, "confidence": 0.6, "key_level": "resistance"},',
            '    "2_hours": {"price": 1240.00, "change_percent": 1.0, "confidence": 0.7, "key_level": "fibonacci_extension"},',
            '    "4_hours": {"price": 1250.00, "change_percent": 1.5, "confidence": 0.8, "key_level": "wave_target"}',
            '  },',
            '  "trading_setup": {',
            '    "action": "long|short|none",',
            '    "reason": "Comprehensive rationale based on multi-method confluence",',
            '    "entry_strategy": "Market/limit order strategy with precise timing",',
            '    "risk_reward_ratio": 2.5,',
            '    "position_size_recommendation": "Percentage of portfolio based on confidence and risk",',
            '    "invalidation_level": "Price level where analysis becomes invalid",',
            '    "max_position_risk": "Maximum account risk percentage for this trade",',
            '    "volatility_adjusted_position": "Position size adjusted for current volatility",',
            '    "session_timing": "Optimal entry timing based on trading sessions"',
            '  },',
            '  "intraday_risk_management": {',
            '    "account_risk_percent": 1.0,',
            '    "max_drawdown_protection": "Stop trading if daily loss exceeds X%",',
            '    "position_sizing_method": "ATR-based|fixed-percent|volatility-adjusted",',
            '    "quick_exit_triggers": ["volume_drying_up", "momentum_failure", "time_stop"],',
            '    "scale_out_levels": [{"level": 1220.0, "percentage": 50}, {"level": 1240.0, "percentage": 100}],',
            '    "max_holding_time": "Maximum time to hold position before forced exit",',
            '    "session_risk_limits": "Reduce position size during low-liquidity sessions"',
            '  },',
            '  "confluence_score": 0.85,',
            '  "method_agreement": {',
            '    "wyckoff": "bullish|bearish|neutral",',
            '    "elliott_wave": "bullish|bearish|neutral",',
            '    "market_structure": "bullish|bearish|neutral",',
            '    "technical_indicators": "bullish|bearish|neutral"',
            '  }',
            "}",
            "",
            "ANALYSIS RULES:",
            "- Only recommend BUY/SELL for intraday_signal if intraday_confidence >= 0.7 AND confluence_score >= 0.7",
            "- Provide entry_price, stop_loss, and target_price only for high-confidence actionable signals",
            "- Base analysis on multi-method confluence - higher agreement = higher confidence",
            "- Include specific Wyckoff phases, Elliott Wave counts, and structural levels",
            "- Provide detailed reasoning for each analytical method used",
            "- Invalidation levels are crucial for risk management",
            "- Confluence_score should reflect agreement between different methods (0.0-1.0)",
            "- Each analysis section should reference specific technical concepts and levels",
            "",
            "INTRADAY TRADING RULES:",
            "- Prioritize scalping opportunities with 30-240 minute holding periods",
            "- Focus on session-based analysis (Asian/European/US market hours)",
            "- Identify micro-momentum breaks on 15m-1h timeframes",
            "- Look for volume confirmation on all intraday entries",
            "- Consider funding rate timing for 8-hour intervals",
            "- Use tight stops (0.5-2% max) for intraday positions",
            "- Minimum risk-reward ratio of 1:2 for intraday signals",
            "",
            "ENHANCED RISK MANAGEMENT RULES:",
            "- Account risk should never exceed 2% per trade",
            "- Position size must be adjusted for current volatility (ATR-based)",
            "- Include daily maximum drawdown limits (3-5% account stop)",
            "- Provide specific exit triggers beyond stop loss",
            "- Scale out at predetermined levels to lock profits",
            "- Reduce position sizes during low-liquidity sessions",
            "- Include time-based stops for momentum failures",
            "- Consider session-specific risk adjustments"
        ])
        
        return "\n".join(prompt_parts)


    def _calculate_24h_change(self, hourly_df: pd.DataFrame | None, current_price: float) -> float:
        """Calculate 24-hour price change percentage."""
        if hourly_df is None or hourly_df.empty or len(hourly_df) < 24:
            return 0.0
        
        try:
            price_24h_ago = hourly_df.iloc[-24]['c']
            return ((current_price - price_24h_ago) / price_24h_ago) * 100
        except (IndexError, KeyError, ZeroDivisionError):
            return 0.0


    async def _perform_ai_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str) -> AIAnalysisResult:
        """Core AI analysis logic using OpenRouter.ai."""

        try:
            # Get current market data
            mid_price = dataframes[Timeframe.MINUTES_15]['c'].iloc[-1] if not dataframes[Timeframe.MINUTES_15].empty else 0.0
            now = int(time.time() * 1000)
            funding_rates = get_funding_with_cache(coin, now, 1)  # Get 1 day of funding rate history
            funding_rate = funding_rates[-1] if funding_rates else None
            
            # Generate prompt for LLM
            prompt = self._generate_llm_prediction_prompt(coin, dataframes, funding_rate, mid_price)
            
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
            return AIAnalysisResult(
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
        ai_result: AIAnalysisResult
    ) -> None:
        """Send AI analysis results to Telegram."""
        
        # Build analysis message
        message = self._build_analysis_message(coin, ai_result)
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)
    
    def _build_analysis_message(self, coin: str, ai_result: AIAnalysisResult) -> str:
        """Build the analysis message text."""
        message = (
            f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n"
            f"📊 <b>Signal:</b> {ai_result.signal.title()}\n"
            f"🎯 <b>Confidence:</b> {ai_result.confidence:.1%}\n"
            f"📈 <b>Prediction:</b> {ai_result.prediction.title()}\n"
            f"⚠️ <b>Risk Level:</b> {ai_result.risk_level.title()}\n"
        )
        
        # Add intraday signal section
        message += self._build_intraday_section(coin, ai_result)
        
        # Add analysis cost information
        if ai_result.analysis_cost > 0:
            message += f"\n💰 <b>Analysis Cost:</b> {fmt_price(ai_result.analysis_cost)} $"
        
        # Add individual analysis sections for better readability
        if ai_result.market_analysis:
            message += f"\n\n📊 <b>Market Analysis:</b>\n{ai_result.market_analysis}"
        
        if ai_result.technical_analysis:
            message += f"\n\n🔧 <b>Technical Analysis:</b>\n{ai_result.technical_analysis}"
        
        if ai_result.risk_assessment:
            message += f"\n\n⚠️ <b>Risk Assessment:</b>\n{ai_result.risk_assessment}"
                
        return message
    
    def _build_intraday_section(self, coin: str, ai_result: AIAnalysisResult) -> str:
        """Build the intraday trading section of the message."""
        if ai_result.intraday_confidence < 0.6 or ai_result.intraday_signal == "hold":
            return ""
        
        section = (
            f"\n💼 <b>Intraday Signal:</b> {ai_result.intraday_signal.title()}\n"
            f"🎯 <b>Intraday Confidence:</b> {ai_result.intraday_confidence:.1%}"
        )
        
        trade_setup = self._build_trade_setup_format(coin, ai_result)
        if trade_setup:
            section += trade_setup
        
        return section
    
    def _build_trade_setup_format(self, coin: str, ai_result: AIAnalysisResult) -> str:
        """Build formatted trade setup."""
        if not (ai_result.entry_price > 0 and (ai_result.stop_loss > 0 or ai_result.target_price > 0)):
            return ""

        enc_side = "L" if ai_result.intraday_signal == "buy" else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(ai_result.stop_loss)}_{fmt_price(ai_result.target_price)}".encode('utf-8')).decode('utf-8')
        trade_link = f"({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

        side = "Long" if ai_result.intraday_signal == "buy" else "Short"

        setup = f"\n\n<b>💰 {side} Trade Setup</b>{trade_link}<b>:</b>"
        
        # Market price (entry price)
        if ai_result.entry_price > 0:
            setup += f"\nMarket price: {ai_result.entry_price:.4f} USDC"
        
        # Stop Loss with percentage
        if ai_result.stop_loss > 0 and ai_result.entry_price > 0:
            if ai_result.intraday_signal == "buy":
                sl_percentage = ((ai_result.stop_loss - ai_result.entry_price) / ai_result.entry_price) * 100
            else:  # short
                sl_percentage = ((ai_result.entry_price - ai_result.stop_loss) / ai_result.entry_price) * 100
            
            setup += f"\nStop Loss: {ai_result.stop_loss:.4f} USDC ({sl_percentage:+.1f}%)"
        
        # Take Profit with percentage
        if ai_result.target_price > 0 and ai_result.entry_price > 0:
            if ai_result.intraday_signal == "buy":
                tp_percentage = ((ai_result.target_price - ai_result.entry_price) / ai_result.entry_price) * 100
            else:  # short
                tp_percentage = ((ai_result.entry_price - ai_result.target_price) / ai_result.entry_price) * 100
            
            setup += f"\nTake Profit: {ai_result.target_price:.4f} USDC ({tp_percentage:+.1f}%)"
        
        return setup


    def _build_price_levels(self, ai_result: AIAnalysisResult) -> str:
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
        
        model = os.getenv("HTB_OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        max_tokens = 1500
        
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
            "response_format": {
                "type": "json_object"
            },
            "usage": {
                "include": "true"
            },
            "max_tokens": max_tokens
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
    
    def _parse_ai_response(self, ai_response: str, coin: str) -> AIAnalysisResult:
        """Parse AI response into structured analysis result."""
        try:
            # Try to parse as JSON first
            response_data = json.loads(ai_response)
            
            # Extract values from JSON response
            signal = response_data.get("signal", "hold").lower()
            confidence = float(response_data.get("confidence", 0.5))
            prediction = response_data.get("prediction", "sideways").lower()
            risk_level = response_data.get("risk_level", "medium").lower()
            intraday_signal = response_data.get("intraday_signal", "hold").lower()
            intraday_confidence = float(response_data.get("intraday_confidence", 0.5))
            entry_price = float(response_data.get("entry_price") or 0.0)
            stop_loss = float(response_data.get("stop_loss") or 0.0)
            target_price = float(response_data.get("target_price") or 0.0)
            
            # Extract structured analysis sections
            market_analysis = response_data.get("market_analysis", "")
            technical_analysis = response_data.get("technical_analysis", "")
            risk_assessment = response_data.get("risk_assessment", "")
            wyckoff_analysis = response_data.get("wyckoff_analysis", "")
            elliott_wave_analysis = response_data.get("elliott_wave_analysis", "")
            market_structure_analysis = response_data.get("market_structure_analysis", "")
            key_drivers = response_data.get("key_drivers", [])
            price_predictions = response_data.get("price_predictions", {})
            
            # Build description from structured sections
            description_parts = []
            if wyckoff_analysis:
                description_parts.append(f"🧭 Wyckoff: {wyckoff_analysis}")
            if elliott_wave_analysis:
                description_parts.append(f"🌊 Elliott Wave: {elliott_wave_analysis}")
            if market_structure_analysis:
                description_parts.append(f"🏗️ Market Structure: {market_structure_analysis}")
            if market_analysis:
                description_parts.append(f"📊 Market: {market_analysis}")
            if technical_analysis:
                description_parts.append(f"🔧 Technical: {technical_analysis}")
            if risk_assessment:
                description_parts.append(f"⚠️ Risk: {risk_assessment}")
            
            description = "\n\n".join(description_parts) if description_parts else response_data.get("analysis", "Analysis not available")
            
            # Determine if we should notify based on signal strength
            min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
            should_notify = signal in ["buy", "sell"] and confidence >= min_confidence
            
            return AIAnalysisResult(
                signal=signal,
                confidence=confidence,
                prediction=prediction,
                risk_level=risk_level,
                should_notify=should_notify,
                description=description,
                intraday_signal=intraday_signal,
                intraday_confidence=intraday_confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                key_drivers=key_drivers,
                price_predictions=price_predictions,
                market_analysis=market_analysis,
                technical_analysis=technical_analysis,
                risk_assessment=risk_assessment
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"JSON parsing failed for {coin}: {str(e)}\n{ai_response}", exc_info=True)
            return AIAnalysisResult(
                description=f"AI analysis for {coin}: Analysis failed due to technical error. Using fallback analysis.",
                signal="hold",
                confidence=0.5,
                analysis_cost=0.0
            )
