import time
import os
import json
import base64
import pandas as pd
from typing import Dict, Any, List
import re
from tzlocal import get_localzone
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
from .analysis_filter import AnalysisFilter
from .openrouter_client import OpenRouterClient

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
            Timeframe.MINUTES_15: 5,
            Timeframe.MINUTES_30: 5,
            Timeframe.HOUR_1: 10,
            Timeframe.HOURS_4: 14,    
        }          
        # Initialize prompt generator, analysis filter, and OpenRouter client
        self.prompt_generator = LLMPromptGenerator(self.timeframe_lookback_days)
        self.analysis_filter = AnalysisFilter()
        self.openrouter_client = OpenRouterClient()
    
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
        }        # Apply basic indicators for AI analysis
        for tf, df in dataframes.items():
            if not df.empty:
                apply_indicators(df, tf)
        
        # Pre-filter: Check if AI analysis is needed
        should_analyze, filter_reason = self.analysis_filter.should_run_llm_analysis(dataframes, coin, interactive_analysis)
        
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
            ai_response, analysis_cost = self.openrouter_client.call_api(prompt)
            
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

        # Get emoji based on signal/prediction
        if ai_result.signal == "buy" or "bullish" in ai_result.prediction.lower() or "up" in ai_result.prediction.lower():
            direction_emoji = "📈"
        elif ai_result.signal == "sell" or "bearish" in ai_result.prediction.lower() or "down" in ai_result.prediction.lower():
            direction_emoji = "📉"
        else:
            direction_emoji = "📊"

        message = f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n\n"
        message += f"<b>{direction_emoji} Market Analysis:</b> {ai_result.recap_heading}\n\n"
        
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
