import time
import os
import json
import pandas as pd
from typing import Dict, Any, List
import re
from tzlocal import get_localzone
import requests
from datetime import datetime
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from logging_utils import logger
from hyperliquid_utils.utils import hyperliquid_utils
from ..candles_cache import get_candles_with_cache
from ..wyckoff.wyckoff_types import Timeframe
from ..data_processor import prepare_dataframe, apply_indicators
from ..funding_rates_cache import get_funding_with_cache, FundingRateEntry
from .prompt_generator import LLMPromptGenerator
from .analysis_filter import AnalysisFilter
from .openrouter_client import OpenRouterClient
from .message_formatter import LLMMessageFormatter

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
        self.timeframe_lookback_days = {
            Timeframe.MINUTES_15: 5,
            Timeframe.MINUTES_30: 5,
            Timeframe.HOUR_1: 14,
            Timeframe.HOURS_4: 21,
        }
        self.prompt_generator = LLMPromptGenerator(self.timeframe_lookback_days)
        self.analysis_filter = AnalysisFilter()
        self.openrouter_client = OpenRouterClient()
        self.message_formatter = LLMMessageFormatter()
    
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

        should_analyze, _ = self.analysis_filter.should_run_llm_analysis(dataframes, coin, interactive_analysis)
        
        if not should_analyze:
            return

        llm_result = await self._perform_llm_analysis(dataframes, coin)

        should_notify = interactive_analysis or llm_result.should_notify
        
        if should_notify:
            mid = float(hyperliquid_utils.info.all_mids()[coin])
            await self.message_formatter.send_llm_analysis_message(context, coin, mid, llm_result)

    async def _perform_llm_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str) -> LLMAnalysisResult:
        """Core AI analysis logic using OpenRouter.ai."""

        try:
            # Get current market data
            mid_price = dataframes[Timeframe.MINUTES_15]['c'].iloc[-1] if not dataframes[Timeframe.MINUTES_15].empty else 0.0
            now = int(time.time() * 1000)
            funding_rates = get_funding_with_cache(coin, now, 5)
            
            model = os.getenv("HTB_OPENROUTER_MAIN_MODEL", "openai/gpt-4.1-nano")
            prompt = self.prompt_generator.generate_prediction_prompt(coin, dataframes, funding_rates, mid_price)

            llm_response = self.openrouter_client.call_api(model, prompt)
            
            # Parse AI response into structured result
            result = self._parse_llm_response(llm_response, coin)
            
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
                confidence=0.5
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
    
    def _parse_llm_response(self, llm_response: str, coin: str) -> LLMAnalysisResult:
        """Parse AI response into structured analysis result."""
        try:
            # Try to parse as JSON first
            response_data = json.loads(llm_response)
            
            # Extract values from JSON response
            signal = response_data.get("signal", "unknown").lower()
            confidence = float(response_data.get("confidence", 0.0))
            prediction = response_data.get("prediction", "unknown").lower()
            risk_level = response_data.get("risk_level", "unknown").lower()
            stop_loss = float(response_data.get("stop_loss") or 0.0)
            target_price = float(response_data.get("target_price") or 0.0)
                        
            # Extract new fields
            recap_heading = response_data.get("recap_heading", "")
            trading_insight = response_data.get("trading_insight", "")
            key_drivers = response_data.get("key_drivers", [])
            
            # Simple description from response or fallback
            description = response_data.get("analysis", "unknown")
            
            # Determine if we should notify based on signal strength
            min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
            should_notify = signal in ["long", "short"] and confidence >= min_confidence
            
            return LLMAnalysisResult(
                signal=signal,
                confidence=confidence,
                prediction=prediction,
                risk_level=risk_level,
                should_notify=should_notify,
                description=description,
                stop_loss=stop_loss,
                target_price=target_price,
                key_drivers=key_drivers,
                recap_heading=recap_heading,
                trading_insight=trading_insight
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"JSON parsing failed for {coin}: {str(e)}\n{llm_response}", exc_info=True)
            return LLMAnalysisResult(
                description=f"AI analysis for {coin}: Analysis failed due to technical error. Using fallback analysis.",
                signal="hold",
                confidence=0.5
            )
