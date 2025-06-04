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
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from ..candles_cache import get_candles_with_cache
from ..wyckoff.wyckoff_types import Timeframe
from ..data_processor import prepare_dataframe, apply_indicators
from ..funding_rates_cache import get_funding_with_cache, FundingRateEntry
from .prompt_generator import LLMPromptGenerator
from .analysis_filter import AnalysisFilter
from .litellm_client import LiteLLMClient
from .message_formatter import LLMMessageFormatter
from .llm_analysis_result import LLMAnalysisResult, LLMAnalysisTradingSetup, Signal, Prediction, RiskLevel
from html import escape as escape_html


class LLMAnalyzer:
    """LLM-based technical analysis implementation."""
    
    def __init__(self):
        self.timeframe_lookback_days = {
            Timeframe.MINUTES_15: 14,
            Timeframe.MINUTES_30: 14,
            Timeframe.HOUR_1: 21,
            Timeframe.HOURS_4: 30,
        }
        self.prompt_generator = LLMPromptGenerator(self.timeframe_lookback_days)
        self.analysis_filter = AnalysisFilter()
        self.llm_client = LiteLLMClient()
        self.message_formatter = LLMMessageFormatter()


    async def send_llm_analysis_filter_message(self, coin, reason, confidence):
        message = f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n\n"
        message += f"<b>Market Analysis:</b> {escape_html(reason)}\n\n"
        message += f"🎯 <b>Confidence:</b> {confidence:.0%}\n"
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)


    async def analyze(self, context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
        """Main LLM analysis entry point."""
        
        now = int(time.time() * 1000)
        local_tz = get_localzone()

        # Get candles for analysis - use lookback days directly
        candles_data = {}
        for tf, lookback_days in self.timeframe_lookback_days.items():
            candles_data[tf] = await get_candles_with_cache(coin, tf, now, lookback_days, hyperliquid_utils.info.candles_snapshot)

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
        funding_rates = get_funding_with_cache(coin, now, 5)
        should_analyze, reason, confidence = await self.analysis_filter.should_run_llm_analysis(dataframes, coin, interactive_analysis, funding_rates)
        
        if should_analyze:
            if not interactive_analysis:
                self.send_llm_analysis_filter_message(coin, reason, confidence)
                return
        else:
            if interactive_analysis:
                self.send_llm_analysis_filter_message(coin, reason, confidence)
            return

        mid = float(hyperliquid_utils.info.all_mids()[coin])
        llm_result = await self._perform_llm_analysis(dataframes, coin, mid, funding_rates)

        should_notify = interactive_analysis or llm_result.should_notify
        
        if should_notify:
            await self.message_formatter.send_llm_analysis_message(context, coin, mid, llm_result)

    async def _perform_llm_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, mid: float, funding_rates: List[FundingRateEntry]) -> LLMAnalysisResult:
        """Core LLM analysis logic"""
        
        model = os.getenv("HTB_LLM_MAIN_MODEL", "unknown")
        prompt = self.prompt_generator.generate_prediction_prompt(coin, dataframes, funding_rates, mid)
        
        llm_response = await self.llm_client.call_api(model, prompt)
        
        return self._parse_llm_response(llm_response)


    def _parse_llm_response(self, llm_response: str) -> LLMAnalysisResult:
        """Parse AI response into structured analysis result."""

        response_data = json.loads(llm_response)
        signal = Signal(response_data.get("signal", "hold").lower())
        confidence = float(response_data.get("confidence", 0.0))
        prediction = Prediction(response_data.get("prediction", "sideways").lower())
        risk_level = RiskLevel(response_data.get("risk_level", "medium").lower())
        
        # Extract new fields
        recap_heading = response_data.get("recap_heading", "")
        trading_insight = response_data.get("trading_insight", "")
        key_drivers = response_data.get("key_drivers", [])
        time_horizon_hours = int(response_data.get("time_horizon_hours", 4))
        
        # Parse trading setup
        trading_setup = None
        trading_setup_data = response_data.get("trading_setup")
        if trading_setup_data:
            trading_setup = LLMAnalysisTradingSetup(
                stop_loss=float(trading_setup_data.get("stop_loss", 0.0)),
                take_profit=float(trading_setup_data.get("take_profit", 0.0))
            )

        # Determine if we should notify based on signal strength
        min_confidence = float(os.getenv("HTB_COINS_ANALYSIS_MIN_CONFIDENCE", "0.65"))
        should_notify = signal in [Signal.LONG, Signal.SHORT] and confidence >= min_confidence
        
        return LLMAnalysisResult(
            signal=signal,
            confidence=confidence,
            prediction=prediction,
            risk_level=risk_level,
            should_notify=should_notify,
            key_drivers=key_drivers,
            recap_heading=recap_heading,
            trading_insight=trading_insight,
            time_horizon_hours=time_horizon_hours,
            trading_setup=trading_setup
        )
