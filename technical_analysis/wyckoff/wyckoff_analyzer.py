import io
import time
from typing import Dict, List, Any
import pandas as pd
from tzlocal import get_localzone

from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from ..candles_cache import get_candles_with_cache
from .wyckoff_types import Timeframe, WyckoffState, SignificantLevelsData
from ..funding_rates_cache import get_funding_with_cache, FundingRateEntry
from .wykcoff_chart import generate_chart
from .mtf.wyckoff_multi_timeframe import analyze_multi_timeframe
from ..data_processor import prepare_dataframe, apply_indicators
from .significant_levels import find_significant_levels
from .wyckoff import detect_wyckoff_phase


class WyckoffAnalyzer:
    """Wyckoff-based technical analysis implementation."""
    
    def __init__(self):
        self.wyckoff_timeframes = {
            Timeframe.MINUTES_15: 28,
            Timeframe.MINUTES_30: 42,
            Timeframe.HOUR_1: 60,
            Timeframe.HOURS_2: 75,
            Timeframe.HOURS_4: 90,
            Timeframe.HOURS_8: 120
        }
    
    async def analyze(self, context: ContextTypes.DEFAULT_TYPE, coin: str, interactive_analysis: bool) -> None:
        """Main Wyckoff analysis entry point."""
        
        now = int(time.time() * 1000)
        funding_rates = get_funding_with_cache(coin, now, 7)
        local_tz = get_localzone()

        # Get candles for all timeframes
        candles_data = await self._get_candles_for_timeframes(coin, now)

        # Check if we have enough data for basic analysis
        if len(candles_data[Timeframe.MINUTES_15]) < 10:
            logger.warning(f"Insufficient candles for technical analysis on {coin}")
            return

        # Prepare dataframes and analyze states
        dataframes = {
            tf: prepare_dataframe(candles, local_tz) 
            for tf, candles in candles_data.items()
        }

        states = {
            tf: self._analyze_timeframe_data(df, tf, funding_rates, local_tz)
            for tf, df in dataframes.items()
        }

        # Add multi-timeframe analysis
        mid = float(hyperliquid_utils.info.all_mids()[coin])
        significant_levels = self._calculate_significant_levels(dataframes, states, mid)
        
        mtf_context = analyze_multi_timeframe(states, coin, mid, significant_levels, interactive_analysis)

        should_notify = interactive_analysis or mtf_context.should_notify

        if should_notify:
            await self._send_wyckoff_analysis_message(
                context, mid, dataframes, states, coin, interactive_analysis, mtf_context.description
            )
    
    async def _get_candles_for_timeframes(self, coin: str, now: int) -> Dict[Timeframe, List[Dict[str, Any]]]:
        """Get candles data for all timeframes with optimized lookback periods."""
        candles_data = {}
        for tf, lookback in self.wyckoff_timeframes.items():
            candles_data[tf] = await get_candles_with_cache(coin, tf, now, lookback, hyperliquid_utils.info.candles_snapshot)
        return candles_data
    
    def _analyze_timeframe_data(self, df: pd.DataFrame, timeframe: Timeframe, funding_rates: List[FundingRateEntry], local_tz) -> WyckoffState:
        """Process data for a single timeframe."""
        if df.empty:
            return WyckoffState.unknown()
        
        apply_indicators(df, timeframe)
        return detect_wyckoff_phase(df, timeframe, funding_rates)
    
    def _calculate_significant_levels(
        self, 
        dataframes: Dict[Timeframe, pd.DataFrame], 
        states: Dict[Timeframe, WyckoffState], 
        mid: float
    ) -> Dict[Timeframe, SignificantLevelsData]:
        """Calculate significant levels for specified timeframes."""
        significant_timeframes = [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]
        return {
            tf: {
                'resistance': resistance,
                'support': support
            }
            for tf in significant_timeframes
            for resistance, support in [find_significant_levels(dataframes[tf], states[tf], mid, tf)]
        }
    
    async def _send_wyckoff_analysis_message(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        mid: float, 
        dataframes: Dict[Timeframe, pd.DataFrame], 
        states: Dict[Timeframe, WyckoffState], 
        coin: str, 
        send_charts: bool, 
        mtf_description: str
    ) -> None:
        """Send Wyckoff analysis results to Telegram."""
        
        if send_charts:
            charts = []
            try:
                charts = generate_chart(dataframes, states, coin, mid)
                
                results_15m = self._get_ta_results(dataframes[Timeframe.MINUTES_15], states[Timeframe.MINUTES_15])
                results_1h = self._get_ta_results(dataframes[Timeframe.HOUR_1], states[Timeframe.HOUR_1])
                results_4h = self._get_ta_results(dataframes[Timeframe.HOURS_4], states[Timeframe.HOURS_4])

                no_wyckoff_data_available = 'No Wyckoff data available'

                # Send all charts in sequence, using copies of the buffers
                for idx, (chart, period, results) in enumerate([
                    (charts[2], "4h", results_4h),
                    (charts[1], "1h", results_1h),
                    (charts[0], "15m", results_15m)
                ]):
                    wyckoff_description = results['wyckoff'].description if results.get('wyckoff') else no_wyckoff_data_available
                    caption = f"<b>{period} indicators:</b>\n{wyckoff_description}"
                    
                    if chart:
                        # Create a copy of the buffer's contents
                        chart_copy = io.BytesIO(chart.getvalue())
                        
                        try:
                            if len(caption) >= 1024:
                                # Send chart and caption separately if caption is too long
                                await context.bot.send_photo(
                                    chat_id=telegram_utils.telegram_chat_id, # type: ignore
                                    photo=chart_copy,
                                    caption=f"<b>{period} chart</b>",
                                    parse_mode=ParseMode.HTML
                                )
                                await telegram_utils.send(caption, parse_mode=ParseMode.HTML)
                            else:
                                # Send together if caption is within limits
                                await context.bot.send_photo(
                                    chat_id=telegram_utils.telegram_chat_id, # type: ignore
                                    photo=chart_copy,
                                    caption=caption,
                                    parse_mode=ParseMode.HTML
                                )
                        finally:
                            chart_copy.close()
                    else:
                        await telegram_utils.send(caption, parse_mode=ParseMode.HTML)

            finally:
                # Clean up the original buffers
                for chart in charts:
                    if chart and not chart.closed:
                        chart.close()
        
        # Add MTF analysis
        await telegram_utils.send(
            f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n"
            f"{mtf_description}",
            parse_mode=ParseMode.HTML
        )
    
    def _get_ta_results(self, df: pd.DataFrame, wyckoff: WyckoffState) -> Dict[str, Any]:
        """Get technical analysis results for a timeframe."""
        # Check if we have enough data points
        if len(df["SuperTrend"]) < 2:
            logger.warning("Insufficient data for technical analysis results")
            return {
                "supertrend_prev": 0,
                "supertrend": 0,
                "supertrend_trend_prev": "unknown",
                "supertrend_trend": "unknown",
                "wyckoff": None
            }

        supertrend_prev, supertrend = df["SuperTrend"].iloc[-2], df["SuperTrend"].iloc[-1]

        return {
            "supertrend_prev": supertrend_prev,
            "supertrend": supertrend,
            "supertrend_trend_prev": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-2] else "downtrend",
            "supertrend_trend": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-1] else "downtrend",
            "wyckoff": wyckoff
        }