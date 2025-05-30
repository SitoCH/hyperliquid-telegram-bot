import base64
from typing import Dict, Any
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from utils import fmt_price, exchange_enabled
from telegram_utils import telegram_utils
from .llm_analysis_result import LLMAnalysisResult


class LLMMessageFormatter:
    """Handles formatting and sending of LLM analysis messages."""
    
    async def send_llm_analysis_message(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        coin: str,
        current_price: float,
        llm_result: LLMAnalysisResult
    ) -> None:
        """Send AI analysis results to Telegram."""
        
        # Build analysis message
        message = self._build_analysis_message(coin, current_price, llm_result)
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)

    def _build_analysis_message(self, coin: str, current_price: float, llm_result: LLMAnalysisResult) -> str:
        """Build the analysis message text."""

        # Get emoji based on signal/prediction
        if llm_result.signal == "long" or "bullish" in llm_result.prediction.lower():
            direction_emoji = "📈"
        elif llm_result.signal == "short" or "bearish" in llm_result.prediction.lower():
            direction_emoji = "📉"
        else:
            direction_emoji = "📊"

        message = f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n\n"
        message += f"<b>{direction_emoji} Market Analysis:</b> {llm_result.recap_heading}\n\n"
        
        # Add signal information
        message += (
            f"📊 <b>Signal:</b> {llm_result.signal.lower()}\n"
            f"🎯 <b>Confidence:</b> {llm_result.confidence:.0%}\n"
            f"📈 <b>Prediction:</b> {llm_result.prediction.lower()}\n"
            f"⚠️ <b>Risk Level:</b> {llm_result.risk_level.lower()}\n"
            f"⏰ <b>Time Horizon:</b> {llm_result.time_horizon_hours}h"
        )

        # Add trading insight if available
        trading_insight = getattr(llm_result, 'trading_insight', '')
        if trading_insight:
            message += f"\n\n💡 <b>Trading Insight:</b>\n{trading_insight}"

        # Add key drivers if available
        if llm_result.key_drivers:
            message += "\n\n🔑 <b>Key Drivers:</b>"
            for driver in llm_result.key_drivers:
                message += f"\n• {driver}"

        # Add timeframe signals if available
        if llm_result.timeframe_signals:
            timeframe_section = self._format_timeframe_signals(llm_result.timeframe_signals, False)
            if timeframe_section:
                message += f"\n\n📊 <b>Timeframe Signals:</b>{timeframe_section}"
        
        trade_setup = self._build_trade_setup_format(coin, current_price, llm_result)
        if trade_setup:
            message += trade_setup
           
        return message

    def _build_trade_setup_format(self, coin: str, current_price: float, llm_result: LLMAnalysisResult) -> str:
        """Build formatted trade setup."""
        if not (current_price > 0 and (llm_result.stop_loss > 0 or llm_result.target_price > 0)):
            return ""

        enc_side = "L" if llm_result.signal == "long" else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(llm_result.stop_loss)}_{fmt_price(llm_result.target_price)}".encode('utf-8')).decode('utf-8')
        trade_link = f"({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

        side = "Long" if llm_result.signal == "long" else "Short"

        setup = f"\n\n<b>💰 {side} Trade Setup</b>{trade_link}<b>:</b>"
        
        setup += f"\nMarket price: {fmt_price(current_price)} USDC"
        
        # Stop Loss with percentage
        if llm_result.stop_loss > 0:
            if llm_result.signal == "long":
                sl_percentage = ((llm_result.stop_loss - current_price) / current_price) * 100
            else:  # short
                sl_percentage = ((current_price - llm_result.stop_loss) / current_price) * 100
            
            setup += f"\nStop Loss: {fmt_price(llm_result.stop_loss)} USDC ({sl_percentage:+.1f}%)"
        
        # Take Profit with percentage
        if llm_result.target_price > 0:
            if llm_result.signal == "long":
                tp_percentage = ((llm_result.target_price - current_price) / current_price) * 100
            else:  # short
                tp_percentage = ((current_price - llm_result.target_price) / current_price) * 100
            
            setup += f"\nTake Profit: {fmt_price(llm_result.target_price)} USDC ({tp_percentage:+.1f}%)"
        
        return setup

    def _build_price_levels(self, current_price: float, llm_result: LLMAnalysisResult) -> str:
        """Build price levels section."""
        if not (current_price > 0 or llm_result.stop_loss > 0 or llm_result.target_price > 0):
            return ""
        
        levels = "\n📋 <b>Price Levels:</b>"
        levels += f"\n   Entry: ${fmt_price(current_price)}"
        if llm_result.stop_loss > 0:
            levels += f"\n   Stop Loss: ${fmt_price(current_price)}"
        if llm_result.target_price > 0:
            levels += f"\n   Target: ${fmt_price(current_price)}"
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