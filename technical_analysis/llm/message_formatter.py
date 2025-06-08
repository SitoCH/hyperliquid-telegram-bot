import base64
from typing import Dict, Any
from telegram.ext import ContextTypes
from telegram.constants import ParseMode
from utils import fmt_price, exchange_enabled
from telegram_utils import telegram_utils
from .llm_analysis_result import LLMAnalysisResult, LLMAnalysisTradingSetup, Signal


class LLMMessageFormatter:
    """Handles formatting and sending of LLM analysis messages."""
    
    async def send_llm_analysis_message(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        coin: str,
        current_price: float,
        llm_result: LLMAnalysisResult,
        short_analysis: bool
    ) -> None:
        """Send AI analysis results to Telegram."""
        
        # Build analysis message
        message = self._build_analysis_message(coin, current_price, llm_result, short_analysis)
        await telegram_utils.send(message, parse_mode=ParseMode.HTML)

    def _build_analysis_message(self, coin: str, current_price: float, llm_result: LLMAnalysisResult, short_analysis: bool) -> str:
        """Build the analysis message text."""

        # Get emoji based on signal
        if llm_result.signal == Signal.LONG:
            direction_emoji = "📈"
        elif llm_result.signal == Signal.SHORT:
            direction_emoji = "📉"
        else:
            direction_emoji = "📊"

        message = f"<b>Technical analysis for {telegram_utils.get_link(coin, f'TA_{coin}')}</b>\n\n"
        message += f"<b>{direction_emoji} Market Analysis:</b> {llm_result.recap_heading}\n\n"
        
        # Add signal information
        message += (
            f"📊 <b>Signal:</b> {llm_result.signal.value}\n"
            f"🎯 <b>Confidence:</b> {llm_result.confidence:.0%}\n"
        )

        if short_analysis:
            return message

        message += (
            f"⚠️ <b>Risk Level:</b> {llm_result.risk_level.value}\n"
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
        
        trade_setup = self._build_trade_setup_format(coin, current_price, llm_result)
        if trade_setup:
            message += trade_setup
           
        return message

    def _build_trade_setup_format(self, coin: str, current_price: float, llm_result: LLMAnalysisResult) -> str:
        """Build formatted trade setup."""

        trading_setup = llm_result.trading_setup

        if not trading_setup or not (current_price > 0 and (trading_setup.stop_loss > 0 or trading_setup.take_profit > 0)):
            return ""

        enc_side = "L" if llm_result.signal == Signal.LONG else "S"
        enc_trade = base64.b64encode(f"{enc_side}_{coin}_{fmt_price(trading_setup.stop_loss)}_{fmt_price(trading_setup.take_profit)}".encode('utf-8')).decode('utf-8')
        trade_link = f"({telegram_utils.get_link('Trade',f'TRD_{enc_trade}')})" if exchange_enabled else ""

        side = "Long" if llm_result.signal == Signal.LONG else "Short"

        setup = f"\n\n<b>💰 {side} Trade Setup</b>{trade_link}<b>:</b>"
        setup += f"\nMarket price: {fmt_price(current_price)} USDC"

        # Stop Loss with percentage
        if trading_setup.stop_loss > 0:
            if llm_result.signal == Signal.LONG:
                sl_percentage = ((trading_setup.stop_loss - current_price) / current_price) * 100
            else:  # short
                sl_percentage = ((current_price - trading_setup.stop_loss) / current_price) * 100
            
            setup += f"\nStop Loss: {fmt_price(trading_setup.stop_loss)} USDC ({sl_percentage:+.1f}%)"
        
        # Take Profit with percentage
        if trading_setup.take_profit > 0:
            if llm_result.signal == Signal.LONG:
                tp_percentage = ((trading_setup.take_profit - current_price) / current_price) * 100
            else:  # short
                tp_percentage = ((current_price - trading_setup.take_profit) / current_price) * 100
            
            setup += f"\nTake Profit: {fmt_price(trading_setup.take_profit)} USDC ({tp_percentage:+.1f}%)"
        
        return setup
