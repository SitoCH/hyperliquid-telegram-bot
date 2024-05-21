import asyncio
import logging
import json
import os
from typing import List

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import UserEventsMsg, Fill

from telegram import (
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class HyperliquidBot:

    def __init__(self):

        telegram_token = os.environ["HYPERLIQUID_TELEGRAM_BOT_TOKEN"]
        self.telegram_chat_id = os.environ["HYPERLIQUID_TELEGRAM_BOT_CHAT_ID"]

        self.user_address = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER"]

        self.hyperliquid_info = Info(constants.MAINNET_API_URL)
        self.hyperliquid_info.subscribe(
            {"type": "userEvents", "user": self.user_address}, self.on_user_events
        )

        self.telegram_app = Application.builder().token(telegram_token).build()

        self.telegram_app.add_handler(CommandHandler("start", self.start))
        self.telegram_app.add_handler(CommandHandler("balance", self.get_balance))

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)

    def send_message(self, msg):
        asyncio.run(
            self.telegram_app.bot.send_message(chat_id=self.telegram_chat_id, text=msg)
        )

    def get_fill_icon(self, closed_pnl: float) -> str:
        return "🟢" if closed_pnl > 0 else "🔴"

    def process_fill(self, fill: Fill) -> None:

        price = float(fill["px"])
        coin = fill["coin"]
        size = fill["sz"]
        amount = price * float(size)
        closed_pnl = float(fill["closedPnl"])
        if fill["dir"] == 'Open Long':
            fill_message = f"🔵 Open Long: {size} {coin} ({amount:,.2f} USDC)"
        elif fill["dir"] == 'Open Short':
            fill_message = f"🔵 Open Short: {size} {coin} ({amount:,.2f} USDC)"
        elif fill["dir"] == 'Close Long':
            fill_message = f"{self.get_fill_icon(closed_pnl)} Close Long: {size} {coin} ({closed_pnl:,.2f} USDC)"
        elif fill["dir"] == 'Close Short':
            fill_message = f"{self.get_fill_icon(closed_pnl)} Close Short: {size} {coin} ({closed_pnl:,.2f} USDC)"
        else:
            fill_message = json.dumps(fill)

        self.send_message(fill_message)

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            fill_events: List[Fill] = user_events_data["fills"]
            for fill in fill_events:
                self.process_fill(fill)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [[KeyboardButton("/balance")]]
        reply_markup = ReplyKeyboardMarkup(
            keyboard, is_persistent=True
        )

        await update.message.reply_text(
            "Welcome! Click the button below to check your balance:",
            reply_markup=reply_markup,
        )

    async def get_balance(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:

        try:
            user_state = self.hyperliquid_info.user_state(self.user_address)
            message = (
                f"Account value: {float(user_state["crossMarginSummary"]["accountValue"]):,.2f} USDC"
            )
        except Exception as e:
            message = f"Failed to fetch balance: {str(e)}"

        await update.message.reply_text(message)


if __name__ == "__main__":
    bot = HyperliquidBot()
    os._exit(0)
