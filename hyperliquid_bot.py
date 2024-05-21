import asyncio
import logging
import json
import os

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import UserEventsMsg

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

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            fill_events = json.dumps(user_events_data["fills"])
            self.send_message(fill_events)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = [[KeyboardButton("/balance")]]
        reply_markup = ReplyKeyboardMarkup(
            keyboard, resize_keyboard=True, one_time_keyboard=True
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
