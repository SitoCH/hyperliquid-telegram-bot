import logging
import json
import os

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import UserEventsMsg

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperliquidBot():

    def __init__(self):
        telegram_token = os.environ["HYPERLIQUID_TELEGRAM_BOT_TOKEN"]
        self.telegram_chat_id = os.environ["HYPERLIQUID_TELEGRAM_BOT_CHAT_ID"]

        self.user_address = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER"]

        self.hyperliquid_info = Info(constants.MAINNET_API_URL)
        self.hyperliquid_info.subscribe({"type": "userEvents", "user": self.user_address}, self.on_user_events)

        self.telegram_app = Application.builder().token(telegram_token).build()

        self.telegram_app.add_handler(CommandHandler("balance", self.get_balance))

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            self.telegram_app.bot.sendMessage(chat_id=self.telegram_chat_id, text=json.dumps(user_events_data["fills"]))

    async def get_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        try:
            user_state = self.hyperliquid_info.user_state(self.user_address)
            message = f"Account value: {user_state.perpetuals_data["crossMarginSummary"]["accountValue"]}"
        except Exception as e:
            message = f"Failed to fetch balance: {str(e)}"

        await update.message.reply_text(message)

if __name__ == '__main__':
    bot = HyperliquidBot()
