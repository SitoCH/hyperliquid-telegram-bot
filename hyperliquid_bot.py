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

        # application.add_handler(CommandHandler("start", self.start))
        # application.add_handler(CommandHandler("orders", self.get_orders))
        self.telegram_app.add_handler(CommandHandler("balance", self.get_balance))

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)

    # def split_message(self, message, max_length=4096):
    #     """Splits a message into chunks of max_length."""
    #     return [message[i:i + max_length] for i in range(0, len(message), max_length)]

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

    # async def get_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    #     if len(context.args) != 1:
    #         await update.message.reply_text('Usage: /orders <your_eth_address>')
    #         return

    #     user_address = context.args[0]
    #     payload = {
    #         "type": "openOrders",
    #         "user": user_address
    #     }
    #     headers = {
    #         "Content-Type": "application/json"
    #     }

    #     try:
    #         response = requests.post(api_url, json=payload, headers=headers)
    #         if response.status_code == 200:
    #             orders = response.json()
    #             if orders:
    #                 message = "Your open orders:\n\n" + "\n".join([str(order) for order in orders])
    #             else:
    #                 message = "You have no open orders."
    #         else:
    #             message = f"Failed to fetch orders: {response.status_code}"
    #     except Exception as e:
    #         message = f"An error occurred: {str(e)}"

    #     for chunk in self.split_message(message):
    #         await update.message.reply_text(chunk)


if __name__ == '__main__':
    bot = HyperliquidBot()
