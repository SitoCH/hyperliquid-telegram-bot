import os
from typing import Any

from telegram import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from telegram import Update
from telegram.ext import Application, ContextTypes
from telegram.ext._handlers.basehandler import BaseHandler
from telegram.ext._utils.types import CCT

from warnings import filterwarnings
from telegram.warnings import PTBUserWarning

filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)


class TelegramUtils:

    enable_exchange_buttons = True if os.environ.get("HYPERLIQUID_TELEGRAM_BOT_USER_VAULT") is not None else False

    reply_markup = ReplyKeyboardMarkup(
        [
            [KeyboardButton("/positions"), KeyboardButton("/orders")],
            [KeyboardButton("/update_orders")] if enable_exchange_buttons else [],
            [KeyboardButton("/sell")] if enable_exchange_buttons else []
        ], resize_keyboard=True
    )

    def __init__(self):
        telegram_token = os.environ["HYPERLIQUID_TELEGRAM_BOT_TOKEN"]
        self.telegram_chat_id = os.environ["HYPERLIQUID_TELEGRAM_BOT_CHAT_ID"]

        self.telegram_app = Application.builder().token(telegram_token).build()

    def send(self, message):
        self.telegram_app.job_queue.run_once(self.send_message, when=0, data=message, chat_id=self.telegram_chat_id)

    def send_and_exit(self, message):
        self.telegram_app.job_queue.run_once(self.send_message_and_exit, when=0, data=message, chat_id=self.telegram_chat_id)

    async def send_message(self, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(
            chat_id=context.job.chat_id, text=context.job.data, reply_markup=self.reply_markup
        )

    async def send_message_and_exit(self, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message(context)
        os._exit(0)

    def add_handler(self, handler: BaseHandler[Any, CCT], group: int = 0) -> None:
        self.telegram_app.add_handler(handler, group)

    def run_polling(self):
        self.telegram_app.job_queue.run_once(self.send_message, when=0, data="Hyperliquid Telegram bot up and running", chat_id=self.telegram_chat_id)

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


telegram_utils = TelegramUtils()
