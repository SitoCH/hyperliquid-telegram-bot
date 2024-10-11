import os
import sys
import datetime
from typing import Any

from telegram import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from telegram import Update
from telegram.ext import Application, ContextTypes, ConversationHandler, CallbackContext
from telegram.ext._handlers.basehandler import BaseHandler
from telegram.ext._utils.types import CCT, JobCallback

from warnings import filterwarnings
from telegram.warnings import PTBUserWarning

from utils import OPERATION_CANCELLED, exchange_enabled, update_orders_enabled

from typing import Optional, Union


filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)


async def conversation_cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text(OPERATION_CANCELLED, reply_markup=telegram_utils.reply_markup)
    return ConversationHandler.END


class TelegramUtils:

    exit_all_command = "exit_all"
    overview_command = "overview"
    ta_command = "ta"

    reply_markup = ReplyKeyboardMarkup(
        [
            [KeyboardButton("/positions"), KeyboardButton(f"/{ta_command}"), KeyboardButton("/orders"), KeyboardButton(f"/{overview_command}")],
            [KeyboardButton("/long"), KeyboardButton("/short")] if exchange_enabled else [],
            [KeyboardButton(f"/{exit_all_command}"), KeyboardButton("/exit")] if exchange_enabled else [],
            [KeyboardButton("/update_orders")] if exchange_enabled and update_orders_enabled else []
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
        sys.exit()

    def add_handler(self, handler: BaseHandler[Any, CCT, Any], group: int = 0) -> None:
        self.telegram_app.add_handler(handler, group)

    def run_repeating(self, callback: JobCallback[CCT],
                      interval: Union[float, datetime.timedelta],
                      first: Optional[Union[float, datetime.timedelta, datetime.datetime, datetime.time]] = None,) -> None:
        self.telegram_app.job_queue.run_repeating(callback, interval=interval, first=first)

    def run_polling(self):
        self.telegram_app.job_queue.run_once(self.send_message, when=0, data="Hyperliquid Telegram bot up and running", chat_id=self.telegram_chat_id)

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


telegram_utils = TelegramUtils()
