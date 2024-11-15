import os
import sys
import datetime
from typing import Any, cast, Tuple, List, Sequence

from telegram import (
    KeyboardButton,
    ReplyKeyboardMarkup,
    Message,
)

from telegram._utils.types import ODVInput
from telegram.ext._utils.types import CCT, JobCallback

from telegram import Update
from telegram.ext import Application, ContextTypes, ConversationHandler, CallbackContext
from telegram.ext._handlers.basehandler import BaseHandler
from telegram.constants import ParseMode

from warnings import filterwarnings
from telegram.warnings import PTBUserWarning

from utils import OPERATION_CANCELLED, exchange_enabled

from typing import Optional, Union


filterwarnings(
    action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning
)


async def conversation_cancel(update: Update, context: CallbackContext) -> int:
    if update.message:
        await telegram_utils.reply(update, OPERATION_CANCELLED)
    return ConversationHandler.END


class TelegramUtils:

    exit_all_command = "exit_all"
    overview_command = "overview"
    ta_command = "ta"

    reply_markup = ReplyKeyboardMarkup(
        [
            [
                KeyboardButton("/positions"),
                KeyboardButton(f"/{ta_command}"),
                KeyboardButton("/orders"),
                KeyboardButton(f"/{overview_command}"),
            ],
            (
                [KeyboardButton("/long"), KeyboardButton("/short")]
                if exchange_enabled
                else []
            ),
            (
                [KeyboardButton(f"/{exit_all_command}"), KeyboardButton("/exit")]
                if exchange_enabled
                else []
            ),
        ],
        resize_keyboard=True,
    )

    def __init__(self):
        telegram_token = os.environ["HTB_TOKEN"]
        self.telegram_chat_id = os.environ["HTB_CHAT_ID"]

        self.telegram_app = Application.builder().token(telegram_token).build()

    def add_buttons(self, buttons: List[str], row_index: int = 0) -> None:
        keyboard: List[List[KeyboardButton]] = [list(row) for row in self.reply_markup.keyboard]
        
        while len(keyboard) <= row_index:
            keyboard.append([])

        row = keyboard[row_index]
        for button_text in buttons:
            row.append(KeyboardButton(button_text))

        keyboard_layout: List[Sequence[KeyboardButton]] = [tuple(row) for row in keyboard]

        self.reply_markup = ReplyKeyboardMarkup(
            keyboard_layout,
            resize_keyboard=True
        )

    async def reply(
        self, update: Update, message: str, parse_mode: ODVInput[str] = None
    ) -> None:
        if update.message:
            await update.message.reply_text(
                message, parse_mode=parse_mode, reply_markup=self.reply_markup
            )

    async def send(self, context: ContextTypes.DEFAULT_TYPE, message: str, parse_mode: ODVInput[str] = None):
        await context.bot.send_message(text=message, parse_mode=parse_mode, chat_id=self.telegram_chat_id)

    def queue_send(self, message: str):
        self.telegram_app.job_queue.run_once(
            self.send_message, when=0, data=message, chat_id=self.telegram_chat_id
        )

    def send_and_exit(self, message: str):
        self.telegram_app.job_queue.run_once(
            self.send_message_and_exit,
            when=0,
            data=message,
            chat_id=self.telegram_chat_id,
        )

    async def send_message(self, context: ContextTypes.DEFAULT_TYPE):
        if context.job and hasattr(context.job, 'chat_id') and hasattr(context.job, 'data'):
            chat_id = str(context.job.chat_id)
            message = str(context.job.data)
            await context.bot.send_message(
                chat_id=chat_id,
                text=message,
                reply_markup=self.reply_markup,
            )

    async def send_message_and_exit(self, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message(context)
        sys.exit()

    def add_handler(self, handler: BaseHandler[Any, CCT, Any], group: int = 0) -> None:
        self.telegram_app.add_handler(handler, group)

    def run_once(self, callback: JobCallback[CCT]):
        self.telegram_app.job_queue.run_once(
            callback, when=0, chat_id=self.telegram_chat_id
        )

    def run_repeating(
        self,
        callback: JobCallback[CCT],
        interval: Union[float, datetime.timedelta],
        first: Optional[
            Union[float, datetime.timedelta, datetime.datetime, datetime.time]
        ] = None,
    ) -> None:
        self.telegram_app.job_queue.run_repeating(
            callback, interval=interval, first=first
        )

    def run_polling(self):
        self.telegram_app.job_queue.run_once(
            self.send_message,
            when=0,
            data="Hyperliquid Telegram bot up and running",
            chat_id=self.telegram_chat_id,
        )

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


telegram_utils = TelegramUtils()
