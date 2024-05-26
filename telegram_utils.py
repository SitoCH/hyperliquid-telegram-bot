import os

from telegram import (
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from telegram.ext import ContextTypes


reply_markup = ReplyKeyboardMarkup(
    [[KeyboardButton("/positions")]], resize_keyboard=True
)


async def send_message(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=context.job.chat_id, text=context.job.data, reply_markup=reply_markup
    )


async def send_message_and_exit(context: ContextTypes.DEFAULT_TYPE):
    await send_message(context)
    os._exit(0)
