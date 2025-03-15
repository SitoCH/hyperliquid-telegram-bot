from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils

async def get_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    message_lines = [
        "<b>Statistics:</b>"
    ]
    await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)