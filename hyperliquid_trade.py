from logging_utils import logger

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext

from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils

SELL_CHOOSING = range(1)


async def sell(update: Update, context: CallbackContext) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = {asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])}

    keyboard = [
        [InlineKeyboardButton(coin, callback_data=coin)]
        for coin in coins
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Choose a position to sell:', reply_markup=reply_markup)
    return SELL_CHOOSING


async def sell_coin(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()

    coin = query.data
    await query.edit_message_text(text=f"Closing {coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange is not None:
            exchange.market_close(coin)
            await query.edit_message_text(text=f"Closed {coin}")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to update orders: {str(e)}")
    
    return ConversationHandler.END


async def trade_cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text('Operation cancelled.', reply_markup=telegram_utils.reply_markup)
    return ConversationHandler.END
