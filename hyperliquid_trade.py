from logging_utils import logger

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext

from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils

SELL_CHOOSING = range(1)

BUY_SELECTING_COIN, BUY_ENTERING_AMOUNT = range(2)


async def buy(update: Update, context: CallbackContext) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    keyboard = [
        [InlineKeyboardButton(coin, callback_data=coin)]
        for coin in coins
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Choose a coin to buy:', reply_markup=reply_markup)
    return BUY_SELECTING_COIN


async def buy_select_coin(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text(text='Operation cancelled')
        return ConversationHandler.END

    context.user_data['selected_coin'] = coin
    await query.edit_message_text(text=f"You selected {coin}. Please enter the amount to buy:")
    return BUY_ENTERING_AMOUNT


async def buy_enter_amount(update: Update, context: CallbackContext) -> int:
    amount = update.message.text
    try:
        amount = float(amount)
    except ValueError:
        await update.message.reply_text("Invalid amount. Please enter a numeric value.")
        return BUY_ENTERING_AMOUNT

    selected_coin = context.user_data.get('selected_coin')
    if selected_coin is None:
        await update.message.reply_text("Error: No coin selected. Please restart the process.")
        return ConversationHandler.END

    await update.message.reply_text("Implementation not complete")

    return ConversationHandler.END


async def sell(update: Update, context: CallbackContext) -> range:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    keyboard = [
        [InlineKeyboardButton(coin, callback_data=coin)]
        for coin in coins
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Choose a coin to sell:', reply_markup=reply_markup)
    return SELL_CHOOSING


async def sell_select_coin(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text(text='Operation cancelled')
        return ConversationHandler.END

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
    await update.message.reply_text('Operation cancelled', reply_markup=telegram_utils.reply_markup)
    return ConversationHandler.END
