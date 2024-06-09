import requests

from logging_utils import logger

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext

from hyperliquid.utils.constants import MAINNET_API_URL

from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED

SELL_CHOOSING = range(1)

BUY_SELECTING_COIN, BUY_ENTERING_AMOUNT = range(2)


def get_coins_by_open_intereset():
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "type": "metaAndAssetCtxs"
    }

    response = requests.post(f"{MAINNET_API_URL}/info", headers=headers, json=data)
    response_data = response.json()
    universe = response_data[0]['universe']
    coin_data = response_data[1]
    coins = []
    for u, c in zip(universe, coin_data):
        name = u["name"]
        volume = float(c["dayNtlVlm"])
        coins.append((name, volume))

    sorted_coins = sorted(coins, key=lambda x: x[1], reverse=True)
    return [coin[0] for coin in sorted_coins[:15]]


async def buy(update: Update, context: CallbackContext) -> int:
    coins = get_coins_by_open_intereset()

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
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text("Loading...")

    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    context.user_data['selected_coin'] = coin
    keyboard = [
        [InlineKeyboardButton(f"{amount}%", callback_data=amount)]
        for amount in [25, 40, 50, 60, 75, 100]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(f"You selected {coin}. Please enter the amount to buy ({float(user_state['withdrawable']):,.2f} USDC available):", reply_markup=reply_markup)

    return BUY_ENTERING_AMOUNT


async def buy_enter_amount(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    amount = query.data
    if amount == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        amount = float(amount)
    except ValueError:
        await query.edit_message_text("Invalid amount.")
        return ConversationHandler.END

    selected_coin = context.user_data.get('selected_coin')
    if selected_coin is None:
        await query.edit_message_text("Error: No coin selected. Please restart the process.")
        return ConversationHandler.END

    await query.edit_message_text(text=f"Buying {selected_coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange is not None:
            # sz = 0
            # exchange.market_open(selected_coin, True, sz)
            await query.edit_message_text(text=f"NOT YET IMPLEMENTED: buy order for {selected_coin}")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to update orders: {str(e)}")

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
    await update.message.reply_text(OPERATION_CANCELLED, reply_markup=telegram_utils.reply_markup)
    return ConversationHandler.END
