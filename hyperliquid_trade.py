import requests

from logging_utils import logger

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext

from hyperliquid.utils.constants import MAINNET_API_URL

from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED

EXIT_CHOOSING = range(1)

SELECTING_COIN, SELECTING_AMOUNT = range(2)


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
    return [coin[0] for coin in sorted_coins[:25]]


def get_enter_reply_markup():
    coins = get_coins_by_open_intereset()

    keyboard = [
        [InlineKeyboardButton(coin, callback_data=coin)]
        for coin in coins
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    return InlineKeyboardMarkup(keyboard)


async def enter_long(update: Update, context: CallbackContext) -> int:
    context.user_data["enter_mode"] = "long"
    await update.message.reply_text('Choose a coin to long:', reply_markup=get_enter_reply_markup())
    return SELECTING_COIN


async def enter_short(update: Update, context: CallbackContext) -> int:
    context.user_data["enter_mode"] = "short"
    await update.message.reply_text('Choose a coin to short:', reply_markup=get_enter_reply_markup())
    return SELECTING_COIN


async def selected_coin(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text("Loading...")

    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    context.user_data["selected_coin"] = coin
    keyboard = [
        [InlineKeyboardButton(f"{amount}%", callback_data=amount)]
        for amount in [25, 40, 50, 60, 75, 100]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(f"You selected {coin}. Please enter the amount to {context.user_data['enter_mode']} ({float(user_state['withdrawable']):,.2f} USDC available):", reply_markup=reply_markup)

    return SELECTING_AMOUNT


def get_leverage(user_state, selected_coin) -> int:
    if len(user_state["assetPositions"]) > 0:
        for asset_position in user_state["assetPositions"]:
            coin = asset_position['position']['coin']
            if coin == selected_coin:
                return int(asset_position['position']['leverage']['value'])

    meta = hyperliquid_utils.info.meta()
    for asset_info in meta["universe"]:
        if asset_info["name"] == selected_coin:
            leverage = int(asset_info["maxLeverage"])
            return min(leverage, 40)
    return 5


def get_liquidation_px(user_state, selected_coin) -> float:
    if len(user_state["assetPositions"]) > 0:
        for asset_position in user_state["assetPositions"]:
            coin = asset_position['position']['coin']
            if coin == selected_coin:
                return float(asset_position['position']['liquidationPx'])

    return 0


async def selected_amount(update: Update, context: CallbackContext) -> int:
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

    await query.edit_message_text(text=f"Creating order for {selected_coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange is not None:
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            available_balance = float(user_state['withdrawable'])
            balance_to_use = available_balance * amount / 100.0
            leverage = get_leverage(user_state, selected_coin)
            exchange.update_leverage(leverage, selected_coin, False)
            mid = float(hyperliquid_utils.info.all_mids()[selected_coin])
            sz_decimals = hyperliquid_utils.get_sz_decimals()
            sz = round(balance_to_use * leverage / mid, sz_decimals[selected_coin])
            if sz * mid < 10.0:
                await query.edit_message_text(text="The order value is less than 10$ and can't be executed")
            else:
                is_long = context.user_data["enter_mode"] == "long"
                open_result = exchange.market_open(selected_coin, is_long, sz)
                logger.info(open_result)
                # set stoploss order
                user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
                liquidation_px = get_liquidation_px(user_state, selected_coin)

                if liquidation_px > 0:
                    trigger_px = liquidation_px * 1.01 if is_long else liquidation_px * 0.99
                else:
                    trigger_px = mid * 0.97 if is_long else mid * 1.03

                limit_px = trigger_px * 0.97 if is_long else trigger_px * 1.03
                stop_order_type = {"trigger": {"triggerPx": round(float(f"{(trigger_px):.5g}"), 6), "isMarket": True, "tpsl": "sl"}}
                stoploss_result = exchange.order(selected_coin, not is_long, sz, round(float(f"{(limit_px):.5g}"), 6), stop_order_type, reduce_only=True)
                logger.info(stoploss_result)

                await query.edit_message_text(text=f"Order exected for {sz} units on {selected_coin} ({leverage}x)")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to update orders: {str(e)}")

    return ConversationHandler.END


async def exit_position(update: Update, context: CallbackContext) -> range:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    if len(coins) == 0:
        await update.message.reply_text('No positions to close', reply_markup=telegram_utils.reply_markup)
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton(coin, callback_data=coin)]
        for coin in coins
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('Choose a coin to sell:', reply_markup=reply_markup)
    return EXIT_CHOOSING


async def exit_selected_coin(update: Update, context: CallbackContext) -> int:
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
