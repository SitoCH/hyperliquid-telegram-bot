import requests
from logging_utils import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext, ContextTypes
from hyperliquid.utils.constants import MAINNET_API_URL
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, px_round

EXIT_CHOOSING, SELECTING_COIN, SELECTING_AMOUNT = range(3)


def get_coins_by_open_interest():
    headers = {"Content-Type": "application/json"}
    data = {"type": "metaAndAssetCtxs"}
    response = requests.post(f"{MAINNET_API_URL}/info", headers=headers, json=data)
    response_data = response.json()
    universe, coin_data = response_data[0]['universe'], response_data[1]

    coins = [(u["name"], float(c["dayNtlVlm"])) for u, c in zip(universe, coin_data)]
    sorted_coins = sorted(coins, key=lambda x: x[1], reverse=True)
    return [coin[0] for coin in reversed(sorted_coins[:75])]


def get_enter_reply_markup():
    coins = get_coins_by_open_interest()
    keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
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
    withdrawable = float(user_state['withdrawable'])
    context.user_data["selected_coin"] = coin

    keyboard = [
        [InlineKeyboardButton(f"{amount}% (~{withdrawable * amount / 100.0:,.2f} USDC)", callback_data=str(amount))]
        for amount in [10, 25, 40, 50, 60, 75, 90, 100]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"You selected {coin}. Please enter the amount to {context.user_data['enter_mode']} ({withdrawable:,.2f} USDC available):", 
        reply_markup=reply_markup
    )

    return SELECTING_AMOUNT


def get_liquidation_px(user_state, selected_coin) -> float:
    for asset_position in user_state.get("assetPositions", []):
        if asset_position['position']['coin'] == selected_coin:
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
    if not selected_coin:
        await query.edit_message_text("Error: No coin selected. Please restart the process.")
        return ConversationHandler.END

    await query.edit_message_text(text=f"Executing orders for {selected_coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            available_balance = float(user_state['withdrawable'])
            balance_to_use = available_balance * amount / 100.0
            leverage = hyperliquid_utils.get_leverage(user_state, selected_coin)
            exchange.update_leverage(leverage, selected_coin, False)
            mid = float(hyperliquid_utils.info.all_mids()[selected_coin])
            sz_decimals = hyperliquid_utils.get_sz_decimals()
            sz = round(balance_to_use * leverage / mid, sz_decimals[selected_coin])
            if sz * mid < 10.0:
                await query.edit_message_text(text="The order value is less than 10 USDC and can't be executed")
                return ConversationHandler.END

            is_long = context.user_data["enter_mode"] == "long"
            open_result = exchange.market_open(selected_coin, is_long, sz)
            logger.info(open_result)

            await place_stop_loss_and_take_profit_orders(exchange, selected_coin, is_long, sz, mid, user_state)

            await query.edit_message_text(text=f"Orders executed for {sz} units on {selected_coin} ({leverage}x)")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to update orders: {str(e)}")

    return ConversationHandler.END


async def place_stop_loss_and_take_profit_orders(exchange, selected_coin, is_long, sz, mid, user_state):
    liquidation_px = get_liquidation_px(user_state, selected_coin)

    if liquidation_px > 0:
        sl_trigger_px = liquidation_px * 1.004 if is_long else liquidation_px * 0.996
    else:
        sl_trigger_px = mid * 0.97 if is_long else mid * 1.03

    sl_limit_px = sl_trigger_px * 0.97 if is_long else sl_trigger_px * 1.03
    sl_order_type = {"trigger": {"triggerPx": px_round(sl_trigger_px), "isMarket": True, "tpsl": "sl"}}
    sl_order_result = exchange.order(selected_coin, not is_long, sz, px_round(sl_limit_px), sl_order_type, reduce_only=True)
    logger.info(sl_order_result)

    tp_trigger_px = mid * 1.0125 if is_long else mid * 0.9875
    tp_limit_px = tp_trigger_px * 1.02 if is_long else tp_trigger_px * 0.98
    tp_order_type = {"trigger": {"triggerPx": px_round(tp_trigger_px), "isMarket": True, "tpsl": "tp"}}
    tp_order_result = exchange.order(selected_coin, not is_long, sz, px_round(tp_limit_px), tp_order_type, reduce_only=True)
    logger.info(tp_order_result)


async def exit_all_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            for asset_position in user_state.get("assetPositions", []):
                coin = asset_position['position']['coin']
                exchange.market_close(coin)
                await update.message.reply_text(text=f"Closed {coin}")
        else:
            await update.message.reply_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await update.message.reply_text(text=f"Failed to exit all positions: {str(e)}")


async def exit_position(update: Update, context: CallbackContext) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    if not coins:
        await update.message.reply_text('No positions to close', reply_markup=telegram_utils.reply_markup)
        return ConversationHandler.END

    keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
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
        if exchange:
            exchange.market_close(coin)
            await query.edit_message_text(text=f"Closed {coin}")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to exit {coin}: {str(e)}")

    return ConversationHandler.END


async def trade_cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text(OPERATION_CANCELLED, reply_markup=telegram_utils.reply_markup)
    return ConversationHandler.END
