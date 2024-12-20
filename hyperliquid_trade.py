from logging_utils import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ConversationHandler, CallbackContext, ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt, px_round, fmt_price

EXIT_CHOOSING, SELECTING_COIN, SELECTING_STOP_LOSS, SELECTING_TAKE_PROFIT, SELECTING_AMOUNT = range(5)


async def enter_long(update: Update, context: CallbackContext) -> int:
    context.user_data["enter_mode"] = "long"
    await update.message.reply_text('Choose a coin to long:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN


async def enter_short(update: Update, context: CallbackContext) -> int:
    context.user_data["enter_mode"] = "short"
    await update.message.reply_text('Choose a coin to short:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN


def ger_price_estimate(mid, decrease, percentage) -> str:
    return fmt_price(mid * (1.0 - percentage / 100.0) if decrease else mid * (1.0 + percentage / 100.0))


async def selected_amount(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    amount = query.data
    if amount == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        context.user_data["amount"] = float(amount)
    except ValueError:
        await query.edit_message_text("Invalid amount.")
        return ConversationHandler.END

    coin = context.user_data["selected_coin"]
    mid = float(hyperliquid_utils.info.all_mids()[coin])
    is_long = context.user_data["enter_mode"] == "long"

    keyboard = [
        [InlineKeyboardButton(f"{stop_loss}% (~{ger_price_estimate(mid, is_long, stop_loss)} USDC)", callback_data=str(stop_loss))]
        for stop_loss in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ]
    keyboard.append([InlineKeyboardButton("Maximum", callback_data='100.0')])
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Please enter the desired stop loss percentage (market price: {fmt_price(mid)} USDC):",
        reply_markup=reply_markup
    )

    return SELECTING_STOP_LOSS


async def selected_stop_loss(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    stop_loss = query.data
    if stop_loss == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    context.user_data["stop_loss"] = stop_loss

    coin = context.user_data["selected_coin"]
    mid = float(hyperliquid_utils.info.all_mids()[coin])
    is_long = context.user_data["enter_mode"] == "long"

    keyboard = [
        [InlineKeyboardButton(f"{take_profit}% (~{ger_price_estimate(mid, not is_long, take_profit)} USDC)", callback_data=str(take_profit))]
        for take_profit in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Please enter the desired take profit percentage (market price: {fmt_price(mid)} USDC):",
        reply_markup=reply_markup
    )

    return SELECTING_TAKE_PROFIT


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
        [InlineKeyboardButton(f"{amount}% (~{fmt(withdrawable * amount / 100.0)} USDC)", callback_data=str(amount))]
        for amount in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"You selected {coin}. Please enter the amount to {context.user_data['enter_mode']}:",
        reply_markup=reply_markup
    )

    return SELECTING_AMOUNT


async def selected_take_profit(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == 'cancel':
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    take_profit_percentage = float(query.data)

    amount = context.user_data.get('amount')
    if not amount:
        await query.edit_message_text("Error: No amount selected. Please restart the process.")
        return ConversationHandler.END

    stop_loss_percentage = float(context.user_data.get('stop_loss'))
    if not stop_loss_percentage:
        await query.edit_message_text("Error: No stop loss selected. Please restart the process.")
        return ConversationHandler.END


    selected_coin = context.user_data.get('selected_coin')
    if not selected_coin:
        await query.edit_message_text("Error: No coin selected. Please restart the process.")
        return ConversationHandler.END

    await query.edit_message_text(text=f"Opening {context.user_data['enter_mode']} for {selected_coin}...")
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

            await place_stop_loss_and_take_profit_orders(exchange, selected_coin, is_long, sz, mid, stop_loss_percentage, take_profit_percentage)

            await query.edit_message_text(text=f"Opened {context.user_data['enter_mode']} for {sz} units on {selected_coin} ({leverage}x)")
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to update orders: {str(e)}")

    return ConversationHandler.END


async def place_stop_loss_and_take_profit_orders(exchange, selected_coin, is_long, sz, mid, stop_loss_percentage: float, take_profit_percentage: float):
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    liquidation_px = float(hyperliquid_utils.get_liquidation_px_str(user_state, selected_coin))

    if liquidation_px > 0.0:
        liquidation_trigger_px = liquidation_px * (1.005 if is_long else 0.995)
        user_trigger_px = mid * (1.0 - stop_loss_percentage / 100.0) if is_long else mid * (1.0 + stop_loss_percentage / 100.0)
        sl_trigger_px = max(liquidation_trigger_px, user_trigger_px) if is_long else min(liquidation_trigger_px, user_trigger_px)
    else:
        sl_trigger_px = mid * 0.98 if is_long else mid * 1.02

    sl_limit_px = sl_trigger_px * 0.97 if is_long else sl_trigger_px * 1.03
    sl_order_type = {"trigger": {"triggerPx": px_round(sl_trigger_px), "isMarket": True, "tpsl": "sl"}}
    sl_order_result = exchange.order(selected_coin, not is_long, sz, px_round(sl_limit_px), sl_order_type, reduce_only=True)
    logger.info(sl_order_result)


    tp_trigger_px = mid * (1.0 + take_profit_percentage / 100.0) if is_long else mid * (1.0 - take_profit_percentage / 100.0)
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
        else:
            await update.message.reply_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await update.message.reply_text(text=f"Failed to exit all positions: {str(e)}")


async def exit_position(update: Update, context: CallbackContext) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    if not coins:
        await telegram_utils.reply(update, 'No positions to close')
        return ConversationHandler.END

    keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await telegram_utils.reply(update, 'Choose a coin to sell:')
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
            await query.delete_message()
        else:
            await query.edit_message_text(text="Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(text=f"Failed to exit {coin}: {str(e)}")

    return ConversationHandler.END
