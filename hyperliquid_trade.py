from typing import Dict, Any, List, Optional, Union, NamedTuple, Tuple
from logging_utils import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ConversationHandler, CallbackContext, ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from telegram.constants import ParseMode
from utils import OPERATION_CANCELLED, fmt, px_round, fmt_price
from technical_analysis.hyperliquid_candles import get_significant_levels
from technical_analysis.wyckoff_types import Timeframe
from tabulate import tabulate, simple_separated_format

EXIT_CHOOSING, SELECTING_COIN, SELECTING_STOP_LOSS, SELECTING_TAKE_PROFIT, SELECTING_AMOUNT, SELECTING_LEVERAGE = range(6)

async def enter_position(update: Update, context: CallbackContext, enter_mode: str) -> int:
    context.user_data.clear()
    context.user_data["enter_mode"] = enter_mode
    if context.args and len(context.args) > 2:
        context.user_data["selected_coin"] = context.args[0]
        context.user_data["stop_loss_price"] = float(context.args[1])
        context.user_data["take_profit_price"] = float(context.args[2])
        message, reply_markup = await get_amount_suggestions(context)
        await telegram_utils.reply(update, message, reply_markup=reply_markup)
        return SELECTING_AMOUNT

    await telegram_utils.reply(update, f'Choose a coin to {enter_mode}:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN

async def enter_long(update: Update, context: CallbackContext) -> int:
    return await enter_position(update, context, "long")


async def enter_short(update: Update, context: CallbackContext) -> int:
    return await enter_position(update, context, "short")


def ger_price_estimate(mid: float, decrease: bool, percentage: float) -> str:
    return fmt_price(mid * (1.0 - percentage / 100.0) if decrease else mid * (1.0 + percentage / 100.0))


async def selected_amount(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    query: CallbackQuery = update.callback_query # type: ignore
    if not query:
        return ConversationHandler.END
    await query.answer()

    amount = query.data
    if not amount or amount == 'cancel':
        await query.edit_message_text(OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        context.user_data["amount"] = float(amount)
    except ValueError:
        await query.edit_message_text("Invalid amount.")
        return ConversationHandler.END

    coin = context.user_data.get("selected_coin", "")
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    current_leverage = hyperliquid_utils.get_leverage(user_state, coin)
    if current_leverage is not None:
        context.user_data["leverage"] = int(current_leverage)
        await send_stop_loss_suggestions(query, context)
        return SELECTING_STOP_LOSS

    try:
        meta: List[Dict[str, Any]] = hyperliquid_utils.info.meta()
        asset_info_map = {
            info["name"]: int(info["maxLeverage"])
            for info in meta.get("universe", [])
        }
            
        max_leverage = asset_info_map.get(coin, 1)
        suggested_leverages = [l for l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30] if l <= max_leverage]

        keyboard = [
            [InlineKeyboardButton(f"{leverage}x", callback_data=str(leverage))]
            for leverage in suggested_leverages
        ]
        keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "Please select the leverage to use:",
            reply_markup=reply_markup
        )
        
        return SELECTING_LEVERAGE
        
    except Exception as e:
        logger.error(f"Error processing metadata for coin {coin}: {str(e)}")
        await query.edit_message_text("Error: Could not process coin metadata. Please try again.")
        return ConversationHandler.END


async def selected_leverage(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    query: CallbackQuery = update.callback_query # type: ignore
    await query.answer()

    leverage = query.data
    if leverage == 'cancel':
        await query.edit_message_text(OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        context.user_data["leverage"] = int(leverage)
    except ValueError:
        await query.edit_message_text("Invalid leverage.")
        return ConversationHandler.END

    if 'stop_loss_price' in context.user_data and 'take_profit_price' in context.user_data:
        await query.delete_message()
        return await open_order(context)

    await send_stop_loss_suggestions(query, context)
    return SELECTING_STOP_LOSS


class PriceSuggestion(NamedTuple):
    type: str
    price: float
    percentage: float

def get_price_suggestions(coin: str, mid: float, is_stop_loss: bool, is_long: bool) -> List[PriceSuggestion]:
    """Get price suggestions for either entry or exit points."""
    resistance_levels, support_levels = get_significant_levels(coin, mid, Timeframe.HOUR_1, 250)
    suggestions: List[PriceSuggestion] = []

    # For take profit (exit), we reverse the direction compared to stop loss (entry)
    if not is_stop_loss:
        is_long = not is_long

    # Add percentage-based suggestions
    for pct in [1.0, 2.0, 3.0, 4.0, 5.0]:
        price = float(mid * (1.0 - pct / 100.0) if is_long else mid * (1.0 + pct / 100.0))
        suggestions.append(PriceSuggestion("Fixed", price, pct))

    # Add level-based suggestions
    if is_long and support_levels:
        valid_supports = [level for level in support_levels if level < mid]
        for level in sorted(valid_supports, reverse=True)[:3]:
            pct = abs((level - mid) / mid * 100)
            suggestions.append(PriceSuggestion("Support", level, pct))
    elif not is_long and resistance_levels:
        valid_resistances = [level for level in resistance_levels if level > mid]
        for level in sorted(valid_resistances)[:3]:
            pct = abs((level - mid) / mid * 100)
            suggestions.append(PriceSuggestion("Resistance", level, pct))

    # Sort suggestions by price (ascending for shorts, descending for longs)
    suggestions.sort(key=lambda x: x.price, reverse=is_long)
    return suggestions


async def get_price_suggestions_text(context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE], is_stop_loss: bool) -> str:
    """Send formatted price suggestions for either stop loss or take profit."""
    coin = context.user_data["selected_coin"]
    mid = float(hyperliquid_utils.info.all_mids()[coin])
    is_long = context.user_data["enter_mode"] == "long"

    suggestions = get_price_suggestions(coin, mid, is_stop_loss, is_long)
    
    table_data = [
        [sugg.type, fmt_price(sugg.price), f"{fmt(sugg.percentage)}%"]
        for sugg in suggestions
    ]
    tablefmt = simple_separated_format('  ')
    table = tabulate(
        table_data,
        headers=["Type", "Price", "Distance"],
        tablefmt=tablefmt,
        colalign=("left", "right", "right")
    )

    return (
        f"Current market price: {fmt_price(mid)} USDC\n"
        f"Suggested {'stop losses' if is_stop_loss else 'take profits'}:\n"
        f"<pre>{table}</pre>\n"
        f"\nEnter your desired {'stop loss' if is_stop_loss else 'take profit'} price in USDC, or 'cancel' to stop:"
    )


async def send_stop_loss_suggestions(query: Any, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> None:
    await query.edit_message_text(f"Loading price suggestions for {context.user_data['selected_coin']}...")
    await query.edit_message_text(await get_price_suggestions_text(context, True), parse_mode=ParseMode.HTML)


async def selected_stop_loss(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    if not update.message:
        return ConversationHandler.END

    stop_loss = update.message.text
    if stop_loss.lower() == 'cancel':
        await telegram_utils.reply(update, OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        stop_loss_price = float(stop_loss)
        if stop_loss_price < 0:
            await telegram_utils.reply(update, "Price must be zero or greater.")
            return SELECTING_STOP_LOSS

        # Only validate stop loss price if it's not zero
        if stop_loss_price > 0:
            coin = context.user_data["selected_coin"]
            mid = float(hyperliquid_utils.info.all_mids()[coin])
            is_long = context.user_data["enter_mode"] == "long"
            
            if is_long and stop_loss_price >= mid:
                await telegram_utils.reply(update, "Stop loss price must be below current market price for long positions.")
                return SELECTING_STOP_LOSS
            elif not is_long and stop_loss_price <= mid:
                await telegram_utils.reply(update, "Stop loss price must be above current market price for short positions.")
                return SELECTING_STOP_LOSS

        context.user_data["stop_loss_price"] = stop_loss_price
    except ValueError:
        await telegram_utils.reply(update, "Invalid price. Please enter a number or 'cancel'.")
        return SELECTING_STOP_LOSS

    await telegram_utils.reply(update, f"Loading price suggestions for {coin}...")
    await telegram_utils.reply(update, await get_price_suggestions_text(context, False), parse_mode=ParseMode.HTML)

    return SELECTING_TAKE_PROFIT


async def selected_coin(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    query: CallbackQuery = update.callback_query # type: ignore
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text(OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text("Loading...")
    context.user_data["selected_coin"] = coin

    message, reply_markup = await get_amount_suggestions(context)
    await query.edit_message_text(message, reply_markup=reply_markup)

    return SELECTING_AMOUNT


async def get_amount_suggestions(context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> Tuple[str, InlineKeyboardMarkup]:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    withdrawable = float(user_state['withdrawable'])

    keyboard = [
        [InlineKeyboardButton(f"{amount}% (~{fmt(withdrawable * amount / 100.0)} USDC)", callback_data=str(amount))]
        for amount in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    return f"You selected {context.user_data['selected_coin']}. Please enter the amount to {context.user_data['enter_mode']}:", reply_markup


async def selected_take_profit(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    if not update.message:
        return ConversationHandler.END

    take_profit = update.message.text
    if take_profit.lower() == 'cancel':
        await telegram_utils.reply(update, OPERATION_CANCELLED)
        return ConversationHandler.END

    try:
        take_profit_price = float(take_profit)
        if take_profit_price <= 0:
            await telegram_utils.reply(update, "Price must be greater than 0.")
            return SELECTING_TAKE_PROFIT

        # Validate take profit price based on position direction
        coin = context.user_data["selected_coin"]
        mid = float(hyperliquid_utils.info.all_mids()[coin])
        is_long = context.user_data["enter_mode"] == "long"
        
        if is_long and take_profit_price <= mid:
            await telegram_utils.reply(update, "Take profit price must be above current market price for long positions.")
            return SELECTING_TAKE_PROFIT
        elif not is_long and take_profit_price >= mid:
            await telegram_utils.reply(update, "Take profit price must be below current market price for short positions.")
            return SELECTING_TAKE_PROFIT

        context.user_data["take_profit_price"] = take_profit_price

    except ValueError:
        await telegram_utils.reply(update, "Invalid price. Please enter a number or 'cancel'.")
        return SELECTING_TAKE_PROFIT

    return await open_order(context)

async def open_order(context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    amount = context.user_data.get('amount')
    if not amount:
        await telegram_utils.send("Error: No amount selected. Please restart the process.")
        return ConversationHandler.END

    stop_loss_price = context.user_data.get('stop_loss_price')
    if not stop_loss_price:
        await telegram_utils.send("Error: No stop loss selected. Please restart the process.")
        return ConversationHandler.END

    take_profit_price = context.user_data.get('take_profit_price')
    if not take_profit_price:
        await telegram_utils.send("Error: No take profit selected. Please restart the process.")
        return ConversationHandler.END

    selected_coin = context.user_data.get('selected_coin')
    if not selected_coin:
        await telegram_utils.send("Error: No coin selected. Please restart the process.")
        return ConversationHandler.END

    await telegram_utils.send(f"Opening {context.user_data['enter_mode']} for {selected_coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            available_balance = float(user_state['withdrawable'])
            balance_to_use = available_balance * amount / 100.0
            leverage = context.user_data.get('leverage', 1)
            exchange.update_leverage(leverage, selected_coin, False)
            mid = float(hyperliquid_utils.info.all_mids()[selected_coin])
            sz_decimals = hyperliquid_utils.get_sz_decimals()
            sz = round(balance_to_use * leverage / mid, sz_decimals[selected_coin])
            if sz * mid < 10.0:
                await telegram_utils.send("The order value is less than 10 USDC and can't be executed")
                return ConversationHandler.END

            is_long = context.user_data["enter_mode"] == "long"
            open_result = exchange.market_open(selected_coin, is_long, sz)
            logger.info(open_result)

            await place_stop_loss_and_take_profit_orders(exchange, selected_coin, is_long, sz, stop_loss_price, take_profit_price)

            await telegram_utils.send(f"Opened {context.user_data['enter_mode']} for {sz} units on {selected_coin} ({leverage}x)")
        else:
            await telegram_utils.send("Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.send(f"Failed to update orders: {str(e)}")

    return ConversationHandler.END


def place_stop_loss_order(exchange: Any, 
    selected_coin: str, 
    is_long: bool, 
    sz: float,
    stop_loss_price: float) -> None:

    sl_trigger_px = stop_loss_price

    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)

    liquidation_px_str = hyperliquid_utils.get_liquidation_px_str(user_state, selected_coin)

    if liquidation_px_str is not None:
        liquidation_px = float(liquidation_px_str)
        if liquidation_px > 0.0:
            liquidation_trigger_px = liquidation_px * (1.0025 if is_long else 0.9975)
            
            if stop_loss_price == 0 or (is_long and stop_loss_price < liquidation_trigger_px) or (not is_long and stop_loss_price > liquidation_trigger_px):
                sl_trigger_px = liquidation_trigger_px
    
    if sl_trigger_px > 0:
        sl_limit_px = sl_trigger_px * 0.97 if is_long else sl_trigger_px * 1.03
        sl_order_type = {"trigger": {"triggerPx": px_round(sl_trigger_px), "isMarket": True, "tpsl": "sl"}}
        sl_order_result = exchange.order(selected_coin, not is_long, sz, px_round(sl_limit_px), sl_order_type, reduce_only=True)
        logger.info(sl_order_result)

async def place_stop_loss_and_take_profit_orders(
    exchange: Any, 
    selected_coin: str, 
    is_long: bool, 
    sz: float,
    stop_loss_price: float, 
    take_profit_price: float
) -> None:

    place_stop_loss_order(exchange, selected_coin, is_long, sz, stop_loss_price)

    if take_profit_price > 0:
        tp_trigger_px = take_profit_price
        tp_limit_px = tp_trigger_px * 1.02 if is_long else tp_trigger_px * 0.98
        tp_order_type = {"trigger": {"triggerPx": px_round(tp_trigger_px), "isMarket": True, "tpsl": "tp"}}
        tp_order_result = exchange.order(selected_coin, not is_long, sz, px_round(tp_limit_px), tp_order_type, reduce_only=True)
        logger.info(tp_order_result)


async def exit_all_positions(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> None:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            for asset_position in user_state.get("assetPositions", []):
                coin = asset_position['position']['coin']
                exchange.market_close(coin)
        else:
            await telegram_utils.reply(update, "Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.reply(update, f"Failed to exit all positions: {str(e)}")


async def exit_position(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    if not coins:
        await telegram_utils.reply(update, 'No positions to close')
        return ConversationHandler.END

    keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
    keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await telegram_utils.reply(update, 'Choose a coin to sell:', reply_markup=reply_markup)
    return EXIT_CHOOSING


async def exit_selected_coin(update: Update, context: Union[CallbackContext, ContextTypes.DEFAULT_TYPE]) -> int:
    query: CallbackQuery = update.callback_query # type: ignore
    await query.answer()

    coin = query.data
    if coin == 'cancel':
        await query.edit_message_text('Operation cancelled')
        return ConversationHandler.END

    await query.edit_message_text(f"Closing {coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            exchange.market_close(coin)
            await query.delete_message()
        else:
            await query.edit_message_text("Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(f"Failed to exit {coin}: {str(e)}")

    return ConversationHandler.END
