import os

from typing import Dict, Any, List, Optional, Union, NamedTuple, Tuple
from logging_utils import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ConversationHandler, CallbackContext, ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from telegram.constants import ParseMode
from utils import OPERATION_CANCELLED, fmt, px_round, fmt_price
from technical_analysis.wyckoff.significant_levels import get_significant_levels_from_timeframe
from technical_analysis.wyckoff.wyckoff_types import Timeframe
from tabulate import tabulate, simple_separated_format

EXIT_CHOOSING, SELECTING_COIN, SELECTING_STOP_LOSS, SELECTING_TAKE_PROFIT, SELECTING_AMOUNT, SELECTING_LEVERAGE = range(6)

# Type alias for context types used throughout the module
ContextType = Union[CallbackContext, ContextTypes.DEFAULT_TYPE]


def _has_sl_tp_set(context: ContextType) -> bool:
    """Check if both stop loss and take profit prices are set in context."""
    return 'stop_loss_price' in context.user_data and 'take_profit_price' in context.user_data  # type: ignore


def _get_mid_price(coin: str) -> float:
    """Get current mid price for a coin."""
    return float(hyperliquid_utils.info.all_mids()[coin])


def _is_long_position(context: ContextType) -> bool:
    """Check if current position is long."""
    return context.user_data["enter_mode"] == "long"  # type: ignore


def _create_keyboard_with_cancel(buttons: List[List[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    """Create an InlineKeyboardMarkup with a cancel button appended."""
    buttons.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
    return InlineKeyboardMarkup(buttons)


async def _handle_callback_cancel(query: CallbackQuery) -> int:
    """Handle cancel action for callback queries."""
    await query.edit_message_text(OPERATION_CANCELLED)
    return ConversationHandler.END


async def _handle_callback_selection(
    update: Update,
    converter: Any,
    invalid_msg: str,
    next_action: Any
) -> int:
    """Generic handler for callback query selections (amount, leverage, coin)."""
    query: CallbackQuery = update.callback_query  # type: ignore
    if not query:
        return ConversationHandler.END
    await query.answer()

    value = query.data
    if not value or value == 'cancel':
        return await _handle_callback_cancel(query)

    try:
        return await next_action(query, converter(value) if converter else value)
    except (ValueError, TypeError):
        await query.edit_message_text(invalid_msg)
        return ConversationHandler.END

async def enter_position(update: Update, context: CallbackContext, enter_mode: str) -> int:
    context.user_data.clear() # type: ignore
    context.user_data["enter_mode"] = enter_mode # type: ignore
    if skip_sl_tp_prompt():
        context.user_data["stop_loss_price"] = "skip" # type: ignore
        context.user_data["take_profit_price"] = "skip" # type: ignore

    if context.args and len(context.args) > 2:
        context.user_data["selected_coin"] = context.args[0] # type: ignore
        context.user_data["stop_loss_price"] = float(context.args[1].replace(",", "")) # type: ignore
        context.user_data["take_profit_price"] = float(context.args[2].replace(",", "")) # type: ignore
        message, reply_markup = await get_amount_suggestions(context)
        await telegram_utils.reply(update, message, reply_markup=reply_markup)
        return SELECTING_AMOUNT

    await telegram_utils.reply(update, f'Choose a coin to {enter_mode}:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN

async def enter_long(update: Update, context: CallbackContext) -> int:
    return await enter_position(update, context, "long")


async def enter_short(update: Update, context: CallbackContext) -> int:
    return await enter_position(update, context, "short")


def skip_sl_tp_prompt() -> bool:
    return os.getenv('HTB_SKIP_SL_TP_PROMPT', 'False') == 'True'

def has_order_error(result: Any) -> bool:
    """Check if order result contains an error."""
    if not result or not isinstance(result, dict):
        return True
    if result.get('status') != 'ok':
        return True
    response = result.get('response', {})
    if response.get('type') == 'order':
        statuses = response.get('data', {}).get('statuses', [])
        return any('error' in status for status in statuses)
    return False

def get_order_error_message(result: Any) -> str:
    """Extract error message from order result."""
    if not result or not isinstance(result, dict):
        return "Unknown error"
    if result.get('status') != 'ok':
        return str(result)
    response = result.get('response', {})
    if response.get('type') == 'order':
        statuses = response.get('data', {}).get('statuses', [])
        for status in statuses:
            if 'error' in status:
                return status['error']
    return "Unknown order error"

async def _process_amount_selection(query: CallbackQuery, context: ContextType, amount: float) -> int:
    """Process the amount selection and determine next step."""
    context.user_data["amount"] = amount  # type: ignore
    coin = context.user_data.get("selected_coin", "")  # type: ignore
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    current_leverage = hyperliquid_utils.get_leverage(user_state, coin)
    
    if current_leverage is not None:
        context.user_data["leverage"] = int(current_leverage)  # type: ignore
        return await _proceed_after_leverage_set(query, context, user_state, coin)

    return await _prompt_for_leverage(query, coin)


async def selected_amount(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, amount: float) -> int:
        return await _process_amount_selection(query, context, amount)
    return await _handle_callback_selection(update, float, "Invalid amount.", process)


async def _proceed_after_leverage_set(query: CallbackQuery, context: ContextType, user_state: Dict[str, Any], coin: str) -> int:
    """Handle the flow after leverage is determined - either open order or prompt for SL."""
    if _has_sl_tp_set(context):
        await query.delete_message()
        return await open_order(context, user_state, _get_mid_price(coin))
    await send_stop_loss_suggestions(query, context)
    return SELECTING_STOP_LOSS


async def _prompt_for_leverage(query: CallbackQuery, coin: str) -> int:
    """Prompt user to select leverage for the given coin."""
    try:
        meta: List[Dict[str, Any]] = hyperliquid_utils.info.meta()
        asset_info_map = {
            info["name"]: int(info["maxLeverage"])
            for info in meta.get("universe", [])  # type: ignore
        }
        max_leverage = asset_info_map.get(coin, 1)
        suggested_leverages = [l for l in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30] if l <= max_leverage]

        keyboard = [
            [InlineKeyboardButton(f"{leverage}x", callback_data=str(leverage))]
            for leverage in suggested_leverages
        ]
        reply_markup = _create_keyboard_with_cancel(keyboard)
        await query.edit_message_text("Please select the leverage to use:", reply_markup=reply_markup)
        return SELECTING_LEVERAGE
    except Exception as e:
        logger.error(f"Error processing metadata for coin {coin}: {str(e)}")
        await query.edit_message_text("Error: Could not process coin metadata. Please try again.")
        return ConversationHandler.END


async def selected_leverage(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, leverage: int) -> int:
        context.user_data["leverage"] = leverage  # type: ignore
        coin = context.user_data.get("selected_coin", "")  # type: ignore
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        return await _proceed_after_leverage_set(query, context, user_state, coin)
    return await _handle_callback_selection(update, int, "Invalid leverage.", process)


class PriceSuggestion(NamedTuple):
    type: str
    price: float
    percentage: float

async def get_price_suggestions(coin: str, mid: float, is_stop_loss: bool, is_long: bool) -> List[PriceSuggestion]:
    """Get price suggestions for either entry or exit points."""
    resistance_levels, support_levels = await get_significant_levels_from_timeframe(coin, mid, Timeframe.HOUR_1, 250)
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


async def get_price_suggestions_text(context: ContextType, is_stop_loss: bool) -> str:
    """Send formatted price suggestions for either stop loss or take profit."""
    coin = context.user_data["selected_coin"]  # type: ignore
    mid = _get_mid_price(coin)
    is_long = _is_long_position(context)

    suggestions = await get_price_suggestions(coin, mid, is_stop_loss, is_long)
    
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

    price_type = 'stop loss' if is_stop_loss else 'take profit'
    return (
        f"Current market price: {fmt_price(mid)} USDC\n"
        f"Suggested {'stop losses' if is_stop_loss else 'take profits'}:\n"
        f"<pre>{table}</pre>\n"
        f"\nEnter your desired {price_type} price in USDC, or 'cancel' to stop:"
    )


async def send_stop_loss_suggestions(query: Any, context: ContextType) -> None:
    await query.edit_message_text(f"Loading price suggestions for {context.user_data['selected_coin']}...")  # type: ignore
    await query.edit_message_text(await get_price_suggestions_text(context, True), parse_mode=ParseMode.HTML)


def _validate_stop_loss_price(price: float, mid: float, is_long: bool) -> Optional[str]:
    """Validate stop loss price. Returns error message if invalid, None if valid."""
    if price < 0:
        return "Price must be zero or greater."
    if price > 0:
        if is_long and price >= mid:
            return "Stop loss price must be below current market price for long positions."
        if not is_long and price <= mid:
            return "Stop loss price must be above current market price for short positions."
    return None


def _validate_take_profit_price(price: float, mid: float, is_long: bool) -> Optional[str]:
    """Validate take profit price. Returns error message if invalid, None if valid."""
    if price <= 0:
        return "Price must be greater than 0."
    if is_long and price <= mid:
        return "Take profit price must be above current market price for long positions."
    if not is_long and price >= mid:
        return "Take profit price must be below current market price for short positions."
    return None


async def _handle_price_input(
    update: Update,
    context: ContextType,
    field_name: str,
    validator: Any,
    current_state: int,
    next_action: Any
) -> int:
    """Generic handler for price input (stop loss or take profit)."""
    if not update.message:
        return ConversationHandler.END

    text = update.message.text
    if text.lower() == 'cancel':  # type: ignore
        await telegram_utils.reply(update, OPERATION_CANCELLED)
        return ConversationHandler.END

    coin = context.user_data["selected_coin"]  # type: ignore
    try:
        price = float(text)  # type: ignore
        mid = _get_mid_price(coin)
        is_long = _is_long_position(context)
        
        error = validator(price, mid, is_long)
        if error:
            await telegram_utils.reply(update, error)
            return current_state

        context.user_data[field_name] = price  # type: ignore
    except ValueError:
        await telegram_utils.reply(update, "Invalid price. Please enter a number or 'cancel'.")
        return current_state

    return await next_action(update, context, coin, mid)


async def _after_stop_loss_set(update: Update, context: ContextType, coin: str, mid: float) -> int:
    """Action after stop loss is set - prompt for take profit."""
    await telegram_utils.reply(update, f"Loading price suggestions for {coin}...")
    await telegram_utils.reply(update, await get_price_suggestions_text(context, False), parse_mode=ParseMode.HTML)
    return SELECTING_TAKE_PROFIT


async def _after_take_profit_set(update: Update, context: ContextType, coin: str, mid: float) -> int:
    """Action after take profit is set - open the order."""
    return await open_order(context, hyperliquid_utils.info.user_state(hyperliquid_utils.address), mid)


async def selected_stop_loss(update: Update, context: ContextType) -> int:
    return await _handle_price_input(
        update, context, "stop_loss_price",
        _validate_stop_loss_price, SELECTING_STOP_LOSS, _after_stop_loss_set
    )


async def selected_take_profit(update: Update, context: ContextType) -> int:
    return await _handle_price_input(
        update, context, "take_profit_price",
        _validate_take_profit_price, SELECTING_TAKE_PROFIT, _after_take_profit_set
    )


async def selected_coin(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, coin: str) -> int:
        await query.edit_message_text("Loading...")
        context.user_data["selected_coin"] = coin  # type: ignore
        message, reply_markup = await get_amount_suggestions(context)
        await query.edit_message_text(message, reply_markup=reply_markup)
        return SELECTING_AMOUNT
    return await _handle_callback_selection(update, None, "Invalid coin.", process)


def calculate_available_margin() -> float:
    perp_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    cross_margin_account_value = float(perp_state['crossMarginSummary']['accountValue'])
    total_margin_used = float(perp_state['crossMarginSummary']['totalMarginUsed'])
    return max(cross_margin_account_value - total_margin_used, 0.0)


async def get_amount_suggestions(context: ContextType) -> Tuple[str, InlineKeyboardMarkup]:
    available_margin = calculate_available_margin()

    keyboard = [
        [InlineKeyboardButton(f"{amount}% (~{fmt(available_margin * amount / 100.0)} USDC)", callback_data=str(amount))]
        for amount in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ]
    reply_markup = _create_keyboard_with_cancel(keyboard)
    coin = context.user_data['selected_coin']  # type: ignore
    enter_mode = context.user_data['enter_mode']  # type: ignore
    return f"You selected {coin}. Please enter the amount to {enter_mode}:", reply_markup

def _validate_order_context(context: ContextType) -> Optional[str]:
    """Validate that all required order data is present. Returns error message if invalid."""
    required_fields = [
        ('amount', 'No amount selected'),
        ('stop_loss_price', 'No stop loss selected'),
        ('take_profit_price', 'No take profit selected'),
        ('selected_coin', 'No coin selected'),
    ]
    for field, error_msg in required_fields:
        if not context.user_data.get(field):  # type: ignore
            return f"Error: {error_msg}. Please restart the process."
    return None


async def open_order(context: ContextType, user_state: Dict[str, Any], mid: float) -> int:
    error = _validate_order_context(context)
    if error:
        await telegram_utils.send(error)
        return ConversationHandler.END

    # After validation, these are guaranteed to be set
    amount: float = context.user_data.get('amount')  # type: ignore
    stop_loss_price: float = context.user_data.get('stop_loss_price')  # type: ignore
    take_profit_price: float = context.user_data.get('take_profit_price')  # type: ignore
    selected_coin: str = context.user_data.get('selected_coin')  # type: ignore
    is_long = _is_long_position(context)

    message = await telegram_utils.send(f"Opening {'long' if is_long else 'short'} for {selected_coin}...")
    try:
        exchange = hyperliquid_utils.get_exchange()
        if not exchange:
            await telegram_utils.send("Exchange is not enabled")
            return ConversationHandler.END

        available_margin = calculate_available_margin()
        balance_to_use = available_margin * amount / 100.0
        leverage: int = context.user_data.get('leverage', 1)  # type: ignore
        use_isolated_leverage = os.getenv('HTB_USE_ISOLATED_LEVERAGE', 'True') == 'True'
        update_leverage_result = exchange.update_leverage(leverage, selected_coin, not use_isolated_leverage)
        logger.info(update_leverage_result)
        
        sz_decimals = hyperliquid_utils.get_sz_decimals()
        sz = round(balance_to_use * leverage / mid, sz_decimals[selected_coin])
        if sz * mid < 10.0:
            await telegram_utils.send("The order value is less than 10 USDC and can't be executed")
            return ConversationHandler.END

        open_result = exchange.market_open(selected_coin, is_long, sz)
        logger.info(open_result)

        if has_order_error(open_result):
            error_msg = get_order_error_message(open_result)
            await telegram_utils.send(f"Failed to open position: {error_msg}")
        elif not skip_sl_tp_prompt():
            await place_stop_loss_and_take_profit_orders(
                exchange, user_state, selected_coin, is_long, sz, stop_loss_price, take_profit_price
            )
        
        await message.delete()  # type: ignore
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.send(f"Failed to update orders: {str(e)}")

    return ConversationHandler.END


def _place_trigger_order(
    exchange: Any,
    coin: str,
    is_long: bool,
    sz: float,
    trigger_px: float,
    tpsl: str
) -> None:
    """Place a trigger order (stop loss or take profit)."""
    if trigger_px <= 0:
        return
    # For SL: limit below trigger for longs, above for shorts
    # For TP: limit above trigger for longs, below for shorts
    if tpsl == "sl":
        limit_px = trigger_px * (0.97 if is_long else 1.03)
    else:
        limit_px = trigger_px * (1.02 if is_long else 0.98)
    
    order_type = {"trigger": {"triggerPx": px_round(trigger_px), "isMarket": True, "tpsl": tpsl}}
    result = exchange.order(coin, not is_long, sz, px_round(limit_px), order_type, reduce_only=True)
    logger.info(result)


def _get_adjusted_stop_loss_trigger(
    user_state: Dict[str, Any],
    coin: str,
    is_long: bool,
    stop_loss_price: float
) -> float:
    """Get stop loss trigger price, adjusted for liquidation if needed."""
    sl_trigger_px = stop_loss_price
    liquidation_px_str = hyperliquid_utils.get_liquidation_px_str(user_state, coin)

    if liquidation_px_str is not None:
        liquidation_px = float(liquidation_px_str)
        if liquidation_px > 0.0:
            liquidation_trigger_px = liquidation_px * (1.0025 if is_long else 0.9975)
            should_use_liquidation = (
                stop_loss_price == 0 or
                (is_long and stop_loss_price < liquidation_trigger_px) or
                (not is_long and stop_loss_price > liquidation_trigger_px)
            )
            if should_use_liquidation:
                sl_trigger_px = liquidation_trigger_px
    
    return sl_trigger_px


def place_stop_loss_order(
    exchange: Any,
    user_state: Dict[str, Any],
    selected_coin: str,
    is_long: bool,
    sz: float,
    stop_loss_price: float
) -> None:
    sl_trigger_px = _get_adjusted_stop_loss_trigger(user_state, selected_coin, is_long, stop_loss_price)
    _place_trigger_order(exchange, selected_coin, is_long, sz, sl_trigger_px, "sl")


async def place_stop_loss_and_take_profit_orders(
    exchange: Any,
    user_state: Dict[str, Any],
    selected_coin: str,
    is_long: bool,
    sz: float,
    stop_loss_price: float,
    take_profit_price: float
) -> None:
    place_stop_loss_order(exchange, user_state, selected_coin, is_long, sz, stop_loss_price)
    _place_trigger_order(exchange, selected_coin, is_long, sz, take_profit_price, "tp")


def close_all_positions_core(exchange: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Close all open positions without doing any messaging."""

    closed: List[str] = []
    errors: List[Tuple[str, str]] = []

    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)

    for asset_position in user_state.get("assetPositions", []):
        coin = asset_position.get('position', {}).get('coin')
        if not coin:
            continue
        try:
            exchange.market_close(coin)
            closed.append(coin)
        except Exception as e:  # pragma: no cover - exchange errors depend on external service
            logger.error(f"Failed to close {coin}: {e}", exc_info=True)
            errors.append((coin, str(e)))

    return closed, errors

async def exit_all_positions(update: Update, context: ContextType) -> None:
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            close_all_positions_core(exchange)
        else:
            await telegram_utils.reply(update, "Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.reply(update, f"Failed to exit all positions: {str(e)}")


async def exit_position(update: Update, context: ContextType) -> int:
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
    coins = sorted({asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])})

    if not coins:
        await telegram_utils.reply(update, 'No positions to close')
        return ConversationHandler.END

    keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
    keyboard.append([InlineKeyboardButton("All", callback_data='all')])
    reply_markup = _create_keyboard_with_cancel(keyboard)
    await telegram_utils.reply(update, 'Choose a coin to close, or select All:', reply_markup=reply_markup)
    return EXIT_CHOOSING


async def _close_position_with_exchange(
    query: CallbackQuery,
    close_action: Any,
    error_prefix: str
) -> int:
    """Helper to close position(s) with exchange error handling."""
    try:
        exchange = hyperliquid_utils.get_exchange()
        if exchange:
            close_action(exchange)
            await query.delete_message()
        else:
            await query.edit_message_text("Exchange is not enabled")
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(f"{error_prefix}: {str(e)}")
    return ConversationHandler.END


async def exit_selected_coin(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, coin: str) -> int:
        if coin == 'all':
            await query.edit_message_text("Closing all positions...")
            return await _close_position_with_exchange(
                query, close_all_positions_core, "Failed to exit all positions"
            )
        await query.edit_message_text(f"Closing {coin}...")
        return await _close_position_with_exchange(
            query, lambda ex: ex.market_close(coin), f"Failed to exit {coin}"
        )
    return await _handle_callback_selection(update, None, "Invalid coin.", process)
