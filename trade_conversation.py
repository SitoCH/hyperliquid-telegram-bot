import os
from typing import Dict, Any, List, Union, Tuple, Callable, Awaitable, Optional
from logging_utils import logger
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import ConversationHandler, CallbackContext, ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils.utils import hyperliquid_utils
from telegram.constants import ParseMode
from utils import OPERATION_CANCELLED, fmt
from trade_pricing import (
    get_price_suggestions_text,
    validate_stop_loss_price,
    validate_take_profit_price
)
from trade_execution import (
    open_order,
    close_all_positions_core,
    calculate_available_margin
)

EXIT_CHOOSING, SELECTING_DEX, SELECTING_COIN, SELECTING_STOP_LOSS, SELECTING_TAKE_PROFIT, SELECTING_AMOUNT, SELECTING_LEVERAGE = range(7)

# Type alias for context types used throughout the module
ContextType = Union[CallbackContext[Any, Any, Any, Any], ContextTypes.DEFAULT_TYPE]


def _has_sl_tp_set(context: ContextType) -> bool:
    """Check if both stop loss and take profit prices are set in context."""
    return 'stop_loss_price' in context.user_data and 'take_profit_price' in context.user_data  # type: ignore


def _get_mid_price(coin: str) -> float:
    """Get current mid price for a coin, resolving the correct DEX."""
    dex = hyperliquid_utils.dex_supported(coin) or ""
    return float(hyperliquid_utils.info.all_mids(dex=dex)[coin])


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
    converter: Optional[Callable[[Any], Any]],
    invalid_msg: str,
    next_action: Callable[[CallbackQuery, Any], Awaitable[int]]
) -> int:
    """Generic handler for callback query selections (amount, leverage, coin)."""
    query: Optional[CallbackQuery] = update.callback_query
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


async def enter_position(update: Update, context: ContextType, enter_mode: str) -> int:
    context.user_data.clear()  # type: ignore
    context.user_data["enter_mode"] = enter_mode  # type: ignore
    if skip_sl_tp_prompt():
        context.user_data["stop_loss_price"] = "skip"  # type: ignore
        context.user_data["take_profit_price"] = "skip"  # type: ignore

    if context.args and len(context.args) > 2:
        context.user_data["selected_coin"] = context.args[0]  # type: ignore
        context.user_data["stop_loss_price"] = float(context.args[1].replace(",", ""))  # type: ignore
        context.user_data["take_profit_price"] = float(context.args[2].replace(",", ""))  # type: ignore
        message, reply_markup = await get_amount_suggestions(context)
        await telegram_utils.reply(update, message, reply_markup=reply_markup)
        return SELECTING_AMOUNT

    # If extra DEXes are configured, ask the user which DEX first
    if hyperliquid_utils.extra_dexes():
        await telegram_utils.reply(
            update,
            f"Choose a DEX to {enter_mode}:",
            reply_markup=hyperliquid_utils.get_dex_reply_markup(),
        )
        return SELECTING_DEX

    await telegram_utils.reply(update, f'Choose a coin to {enter_mode}:', reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return SELECTING_COIN


async def enter_long(update: Update, context: ContextType) -> int:
    return await enter_position(update, context, "long")


async def enter_short(update: Update, context: ContextType) -> int:
    return await enter_position(update, context, "short")


async def selected_dex(update: Update, context: ContextType) -> int:
    """Handle DEX selection from the /long or /short flow."""
    async def process(query: CallbackQuery, raw: str) -> int:
        await query.edit_message_text("Loading...")
        # callback_data is "__dex__" for default or "__dex__xyz" for extra DEXes
        dex = raw.removeprefix("__dex__")
        context.user_data["selected_dex"] = dex  # type: ignore
        label = "Default (Hyperliquid)" if not dex else dex.upper()
        coins_markup = hyperliquid_utils.get_coins_reply_markup(dex=dex)
        await query.edit_message_text(
            f"Choose a coin on {label}:",
            reply_markup=coins_markup,
        )
        return SELECTING_COIN
    return await _handle_callback_selection(update, None, "Invalid DEX.", process)


def skip_sl_tp_prompt() -> bool:
    return os.getenv('HTB_SKIP_SL_TP_PROMPT', 'False') == 'True'


async def _process_amount_selection(query: CallbackQuery, context: ContextType, amount: float) -> int:
    """Process the amount selection and determine next step."""
    context.user_data["amount"] = amount  # type: ignore
    coin = context.user_data.get("selected_coin", "")  # type: ignore
    dex = hyperliquid_utils.dex_supported(coin) or ""
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex)
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
        await open_order(
            context.user_data,  # type: ignore
            user_state,
            _get_mid_price(coin),
            _is_long_position(context),
            skip_sl_tp_prompt()
        )
        return ConversationHandler.END
    await send_stop_loss_suggestions(query, context)
    return SELECTING_STOP_LOSS


async def _prompt_for_leverage(query: CallbackQuery, coin: str) -> int:
    """Prompt user to select leverage for the given coin."""
    try:
        dex = hyperliquid_utils.dex_supported(coin) or ""
        meta: List[Dict[str, Any]] = hyperliquid_utils.info.meta(dex=dex)
        asset_info_map = {
            info["name"]: int(info["maxLeverage"])
            for info in meta.get("universe", [])  # type: ignore
        }
        max_leverage = asset_info_map.get(coin, 1)
        suggested_leverages = [lev for lev in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30] if lev <= max_leverage]

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
        dex = hyperliquid_utils.dex_supported(coin) or ""
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex)
        return await _proceed_after_leverage_set(query, context, user_state, coin)
    return await _handle_callback_selection(update, int, "Invalid leverage.", process)


async def send_stop_loss_suggestions(query: CallbackQuery, context: ContextType) -> None:
    coin = context.user_data['selected_coin']  # type: ignore
    is_long = _is_long_position(context)
    await query.edit_message_text(f"Loading price suggestions for {coin}...")
    await query.edit_message_text(
        await get_price_suggestions_text(coin, True, is_long),
        parse_mode=ParseMode.HTML
    )


async def _handle_price_input(
    update: Update,
    context: ContextType,
    field_name: str,
    validator: Callable[[float, float, bool], Optional[str]],
    current_state: int,
    next_action: Callable[[Update, ContextType, str, float], Awaitable[int]]
) -> int:
    """Generic handler for price input (stop loss or take profit)."""
    if not update.message:
        return ConversationHandler.END

    text = update.message.text
    if text and text.lower() == 'cancel':
        await telegram_utils.reply(update, OPERATION_CANCELLED)
        return ConversationHandler.END

    coin = context.user_data["selected_coin"]  # type: ignore
    try:
        price = float(text) if text else 0.0
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
    is_long = _is_long_position(context)
    await telegram_utils.reply(update, f"Loading price suggestions for {coin}...")
    await telegram_utils.reply(
        update,
        await get_price_suggestions_text(coin, False, is_long),
        parse_mode=ParseMode.HTML
    )
    return SELECTING_TAKE_PROFIT


async def _after_take_profit_set(update: Update, context: ContextType, coin: str, mid: float) -> int:
    """Action after take profit is set - open the order."""
    dex = hyperliquid_utils.dex_supported(coin) or ""
    await open_order(
        context.user_data,  # type: ignore
        hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex),
        mid,
        _is_long_position(context),
        skip_sl_tp_prompt()
    )
    return ConversationHandler.END


async def selected_stop_loss(update: Update, context: ContextType) -> int:
    return await _handle_price_input(
        update, context, "stop_loss_price",
        validate_stop_loss_price, SELECTING_STOP_LOSS, _after_stop_loss_set
    )


async def selected_take_profit(update: Update, context: ContextType) -> int:
    return await _handle_price_input(
        update, context, "take_profit_price",
        validate_take_profit_price, SELECTING_TAKE_PROFIT, _after_take_profit_set
    )


async def selected_coin(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, coin: str) -> int:
        await query.edit_message_text("Loading...")
        context.user_data["selected_coin"] = coin  # type: ignore
        message, reply_markup = await get_amount_suggestions(context)
        await query.edit_message_text(message, reply_markup=reply_markup)
        return SELECTING_AMOUNT
    return await _handle_callback_selection(update, None, "Invalid coin.", process)


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
    """Show positions from all DEXes (default + extra) for closing."""
    coins: list[str] = []
    dexes_to_check: list[str] = [""] + hyperliquid_utils.extra_dexes()

    for dex in dexes_to_check:
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex)
        coins.extend(
            ap['position']['coin']
            for ap in user_state.get("assetPositions", [])
        )

    coins = sorted(set(coins))

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
    coin: Optional[str],
) -> int:
    """Close position(s) on the correct DEX with error handling.

    Args:
        coin: The coin to close. If None or 'all', closes all positions.
    """
    try:
        if coin and coin != 'all':
            # Close a specific coin on its DEX
            dex = hyperliquid_utils.dex_supported(coin) or ""
            exchange = hyperliquid_utils.get_exchange(dex=dex)
            if not exchange:
                await query.edit_message_text("Exchange is not enabled")
                return ConversationHandler.END
            exchange.market_close(coin)
            await query.delete_message()
        else:
            # Close all positions across all DEXes
            exchange = hyperliquid_utils.get_exchange()
            if not exchange:
                await query.edit_message_text("Exchange is not enabled")
                return ConversationHandler.END
            close_all_positions_core(exchange)
            await query.delete_message()
    except Exception as e:
        logger.critical(e, exc_info=True)
        await query.edit_message_text(f"Failed to close position: {str(e)}")
    return ConversationHandler.END


async def exit_selected_coin(update: Update, context: ContextType) -> int:
    async def process(query: CallbackQuery, coin: str) -> int:
        if coin == 'all':
            await query.edit_message_text("Closing all positions...")
            return await _close_position_with_exchange(query, None)
        await query.edit_message_text(f"Closing {coin}...")
        return await _close_position_with_exchange(query, coin)
    return await _handle_callback_selection(update, None, "Invalid coin.", process)
