import os
from typing import Dict, Any, List, Tuple, Union, Literal
from hyperliquid.exchange import Exchange, OrderType
from logging_utils import logger
from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from trade_pricing import px_round, get_adjusted_stop_loss_trigger


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
                return str(status['error'])
    return "Unknown order error"


def calculate_available_margin(dex: str = "") -> float:
    """Calculate available margin for trading on a specific perp DEX.

    Args:
        dex: The perp DEX name ('' for default Hyperliquid DEX).
    """
    user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex)
    cross_margin_account_value = float(user_state['crossMarginSummary']['accountValue'])
    total_margin_used = float(user_state['crossMarginSummary']['totalMarginUsed'])
    return max(cross_margin_account_value - total_margin_used, 0.0)


def _validate_order_context(user_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that all required order data is present. Returns (is_valid, error_message)."""
    required_fields = [
        ('amount', 'No amount selected'),
        ('stop_loss_price', 'No stop loss selected'),
        ('take_profit_price', 'No take profit selected'),
        ('selected_coin', 'No coin selected'),
    ]
    for field, error_msg in required_fields:
        if not user_data.get(field):
            return False, f"Error: {error_msg}. Please restart the process."
    return True, ""


def _determine_leverage_mode(coin: str) -> tuple[str, bool]:
    """Determine the correct DEX and leverage mode (cross/isolated) for a coin.

    Returns:
        (dex_name, is_cross) where dex_name is '' for the default DEX.
        is_cross is False for isolated-only tokens regardless of settings.
    """
    dex = hyperliquid_utils.dex_supported(coin) or ""
    only_isolated = hyperliquid_utils.get_isolated_only(coin)
    use_isolated = os.getenv('HTB_USE_ISOLATED_LEVERAGE', 'True') == 'True'
    is_cross = not use_isolated and not only_isolated
    return dex, is_cross


async def open_order(user_data: Dict[str, Any], user_state: Dict[str, Any], mid: float, is_long: bool, skip_sl_tp: bool) -> None:
    is_valid, error = _validate_order_context(user_data)
    if not is_valid:
        await telegram_utils.send(error)
        return

    # After validation, these are guaranteed to be set
    amount: float = user_data['amount']
    stop_loss_price: float = user_data['stop_loss_price']
    take_profit_price: float = user_data['take_profit_price']
    selected_coin: str = user_data['selected_coin']

    message = await telegram_utils.send(f"Opening {'long' if is_long else 'short'} for {selected_coin}...")
    try:
        dex, is_cross = _determine_leverage_mode(selected_coin)
        exchange = hyperliquid_utils.get_exchange(dex=dex)
        if not exchange:
            await telegram_utils.send("Exchange is not enabled")
            return

        available_margin = calculate_available_margin(dex=dex)
        balance_to_use = available_margin * amount / 100.0
        leverage: int = user_data.get('leverage', 1)
        update_leverage_result = exchange.update_leverage(leverage, selected_coin, is_cross)
        logger.info(update_leverage_result)

        sz_decimals = hyperliquid_utils.get_sz_decimals(dex=dex)
        sz = round(balance_to_use * leverage / mid, sz_decimals[selected_coin])
        if sz * mid < 10.0:
            await telegram_utils.send("The order value is less than 10 USDC and can't be executed")
            return

        open_result = exchange.market_open(selected_coin, is_long, sz)
        logger.info(open_result)

        if has_order_error(open_result):
            error_msg = get_order_error_message(open_result)
            await telegram_utils.send(f"Failed to open position: {error_msg}")
        elif not skip_sl_tp:
            place_stop_loss_and_take_profit_orders(
                exchange, user_state, selected_coin, is_long, sz, stop_loss_price, take_profit_price, sz_decimals[selected_coin]
            )

        if message:
            await message.delete()
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.send(f"Failed to update orders: {str(e)}")


def _place_trigger_order(
    exchange: Exchange,
    coin: str,
    is_long: bool,
    sz: float,
    trigger_px: float,
    tpsl: Union[Literal["tp"], Literal["sl"]],
    sz_decimals: int
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

    order_type: OrderType = {"trigger": {"triggerPx": px_round(trigger_px, sz_decimals), "isMarket": True, "tpsl": tpsl}}
    result = exchange.order(coin, not is_long, sz, px_round(limit_px, sz_decimals), order_type, reduce_only=True)
    logger.info(result)


def place_stop_loss_order(
    exchange: Exchange,
    user_state: Dict[str, Any],
    selected_coin: str,
    is_long: bool,
    sz: float,
    stop_loss_price: float,
    sz_decimals: int
) -> None:
    sl_trigger_px = get_adjusted_stop_loss_trigger(user_state, selected_coin, is_long, stop_loss_price)
    _place_trigger_order(exchange, selected_coin, is_long, sz, sl_trigger_px, "sl", sz_decimals)


def place_stop_loss_and_take_profit_orders(
    exchange: Exchange,
    user_state: Dict[str, Any],
    selected_coin: str,
    is_long: bool,
    sz: float,
    stop_loss_price: float,
    take_profit_price: float,
    sz_decimals: int
) -> None:
    place_stop_loss_order(exchange, user_state, selected_coin, is_long, sz, stop_loss_price, sz_decimals)
    _place_trigger_order(exchange, selected_coin, is_long, sz, take_profit_price, "tp", sz_decimals)


def close_all_positions_core(exchange: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Close all open positions across all configured DEXes.

    Closes positions on the default DEX plus any extra DEXes (e.g. XYZ).
    Uses the provided exchange for the default DEX and creates/caches
    per-DEX exchanges for the rest.

    Returns:
        (closed_coins, errors) where errors is a list of (coin, error_message).
    """
    closed: List[str] = []
    errors: List[Tuple[str, str]] = []

    # Collect positions from default DEX + all extra DEXes
    dexes_to_check: list[str] = [""] + hyperliquid_utils.extra_dexes()

    for dex in dexes_to_check:
        dex_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address, dex=dex)

        for asset_position in dex_state.get("assetPositions", []):
            coin = asset_position.get('position', {}).get('coin')
            if not coin:
                continue

            try:
                # Use the default exchange for the default DEX, get/create
                # per-DEX exchange for extra DEXes
                if dex == "" or dex is None:
                    dex_exchange = exchange
                else:
                    dex_exchange = hyperliquid_utils.get_exchange(dex=dex)
                    if not dex_exchange:
                        errors.append((coin, f"No exchange for DEX '{dex}'"))
                        continue

                dex_exchange.market_close(coin)
                closed.append(coin)
            except Exception as e:  # pragma: no cover - exchange errors depend on external service
                logger.error(f"Failed to close {coin} on DEX '{dex}': {e}", exc_info=True)
                errors.append((coin, str(e)))

    return closed, errors
