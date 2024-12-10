from collections import defaultdict
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import fmt, px_round


async def get_orders_from_hyperliquid():
    open_orders = hyperliquid_utils.info.frontend_open_orders(hyperliquid_utils.address)
    grouped_data = defaultdict(lambda: defaultdict(list))
    for order in open_orders:
        grouped_data[order["coin"]][order["orderType"]].append(order)
    return {coin: dict(order_types) for coin, order_types in sorted(grouped_data.items())}


def get_sl_tp_orders(order_types, is_long: bool):
    sl_raw_orders = order_types.get('Stop Market', [])
    sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    tp_raw_orders = order_types.get('Take Profit Market', [])
    tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=is_long)
    return sl_raw_orders, tp_raw_orders


async def get_open_orders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        grouped_data = await get_orders_from_hyperliquid()
        all_mids = hyperliquid_utils.info.all_mids()
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        message_lines = []

        # Get all coins with positions
        coins_with_positions = {
            pos['position']['coin']: pos['position']
            for pos in user_state.get("assetPositions", [])
            if float(pos['position']['szi']) != 0
        }

        # Combine coins from orders and positions
        all_coins = set(list(grouped_data.keys()) + list(coins_with_positions.keys()))

        for coin in sorted(all_coins):
            message_lines.append(f"<b>{coin}</b>")
            mid = float(all_mids[coin])
            position = coins_with_positions.get(coin)
            
            # If we have a position for this coin
            if position:
                is_long = float(position['szi']) > 0
                message_lines.append(f"Mode: {'long' if is_long else 'short'}")
                message_lines.append(f"Leverage: {position['leverage']['value']}x")

                order_types = grouped_data.get(coin, {})
                sl_raw_orders, tp_raw_orders = get_sl_tp_orders(order_types, is_long)

                tp_orders = format_orders(tp_raw_orders, mid, percentage_format=lambda triggerPx, mid: (triggerPx / mid - 1) * 100)
                sl_orders = format_orders(sl_raw_orders, mid, percentage_format=lambda triggerPx, mid: (1 - triggerPx / mid) * 100)

                table_orders = tp_orders + [["Current", all_mids[coin], ""]] + sl_orders

                entry_px = float(position['entryPx'])
                liquidation_px = float(position['liquidationPx'])

                entry_distance = (entry_px / mid - 1) * 100
                liq_distance = (liquidation_px / mid - 1) * 100

                table_orders = insert_order(table_orders, ["Entry", entry_px, f"{fmt(entry_distance)}%"], is_long)
                table_orders = insert_order(table_orders, ["Liq.", liquidation_px, f"{fmt(liq_distance)}%"], is_long)

                if not is_long:
                    table_orders.reverse()

                table = tabulate(
                    table_orders,
                    headers=["Size", "Trigger price", "Distance"],
                    tablefmt=simple_separated_format(' '),
                    colalign=("right", "right", "right")
                )
                message_lines.append(f"<pre>{table}</pre>")

        if not message_lines:
            message_lines.append("No open orders or positions")

        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)

    except Exception as e:
        await telegram_utils.reply(update, f"Failed to check orders: {str(e)}")


def format_orders(raw_orders, mid, percentage_format):
    return [
        [
            order['sz'],
            order['triggerPx'],
            f"{fmt(percentage_format(float(order['triggerPx']), mid))}%"
        ]
        for order in raw_orders
    ]


def insert_order(orders, new_order, is_long):
    order_px = float(new_order[1])
    inserted = False

    for i in range(len(orders)):
        if (order_px > float(orders[i][1]) if is_long else order_px < float(orders[i][1])):
            orders.insert(i, new_order)
            inserted = True
            break

    if not inserted:
        orders.append(new_order)

    return orders
