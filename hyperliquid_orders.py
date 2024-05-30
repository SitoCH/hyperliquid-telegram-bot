
from collections import defaultdict

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram_utils import reply_markup

from hyperliquid_utils import hyperliquid_info, user_address


async def get_open_orders(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:

    try:
        open_orders = hyperliquid_info.frontend_open_orders(user_address)
        grouped_data = defaultdict(lambda: defaultdict(list))

        for order in open_orders:
            coin = order["coin"]
            order_type = order["orderType"]
            grouped_data[coin][order_type].append(order)

        grouped_data = {coin: dict(order_types) for coin, order_types in grouped_data.items()}

        all_mids = hyperliquid_info.all_mids()

        for coin, order_types in grouped_data.items():

            message_lines = [
                f"<b>Coin {coin}:</b>"
            ]
            mid = float(all_mids[coin])

            tp_raw_orders = order_types.get('Take Profit Market', [])
            tp_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)
            tp_orders = [
                        [
                            f"{order['sz']}",
                            f"{order['triggerPx']}",
                            f"{((float(order['triggerPx']) / mid - 1) * 100):.2f} %"
                        ]
                for order in tp_raw_orders
            ]

            sl_raw_orders = order_types.get('Stop Market', [])
            sl_raw_orders.sort(key=lambda x: x["triggerPx"], reverse=True)
            sl_orders = [
                        [
                            f"{order['sz']}",
                            f"{order['triggerPx']}",
                            f"{((1 - float(order['triggerPx']) / mid) * 100):.2f} %"
                        ]
                for order in sl_raw_orders
            ]

            tablefmt = simple_separated_format(' ')
            table = tabulate(
                tp_orders + [["Current", all_mids[coin], ""]] + sl_orders,
                headers=[
                    "Size",
                    "Trigger price",
                    "Distance"
                ],
                tablefmt=tablefmt
            )

            message_lines.append(f"<pre>{table}</pre>")

            message = '\n'.join(message_lines)
            await update.message.reply_text(text=message, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

    except Exception as e:
        await update.message.reply_text(text=f"Failed to check orders: {str(e)}")

