
import json
import os

from typing import List

from hyperliquid.utils.types import UserEventsMsg, Fill

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import CommandHandler, ContextTypes, CallbackQueryHandler, ConversationHandler

from hyperliquid_orders import get_open_orders, update_open_orders, update_orders_command
from hyperliquid_trade import SELECTING_COIN, SELECTING_AMOUNT, EXIT_CHOOSING, enter_long, enter_short, selected_amount, selected_coin, exit_position, exit_selected_coin, trade_cancel
from hyperliquid_utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import exchange_enabled


class HyperliquidBot:

    def __init__(self):

        hyperliquid_utils.info.subscribe(
            {"type": "userEvents", "user": hyperliquid_utils.address}, self.on_user_events
        )

        telegram_utils.add_handler(CommandHandler("start", self.start))
        telegram_utils.add_handler(CommandHandler("positions", self.get_positions))
        telegram_utils.add_handler(CommandHandler("orders", get_open_orders))

        if exchange_enabled:
            telegram_utils.add_handler(CommandHandler("update_orders", update_orders_command))

            telegram_utils.run_repeating(update_open_orders, interval=600, first=15)

            sell_conv_handler = ConversationHandler(
                entry_points=[CommandHandler('exit', exit_position)],
                states={
                    EXIT_CHOOSING: [CallbackQueryHandler(exit_selected_coin)]
                },
                fallbacks=[CommandHandler('cancel', trade_cancel)]
            )
            telegram_utils.add_handler(sell_conv_handler)

            enter_long_conv_handler = ConversationHandler(
                entry_points=[CommandHandler('long', enter_long)],
                states={
                    SELECTING_COIN: [CallbackQueryHandler(selected_coin)],
                    SELECTING_AMOUNT: [CallbackQueryHandler(selected_amount)]
                },
                fallbacks=[CommandHandler('cancel', trade_cancel)]
            )
            telegram_utils.add_handler(enter_long_conv_handler)

            enter_short_conv_handler = ConversationHandler(
                entry_points=[CommandHandler('short', enter_short)],
                states={
                    SELECTING_COIN: [CallbackQueryHandler(selected_coin)],
                    SELECTING_AMOUNT: [CallbackQueryHandler(selected_amount)]
                },
                fallbacks=[CommandHandler('cancel', trade_cancel)]
            )
            telegram_utils.add_handler(enter_short_conv_handler)

        telegram_utils.run_polling()


    def get_fill_icon(self, closed_pnl: float) -> str:
        return "🟢" if closed_pnl > 0 else "🔴"

    def get_fill_description(self, initial_message: str, coin: str, size: str, fee: float, fee_token: str, amount: float = None, closed_pnl: float = None) -> str:
        fill_description = [
            initial_message,
            f"Coin: {coin}"
        ]

        if amount is not None:
            fill_description.append(f"Amount: {amount:,.02f} USDC")

        if closed_pnl is not None:
            fill_description.append(f"Profit: {closed_pnl:,.02f} USDC")

        fill_description.append(f"Size: {size}")
        fill_description.append(f"Fee: {fee:,.02f} {fee_token}")

        return '\n'.join(fill_description)

    def process_fill(self, fill: Fill) -> None:

        price = float(fill["px"])
        coin = fill["coin"]
        size = fill["sz"]
        fee = float(fill["fee"])
        fee_token = fill["feeToken"]
        amount = price * float(size)
        closed_pnl = float(fill["closedPnl"])
        if fill["dir"] == 'Open Long':
            fill_message = self.get_fill_description("🔵 Opened long:", coin, size, fee, fee_token, amount)
        elif fill["dir"] == 'Open Short':
            fill_message = self.get_fill_description("🔵 Opened short:", coin, size, fee, fee_token, amount)
        elif fill["dir"] == 'Close Long':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Closed long:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
        elif fill["dir"] == 'Close Short':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Closed short:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
        elif fill["dir"] == 'Buy':
            fill_message = self.get_fill_description("🔵 Bought spot:", coin, size, fee, fee_token, amount)
        elif fill["dir"] == 'Sell':
            fill_message = self.get_fill_description("🔵 Sold spot:", coin, size, fee, fee_token, amount)
        elif fill["dir"] == 'Liquidated Isolated Long':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Liquidated isolated long:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
        elif fill["dir"] == 'Long > Short':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Long -> short:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
        else:
            fill_message = json.dumps(fill)

        telegram_utils.send(fill_message)

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            fill_events: List[Fill] = user_events_data["fills"]
            for fill in fill_events:
                self.process_fill(fill)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        await update.message.reply_text(
            "Welcome! Click the button below to check the account's positions.",
            reply_markup=telegram_utils.reply_markup,
        )

    async def get_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:

        try:
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            print(user_state)
            total_balance = float(user_state['marginSummary']['accountValue'])

            message_lines = [
                "<b>Perps positions:</b>",
                f"Total balance: {total_balance:,.2f} USDC",
                f"Available balance: {float(user_state['withdrawable']):,.2f} USDC",
            ]

            tablefmt = simple_separated_format(' ')
            if len(user_state["assetPositions"]) > 0:

                total_pnl = sum(
                    float(asset_position['position']['unrealizedPnl'])
                    for asset_position in user_state["assetPositions"]
                )
                message_lines.append(f"Unrealized profit: {total_pnl:,.2f} USDC")

                message_lines.append("Open positions:")

                sorted_positions = sorted(
                    user_state["assetPositions"],
                    key=lambda x: float(x['position']['positionValue']),
                    reverse=True
                )

                table = tabulate(
                    [
                        [
                            f"{asset_position['position']['szi']}",
                            f"{asset_position['position']['coin']}",
                            f"{float(asset_position['position']['positionValue']):,.2f}",
                            f"{float(asset_position['position']['unrealizedPnl']):,.2f}"
                        ]
                        for asset_position in sorted_positions
                    ],
                    headers=[
                        "Size",
                        "Coin",
                        "Value ($)",
                        "PnL ($)",
                    ],
                    tablefmt=tablefmt,
                    colalign=("right", "left", "right", "right")
                )

                message_lines.append(f"<pre>{table}</pre>")

            spot_user_state = hyperliquid_utils.info.spot_user_state(hyperliquid_utils.address)
            if len(spot_user_state['balances']) > 0:
                message_lines.append("<b>Spot positions:</b>")

                spot_table = tabulate(
                    [
                        [
                            f"{float(balance['total']):,.2f}",
                            balance["coin"],
                        ]
                        for balance in spot_user_state['balances']
                    ],
                    headers=[
                        "Total",
                        "Coin"
                    ],
                    tablefmt=tablefmt,
                    colalign=("right", "left")
                )

                message_lines.append(f"<pre>{spot_table}</pre>")

            message = '\n'.join(message_lines)

        except Exception as e:
            message = f"Failed to fetch positions: {str(e)}"

        await update.message.reply_text(text=message, parse_mode=ParseMode.HTML, reply_markup=telegram_utils.reply_markup)


if __name__ == "__main__":
    bot = HyperliquidBot()
    os._exit(0)
