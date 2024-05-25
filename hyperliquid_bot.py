import logging
import json
import os

from typing import List

from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.utils.types import UserEventsMsg, Fill

from tabulate import simple_separated_format, tabulate

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from telegram_utils import send_message, reply_markup

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class HyperliquidBot:

    def __init__(self):

        telegram_token = os.environ["HYPERLIQUID_TELEGRAM_BOT_TOKEN"]
        self.telegram_chat_id = os.environ["HYPERLIQUID_TELEGRAM_BOT_CHAT_ID"]

        self.user_address = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER"]

        self.hyperliquid_info = Info(constants.MAINNET_API_URL)
        self.hyperliquid_info.subscribe(
            {"type": "userEvents", "user": self.user_address}, self.on_user_events
        )

        self.telegram_app = Application.builder().token(telegram_token).build()

        self.telegram_app.add_handler(CommandHandler("start", self.start))
        self.telegram_app.add_handler(CommandHandler("positions", self.get_positions))

        self.hyperliquid_info.ws_manager.ws.on_error = self.on_websocket_error
        self.hyperliquid_info.ws_manager.ws.on_close = self.on_websocket_close

        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


    def on_websocket_error(self, ws, error):
        logger.error(f"Websocket error: {error}")
        self.hyperliquid_info.ws_manager.ws.close()
        self.hyperliquid_info.ws_manager.ws.run_forever()

    def on_websocket_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Websocket closed: {close_msg}")

    def get_fill_icon(self, closed_pnl: float) -> str:
        return "🟢" if closed_pnl > 0 else "🔴"

    def get_fill_description(self, initial_message: str, coin: str, size: str, amount: float = None, closed_pnl: float = None) -> str:
        fill_description = [
            initial_message,
            f"Coin: {coin}",
            f"Size: {size}",
            *(f"Amount: {amount:,.2f} USDC" if amount is not None else []),
            *(f"Profit: {closed_pnl:,.2f} USDC" if closed_pnl is not None else []),
        ]

        return '\n'.join(fill_description)

    def process_fill(self, fill: Fill) -> None:

        price = float(fill["px"])
        coin = fill["coin"]
        size = fill["sz"]
        amount = price * float(size)
        closed_pnl = float(fill["closedPnl"])
        if fill["dir"] == 'Open Long':
            fill_message = self.get_fill_description("🔵 Opened long:", coin, size, amount)
        elif fill["dir"] == 'Open Short':
            fill_message = self.get_fill_description("🔵 Opened short:", coin, size, amount)
        elif fill["dir"] == 'Close Long':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Closed long:", coin, size, closed_pnl=closed_pnl)
        elif fill["dir"] == 'Close Short':
            fill_message = self.get_fill_description(f"{self.get_fill_icon(closed_pnl)} Closed short:", coin, size, closed_pnl=closed_pnl)
        else:
            fill_message = json.dumps(fill)

        self.telegram_app.job_queue.run_once(send_message, when=0, data=fill_message, chat_id=self.telegram_chat_id)

    def on_user_events(self, user_events: UserEventsMsg) -> None:
        user_events_data = user_events["data"]
        if "fills" in user_events_data:
            fill_events: List[Fill] = user_events_data["fills"]
            for fill in fill_events:
                self.process_fill(fill)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        await update.message.reply_text(
            "Welcome! Click the button below to check the account's balance.",
            reply_markup=reply_markup,
        )

    async def get_positions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:

        try:
            user_state = self.hyperliquid_info.user_state(self.user_address)

            message_lines = [
                "<b>Account positions:</b>",
                f"Total balance: {float(user_state["crossMarginSummary"]["accountValue"]):,.2f} USDC",
                f"Available balance: {float(user_state["withdrawable"]):,.2f} USDC",
            ]

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

                tablefmt = simple_separated_format(' ')
                table = tabulate(
                    [
                        [
                            f"{asset_position['position']["szi"]}",
                            f"{asset_position['position']["coin"]}",
                            f"{float(asset_position['position']["positionValue"]):,.02f}",
                            f"{float(asset_position['position']["unrealizedPnl"]):,.02f}"
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
                    colalign=("right", "left", "right", "right"),
                    floatfmt=(".1f", ".3f")
                )

                message_lines.append(f"<pre>{table}</pre>")

                message = '\n'.join(message_lines)

        except Exception as e:
            message = f"Failed to fetch positions: {str(e)}"

        await update.message.reply_text(text=message, parse_mode=ParseMode.HTML)


if __name__ == "__main__":
    bot = HyperliquidBot()
    os._exit(0)
