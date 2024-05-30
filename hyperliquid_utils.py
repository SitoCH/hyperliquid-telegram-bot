import os

from logging_utils import logger

from hyperliquid.info import Info
from hyperliquid.utils import constants

from telegram_utils import send_message_and_exit

user_address = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER"]

hyperliquid_info = Info(constants.MAINNET_API_URL)


def setup_hyperliquid() -> None:

    hyperliquid_info.ws_manager.ws.on_error = on_websocket_error
    hyperliquid_info.ws_manager.ws.on_close = on_websocket_close


def on_websocket_error(self, ws, error):
    logger.error(f"Websocket error: {error}")
    self.telegram_app.job_queue.run_once(send_message_and_exit, when=0, data="Websocket error, restarting the application...", chat_id=self.telegram_chat_id)


def on_websocket_close(self, ws, close_status_code, close_msg):
    logger.warning(f"Websocket closed: {close_msg}")
