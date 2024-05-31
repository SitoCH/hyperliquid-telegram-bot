import os

import eth_account
from eth_account.signers.local import LocalAccount

from logging_utils import logger

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from telegram_utils import send_message_and_exit


class HyperliquidUtils:

    def __init__(self):

        self.user_address = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER"]

        self.info = Info(constants.MAINNET_API_URL)
        self.info.ws_manager.ws.on_error = self.on_websocket_error
        self.info.ws_manager.ws.on_close = self.on_websocket_close

        key_file = os.environ["HYPERLIQUID_TELEGRAM_BOT_KEY_FILE"]
        if key_file and os.path.isfile(key_file):
            with open(key_file, 'r') as file:
                file_content = file.read()
                account: LocalAccount = eth_account.Account.from_key(file_content)
                self.exchange = Exchange(account, constants.MAINNET_API_URL, account_address=self.user_address)
            logger.info("Key file found, exchange enabled")


    def on_websocket_error(self, ws, error):
        logger.error(f"Websocket error: {error}")
        self.telegram_app.job_queue.run_once(send_message_and_exit, when=0, data="Websocket error, restarting the application...", chat_id=self.telegram_chat_id)


    def on_websocket_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Websocket closed: {close_msg}")


hyperliquid_utils = HyperliquidUtils()
