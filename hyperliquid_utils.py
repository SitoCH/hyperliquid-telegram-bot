import os

import eth_account
from eth_account.signers.local import LocalAccount

from logging_utils import logger

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from telegram_utils import telegram_utils


class HyperliquidUtils:

    def __init__(self):

        user_wallet = os.environ["HYPERLIQUID_TELEGRAM_BOT_USER_WALLET"]
        user_vault = os.environ.get("HYPERLIQUID_TELEGRAM_BOT_USER_VAULT")
        self.address = user_vault if user_vault is not None else user_wallet

        self.info = Info(constants.MAINNET_API_URL)
        self.info.ws_manager.ws.on_error = self.on_websocket_error
        self.info.ws_manager.ws.on_close = self.on_websocket_close

    def on_websocket_error(self, ws, error):
        logger.error(f"Websocket error: {error}")
        telegram_utils.send_and_exit("Websocket error, restarting the application...")

    def on_websocket_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Websocket closed: {close_msg}")

    def get_exchange(self):
        key_file = os.environ.get("HYPERLIQUID_TELEGRAM_BOT_KEY_FILE")
        if key_file is not None and os.path.isfile(key_file):
            with open(key_file, 'r') as file:
                file_content = file.read()
                account: LocalAccount = eth_account.Account.from_key(file_content)
                return Exchange(account, constants.MAINNET_API_URL, vault_address=os.environ.get("HYPERLIQUID_TELEGRAM_BOT_USER_VAULT"), account_address=os.environ["HYPERLIQUID_TELEGRAM_BOT_USER_WALLET"])

    def get_sz_decimals(self):
        meta = self.info.meta()
        sz_decimals = {}
        for asset_info in meta["universe"]:
            sz_decimals[asset_info["name"]] = asset_info["szDecimals"]
        return sz_decimals

    def get_entry_px(self, user_state, selected_coin) -> float:
        for asset_position in user_state.get("assetPositions", []):
            if asset_position['position']['coin'] == selected_coin:
                return float(asset_position['position']['entryPx'])
        return 0.0

    def get_unrealized_pnl(self, user_state, selected_coin) -> float:
        for asset_position in user_state.get("assetPositions", []):
            if asset_position['position']['coin'] == selected_coin:
                return float(asset_position['position']['unrealizedPnl'])
        return 0.0


    def get_margin_used(self, user_state, selected_coin) -> float:
        for asset_position in user_state.get("assetPositions", []):
            if asset_position['position']['coin'] == selected_coin:
                return float(asset_position['position']['marginUsed'])
        return 0.0


    def get_leverage(self, user_state, selected_coin) -> int:
        for asset_position in user_state.get("assetPositions", []):
            if asset_position['position']['coin'] == selected_coin:
                return int(asset_position['position']['leverage']['value'])

        meta = hyperliquid_utils.info.meta()
        for asset_info in meta.get("universe", []):
            if asset_info["name"] == selected_coin:
                return min(int(asset_info["maxLeverage"]), 40)

        return 5



hyperliquid_utils = HyperliquidUtils()
