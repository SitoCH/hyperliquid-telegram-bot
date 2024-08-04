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


    def _get_asset_position(self, user_state, selected_coin):
        return next(
            (asset_position['position'] for asset_position in user_state.get("assetPositions", [])
             if asset_position['position']['coin'] == selected_coin),
            None
        )

    def get_size(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['szi']) if position else 0.0

    def get_entry_px_str(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return position['entryPx'] if position else None

    def get_liquidation_px_str(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return position['liquidationPx'] if position else None

    def get_entry_px(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['entryPx']) if position else 0.0


    def get_unrealized_pnl(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['unrealizedPnl']) if position else 0.0


    def get_return_on_equity(self, user_state, selected_coin) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['returnOnEquity']) if position else 0.0


    def get_leverage(self, user_state, selected_coin) -> int:
        position = self._get_asset_position(user_state, selected_coin)
        if position:
            return int(position['leverage']['value'])

        meta = self.info.meta()
        asset_info = next(
            (info for info in meta.get("universe", []) if info["name"] == selected_coin),
            None
        )
        if asset_info:
            max_leverage = int(asset_info["maxLeverage"])
            if max_leverage > 20:
                return 20
            if max_leverage > 10:
                return 15
            if max_leverage > 5:
                return 8
        return 5


    def get_coins_with_open_positions(self):
        user_state = self.info.user_state(self.address)
        return [asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])]


hyperliquid_utils = HyperliquidUtils()
