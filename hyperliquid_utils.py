import os
import requests
from typing import List, Set, Dict, Any, ClassVar

import eth_account
from eth_account.signers.local import LocalAccount
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from logging_utils import logger

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from telegram_utils import telegram_utils


class HyperliquidUtils:

    COINGECKO_URL: ClassVar[str] = "https://api.coingecko.com/api/v3/coins/markets"

    def __init__(self):

        user_wallet = os.environ.get("HTB_USER_WALLET")
        if user_wallet is None:
            logger.error("HTB_USER_WALLET environment variable is not set")
            
        user_vault = os.environ.get("HTB_USER_VAULT")
        self.address = user_vault if user_vault is not None else user_wallet

        self.info = Info(constants.MAINNET_API_URL, True)

    def init_websocket(self):
        """Initialize WebSocket connection for live data."""
        self.info = Info(constants.MAINNET_API_URL, False)
        if hasattr(self.info, 'ws_manager') and self.info.ws_manager:
            self.info.ws_manager.ws.on_error = self.on_websocket_error
            self.info.ws_manager.ws.on_close = self.on_websocket_close

    def on_websocket_error(self, ws, error):
        logger.error(f"Websocket error: {error}")
        telegram_utils.send_and_exit("Websocket error, restarting the application...")

    def on_websocket_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Websocket closed: {close_msg}")
        telegram_utils.send_and_exit("Websocket error, restarting the application...")

    def get_exchange(self):
        key_file = os.environ.get("HTB_KEY_FILE")
        if key_file is not None and os.path.isfile(key_file):
            with open(key_file, 'r') as file:
                file_content = file.read()
                account: LocalAccount = eth_account.Account.from_key(file_content)
                return Exchange(account, constants.MAINNET_API_URL, vault_address=os.environ.get("HTB_USER_VAULT"), account_address=os.environ["HTB_USER_WALLET"])

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


    def get_leverage(self, user_state, selected_coin) -> int | None:
        position = self._get_asset_position(user_state, selected_coin)
        if position:
            return int(position['leverage']['value'])
        return None


    def get_coins_with_open_positions(self):
        user_state = self.info.user_state(self.address)
        return [asset_position['position']['coin'] for asset_position in user_state.get("assetPositions", [])]

    def get_coins_by_open_interest(self) -> List[str]:
        response_data = self.info.meta_and_asset_ctxs()
        universe, coin_data = response_data[0]['universe'], response_data[1]

        coins = [(u["name"], float(c["dayNtlVlm"])) for u, c in zip(universe, coin_data)]
        sorted_coins = sorted(coins, key=lambda x: x[1], reverse=True)
        return [coin[0] for coin in reversed(sorted_coins[:75])]

    def get_coins_reply_markup(self):
        coins = self.get_coins_by_open_interest()
        keyboard = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in coins]
        keyboard.append([InlineKeyboardButton("Cancel", callback_data='cancel')])
        return InlineKeyboardMarkup(keyboard)

    def get_hyperliquid_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Hyperliquid format if needed."""
        symbol_mapping = {
            "SHIB": "kSHIB",
            "PEPE": "kPEPE",
            "FLOKI": "kFLOKI",
            "BONK": "kBONK"
        }
        return symbol_mapping.get(symbol, symbol)

    def fetch_cryptos(self, params: Dict[str, Any], page_count: int = 1) -> List[Dict]:
        """Fetch crypto data from CoinGecko API with configurable pagination."""
        all_cryptos = []
        try:
            for page in range(1, page_count + 1):
                params["page"] = page
                response = requests.get(self.COINGECKO_URL, params=params)
                response.raise_for_status()
                cryptos = response.json()
                all_cryptos.extend(cryptos)

            for crypto in all_cryptos:
                crypto["symbol"] = self.get_hyperliquid_symbol(crypto["symbol"].upper())
            return all_cryptos
        except requests.RequestException as e:
            logger.error(f"Error fetching crypto data: {e}")
            return []

hyperliquid_utils = HyperliquidUtils()
