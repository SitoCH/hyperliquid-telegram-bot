import os
import requests
from typing import List, Dict, Any, ClassVar, Optional, Tuple, Callable

import eth_account
from eth_account.signers.local import LocalAccount
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from logging_utils import logger

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from telegram_utils import telegram_utils
from .hyperliquid_info_proxy import InfoProxy


class HyperliquidUtils:

    COINGECKO_URL: ClassVar[str] = "https://api.coingecko.com/api/v3/coins/markets"

    def __init__(self) -> None:

        user_wallet: Optional[str] = os.environ.get("HTB_USER_WALLET")
        if user_wallet is None:
            logger.error("HTB_USER_WALLET environment variable is not set")
            raise ValueError("HTB_USER_WALLET environment variable is required")

        user_vault: Optional[str] = os.environ.get("HTB_USER_VAULT")
        self.address: str = user_vault if user_vault is not None else user_wallet

        self._info: Optional[InfoProxy] = None
        self._exchange_cache: dict[str, Exchange] = {}
        self._extra_dexes: list[str] = [
            s.strip()
            for s in os.environ.get("HTB_EXTRA_DEXES", "xyz").split(",")
            if s.strip()
        ]
        self._ws_subscriptions: list[tuple[dict[str, Any], Any]] = []
        self._reconnecting: bool = False

    @property
    def info(self) -> InfoProxy:
        """Lazy initialization of InfoProxy to avoid network calls during module import."""
        if self._info is None:
            self._info = InfoProxy(Info(constants.MAINNET_API_URL, True))
        return self._info

    @info.setter
    def info(self, value: InfoProxy) -> None:
        self._info = value

    @info.deleter
    def info(self) -> None:
        self._info = None

    def init_websocket(self) -> None:
        """Initialize WebSocket connection for live data."""
        self._init_websocket_inner()

    def _init_websocket_inner(self) -> None:
        """Create a fresh WebSocket connection and attach handlers."""
        self.info = InfoProxy(Info(constants.MAINNET_API_URL, False))
        if hasattr(self.info._info, 'ws_manager') and self.info._info.ws_manager:
            self.info._info.ws_manager.ws.on_error = self._on_websocket_error
            self.info._info.ws_manager.ws.on_close = self._on_websocket_close
        # Re-subscribe any stored subscriptions on the new connection
        for subscription, callback in self._ws_subscriptions:
            try:
                self.info.subscribe(subscription, callback)
            except Exception as e:
                logger.error(f"Failed to re-subscribe {subscription}: {e}")

    def subscribe_websocket(self, subscription: dict[str, Any], callback: Any) -> None:
        """Subscribe to a websocket event and store it for reconnection."""
        self._ws_subscriptions.append((subscription, callback))
        try:
            self.info.subscribe(subscription, callback)
        except Exception as e:
            logger.error(f"Failed to subscribe {subscription}: {e}")

    def _on_websocket_error(self, ws: Any, error: Any) -> None:
        logger.error(f"Websocket error: {error}")
        self._notify_and_reconnect("⚠️", "WebSocket connection error")

    def _on_websocket_close(self, ws: Any, close_status_code: int, close_msg: str) -> None:
        logger.warning(f"Websocket closed: {close_msg}")
        self._notify_and_reconnect("🔌", "WebSocket disconnected")

    def _notify_and_reconnect(self, icon: str, reason: str) -> None:
        """Notify user and schedule reconnect, debounced to prevent duplicates."""
        if self._reconnecting:
            logger.info(f"Ignoring {reason} — reconnection already in progress")
            return
        self._reconnecting = True
        telegram_utils.queue_send(f"{icon} {reason} — reconnecting...")
        if not telegram_utils.telegram_app or not telegram_utils.telegram_app.job_queue:
            logger.warning("Telegram app not ready, reconnecting immediately")
            self._do_reconnect()
            return
        telegram_utils.telegram_app.job_queue.run_once(
            self._do_reconnect_job,
            when=5,
            job_kwargs={'misfire_grace_time': 60},
        )

    async def _do_reconnect_job(self, context: Any) -> None:
        """Job callback that performs the actual reconnect."""
        self._do_reconnect()

    def _do_reconnect(self) -> None:
        """Perform the actual WebSocket reconnection."""
        try:
            # Disconnect old websocket if still alive
            try:
                if self._info is not None and hasattr(self._info._info, 'disconnect_websocket'):
                    self._info._info.disconnect_websocket()
            except Exception:
                pass

            # Clear the old info to force fresh creation
            self._info = None

            # Create fresh connection with re-subscription
            self._init_websocket_inner()
            self._reconnecting = False
            logger.info("WebSocket reconnected successfully")
            telegram_utils.queue_send("✅ WebSocket reconnected")
        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}", exc_info=True)
            self._reconnecting = False
            telegram_utils.queue_send(f"❌ WebSocket reconnect failed: {e}")

    def get_exchange(self, dex: str = "") -> Optional[Exchange]:
        """Get or create a cached Exchange for the given perp DEX.

        Args:
            dex: The perp DEX name (e.g. '' for default, 'xyz', 'flx').
                 Exchanges are cached per dex name.

        Returns:
            Exchange instance or None if HTB_KEY_FILE is not configured.
        """
        if dex in self._exchange_cache:
            return self._exchange_cache[dex]

        key_file: Optional[str] = os.environ.get("HTB_KEY_FILE")
        if key_file is not None and os.path.isfile(key_file):
            with open(key_file, 'r') as file:
                file_content: str = file.read()
                ascii_only: str = file_content.encode("ascii", "ignore").decode("ascii").strip()
                account: LocalAccount = eth_account.Account.from_key(ascii_only)
                perp_dexs = [dex] if dex else None  # None = default DEX only
                exchange = Exchange(
                    account, constants.MAINNET_API_URL,
                    vault_address=os.environ.get("HTB_USER_VAULT"),
                    account_address=os.environ["HTB_USER_WALLET"],
                    perp_dexs=perp_dexs,
                )
                self._exchange_cache[dex] = exchange
                return exchange
        return None

    @staticmethod
    def dex_supported(coin: str) -> Optional[str]:
        """Detect the perp DEX a coin belongs to based on its prefix.

        Coins from builder-deployed perp DEXes use the ``{dex_name}:`` prefix
        (e.g. ``xyz:AAPL``, ``xyz:SPCX``).  Default DEX coins have no prefix.

        Returns:
            The dex name (e.g. 'xyz', 'flx') or None for the default DEX.
        """
        if ":" in coin:
            parts = coin.split(":", 1)
            if parts[0] and parts[1]:
                return parts[0]
        return None

    @staticmethod
    def strip_dex_prefix(coin: str) -> str:
        """Strip the ``{dex_name}:`` prefix from a coin name."""
        if ":" in coin:
            return coin.split(":", 1)[1]
        return coin

    def get_isolated_only(self, coin: str) -> bool:
        """Check if a coin is isolated-only (can't use cross margin).

        Queries the meta for the relevant DEX to check the onlyIsolated flag.
        """
        dex = self.dex_supported(coin) or ""
        meta = self.info.meta(dex=dex)
        for asset_info in meta.get("universe", []):
            if asset_info["name"] == coin:
                return bool(asset_info.get("onlyIsolated", False))
        return False

    def get_sz_decimals(self) -> Dict[str, int]:
        meta: Dict[str, Any] = self.info.meta()
        sz_decimals: Dict[str, int] = {}
        for asset_info in meta["universe"]:
            sz_decimals[asset_info["name"]] = asset_info["szDecimals"]
        return sz_decimals

    def _get_asset_position(self, user_state: Dict[str, Any], selected_coin: str) -> Optional[Dict[str, Any]]:
        return next(
            (asset_position['position'] for asset_position in user_state.get("assetPositions", [])
             if asset_position['position']['coin'] == selected_coin),
            None
        )

    def get_size(self, user_state: Dict[str, Any], selected_coin: str) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['szi']) if position else 0.0

    def get_entry_px_str(self, user_state: Dict[str, Any], selected_coin: str) -> Optional[str]:
        position = self._get_asset_position(user_state, selected_coin)
        return position['entryPx'] if position else None

    def get_liquidation_px_str(self, user_state: Dict[str, Any], selected_coin: str) -> Optional[str]:
        position = self._get_asset_position(user_state, selected_coin)
        return position['liquidationPx'] if position else None

    def get_entry_px(self, user_state: Dict[str, Any], selected_coin: str) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['entryPx']) if position else 0.0

    def get_unrealized_pnl(self, user_state: Dict[str, Any], selected_coin: str) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['unrealizedPnl']) if position else 0.0

    def get_return_on_equity(self, user_state: Dict[str, Any], selected_coin: str) -> float:
        position = self._get_asset_position(user_state, selected_coin)
        return float(position['returnOnEquity']) if position else 0.0

    def get_leverage(self, user_state: Dict[str, Any], selected_coin: str) -> Optional[int]:
        position = self._get_asset_position(user_state, selected_coin)
        if position:
            return int(position['leverage']['value'])
        return None

    def get_coins_with_open_positions(self) -> List[str]:
        """Get coins with open positions across all DEXes."""
        coins: list[str] = []
        dexes: list[str] = [""] + self._extra_dexes
        for dex in dexes:
            user_state = self.info.user_state(self.address, dex=dex)
            coins.extend(
                ap['position']['coin']
                for ap in user_state.get("assetPositions", [])
            )
        return coins

    def get_coins_by_traded_volume(self) -> List[str]:
        """Get coins available for trading across all DEXes, sorted by volume."""
        # Default DEX coins
        response_data: Any = self.info.meta_and_asset_ctxs()
        universe: List[Dict[str, Any]] = response_data[0]['universe']
        coin_data: List[Dict[str, Any]] = response_data[1]
        coins: list[tuple[str, float]] = [(u["name"], float(c["dayNtlVlm"])) for u, c in zip(universe, coin_data)]

        # Extra DEX coins
        for dex in self._extra_dexes:
            try:
                dex_ctxs = self.info.meta_and_asset_ctxs(dex=dex)
                dex_universe = dex_ctxs[0]['universe']
                dex_coin_data = dex_ctxs[1]
                coins.extend(
                    (u["name"], float(c.get("dayNtlVlm", 0)))
                    for u, c in zip(dex_universe, dex_coin_data)
                )
            except Exception as e:
                logger.warning(f"Failed to fetch coins for DEX '{dex}': {e}")

        sorted_coins = sorted(coins, key=lambda x: x[1], reverse=True)
        return [coin[0] for coin in reversed(sorted_coins[:75])]

    def extra_dexes(self) -> List[str]:
        """Return the list of extra perp DEX names to query (e.g. ['xyz'])."""
        return list(self._extra_dexes)

    def get_coins_reply_markup(self) -> InlineKeyboardMarkup:
        coins: List[str] = self.get_coins_by_traded_volume()
        open_position_coins: set[str] = set(self.get_coins_with_open_positions())
        prioritized: List[str] = [c for c in coins if c in open_position_coins]
        others: List[str] = [c for c in coins if c not in open_position_coins]
        ordered_coins: List[str] = others + prioritized

        keyboard: List[List[InlineKeyboardButton]] = [[InlineKeyboardButton(coin, callback_data=coin)] for coin in ordered_coins]
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

    def fetch_cryptos(self, params: Dict[str, Any], page_count: int = 1) -> List[Dict[str, Any]]:
        """Fetch crypto data from CoinGecko API with configurable pagination."""
        all_cryptos: List[Dict[str, Any]] = []
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
