import requests
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from typing import List, Dict, Optional, Any
from hyperliquid.info import Info
from hyperliquid_utils import hyperliquid_utils
from telegram.constants import ParseMode
from telegram_utils import telegram_utils
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from utils import exchange_enabled, fmt


class EtfStrategy:
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    LIMIT_PERCENTAGE = 0.25
    EXCLUDED_SYMBOLS = {"STETH", "WSTETH", "WBTC", "WETH", "WEETH", "LEO", "USDS"}
    LEVERAGE = 5.0
    MIN_USDC_BALANCE = 2.0
    USDC_BALANCE_PERCENT = 0.01

    def fetch_cryptos(self, url: str, params: Dict) -> List[Dict]:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching crypto data: {e}")
            return []

    def filter_top_cryptos(
        self,
        cryptos: List[Dict],
        market_cap_max_limit: int,
        coins_to_include: int,
        minimum_price_change: float,
    ) -> List[Dict]:
        filtered_cryptos = [
            {
                "name": coin["name"],
                "symbol": coin["symbol"].upper(),
                "market_cap": coin["market_cap"],
                "price_change_percentage_1y_in_currency": coin[
                    "price_change_percentage_1y_in_currency"
                ],
            }
            for coin in cryptos
            if (
                coin["symbol"].upper() not in self.EXCLUDED_SYMBOLS
                and coin["market_cap"] / 1_000_000_000 <= market_cap_max_limit
                and (
                    coin["price_change_percentage_1y_in_currency"] is None
                    or coin["price_change_percentage_1y_in_currency"]
                    > minimum_price_change
                )
                and (
                    abs(coin["price_change_percentage_24h_in_currency"])
                    > self.LIMIT_PERCENTAGE
                    or abs(coin["price_change_percentage_30d_in_currency"])
                    > self.LIMIT_PERCENTAGE
                )
            )
        ]
        return sorted(
            filtered_cryptos,
            key=lambda x: x["market_cap"],
            reverse=True,
        )[:coins_to_include]

    def calculate_account_values(self, user_state: Dict) -> tuple[dict[Any, float], float, float]:
        position_values = {
            pos["position"]["coin"].lstrip("k"): float(pos["position"]["positionValue"])
            for pos in user_state["assetPositions"]
        }
        usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
        total_account_value = sum(position_values.values()) + usdc_balance * self.LEVERAGE
        usdc_target_balance = max(
            total_account_value * self.USDC_BALANCE_PERCENT / self.LEVERAGE,
            self.MIN_USDC_BALANCE,
        )
        return position_values, total_account_value, usdc_target_balance

    def generate_table_data(
        self,
        top_cryptos: List[Dict],
        position_values: Dict[str, float],
        tradeable_balance: float,
        total_market_cap: float,
        cryptos: List[Dict],
        usdc_balance: float,
        usdc_target_balance: float,
    ) -> List[List]:
        table_data = []
        top_crypto_symbols = {coin["symbol"] for coin in top_cryptos}

        for idx, coin in enumerate(top_cryptos, start=1):
            symbol = coin["symbol"]
            price_change_1y = coin.get("price_change_percentage_1y_in_currency")
            price_change_1y_str = (
                f"{price_change_1y:.2f}" if price_change_1y is not None else "-"
            )
            market_cap_billion = coin["market_cap"] / 1_000_000_000
            rel_market_cap_pct = (coin["market_cap"] / total_market_cap) * 100
            target_value = (rel_market_cap_pct / 100) * tradeable_balance
            position_value = position_values.get(symbol, 0.0)
            curr_value_pct = (position_value / tradeable_balance) * 100
            difference = target_value - position_value

            table_data.extend(
                [
                    [idx, coin["name"], symbol],
                    ["", "1Y", f"{price_change_1y_str}%"],
                    [
                        "",
                        "Market cap",
                        f"{format(market_cap_billion, ',.0f')}B",
                        f"{fmt(rel_market_cap_pct)}%"
                    ],
                    ["", "Target value", f"{fmt(target_value)}$"],
                    [
                        "",
                        "Current value",
                        f"{fmt(position_value)}$",
                        f"{fmt(curr_value_pct)}%",
                    ],
                    ["", "USDC to buy", f"{fmt(difference)}$"],
                    ["", "", ""],
                ]
            )

        other_coins = [
            coin for coin in position_values.keys() if coin.upper() not in top_crypto_symbols
        ]
        full_cryptos = {coin["symbol"].upper(): coin for coin in cryptos}

        for symbol in other_coins:
            position_value = position_values[symbol]
            coin_data = full_cryptos.get(symbol.upper())

            if coin_data:
                market_cap = coin_data["market_cap"]
                market_cap_billion = market_cap / 1_000_000_000
                price_change_1y = coin_data.get(
                    "price_change_percentage_1y_in_currency"
                )
                price_change_1y_str = (
                    f"{price_change_1y:.2f}" if price_change_1y is not None else "-"
                )

                table_data.extend(
                    [
                        ["", coin_data["name"], symbol],
                        ["", "1Y", f"{price_change_1y_str}%"],
                        ["", "Market cap", f"{format(market_cap_billion, ',.0f')}B"],
                        ["", "Current value", f"{fmt(position_value)}$"],
                    ]
                )
            else:
                table_data.extend(
                    [
                        ["", "", symbol],
                        ["", "Current value", f"{fmt(position_value)}$"],
                    ]
                )

        table_data.extend(
            [
                ["", "USDC", ""],
                ["", "Target value", f"{fmt(usdc_target_balance)}$"],
                ["", "Current value", f"{fmt(usdc_balance)}$"],
                ["", "USDC to buy", f"{fmt(usdc_target_balance - usdc_balance)}$"],
            ]
        )

        return table_data

    async def display_crypto_info(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_address: str,
        cryptos: List[Dict],
        market_cap_max_limit: int,
        coins_to_include: int,
        minimum_price_change: float,
    ) -> None:
        try:
            top_cryptos = self.filter_top_cryptos(
                cryptos, market_cap_max_limit, coins_to_include, minimum_price_change
            )
            user_state = hyperliquid_utils.info.user_state(user_address)
            position_values, total_account_value, usdc_target_balance = self.calculate_account_values(
                user_state
            )

            total_market_cap = sum(coin["market_cap"] for coin in top_cryptos)
            usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
            tradeable_balance = total_account_value - usdc_target_balance

            table_data = self.generate_table_data(
                top_cryptos,
                position_values,
                tradeable_balance,
                total_market_cap,
                cryptos,
                usdc_balance,
                usdc_target_balance,
            )

            table = tabulate(
                table_data,
                headers=["", "", "", ""],
                tablefmt=simple_separated_format(" "),
                colalign=("right", "right", "right"),
            )

            message = [
                f"Leveraged account value: {total_account_value:.2f} USDC",
                f"<pre>{table}</pre>",
            ]

            await context.bot.send_message(
                chat_id=telegram_utils.telegram_chat_id,
                text="\n".join(message),
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error(f"Error displaying crypto info: {str(e)}")

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE) -> Optional[bool]:





        try:
            cryptos = self.fetch_cryptos(
                self.COINGECKO_URL,
                {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 50,
                    "page": 1,
                    "sparkline": "false",
                    "price_change_percentage": "24h,30d,1y",
                },
            )
            await self.display_crypto_info(
                context, hyperliquid_utils.address, cryptos, 10000, 5, 15.0
            )
            return True
        except Exception as e:
            logger.error(f"Error executing ETF strategy: {str(e)}")
            return False
