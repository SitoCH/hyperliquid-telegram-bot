import requests
import os
from logging_utils import logger
from tabulate import simple_separated_format, tabulate
from typing import List, Dict, Optional, Any, Tuple
from hyperliquid.info import Info
from hyperliquid_utils import hyperliquid_utils
from telegram.constants import ParseMode
from telegram_utils import telegram_utils
from hyperliquid_trade import exit_all_positions
from telegram.ext import (
    CallbackContext,
    ContextTypes,
    ConversationHandler,
    CommandHandler,
)
from utils import exchange_enabled, fmt
from telegram import Update


class EtfStrategy:
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    LIMIT_PERCENTAGE = 0.25
    EXCLUDED_SYMBOLS = {"STETH", "WSTETH", "WBTC", "WETH", "WEETH", "LEO", "USDS"}
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

    def calculate_account_values(
        self, user_state: Dict, leverage: int
    ) -> tuple[dict[Any, float], float, float]:
        position_values = {
            pos["position"]["coin"].lstrip("k"): float(pos["position"]["positionValue"])
            for pos in user_state["assetPositions"]
        }
        usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
        total_account_value = (
            sum(position_values.values()) + usdc_balance * leverage
        )
        usdc_target_balance = max(
            total_account_value * self.USDC_BALANCE_PERCENT / leverage,
            self.MIN_USDC_BALANCE,
        )
        return position_values, total_account_value, usdc_target_balance

    def calculate_allocations(
        self,
        top_cryptos: List[Dict],
        position_values: Dict[str, float],
        tradeable_balance: float,
        total_market_cap: float,
        cryptos: List[Dict],
        usdc_balance: float,
        usdc_target_balance: float,
    ) -> Tuple[List[Dict], List[Dict], float]:
        top_crypto_symbols = {coin["symbol"] for coin in top_cryptos}
        allocation_data = []
        
        for coin in top_cryptos:
            symbol = coin["symbol"]
            rel_market_cap_pct = (coin["market_cap"] / total_market_cap) * 100
            target_value = (rel_market_cap_pct / 100) * tradeable_balance
            position_value = position_values.get(symbol, 0.0)
            difference = target_value - position_value
            
            allocation_data.append({
                "name": coin["name"],
                "symbol": symbol,
                "target_value": target_value,
                "current_value": position_value,
                "difference": difference,
                "market_cap": coin["market_cap"],
                "price_change_1y": coin.get("price_change_percentage_1y_in_currency")
            })

        other_positions = []
        for symbol, value in position_values.items():
            if symbol.upper() not in top_crypto_symbols:
                coin_data = next((c for c in cryptos if c["symbol"].upper() == symbol.upper()), None)
                other_positions.append({
                    "symbol": symbol,
                    "name": coin_data["name"] if coin_data else symbol,
                    "current_value": value,
                    "market_cap": coin_data["market_cap"] if coin_data else None,
                    "price_change_1y": coin_data.get("price_change_percentage_1y_in_currency") if coin_data else None
                })

        usdc_difference = usdc_target_balance - usdc_balance
        
        return allocation_data, other_positions, usdc_difference

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
        allocation_data, other_positions, usdc_difference = self.calculate_allocations(
            top_cryptos, position_values, tradeable_balance, total_market_cap,
            cryptos, usdc_balance, usdc_target_balance
        )
        
        table_data = []
        
        for idx, coin in enumerate(allocation_data, start=1):
            price_change_1y_str = (
                f"{coin['price_change_1y']:.2f}" if coin['price_change_1y'] is not None else "-"
            )
            market_cap_billion = coin["market_cap"] / 1_000_000_000
            curr_value_pct = (coin["current_value"] / tradeable_balance) * 100
            rel_market_cap_pct = (coin["market_cap"] / total_market_cap) * 100

            table_data.extend([
                [idx, coin["name"], coin["symbol"]],
                ["", "1Y", f"{price_change_1y_str}%"],
                [
                    "",
                    "Market cap",
                    f"{format(market_cap_billion, ',.0f')}B",
                    f"{fmt(rel_market_cap_pct)}%",
                ],
                ["", "Target value", f"{fmt(coin['target_value'])}$"],
                [
                    "",
                    "Current value",
                    f"{fmt(coin['current_value'])}$",
                    f"{fmt(curr_value_pct)}%",
                ],
                ["", "USDC to buy", f"{fmt(coin['difference'])}$"],
                ["", "", ""],
            ])

        for pos in other_positions:
            if pos.get("market_cap"):
                market_cap_billion = pos["market_cap"] / 1_000_000_000
                price_change_1y_str = (
                    f"{pos['price_change_1y']:.2f}" if pos['price_change_1y'] is not None else "-"
                )
                
                table_data.extend([
                    ["", pos["name"], pos["symbol"]],
                    ["", "1Y", f"{price_change_1y_str}%"],
                    ["", "Market cap", f"{format(market_cap_billion, ',.0f')}B"],
                    ["", "Current value", f"{fmt(pos['current_value'])}$"],
                    ["", "", ""],
                ])
            else:
                table_data.extend([
                    ["", "", pos["symbol"]],
                    ["", "Current value", f"{fmt(pos['current_value'])}$"],
                    ["", "", ""],
                ])

        table_data.extend([
            ["", "USDC", ""],
            ["", "Target value", f"{fmt(usdc_target_balance)}$"],
            ["", "Current value", f"{fmt(usdc_balance)}$"],
            ["", "USDC to buy", f"{fmt(usdc_difference)}$"],
        ])

        return table_data

    def get_strategy_data(self) -> Tuple[List[Dict], int, int, float, int]:
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
        coins_number = int(os.getenv("HTB_ETF_STRATEGY_COINS_NUMBER", "5"))
        leverage = int(os.getenv("HTB_ETF_STRATEGY_LEVERAGE", "5"))
        min_yearly_performance = int(
            os.getenv("HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE", "15.0")
        )
        max_market_cap = int(os.getenv("HTB_ETF_STRATEGY_MAX_MARKET_CAP", "10000"))
        
        return cryptos, max_market_cap, coins_number, min_yearly_performance, leverage

    async def display_crypto_info(
        self,
        update: Update,
        user_address: str,
        cryptos: List[Dict],
        market_cap_max_limit: int,
        coins_to_include: int,
        minimum_price_change: float,
        leverage: int
    ) -> None:
        try:
            top_cryptos = self.filter_top_cryptos(
                cryptos, market_cap_max_limit, coins_to_include, minimum_price_change
            )
            user_state = hyperliquid_utils.info.user_state(user_address)
            position_values, total_account_value, usdc_target_balance = (
                self.calculate_account_values(user_state, leverage)
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

            await telegram_utils.reply(
                update, "\n".join(message), parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Error displaying crypto info: {str(e)}")

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            cryptos, max_market_cap, coins_number, min_yearly_performance, leverage = self.get_strategy_data()
            await self.display_crypto_info(
                update,
                hyperliquid_utils.address,
                cryptos,
                max_market_cap,
                coins_number,
                min_yearly_performance,
                leverage
            )
        except Exception as e:
            logger.error(f"Error executing ETF strategy: {str(e)}")


    def get_hyperliquid_symbol(self, symbol: str):
        if symbol == "SHIB":
            return "kSHIB"
        if symbol == "PEPE":
            return "kPEPE"
        return symbol

    async def rebalance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await telegram_utils.reply(update, "Closing all current positions...")
            
            exchange = hyperliquid_utils.get_exchange()
            await exit_all_positions(update, context)
            
            await telegram_utils.reply(update, "Opening new positions based on current market data...")
            
            cryptos, max_market_cap, coins_number, min_yearly_performance, leverage = self.get_strategy_data()
            
            top_cryptos = self.filter_top_cryptos(
                cryptos, max_market_cap, coins_number, min_yearly_performance
            )
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            position_values, total_account_value, usdc_target_balance = (
                self.calculate_account_values(user_state, leverage)
            )

            total_market_cap = sum(coin["market_cap"] for coin in top_cryptos)
            usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
            tradeable_balance = total_account_value - usdc_target_balance

            allocation_data, other_positions, usdc_difference = self.calculate_allocations(
                top_cryptos,
                position_values,
                tradeable_balance,
                total_market_cap,
                cryptos,
                usdc_balance,
                usdc_target_balance,
            )
                    
            for coin in allocation_data:
                try:
                    symbol = coin["symbol"]
                    difference = coin["difference"]
                    if difference < 10.0:
                        logger.info(f"The order value for {symbol} is less than 10 USDC ({fmt(difference)} USDC) and can't be executed")
                        continue

                    hyperliquid_symbol = self.get_hyperliquid_symbol(symbol) 
                    exchange.update_leverage(leverage, hyperliquid_symbol, False)
                    mid = float(hyperliquid_utils.info.all_mids()[hyperliquid_symbol])
                    sz_decimals = hyperliquid_utils.get_sz_decimals()
                    sz = round(difference / mid, sz_decimals[hyperliquid_symbol])
                    logger.info(f"Need to buy {fmt(difference)} USDC worth of {hyperliquid_symbol}: {sz} units")
                    open_result = exchange.market_open(hyperliquid_symbol, True, sz)
                    logger.info(open_result)
                except Exception as e:
                    logger.error(f"Unable to open position for {symbol}: {str(e)}")
                    continue
            
            await telegram_utils.reply(update, "Rebalancing completed")
        except Exception as e:
            logger.critical(e, exc_info=True)
            await telegram_utils.reply(
                update, f"Error during rebalancing: {str(e)}", parse_mode=ParseMode.HTML
            )

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE):
        rebalance_button_text = "etf_rebalance"
        telegram_utils.add_buttons([f"/{rebalance_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(rebalance_button_text, self.rebalance))

        analyze_button_text = "etf_analyze"
        telegram_utils.add_buttons([f"/{analyze_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))
