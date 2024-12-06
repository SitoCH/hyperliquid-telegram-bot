from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, ClassVar
from dataclasses import dataclass
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from telegram.constants import ParseMode
from tabulate import simple_separated_format, tabulate
from hyperliquid_utils import hyperliquid_utils
from telegram_utils import telegram_utils
from hyperliquid_trade import exit_all_positions
from logging_utils import logger
from utils import fmt


@dataclass
class BaseStrategyConfig:
    leverage: int


class BaseStrategy(ABC):
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    MIN_USDC_BALANCE: ClassVar[float] = 2.0
    USDC_BALANCE_PERCENT: ClassVar[float] = 0.01
    _config: BaseStrategyConfig

    @property
    def config(self) -> BaseStrategyConfig:
        return self._config

    def get_hyperliquid_symbol(self, symbol: str) -> str:
        symbol_mapping = {
            "SHIB": "kSHIB",
            "PEPE": "kPEPE",
            "FLOKI": "kFLOKI",
            "BONK": "kBONK"
        }
        return symbol_mapping.get(symbol, symbol)

    def calculate_account_values(
        self, user_state: Dict, leverage: int
    ) -> tuple[dict[Any, float], float, float]:
        position_values = {
            pos["position"]["coin"]: float(pos["position"]["positionValue"])
            for pos in user_state["assetPositions"]
        }
        usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
        total_account_value = sum(position_values.values()) + usdc_balance * leverage
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
            if symbol not in top_crypto_symbols:
                coin_data = next((c for c in cryptos if c["symbol"] == symbol), None)
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
            curr_value_pct = (coin["current_value"] / tradeable_balance) * 100 if tradeable_balance > 0 else 0
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

    async def display_crypto_info(
        self,
        update: Update,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> None:
        try:
            top_cryptos = self.filter_top_cryptos(cryptos, all_mids, meta)
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            position_values, total_account_value, usdc_target_balance = (
                self.calculate_account_values(user_state, self.config.leverage)
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
            cryptos, all_mids, meta = self.get_strategy_params()
            await self.display_crypto_info(update, cryptos, all_mids, meta)
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")

    async def rebalance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await telegram_utils.reply(update, "Closing all current positions...")
            
            exchange = hyperliquid_utils.get_exchange()
            await exit_all_positions(update, context)
            
            await telegram_utils.reply(update, "Opening new positions based on current market data...")
            
            cryptos, all_mids, meta = self.get_strategy_params()
            
            top_cryptos = self.filter_top_cryptos(cryptos, all_mids, meta)
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            position_values, total_account_value, usdc_target_balance = (
                self.calculate_account_values(user_state, self.config.leverage)
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

                    exchange.update_leverage(self.config.leverage, symbol, False)
                    mid = float(all_mids[symbol])
                    sz_decimals = hyperliquid_utils.get_sz_decimals()
                    sz = round(difference / mid, sz_decimals[symbol])
                    logger.info(f"Need to buy {fmt(difference)} USDC worth of {symbol}: {sz} units")
                    open_result = exchange.market_open(symbol, True, sz)
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

    @abstractmethod
    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        pass

    @abstractmethod
    def filter_top_cryptos(
        self,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> List[Dict]:
        pass
