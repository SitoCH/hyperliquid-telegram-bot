import os
import datetime
import requests
import random
from dataclasses import dataclass
from telegram.ext import ContextTypes, CommandHandler
from typing import List, Dict, Set, Optional, Tuple
from strategies.base_strategy.base_strategy import BaseStrategy, BaseStrategyConfig
from logging_utils import logger
from hyperliquid_utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt


@dataclass
class EtfConfig:
    coins_number: int
    coins_offset: int
    min_yearly_performance: float
    excluded_symbols: Set[str]
    category: Optional[str]


class EtfStrategy(BaseStrategy):

    def __init__(self):
        leverage = int(os.getenv("HTB_ETF_STRATEGY_LEVERAGE", "5"))
        self._config = BaseStrategyConfig(leverage=leverage)
        self._etf_config = EtfConfig(
            coins_number=int(os.getenv("HTB_ETF_STRATEGY_COINS_NUMBER", "5")),
            coins_offset=int(os.getenv("HTB_ETF_STRATEGY_COINS_OFFSET", "0")),
            min_yearly_performance=float(os.getenv("HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE", "15.0")),
            excluded_symbols=set(os.getenv("HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS", "").split(",")),
            category=os.getenv("HTB_ETF_STRATEGY_CATEGORY")
        )

    def fetch_cryptos(self, url: str, params: Dict) -> List[Dict]:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            cryptos = response.json()
            for crypto in cryptos:
                crypto["symbol"] = self.get_hyperliquid_symbol(crypto["symbol"].upper())
            return cryptos
        except requests.RequestException as e:
            logger.error(f"Error fetching crypto data: {e}")
            return []

    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,30d,1y",
        }
        
        if self._etf_config.category:
            params["category"] = self._etf_config.category
        
        cryptos = self.fetch_cryptos(self.COINGECKO_URL, params)
        all_mids = hyperliquid_utils.info.all_mids()
        meta = hyperliquid_utils.info.meta()
        
        return cryptos, all_mids, meta

    def filter_top_cryptos(
        self,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> List[Dict]:
        filtered_cryptos = []

        asset_info_map = {
            info["name"]: int(info["maxLeverage"])
            for info in meta.get("universe", [])
        }
        
        for coin in cryptos:
            symbol = coin["symbol"]
            yearly_change = coin["price_change_percentage_1y_in_currency"]

            if symbol in self._etf_config.excluded_symbols:
                logger.info(f"Excluding {symbol}: in HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS")
                continue
                
            if symbol not in all_mids:
                logger.info(f"Excluding {symbol}: not available on Hyperliquid")
                continue
                
            if yearly_change is not None and yearly_change <= self._etf_config.min_yearly_performance:
                logger.info(f"Excluding {symbol}: yearly change {fmt(yearly_change)}% <= {self._etf_config.min_yearly_performance}%")
                continue

            max_leverage = asset_info_map.get(symbol)
            if max_leverage is not None and self._config.leverage > max_leverage:
                logger.info(f"Excluding {symbol}: strategy leverage {self._config.leverage} exceeds max allowed leverage {max_leverage}")
                continue

            filtered_cryptos.append({
                "name": coin["name"],
                "symbol": symbol,
                "market_cap": coin["market_cap"],
                "price_change_percentage_1y_in_currency": yearly_change,
            })

        sorted_cryptos = sorted(
            filtered_cryptos,
            key=lambda x: x["market_cap"],
            reverse=True,
        )
        
        if self._etf_config.coins_offset > 0:
            for coin in sorted_cryptos[:self._etf_config.coins_offset]:
                logger.info(f"Skipping {coin['symbol']} due to coins_offset={self._etf_config.coins_offset}")
        
        # Apply offset and limit
        return sorted_cryptos[self._etf_config.coins_offset:self._etf_config.coins_offset + self._etf_config.coins_number]

    async def init_strategy(self, context: ContextTypes.DEFAULT_TYPE):
        rebalance_button_text = "rebalance"
        telegram_utils.add_buttons([f"/{rebalance_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(rebalance_button_text, self.rebalance))

        analyze_button_text = "analyze"
        telegram_utils.add_buttons([f"/{analyze_button_text}"], 1)
        telegram_utils.add_handler(CommandHandler(analyze_button_text, self.analyze))

        telegram_utils.run_repeating(
            self.check_position_allocation_drifts,
            interval=datetime.timedelta(minutes=random.randint(50, 70))
        )
