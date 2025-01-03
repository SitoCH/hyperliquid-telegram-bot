from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, ClassVar, Optional
from dataclasses import dataclass, field
import requests
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
    """Configuration for base trading strategy.
    
    Attributes:
        leverage: Trading leverage to use (1-10)
        min_yearly_performance: Minimum acceptable yearly performance in percent
        min_position_size: Minimum position size in USDC
        min_usdc_balance: Minimum USDC balance to maintain
        usdc_balance_percent: Percentage of total account value to keep as USDC
        min_rebalance_difference: Minimum difference in USDC to trigger rebalance
        drift_check_threshold: Minimum unrealized PnL to check position drifts
        position_drift_threshold: Minimum position difference to trigger drift alert
    """
    leverage: int
    min_yearly_performance: float = 15.0
    min_position_size: float = 10.0
    min_usdc_balance: float = 2.0
    usdc_balance_percent: float = 0.01
    min_rebalance_difference: float = 25.0
    drift_check_threshold: float = 25.0
    position_drift_threshold: float = 25.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.leverage <= 10:
            raise ValueError("Leverage must be between 1 and 10")
        if self.min_yearly_performance < 0:
            raise ValueError("Minimum yearly performance cannot be negative")
        if self.min_position_size < 10.0:
            raise ValueError("Minimum position size cannot be less than 10 USDC")
        if self.min_usdc_balance < 0:
            raise ValueError("Minimum USDC balance cannot be negative")
        if not 0 < self.usdc_balance_percent < 1:
            raise ValueError("USDC balance percentage must be between 0 and 1")
    
    @classmethod
    def default(cls) -> 'BaseStrategyConfig':
        """Create a configuration with default values."""
        return cls(
            leverage=3,
            min_yearly_performance=15.0,
            min_position_size=10.0,
            min_usdc_balance=2.0,
            usdc_balance_percent=0.01,
            min_rebalance_difference=25.0,
            drift_check_threshold=25.0,
            position_drift_threshold=25.0
        )


@dataclass
class AllocationData:
    name: str
    symbol: str
    target_value: float
    current_value: float
    difference: float
    market_cap: float
    price_change_1y: Optional[float]


@dataclass
class PositionData:
    name: str
    symbol: str
    current_value: float
    market_cap: Optional[float] = None
    price_change_1y: Optional[float] = None


class BaseStrategy(ABC):
    """
    Base class for implementing trading strategies.
    Provides common functionality for analyzing and rebalancing positions.
    """
    _config: BaseStrategyConfig

    def __init__(self, config: Optional[BaseStrategyConfig] = None):
        self._config = config or BaseStrategyConfig.default()

    @property
    def config(self) -> BaseStrategyConfig:
        return self._config

    def calculate_account_values(
        self, user_state: Dict, leverage: int
    ) -> Tuple[Dict[str, float], float, float]:
        """Calculate various account values including positions and balances."""
        position_values = {
            pos["position"]["coin"]: float(pos["position"]["positionValue"])
            for pos in user_state["assetPositions"]
        }
        usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
        total_account_value = sum(position_values.values()) + usdc_balance * leverage
        usdc_target_balance = max(
            total_account_value * self.config.usdc_balance_percent / leverage,
            self.config.min_usdc_balance,
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
    ) -> Tuple[List[AllocationData], List[PositionData], float]:
        """Calculate allocation data for positions and USDC."""
        top_crypto_symbols = {coin["symbol"] for coin in top_cryptos}
        allocation_data = []
        
        initial_targets = {}
        valid_coins = []
        skipped_value = 0.0
        
        for coin in top_cryptos:
            symbol = coin["symbol"]
            rel_market_cap_pct = (coin["market_cap"] / total_market_cap) * 100
            target_value = (rel_market_cap_pct / 100) * tradeable_balance
            
            if target_value < self.config.min_position_size:
                logger.info(f"Skipping {symbol}: target value {fmt(target_value)} USDC is below minimum")
                skipped_value += target_value
                continue
                
            valid_coins.append(coin)
            initial_targets[symbol] = target_value
            
        if valid_coins:
            if skipped_value > 0:
                valid_total = sum(initial_targets.values())
                redistribution_ratio = (valid_total + skipped_value) / valid_total if valid_total > 0 else 0
            else:
                redistribution_ratio = 1.0

            for coin in valid_coins:
                symbol = coin["symbol"]
                position_value = position_values.get(symbol, 0.0)
                target_value = initial_targets[symbol] * redistribution_ratio
                
                allocation_data.append(AllocationData(
                    name=coin["name"],
                    symbol=symbol,
                    target_value=target_value,
                    current_value=position_value,
                    difference=target_value - position_value,
                    market_cap=coin["market_cap"],
                    price_change_1y=coin.get("price_change_percentage_1y_in_currency")
                ))

        other_positions = []
        for symbol, value in position_values.items():
            if symbol not in top_crypto_symbols:
                coin_data = next((c for c in cryptos if c["symbol"] == symbol), None)
                market_cap = coin_data["market_cap"] if coin_data else None
                price_change_1y = (
                    coin_data.get("price_change_percentage_1y_in_currency")
                    if coin_data else None
                )
                other_positions.append(PositionData(
                    name=coin_data["name"] if coin_data else symbol,
                    symbol=symbol,
                    current_value=value,
                    market_cap=market_cap,
                    price_change_1y=price_change_1y
                ))

        usdc_difference = usdc_target_balance - usdc_balance
        
        return allocation_data, other_positions, usdc_difference

    def generate_table_rows(
        self,
        allocation: AllocationData,
        tradeable_balance: float,
        total_market_cap: float,
        idx: Optional[int] = None,
    ) -> List[List[Any]]:
        """Generate table rows for a single allocation."""
        market_cap_billion = allocation.market_cap / 1_000_000_000
        curr_value_pct = (allocation.current_value / tradeable_balance) * 100 if tradeable_balance > 0 else 0
        rel_market_cap_pct = (allocation.market_cap / total_market_cap) * 100
        price_change_1y_str = f"{allocation.price_change_1y:.2f}" if allocation.price_change_1y is not None else "-"

        return [
            [idx if idx else "", allocation.name, allocation.symbol],
            ["", "1Y", f"{price_change_1y_str}%"],
            [
                "",
                "Market cap",
                f"{format(market_cap_billion, ',.0f')}B",
                f"{fmt(rel_market_cap_pct)}%",
            ],
            ["", "Target value", f"{fmt(allocation.target_value)}$"],
            [
                "",
                "Current value",
                f"{fmt(allocation.current_value)}$",
                f"{fmt(curr_value_pct)}%",
            ],
            ["", "USDC to buy", f"{fmt(allocation.difference)}$"],
            ["", "", ""],
        ]

    def generate_table_data(
        self,
        allocation_data: List[AllocationData],
        other_positions: List[PositionData],
        tradeable_balance: float,
        total_market_cap: float,
        usdc_balance: float,
        usdc_target_balance: float,
        usdc_difference: float,
    ) -> List[List[Any]]:
        """Generate complete table data for display."""
        table_data = []
        
        for idx, allocation in enumerate(allocation_data, start=1):
            table_data.extend(
                self.generate_table_rows(allocation, tradeable_balance, total_market_cap, idx)
            )

        for pos in other_positions:
            if pos.market_cap:
                market_cap_billion = pos.market_cap / 1_000_000_000
                price_change_1y_str = f"{pos.price_change_1y:.2f}" if pos.price_change_1y is not None else "-"
                
                table_data.extend([
                    ["", pos.name, pos.symbol],
                    ["", "1Y", f"{price_change_1y_str}%"],
                    ["", "Market cap", f"{format(market_cap_billion, ',.0f')}B"],
                    ["", "Current value", f"{fmt(pos.current_value)}$"],
                    ["", "", ""],
                ])
            else:
                table_data.extend([
                    ["", "", pos.symbol],
                    ["", "Current value", f"{fmt(pos.current_value)}$"],
                    ["", "", ""],
                ])

        table_data.extend([
            ["", "USDC", ""],
            ["", "Target value", f"{fmt(usdc_target_balance)}$"],
            ["", "Current value", f"{fmt(usdc_balance)}$"],
            ["", "USDC to buy", f"{fmt(usdc_difference)}$"],
        ])

        return table_data

    async def check_position_allocation_drifts(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Monitor and report significant position allocation drifts."""
        try:
            user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
            if user_state["assetPositions"]:
                unrealized_pnl = sum(
                    float(asset_position['position']['unrealizedPnl'])
                    for asset_position in user_state["assetPositions"]
                )

                if unrealized_pnl < self.config.drift_check_threshold:
                    return

            cryptos, all_mids, meta = self.get_strategy_params()
            top_cryptos = self.filter_top_cryptos(cryptos, all_mids, meta)
            position_values, total_account_value, usdc_target_balance = (
                self.calculate_account_values(user_state, self.config.leverage)
            )

            total_market_cap = sum(coin["market_cap"] for coin in top_cryptos)
            usdc_balance = float(user_state["crossMarginSummary"]["totalRawUsd"])
            tradeable_balance = total_account_value - usdc_target_balance

            allocation_data, _, _ = self.calculate_allocations(
                top_cryptos,
                position_values,
                tradeable_balance,
                total_market_cap,
                cryptos,
                usdc_balance,
                usdc_target_balance,
            )

            for allocation in allocation_data:
                if abs(allocation.difference) > self.config.position_drift_threshold:
                    emoji = "🔼" if allocation.difference > 0.0 else "🔽"
                    message = [
                        f"{emoji} <b>Coin difference alert</b> {emoji}",
                        f"Coin: {allocation.name} ({allocation.symbol})",
                        f"Target value: {fmt(allocation.target_value)} USDC",
                        f"Current value: {fmt(allocation.current_value)} USDC",
                        f"Difference: {fmt(allocation.difference)} USDC",
                    ]
                    await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)

        except Exception as e:
            logger.error(f"Error checking allocation drifts: {str(e)}", exc_info=True)
            await telegram_utils.send(f"Error checking allocation drifts: {str(e)}")

    async def display_crypto_info(
        self,
        update: Update,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> None:
        """Display comprehensive crypto portfolio information."""
        try:
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

            table_data = self.generate_table_data(
                allocation_data,
                other_positions,
                tradeable_balance,
                total_market_cap,
                usdc_balance,
                usdc_target_balance,
                usdc_difference,
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
            logger.error(f"Error displaying crypto info: {str(e)}", exc_info=True)

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Analyze current portfolio state and display information."""
        try:
            cryptos, all_mids, meta = self.get_strategy_params()
            await self.display_crypto_info(update, cryptos, all_mids, meta)
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")

    async def rebalance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Rebalance portfolio according to strategy parameters."""
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
            
            # Calculate available balance after keeping target USDC
            available_usdc = max(0, usdc_balance - usdc_target_balance)
            if available_usdc < self.config.min_position_size:  # Minimum amount to trade
                await telegram_utils.reply(
                    update, 
                    f"Insufficient balance for trading. Available: {fmt(available_usdc)} USDC after keeping {fmt(usdc_target_balance)} USDC as reserve"
                )
                return
                
            # Calculate tradeable balance using only available USDC
            tradeable_balance = available_usdc * self.config.leverage

            allocation_data, other_positions, _ = self.calculate_allocations(
                top_cryptos,
                position_values,
                tradeable_balance,
                total_market_cap,
                cryptos,
                usdc_balance,
                usdc_target_balance,
            )
                    
            for allocation in allocation_data:
                try:
                    if allocation.target_value < self.config.min_position_size:  # Check target value instead of difference
                        logger.info(
                            f"The target value for {allocation.symbol} is less than 10 USDC "
                            f"({fmt(allocation.target_value)} USDC) and can't be executed"
                        )
                        continue

                    exchange.update_leverage(self.config.leverage, allocation.symbol, False)
                    mid = float(all_mids[allocation.symbol])
                    sz_decimals = hyperliquid_utils.get_sz_decimals()
                    # Use target_value instead of difference for new positions
                    sz = round(allocation.target_value / mid, sz_decimals[allocation.symbol])
                    logger.info(f"Opening position for {allocation.symbol}: {sz} units (value: {fmt(allocation.target_value)} USDC)")
                    open_result = exchange.market_open(allocation.symbol, True, sz)
                    logger.info(open_result)
                except Exception as e:
                    logger.error(f"Unable to open position for {allocation.symbol}: {str(e)}")
                    continue
            
            await telegram_utils.reply(update, "Rebalancing completed")
        except Exception as e:
            logger.critical(e, exc_info=True)
            await telegram_utils.reply(
                update, f"Error during rebalancing: {str(e)}", parse_mode=ParseMode.HTML
            )

    @abstractmethod
    def get_strategy_params(self) -> Tuple[List[Dict], Dict[str, str], Dict]:
        """Get strategy-specific parameters including crypto data and exchange info."""
        pass

    @abstractmethod
    def filter_top_cryptos(
        self,
        cryptos: List[Dict],
        all_mids: Dict[str, str],
        meta: Dict
    ) -> List[Dict]:
        """Filter and sort cryptos according to strategy criteria."""
        pass
