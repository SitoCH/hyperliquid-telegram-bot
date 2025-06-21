from dataclasses import dataclass
from typing import Dict, List, Any
from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hyperliquid_utils.utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt
from logging_utils import logger

@dataclass
class PortfolioBalance:
    perp_total: float
    perp_withdrawable: float 
    perp_margin_available: float 
    spot_total: float
    stacked_total: float
    cross_margin_ratio: float
    cross_account_leverage: float
    
    @property
    def total(self) -> float:
        return self.perp_total + self.spot_total + self.stacked_total
        
def _calculate_spot_balance(spot_state: Dict[str, Any], token_prices: Dict[str, float]) -> float:
    """Calculate total spot balance from user state and token prices."""
    return sum(
        float(balance['total']) * token_prices.get(balance['coin'], 0.0)
        for balance in spot_state.get('balances', [])
    )

def _calculate_stacked_balance(staking_summary: Dict[str, float], token_prices: Dict[str, float]) -> float:
    """Calculate total stacked balance from delegator info and token prices."""
    hype_price = token_prices.get("HYPE", 0.0)
    
    delegated_value = float(staking_summary["delegated"]) * hype_price
    undelegated_value = float(staking_summary["undelegated"]) * hype_price
    pending_value = float(staking_summary["totalPendingWithdrawal"]) * hype_price
    
    return delegated_value + undelegated_value + pending_value


def _get_portfolio_balance() -> PortfolioBalance:
    """Get current portfolio balance information."""
    address = hyperliquid_utils.address
    perp_state = hyperliquid_utils.info.user_state(address)
    spot_state = hyperliquid_utils.info.spot_user_state(address)
    token_prices = _get_token_prices()
    staking_summary = hyperliquid_utils.info.user_staking_summary(address)

    cross_margin_account_value =float(perp_state['crossMarginSummary']['accountValue'])
    cross_margin_positions_value =float(perp_state['crossMarginSummary']['totalNtlPos'])
    maintenance_margin = float(perp_state['crossMaintenanceMarginUsed'])

    total_margin_used = float(perp_state['crossMarginSummary']['totalMarginUsed'])
    
    return PortfolioBalance(
        perp_total=float(perp_state['marginSummary']['accountValue']),
        perp_withdrawable = float(perp_state['withdrawable']),
        perp_margin_available = cross_margin_account_value - total_margin_used if total_margin_used > 0.0 else 0.0,
        spot_total=_calculate_spot_balance(spot_state, token_prices),
        stacked_total=_calculate_stacked_balance(staking_summary, token_prices),
        cross_margin_ratio = maintenance_margin / cross_margin_account_value if cross_margin_account_value > 0 else 0.0,
        cross_account_leverage = cross_margin_positions_value / cross_margin_account_value if cross_margin_account_value > 0 else 0.0,
    )

def _format_portfolio_message(balance: PortfolioBalance) -> List[str]:
    """Format portfolio balance information as message lines."""
    message = [
        "<b>Portfolio:</b>",
        f"Total balance: {fmt(balance.total)} USDC",
    ]
     
    if balance.spot_total > 0:
        message.extend([
        "<b>Spot positions:</b>",
        f"Total balance: {fmt(balance.spot_total)} USDC", 
        ])

    if balance.stacked_total > 0:
        message.extend([
        "<b>Stacked positions:</b>",
        f"Total balance: {fmt(balance.stacked_total)} USDC", 
        ])

    message.extend([
        "<b>Perps positions:</b>",
        f"Total balance: {fmt(balance.perp_total)} USDC", 
        f"Withdrawable balance: {fmt(balance.perp_withdrawable)} USDC",
    ])

    if balance.perp_margin_available > 0.0:
        message.extend([
            f"Available to trade: {fmt(balance.perp_margin_available)} USDC"
        ])

    if balance.cross_margin_ratio > 0:
        message.extend([
            f"Cross margin ratio: {fmt(balance.cross_margin_ratio * 100)}%",
            f"Cross account leverage: {fmt(balance.cross_account_leverage)}x"
        ])

    return message

async def get_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for getting current portfolio positions."""
    try:
        balance = _get_portfolio_balance()
        message_lines = _format_portfolio_message(balance)
        
        all_mids = hyperliquid_utils.info.all_mids()
        perp_user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        tablefmt = simple_separated_format('  ')
        if perp_user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in perp_user_state["assetPositions"]
            )
            message_lines.append(f"Unrealized profit: {fmt(total_pnl)} USDC")
            await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)

            sorted_positions = sorted(
                perp_user_state["assetPositions"],
                key=lambda x: float(x['position']['positionValue']),
                reverse=True
            )

            for asset_position in sorted_positions:
                coin = asset_position['position']['coin']
                coin_message_lines = [
                    f"<b>{telegram_utils.get_link(coin, f'TA_{coin}')}:</b>"
                ]
                table_data = []
                table_data.append(
                    [
                        "PnL",
                        f"{fmt(float(asset_position['position']['unrealizedPnl']))}$",
                        f"({fmt(float(asset_position['position']['returnOnEquity']) * 100.0)}%)"
                    ]
                )
                table_data.append(
                    [
                        "Entry price",
                        "",
                        f"{asset_position['position']['entryPx']}"
                    ]
                )
                table_data.append(
                    [
                        "Mid price",
                        "",
                        f"{all_mids[asset_position['position']['coin']]}"
                    ]
                )
                table_data.append(
                    [
                        "Margin used",
                        "",
                        f"{fmt(float(asset_position['position']['marginUsed']))}$"
                    ]
                )
                table_data.append(
                    [
                        "Leverage",
                        "",
                        f"{asset_position['position']['leverage']['value']}x"
                    ]
                )
                table_data.append(
                    [
                        "Funding",
                        "",
                        f"{fmt(float(asset_position['position']['cumFunding']['sinceOpen']) * -1.0)}$"
                    ]
                )
                table_data.append(
                    [
                        "Pos. value",
                        "",
                        f"{fmt(float(asset_position['position']['positionValue']))}$"
                    ]
                )
                table_data.append(
                    [
                        "Size",
                        "",
                        f"{asset_position['position']['szi']}"
                    ]
                )
                table = tabulate(
                    table_data,
                    headers=[" ", " ", " ", " "],
                    tablefmt=tablefmt,
                    colalign=("right", "right", "right")
                )

                coin_message_lines.append(f"<pre>{table}</pre>")

                await telegram_utils.reply(update, '\n'.join(coin_message_lines), parse_mode=ParseMode.HTML)
        else:
            await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)

        spot_messages = await spot_positions_messages(tablefmt, hyperliquid_utils.info.spot_user_state(hyperliquid_utils.address))
        if len(spot_messages) > 0:
            await telegram_utils.reply(update, '\n'.join(spot_messages), parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        await telegram_utils.reply(
            update,
            "Error getting positions. Please try again later."
        )

async def get_overview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        balance = _get_portfolio_balance()
        message_lines = _format_portfolio_message(balance)

        perp_user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        tablefmt = simple_separated_format(' ')
        if perp_user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in perp_user_state["assetPositions"]
            )
            message_lines.append(f"Unrealized profit: {fmt(total_pnl)} USDC")
            message_lines.append("")

            sorted_positions = sorted(
                perp_user_state["assetPositions"],
                key=lambda x: float(x['position']['positionValue']),
                reverse=True
            )

            table = tabulate(
                [
                    [
                        "(L)" if float(position['position']['szi']) > 0 else "(S)",
                        position['position']['coin'],
                        f"{fmt(float(position['position']['positionValue']))}$",
                        f"{fmt(float(position['position']['unrealizedPnl']))}$",
                        f"{fmt(float(position['position']['returnOnEquity']) * 100.0)}%"
                    ]
                    for position in sorted_positions
                ],
                headers=["", "Coin", "Balance", "PnL $", "PnL %"],
                tablefmt=tablefmt,
                colalign=("left", "left", "right", "right", "right")
            )

            message_lines.append(f"<pre>{table}</pre>")

        spot_messages = await spot_positions_messages(tablefmt, hyperliquid_utils.info.spot_user_state(hyperliquid_utils.address))
        message_lines += spot_messages
        await telegram_utils.reply(update, '\n'.join(message_lines), parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.reply(update, f"Failed to fetch positions: {str(e)}")

def _get_token_prices():
    """Get token prices from market data."""
    metadata, market_data = hyperliquid_utils.info.spot_meta_and_asset_ctxs()
    tokens, universe = metadata["tokens"], metadata["universe"]
    market_data_map = {i: data for i, data in enumerate(market_data)}
    token_prices = {"USDC": 1.0}

    for pair in universe:
        market = market_data_map.get(pair["index"])
        if not market or not (mid_price := float(market.get("midPx") or 0)):
            continue
        
        base_token, quote_token = (tokens[idx]["name"] for idx in pair["tokens"])
        
        if quote_token == "USDC":
            token_prices[base_token] = mid_price
        elif base_token == "USDC":
            token_prices[quote_token] = 1 / mid_price
        else:
            if base_token in token_prices:
                token_prices[quote_token] = token_prices[base_token] / mid_price
            elif quote_token in token_prices:
                token_prices[base_token] = token_prices[quote_token] * mid_price

    return token_prices


async def spot_positions_messages(tablefmt, spot_user_state):
    """Generate messages for spot positions, sorted by USD value."""

    if not spot_user_state['balances']:
        return []

    token_prices = _get_token_prices()
    
    positions = []
    for balance in spot_user_state['balances']:
        token = balance['coin']
        amount = float(balance['total'])
        entry_value = float(balance['entryNtl'])
        price = token_prices.get(token, 0.0)
        usd_value = price * amount
        
        if usd_value > 1.0:
            pnl = usd_value - entry_value
            pnl_percentage = (pnl / entry_value * 100) if entry_value != 0 else 0
            positions.append({
                'token': token,
                'usd_value': usd_value,
                'pnl': pnl,
                'pnl_percentage': pnl_percentage
            })

    if len(positions) == 0:
        return []

    positions.sort(key=lambda x: x['usd_value'], reverse=True)

    table = tabulate(
        [
            [
                pos['token'],
                f"{fmt(pos['usd_value'])}$",
                f"{fmt(pos['pnl'])}$" if pos['token'] != 'USDC' else '',
                f"({fmt(pos['pnl_percentage'])}%)" if pos['token'] != 'USDC' else ''
            ]
            for pos in positions
        ],
        headers=["Coin", "Balance", "PnL", ""],
        tablefmt=tablefmt,
        colalign=("left", "right", "right", "right")
    )
    
    return [
        "",
        "<b>Spot positions:</b>",
        f"<pre>{table}</pre>"
    ]
