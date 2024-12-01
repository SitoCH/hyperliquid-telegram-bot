from tabulate import simple_separated_format, tabulate
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from hyperliquid_utils import hyperliquid_utils
from telegram_utils import telegram_utils
from utils import fmt
from logging_utils import logger

async def check_profit_percentage(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        total_balance = float(user_state['marginSummary']['accountValue'])
        available_balance = float(user_state['withdrawable'])
        
        if available_balance > 100:
            message = [
                "💰 <b>Available balance alert</b> 💰",
                f"Total balance: {fmt(total_balance)} USDC",
                f"Available balance: {fmt(available_balance)} USDC",
            ]
            await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
        
        if user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in user_state["assetPositions"]
            )
            
            pnl_percentage = (total_pnl / total_balance) * 100
            
            if abs(pnl_percentage) > 10:
                emoji = "🚀" if pnl_percentage > 10 else "📉"
                message = [
                    f"{emoji} <b>Unrealized profit alert</b> {emoji}",
                    f"Total balance: {fmt(total_balance)} USDC",
                    f"Unrealized profit: {fmt(total_pnl)} USDC ({fmt(pnl_percentage)}%)",
                ]
                await telegram_utils.send('\n'.join(message), parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.critical(e, exc_info=True)

async def get_positions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        all_mids = hyperliquid_utils.info.all_mids()
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        total_balance = float(user_state['marginSummary']['accountValue'])
        available_balance = float(user_state['withdrawable'])

        perp_message_lines = [
            "<b>Perps positions:</b>",
            f"Total balance: {fmt(total_balance)} USDC",
            f"Available balance: {fmt(available_balance)} USDC",
        ]

        tablefmt = simple_separated_format('  ')
        if user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in user_state["assetPositions"]
            )
            perp_message_lines.append(f"Unrealized profit: {fmt(total_pnl)} USDC")
            await telegram_utils.reply(update, '\n'.join(perp_message_lines), parse_mode=ParseMode.HTML)

            sorted_positions = sorted(
                user_state["assetPositions"],
                key=lambda x: float(x['position']['positionValue']),
                reverse=True
            )

            for asset_position in sorted_positions:
                coin_message_lines = [
                    f"<b>{asset_position['position']['coin']}:</b>"
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
            await telegram_utils.reply(update, '\n'.join(perp_message_lines), parse_mode=ParseMode.HTML)

        spot_messages = await spot_positions_messages(tablefmt)
        if len(spot_messages) > 0:
            await telegram_utils.reply(update, '\n'.join(spot_messages), parse_mode=ParseMode.HTML)

    except Exception as e:
        await telegram_utils.reply(update, f"Failed to fetch positions: {str(e)}")

async def get_overview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user_state = hyperliquid_utils.info.user_state(hyperliquid_utils.address)
        total_balance = float(user_state['marginSummary']['accountValue'])
        available_balance = float(user_state['withdrawable'])

        message_lines = [
            "<b>Perps positions:</b>",
            f"Total balance: {fmt(total_balance)} USDC",
            f"Available balance: {fmt(available_balance)} USDC",
        ]

        tablefmt = simple_separated_format(' ')
        if user_state["assetPositions"]:
            total_pnl = sum(
                float(asset_position['position']['unrealizedPnl'])
                for asset_position in user_state["assetPositions"]
            )
            message_lines.append(f"Unrealized profit: {fmt(total_pnl)} USDC")

            sorted_positions = sorted(
                user_state["assetPositions"],
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
                        f"({fmt(float(position['position']['returnOnEquity']) * 100.0)}%)"
                    ]
                    for position in sorted_positions
                ],
                headers=["", "Coin", "Balance", "PnL", ""],
                tablefmt=tablefmt,
                colalign=("left", "left", "right", "right", "left")
            )

            message_lines.append(f"<pre>{table}</pre>")

        spot_messages = await spot_positions_messages(tablefmt)
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


async def spot_positions_messages(tablefmt):
    """Generate messages for spot positions, sorted by USD value."""
    spot_user_state = hyperliquid_utils.info.spot_user_state(hyperliquid_utils.address)
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
            positions.append({
                'token': token,
                'amount': amount,
                'usd_value': usd_value,
                'entry_value': entry_value
            })
    
    positions.sort(key=lambda x: x['usd_value'], reverse=True)
    
    table = tabulate(
        [
            [
                pos['token'],
                f"{fmt(pos['amount'])}",
                f"{fmt(pos['usd_value'])}$",
                f"{fmt(pos['usd_value'] - pos['entry_value'])}$" if pos['token'] != 'USDC' else ''
            ]
            for pos in positions
        ],
        headers=["Coin", "Balance", "Pos. value", "PnL"],
        tablefmt=tablefmt,
        colalign=("left", "right", "right", "right")
    )
    
    return [
        "<b>Spot positions:</b>",
        f"<pre>{table}</pre>"
    ]
