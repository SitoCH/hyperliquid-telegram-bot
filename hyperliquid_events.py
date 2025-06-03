import json
from typing import List
from hyperliquid.utils.types import UserEventsMsg, Fill
from telegram_utils import telegram_utils
from logging_utils import logger

def get_fill_icon(closed_pnl: float) -> str:
    return "🟢" if closed_pnl > 0 else "🔴"

def get_fill_description(initial_message: str, coin: str, size: str, fee: float, fee_token: str, amount: float | None = None, closed_pnl: float | None = None) -> str:
    fill_description = [
        initial_message,
        f"Coin: {telegram_utils.get_link(coin, f'TA_{coin}')}"
    ]

    if amount is not None:
        fill_description.append(f"Amount: {amount:,.02f} USDC")

    if closed_pnl is not None:
        fill_description.append(f"Profit: {closed_pnl:,.02f} USDC")

    fill_description.append(f"Size: {size}")
    fill_description.append(f"Fee: {fee:,.02f} {fee_token}")

    return '\n'.join(fill_description)

def process_fill(fill: Fill) -> None:
    price = float(fill["px"])
    coin = fill["coin"]
    size = fill["sz"]
    fee = float(fill["fee"])
    fee_token = fill["feeToken"]
    amount = price * float(size)
    closed_pnl = float(fill["closedPnl"])
    if fill["dir"] == 'Open Long':
        fill_message = get_fill_description("🔵 Opened long:", coin, size, fee, fee_token, amount)
    elif fill["dir"] == 'Open Short':
        fill_message = get_fill_description("🔵 Opened short:", coin, size, fee, fee_token, amount)
    elif fill["dir"] == 'Close Long':
        fill_message = get_fill_description(f"{get_fill_icon(closed_pnl)} Closed long:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
    elif fill["dir"] == 'Close Short':
        fill_message = get_fill_description(f"{get_fill_icon(closed_pnl)} Closed short:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
    elif fill["dir"] == 'Buy':
        fill_message = get_fill_description("🔵 Bought spot:", coin, size, fee, fee_token, amount)
    elif fill["dir"] == 'Sell':
        fill_message = get_fill_description("🔵 Sold spot:", coin, size, fee, fee_token, amount)
    elif fill["dir"] == 'Liquidated Isolated Long':
        fill_message = get_fill_description(f"{get_fill_icon(closed_pnl)} Liquidated isolated long:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
    elif fill["dir"] == 'Long > Short':
        fill_message = get_fill_description(f"{get_fill_icon(closed_pnl)} Long -> short:", coin, size, fee, fee_token, closed_pnl=closed_pnl)
    elif fill["dir"] == 'Spot Dust Conversion':
        fill_message = get_fill_description("🧹 Dust conversion:", coin, size, fee, fee_token, amount)
    else:
        fill_message = json.dumps(fill)

    telegram_utils.queue_send(fill_message)

def on_user_events(user_events: UserEventsMsg) -> None:
    user_events_data = user_events["data"]
    if "fills" in user_events_data:
        fill_events: List[Fill] = user_events_data["fills"]
        for fill in fill_events:
            process_fill(fill)
            logger.info(fill)
