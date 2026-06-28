from typing import List, Optional, NamedTuple, Dict, Any
from technical_analysis.wyckoff.significant_levels import get_significant_levels_from_timeframe
from technical_analysis.wyckoff.wyckoff_types import Timeframe
from hyperliquid_utils.utils import hyperliquid_utils
from utils import fmt, fmt_price
from tabulate import tabulate, simple_separated_format


class PriceSuggestion(NamedTuple):
    type: str
    price: float
    percentage: float


def _get_mid_price(coin: str) -> float:
    """Get current mid price for a coin."""
    dex = hyperliquid_utils.dex_supported(coin) or ""
    return float(hyperliquid_utils.info.all_mids(dex=dex)[coin])


async def get_price_suggestions(coin: str, mid: float, is_stop_loss: bool, is_long: bool) -> List[PriceSuggestion]:
    """Get price suggestions for either entry or exit points."""
    resistance_levels, support_levels = await get_significant_levels_from_timeframe(coin, mid, Timeframe.HOUR_1, 250)
    suggestions: List[PriceSuggestion] = []

    # For take profit (exit), we reverse the direction compared to stop loss (entry)
    if not is_stop_loss:
        is_long = not is_long

    # Add percentage-based suggestions
    for pct in [1.0, 2.0, 3.0, 4.0, 5.0]:
        price = float(mid * (1.0 - pct / 100.0) if is_long else mid * (1.0 + pct / 100.0))
        suggestions.append(PriceSuggestion("Fixed", price, pct))

    # Add level-based suggestions
    if is_long and support_levels:
        valid_supports = [level for level in support_levels if level < mid]
        for level in sorted(valid_supports, reverse=True)[:3]:
            pct = abs((level - mid) / mid * 100)
            suggestions.append(PriceSuggestion("Support", level, pct))
    elif not is_long and resistance_levels:
        valid_resistances = [level for level in resistance_levels if level > mid]
        for level in sorted(valid_resistances)[:3]:
            pct = abs((level - mid) / mid * 100)
            suggestions.append(PriceSuggestion("Resistance", level, pct))

    # Sort suggestions by price (ascending for shorts, descending for longs)
    suggestions.sort(key=lambda x: x.price, reverse=is_long)
    return suggestions


async def get_price_suggestions_text(coin: str, is_stop_loss: bool, is_long: bool) -> str:
    """Send formatted price suggestions for either stop loss or take profit."""
    mid = _get_mid_price(coin)
    suggestions = await get_price_suggestions(coin, mid, is_stop_loss, is_long)

    table_data = [
        [sugg.type, fmt_price(sugg.price), f"{fmt(sugg.percentage)}%"]
        for sugg in suggestions
    ]
    tablefmt = simple_separated_format('  ')
    table = tabulate(
        table_data,
        headers=["Type", "Price", "Distance"],
        tablefmt=tablefmt,
        colalign=("left", "right", "right")
    )

    price_type = 'stop loss' if is_stop_loss else 'take profit'
    return (
        f"Current market price: {fmt_price(mid)} USDC\n"
        f"Suggested {'stop losses' if is_stop_loss else 'take profits'}:\n"
        f"<pre>{table}</pre>\n"
        f"\nEnter your desired {price_type} price in USDC, or 'cancel' to stop:"
    )


def validate_stop_loss_price(price: float, mid: float, is_long: bool) -> Optional[str]:
    """Validate stop loss price. Returns error message if invalid, None if valid."""
    if price < 0:
        return "Price must be zero or greater."
    if price > 0:
        if is_long and price >= mid:
            return "Stop loss price must be below current market price for long positions."
        if not is_long and price <= mid:
            return "Stop loss price must be above current market price for short positions."
    return None


def validate_take_profit_price(price: float, mid: float, is_long: bool) -> Optional[str]:
    """Validate take profit price. Returns error message if invalid, None if valid."""
    if price <= 0:
        return "Price must be greater than 0."
    if is_long and price <= mid:
        return "Take profit price must be above current market price for long positions."
    if not is_long and price >= mid:
        return "Take profit price must be below current market price for short positions."
    return None


def px_round(px: float, sz_decimals: int) -> float:
    max_decimals = 6
    if px > 100_000:
        px = round(px)
    else:
        px = round(float(f"{px:.5g}"), max_decimals - sz_decimals)
    return px


def get_adjusted_stop_loss_trigger(
    user_state: Dict[str, Any],
    coin: str,
    is_long: bool,
    stop_loss_price: float
) -> float:
    """Get stop loss trigger price, adjusted for liquidation if needed."""
    sl_trigger_px = stop_loss_price
    liquidation_px_str = hyperliquid_utils.get_liquidation_px_str(user_state, coin)

    if liquidation_px_str is not None:
        liquidation_px = float(liquidation_px_str)
        if liquidation_px > 0.0:
            liquidation_trigger_px = liquidation_px * (1.0025 if is_long else 0.9975)
            should_use_liquidation = (
                stop_loss_price == 0
                or (is_long and stop_loss_price < liquidation_trigger_px)
                or (not is_long and stop_loss_price > liquidation_trigger_px)
            )
            if should_use_liquidation:
                sl_trigger_px = liquidation_trigger_px

    return sl_trigger_px
