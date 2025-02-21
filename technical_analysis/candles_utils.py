import os
import time

from typing import Set, List, Dict, Any, Optional, cast, Tuple
from hyperliquid_utils.utils import hyperliquid_utils
from logging_utils import logger

async def get_coins_to_analyze(all_mids: Dict[str, Any]) -> Set[str]:
    """Get the set of coins to analyze based on configuration."""
    coins_to_analyze: Set[str] = set()
    
    # Get top N coins by open interest if configured
    top_coins = os.environ.get("HTB_TOP_COINS_TO_ANALYZE")
    if top_coins:
        coins_to_analyze.update(hyperliquid_utils.get_coins_by_open_interest()[:int(top_coins)])

    # Add explicitly configured coins
    configured_coins = os.getenv("HTB_COINS_TO_ANALYZE", "").split(",")
    coins_to_analyze.update(coin for coin in configured_coins if coin and coin in all_mids)
    
    # Add coins with open orders if configured
    if os.getenv('HTB_ANALYZE_COINS_WITH_OPEN_ORDERS', 'False') == 'True':
        coins_to_analyze.update(
            coin for coin in hyperliquid_utils.get_coins_with_open_positions()
            if coin in all_mids
        )
    
    return coins_to_analyze