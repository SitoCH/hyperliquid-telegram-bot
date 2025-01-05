import os
import time
import asyncio

from typing import Set, List, Dict, Any, Optional, cast, Tuple
from hyperliquid_utils import hyperliquid_utils
from logging_utils import logger

async def get_coins_to_analyze(all_mids: Dict[str, Any]) -> Set[str]:
    """Get the set of coins to analyze based on configuration."""
    coins_to_analyze: Set[str] = set()
    
    # Add explicitly configured coins
    configured_coins = os.getenv("HTB_COINS_TO_ANALYZE", "").split(",")
    coins_to_analyze.update(coin for coin in configured_coins if coin and coin in all_mids)
    
    # Add coins from configured categories
    if categories := os.getenv("HTB_CATEGORIES_TO_ANALYZE"):
        for category in categories.split(","):
            if not category.strip():
                continue
                
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 30,
                "sparkline": "false",
                "category": category.strip(),
                "price_change_percentage": "24h,30d,1y",
            }
            
            try:
                cryptos = hyperliquid_utils.fetch_cryptos(params)
                coins_to_analyze.update(
                    crypto["symbol"] for crypto in cryptos 
                    if crypto["symbol"] in all_mids
                )
            except Exception as e:
                logger.error(f"Error fetching cryptos for category {category}: {str(e)}", exc_info=True)
            
            await asyncio.sleep(3)
    
    # Add coins with open orders if configured
    if os.getenv('HTB_ANALYZE_COINS_WITH_OPEN_ORDERS', 'False') == 'True':
        coins_to_analyze.update(
            coin for coin in hyperliquid_utils.get_coins_with_open_positions()
            if coin in all_mids
        )
    
    return coins_to_analyze