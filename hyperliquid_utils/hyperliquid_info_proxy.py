import time
from typing import Any, Dict
from hyperliquid.info import Info
from .hyperliquid_ratelimiter import hyperliquid_rate_limiter
from logging_utils import logger

class InfoProxy:
    # Weight definitions for each endpoint
    WEIGHTS = {
        'all_mids': 2,
        'user_state': 2,
        'spot_user_state': 2,
        'spot_meta_and_asset_ctxs': 20,
        'meta_and_asset_ctxs': 20,
        'user_staking_summary': 20,
        'user_fills': 20,
        'frontend_open_orders': 20,
        'meta': 20,
        'subscribe': 0,
        'candles_snapshot': 20,
        'funding_history': 20
    }

    def __init__(self, info: Info):
        self._info = info

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._info, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                result = attr(*args, **kwargs)
                if name in self.WEIGHTS:
                    weight = self.WEIGHTS[name]
                    hyperliquid_rate_limiter.add_weight(weight)
                else:
                    logger.warning(f"InfoProxy: method '{name}' called but not found in WEIGHTS dictionary")
                return result
            return wrapped
        return attr
