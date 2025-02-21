import time
from math import floor
from logging_utils import logger

class HyperliquidRateLimiter:

    def __init__(self, max_weight_per_minute=1000):
        self.max_weight_per_minute = max_weight_per_minute
        self.current_minute = self._get_current_minute()
        self.current_weight = 0
    

    def _get_current_minute(self) -> int:
        """Get the current minute timestamp by rounding down to nearest minute"""
        return floor(time.time() / 60) * 60
    

    def _reset_if_new_minute(self) -> None:
        current = self._get_current_minute()
        if current > self.current_minute:
            self.current_minute = current
            self.current_weight = 0
    

    def add_weight(self, weight: int) -> None:
        self._reset_if_new_minute()
        self.current_weight += weight
        # logger.debug(f"Rate limiter current weight: {self.current_weight}")
    

    def get_next_available_time(self, weight: int) -> float:
        """Returns seconds until the weight can be accommodated"""
        self._reset_if_new_minute()
        
        if self.current_weight + weight <= self.max_weight_per_minute:
            return 0
            
        # Calculate time until next minute
        current_time = time.time()
        next_minute = self.current_minute + 60
        return max(0, next_minute - current_time)


# Global instance
hyperliquid_rate_limiter = HyperliquidRateLimiter()
