import time
from typing import Optional

class HyperliquidRateLimiter:
    def __init__(self, max_weight_per_minute: int = 900):
        self.max_weight_per_minute = max_weight_per_minute
        self.current_minute: int = 0
        self.current_weight: int = 0

    def add_weight(self, weight: int) -> None:
        current_minute = int(time.time() / 60)
        if current_minute != self.current_minute:
            self.current_minute = current_minute
            self.current_weight = 0
        self.current_weight += weight

    def get_next_available_time(self, weight: int) -> float:
        """
        Returns the number of seconds to wait before the next request can be made.
        If the weight would exceed the limit, returns seconds until next minute.
        """
        current_minute = int(time.time() / 60)
        current_second = time.time() % 60
        
        if current_minute != self.current_minute:
            # New minute started, no wait needed
            return 0
            
        if self.current_weight + weight <= self.max_weight_per_minute:
            # Can execute immediately
            return 0
            
        # Need to wait until next minute
        return 60 - current_second

# Global rate limiter instance
hyperliquid_rate_limiter = HyperliquidRateLimiter()
