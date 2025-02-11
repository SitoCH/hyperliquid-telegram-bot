import time
from typing import Deque
from collections import deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_weight_per_minute: int, weight_per_call: int):
        self.max_weight_per_minute = max_weight_per_minute
        self.weight_per_call = weight_per_call
        self.timestamps: Deque[float] = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Remove timestamps older than 1 minute
            while self.timestamps and self.timestamps[0] < minute_ago:
                self.timestamps.popleft()
            
            # Calculate current weight
            current_weight = len(self.timestamps) * self.weight_per_call
            
            # Wait if adding another call would exceed the limit
            if current_weight + self.weight_per_call > self.max_weight_per_minute:
                sleep_time = self.timestamps[0] - minute_ago
                time.sleep(max(0, sleep_time))
                # Clean up old timestamps again after waiting
                while self.timestamps and self.timestamps[0] < time.time() - 60:
                    self.timestamps.popleft()
            
            # Add current timestamp
            self.timestamps.append(now)
