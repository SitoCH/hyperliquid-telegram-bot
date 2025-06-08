from typing import Dict, Any, List
from enum import Enum

class Signal(Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class LLMAnalysisTradingSetup:

    def __init__(
        self,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit

class LLMAnalysisResult:
    """Container for LLM analysis results."""
    
    def __init__(
        self,
        signal: Signal = Signal.HOLD,
        confidence: float = 0.5,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        should_notify: bool = False,
        key_drivers: List[str] | None = None,
        recap_heading: str = "",
        trading_insight: str = "",
        time_horizon_hours: int = 4,
        trading_setup: LLMAnalysisTradingSetup | None = None
    ):
        self.signal = signal
        self.confidence = confidence
        self.risk_level = risk_level
        self.should_notify = should_notify

        self.key_drivers = key_drivers or []
        self.recap_heading = recap_heading
        self.trading_insight = trading_insight
        self.time_horizon_hours = time_horizon_hours
        self.trading_setup = trading_setup