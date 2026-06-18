import pytest
import os
import warnings
from unittest.mock import MagicMock, AsyncMock, create_autospec
from telegram.warnings import PTBUserWarning

# Suppress PTB warnings globally during tests
warnings.simplefilter("ignore", category=PTBUserWarning)

from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from technical_analysis.wyckoff.wyckoff_types import (
    WyckoffState, WyckoffPhase, VolumeState, MarketPattern,
    VolatilityState, EffortResult, CompositeAction, WyckoffSign,
    FundingState,
)
from telegram_utils import TelegramUtils

# Set required environment variable before any module imports
os.environ.setdefault("HTB_USER_WALLET", "0x0000000000000000000000000000000000000000")


@pytest.fixture
def wyckoff_state():
    """Fixture factory for WyckoffState with sensible defaults."""
    def _build(**overrides) -> WyckoffState:
        defaults = dict(
            phase=WyckoffPhase.RANGING,
            uncertain_phase=False,
            volume=VolumeState.NEUTRAL,
            pattern=MarketPattern.RANGING,
            volatility=VolatilityState.NORMAL,
            is_spring=False,
            is_upthrust=False,
            effort_vs_result=EffortResult.NEUTRAL,
            composite_action=CompositeAction.NEUTRAL,
            wyckoff_sign=WyckoffSign.NONE,
            funding_state=FundingState.NEUTRAL,
            description="Default test state",
        )
        defaults.update(overrides)
        return WyckoffState(**defaults)  # type: ignore[arg-type]
    return _build


@pytest.fixture
def mock_exchange():
    mock = MagicMock(spec=Exchange)
    mock.market_open = MagicMock()
    mock.market_close = MagicMock()
    mock.order = MagicMock()
    mock.update_leverage = MagicMock()
    mock.cancel = MagicMock()
    mock.bulk_cancel = MagicMock()
    mock.schedule_cancel = MagicMock()
    return mock


@pytest.fixture
def mock_hyperliquid_info():
    mock_info = create_autospec(Info, instance=True)
    mock_info.all_mids = MagicMock()
    mock_info.user_state = MagicMock()
    mock_info.meta = MagicMock()
    mock_info.meta_and_asset_ctxs = MagicMock()
    mock_info.frontend_open_orders = MagicMock()
    mock_info.user_fills = MagicMock()
    mock_info.user_fills_by_time = MagicMock()
    mock_info.candles_snapshot = MagicMock()
    mock_info.funding_history = MagicMock()
    mock_info.spot_user_state = MagicMock()
    mock_info.spot_meta_and_asset_ctxs = MagicMock()
    return mock_info


@pytest.fixture
def mock_telegram_utils():
    mock_utils = create_autospec(TelegramUtils, instance=True)
    mock_utils.reply = AsyncMock()
    mock_utils.send = AsyncMock()
    mock_utils.send_and_exit = AsyncMock()
    mock_utils.add_handler = MagicMock()
    mock_utils.run_polling = MagicMock()
    return mock_utils


@pytest.fixture
def sample_user_state():
    return {
        "assetPositions": [
            {
                "position": {
                    "coin": "BTC",
                    "szi": "1.5",
                    "entryPx": "40000",
                    "liquidationPx": "39000",
                    "leverage": {"value": "10", "type": "cross"},
                    "cumFunding": {"allTime": "50"},
                    "returnOnEquity": "0.15",
                }
            },
            {
                "position": {
                    "coin": "ETH",
                    "szi": "-10.0",
                    "entryPx": "2000",
                    "liquidationPx": "2100",
                    "leverage": {"value": "5", "type": "cross"},
                    "cumFunding": {"allTime": "-20"},
                    "returnOnEquity": "-0.05",
                }
            },
        ],
        "crossMarginSummary": {
            "accountValue": "10000",
            "totalMarginUsed": "3000",
            "totalNtlPos": "2500",
            "totalRawUsd": "7000",
        },
        "marginSummary": {
            "accountValue": "10000",
            "totalMarginUsed": "3000",
            "totalNtlPos": "2500",
            "totalRawUsd": "7000",
        },
        "time": 1700000000000,
    }


@pytest.fixture
def sample_all_mids():
    return {
        "BTC": "41000",
        "ETH": "2200",
        "SOL": "120",
        "ARB": "1.2",
        "OP": "3.5",
    }


@pytest.fixture
def sample_open_orders():
    return [
        {"coin": "BTC", "oid": 1, "side": "B", "sz": "1.0", "limitPx": "40000", "orderType": {"limit": {"tif": "Gtc"}}},
        {"coin": "ETH", "oid": 2, "side": "A", "sz": "5.0", "limitPx": "2100", "orderType": {"limit": {"tif": "Gtc"}}},
        {"coin": "BTC", "oid": 3, "side": "A", "sz": "2.0", "limitPx": "42000", "orderType": {"trigger": {"triggerPx": "42000", "isMarket": True}}},
    ]


@pytest.fixture
def sample_positions():
    return [
        {"coin": "BTC", "szi": "1.5", "entryPx": "40000", "liquidationPx": "39000", "leverage": {
            "value": "10", "type": "cross"}, "cumFunding": {"allTime": "50"}, "returnOnEquity": "0.15"},
        {"coin": "ETH", "szi": "-10.0", "entryPx": "2000", "liquidationPx": "2100", "leverage": {"value": "5", "type": "cross"}, "cumFunding": {"allTime": "-20"}, "returnOnEquity": "-0.05"},
    ]
