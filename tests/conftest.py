import pytest
import os
from unittest.mock import MagicMock, AsyncMock

# Set required environment variable before any module imports
os.environ.setdefault("HTB_USER_WALLET", "0x0000000000000000000000000000000000000000")

@pytest.fixture
def mock_hyperliquid_info():
    mock_info = MagicMock()
    mock_info.user_state = MagicMock()
    return mock_info

@pytest.fixture
def mock_telegram_utils():
    mock_utils = MagicMock()
    mock_utils.send = AsyncMock()
    return mock_utils
