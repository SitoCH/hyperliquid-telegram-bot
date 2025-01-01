import pytest
from unittest.mock import MagicMock, AsyncMock

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
