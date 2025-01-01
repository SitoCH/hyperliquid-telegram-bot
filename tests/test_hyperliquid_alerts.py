import pytest
from unittest.mock import patch, MagicMock
from telegram.ext import ContextTypes

from hyperliquid_alerts import check_profit_percentage

@pytest.fixture
def mock_context():
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)

@pytest.mark.asyncio
async def test_check_profit_high_available_balance(mock_hyperliquid_info, mock_telegram_utils, mock_context):
    with patch('hyperliquid_alerts.hyperliquid_utils.info', mock_hyperliquid_info), \
         patch('hyperliquid_alerts.telegram_utils', mock_telegram_utils):
        
        # Setup mock response
        mock_hyperliquid_info.user_state.return_value = {
            'marginSummary': {'accountValue': '1000.0'},
            'withdrawable': '150.0',
            'assetPositions': []
        }

        await check_profit_percentage(mock_context)

        # Verify alert was sent
        assert mock_telegram_utils.send.called
        args = mock_telegram_utils.send.call_args[0][0]
        assert '💰 <b>Available balance alert</b> 💰' in args
        assert '150.00 USDC' in args

@pytest.mark.asyncio
async def test_check_profit_high_pnl_percentage(mock_hyperliquid_info, mock_telegram_utils, mock_context):
    with patch('hyperliquid_alerts.hyperliquid_utils.info', mock_hyperliquid_info), \
         patch('hyperliquid_alerts.telegram_utils', mock_telegram_utils):
        
        # Setup mock response with high profit
        mock_hyperliquid_info.user_state.return_value = {
            'marginSummary': {'accountValue': '1000.0'},
            'withdrawable': '50.0',
            'assetPositions': [
                {'position': {'unrealizedPnl': '600.0'}}
            ]
        }

        await check_profit_percentage(mock_context)

        # Verify profit alert was sent
        assert mock_telegram_utils.send.called
        args = mock_telegram_utils.send.call_args[0][0]
        assert '🚀 <b>Unrealized profit alert</b> 🚀' in args
        assert '600.00 USDC' in args

@pytest.mark.asyncio
async def test_check_profit_high_loss_percentage(mock_hyperliquid_info, mock_telegram_utils, mock_context):
    with patch('hyperliquid_alerts.hyperliquid_utils.info', mock_hyperliquid_info), \
         patch('hyperliquid_alerts.telegram_utils', mock_telegram_utils):
        
        # Setup mock response with high loss
        mock_hyperliquid_info.user_state.return_value = {
            'marginSummary': {'accountValue': '1000.0'},
            'withdrawable': '50.0',
            'assetPositions': [
                {'position': {'unrealizedPnl': '-600.0'}}
            ]
        }

        await check_profit_percentage(mock_context)

        # Verify loss alert was sent
        assert mock_telegram_utils.send.called
        args = mock_telegram_utils.send.call_args[0][0]
        assert '📉 <b>Unrealized loss alert</b> 📉' in args
        assert '-600.00 USDC' in args

@pytest.mark.asyncio
async def test_check_profit_no_alerts_needed(mock_hyperliquid_info, mock_telegram_utils, mock_context):
    with patch('hyperliquid_alerts.hyperliquid_utils.info', mock_hyperliquid_info), \
         patch('hyperliquid_alerts.telegram_utils', mock_telegram_utils):
        
        # Setup mock response with normal values
        mock_hyperliquid_info.user_state.return_value = {
            'marginSummary': {'accountValue': '1000.0'},
            'withdrawable': '50.0',
            'assetPositions': [
                {'position': {'unrealizedPnl': '100.0'}}
            ]
        }

        await check_profit_percentage(mock_context)

        # Verify no alerts were sent
        assert not mock_telegram_utils.send.called
