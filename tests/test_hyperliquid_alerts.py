import pytest
from unittest.mock import patch, MagicMock
from telegram.ext import ContextTypes
import os

from hyperliquid_alerts import check_profit_percentage, check_positions_to_close

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

@pytest.mark.asyncio
async def test_check_positions_to_close_stale_dynamic_threshold(mock_hyperliquid_info, mock_telegram_utils, mock_context):
    with patch('hyperliquid_alerts.hyperliquid_utils.info', mock_hyperliquid_info), \
         patch('hyperliquid_alerts.telegram_utils', mock_telegram_utils), \
         patch('hyperliquid_alerts.time.time') as mock_time, \
         patch.dict(os.environ, {'HTB_STALE_POSITION_MIN_PNL': '5'}):
        
        # Set current time to a fixed value
        current_time = 1000000.0
        mock_time.return_value = current_time
        
        # 4 hours in seconds
        stale_duration = 4 * 60 * 60
        
        # Case 1: Position age 6h (exactly stale), PnL 6% (> 5%). Should alert.
        trade_time_1 = (current_time - stale_duration - 1) * 1000 # Just over 6h
        
        # Case 2: Position age 12h (2x stale), PnL 3% (> 2.5%). Should alert.
        trade_time_2 = (current_time - (stale_duration * 2) - 1) * 1000
        
        # Case 3: Position age 12h (2x stale), PnL 2% (< 2.5%). Should NOT alert.
        trade_time_3 = (current_time - (stale_duration * 2) - 1) * 1000
        
        # Setup mock response
        mock_hyperliquid_info.user_state.return_value = {
            'assetPositions': [
                {
                    'position': {
                        'coin': 'BTC',
                        'unrealizedPnl': '100.0',
                        'returnOnEquity': '0.06', # 6%
                        'szi': '1',
                        'entryPx': '50000'
                    }
                },
                {
                    'position': {
                        'coin': 'ETH',
                        'unrealizedPnl': '50.0',
                        'returnOnEquity': '0.03', # 3%
                        'szi': '10',
                        'entryPx': '3000'
                    }
                },
                {
                    'position': {
                        'coin': 'SOL',
                        'unrealizedPnl': '20.0',
                        'returnOnEquity': '0.02', # 2%
                        'szi': '100',
                        'entryPx': '100'
                    }
                }
            ]
        }
        
        mock_hyperliquid_info.frontend_open_orders.return_value = [
            {'coin': 'BTC', 'timestamp': str(int(trade_time_1))},
            {'coin': 'ETH', 'timestamp': str(int(trade_time_2))},
            {'coin': 'SOL', 'timestamp': str(int(trade_time_3))}
        ]

        await check_positions_to_close(mock_context)

        # Verify alerts
        assert mock_telegram_utils.send.call_count == 2
        
        calls = mock_telegram_utils.send.call_args_list
        messages = [call[0][0] for call in calls]
        
        assert any('Stale position on BTC' in msg for msg in messages)
        assert any('Stale position on ETH' in msg for msg in messages)
        assert not any('Stale position on SOL' in msg for msg in messages)
