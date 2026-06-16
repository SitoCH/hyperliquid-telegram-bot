import pytest
from unittest.mock import MagicMock, patch
from strategies.default_strategy.default_strategy import DefaultStrategy

class TestDefaultStrategy:
    @pytest.mark.asyncio
    async def test_init_strategy_success(self):
        strategy = DefaultStrategy()
        context = MagicMock()
        
        with patch('strategies.default_strategy.default_strategy.logger') as mock_logger:
            await strategy.init_strategy(context)
            mock_logger.info.assert_called_with('Running default strategy')

    @pytest.mark.asyncio
    async def test_init_strategy_error(self):
        strategy = DefaultStrategy()
        context = MagicMock()
        
        # Test error handling by making logger.info raise an exception
        with patch('strategies.default_strategy.default_strategy.logger') as mock_logger:
            mock_logger.info.side_effect = Exception("Test error")
            await strategy.init_strategy(context)
            mock_logger.error.assert_called()
            args, _ = mock_logger.error.call_args
            assert "Error executing default strategy: Test error" in args[0]
