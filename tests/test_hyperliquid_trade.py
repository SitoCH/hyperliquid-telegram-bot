import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram.ext import ConversationHandler

from hyperliquid_trade import (
    EXIT_CHOOSING,
    SELECTING_COIN,
    SELECTING_STOP_LOSS,
    SELECTING_TAKE_PROFIT,
    SELECTING_AMOUNT,
    SELECTING_LEVERAGE,
    enter_position,
    enter_long,
    enter_short,
    skip_sl_tp_prompt,
    has_order_error,
    get_order_error_message,
    selected_amount,
    selected_leverage,
    get_price_suggestions,
    PriceSuggestion,
    selected_stop_loss,
    selected_coin,
    calculate_available_margin,
    get_amount_suggestions,
    selected_take_profit,
    open_order,
    place_stop_loss_order,
    place_stop_loss_and_take_profit_orders,
    close_all_positions_core,
    exit_all_positions,
    exit_position,
    exit_selected_coin,
    _has_sl_tp_set,
    _is_long_position,
    _validate_stop_loss_price,
    _validate_take_profit_price,
    _validate_order_context,
    _handle_callback_selection,
    _handle_callback_cancel,
)


class TestHandleCallbackCancel:
    """Tests for _handle_callback_cancel helper."""
    
    @pytest.mark.asyncio
    async def test_cancel_edits_message_and_ends(self):
        query = AsyncMock()
        result = await _handle_callback_cancel(query)
        query.edit_message_text.assert_called_once()
        assert result == ConversationHandler.END


class TestHandleCallbackSelection:
    """Tests for _handle_callback_selection helper."""
    
    @pytest.mark.asyncio
    async def test_no_query_returns_end(self):
        update = MagicMock()
        update.callback_query = None
        result = await _handle_callback_selection(update, int, "Error", AsyncMock())
        assert result == ConversationHandler.END
    
    @pytest.mark.asyncio
    async def test_cancel_data_calls_cancel_handler(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'cancel'
        update.callback_query = query
        
        result = await _handle_callback_selection(update, int, "Error", AsyncMock())
        query.answer.assert_called_once()
        query.edit_message_text.assert_called_once()
        assert result == ConversationHandler.END
    
    @pytest.mark.asyncio
    async def test_none_data_calls_cancel_handler(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = None
        update.callback_query = query
        
        result = await _handle_callback_selection(update, int, "Error", AsyncMock())
        assert result == ConversationHandler.END
    
    @pytest.mark.asyncio
    async def test_valid_data_with_converter_calls_next_action(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = '42'
        update.callback_query = query
        
        next_action = AsyncMock(return_value=SELECTING_STOP_LOSS)
        result = await _handle_callback_selection(update, int, "Error", next_action)
        
        next_action.assert_called_once_with(query, 42)
        assert result == SELECTING_STOP_LOSS
    
    @pytest.mark.asyncio
    async def test_valid_data_without_converter_passes_raw_value(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'BTC'
        update.callback_query = query
        
        next_action = AsyncMock(return_value=SELECTING_AMOUNT)
        result = await _handle_callback_selection(update, None, "Error", next_action)
        
        next_action.assert_called_once_with(query, 'BTC')
        assert result == SELECTING_AMOUNT
    
    @pytest.mark.asyncio
    async def test_invalid_conversion_shows_error(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'not_a_number'
        update.callback_query = query
        
        result = await _handle_callback_selection(update, int, "Invalid number.", AsyncMock())
        
        query.edit_message_text.assert_called_once_with("Invalid number.")
        assert result == ConversationHandler.END


class TestHelperFunctions:
    """Tests for helper functions introduced during refactoring."""
    
    def test_has_sl_tp_set_both_present(self):
        context = MagicMock()
        context.user_data = {'stop_loss_price': 45000, 'take_profit_price': 55000}
        assert _has_sl_tp_set(context) is True

    def test_has_sl_tp_set_only_sl(self):
        context = MagicMock()
        context.user_data = {'stop_loss_price': 45000}
        assert _has_sl_tp_set(context) is False

    def test_has_sl_tp_set_only_tp(self):
        context = MagicMock()
        context.user_data = {'take_profit_price': 55000}
        assert _has_sl_tp_set(context) is False

    def test_has_sl_tp_set_neither(self):
        context = MagicMock()
        context.user_data = {}
        assert _has_sl_tp_set(context) is False

    def test_is_long_position_long(self):
        context = MagicMock()
        context.user_data = {'enter_mode': 'long'}
        assert _is_long_position(context) is True

    def test_is_long_position_short(self):
        context = MagicMock()
        context.user_data = {'enter_mode': 'short'}
        assert _is_long_position(context) is False


class TestValidateStopLossPrice:
    def test_negative_price(self):
        assert _validate_stop_loss_price(-100, 50000, True) == "Price must be zero or greater."

    def test_zero_price_valid(self):
        assert _validate_stop_loss_price(0, 50000, True) is None

    def test_long_sl_above_market_invalid(self):
        error = _validate_stop_loss_price(55000, 50000, True)
        assert error == "Stop loss price must be below current market price for long positions."

    def test_long_sl_below_market_valid(self):
        assert _validate_stop_loss_price(45000, 50000, True) is None

    def test_short_sl_below_market_invalid(self):
        error = _validate_stop_loss_price(45000, 50000, False)
        assert error == "Stop loss price must be above current market price for short positions."

    def test_short_sl_above_market_valid(self):
        assert _validate_stop_loss_price(55000, 50000, False) is None


class TestValidateTakeProfitPrice:
    def test_zero_price_invalid(self):
        assert _validate_take_profit_price(0, 50000, True) == "Price must be greater than 0."

    def test_negative_price_invalid(self):
        assert _validate_take_profit_price(-100, 50000, True) == "Price must be greater than 0."

    def test_long_tp_below_market_invalid(self):
        error = _validate_take_profit_price(45000, 50000, True)
        assert error == "Take profit price must be above current market price for long positions."

    def test_long_tp_above_market_valid(self):
        assert _validate_take_profit_price(55000, 50000, True) is None

    def test_short_tp_above_market_invalid(self):
        error = _validate_take_profit_price(55000, 50000, False)
        assert error == "Take profit price must be below current market price for short positions."

    def test_short_tp_below_market_valid(self):
        assert _validate_take_profit_price(45000, 50000, False) is None


class TestValidateOrderContext:
    def test_missing_amount(self):
        context = MagicMock()
        context.user_data = {'stop_loss_price': 45000, 'take_profit_price': 55000, 'selected_coin': 'BTC'}
        error = _validate_order_context(context)
        assert "No amount selected" in error

    def test_missing_stop_loss(self):
        context = MagicMock()
        context.user_data = {'amount': 100, 'take_profit_price': 55000, 'selected_coin': 'BTC'}
        error = _validate_order_context(context)
        assert "No stop loss selected" in error

    def test_missing_take_profit(self):
        context = MagicMock()
        context.user_data = {'amount': 100, 'stop_loss_price': 45000, 'selected_coin': 'BTC'}
        error = _validate_order_context(context)
        assert "No take profit selected" in error

    def test_missing_coin(self):
        context = MagicMock()
        context.user_data = {'amount': 100, 'stop_loss_price': 45000, 'take_profit_price': 55000}
        error = _validate_order_context(context)
        assert "No coin selected" in error

    def test_all_present_valid(self):
        context = MagicMock()
        context.user_data = {
            'amount': 100,
            'stop_loss_price': 45000,
            'take_profit_price': 55000,
            'selected_coin': 'BTC'
        }
        assert _validate_order_context(context) is None


class TestSkipSlTpPrompt:
    def test_skip_when_env_true(self):
        with patch.dict('os.environ', {'HTB_SKIP_SL_TP_PROMPT': 'True'}):
            assert skip_sl_tp_prompt() is True

    def test_no_skip_when_env_false(self):
        with patch.dict('os.environ', {'HTB_SKIP_SL_TP_PROMPT': 'False'}):
            assert skip_sl_tp_prompt() is False

    def test_no_skip_when_env_not_set(self):
        with patch.dict('os.environ', {}, clear=True):
            assert skip_sl_tp_prompt() is False


class TestHasOrderError:
    def test_none_result(self):
        assert has_order_error(None) is True

    def test_non_dict_result(self):
        assert has_order_error("error") is True
        assert has_order_error(123) is True

    def test_status_not_ok(self):
        result = {'status': 'error'}
        assert has_order_error(result) is True

    def test_order_with_error_in_statuses(self):
        result = {
            'status': 'ok',
            'response': {
                'type': 'order',
                'data': {
                    'statuses': [{'error': 'Insufficient funds'}]
                }
            }
        }
        assert has_order_error(result) is True

    def test_order_without_error(self):
        result = {
            'status': 'ok',
            'response': {
                'type': 'order',
                'data': {
                    'statuses': [{'filled': {'oid': 123}}]
                }
            }
        }
        assert has_order_error(result) is False

    def test_order_empty_statuses(self):
        result = {
            'status': 'ok',
            'response': {
                'type': 'order',
                'data': {
                    'statuses': []
                }
            }
        }
        assert has_order_error(result) is False


class TestGetOrderErrorMessage:
    def test_none_result(self):
        assert get_order_error_message(None) == "Unknown error"

    def test_non_dict_result(self):
        assert get_order_error_message("error") == "Unknown error"

    def test_status_not_ok(self):
        result = {'status': 'error', 'message': 'Failed'}
        assert "error" in get_order_error_message(result).lower()

    def test_error_in_statuses(self):
        result = {
            'status': 'ok',
            'response': {
                'type': 'order',
                'data': {
                    'statuses': [{'error': 'Insufficient funds'}]
                }
            }
        }
        assert get_order_error_message(result) == 'Insufficient funds'

    def test_no_error_in_statuses(self):
        result = {
            'status': 'ok',
            'response': {
                'type': 'order',
                'data': {
                    'statuses': [{'filled': {'oid': 123}}]
                }
            }
        }
        assert get_order_error_message(result) == "Unknown order error"


class TestEnterPosition:
    @pytest.mark.asyncio
    async def test_enter_long_calls_enter_position(self):
        update = MagicMock()
        context = MagicMock()
        context.args = []
        context.user_data = {}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.get_coins_reply_markup.return_value = MagicMock()

            result = await enter_long(update, context)

            assert result == SELECTING_COIN
            assert context.user_data['enter_mode'] == 'long'

    @pytest.mark.asyncio
    async def test_enter_short_calls_enter_position(self):
        update = MagicMock()
        context = MagicMock()
        context.args = []
        context.user_data = {}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.get_coins_reply_markup.return_value = MagicMock()

            result = await enter_short(update, context)

            assert result == SELECTING_COIN
            assert context.user_data['enter_mode'] == 'short'

    @pytest.mark.asyncio
    async def test_enter_position_with_args_goes_to_amount(self):
        update = MagicMock()
        context = MagicMock()
        context.args = ['BTC', '45000', '55000']
        context.user_data = {}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.get_amount_suggestions', new_callable=AsyncMock) as mock_suggestions:
            mock_tg.reply = AsyncMock()
            mock_suggestions.return_value = ("message", MagicMock())

            result = await enter_position(update, context, "long")

            assert result == SELECTING_AMOUNT
            assert context.user_data['selected_coin'] == 'BTC'
            assert context.user_data['stop_loss_price'] == 45000.0
            assert context.user_data['take_profit_price'] == 55000.0

    @pytest.mark.asyncio
    async def test_enter_position_with_skip_sl_tp(self):
        update = MagicMock()
        context = MagicMock()
        context.args = []
        context.user_data = {}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.skip_sl_tp_prompt', return_value=True):
            mock_tg.reply = AsyncMock()
            mock_hl.get_coins_reply_markup.return_value = MagicMock()

            result = await enter_position(update, context, "long")

            assert result == SELECTING_COIN
            assert context.user_data['stop_loss_price'] == 'skip'
            assert context.user_data['take_profit_price'] == 'skip'


class TestSelectedAmount:
    @pytest.mark.asyncio
    async def test_cancel_operation(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'cancel'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        result = await selected_amount(update, context)

        assert result == ConversationHandler.END
        query.edit_message_text.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_amount(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'invalid'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        result = await selected_amount(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_valid_amount_existing_leverage(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = '50'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC'}

        mock_user_state = {'assetPositions': []}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.send_stop_loss_suggestions', new_callable=AsyncMock) as mock_send_sl:
            mock_hl.info.user_state.return_value = mock_user_state
            mock_hl.get_leverage.return_value = 10
            mock_hl.address = "test_address"

            result = await selected_amount(update, context)

            assert result == SELECTING_STOP_LOSS
            assert context.user_data['amount'] == 50.0
            assert context.user_data['leverage'] == 10

    @pytest.mark.asyncio
    async def test_no_callback_query(self):
        update = MagicMock()
        update.callback_query = None
        context = MagicMock()

        result = await selected_amount(update, context)

        assert result == ConversationHandler.END


class TestSelectedLeverage:
    @pytest.mark.asyncio
    async def test_cancel_leverage(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'cancel'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        result = await selected_leverage(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_invalid_leverage(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'invalid'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        result = await selected_leverage(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_valid_leverage_without_sl_tp(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = '10'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC'}

        with patch('hyperliquid_trade.send_stop_loss_suggestions', new_callable=AsyncMock) as mock_send_sl:
            result = await selected_leverage(update, context)

            assert result == SELECTING_STOP_LOSS
            assert context.user_data['leverage'] == 10


class TestGetPriceSuggestions:
    @pytest.mark.asyncio
    async def test_long_stop_loss_suggestions(self):
        with patch('hyperliquid_trade.get_significant_levels_from_timeframe', new_callable=AsyncMock) as mock_levels:
            mock_levels.return_value = ([55000, 60000], [45000, 40000])

            suggestions = await get_price_suggestions('BTC', 50000, is_stop_loss=True, is_long=True)

            # Should have percentage-based and level-based suggestions
            assert len(suggestions) > 0
            assert all(isinstance(s, PriceSuggestion) for s in suggestions)

            # For long stop loss, all prices should be below mid
            fixed_suggestions = [s for s in suggestions if s.type == 'Fixed']
            for s in fixed_suggestions:
                assert s.price < 50000

    @pytest.mark.asyncio
    async def test_short_stop_loss_suggestions(self):
        with patch('hyperliquid_trade.get_significant_levels_from_timeframe', new_callable=AsyncMock) as mock_levels:
            mock_levels.return_value = ([55000, 60000], [45000, 40000])

            suggestions = await get_price_suggestions('BTC', 50000, is_stop_loss=True, is_long=False)

            # For short stop loss, fixed prices should be above mid
            fixed_suggestions = [s for s in suggestions if s.type == 'Fixed']
            for s in fixed_suggestions:
                assert s.price > 50000

    @pytest.mark.asyncio
    async def test_long_take_profit_suggestions(self):
        with patch('hyperliquid_trade.get_significant_levels_from_timeframe', new_callable=AsyncMock) as mock_levels:
            mock_levels.return_value = ([55000, 60000], [45000, 40000])

            suggestions = await get_price_suggestions('BTC', 50000, is_stop_loss=False, is_long=True)

            # For long take profit (is_long is reversed), fixed prices should be above mid
            fixed_suggestions = [s for s in suggestions if s.type == 'Fixed']
            for s in fixed_suggestions:
                assert s.price > 50000


class TestSelectedStopLoss:
    @pytest.mark.asyncio
    async def test_cancel_stop_loss(self):
        update = MagicMock()
        message = MagicMock()
        message.text = 'cancel'
        update.message = message

        context = MagicMock()

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.reply = AsyncMock()

            result = await selected_stop_loss(update, context)

            assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_invalid_stop_loss_price(self):
        update = MagicMock()
        message = MagicMock()
        message.text = 'invalid'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.reply = AsyncMock()

            result = await selected_stop_loss(update, context)

            assert result == SELECTING_STOP_LOSS

    @pytest.mark.asyncio
    async def test_negative_stop_loss_price(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '-100'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_stop_loss(update, context)

            assert result == SELECTING_STOP_LOSS

    @pytest.mark.asyncio
    async def test_long_stop_loss_above_market(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '55000'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_stop_loss(update, context)

            assert result == SELECTING_STOP_LOSS

    @pytest.mark.asyncio
    async def test_short_stop_loss_below_market(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '45000'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'short'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_stop_loss(update, context)

            assert result == SELECTING_STOP_LOSS

    @pytest.mark.asyncio
    async def test_valid_stop_loss_proceeds_to_take_profit(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '45000'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.get_price_suggestions_text', new_callable=AsyncMock) as mock_suggestions:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}
            mock_suggestions.return_value = "Take profit suggestions"

            result = await selected_stop_loss(update, context)

            assert result == SELECTING_TAKE_PROFIT
            assert context.user_data['stop_loss_price'] == 45000.0


class TestSelectedCoin:
    @pytest.mark.asyncio
    async def test_cancel_coin_selection(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'cancel'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        result = await selected_coin(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_valid_coin_selection(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'BTC'
        update.callback_query = query

        context = MagicMock()
        context.user_data = {}

        with patch('hyperliquid_trade.get_amount_suggestions', new_callable=AsyncMock) as mock_suggestions:
            mock_suggestions.return_value = ("message", MagicMock())

            result = await selected_coin(update, context)

            assert result == SELECTING_AMOUNT
            assert context.user_data['selected_coin'] == 'BTC'


class TestCalculateAvailableMargin:
    def test_calculate_available_margin(self):
        mock_perp_state = {
            'crossMarginSummary': {
                'accountValue': '10000',
                'totalMarginUsed': '3000'
            }
        }

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.info.user_state.return_value = mock_perp_state
            mock_hl.address = "test_address"

            result = calculate_available_margin()

            assert result == 7000.0

    def test_calculate_available_margin_negative_returns_zero(self):
        mock_perp_state = {
            'crossMarginSummary': {
                'accountValue': '3000',
                'totalMarginUsed': '5000'
            }
        }

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.info.user_state.return_value = mock_perp_state
            mock_hl.address = "test_address"

            result = calculate_available_margin()

            assert result == 0.0


class TestGetAmountSuggestions:
    @pytest.mark.asyncio
    async def test_get_amount_suggestions(self):
        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.calculate_available_margin', return_value=10000.0):
            message, reply_markup = await get_amount_suggestions(context)

            assert 'BTC' in message
            assert 'long' in message
            assert reply_markup is not None


class TestSelectedTakeProfit:
    @pytest.mark.asyncio
    async def test_cancel_take_profit(self):
        update = MagicMock()
        message = MagicMock()
        message.text = 'cancel'
        update.message = message

        context = MagicMock()

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.reply = AsyncMock()

            result = await selected_take_profit(update, context)

            assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_invalid_take_profit(self):
        update = MagicMock()
        message = MagicMock()
        message.text = 'invalid'
        update.message = message

        context = MagicMock()

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.reply = AsyncMock()

            result = await selected_take_profit(update, context)

            assert result == SELECTING_TAKE_PROFIT

    @pytest.mark.asyncio
    async def test_zero_take_profit(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '0'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_take_profit(update, context)

            assert result == SELECTING_TAKE_PROFIT

    @pytest.mark.asyncio
    async def test_long_take_profit_below_market(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '45000'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'long'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_take_profit(update, context)

            assert result == SELECTING_TAKE_PROFIT

    @pytest.mark.asyncio
    async def test_short_take_profit_above_market(self):
        update = MagicMock()
        message = MagicMock()
        message.text = '55000'
        update.message = message

        context = MagicMock()
        context.user_data = {'selected_coin': 'BTC', 'enter_mode': 'short'}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg, \
             patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.info.all_mids.return_value = {'BTC': '50000'}

            result = await selected_take_profit(update, context)

            assert result == SELECTING_TAKE_PROFIT


class TestOpenOrder:
    @pytest.mark.asyncio
    async def test_open_order_no_amount(self):
        context = MagicMock()
        context.user_data = {}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.send = AsyncMock()

            result = await open_order(context, {}, 50000.0)

            assert result == ConversationHandler.END
            mock_tg.send.assert_called_with("Error: No amount selected. Please restart the process.")

    @pytest.mark.asyncio
    async def test_open_order_no_stop_loss(self):
        context = MagicMock()
        context.user_data = {'amount': 100}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.send = AsyncMock()

            result = await open_order(context, {}, 50000.0)

            assert result == ConversationHandler.END
            mock_tg.send.assert_called_with("Error: No stop loss selected. Please restart the process.")

    @pytest.mark.asyncio
    async def test_open_order_no_take_profit(self):
        context = MagicMock()
        context.user_data = {'amount': 100, 'stop_loss_price': 45000}

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.send = AsyncMock()

            result = await open_order(context, {}, 50000.0)

            assert result == ConversationHandler.END
            mock_tg.send.assert_called_with("Error: No take profit selected. Please restart the process.")

    @pytest.mark.asyncio
    async def test_open_order_no_coin(self):
        context = MagicMock()
        context.user_data = {
            'amount': 100,
            'stop_loss_price': 45000,
            'take_profit_price': 55000
        }

        with patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_tg.send = AsyncMock()

            result = await open_order(context, {}, 50000.0)

            assert result == ConversationHandler.END
            mock_tg.send.assert_called_with("Error: No coin selected. Please restart the process.")


class TestPlaceStopLossOrder:
    def test_place_stop_loss_order_long(self):
        mock_exchange = MagicMock()
        user_state = {}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.get_liquidation_px_str.return_value = None

            place_stop_loss_order(
                mock_exchange, user_state, 'BTC', True, 1.0, 45000.0, 6
            )

            mock_exchange.order.assert_called_once()

    def test_place_stop_loss_order_with_liquidation_adjustment_long(self):
        mock_exchange = MagicMock()
        user_state = {}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            # Liquidation price that would trigger adjustment
            mock_hl.get_liquidation_px_str.return_value = "46000"

            place_stop_loss_order(
                mock_exchange, user_state, 'BTC', True, 1.0, 45000.0, 6
            )

            mock_exchange.order.assert_called_once()

    def test_place_stop_loss_order_zero_price(self):
        mock_exchange = MagicMock()
        user_state = {}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.get_liquidation_px_str.return_value = None

            place_stop_loss_order(
                mock_exchange, user_state, 'BTC', True, 1.0, 0.0, 6
            )

            # No order should be placed with zero stop loss and no liquidation
            mock_exchange.order.assert_not_called()


class TestPlaceStopLossAndTakeProfitOrders:
    @pytest.mark.asyncio
    async def test_place_both_orders(self):
        mock_exchange = MagicMock()
        user_state = {}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.place_stop_loss_order') as mock_sl:
            mock_hl.get_liquidation_px_str.return_value = None

            await place_stop_loss_and_take_profit_orders(
                mock_exchange, user_state, 'BTC', True, 1.0, 45000.0, 55000.0, 6
            )

            mock_sl.assert_called_once()
            mock_exchange.order.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_take_profit_when_zero(self):
        mock_exchange = MagicMock()
        user_state = {}

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.place_stop_loss_order') as mock_sl:
            mock_hl.get_liquidation_px_str.return_value = None

            await place_stop_loss_and_take_profit_orders(
                mock_exchange, user_state, 'BTC', True, 1.0, 45000.0, 0.0, 6
            )

            mock_sl.assert_called_once()
            mock_exchange.order.assert_not_called()


class TestCloseAllPositionsCore:
    def test_close_all_positions_empty(self):
        mock_exchange = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.info.user_state.return_value = {'assetPositions': []}
            mock_hl.address = "test_address"

            closed, errors = close_all_positions_core(mock_exchange)

            assert closed == []
            assert errors == []

    def test_close_all_positions_success(self):
        mock_exchange = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.info.user_state.return_value = {
                'assetPositions': [
                    {'position': {'coin': 'BTC'}},
                    {'position': {'coin': 'ETH'}}
                ]
            }
            mock_hl.address = "test_address"

            closed, errors = close_all_positions_core(mock_exchange)

            assert 'BTC' in closed
            assert 'ETH' in closed
            assert errors == []
            assert mock_exchange.market_close.call_count == 2

    def test_close_all_positions_missing_coin(self):
        mock_exchange = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.info.user_state.return_value = {
                'assetPositions': [
                    {'position': {}},  # Missing coin
                    {'position': {'coin': 'ETH'}}
                ]
            }
            mock_hl.address = "test_address"

            closed, errors = close_all_positions_core(mock_exchange)

            assert 'ETH' in closed
            assert len(closed) == 1


class TestExitAllPositions:
    @pytest.mark.asyncio
    async def test_exit_all_no_exchange(self):
        update = MagicMock()
        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_hl.get_exchange.return_value = None
            mock_tg.reply = AsyncMock()

            await exit_all_positions(update, context)

            mock_tg.reply.assert_called_with(update, "Exchange is not enabled")

    @pytest.mark.asyncio
    async def test_exit_all_success(self):
        update = MagicMock()
        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.close_all_positions_core') as mock_close:
            mock_hl.get_exchange.return_value = MagicMock()
            mock_close.return_value = (['BTC'], [])

            await exit_all_positions(update, context)

            mock_close.assert_called_once()


class TestExitPosition:
    @pytest.mark.asyncio
    async def test_exit_no_positions(self):
        update = MagicMock()
        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_hl.info.user_state.return_value = {'assetPositions': []}
            mock_hl.address = "test_address"
            mock_tg.reply = AsyncMock()

            result = await exit_position(update, context)

            assert result == ConversationHandler.END
            mock_tg.reply.assert_called_with(update, 'No positions to close')

    @pytest.mark.asyncio
    async def test_exit_with_positions(self):
        update = MagicMock()
        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.telegram_utils') as mock_tg:
            mock_hl.info.user_state.return_value = {
                'assetPositions': [
                    {'position': {'coin': 'BTC'}},
                    {'position': {'coin': 'ETH'}}
                ]
            }
            mock_hl.address = "test_address"
            mock_tg.reply = AsyncMock()

            result = await exit_position(update, context)

            assert result == EXIT_CHOOSING
            mock_tg.reply.assert_called()


class TestExitSelectedCoin:
    @pytest.mark.asyncio
    async def test_exit_cancel(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'cancel'
        update.callback_query = query

        context = MagicMock()

        result = await exit_selected_coin(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_exit_all_positions_via_menu(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'all'
        update.callback_query = query

        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl, \
             patch('hyperliquid_trade.close_all_positions_core') as mock_close:
            mock_hl.get_exchange.return_value = MagicMock()
            mock_close.return_value = (['BTC'], [])

            result = await exit_selected_coin(update, context)

            assert result == ConversationHandler.END
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_specific_coin(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'BTC'
        update.callback_query = query

        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_exchange = MagicMock()
            mock_hl.get_exchange.return_value = mock_exchange

            result = await exit_selected_coin(update, context)

            assert result == ConversationHandler.END
            mock_exchange.market_close.assert_called_with('BTC')

    @pytest.mark.asyncio
    async def test_exit_no_exchange(self):
        update = MagicMock()
        query = AsyncMock()
        query.data = 'BTC'
        update.callback_query = query

        context = MagicMock()

        with patch('hyperliquid_trade.hyperliquid_utils') as mock_hl:
            mock_hl.get_exchange.return_value = None

            result = await exit_selected_coin(update, context)

            assert result == ConversationHandler.END
            query.edit_message_text.assert_called_with("Exchange is not enabled")
