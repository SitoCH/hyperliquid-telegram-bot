import base64
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from datetime import datetime, timedelta, timezone
from telegram.ext import ConversationHandler

from hyperliquid_bot import main, load_strategy, start, shutdown as shutdown_func


def make_update_context(context_args=None):
    update = MagicMock()
    update.message = MagicMock()
    update.message.delete = AsyncMock()
    context = MagicMock()
    context.args = context_args or []
    context.user_data = {}
    return update, context


class TestLoadStrategy:
    def test_loads_strategy_successfully(self):
        mock_strategy_instance = MagicMock()
        mock_module = MagicMock()
        mock_module.TestStrategy = MagicMock(return_value=mock_strategy_instance)

        with patch("hyperliquid_bot.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_module
            result = load_strategy("test_strategy")

        assert result is mock_strategy_instance

    def test_returns_none_on_module_not_found(self):
        with patch("hyperliquid_bot.importlib") as mock_importlib, \
                patch("hyperliquid_bot.logger") as mock_logger:
            mock_importlib.import_module.side_effect = ModuleNotFoundError("no module")
            result = load_strategy("nonexistent")

        assert result is None
        mock_logger.critical.assert_called_once()

    def test_returns_none_on_attribute_error(self):
        with patch("hyperliquid_bot.importlib") as mock_importlib, \
                patch("hyperliquid_bot.logger") as mock_logger:
            mock_importlib.import_module.return_value = object()
            result = load_strategy("bad_strategy")

        assert result is None
        mock_logger.critical.assert_called_once()

    def test_strategy_name_title_conversion(self):
        mock_strategy_instance = MagicMock()
        mock_module = MagicMock()
        mock_module.FixedTokenStrategy = MagicMock(return_value=mock_strategy_instance)

        with patch("hyperliquid_bot.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_module
            result = load_strategy("fixed_token_strategy")

        assert result is mock_strategy_instance

    def test_constructed_with_no_args(self):
        mock_strategy_instance = MagicMock()
        mock_module = MagicMock()
        mock_module.AlphaG = MagicMock(return_value=mock_strategy_instance)

        with patch("hyperliquid_bot.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_module
            result = load_strategy("alpha_g")

        assert result is mock_strategy_instance
        mock_module.AlphaG.assert_called_once_with()

    def test_module_name_format(self):
        mock_module = MagicMock()
        mock_module.EtfStrategy = MagicMock(return_value=MagicMock())

        with patch("hyperliquid_bot.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_module
            load_strategy("etf_strategy")

        mock_importlib.import_module.assert_called_once_with("strategies.etf_strategy.etf_strategy")


class TestStart:
    @pytest.mark.asyncio
    async def test_no_args_sends_welcome(self, mock_telegram_utils):
        update, context = make_update_context()

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils):
            result = await start(update, context)

        assert result == ConversationHandler.END
        mock_telegram_utils.reply.assert_called_once_with(update, ANY)
        assert "Welcome" in mock_telegram_utils.reply.call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_args_returns_end(self, mock_telegram_utils):
        update, context = make_update_context()

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils):
            result = await start(update, context)

        assert result == ConversationHandler.END

    @pytest.mark.asyncio
    async def test_ta_prefix_deletes_message_and_calls_execute_ta(self, mock_telegram_utils):
        update, context = make_update_context(["TA_BTC"])
        mock_execute_ta = AsyncMock(return_value=ConversationHandler.END)

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils), \
                patch("hyperliquid_bot.execute_ta", mock_execute_ta):
            result = await start(update, context)

        assert result == ConversationHandler.END
        update.message.delete.assert_called_once()
        mock_execute_ta.assert_called_once_with(update, context)
        assert context.args == ["BTC"]

    @pytest.mark.asyncio
    async def test_trd_long_prefix_decodes_and_calls_enter_long(self, mock_telegram_utils):
        update, context = make_update_context()
        raw_param = "TRD_" + base64.b64encode(b"L_BTC_45000_55000").decode()
        context.args = [raw_param]
        mock_enter_long = AsyncMock(return_value=ConversationHandler.END)

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils), \
                patch("hyperliquid_bot.enter_long", mock_enter_long):
            result = await start(update, context)

        assert result == ConversationHandler.END
        update.message.delete.assert_called_once()
        mock_enter_long.assert_called_once_with(update, context)
        assert context.args == ["BTC", "45000", "55000"]

    @pytest.mark.asyncio
    async def test_trd_short_prefix_decodes_and_calls_enter_short(self, mock_telegram_utils):
        update, context = make_update_context()
        raw_param = "TRD_" + base64.b64encode(b"S_ETH_2000_1800").decode()
        context.args = [raw_param]
        mock_enter_short = AsyncMock(return_value=ConversationHandler.END)

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils), \
                patch("hyperliquid_bot.enter_short", mock_enter_short):
            result = await start(update, context)

        assert result == ConversationHandler.END
        update.message.delete.assert_called_once()
        mock_enter_short.assert_called_once_with(update, context)
        assert context.args == ["ETH", "2000", "1800"]

    @pytest.mark.asyncio
    async def test_trd_without_valid_side_returns_end(self, mock_telegram_utils):
        update, context = make_update_context()
        raw_param = "TRD_" + base64.b64encode(b"X_SOL_100_200").decode()
        context.args = [raw_param]
        mock_enter_long = AsyncMock()
        mock_enter_short = AsyncMock()

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils), \
                patch("hyperliquid_bot.enter_long", mock_enter_long), \
                patch("hyperliquid_bot.enter_short", mock_enter_short):
            result = await start(update, context)

        assert result == ConversationHandler.END
        update.message.delete.assert_called_once()
        mock_enter_long.assert_not_called()
        mock_enter_short.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_arg_prefix_returns_end(self, mock_telegram_utils):
        update, context = make_update_context(["UNKNOWN_param"])

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils):
            result = await start(update, context)

        assert result == ConversationHandler.END
        update.message.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_ta_path_propagates_execute_ta_return_value(self, mock_telegram_utils):
        update, context = make_update_context(["TA_ETH"])
        expected_state = 42
        mock_execute_ta = AsyncMock(return_value=expected_state)

        with patch("hyperliquid_bot.telegram_utils", mock_telegram_utils), \
                patch("hyperliquid_bot.execute_ta", mock_execute_ta):
            result = await start(update, context)

        assert result == expected_state


class TestShutdown:
    @pytest.mark.asyncio
    async def test_logs_and_exits(self):
        mock_application = MagicMock()

        with patch("hyperliquid_bot.logger") as mock_logger, \
                patch("hyperliquid_bot.os._exit") as mock_exit:
            await shutdown_func(mock_application)

        mock_logger.info.assert_called_once_with("Shutting down Hyperliquid Telegram bot...")
        mock_exit.assert_called_once_with(0)


class TestMainHelpers:
    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_websocket_initialized(self, mock_tz, mock_dt, mock_tg, mock_hl):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime

        main()

        mock_hl.init_websocket.assert_called_once()

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_subscribes_to_user_events(self, mock_tz, mock_dt, mock_tg, mock_hl):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime

        main()

        mock_hl.info.subscribe.assert_called_once_with(
            {"type": "userEvents", "user": mock_hl.address},
            ANY
        )


class TestMain:
    WITHOUT_EXCHANGE_HANDLER_COUNT = 6  # start, overview, stats, positions, orders, ta
    WITH_EXCHANGE_HANDLER_COUNT = 9  # above + exit, long, short

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_sets_up_handlers_and_starts_polling(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

        mock_hl.init_websocket.assert_called_once()
        mock_hl.info.subscribe.assert_called_once()
        assert mock_tg.add_handler.call_count == self.WITHOUT_EXCHANGE_HANDLER_COUNT
        mock_tg.run_repeating.assert_called_once()
        mock_tg.run_polling.assert_called_once_with(shutdown_func)

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", True)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {"HTB_STRATEGY": "test_strategy"}, clear=True)
    def test_main_with_exchange_loads_strategy_and_adds_handlers(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        with patch("hyperliquid_bot.load_strategy") as mock_load:
            mock_strategy = MagicMock()
            mock_load.return_value = mock_strategy
            main()

        mock_load.assert_called_once_with("test_strategy")
        mock_tg.run_once.assert_called_once_with(mock_strategy.init_strategy)
        assert mock_tg.add_handler.call_count == self.WITH_EXCHANGE_HANDLER_COUNT

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", True)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_exchange_no_strategy_does_not_load(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

        mock_tg.run_once.assert_not_called()

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", True)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_schedules_candle_analysis_when_exchange_enabled(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

        assert mock_tg.run_repeating.call_count >= 2

        interval_args = [c[1].get("interval") for c in mock_tg.run_repeating.call_args_list]
        assert timedelta(hours=1) in interval_args

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_disabled_exchange_logs_message(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        with patch("hyperliquid_bot.logger") as mock_logger:
            main()

        mock_logger.info.assert_any_call("Exchange orders disabled")
        assert mock_tg.add_handler.call_count == self.WITHOUT_EXCHANGE_HANDLER_COUNT

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {"HTB_MONITOR_STALE_POSITIONS": "True"}, clear=True)
    def test_main_schedules_stale_position_check_when_enabled(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

        call_args_list = mock_tg.run_repeating.call_args_list
        stale_intervals = [c[1].get("interval") for c in call_args_list]
        assert timedelta(hours=1) in stale_intervals

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_main_does_not_schedule_stale_check_when_disabled(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

        assert mock_tg.run_repeating.call_count == 1

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {"HTB_MONITOR_STALE_POSITIONS": "True"}, clear=True)
    def test_stale_check_uses_date_based_first_calculation(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        mock_now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_dt.datetime = datetime
        mock_hl.info.user_state.return_value = {}

        main()

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", False)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {"HTB_MONITOR_STALE_POSITIONS": "True"}, clear=True)
    def test_stale_check_calculation_when_past_50_minutes(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        # Current time is 12:55:00, next_stale_check initially 12:50:00
        # It should be incremented to 13:50:00
        mock_now = datetime(2024, 1, 15, 12, 55, 0, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_hl.info.user_state.return_value = {}

        main()

        call_args_list = mock_tg.run_repeating.call_args_list
        stale_calls = [
            c for c in call_args_list
            if c[1].get("interval") == timedelta(hours=1)
        ]
        assert len(stale_calls) >= 1
        first_call_kwargs = stale_calls[0].kwargs
        expected_first = datetime(2024, 1, 15, 13, 50, 0, tzinfo=timezone.utc)
        assert first_call_kwargs.get("first") == expected_first

    @patch("hyperliquid_bot.hyperliquid_utils")
    @patch("hyperliquid_bot.telegram_utils")
    @patch("hyperliquid_bot.exchange_enabled", True)
    @patch("hyperliquid_bot.datetime")
    @patch("hyperliquid_bot.get_localzone")
    @patch.dict(os.environ, {}, clear=True)
    def test_candle_analysis_calculation_when_past_15_seconds(
        self, mock_tz, mock_dt, mock_tg, mock_hl
    ):
        mock_tz.return_value = timezone.utc
        # Current time is 12:00:20, next_hour initially 12:00:15
        # It should be incremented to 13:00:15
        mock_now = datetime(2024, 1, 15, 12, 0, 20, tzinfo=timezone.utc)
        mock_dt.datetime.now.return_value = mock_now
        mock_dt.timedelta = timedelta
        mock_hl.info.user_state.return_value = {}

        main()

        call_args_list = mock_tg.run_repeating.call_args_list
        # The one with interval=1 hour and first is around next hour
        candle_calls = [
            c for c in call_args_list
            if c[1].get("interval") == timedelta(hours=1)
            and c[1].get("first") is not None
            and c[1].get("first").minute == 0
        ]
        assert len(candle_calls) >= 1
        first_call_kwargs = candle_calls[0].kwargs
        expected_first = datetime(2024, 1, 15, 13, 0, 15, tzinfo=timezone.utc)
        assert first_call_kwargs.get("first") == expected_first
