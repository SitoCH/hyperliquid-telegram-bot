import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock


os.environ.setdefault("HTB_TOKEN", "fake_token_12345")
os.environ.setdefault("HTB_CHAT_ID", "-987654321")


@pytest.fixture(autouse=True)
def mock_application():
    with patch('telegram_utils.Application.builder') as mock_builder:
        mock_builder_instance = MagicMock()
        mock_builder.return_value = mock_builder_instance
        mock_builder_instance.token.return_value = mock_builder_instance
        mock_app = MagicMock()
        mock_app.bot = MagicMock()
        mock_app.bot.username = "test_bot"
        mock_app.job_queue = MagicMock()
        mock_app.job_queue.run_once = MagicMock()
        mock_app.job_queue.run_repeating = MagicMock()
        mock_builder_instance.build.return_value = mock_app
        yield mock_app


class TestTelegramUtilsInit:
    def test_init_success(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        assert instance.telegram_chat_id == "-987654321"
        assert instance.telegram_app is not None

    def test_init_missing_token(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                _ = instance.telegram_app

    def test_init_missing_chat_id(self):
        with patch.dict(os.environ, {"HTB_TOKEN": "tok"}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                _ = instance.telegram_app


class TestTelegramUtilsReplyMarkup:
    def test_reply_markup_has_positions_and_orders(self):
        from telegram_utils import TelegramUtils
        keyboard = TelegramUtils.reply_markup.keyboard
        all_buttons = [btn.text for row in keyboard for btn in row]
        assert "/positions" in all_buttons
        assert "/orders" in all_buttons


class TestTelegramUtilsAddButtons:
    def test_add_buttons(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        instance.add_buttons(["/new_button"])
        all_buttons = [btn.text for row in instance.reply_markup.keyboard for btn in row]
        assert "/new_button" in all_buttons

    def test_add_buttons_custom_row(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        instance.add_buttons(["/btn1", "/btn2"], row_index=5)
        keyboard = instance.reply_markup.keyboard
        assert len(keyboard) > 5
        row_buttons = [btn.text for btn in keyboard[5]]
        assert "/btn1" in row_buttons
        assert "/btn2" in row_buttons


@pytest.mark.asyncio
class TestTelegramUtilsReply:
    async def test_reply_with_message(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        await instance.reply(update, "Hello")
        update.message.reply_text.assert_called_once_with(
            "Hello",
            parse_mode=None,
            reply_markup=instance.reply_markup,
        )

    async def test_reply_with_parse_mode(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        await instance.reply(update, "<b>Bold</b>", parse_mode="HTML")
        update.message.reply_text.assert_called_once_with(
            "<b>Bold</b>",
            parse_mode="HTML",
            reply_markup=instance.reply_markup,
        )

    async def test_reply_with_custom_markup(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        update = MagicMock()
        update.message = MagicMock()
        custom_markup = MagicMock()
        update.message.reply_text = AsyncMock()
        await instance.reply(update, "test", reply_markup=custom_markup)
        update.message.reply_text.assert_called_once_with(
            "test",
            parse_mode=None,
            reply_markup=custom_markup,
        )

    async def test_reply_no_message(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        update = MagicMock()
        update.message = None
        result = await instance.reply(update, "Hello")
        assert result is None


@pytest.mark.asyncio
class TestTelegramUtilsSend:
    async def test_send(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        mock_application.bot.send_message = AsyncMock()
        await instance.send("Test message")
        mock_application.bot.send_message.assert_called_once_with(
            text="Test message",
            parse_mode=None,
            chat_id="-987654321",
        )

    async def test_send_with_parse_mode(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        mock_application.bot.send_message = AsyncMock()
        await instance.send("<b>Bold</b>", parse_mode="HTML")
        mock_application.bot.send_message.assert_called_once_with(
            text="<b>Bold</b>",
            parse_mode="HTML",
            chat_id="-987654321",
        )

    async def test_send_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                await instance.send("Test")


class TestTelegramUtilsSendAndExit:
    def test_send_and_exit(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        instance.send_and_exit("Goodbye")
        mock_application.job_queue.run_once.assert_called_once()

    def test_send_and_exit_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.send_and_exit("Goodbye")


class TestTelegramUtilsQueueSend:
    def test_queue_send(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        instance.queue_send("Queued message")
        mock_application.job_queue.run_once.assert_called_once()

    def test_queue_send_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.queue_send("Queued")


@pytest.mark.asyncio
class TestTelegramUtilsSendMessage:
    async def test_send_message(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        context = MagicMock()
        context.job.chat_id = "-987654321"
        context.job.data = "Test"
        context.bot.send_message = AsyncMock()
        await instance.send_message(context)
        context.bot.send_message.assert_called_once()

    async def test_send_message_no_job(self):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        context = MagicMock()
        context.job = None
        await instance.send_message(context)


@pytest.mark.asyncio
class TestTelegramUtilsSendMessageAndExit:
    async def test_send_message_and_exit(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        context = MagicMock()
        context.job.chat_id = "-987654321"
        context.job.data = "Exit msg"
        context.bot.send_message = AsyncMock()
        with patch('telegram_utils.os._exit') as mock_exit:
            await instance.send_message_and_exit(context)
            context.bot.send_message.assert_called_once()
            mock_exit.assert_called_once_with(0)

    async def test_send_message_and_exit_exception(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        context = MagicMock()
        context.job.chat_id = "-987654321"
        context.job.data = "Fail"
        context.bot.send_message = AsyncMock(side_effect=Exception("network error"))
        with patch('telegram_utils.os._exit') as mock_exit:
            await instance.send_message_and_exit(context)
            mock_exit.assert_called_once_with(0)


class TestTelegramUtilsAddHandler:
    def test_add_handler(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        handler = MagicMock()
        instance.add_handler(handler, group=1)
        mock_application.add_handler.assert_called_once_with(handler, 1)

    def test_add_handler_default_group(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        handler = MagicMock()
        instance.add_handler(handler)
        mock_application.add_handler.assert_called_once_with(handler, 0)

    def test_add_handler_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.add_handler(MagicMock())


class TestTelegramUtilsRunOnce:
    def test_run_once(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        callback = MagicMock()
        instance.run_once(callback)
        mock_application.job_queue.run_once.assert_called_once()

    def test_run_once_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.run_once(MagicMock())


class TestTelegramUtilsRunRepeating:
    def test_run_repeating(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        callback = MagicMock()
        import datetime
        instance.run_repeating(callback, interval=datetime.timedelta(seconds=60))
        mock_application.job_queue.run_repeating.assert_called_once()

    def test_run_repeating_float_interval(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        callback = MagicMock()
        instance.run_repeating(callback, interval=60.0)
        mock_application.job_queue.run_repeating.assert_called_once()

    def test_run_repeating_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.run_repeating(MagicMock(), interval=60)


class TestTelegramUtilsRunPolling:
    def test_run_polling(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        shutdown = MagicMock()
        instance.run_polling(shutdown)
        assert mock_application.post_shutdown == shutdown
        mock_application.run_polling.assert_called_once()

    def test_run_polling_no_app(self):
        with patch.dict(os.environ, {}, clear=True):
            from telegram_utils import TelegramUtils
            instance = TelegramUtils()
            with pytest.raises(AttributeError):
                instance.run_polling(MagicMock())


class TestTelegramUtilsGetLink:
    def test_get_link(self, mock_application):
        from telegram_utils import TelegramUtils
        instance = TelegramUtils()
        result = instance.get_link("Click here", "start_action")
        assert "Click here" in result
        assert "start_action" in result
        assert "https://t.me/test_bot" in result


class TestTelegramUtilsConversationCancel:
    @pytest.mark.asyncio
    async def test_conversation_cancel_with_message(self):
        from telegram_utils import conversation_cancel, telegram_utils
        update = MagicMock()
        update.message = MagicMock()
        with patch.object(telegram_utils, 'reply', new_callable=AsyncMock) as mock_reply:
            result = await conversation_cancel(update, MagicMock())
            assert result is not None
            mock_reply.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_cancel_no_message(self):
        from telegram_utils import conversation_cancel
        update = MagicMock()
        update.message = None
        result = await conversation_cancel(update, MagicMock())
        assert result is not None
