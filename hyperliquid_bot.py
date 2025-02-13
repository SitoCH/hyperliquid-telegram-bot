import sys
import datetime
import importlib
import os
import random
import base64
from tzlocal import get_localzone

from logging_utils import logger
from telegram.ext import CommandHandler, ConversationHandler, CallbackQueryHandler, MessageHandler, filters

from technical_analysis.hyperliquid_candles import SELECTING_COIN_FOR_TA, analyze_candles, execute_ta, selected_coin_for_ta
from hyperliquid_orders import get_open_orders
from hyperliquid_trade import SELECTING_COIN, SELECTING_AMOUNT, EXIT_CHOOSING, SELECTING_STOP_LOSS, SELECTING_TAKE_PROFIT, SELECTING_LEVERAGE, enter_long, enter_short, exit_all_positions, selected_amount, selected_coin, exit_position, exit_selected_coin, selected_stop_loss, selected_take_profit, selected_leverage
from hyperliquid_utils import hyperliquid_utils
from hyperliquid_positions import get_positions, get_overview
from hyperliquid_alerts import check_profit_percentage
from hyperliquid_events import on_user_events
from telegram_utils import conversation_cancel, telegram_utils
from utils import exchange_enabled


def main() -> None:
    hyperliquid_utils.init_websocket()

    hyperliquid_utils.info.subscribe(
        {"type": "userEvents", "user": hyperliquid_utils.address}, on_user_events
    )
    
    enter_position_states = {
            SELECTING_COIN: [CallbackQueryHandler(selected_coin)],
            SELECTING_AMOUNT: [CallbackQueryHandler(selected_amount)],
            SELECTING_LEVERAGE: [CallbackQueryHandler(selected_leverage)],
            SELECTING_STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, selected_stop_loss)],
            SELECTING_TAKE_PROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, selected_take_profit)]
        }

    start_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states=enter_position_states, # type: ignore
        fallbacks=[CommandHandler('cancel', conversation_cancel)]
    )
    telegram_utils.add_handler(start_conv_handler)

    telegram_utils.add_handler(CommandHandler(telegram_utils.overview_command, get_overview))
    telegram_utils.add_handler(CommandHandler("positions", get_positions))
    telegram_utils.add_handler(CommandHandler("orders", get_open_orders))
    telegram_utils.add_handler(CommandHandler(telegram_utils.exit_all_command, exit_all_positions))
    ta_conv_handler = ConversationHandler(
        entry_points=[CommandHandler(telegram_utils.ta_command, execute_ta)],
        states={
            SELECTING_COIN_FOR_TA: [CallbackQueryHandler(selected_coin_for_ta)]
        },
        fallbacks=[CommandHandler('cancel', conversation_cancel)]
    )
    telegram_utils.add_handler(ta_conv_handler)

    telegram_utils.run_repeating(
        check_profit_percentage,
        interval=datetime.timedelta(minutes=random.randint(6 * 60 - 10, 6 * 60 + 10))
    )

    if exchange_enabled:
        strategy_name = os.environ.get("HTB_STRATEGY")
        if strategy_name is not None:
            strategy = load_strategy(strategy_name)
            if strategy:
                telegram_utils.run_once(strategy.init_strategy)
            logger.info(f'Exchange order enabled and loaded the strategy "{strategy_name}"')

        current_time = datetime.datetime.now(get_localzone())
        next_hour = current_time.replace(minute=0, second=15, microsecond=0)
        if current_time >= next_hour:
            next_hour += datetime.timedelta(hours=1)
        telegram_utils.run_repeating(analyze_candles, interval=datetime.timedelta(hours=1.0), first=next_hour)
        # telegram_utils.run_once(analyze_candles)

        sell_conv_handler = ConversationHandler(
            entry_points=[CommandHandler('exit', exit_position)],
            states={
                EXIT_CHOOSING: [CallbackQueryHandler(exit_selected_coin)]
            },
            fallbacks=[CommandHandler('cancel', conversation_cancel)]
        )
        telegram_utils.add_handler(sell_conv_handler)

        enter_long_conv_handler = ConversationHandler(
            entry_points=[CommandHandler('long', enter_long)],
            states=enter_position_states, # type: ignore
            fallbacks=[CommandHandler('cancel', conversation_cancel)]
        )
        telegram_utils.add_handler(enter_long_conv_handler)

        enter_short_conv_handler = ConversationHandler(
            entry_points=[CommandHandler('short', enter_short)],
            states=enter_position_states, # type: ignore
            fallbacks=[CommandHandler('cancel', conversation_cancel)]
        )
        telegram_utils.add_handler(enter_short_conv_handler)

    else:
        logger.info('Exchange orders disabled')

    telegram_utils.run_polling(shutdown)
    #await telegram_utils.stop()

def load_strategy(strategy_name):
    module_name = f"strategies.{strategy_name}.{strategy_name}"
    try:
        strategy_module = importlib.import_module(module_name)
        strategy_class = getattr(strategy_module, strategy_name.title().replace('_', ''))
        return strategy_class()
    except (ModuleNotFoundError, AttributeError) as e:
        logger.critical(e, exc_info=True)
        return None

async def start(update, context):
    if context.args:
        raw_param = context.args[0]
        if raw_param.startswith("TA_"):
            context.args = [raw_param[3:]]
            await update.message.delete()
            await execute_ta(update, context)    
        elif raw_param.startswith("TRD_"):
            await update.message.delete()
            decoded_params = base64.b64decode(raw_param[3:]).decode('utf-8')
            side, coin, sl, tp = decoded_params.split('_')
            context.args = [coin, sl, tp]
            if side == 'L':
                return await enter_long(update, context)
            elif side == 'S':
                return await enter_short(update, context)
    else:
        await telegram_utils.reply(update, "Welcome! Click the button below to check the account's positions.")
    return ConversationHandler.END

async def shutdown(application):
    logger.info("Shutting down Hyperliquid Telegram bot...")
    os._exit(0)

if __name__ == "__main__":
    main()
