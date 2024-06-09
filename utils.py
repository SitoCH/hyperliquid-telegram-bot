import os

exchange_enabled = True if os.environ.get("HYPERLIQUID_TELEGRAM_BOT_KEY_FILE") is not None else False


OPERATION_CANCELLED = 'Operation cancelled'
