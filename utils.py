import os

exchange_enabled = True if os.environ.get("HYPERLIQUID_TELEGRAM_BOT_KEY_FILE") is not None and os.path.isfile(os.environ.get("HYPERLIQUID_TELEGRAM_BOT_KEY_FILE")) else False


OPERATION_CANCELLED = 'Operation cancelled'


def fmt(value: float) -> str:
    return format(value, ',.2f')


def px_round(value: float) -> str:
    return round(float(f"{value:.5g}"), 6)
