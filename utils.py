import os

exchange_enabled = True if os.environ.get("HTB_KEY_FILE") is not None and os.path.isfile(os.environ.get("HTB_KEY_FILE")) else False


OPERATION_CANCELLED = 'Operation cancelled'


def fmt_price(price: float) -> str:
    if price > 1000:
        return format(price, ',.0f')
    if price > 1:
        return format(price, ',.2f')
    return format(price, ',.4f')


def fmt(value: float) -> str:
    return format(value, ',.2f')


def px_round(value: float) -> str:
    return round(float(f"{value:.5g}"), 6)
