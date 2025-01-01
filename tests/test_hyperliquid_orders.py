import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from collections import defaultdict
from hyperliquid_orders import (
    get_orders_from_hyperliquid,
    get_sl_tp_orders,
    format_orders,
    insert_order
)

@pytest.fixture
def mock_orders():
    return [
        {"coin": "BTC", "orderType": "Stop Market", "triggerPx": "25000", "sz": "1.0"},
        {"coin": "BTC", "orderType": "Take Profit Market", "triggerPx": "35000", "sz": "0.5"},
        {"coin": "ETH", "orderType": "Stop Market", "triggerPx": "1800", "sz": "10.0"}
    ]

@pytest.fixture
def mock_hyperliquid_utils():
    mock = MagicMock()
    mock.info.frontend_open_orders = MagicMock(return_value=[
        {"coin": "BTC", "orderType": "Stop Market", "triggerPx": "25000", "sz": "1.0"},
        {"coin": "BTC", "orderType": "Take Profit Market", "triggerPx": "35000", "sz": "0.5"}
    ])
    mock.address = "test_address"
    return mock

@pytest.mark.asyncio
async def test_get_orders_from_hyperliquid(mock_hyperliquid_utils):
    with patch('hyperliquid_orders.hyperliquid_utils', mock_hyperliquid_utils):
        result = await get_orders_from_hyperliquid()
        assert "BTC" in result
        assert "Stop Market" in result["BTC"]
        assert "Take Profit Market" in result["BTC"]
        assert len(result["BTC"]["Stop Market"]) == 1
        assert len(result["BTC"]["Take Profit Market"]) == 1

def test_get_sl_tp_orders():
    order_types = {
        "Stop Market": [
            {"triggerPx": "25000"},
            {"triggerPx": "24000"}
        ],
        "Take Profit Market": [
            {"triggerPx": "35000"},
            {"triggerPx": "36000"}
        ]
    }
    
    # Test long position
    sl_orders, tp_orders = get_sl_tp_orders(order_types, is_long=True)
    assert float(sl_orders[0]["triggerPx"]) > float(sl_orders[1]["triggerPx"])
    assert float(tp_orders[0]["triggerPx"]) > float(tp_orders[1]["triggerPx"])
    
    # Test short position
    sl_orders, tp_orders = get_sl_tp_orders(order_types, is_long=False)
    assert float(sl_orders[0]["triggerPx"]) < float(sl_orders[1]["triggerPx"])
    assert float(tp_orders[0]["triggerPx"]) < float(tp_orders[1]["triggerPx"])

def test_format_orders():
    raw_orders = [
        {"sz": "1.0", "triggerPx": "30000"},
        {"sz": "0.5", "triggerPx": "31000"}
    ]
    mid = 29000
    percentage_format = lambda triggerPx, mid: (float(triggerPx) / mid - 1) * 100
    
    result = format_orders(raw_orders, mid, percentage_format)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert result[0][0] == "1.0"
    assert result[0][1] == "30000"
    assert "%" in result[0][2]
