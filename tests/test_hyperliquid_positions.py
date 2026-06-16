import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from hyperliquid_positions import (
    PortfolioBalance,
    _calculate_spot_balance,
    _calculate_stacked_balance,
    _get_portfolio_balance,
    _format_portfolio_message,
    _get_token_prices,
    get_positions,
    get_overview,
    spot_positions_messages,
    vault_positions_messages,
)


class TestPortfolioBalance:
    def test_vault_total(self):
        vaults = [
            {"equity": "1000", "vaultAddress": "0xabc"},
            {"equity": "2000", "vaultAddress": "0xdef"},
        ]
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=1500,
            spot_total=3000, stacked_total=1000, vaults=vaults,
            cross_margin_ratio=0.1, cross_account_leverage=2.0,
        )
        assert balance.vault_total == 3000.0

    def test_vault_total_empty(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=1500,
            spot_total=3000, stacked_total=1000, vaults=[],
            cross_margin_ratio=0.1, cross_account_leverage=2.0,
        )
        assert balance.vault_total == 0.0

    def test_total(self):
        vaults = [{"equity": "500", "vaultAddress": "0xabc"}]
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=1500,
            spot_total=3000, stacked_total=1000, vaults=vaults,
            cross_margin_ratio=0.1, cross_account_leverage=2.0,
        )
        assert balance.total == 9500.0

    def test_vault_total_handles_missing_equity(self):
        vaults = [{"vaultAddress": "0xabc"}]
        balance = PortfolioBalance(
            perp_total=0, perp_withdrawable=0, perp_margin_available=0,
            spot_total=0, stacked_total=0, vaults=vaults,
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        assert balance.vault_total == 0.0


class TestCalculateSpotBalance:
    def test_empty_balances(self):
        assert _calculate_spot_balance({"balances": []}, {"BTC": 50000}) == 0.0

    def test_single_balance(self):
        spot_state = {"balances": [{"coin": "BTC", "total": "2.0"}]}
        token_prices = {"BTC": 50000}
        assert _calculate_spot_balance(spot_state, token_prices) == 100000.0

    def test_multiple_balances(self):
        spot_state = {
            "balances": [
                {"coin": "BTC", "total": "1.0"},
                {"coin": "ETH", "total": "10.0"},
            ]
        }
        token_prices = {"BTC": 50000, "ETH": 2000}
        assert _calculate_spot_balance(spot_state, token_prices) == 70000.0

    def test_missing_price_defaults_to_zero(self):
        spot_state = {"balances": [{"coin": "UNKNOWN", "total": "5.0"}]}
        token_prices = {"BTC": 50000}
        assert _calculate_spot_balance(spot_state, token_prices) == 0.0

    def test_missing_balances_key(self):
        assert _calculate_spot_balance({}, {"BTC": 50000}) == 0.0


class TestCalculateStackedBalance:
    def test_all_components(self):
        staking_summary = {
            "delegated": "1000", "undelegated": "500", "totalPendingWithdrawal": "200",
        }
        assert _calculate_stacked_balance(staking_summary, {"HYPE": 10.0}) == 17000.0

    def test_zero_hype_price(self):
        staking_summary = {
            "delegated": "1000", "undelegated": "500", "totalPendingWithdrawal": "200",
        }
        assert _calculate_stacked_balance(staking_summary, {"HYPE": 0.0}) == 0.0

    def test_missing_hype_price(self):
        staking_summary = {
            "delegated": "1000", "undelegated": "500", "totalPendingWithdrawal": "0",
        }
        assert _calculate_stacked_balance(staking_summary, {}) == 0.0


class TestGetTokenPrices:
    def test_basic_prices(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "HYPE"}],
            "universe": [{"tokens": [1, 0], "index": 0}],
        }
        market_data = [{"midPx": "25.0"}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert prices["USDC"] == 1.0
        assert prices["HYPE"] == 25.0

    def test_inverse_quote(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "BTC"}],
            "universe": [{"tokens": [0, 1], "index": 0}],
        }
        market_data = [{"midPx": "0.00002"}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert prices["BTC"] == pytest.approx(50000.0, rel=1e-9)

    def test_skip_no_mid_price(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "HYPE"}],
            "universe": [{"tokens": [1, 0], "index": 0}],
        }
        market_data = [{}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert "HYPE" not in prices

    def test_skip_no_market_data(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "HYPE"}],
            "universe": [{"tokens": [1, 0], "index": 0}],
        }
        market_data = [None]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert "HYPE" not in prices

    def test_infer_quote_via_base(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "BTC"}, {"name": "ETH"}],
            "universe": [
                {"tokens": [1, 0], "index": 0},
                {"tokens": [2, 1], "index": 1},
            ],
        }
        market_data = [{"midPx": "50000"}, {"midPx": "0.05"}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert prices["BTC"] == 50000.0
        assert prices["ETH"] == 2500.0

    def test_infer_base_via_quote(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "BTC"}, {"name": "ETH"}],
            "universe": [
                {"tokens": [1, 0], "index": 0},
                {"tokens": [1, 2], "index": 1},
            ],
        }
        market_data = [{"midPx": "50000"}, {"midPx": "20"}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert prices["BTC"] == 50000.0
        assert prices["ETH"] == 2500.0

    def test_neither_base_nor_quote_known(self):
        metadata = {
            "tokens": [{"name": "USDC"}, {"name": "AAA"}, {"name": "BBB"}],
            "universe": [
                {"tokens": [1, 2], "index": 0},
            ],
        }
        market_data = [{"midPx": "10.0"}]

        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (metadata, market_data)
            prices = _get_token_prices()

        assert "AAA" not in prices
        assert "BBB" not in prices


class TestFormatPortfolioMessage:
    def test_basic_message(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=1500,
            spot_total=0, stacked_total=0, vaults=[],
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "<b>Portfolio:</b>" in lines
        assert "Total balance: 5,000.00 USDC" in lines
        assert "Perps positions" in "\n".join(lines)
        assert "Withdrawable balance: 2,000.00 USDC" in lines
        assert "Available to trade: 1,500.00 USDC" in lines

    def test_with_spot_positions(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=0,
            spot_total=3000, stacked_total=0, vaults=[],
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "<b>Spot positions:</b>" in lines
        assert "Total balance: 3,000.00 USDC" in lines

    def test_with_stacked_positions(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=0,
            spot_total=0, stacked_total=1000, vaults=[],
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "<b>Stacked positions:</b>" in lines
        assert "Total balance: 1,000.00 USDC" in lines

    def test_with_vault_positions(self):
        vaults = [{"equity": "500", "vaultAddress": "0xabc"}]
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=0,
            spot_total=0, stacked_total=0, vaults=vaults,
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "<b>Vault positions:</b>" in lines
        assert "Total balance: 500.00 USDC" in lines

    def test_with_cross_margin_info(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=500,
            spot_total=0, stacked_total=0, vaults=[],
            cross_margin_ratio=0.1, cross_account_leverage=2.5,
        )
        lines = _format_portfolio_message(balance)
        assert "Cross margin ratio: 10.00%" in lines
        assert "Cross account leverage: 2.50x" in lines

    def test_zero_margin_available_hides_available_to_trade(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=0.0,
            spot_total=0, stacked_total=0, vaults=[],
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "Available to trade" not in "\n".join(lines)

    def test_zero_cross_margin_ratio_hides_cross_info(self):
        balance = PortfolioBalance(
            perp_total=5000, perp_withdrawable=2000, perp_margin_available=0,
            spot_total=0, stacked_total=0, vaults=[],
            cross_margin_ratio=0.0, cross_account_leverage=0.0,
        )
        lines = _format_portfolio_message(balance)
        assert "Cross margin ratio" not in "\n".join(lines)
        assert "Cross account leverage" not in "\n".join(lines)

    def test_everything_included(self):
        vaults = [{"equity": "500", "vaultAddress": "0xabc"}]
        balance = PortfolioBalance(
            perp_total=10000, perp_withdrawable=5000, perp_margin_available=2000,
            spot_total=3000, stacked_total=1000, vaults=vaults,
            cross_margin_ratio=0.05, cross_account_leverage=1.5,
        )
        lines = _format_portfolio_message(balance)

        text = "\n".join(lines)
        assert "<b>Portfolio:</b>" in text
        assert "Total balance: 14,500.00 USDC" in text
        assert "<b>Spot positions:</b>" in text
        assert "<b>Stacked positions:</b>" in text
        assert "<b>Vault positions:</b>" in text
        assert "<b>Perps positions:</b>" in text
        assert "Available to trade: 2,000.00 USDC" in text
        assert "Cross margin ratio: 5.00%" in text
        assert "Cross account leverage: 1.50x" in text


def make_user_state(
    positions=None,
    cross_margin_account_value="10000",
    total_margin_used="3000",
    total_ntl_pos="2500",
    margin_account_value="10000",
    withdrawable="5000",
    cross_maintenance_margin="500",
):
    return {
        "assetPositions": positions or [],
        "crossMarginSummary": {
            "accountValue": cross_margin_account_value,
            "totalMarginUsed": total_margin_used,
            "totalNtlPos": total_ntl_pos,
            "totalRawUsd": "7000",
        },
        "marginSummary": {
            "accountValue": margin_account_value,
            "totalMarginUsed": total_margin_used,
            "totalNtlPos": total_ntl_pos,
            "totalRawUsd": "7000",
        },
        "crossMaintenanceMarginUsed": cross_maintenance_margin,
        "withdrawable": withdrawable,
        "time": 1700000000000,
    }


class TestGetPortfolioBalance:
    def setup_mocks(self, mock_hl, perp_state=None, spot_state=None,
                    staking_summary=None, vault_equities=None,
                    spot_meta=None):
        mock_hl.address = "0xtest"
        mock_hl.info.user_state.return_value = perp_state or make_user_state()
        mock_hl.info.spot_user_state.return_value = spot_state or {"balances": []}
        mock_hl.info.user_staking_summary.return_value = staking_summary or {
            "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
        }
        mock_hl.info.user_vault_equities.return_value = vault_equities or []
        if spot_meta is not None:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = spot_meta
        else:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )

    def test_basic_balance(self):
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            self.setup_mocks(mock_hl)
            balance = _get_portfolio_balance()

        assert balance.perp_total == 10000.0
        assert balance.perp_withdrawable == 5000.0
        assert balance.perp_margin_available == 7000.0
        assert balance.spot_total == 0.0
        assert balance.stacked_total == 0.0
        assert balance.vault_total == 0.0
        assert balance.cross_margin_ratio == 0.05
        assert balance.cross_account_leverage == 0.25

    def test_with_spot_and_stacked(self):
        spot_state = {"balances": [{"coin": "USDC", "total": "1000"}]}
        staking_summary = {
            "delegated": "100", "undelegated": "50", "totalPendingWithdrawal": "10"
        }
        spot_meta = (
            {
                "tokens": [{"name": "USDC"}, {"name": "HYPE"}],
                "universe": [{"tokens": [1, 0], "index": 0}],
            },
            [{"midPx": "25.0"}],
        )
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            self.setup_mocks(mock_hl, spot_state=spot_state,
                             staking_summary=staking_summary, spot_meta=spot_meta)
            balance = _get_portfolio_balance()

        assert balance.spot_total == 1000.0
        assert balance.stacked_total == 4000.0

    def test_with_vaults(self):
        vault_equities = [
            {"equity": "5000", "vaultAddress": "0xabc"},
            {"equity": "3000", "vaultAddress": "0xdef"},
        ]
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            self.setup_mocks(mock_hl, vault_equities=vault_equities)
            balance = _get_portfolio_balance()

        assert balance.vault_total == 8000.0
        assert len(balance.vaults) == 2

    def test_zero_cross_margin_account_value_prevents_division_by_zero(self):
        perp_state = make_user_state(cross_margin_account_value="0")
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            self.setup_mocks(mock_hl, perp_state=perp_state)
            balance = _get_portfolio_balance()

        assert balance.cross_margin_ratio == 0.0
        assert balance.cross_account_leverage == 0.0

    def test_perp_margin_available_clamped_to_zero(self):
        perp_state = make_user_state(
            cross_margin_account_value="1000", total_margin_used="2000"
        )
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            self.setup_mocks(mock_hl, perp_state=perp_state)
            balance = _get_portfolio_balance()

        assert balance.perp_margin_available == 0.0


class TestSpotPositionsMessages:
    @pytest.mark.asyncio
    async def test_empty_balances(self):
        result = await spot_positions_messages("grid", {"balances": []})
        assert result == []

    @pytest.mark.asyncio
    async def test_no_balances_key(self):
        result = await spot_positions_messages("grid", {})
        assert result == []

    @pytest.mark.asyncio
    async def test_balances_below_threshold(self):
        spot_state = {"balances": [{"coin": "SHIB", "total": "0.000001", "entryNtl": "0"}]}
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )
            result = await spot_positions_messages("grid", spot_state)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_position_shows_pnl(self):
        spot_state = {"balances": [{"coin": "BTC", "total": "1.0", "entryNtl": "45000"}]}
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {
                    "tokens": [{"name": "USDC"}, {"name": "BTC"}],
                    "universe": [{"tokens": [1, 0], "index": 0}],
                },
                [{"midPx": "50000"}],
            )
            result = await spot_positions_messages("grid", spot_state)

        assert len(result) == 3
        assert "<b>Spot positions:</b>" in result
        assert "50,000.00$" in result[2]

    @pytest.mark.asyncio
    async def test_multiple_positions_sorted_by_value(self):
        spot_state = {
            "balances": [
                {"coin": "ETH", "total": "10.0", "entryNtl": "18000"},
                {"coin": "BTC", "total": "1.0", "entryNtl": "45000"},
            ]
        }
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {
                    "tokens": [{"name": "USDC"}, {"name": "BTC"}, {"name": "ETH"}],
                    "universe": [
                        {"tokens": [1, 0], "index": 0},
                        {"tokens": [2, 0], "index": 1},
                    ],
                },
                [{"midPx": "50000"}, {"midPx": "2000"}],
            )
            result = await spot_positions_messages("grid", spot_state)

        assert len(result) == 3
        assert "50,000.00$" in result[2]
        assert "20,000.00$" in result[2]

    @pytest.mark.asyncio
    async def test_zero_entry_value_pnl_percentage(self):
        spot_state = {"balances": [{"coin": "BTC", "total": "1.0", "entryNtl": "0"}]}
        with patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {
                    "tokens": [{"name": "USDC"}, {"name": "BTC"}],
                    "universe": [{"tokens": [1, 0], "index": 0}],
                },
                [{"midPx": "50000"}],
            )
            result = await spot_positions_messages("grid", spot_state)
        assert len(result) == 3


class TestVaultPositionsMessages:
    def test_empty_vaults(self):
        assert vault_positions_messages("grid", []) == []

    def test_single_vault(self):
        vaults = [{"equity": "5000", "vaultAddress": "0x1234567890abcdef1234567890abcdef12345678"}]
        result = vault_positions_messages("grid", vaults)
        assert len(result) == 3
        assert "<b>Vault positions:</b>" in result
        assert "5,000.00$" in result[2]

    def test_vault_address_truncated(self):
        vaults = [{"equity": "5000", "vaultAddress": "0x1234567890abcdef1234567890abcdef12345678"}]
        result = vault_positions_messages("grid", vaults)
        assert "0x1234" in result[2]
        assert "5678" in result[2]

    def test_multiple_vaults_sorted_by_equity(self):
        vaults = [
            {"equity": "3000", "vaultAddress": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            {"equity": "5000", "vaultAddress": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
        ]
        result = vault_positions_messages("grid", vaults)
        assert len(result) == 3
        assert "5,000.00$" in result[2]


def make_asset_position(coin, szi, entry_px, unrealized_pnl,
                         return_on_equity, margin_used, position_value,
                         leverage_value, cum_funding_since_open):
    return {
        "position": {
            "coin": coin, "szi": szi, "entryPx": entry_px,
            "unrealizedPnl": unrealized_pnl,
            "returnOnEquity": return_on_equity,
            "marginUsed": margin_used,
            "positionValue": position_value,
            "leverage": {"value": leverage_value, "type": "cross"},
            "cumFunding": {"allTime": "0", "sinceOpen": cum_funding_since_open},
        },
    }


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_no_positions_sends_portfolio_only(self):
        update = MagicMock()
        context = MagicMock()

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_tg.get_link = MagicMock(side_effect=lambda text, action: text)
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.return_value = make_user_state(positions=[])
            mock_hl.info.spot_user_state.return_value = {"balances": []}
            mock_hl.info.user_staking_summary.return_value = {
                "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
            }
            mock_hl.info.user_vault_equities.return_value = []
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )
            mock_hl.info.all_mids.return_value = {}

            with patch("hyperliquid_positions._get_token_prices", return_value={"USDC": 1.0}):
                await get_positions(update, context)

            assert mock_tg.reply.called
            message = mock_tg.reply.call_args[0][1]
            assert "<b>Portfolio:</b>" in message
            assert "Total balance: 10,000.00 USDC" in message

    @pytest.mark.asyncio
    async def test_with_perp_positions_sends_position_details(self):
        update = MagicMock()
        context = MagicMock()

        positions = [
            make_asset_position("BTC", "1.5", "40000", "1500", "0.15",
                                "6000", "60000", "10", "50"),
        ]

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_tg.get_link = MagicMock(side_effect=lambda text, action: text)
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.return_value = make_user_state(positions=positions)
            mock_hl.info.spot_user_state.return_value = {"balances": []}
            mock_hl.info.user_staking_summary.return_value = {
                "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
            }
            mock_hl.info.user_vault_equities.return_value = []
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )
            mock_hl.info.all_mids.return_value = {"BTC": "41000"}

            with patch("hyperliquid_positions._get_token_prices", return_value={"USDC": 1.0}):
                await get_positions(update, context)

            assert mock_tg.reply.call_count >= 2
            first_message = mock_tg.reply.call_args_list[0][0][1]
            assert "Unrealized profit" in first_message

    @pytest.mark.asyncio
    async def test_error_handling(self):
        update = MagicMock()
        context = MagicMock()

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.side_effect = Exception("API error")

            await get_positions(update, context)

            assert mock_tg.reply.called
            message = mock_tg.reply.call_args[0][1]
            assert "Error getting positions" in message

    @pytest.mark.asyncio
    async def test_sends_spot_messages_when_present(self):
        update = MagicMock()
        context = MagicMock()

        positions = [
            make_asset_position("BTC", "1.5", "40000", "1500", "0.15",
                                "6000", "60000", "10", "50"),
        ]

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_tg.get_link = MagicMock(side_effect=lambda text, action: text)
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.return_value = make_user_state(positions=positions)
            mock_hl.info.spot_user_state.return_value = {
                "balances": [{"coin": "USDC", "total": "5000", "entryNtl": "5000"}]
            }
            mock_hl.info.user_staking_summary.return_value = {
                "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
            }
            mock_hl.info.user_vault_equities.return_value = []
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [{"name": "USDC"}], "universe": []}, [],
            )
            mock_hl.info.all_mids.return_value = {"BTC": "41000"}

            with patch("hyperliquid_positions._get_token_prices", return_value={"USDC": 1.0}):
                await get_positions(update, context)

            assert mock_tg.reply.call_count >= 2


class TestGetOverview:
    @pytest.mark.asyncio
    async def test_no_positions(self):
        update = MagicMock()
        context = MagicMock()

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_tg.get_link = MagicMock(side_effect=lambda text, action: text)
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.return_value = make_user_state(positions=[])
            mock_hl.info.spot_user_state.return_value = {"balances": []}
            mock_hl.info.user_staking_summary.return_value = {
                "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
            }
            mock_hl.info.user_vault_equities.return_value = []
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )

            with patch("hyperliquid_positions._get_token_prices", return_value={"USDC": 1.0}):
                await get_overview(update, context)

            assert mock_tg.reply.called
            message = mock_tg.reply.call_args[0][1]
            assert "<b>Portfolio:</b>" in message

    @pytest.mark.asyncio
    async def test_with_positions_shows_table(self):
        update = MagicMock()
        context = MagicMock()

        positions = [
            make_asset_position("BTC", "1.5", "40000", "1500", "0.15",
                                "6000", "60000", "10", "50"),
            make_asset_position("ETH", "-10", "2000", "-500", "-0.05",
                                "2000", "20000", "5", "-20"),
        ]

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_tg.get_link = MagicMock(side_effect=lambda text, action: text)
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.return_value = make_user_state(positions=positions)
            mock_hl.info.spot_user_state.return_value = {"balances": []}
            mock_hl.info.user_staking_summary.return_value = {
                "delegated": "0", "undelegated": "0", "totalPendingWithdrawal": "0"
            }
            mock_hl.info.user_vault_equities.return_value = []
            mock_hl.info.spot_meta_and_asset_ctxs.return_value = (
                {"tokens": [], "universe": []}, [],
            )

            with patch("hyperliquid_positions._get_token_prices", return_value={"USDC": 1.0}):
                await get_overview(update, context)

            assert mock_tg.reply.called
            message = mock_tg.reply.call_args[0][1]
            assert "Unrealized profit" in message
            assert "1,000.00 USDC" in message

    @pytest.mark.asyncio
    async def test_error_handling(self):
        update = MagicMock()
        context = MagicMock()

        with patch("hyperliquid_positions.telegram_utils") as mock_tg, \
                patch("hyperliquid_positions.hyperliquid_utils") as mock_hl:
            mock_tg.reply = AsyncMock()
            mock_hl.address = "0xtest"
            mock_hl.info.user_state.side_effect = Exception("API error")

            await get_overview(update, context)

            assert mock_tg.reply.called
            message = mock_tg.reply.call_args[0][1]
            assert "Failed to fetch positions" in message
