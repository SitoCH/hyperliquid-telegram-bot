"""Tests for hyperliquid-python-sdk v0.24.0 compatibility.

Verifies that the SDK update from v0.18.0 to v0.24.0 doesn't introduce
regressions and that the fix for the IndexError in Info.__init__ (caused by
non-sequential token indices in the spot_meta response) works correctly.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestInfoInitCompatibility:
    """Tests that Info.__init__ handles spot_meta correctly with SDK v0.24.0.

    Root cause: v0.18.0 indexed spot_meta["tokens"] by position
    (spot_meta["tokens"][base]) which crashed with IndexError when token
    indices in the API response were not sequential starting from 0.
    v0.24.0 builds a dict (token_by_index) and looks up by key instead.
    """

    def test_spot_meta_with_non_sequential_token_indices(self):
        """Verify no IndexError when token indices skip values (the exact fix)."""
        from hyperliquid.info import Info

        spot_meta = {
            "tokens": [
                {"index": 0, "name": "USDC", "szDecimals": 8},
                {"index": 2, "name": "BTC", "szDecimals": 5},   # non-sequential
                {"index": 5, "name": "ETH", "szDecimals": 4},   # non-sequential
            ],
            "universe": [
                {"name": "BTC/USDC", "index": 0, "tokens": [2, 0]},
                {"name": "ETH/USDC", "index": 1, "tokens": [5, 0]},
            ],
        }

        info = Info(
            base_url="https://api.hyperliquid.xyz",
            skip_ws=True,
            spot_meta=spot_meta,
        )

        assert info.coin_to_asset["BTC/USDC"] == 10000
        assert info.coin_to_asset["ETH/USDC"] == 10001
        assert info.name_to_coin["BTC/USDC"] == "BTC/USDC"
        assert info.asset_to_sz_decimals[10000] == 5
        assert info.asset_to_sz_decimals[10001] == 4

    def test_spot_meta_with_sequential_token_indices(self):
        """Verify backward compatibility with sequential token indices."""
        from hyperliquid.info import Info

        spot_meta = {
            "tokens": [
                {"index": 0, "name": "USDC", "szDecimals": 8},
                {"index": 1, "name": "BTC", "szDecimals": 5},
                {"index": 2, "name": "ETH", "szDecimals": 4},
            ],
            "universe": [
                {"name": "BTC/USDC", "index": 0, "tokens": [1, 0]},
                {"name": "ETH/USDC", "index": 1, "tokens": [2, 0]},
            ],
        }

        info = Info(
            base_url="https://api.hyperliquid.xyz",
            skip_ws=True,
            spot_meta=spot_meta,
        )

        assert info.coin_to_asset["BTC/USDC"] == 10000
        assert info.coin_to_asset["ETH/USDC"] == 10001
        assert info.name_to_coin["BTC/USDC"] == "BTC/USDC"
        assert info.asset_to_sz_decimals[10000] == 5
        assert info.asset_to_sz_decimals[10001] == 4

    def test_spot_meta_with_realistic_data(self):
        """Verify with a realistic snapshot of Hyperliquid spot_meta data."""
        from hyperliquid.info import Info

        spot_meta = {
            "tokens": [
                {"index": 0, "name": "USDC", "szDecimals": 8},
                {"index": 1, "name": "kSHIB", "szDecimals": 0},
                {"index": 6, "name": "SOL", "szDecimals": 2},
                {"index": 15, "name": "kPEPE", "szDecimals": 0},
            ],
            "universe": [
                {"name": "kSHIB/USDC", "index": 0, "tokens": [1, 0]},
                {"name": "SOL/USDC", "index": 1, "tokens": [6, 0]},
                {"name": "kPEPE/USDC", "index": 2, "tokens": [15, 0]},
            ],
        }

        info = Info(
            base_url="https://api.hyperliquid.xyz",
            skip_ws=True,
            spot_meta=spot_meta,
        )

        assert "kSHIB/USDC" in info.coin_to_asset
        assert info.coin_to_asset["kSHIB/USDC"] == 10000
        assert info.coin_to_asset["SOL/USDC"] == 10001
        assert info.coin_to_asset["kPEPE/USDC"] == 10002
        # Perp assets also get populated
        assert 0 in info.asset_to_sz_decimals
        assert 1 in info.asset_to_sz_decimals
        assert 2 in info.asset_to_sz_decimals

    def test_name_to_coin_spot_pair_name_registered(self):
        """Verify spot pair names like 'BTC/USDC' are added to name_to_coin."""
        from hyperliquid.info import Info

        spot_meta = {
            "tokens": [
                {"index": 0, "name": "USDC", "szDecimals": 8},
                {"index": 1, "name": "BTC", "szDecimals": 5},
            ],
            "universe": [
                {"name": "BTC/USDC", "index": 0, "tokens": [1, 0]},
            ],
        }

        info = Info(
            base_url="https://api.hyperliquid.xyz",
            skip_ws=True,
            spot_meta=spot_meta,
        )

        # Both the raw coin name and the pair name should be registered
        assert info.name_to_coin["BTC"] == "BTC"
        assert info.name_to_coin["BTC/USDC"] == "BTC/USDC"


class TestSdkVersion:
    """Verify the SDK is at the expected minimum version."""

    def test_sdk_version_is_at_least_0_24_0(self):
        import hyperliquid

        __version__ = getattr(hyperliquid, "__version__", None)
        if __version__ is None:
            # uv-installed git-based packages may lack __version__;
            # verify by checking that Info has the spot_meta fix instead
            from hyperliquid.info import Info
            import inspect

            sig = inspect.signature(Info.__init__)
            source = inspect.getsource(Info.__init__)
            assert "token_by_index" in source, (
                "SDK missing token_by_index dict lookup — "
                "upgrade to v0.24.0+ required"
            )
            return

        major, minor, _ = [int(x) for x in __version__.split(".")]
        assert (major, minor) >= (0, 24), (
            f"Expected hyperliquid-python-sdk >= 0.24.0, got {__version__}"
        )

    def test_info_init_accepts_skip_ws_and_spot_meta(self):
        """Verify Info.__init__ signature includes skip_ws and spot_meta params."""
        import inspect
        from hyperliquid.info import Info

        sig = inspect.signature(Info.__init__)
        params = list(sig.parameters.keys())
        assert "skip_ws" in params, (
            f"Info.__init__ missing 'skip_ws' parameter. Got: {params}"
        )
        assert "spot_meta" in params, (
            f"Info.__init__ missing 'spot_meta' parameter. Got: {params}"
        )


class TestInitWebsocket:
    """Verifies that init_websocket in the bot code is compatible with the SDK update."""

    def test_init_websocket_calls_info_with_skip_ws_false(self):
        """Verify init_websocket creates Info with skip_ws=False."""
        with patch("hyperliquid_utils.utils.Info") as mock_info, \
                patch("hyperliquid_utils.utils.InfoProxy") as mock_infoproxy:

            from hyperliquid_utils.utils import HyperliquidUtils

            instance = HyperliquidUtils()
            instance._info = None  # Reset to force new creation
            instance.init_websocket()

            # Should create Info with skip_ws=False for websocket connection
            call_args = mock_info.call_args
            assert call_args is not None
            args, kwargs = call_args
            assert args[1] is False, (
                f"Expected Info(..., skip_ws=False), got args={args}"
            )

    def test_init_websocket_sets_on_error_and_on_close(self):
        """Verify error/close handlers are attached to the websocket."""
        mock_ws_manager = MagicMock()
        mock_ws_manager.ws = MagicMock()
        mock_info_instance = MagicMock()
        mock_info_instance.ws_manager = mock_ws_manager

        with patch("hyperliquid_utils.utils.Info", return_value=mock_info_instance), \
                patch("hyperliquid_utils.utils.InfoProxy") as mock_infoproxy:

            from hyperliquid_utils.utils import HyperliquidUtils

            instance = HyperliquidUtils()
            instance._info = None
            instance.init_websocket()

            assert mock_ws_manager.ws.on_error is not None
            assert mock_ws_manager.ws.on_close is not None

    def test_name_to_coin_has_coin_and_pair_names(self):
        """Verify both coin names and spot pair names coexist in name_to_coin."""
        from hyperliquid.info import Info

        spot_meta = {
            "tokens": [
                {"index": 0, "name": "USDC", "szDecimals": 8},
                {"index": 1, "name": "BTC", "szDecimals": 5},
                {"index": 3, "name": "ETH", "szDecimals": 4},
            ],
            "universe": [
                {"name": "BTC/USDC", "index": 0, "tokens": [1, 0]},
                {"name": "ETH/USDC", "index": 1, "tokens": [3, 0]},
            ],
        }

        info = Info(
            base_url="https://api.hyperliquid.xyz",
            skip_ws=True,
            spot_meta=spot_meta,
        )

        assert "BTC" in info.name_to_coin
        assert "BTC/USDC" in info.name_to_coin
        assert "ETH" in info.name_to_coin
        assert "ETH/USDC" in info.name_to_coin