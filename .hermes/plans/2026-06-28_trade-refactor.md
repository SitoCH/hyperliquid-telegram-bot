# Trade Execution Refactor — Multi-DEX + Isolated Leverage

## Goal
Refactor `trade_execution.py` / `trade_conversation.py` to:
1. Route trades to the correct builder-deployed perp DEX based on coin prefix (e.g. `xyz:SPCX` → XYZ DEX)
2. Handle `onlyIsolated` tokens gracefully (don't force cross margin)
3. Fix exit/close to work across all DEXes

## Files
- `hyperliquid_utils/utils.py` — add DEX-aware exchange caching  
- `trade_execution.py` — core trading functions  
- `trade_conversation.py` — exit/close position flows  
- `tests/test_hyperliquid_trade.py` — new test classes  
- `tests/conftest.py` — new fixtures if needed

## Design Decisions

### DEX detection from coin name
Coins from builder DEXes are prefixed with `{dex_name}:` (e.g. `xyz:AAPL`, `xyz:SPCX`). Default DEX coins have no prefix. A helper `_resolve_dex(coin: str) -> tuple[str, str]` returns `(dex_name, bare_coin)`.

### Exchange caching
`HyperliquidUtils` caches Exchange instances per dex name. `get_exchange(dex_name=None)` creates/returns cached Exchange configured with `perp_dexs=[dex_name]` for the matching DEX.

### Isolated-only detection
The `meta(dex=...)` response includes `onlyIsolated: bool` per asset. Before setting leverage, check this flag: if `onlyIsolated=True`, always use isolated mode regardless of `HTB_USE_ISOLATED_LEVERAGE`.

### Close/exit across DEXes
- `close_all_positions_core()` iterates not just the default DEX's assetPositions but all configured extra DEXes
- `exit_position()` shows positions from all DEXes
- `exit_selected_coin()` routes the `market_close` to the correct dex's Exchange

## Implementation Plan

### Task 1: Add DEX resolution + exchange caching to HyperliquidUtils

**Files:** `hyperliquid_utils/utils.py`

- Add `_exchange_cache: dict[str, Exchange]` to `__init__`
- Add `dex_supported(coin: str) -> Optional[str]` — returns dex name if coin has a dex prefix, None otherwise
- Modify `get_exchange(dex: str = "")` — cache per dex, init with `perp_dexs=[dex]`

**Tests:**
- `test_resolve_dex_default_coin` — `dex_supported("BTC")` → None
- `test_resolve_dex_xyz_coin` — `dex_supported("xyz:SPCX")` → "xyz"
- `test_resolve_dex_vntl_coin` — `dex_supported("vntl:SPACEX")` → "vntl"
- `test_get_exchange_caches` — two calls with same dex return same object
- `test_get_exchange_different_dexes` — different dexes return different objects

### Task 2: Refactor open_order for DEX awareness

**Files:** `trade_execution.py`

- Extract `_determine_leverage_mode(coin: str, dex: str) -> bool` — checks `onlyIsolated` in meta, returns `is_cross`
- Modify `open_order()` to:
  1. Detect DEX from coin name
  2. Get the correct Exchange instance
  3. Use per-DEX `meta(dex=dex)` to check `onlyIsolated`
  4. Pass correct `is_cross` to `update_leverage`
  5. Use per-DEX `get_sz_decimals(dex=dex)` if needed

**Tests:**
- `test_determine_leverage_default_dex_cross` — BTC without onlyIsolated → is_cross=True
- `test_determine_leverage_xyz_dex_cross` — xyz:AAPL without onlyIsolated → is_cross=True
- `test_determine_leverage_isolated_only` — token with onlyIsolated=True → is_cross=False
- `test_open_order_xyz_dex` — full flow with xyz:SPCX
- `test_open_order_handles_missing_sz_decimals`

### Task 3: Refactor close/exit for DEX awareness

**Files:** `trade_execution.py`, `trade_conversation.py`

- `close_all_positions_core()`:
  - Query default DEX + all extra DEXes for assetPositions
  - Close each position using the correct dex's Exchange
- `exit_position()`:
  - Include positions from all DEXes in the selection list
- `exit_selected_coin()`:
  - Route `market_close(coin)` to the correct dex's Exchange

**Tests:**
- `test_close_all_with_xyz_positions`
- `test_close_all_positions_from_multiple_dexes`
- `test_exit_selected_coin_xyz`
- `test_exit_position_shows_xyz_coins`
