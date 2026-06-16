---
name: hyperliquid-telegram-bot
description: >
  Technical reference for the hyperliquid-telegram-bot — a Python/Telegram bot monitoring Hyperliquid perpetual
  futures with real-time fills, Wyckoff/LLM technical analysis, automated trading strategies, and portfolio
  statistics. Use when working on any part of this codebase: adding features, fixing bugs, writing tests, or
  understanding the architecture.
---

# hyperliquid-telegram-bot

Telegram bot for Hyperliquid perpetual futures monitoring, analysis, and trading. Built with Python 3.10,
`python-telegram-bot`, `hyperliquid-python-sdk`, Wyckoff technical analysis, optional LLM analysis via LiteLLM,
and dynamic trading strategies.

## Architecture

```
Telegram User
    |
    v
telegram_utils.py  (python-telegram-bot Application, polling loop)
    |
    +-- hyperliquid_bot.py  [Entry point: wires handlers, websocket, periodic jobs, strategies]
    |       |
    |       +-- Command handlers: /start, /positions, /orders, /ta, /long, /short, /exit, /overview, /stats
    |       +-- Periodic jobs: PnL alerts (~6h randomized), stale position checks (hourly), hourly TA
    |       +-- Strategy loader: importlib.import_module(f"strategies.{name}.{name}")
    |
    +-- hyperliquid_utils/utils.py  [HyperliquidUtils singleton]
    |       +-- Info (read API, wrapped in InfoProxy for rate-limit tracking)
    |       +-- Exchange (write API, requires HTB_KEY_FILE)
    |       +-- WebSocket (userEvents subscription for real-time fill notifications)
    |
    +-- technical_analysis/
    |       +-- hyperliquid_candles.py  [Mode router: HTB_ANALYSIS_MODE → WyckoffAnalyzer | LLMAnalyzer]
    |       +-- data_processor.py  [OHLCV → indicators: BB, ATR, MACD, RSI, SuperTrend, ADX, Fib, Pivot, VWAP]
    |       +-- wyckoff/  [Wyckoff phase detection, MTF alignment, chart generation]
    |       +-- llm/  [LiteLLM-powered analysis with structured JSON prompts and response parsing]
    |
    +-- strategies/  [Dynamically loaded strategy modules]
            +-- base_strategy/  [ABC with analyze(), rebalance(), drift monitoring]
            +-- etf_strategy/  [Top N market-cap coins, category filtering, yearly perf filter]
            +-- fixed_token_strategy/  [Fixed token set with configurable leverage]
            +-- alpha_g_strategy/  [Multi-day surge/crash detection with ATR thresholds]
```

### Key Design Decisions

- **Singletons**: `hyperliquid_utils` (HyperliquidUtils) and `telegram_utils` (TelegramUtils) are module-level singletons. Import them directly.
- **Conversation handlers**: Telegram's `ConversationHandler` with named state constants for multi-step trading flows. See `hyperliquid_trade.py` for pattern (`SELECTING_COIN`, `SELECTING_AMOUNT`, etc.).
- **Strategy pattern**: Abstract `BaseStrategy` in `strategies/base_strategy/`, concrete strategies in subdirectories, loaded dynamically by `importlib` in `hyperliquid_bot.py:load_strategy()`.
- **Env-driven config**: Everything comes from `HTB_*` env vars. Defaults are safe for real-money accounts (analysis-only unless explicitly configured to trade).
- **Rate limiting**: `InfoProxy` wraps all Hyperliquid API reads and tracks weight consumption. Max 1000 weight per minute.
- **Caching**: `candles_cache.py` and `funding_rates_cache.py` deduplicate API calls with TTL.

## Entry Point: `hyperliquid_bot.py`

`main()` sequence:
1. `hyperliquid_utils.init_websocket()` — upgrades Info to WebSocket connection
2. Subscribes to `userEvents` → `hyperliquid_events.on_user_events`
3. Registers `/start` conversation handler (with deep-link param support: `TA_<coin>`, `TRD_<side>_<coin>_<sl>_<tp>`)
4. Registers `/positions`, `/orders`, `/overview`, `/stats` read-only handlers
5. Registers `/ta` conversation handler
6. Schedules periodic jobs:
   - `check_profit_percentage` every ~6h (randomized ±20m)
   - `check_positions_to_close` every 1h (if `HTB_MONITOR_STALE_POSITIONS=True`)
7. If `exchange_enabled` (HTB_KEY_FILE exists):
   - Loads strategy via `load_strategy(strategy_name)` from `HTB_STRATEGY`
   - Schedules `analyze_candles` every 1h
   - Registers `/long`, `/short`, `/exit` conversation handlers
8. `telegram_utils.run_polling(shutdown)` — starts polling loop

## Project Structure Reference

### Top-Level Modules

| File | Purpose |
|------|---------|
| `hyperliquid_bot.py` | Entry point; wires all handlers and periodic jobs |
| `hyperliquid_trade.py` | `/long`, `/short`, `/exit` conversation handlers; places market + trigger orders |
| `hyperliquid_positions.py` | `/positions`, `/overview` — fetches perpetual/spot/staking/vault balances |
| `hyperliquid_orders.py` | `/orders` — groups open orders by coin with trigger prices |
| `hyperliquid_events.py` | WebSocket `userEvents` handler → formats fills and sends to Telegram |
| `hyperliquid_alerts.py` | Periodic PnL alerts and stale position monitoring/auto-closing |
| `telegram_utils.py` | `TelegramUtils` singleton: message sending, keyboards, job scheduling, polling |
| `logging_utils.py` | Logger configuration (INFO level, suppresses httpx/apscheduler) |
| `utils.py` | `exchange_enabled` flag, `fmt_price()`, `fmt()`, `log_execution_time` decorator |

### `hyperliquid_utils/`

| File | Purpose |
|------|---------|
| `utils.py` | `HyperliquidUtils` class: Info proxy, Exchange init (from key file), WebSocket lifecycle, coin listings |
| `hyperliquid_info_proxy.py` | `InfoProxy` — wraps `Info` methods, tracks API weight per endpoint |
| `hyperliquid_ratelimiter.py` | Rate limiter (1000 weight/minute), `get_next_available_time()` |

### `technical_analysis/`

| File | Purpose |
|------|---------|
| `hyperliquid_candles.py` | Mode router; `analyze_candles()` → selects coins, rate-limits, dispatches to Wyckoff or LLM |
| `data_processor.py` | OHLCV pipeline: `prepare_dataframe()`, `remove_partial_candle()`, `apply_indicators()` |
| `candles_utils.py` | Coin selection: `get_coins_to_analyze()` from env vars |
| `candles_cache.py` | Candle data cache with TTL deduplication |
| `funding_rates_cache.py` | Historical funding rate cache |

#### Wyckoff (`wyckoff/`)

| File | Purpose |
|------|---------|
| `wyckoff_analyzer.py` | Orchestrator: fetches 5 timeframes, detects phases, generates charts, sends messages |
| `wyckoff_types.py` | Core enums: `Timeframe`, `WyckoffState`, `TimeframeSettings`, `SignificantLevelsData` |
| `wyckoff.py` | `detect_wyckoff_phase()` — phase detection from indicator data |
| `wyckoff_chart.py` | Chart generation with mplfinance (Heikin Ashi toggle support) |
| `wyckoff_description.py` | Human-readable Wyckoff state descriptions |
| `significant_levels.py` | Support/resistance from swing points |
| `phase_hysteresis.py` | State smoothing to prevent rapid phase flips |
| `mtf/` | Multi-timeframe aggregation: `mtf_alignment.py`, `mtf_direction.py`, `trade_suggestion.py` |

#### LLM (`llm/`)

| File | Purpose |
|------|---------|
| `llm_analyzer.py` | Orchestrator: data → filter → LLM call → parse JSON response |
| `prompt_generator.py` | Deterministic, schema-guided prompts from OHLCV + funding data |
| `litellm_client.py` | LiteLLM wrapper for model-agnostic API calls |
| `analysis_filter.py` | Pre-LLM filter: decides if analysis is worth running |
| `message_formatter.py` | Formats `LLMAnalysisResult` for Telegram |
| `llm_analysis_result.py` | `LLMAnalysisResult`, `Signal` enum, `RiskLevel` enum |

### `strategies/`

| Module | Base Class | Commands | Description |
|--------|-----------|----------|-------------|
| `etf_strategy/` | `BaseStrategy` | `/rebalance`, `/analyze` | Top N coins by market cap with category/yearly perf filter |
| `fixed_token_strategy/` | `BaseStrategy` | `/rebalance`, `/analyze` | Fixed token set with leverage and perf filter |
| `alpha_g_strategy/` | Standalone | `/analyze` | Multi-day surge/crash detection with ATR thresholds, BB re-entry |
| `default_strategy/` | `BaseStrategy` | — | Minimal stub |

Strategy naming convention for `load_strategy()`: module path is `strategies.{name}.{name}`, class name is `name.title().replace('_', '')`. Example: `etf_strategy` → class `EtfStrategy` in `strategies/etf_strategy/etf_strategy.py`.

### `bot_statistics/`

| File | Purpose |
|------|---------|
| `hyperliquid_stats.py` | `/stats`: win/loss, PnL, fees, benchmarks vs BTC HODL and S&P 500 |
| `btc_price_utils.py` | Historical BTC price fetcher |
| `sp500_price_utils.py` | S&P 500 price fetcher (Alpha Vantage API) |

## Environment Configuration

All `HTB_*` env vars have safe defaults for real-money accounts. The bot is read-only by default — trading requires `HTB_KEY_FILE`.

### Required

| Variable | Purpose |
|----------|---------|
| `HTB_TOKEN` | Telegram bot token from @BotFather |
| `HTB_CHAT_ID` | Telegram chat/user ID |
| `HTB_USER_WALLET` | Hyperliquid wallet address to monitor |

### Analysis Mode

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `HTB_ANALYSIS_MODE` | `wyckoff` or `llm` | `wyckoff` | Switches between Wyckoff and LLM TA pipelines |
| `HTB_USE_HEIKIN_ASHI` | `True` or `False` | `False` | Heikin Ashi candles for charts; calculations always use real OHLCV |
| `HTB_LLM_MAIN_MODEL` | model name | `unknown` | Main LLM for analysis; must be set when `HTB_ANALYSIS_MODE=llm` |
| `HTB_LLM_FAST_MODEL` | model name | `unknown` | Fast model for filter; must be set when `HTB_ANALYSIS_MODE=llm` |
| `HTB_ALWAYS_RUN_LLM_FILTER` | `True` or `False` | `False` | Force LLM filter before analysis |
| `HTB_TRADE_MIN_RR` | float | `1.4` | Min risk/reward ratio for LLM trade proposals |

### TA Coin Selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HTB_COINS_TO_ANALYZE` | comma-separated | None | Specific coins for hourly TA |
| `HTB_TOP_COINS_TO_ANALYZE` | int | None | Number of top coins by volume to analyze |
| `HTB_TOP_COINS_OFFSET` | int | `0` | Skip N top coins before selecting |
| `HTB_ANALYZE_COINS_WITH_OPEN_ORDERS` | bool | `False` | Include coins with open orders |
| `HTB_COINS_ANALYSIS_MIN_CONFIDENCE` | float | `0.65` | Min confidence to send analysis |

### Trading & Risk

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HTB_KEY_FILE` | path | None | Private key file path (enables trading) |
| `HTB_USE_ISOLATED_LEVERAGE` | bool | `True` | Isolated vs cross margin |
| `HTB_SKIP_SL_TP_PROMPT` | bool | `False` | Skip SL/TP prompt when opening positions |
| `HTB_MONITOR_STALE_POSITIONS` | bool | `False` | Monitor positions >4h with positive PnL |
| `HTB_STALE_POSITION_MIN_PNL` | float | `5` | Min PnL% to consider stale |
| `HTB_AUTO_CLOSE_STALE_POSITIONS` | bool | `False` | Auto-close stale positions instead of alerting |

### Benchmarking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HTB_ALPHAVANTAGE_API_KEY` | string | None | Required for S&P 500 benchmark comparison |

### Strategy-Specific Config

See `strategies/<name>/` modules for `HTB_<STRATEGY>_*` env vars. Key patterns:
- ETF: `HTB_ETF_STRATEGY_COINS_NUMBER`, `HTB_ETF_STRATEGY_COINS_OFFSET`, `HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE`, `HTB_ETF_STRATEGY_LEVERAGE`, `HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS`, `HTB_ETF_STRATEGY_CATEGORY`
- Fixed Token: `HTB_FIXED_TOKEN_STRATEGY_TOKENS`, `HTB_FIXED_TOKEN_STRATEGY_LEVERAGE`, `HTB_FIXED_TOKEN_STRATEGY_MIN_YEARLY_PERFORMANCE`
- AlphaG: `HTB_ALPHA_G_STRATEGY_LOOKBACK_DAYS`, `HTB_ALPHA_G_STRATEGY_THRESHOLD_PCT`

## Telegram Conversation Flow

Multi-step trades use `ConversationHandler` with state constants defined in each module:

```python
# From hyperliquid_trade.py — enter position flow
states = {
    SELECTING_COIN: [CallbackQueryHandler(selected_coin)],
    SELECTING_AMOUNT: [CallbackQueryHandler(selected_amount)],
    SELECTING_LEVERAGE: [CallbackQueryHandler(selected_leverage)],
    SELECTING_STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, selected_stop_loss)],
    SELECTING_TAKE_PROFIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, selected_take_profit)]
}
```

Deep-link params from `/start` allow direct entry:
- `TA_<coin>` → skip to TA flow
- `TRD_<base64(side_coin_sl_tp)>` → skip to trade flow

## Testing

- **Framework**: pytest with pytest-asyncio (auto mode), pytest-cov, pytest-mock
- **Run**: `uv run pytest -q` from repo root (or `uv run pytest --cov` for coverage)
- **Test paths**: `tests/` mirrors source structure
- **Fixtures**: `conftest.py` sets `HTB_USER_WALLET`, provides `mock_hyperliquid_info` and `mock_telegram_utils`
- **Pattern**: Mock Hyperliquid API and Telegram utils; test logic in isolation without real connections

## Development Conventions

- **Types**: Use `wyckoff_types.py` `Timeframe` and `TimeframeSettings` instead of hardcoded numeric values
- **Timeframes**: MTF groups in `wyckoff_types.py`: `SHORT_TERM_TIMEFRAMES`, `INTERMEDIATE_TIMEFRAMES`, `CONTEXT_TIMEFRAMES`
- **Caching**: Route candle/funding fetches through `candles_cache.py` / `funding_rates_cache.py`
- **Rate limits**: API calls must go through `hyperliquid_rate_limiter`
- **Logging**: Use `logging_utils.logger`; `info` for normal ops, `error`/`critical` with `exc_info=True` for failures
- **Telegram UX**: Use `telegram_utils` helpers for replying; keep messages concise and HTML-safe
- **Analysis logic**: Keep deterministic (no randomness in TA); non-determinism breaks tests and LLM prompts

## Extension Points

### New TA features
1. Add indicator calculations in `technical_analysis/data_processor.py:apply_indicators()`
2. Expose via `TimeframeSettings` config in `wyckoff_types.py`
3. Reuse existing `WyckoffState`, MTF groupings, description generators

### New LLM signals
1. Extend `LLMPromptGenerator` in `technical_analysis/llm/prompt_generator.py`
2. Keep JSON response schema strict and backward-compatible
3. Prefer derived metrics from existing data over new external dependencies

### New strategies
1. Create `strategies/<name>/<name>.py` with class `Name` matching `name.title().replace('_', '')`
2. Extend `BaseStrategy` from `strategies/base_strategy/` for ETF/fixed-token style strategies, or standalone for custom (like AlphaG)
3. Follow env-driven config pattern with `HTB_<STRATEGY>_*` vars
4. Register command handlers using `telegram_utils.add_handler()` from `init_strategy()`
