# Copilot Instructions for `hyperliquid-telegram-bot`

These rules guide AI coding agents working in this repo. Focus on keeping the bot stable, deterministic, and safe with real Hyperliquid accounts.

## Big Picture
- This is a **Telegram bot for Hyperliquid perpetuals**: monitoring positions/orders, sending alerts, and optionally executing trades.
- Entry point: `hyperliquid_bot.py` wires Telegram handlers, Hyperliquid websocket subscriptions, periodic jobs, and optional strategies.
- Technical analysis is in `technical_analysis/` with two analysis modes:
  - **Wyckoff mode** (`HTB_ANALYSIS_MODE=wyckoff`, default): indicator-heavy, multi-timeframe Wyckoff analysis and chart generation.
  - **LLM mode** (`HTB_ANALYSIS_MODE=llm`): uses `technical_analysis/llm/` to build prompts and call LLMs via LiteLLM.
- Hyperliquid integration lives in `hyperliquid_utils/` and top-level `hyperliquid_*.py` files; trading strategies are under `strategies/`.

## Run, Build, and Test
- Local tests (preferred): from repo root run:
  - `uv run pytest -q` (matches CI via `pyproject.toml` and `.github/workflows/test.yml`).
- Docker image build & run (used in README examples and CI):
  - Build: `docker build -t sito/hyperliquid-telegram-bot:local .`
  - Run (minimal): mount env vars `HTB_TOKEN`, `HTB_CHAT_ID`, `HTB_USER_WALLET`.
- Do **not** assume tests are run via `python -m pytest`; default is `uv run pytest`.

## Key Modules and Data Flow
- `hyperliquid_bot.py`
  - Initializes Hyperliquid websocket (`hyperliquid_utils.init_websocket()`), subscribes to `userEvents`, and starts the Telegram polling loop.
  - Registers command handlers routed through `telegram_utils`:
    - `/start` → `start` (decodes deep-link params for TA/TRADE shortcuts).
    - `/positions`, `/orders`, overview/stats commands → read-only queries.
    - `/ta` → technical analysis flow via `technical_analysis.hyperliquid_candles`.
    - `/long`, `/short`, `/exit` → trading conversations using `hyperliquid_trade` & `hyperliquid_orders`.
  - Schedules periodic jobs for:
    - PnL/alerts: `hyperliquid_alerts.check_profit_percentage` and optionally `check_positions_to_close`.
    - Hourly TA on multiple coins: `technical_analysis.hyperliquid_candles.analyze_candles`.
- `technical_analysis/hyperliquid_candles.py`
  - Controls TA mode via `AnalysisMode` and `HTB_ANALYSIS_MODE`.
  - `analyze_candles` → selects coins (`candles_utils.get_coins_to_analyze`), applies rate limiting (`hyperliquid_rate_limiter`), and dispatches one coin at a time.
  - Delegates to `WyckoffAnalyzer` or `LLMAnalyzer` which produce messages/charts via `telegram_utils`.
- Wyckoff stack (`technical_analysis/wyckoff/`)
  - `wyckoff_types.py`: core enums, `WyckoffState`, timeframe definitions and settings used across TA (including MTF analysis and significant levels).
  - `data_processor.py`: prepares OHLCV data and applies indicators based on `Timeframe` and `TimeframeSettings`.
  - `significant_levels.py`, `wyckoff_description.py`, `mtf/*`: compute levels, multi-timeframe states, and human-readable descriptions.
- LLM stack (`technical_analysis/llm/`)
  - `prompt_generator.py`: constructs deterministic, schema-guided prompts from OHLCV + funding data.
  - `llm_analyzer.py`: orchestrates data gathering, calls to LiteLLM, and post-processing of model responses.
- Strategies (`strategies/*`)
  - `base_strategy` defines common configuration; specific strategies (e.g. `etf_strategy`, `fixed_token_strategy`, `alpha_g_strategy`) live in matching subfolders and are loaded dynamically via `HTB_STRATEGY`.

## Project-Specific Conventions
- **Environment-driven behavior**
  - All runtime configuration (Telegram, Hyperliquid, TA mode, strategies, LLM models) comes from environment variables; mirror the naming/usage pattern in `README.md` when adding new config.
  - Default behaviors should be **safe** for real-money accounts (e.g. analysis-only unless explicitly configured to trade).
- **Telegram UX**
  - Use `telegram_utils` helpers for replying, keyboards, and scheduling; keep bot responses concise and HTML-safe.
  - Conversation handlers use clearly named state constants (e.g. `SELECTING_COIN_FOR_TA`, `EXIT_CHOOSING`); re-use this style when adding new flows.
- **Timeframes and TA settings**
  - Always reference `Timeframe` and `TimeframeSettings` from `wyckoff_types.py` rather than hardcoding numeric values.
  - Multi-timeframe logic is grouped into `SHORT_TERM_TIMEFRAMES`, `INTERMEDIATE_TIMEFRAMES`, `CONTEXT_TIMEFRAMES`; respect these groupings when aggregating signals.
- **Caching & rate limiting**
  - Candle and funding data should go through `technical_analysis/candles_cache.py` and `funding_rates_cache.py` instead of hitting Hyperliquid directly.
  - API calls must respect `hyperliquid_rate_limiter` to prevent rate-limit issues.
- **Logging & errors**
  - Use the shared `logging_utils.logger`; log at `info` for normal operations and `error`/`critical` with `exc_info=True` for failures.
  - For failures in interactive flows, also notify users via `telegram_utils` (see `analyze_candles_for_coin`).

## Testing and Safety When Modifying Code
- Keep existing tests under `tests/` passing; when adding TA/strategy logic, add or extend tests beside the relevant module where patterns already exist.
- Avoid introducing non-determinism in analysis logic (randomness, time-sensitive branching without explicit inputs), as this would make tests brittle and LLM prompts harder to reason about.
- When touching trading code (`hyperliquid_trade.py`, `hyperliquid_orders.py`, strategies), **do not** change default order sizes, leverage, or risk parameters without explicit intent.

## How Agents Should Extend the Project
- For new TA features:
  - Add indicator calculations in `technical_analysis/data_processor.py` and expose them via `TimeframeSettings` / DF columns.
  - Re-use `WyckoffState`, MTF groupings, and existing description generators rather than inventing parallel abstractions.
- For new LLM signals:
  - Extend `LLMPromptGenerator` in `technical_analysis/llm/prompt_generator.py`, keeping the JSON response schema strict and backward compatible.
  - Prefer enriching prompts with **derived metrics** from existing data rather than introducing new external dependencies.
- For new strategies:
  - Create `strategies/<name>/<name>.py` with a class name matching `name.title().replace('_', '')` so `load_strategy` in `hyperliquid_bot.py` can import it.
  - Base it on `BaseStrategyConfig`/patterns in existing strategy modules and respect environment-driven configuration.

If any part of this guidance feels incomplete or unclear for your workflow, tell me which area (TA, LLM, strategies, or deployment), and I’ll refine these instructions further.