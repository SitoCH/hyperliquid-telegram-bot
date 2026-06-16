# Contributing

Contributions are welcome! This project follows a standard fork-and-PR workflow. Please read the guidelines below before submitting changes.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/hyperliquid-telegram-bot.git`
3. Set up the development environment:
   ```bash
   cd hyperliquid-telegram-bot
   uv sync --frozen
   ```
4. Run tests to verify your setup: `uv run pytest -q`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/YourFeatureName`
2. Make your changes
3. Run the tests: `uv run pytest -q`
4. Keep code style consistent with the existing codebase (see `.flake8` for linting rules)
5. Commit with clear, descriptive messages
6. Push to your fork: `git push origin feature/YourFeatureName`
7. Open a Pull Request against the `main` branch

## Code Conventions

- **Configuration**: All runtime configuration must use `HTB_*` environment variables. Defaults must be safe for real-money accounts (analysis-only unless explicitly configured to trade).
- **Telegram UX**: Use `telegram_utils` helpers for replying, keyboards, and job scheduling. Keep bot responses concise and HTML-safe.
- **Timeframes**: Reference `Timeframe` and `TimeframeSettings` from `wyckoff_types.py` rather than hardcoding values.
- **Caching**: Route candle and funding data fetches through `candles_cache.py` and `funding_rates_cache.py`.
- **Rate Limiting**: All API calls must respect `hyperliquid_rate_limiter`.
- **Logging**: Use `logging_utils.logger`; `info` for normal operations, `error`/`critical` with `exc_info=True` for failures.
- **Analysis Logic**: Keep deterministic (no randomness in TA); non-determinism makes tests brittle and prompts harder to reason about.
- **Testing**: Add or extend tests beside the relevant module. Do not change default risk parameters or order sizes in trading code without explicit intent.

## Adding New Strategies

1. Create `strategies/<name>/<name>.py` with a class name matching `name.title().replace('_', '')`
2. Base it on `BaseStrategyConfig` from `strategies/base_strategy/`
3. Use `HTB_<STRATEGY>_*` environment variables for configuration
4. Register command handlers using `telegram_utils.add_handler()` from `init_strategy()`

## Adding New TA Features

1. Add indicator calculations in `technical_analysis/data_processor.py:apply_indicators()`
2. Expose via `TimeframeSettings` in `wyckoff_types.py`
3. Reuse existing `WyckoffState`, MTF groupings, and description generators

## Adding New LLM Signals

1. Extend `LLMPromptGenerator` in `technical_analysis/llm/prompt_generator.py`
2. Keep the JSON response schema strict and backward-compatible
3. Prefer enriching prompts with derived metrics from existing data over new external dependencies

## Questions?

If any part of this guidance feels unclear, open an issue or start a discussion.