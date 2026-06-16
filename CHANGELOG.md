# Changelog

All notable changes to the hyperliquid-telegram-bot project are documented in this file.

## Unreleased

- Added SKILL.md for AI agent development reference
- Refactored websocket event handling (removed user fundings subscription)
- Improved momentum scoring and volume metrics for crypto analysis
- Enhanced trading logic with price validation and callback handling
- Added vault support to portfolio balance calculations
- Added ADX indicator and improved entry timing checks
- Implemented dynamic stale position closure thresholds based on age and PnL
- Enhanced trend analysis with phase-aware signal direction
- Adjusted alignment scores and decision thresholds for signal filtering
- Added 8-hour timeframe to trade suggestion levels

## 2025-01 (Development Peak)

### Features
- Multi-timeframe Wyckoff state analysis and notification logic
- Wyckoff chart generation with Heikin Ashi and significant levels
- Configurable timeframe settings via `Timeframe` enum
- Funding rate caching and analysis with outlier detection
- Disk caching for candle data with load/save/clear
- Actionable Wyckoff signal detection with enhanced confirmation

### Improvements
- Phase hysteresis to reduce flip-flopping in Wyckoff state detection
- Crypto-specific adjustments for volume and effort thresholds
- Enhanced significant levels detection with adaptive parameters
- Better volatility handling and clustering/scoring for support/resistance
- Optimized SuperTrend and MACD settings for crypto markets

### Refactoring
- Extracted candle analysis into dedicated utility modules
- Organized technical analysis into package structure
- Simplified phase detection to analyze only the last period
- Improved funding rates caching with typed dataclasses

### Testing
- Dedicated candle cache tests
- Tests for significant levels analysis

## 2024-12

### Features
- ETF strategy with Coingecko category filtering
- Fixed token strategy with configurable token set
- Delta-neutral strategy for market-neutral positions
- PnL percentage alerts with configurable thresholds
- Spot token balance and PnL display
- Portfolio rebalancing commands for ETF strategy

### Improvements
- Increased PnL alert threshold from 10% to 20%
- Maximum leverage raised for trading
- Better table formatting for positions and orders
- Improved order update logic based on PnL

### Refactoring
- Moved alerts to dedicated file
- Refactored position commands
- General BaseStrategy refactor

## 2024-11

### Features
- Custom bot strategies (ETF strategy first draft)
- Rebalance command for ETF strategy
- Coingecko category filtering for ETF strategy
- Leverage as configurable environment variable
- CoinGecko-to-Hyperliquid symbol mapping

### Improvements
- Docker image size reduction
- Rate-limited TA when scanning multiple coins
- Increased candles used for technical analysis
- Configurable SL and TP prices when opening positions

### CI/CD
- GitHub Actions test and Docker publish workflows

## 2024-10

- Switched to `uv` as package manager

## 2024-08

### Features
- Heikin Ashi candle charts
- MACD indicator replacing Volume chart
- EMA indicators on TA charts
- Multi-timeframe charts (5m / 1h / 4h) in TA results
- VWAP indicator
- Configurable stop loss percentage when opening positions

### Improvements
- Chart filtering by time range
- Reduced candle download count for performance
- Improved indicator table formatting
- Fixed scheduled `analyze_candles`

## 2024-07

### Features
- First draft of TA-based notifications
- SuperTrend indicator integration
- Z-score trend analysis
- `/overview` command for portfolio overview
- 4h candle TA support
- Configurable take-profit percentage

### Improvements
- Configurable leverage defaults
- Better indicator table formatting
- Hourly scheduled TA runs
- Refactored `get_open_orders`

### Fixes
- Spot positions without price data
- TP order error handling

## 2024-06

### Features
- Buy/sell order commands with conversation flow
- Stop loss order placement
- Take profit order placement
- Exit all positions command
- Automatic SL/TP order updates based on unrealized PnL

### Improvements
- Leverage configuration and limits
- SL distance limit adjustments
- Support for both long and short orders
- Isolated margin by default

### Refactoring
- Extracted bot logic to separate files
- Moved Telegram logic to `TelegramUtils` singleton
- Exchange setup via key file

## 2024-05 (Initial Release)

### Features
- Telegram bot for Hyperliquid position monitoring
- Real-time fill notifications via websocket
- Position, order, and balance display
- Docker support
- GitHub Actions CI/CD

### Infrastructure
- Initial project structure
- Dockerfile multi-stage build
- Docker publish workflow
- GPL-3.0 license