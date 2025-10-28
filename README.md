# hyperliquid-telegram-bot

A Telegram bot for Hyperliquid that monitors cryptocurrency trading positions and sends notifications for filled orders. Inspired by [Freqtrade](https://www.freqtrade.io/en/stable/).

## Features

- 🔍 Monitor open positions for any wallet address
- 📊 Real-time notifications for filled orders
- 📈 Support for custom trading strategies
- ⏰ Periodic coin analysis
- 🐳 Docker support for easy deployment

## Quick Start

1. Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  hyperliquid_bot:
    image: sito/hyperliquid-telegram-bot:latest
    container_name: hyperliquid_bot
    environment:
      HTB_TOKEN: "<TELEGRAM BOT TOKEN>"
      HTB_CHAT_ID: "<TELEGRAM CHAT ID>"
      HTB_USER_WALLET: "<ADDRESS TO WATCH>"
    restart: unless-stopped
```

2. Run the bot:
```bash
docker-compose up -d
```

## Configuration

### Required Parameters

#### Telegram Setup
1. Create a new bot through [@BotFather](https://t.me/BotFather):
   - Send `/newbot` command
   - Follow the setup procedure
   - Copy the provided bot token to `HTB_TOKEN`
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot):
   - The "Id" field is your `HTB_CHAT_ID`

#### Hyperliquid Parameters
- `HTB_USER_WALLET`: The wallet address to monitor (required)
- `HTB_USER_VAULT`: Vault address (optional, for vault monitoring)
- `HTB_KEY_FILE`: Path to private key file (optional, for order management)

### Optional Parameters

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| HTB_COINS_TO_ANALYZE | Coins to analyze hourly | "BTC,ETH" | None |
| HTB_TOP_COINS_TO_ANALYZE | Number of top coins by open interest to analyze | "10" | None |
| HTB_TOP_COINS_OFFSET | Number of top coins to skip before selecting coins for analysis | "0" | 0 |
| HTB_ANALYZE_COINS_WITH_OPEN_ORDERS | Include coins with open orders in analysis | "True" or "False" | "False" |
| HTB_COINS_ANALYSIS_MIN_CONFIDENCE | Minimum confidence level for coin analysis notifications | "0.75" | 0.75 |
| HTB_MONITOR_STALE_POSITIONS | Monitor positions older than 1 day with positive PnL | "True" or "False" | "False" |
| HTB_STALE_POSITION_MIN_PNL | Minimum PnL% to consider a position as stale. Provide a float like "5" for 5% (percent sign not supported) | "5" | 5 |
| HTB_AUTO_CLOSE_STALE_POSITIONS | Automatically close stale positions instead of just alerting | "True" or "False" | "False" |
| HTB_USE_ISOLATED_LEVERAGE | Use isolated leverage instead of cross | "True" or "False" | "True" |
| HTB_ALPHAVANTAGE_API_KEY | API key for Alpha Vantage (required for performance comparison against S&P500) | "XXXXXXXXXXXXXXXX" | None |
| HTB_ANALYSIS_MODE | Analysis mode for technical analysis | "wyckoff" or "llm" | "wyckoff" |
| HTB_SKIP_SL_TP_PROMPT | Skip asking for stop loss and take profit when opening a position | "True" or "False" | "False" |
| HTB_LLM_MAIN_MODEL | Main model for LLM analysis | "gpt-4" | "gpt-4" |
| HTB_LLM_FAST_MODEL | Fast model for filtering analysis | "gpt-3.5-turbo" | "gpt-3.5-turbo" |
| HTB_ALWAYS_RUN_LLM_FILTER | Always run LLM filter (use filter before analysis) | "True" or "False" | "False" |
| HTB_TRADE_MIN_RR | Minimum Risk:Reward ratio required to propose a trade | "1.6" | 1.4 |
| HTB_USE_HEIKIN_ASHI | Use Heikin Ashi candles in generated charts | "True" or "False" | False |

## Technical Analysis Modes

The bot supports two different technical analysis modes that can be switched using the `HTB_ANALYSIS_MODE` environment variable:

### Wyckoff Analysis Mode (Default)

The default mode uses traditional Wyckoff methodology to analyze market phases across multiple timeframes with volume pattern detection and chart generation.

You can toggle the chart candle type to standard candles by setting:

```yaml
environment:
  HTB_USE_HEIKIN_ASHI: "False"
```

When enabled (the default), charts use Heikin Ashi candles for smoother visualization while calculations (indicators and level detection) still use the original OHLCV data.

### LLM Analysis Mode

An AI-powered mode that uses large language models to provide natural language analysis, confidence scoring, risk assessment, and price predictions with timeframes.

To use LLM mode, you need to:

1. Set `HTB_ANALYSIS_MODE=llm`
2. Configure the appropriate environment variables for LiteLLM (e.g., `LITELLM_PROXY_API_BASE`, `LITELLM_PROXY_API_KEY`, etc.)
3. Optionally customize the models

Example configuration for LLM mode:

```yaml
environment:
  HTB_ANALYSIS_MODE: "llm"
  LITELLM_PROXY_API_BASE: "http://localhost:4000"
  LITELLM_PROXY_API_KEY: "sk-xxxxx"
  HTB_LLM_MAIN_MODEL: "gpt-4"
  HTB_LLM_FAST_MODEL: "gpt-3.5-turbo"
```

## Trading Strategies

The bot supports custom trading strategies in the `strategies/` directory:

- `default_strategy`: Basic trading strategy implementation
- `etf_strategy`: Strategy focused on ETF-like trading behavior that automatically allocates funds across top cryptocurrencies based on market capitalization
- `fixed_token_strategy`: Strategy that trades a fixed set of tokens with configurable parameters
- `alpha_g_strategy`: Strategy that detects strong multi-day surges/crashes and flags potential daily reversal signals

### ETF Strategy Configuration

The ETF strategy can be configured using the following environment variables:

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| HTB_ETF_STRATEGY_COINS_NUMBER | Number of top coins to include in portfolio | "5" | 5 |
| HTB_ETF_STRATEGY_COINS_OFFSET | Number of top coins to skip (useful to exclude BTC/ETH) | "2" | 0 |
| HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE | Minimum 1-year performance percentage to include a coin | "15.0" | 15.0 |
| HTB_ETF_STRATEGY_LEVERAGE | Leverage to use for positions | "5" | 5 |
| HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS | Comma-separated list of symbols to exclude | "DOGE,SHIB" | "" |
| HTB_ETF_STRATEGY_CATEGORY | Filter coins by category (e.g., "layer-1") | "defi" | None |

Example configuration in docker-compose.yml:
```yaml
version: '3'
services:
  hyperliquid_bot:
    image: sito/hyperliquid-telegram-bot:latest
    container_name: hyperliquid_bot
    environment:
      HTB_TOKEN: "<TELEGRAM BOT TOKEN>"
      HTB_CHAT_ID: "<TELEGRAM CHAT ID>"
      HTB_USER_WALLET: "<ADDRESS TO WATCH>"
      HTB_ETF_STRATEGY_COINS_NUMBER: "3"
      HTB_ETF_STRATEGY_COINS_OFFSET: "2"
      HTB_ETF_STRATEGY_MIN_YEARLY_PERFORMANCE: "20.0"
      HTB_ETF_STRATEGY_LEVERAGE: "3"
      HTB_ETF_STRATEGY_EXCLUDED_SYMBOLS: "DOGE,SHIB"
      HTB_ETF_STRATEGY_CATEGORY: "layer-1"
    restart: unless-stopped
```

This configuration would create a portfolio that:
- Skips the top 2 coins (typically BTC and ETH)
- Includes the next 3 top layer-1 coins by market cap
- Only includes coins with >20% yearly performance
- Uses 3x leverage
- Excludes DOGE and SHIB from consideration

### Fixed Token Strategy Configuration

The Fixed Token strategy allows you to trade a specific set of tokens with configurable parameters. It can be configured using the following environment variables:

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| HTB_FIXED_TOKEN_STRATEGY_TOKENS | Comma-separated list of tokens to trade | "BTC,ETH" | "BTC,ETH" |
| HTB_FIXED_TOKEN_STRATEGY_LEVERAGE | Leverage to use for positions | "5" | 5 |
| HTB_FIXED_TOKEN_STRATEGY_MIN_YEARLY_PERFORMANCE | Minimum 1-year performance percentage to include a token | "15.0" | 15.0 |

Example configuration in docker-compose.yml:
```yaml
version: '3'
services:
  hyperliquid_bot:
    image: sito/hyperliquid-telegram-bot:latest
    container_name: hyperliquid_bot
    environment:
      HTB_TOKEN: "<TELEGRAM BOT TOKEN>"
      HTB_CHAT_ID: "<TELEGRAM CHAT ID>"
      HTB_USER_WALLET: "<ADDRESS TO WATCH>"
      HTB_FIXED_TOKEN_STRATEGY_TOKENS: "BTC,ETH,SOL"
      HTB_FIXED_TOKEN_STRATEGY_LEVERAGE: "3"
      HTB_FIXED_TOKEN_STRATEGY_MIN_YEARLY_PERFORMANCE: "20.0"
    restart: unless-stopped
```

This configuration would:
- Trade only BTC, ETH, and SOL
- Use 3x leverage for positions
- Only include tokens with >20% yearly performance

### AlphaG Strategy Configuration

The AlphaG strategy analyzes top market-cap coins to detect strong multi-day moves and highlights potential intraday reversals. You can configure its sensitivity with the following environment variables:

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| HTB_ALPHA_G_STRATEGY_LOOKBACK_DAYS | Number of full daily candles to consider when classifying a surge/crash | "3" | 3 |
| HTB_ALPHA_G_STRATEGY_THRESHOLD_PCT | Absolute percentage threshold to qualify a surge/crash over the lookback window | "20.0" | 20.0 |

Example configuration in docker-compose.yml:
```yaml
version: '3'
services:
  hyperliquid_bot:
    image: sito/hyperliquid-telegram-bot:latest
    container_name: hyperliquid_bot
    environment:
      HTB_TOKEN: "<TELEGRAM BOT TOKEN>"
      HTB_CHAT_ID: "<TELEGRAM CHAT ID>"
      HTB_USER_WALLET: "<ADDRESS TO WATCH>"
      HTB_ALPHA_G_STRATEGY_LOOKBACK_DAYS: "3"        # Number of full candles (days)
      HTB_ALPHA_G_STRATEGY_THRESHOLD_PCT: "20.0"     # Surge/Crash threshold in % over lookback
    restart: unless-stopped
```

## Contributing

Contributions are welcome! For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
