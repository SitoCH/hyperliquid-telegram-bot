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
| HTB_ANALYZE_COINS_WITH_OPEN_ORDERS | Include coins with open orders in analysis | "True" | False |
| HTB_CATEGORIES_TO_ANALYZE | CoinGecko categories to analyze (comma-separated) | "layer-1,defi" | None |

## Trading Strategies

The bot supports custom trading strategies in the `strategies/` directory:

- `default_strategy`: Basic trading strategy implementation
- `etf_strategy`: Strategy focused on ETF-like trading behavior that automatically allocates funds across top cryptocurrencies based on market capitalization
- `fixed_token_strategy`: Strategy that trades a fixed set of tokens with configurable parameters

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

## Contributing

Contributions are welcome! For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
