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

## Trading Strategies

The bot supports custom trading strategies in the `strategies/` directory:

- `default_strategy`: Basic trading strategy implementation
- `etf_strategy`: Strategy focused on ETF-like trading behavior

To implement a new strategy:
1. Create a new directory under `strategies/`
2. Implement your strategy following the base strategy interface
3. Register your strategy in the bot configuration

## Contributing

Contributions are welcome! For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
