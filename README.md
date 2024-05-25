# hyperliquid-telegram-bot

This is a Telegram bot for Hyperliquid that allows you to check the open positions of a given address and receive updates when an order is filled.
The inspiration for it comes from Freqtrade: https://www.freqtrade.io/en/stable/

## Parameters required

### Telegram

In order to create a Telegram bot you need the `HYPERLIQUID_TELEGRAM_BOT_TOKEN` from @BotFather. Send the command `/newbot` to it, follow the steps and copy the Bot token given to you at the end of the procedure.
The second parameter needed, `HYPERLIQUID_TELEGRAM_BOT_CHAT_ID`, can be obtained form the @userinfobot (the field is named `Id`).

### Hyperliquid

The only parameter, `HYPERLIQUID_TELEGRAM_BOT_USER`, needed from Hyperliquid is the user / Vault to watch

## Docker Compose

The bot can be started adding it as a normal Docker Compose service:

```
---
version: '3'
services:

  hyperliquid_bot:
    image: sito/hyperliquid-telegram-bot:latest
    container_name: hyperliquid_bot
    environment:
      HYPERLIQUID_TELEGRAM_BOT_TOKEN: "<TELEGRAM BOT TOKEN>"
      HYPERLIQUID_TELEGRAM_BOT_CHAT_ID: "<TELEGRAM CHAT ID>"
      HYPERLIQUID_TELEGRAM_BOT_USER: "<ADDRESS TO WATCH>"
    restart: unless-stopped
```