# hyperliquid-telegram-bot

This is a Telegram bot for Hyperliquid that allows you to check the open positions of a given address and receive updates when an order is filled.
The inspiration for it comes from Freqtrade: https://www.freqtrade.io/en/stable/

## Parameters required

### Telegram

In order to create a Telegram bot you need the `HTB_TOKEN` from @BotFather. Send the command `/newbot` to it, follow the steps and copy the Bot token given to you at the end of the procedure.
The second parameter needed, `HTB_CHAT_ID`, can be obtained form the @userinfobot (the field is named `Id`).

### Hyperliquid

There only mandatory parameter, `HTB_USER_WALLET`, is the user's wallet address.
If the bot is monitoring a Vault you need to set also `HTB_USER_VAULT`.
`HTB_KEY_FILE` is instead required if the bot needs to sign transactions so that it can manage orders. The file must contain the private key of the wallet.

### Other environment variables

| Variable    | Description | Example |
| -------- | ------- |
| HTB_COINS_TO_ANALYZE    | Coins to anylze every hour | "BTC,ETH"|
| HTB_ANALYZE_COINS_WITH_OPEN_ORDERS    | Analyze also coins that have open orders |"True"|

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
      HTB_TOKEN: "<TELEGRAM BOT TOKEN>"
      HTB_CHAT_ID: "<TELEGRAM CHAT ID>"
      HTB_USER_WALLET: "<ADDRESS TO WATCH>"
    restart: unless-stopped
```