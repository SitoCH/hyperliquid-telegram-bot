# Security Policy

## Reporting a Vulnerability

This project interacts with real cryptocurrency accounts on Hyperliquid. Security is critical.

If you discover a security vulnerability, please report it privately by:

1. **Opening a security advisory** at https://github.com/SitoCH/hyperliquid-telegram-bot/security/advisories
2. Alternatively, contact the maintainer directly via the GitHub profile

Please do **not** report security vulnerabilities through public GitHub issues.

## What to Include

When reporting, please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact (e.g., fund loss, unauthorized trading, data exposure)
- Suggested fix (if known)

## Scope

The following are in scope:
- The bot's trading logic (`trade_conversation.py`, `trade_execution.py`, `trade_pricing.py`, `hyperliquid_orders.py`)
- Private key handling (`hyperliquid_utils/utils.py`, `HTB_KEY_FILE`)
- API rate limiting and abuse prevention
- Environment variable leakage in logs or error messages
- WebSocket event handling that could lead to unintended trades

## Security Best Practices

- **Private keys**: Store `HTB_KEY_FILE` securely. Use read-only wallets for monitoring-only setups.
- **Environment variables**: Never commit `.env` files or hardcoded tokens.
- **Docker**: Keep the Docker image updated; the CI pipeline publishes `latest` tags.
- **API keys**: Rotate Telegram bot tokens and Hyperliquid keys periodically.
- **Backup**: Backup your `HTB_KEY_FILE` and wallet recovery phrases separately.

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest (Docker `latest` tag) | ✅ |
| Previous versions | ❌ |

We recommend always running the latest Docker image or the latest commit on `main`.