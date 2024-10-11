FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY ["*.py", "pyproject.toml", "uv.lock", "./"]

RUN apt-get update && apt-get install -y git && apt-get clean && uv sync --frozen

CMD ["uv", "run", "hyperliquid_bot.py"]
