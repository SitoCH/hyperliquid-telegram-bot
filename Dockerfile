# syntax=docker/dockerfile:1.4

FROM python:3.10-slim-bullseye AS base

FROM ghcr.io/astral-sh/uv:latest AS uv

FROM base AS builder
WORKDIR /app

COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen && \
    # Cleanup system
    apt-get purge -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . ./

FROM base AS final
WORKDIR /app

COPY --from=builder /app/ /app/

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "hyperliquid_bot.py"]
