# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM python:3.10-slim-bullseye AS base

FROM base AS uv-amd64
ADD https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz

FROM base AS uv-arm64
ADD https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz /tmp/uv.tar.gz

FROM uv-$TARGETARCH AS uv
RUN tar xf /tmp/uv.tar.gz -C /bin && rm /tmp/uv.tar.gz

FROM base AS builder

COPY --from=uv /bin/uv /bin/uv

WORKDIR /app

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
