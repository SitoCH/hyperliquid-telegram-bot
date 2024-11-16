# Build stage
FROM python:3.10-alpine as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN apk add --no-cache --virtual .build-deps git && \
    uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen && \
    apk del .build-deps && \
    rm -rf /root/.cache /tmp/*

# Copy application code
COPY . ./

# Final stage
FROM python:3.10-alpine

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/ /app/

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "hyperliquid_bot.py"]
