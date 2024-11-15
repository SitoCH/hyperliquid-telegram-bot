# Build stage
FROM python:3.10-slim as builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen

# Copy application code
COPY *.py strategies/ ./

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/*.py /app/
COPY --from=builder /app/strategies /app/strategies

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "hyperliquid_bot.py"]