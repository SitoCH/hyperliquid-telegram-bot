# Build stage
FROM python:3.10-slim-bullseye AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy only dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment and cleanup in the same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    # Minimal dependencies for matplotlib
    libfreetype6 \
    libpng16-16 \
    && uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv sync --frozen && \
    # Configure matplotlib to use Agg backend
    mkdir -p /app/.venv/lib/python3.10/site-packages/matplotlib && \
    echo "backend: Agg" > /app/.venv/lib/python3.10/site-packages/matplotlib/mpl-data/matplotlibrc && \
    # Cleanup system
    apt-get purge -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary Python files
COPY . ./

# Final stage
FROM python:3.10-slim-bullseye

WORKDIR /app

# Install only runtime dependencies for matplotlib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/ /app/

# Use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set Python to not write bytecode files and run in unbuffered mode
ENV MPLBACKEND=Agg

CMD ["python", "hyperliquid_bot.py"]
