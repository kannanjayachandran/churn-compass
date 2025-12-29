FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PYTHONPATH=/app/src

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Copy source (needed before sync for editable install)
COPY src/ /app/src/
COPY README.md ./

# Install dependencies AND project in one step, excluding dev group
RUN uv sync --frozen --no-cache --no-group dev

EXPOSE 8000

# Use 'uv run' to ensure the environment is correctly used
CMD ["uv", "run", "uvicorn", "churn_compass.api.main:app", "--host", "0.0.0.0", "--port", "8000"]