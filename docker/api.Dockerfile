FROM python:3.13-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen to ensure exact versions from lockfile)
RUN uv sync --frozen --no-cache --no-install-project --no-group dev

# copy metadata first
COPY pyproject.toml README.md ./

# Copy source
COPY src/ /app/src/

# Install the project itself
RUN uv pip install --no-cache -e .

EXPOSE 8000

# Use 'uv run' to ensure the environment is correctly used
CMD ["uv", "run", "uvicorn", "churn_compass.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
