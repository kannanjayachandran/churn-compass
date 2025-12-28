FROM oven/bun:canary-alpine

WORKDIR /app

# Copy dependency manifests
COPY frontend/package.json frontend/bun.lock ./

# Install dependencies (frozen to ensure exact versions from lockfile)
RUN bun install --frozen-lockfile

# Copy the rest of the frontend code
COPY frontend ./

# Use a non-root user for security
USER bun

EXPOSE 3000

CMD ["bun", "run", "dev"]
