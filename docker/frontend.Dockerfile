FROM oven/bun:canary-alpine AS builder

WORKDIR /app

# Copy dependency manifests
COPY frontend/package.json frontend/bun.lock ./

# Install dependencies (frozen to ensure exact versions from lockfile)
RUN bun install --frozen-lockfile

# Copy the rest of the frontend code
COPY frontend ./

RUN bun run build

# ---------- Runtime stage ----------
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html


EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]