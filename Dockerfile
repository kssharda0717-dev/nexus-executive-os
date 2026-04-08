# ── NEXUS — Multi-stage production Dockerfile ──────────
# Optimized for Google Cloud Run: fast cold starts, small image, non-root user.

# Stage 1: Build dependencies
FROM python:3.13-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production image
FROM python:3.13-slim AS production

# Security: non-root user
RUN groupadd -r nexus && useradd -r -g nexus -d /app -s /sbin/nologin nexus

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directory for SQLite (writable by nexus user)
RUN mkdir -p /app/data && chown -R nexus:nexus /app

# Switch to non-root user
USER nexus

# Cloud Run sets PORT env var; default to 8080 (Cloud Run standard)
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

EXPOSE ${PORT}

# Run with uvicorn — single worker for SQLite compatibility,
# but multiple could be used with a proper DB backend
CMD ["sh", "-c", "python -m uvicorn main:app --host $HOST --port $PORT --workers 1 --log-level info --access-log"]
