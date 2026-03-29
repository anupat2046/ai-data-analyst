# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder
# Install Python packages into an isolated prefix so the final image stays lean.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# System libs needed to compile some packages (scipy, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="AI Data Analyst"
LABEL description="AI-powered data analytics — FastAPI backend + Streamlit frontend"

WORKDIR /app

# Apply all security patches, then install runtime-only system libraries.
# curl is intentionally omitted — health checks use a Python one-liner instead.
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p uploads data

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose both service ports
EXPOSE 8000 8501

# Default command — overridden by docker-compose per service
CMD ["bash", "start.sh"]
