# Multi-stage build for optimal image size
FROM python:3.11-slim as builder

# Install UV for fast dependency installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync --frozen --no-dev

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 researcher && \
    mkdir -p /app && \
    chown -R researcher:researcher /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=researcher:researcher /app/.venv /app/.venv

# Copy application code
COPY --chown=researcher:researcher ai_deep_research_assistant/ ./ai_deep_research_assistant/
COPY --chown=researcher:researcher pyproject.toml ./

# Switch to non-root user
USER researcher

# Set Python path
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Create directory for output files
RUN mkdir -p /app/output

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - run in interactive mode
ENTRYPOINT ["python", "-m", "ai_deep_research_assistant.main"]

# Allow command line arguments to be passed
CMD []