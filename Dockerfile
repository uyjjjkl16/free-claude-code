
FROM python:3.14-slim

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y curl passwd \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Create a non-root user
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home appuser

# Copy project files
COPY . .

# Give the application user access to the project
RUN chown -R appuser:appuser /app \
    && mkdir -p /app/agent_workspace \
    && chown -R appuser:appuser /app/agent_workspace

# Run without root privileges
USER appuser

EXPOSE 10000

CMD ["uv", "run", "--no-sync", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "10000"]
