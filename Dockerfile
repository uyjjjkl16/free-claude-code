FROM python:3.14-slim

# Instalar Node.js e npm necessários para o Claude Code
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Instalar o Claude Code globalmente
RUN npm install -g @anthropic-ai/claude-code

COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, so code changes don't re-run pip install
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY . .

EXPOSE 8082
CMD ["uv", "run", "uvicorn", "servidor:app", "--host", "0.0.0.0", "--port", "10000"]
