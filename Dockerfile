FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY server.py chatbot.py chatbot_langgraph.py constants.py messages_class.py ./

# Copy data files if any exist alongside the app
COPY *.txt ./

# Expose the port uvicorn will listen on
EXPOSE 8000

# Run uvicorn via uv; restart=always handled by Docker restart policy
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
