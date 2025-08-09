# Use official UV image with Python 3.12
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Create data and temp folders
RUN mkdir -p /app/data /app/temp

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv sync
RUN uv sync --locked --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --locked

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the command to run the application with Unicorn
# Run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio", "--limit-concurrency", "400","--workers","20"]