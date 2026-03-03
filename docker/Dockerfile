FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY python/ ./python/
COPY configs/ ./configs/

# Install Python dependencies
RUN uv sync --no-dev

# Data volumes (scenes and indexes)
VOLUME ["/data/scenes", "/data/indexes"]

EXPOSE 8000

ENV SQ_SCENE_ROOT=/data/scenes
ENV SQ_INDEX_ROOT=/data/indexes
ENV SQ_LOG_LEVEL=INFO

CMD ["uv", "run", "uvicorn", "python.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
