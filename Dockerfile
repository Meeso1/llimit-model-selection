# Use a CUDA-enabled base image for GPU-accelerated training
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Install uv from the official binary distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
# HF_HOME ensures HuggingFace downloads go to a specific location we can cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    HF_HOME=/app/.cache/huggingface \
    UV_LINK_MODE=copy

WORKDIR /app

# To leverage Docker layer caching, we copy only the dependency files first
COPY pyproject.toml uv.lock .python-version ./

# Install Python 3.13 and sync dependencies
# This creates a .venv in the container
RUN uv python install 3.13 && \
    uv sync --frozen --no-install-project

# Pre-download the HuggingFace dataset into the image
# This avoids the "few minutes" download during pod startup
RUN uv run python -c "import datasets; datasets.load_dataset('lmarena-ai/arena-human-preference-140k')"

# Copy the rest of the project
# This includes the source code and any pre-existing LFS files in your local copy
COPY . .

# Final sync to install the project itself
RUN uv sync --frozen

# Default to bash for interactive use in RunPod
CMD ["/bin/bash"]

