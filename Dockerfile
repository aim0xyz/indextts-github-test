FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including CUDA support
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    ffmpeg \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime for GPU support
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-runtime-12-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements (including PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .

# Create necessary directories
RUN mkdir -p /tmp/indextts/{indextts2_checkpoints,voices,cache} && \
    chmod 755 /tmp/indextts /tmp/indextts/*

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "api.py"]