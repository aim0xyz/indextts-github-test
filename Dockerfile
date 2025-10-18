# Use official NVIDIA CUDA runtime base for GPU support (Ubuntu 22.04, CUDA 12.2 - matches RunPod)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python 3.11 and essentials
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    && (ln -sf /usr/bin/python3.11 /usr/bin/python || true) \
    && (ln -sf /usr/bin/pip3 /usr/bin/pip || true) \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python requirements (PyTorch with CUDA first for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1+cu121 torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Install index_tts from Git with proper authentication handling
# Option 1: Try pip install first (works for public repos)
RUN (pip install --no-cache-dir git+https://github.com/IndexTeam/IndexTTS.git && echo "✅ IndexTTS installed via pip") || \
    # Option 2: Manual git clone with authentication (for private repos)
    (echo "🔑 Setting up git credentials..." && \
     git config --global user.email "docker@example.com" && \
     git config --global user.name "Docker Build" && \
     git clone https://github.com/IndexTeam/IndexTTS.git /tmp/indextts && \
     cd /tmp/indextts && \
     pip install -e . && \
     cd / && rm -rf /tmp/indextts && \
     echo "✅ IndexTTS installed via manual git clone")

# Alternative: If IndexTTS installation fails, you can:
# 1. Make sure the repository is public or use a personal access token
# 2. Use this alternative installation method:
# RUN git clone https://github.com/IndexTeam/IndexTTS.git /tmp/indextts && \
#     cd /tmp/indextts && \
#     pip install -e . && \
#     cd / && rm -rf /tmp/indextts

# Copy your application code
COPY api.py .
# If you have presets.json or other files:
# COPY presets.json .

# Create ephemeral directories (your api.py uses /tmp, which persists during worker lifetime)
RUN mkdir -p /tmp/indextts/{indextts2_checkpoints,voices,cache} && \
    chmod 755 /tmp/indextts /tmp/indextts/*

# Environment variables for performance and HF downloads
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Use GPU 0 (RunPod assigns one per worker)
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for FastAPI (RunPod job API uses internal handler, but good practice)
EXPOSE 8000

# Health check (assumes /health endpoint from your api.py)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn for FastAPI (better than plain python for prod; matches your api.py)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]