FROM runpod/pytorch:2.0.1-py3.11-cuda11.8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
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