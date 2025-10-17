FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade huggingface_hub

# Copy code
COPY api.py .

# Create temporary dirs for runtime storage
RUN mkdir -p /tmp/indextts/{indextts2_checkpoints,voices,cache}

EXPOSE 8000
CMD ["python", "api.py"]