FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install deps (explicitly upgrade HF hub for CLI)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade huggingface_hub

# Download model during build (to /app/indextts2_checkpoints)
# Use --token-env for safe token passing (no shell expansion issues)
RUN huggingface-cli login --token-env HF_TOKEN --add-to-git-credential && \
    huggingface-cli download IndexTeam/IndexTTS-2 --local-dir /app/indextts2_checkpoints && \
    huggingface-cli logout

# Copy code
COPY api.py .

# Create temp dirs for runtime
RUN mkdir -p /tmp/indextts/{indextts2_checkpoints,voices,cache}

EXPOSE 8000

CMD ["python", "api.py"]