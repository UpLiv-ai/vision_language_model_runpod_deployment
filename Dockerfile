# Use the compatible PyTorch 2.4 base
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install standard dependencies
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    sentencepiece \
    accelerate \
    Pillow \
    einops \
    runpod \
    requests

# FORCE install compatible Flash Attention (Critical for MiniCPM on Torch 2.4)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY handler.py .

CMD ["python", "handler.py"]