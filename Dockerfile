# 1. Specify the Base Image
# Use the RunPod PyTorch image with Python 3.11 and CUDA 12.8.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. Set the Working Directory
# This is where your code and dependencies will live inside the container.
WORKDIR /app

# 3. Install Python Dependencies
# Includes your list plus 'runpod' and 'requests', which are required by the handler.
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    accelerate \
    Pillow \
    einops \
    runpod \
    requests

# 4. Copy Your Handler Script
# This copies the handler.py file from your local directory into the container.
COPY handler.py .

# 5. Set the Default Command
# This tells the container to run your handler script when it starts.
CMD ["python", "handler.py"]