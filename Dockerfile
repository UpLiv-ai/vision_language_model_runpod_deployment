# 1. Base Image
# We switch to the newer RunPod image that supports PyTorch 2.4+ and CUDA 12.4
# which is required by MiniCPM-o 4.5.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Set Working Directory
WORKDIR /app

# 3. System & Python Updates
# Good practice to ensure pip is up to date before installing heavy libraries
RUN python -m pip install --upgrade pip

# 4. Install Dependencies
# We combine these into a single RUN command to keep the image layer size optimized.
# Note: We explicitly install torch>=2.4.0 to ensure compatibility, even if the base has it.
RUN pip install --no-cache-dir \
    "torch>=2.4.0" \
    "torchaudio>=2.4.0" \
    "torchvision>=0.19.0" \
    "transformers==4.51.0" \
    "minicpmo-utils>=1.0.5" \
    accelerate \
    sentencepiece \
    Pillow \
    einops \
    runpod \
    requests \
    numpy

# 5. Copy Handler
# Copies your updated handler.py into the container
COPY handler.py .

# 6. Start the Handler
# The "-u" flag ensures Python output is unbuffered, so logs show up immediately in RunPod.
CMD ["python", "-u", "handler.py"]