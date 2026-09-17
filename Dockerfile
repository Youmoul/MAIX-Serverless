FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODEL=qwen3:8b

WORKDIR /app

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------
# AudioCraft/PyAV/FFmpeg runtime + build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    ca-certificates \
    git \
    build-essential \
    pkg-config \
    ffmpeg \
    zstd \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python tooling
# ---------------------------------------------------------
RUN python3.10 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# ---------------------------------------------------------
# PyTorch CUDA 11.8
# ---------------------------------------------------------
# This matches the MAIX stack we already validated.
RUN python3.10 -m pip install \
    torch==2.1.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------
# MAIX Python dependencies
# ---------------------------------------------------------
COPY requirements-serverless.txt /app/requirements-serverless.txt

RUN python3.10 -m pip install \
    -r /app/requirements-serverless.txt

# ---------------------------------------------------------
# Ollama runtime
# ---------------------------------------------------------
# The Python "ollama" package is only the client.
# We also need the actual Ollama runtime inside the container.
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---------------------------------------------------------
# MAIX application
# ---------------------------------------------------------
COPY config.py /app/config.py
COPY maestro.py /app/maestro.py
COPY audio_utils.py /app/audio_utils.py
COPY music_engine.py /app/music_engine.py
COPY handler.py /app/handler.py

# ---------------------------------------------------------
# RunPod Serverless worker
# ---------------------------------------------------------
CMD ["python3.10", "-u", "/app/handler.py"]
