FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODEL=qwen3:8b

WORKDIR /app

# AudioCraft/PyAV/FFmpeg runtime + build dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    curl ca-certificates git build-essential pkg-config ffmpeg zstd \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Use the CUDA 11.8 PyTorch wheels that match the proven MAIX stack.
RUN python3.10 -m pip install \
    torch==2.1.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements-serverless.txt /app/
RUN python3.10 -m pip install -r requirements-serverless.txt

# Ollama runtime lives outside Python; the Python "ollama" package above is only the client.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Bake Qwen into the image so workers do not download it for every cold start.


CMD ["python3.10", "-u", "/app/handler.py"]
