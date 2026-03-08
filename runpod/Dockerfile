FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV LTX_APP_DATA_DIR=/workspace

# System dependencies + Python 3.12 (deadsnakes PPA) + Node.js for frontend build
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nodejs \
    && npm install -g pnpm@10 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install PyTorch 2.8 with CUDA 12.8 (pinned for reproducibility)
RUN pip install --no-cache-dir \
    "torch==2.8.0" \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install uv for fast package installs
RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Frontend build (web-only, no Electron) ──
COPY frontend/ /app/frontend/
COPY package.json pnpm-lock.yaml* tsconfig*.json vite.config.web.ts index.html /app/
COPY postcss.config.js* tailwind.config.js* /app/
RUN pnpm install --frozen-lockfile --ignore-scripts && pnpm run build:web \
    && mkdir -p /app/backend/static \
    && cp -r /app/dist-web/* /app/backend/static/ \
    && rm -rf /app/node_modules /app/dist-web /app/frontend

# ── Backend ──
COPY backend/ /app/backend/

# Install backend dependencies (without torch, already installed)
RUN cd /app/backend && pip install --no-cache-dir \
    pillow>=10.3.0 \
    huggingface-hub>=0.23.0 \
    tqdm>=4.66.0 \
    pynvml>=11.5.0 \
    pydantic>=2.7.0 \
    ftfy>=6.0.0 \
    imageio>=2.37.2 \
    imageio-ffmpeg>=0.6.0 \
    peft>=0.13.2 \
    protobuf>=3.20.0 \
    "transformers>=4.52,<5" \
    sentencepiece>=0.1.99 \
    "sageattention>=1.0.0" \
    opencv-python-headless>=4.8.0 \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    python-multipart>=0.0.9 \
    requests>=2.31 \
    triton

# Install git-based deps
RUN pip install --no-cache-dir \
    "diffusers @ git+https://github.com/huggingface/diffusers.git@01de02e8b4f2cc91df4f3e91cb6535ebcbeb490c" \
    "ltx-core @ git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core&egg=ltx-core" \
    "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines&egg=ltx-pipelines"

# Copy runpod scripts
COPY runpod/start.sh /app/start.sh
COPY runpod/watchdog.py /app/watchdog.py

RUN chmod +x /app/start.sh

EXPOSE 8000

WORKDIR /app/backend

CMD ["/app/start.sh"]
