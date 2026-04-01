FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install Python 3.12 + system deps (including XAI/MediaPipe/SAM requirements)
RUN apt-get update && \
    apt-get install -y \
        python3-pip python3-dev python-is-python3 \
        ffmpeg libsm6 libxext6 libgl1-mesa-glx \
        libglib2.0-0 \
        cmake build-essential \
        curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    # Install PyTorch with CUDA 11.8 wheels first (GPU-enabled)
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install segment-anything from source (Meta's SAM)
    pip install --no-cache-dir \
        git+https://github.com/facebookresearch/segment-anything.git

# Create checkpoint and data directories
RUN mkdir -p /app/checkpoints /app/data/tcav_concepts

# Copy backend code and .env
COPY backend/ ./backend
COPY .env /app/.env

# Copy wait script
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy sample data (ensure ml_models is inside backend)
COPY data/ /app/data
COPY GenD_PE_L/ /app/GenD_PE_L
COPY ASVspoof/ /app/ASVspoof


EXPOSE 8000

WORKDIR /app/backend

# Run FastAPI (reload for dev)
CMD /wait-for-it.sh postgres:5432 --timeout=5 -- uvicorn main:app --host 0.0.0.0 --port 8000 --reload
