FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# RUN nvidia-smi

# # python 3.12
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y libgl1-mesa-glx

WORKDIR /app

# requirements for backend
COPY requirements.txt .
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# pytorch with gpu support
RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# backend code and .env file
COPY backend/ ./backend
COPY .env /app/.env

# add scripts and make them executable
COPY start_workers.sh /app/start_workers.sh
RUN chmod +x /app/start_workers.sh

COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# add sample data for testing
COPY data/ /app/data

# expose port
EXPOSE 8000

# run fastapi server with reload, after waiting for postgres
WORKDIR /app/backend
CMD /wait-for-it.sh postgres:5432 --timeout=5 -- uvicorn main:app --host 0.0.0.0 --port 8000 --reload
