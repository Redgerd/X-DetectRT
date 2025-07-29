FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Install Python 3.12 + system deps
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 ffmpeg libsm6 libxext6 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code and .env
COPY backend/ ./backend
COPY .env /app/.env

# Copy wait script
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# Copy sample data (ensure ml_models is inside backend)
COPY data/ /app/data

EXPOSE 8000

WORKDIR /app/backend

# Run FastAPI (reload for dev)
CMD /wait-for-it.sh postgres:5432 --timeout=5 -- uvicorn main:app --host 0.0.0.0 --port 8000 --reload
