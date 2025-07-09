FROM python:3.12-slim

# requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
# use cache volume to speed up the build
# RUN pip install --cache-dir=/root/.cache/pip --prefer-binary -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# cv2 requirements
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# copy the backend code and environment variables
COPY backend/ ./backend
COPY .env /app/.env

# get the bash scripts and make them executable
COPY start_workers.sh /app/start_workers.sh
RUN chmod +x /app/start_workers.sh

COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

# run the server
WORKDIR /app/backend
EXPOSE 8000
CMD /wait-for-it.sh postgres:5432 --timeout=5 -- uvicorn main:app --host 0.0.0.0 --port 8000 --reload