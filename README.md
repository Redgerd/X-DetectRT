[![FastAPI](https://img.shields.io/badge/FastAPI-%2300C7B7.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-37814A.svg?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-%23DC382D.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![MinIO](https://img.shields.io/badge/MinIO-EF3C3C.svg?style=for-the-badge&logo=minio&logoColor=white)](https://min.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%23336791.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23004888.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Langchain](https://img.shields.io/badge/Langchain-%231A202C.svg?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)


## Problem Statement
Deepfake content is a growing threat to the authenticity of digital media. This project aims to detect manipulated (deepfake) videos by analyzing both visual and audio components using machine learning models. The backend system is designed to provide accurate and explainable results to help verify media authenticity.

## Objectives


- Build a backend system to detect deepfakes in uploaded videos.
- Extract frames and audio from videos for individual analysis.
- Apply trained ML models to assess the authenticity of content.
- Use Explainable AI (XAI) to generate interpretable results.
- Implement asynchronous and distributed processing using Celery.
- Store and serve detection results through an API using object storage.
- (Optional) Use AI agents to manage decision-making and generate summaries.

## System Architecture

**Components:**

- `FastAPI` – API server for upload and response
- `Celery` – Distributed task queue manager
- `Redis` – Message broker for Celery
- `Object Storage` – MinIO for storing video, frames, audio, and outputs
- `ML Models` – Trained visual and audio models for detection
- `XAI Module` – Grad-CAM, SHAP for explainability
- `Database` – PostgreSQL for metadata, scores, and user info
- `AI Agents` *(Optional)* – Decision logic via crewAI or langchain

![image](https://github.com/user-attachments/assets/afe712c2-918f-4a05-be05-60ff955e6430)


## Tech Stack

| Layer              | Technology           |
|--------------------|----------------------|
| API Backend        | FastAPI              |
| Task Queue         | Celery + Redis       |
| Video Processing   | OpenCV, ffmpeg       |
| Audio Processing   | MoviePy, Pydub       |
| ML Inference       | TensorFlow           |
| Explainability     | Grad-CAM, SHAP       |
| AI Agents (opt.)   | crewAI, langchain    |
| Data Storage       | PostgreSQL           |
| File Storage       | MinIO                |
| Deployment         | Docker               |

## Deepfake Detection Pipeline

1. User uploads a video via FastAPI.
2. Video is stored in object storage (e.g., MinIO).
3. Video is split into frames and audio.
4. Celery workers process frames and audio asynchronously.
5. ML models run inference to detect manipulation.
6. Scores are aggregated and explained using XAI.
7. Results and visuals are saved and returned via API.

## Explainable AI (XAI)

- **Purpose:** Enable transparency behind predictions.
- **Methods Used:**
  - `Grad-CAM` – Heatmap for CNN layers in visual model
  - `SHAP` / `LIME` – Feature importance for tabular/audio data
- **Outputs:**
  - Frame-wise heatmaps
  - Text-based explanation summaries
  - Optional downloadable reports (PDF/JSON

## AI Agents (Optional Module)

- Dynamically select detection path (audio/video/both).
- Generate human-like explanations using LLMs.
- Tools: `crewAI`, `langchain`, or custom agents.

## Machine Learning Overview

- **Visual Model:** CNN-based model trained on FaceForensics++, DFDC.
- **Audio Model:** Spectrogram or MFCC-based voice classifiers.
- **Metrics:** Accuracy, precision, recall, F1-score.
- **Scoring:** Weighted average of visual and audio model results.

## Output Example

- Verdict: **Fake**
- Confidence: **91.3%**
- Fake Frames: **72%**
- Audio Score: **85% fake likelihood**
- XAI Summary: "Fake regions detected in facial area between 00:10–00:25"
- Heatmap_url: "https://yourstorage.com/outputs/video123_heatmap.png"

## Security & Scalability

- Optional JWT authentication for secure API access.
- Rate limiting and request validation.
- Docker-based containerization for local and cloud deployment.
- Compatible with GPU acceleration.
- Scalable with object storage + task queue

## Future Improvements

- Add real-time video streaming support
- Use reinforcement learning agents for adaptive decisions
- Deploy to cloud (AWS/GCP) with Kubernetes
