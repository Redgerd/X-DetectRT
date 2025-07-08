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

- Build a backend pipeline for deepfake detection from user-uploaded videos.
- Extract frames and audio segments for separate, optimized analysis.
- Apply trained ML models to identify visual and audio inconsistencies.
- Generate interpretable results using Explainable AI techniques.
- Integrate LLMs to:
  - Aid in interpreting detection metadata and frame outputs
  - Produce user-friendly, natural language explanations
- Ensure efficient, asynchronous processing with Celery workers.
- Persist videos and detection results in MinIO and PostgreSQL.
- Expose predictions and reports via a FastAPI-based REST API.

## System Architecture

**Components:**

- `FastAPI` – REST API for upload and results  
- `Celery` – Distributed task manager for background tasks  
- `Redis` – Message broker for task queue  
- `MinIO` – S3-compatible storage for all media artifacts  
- `PostgreSQL` – Metadata and result storage  
- `TensorFlow` – Visual/audio deepfake classification  
- `LLMs` – For natural language output  
- `Docker` – For isolated development and deployment environments

![image](https://github.com/user-attachments/assets/4f8d85ee-8237-4e9e-b873-f59d6c5a2900)

## Tech Stack

| Layer              | Technology           |
|--------------------|----------------------|
| API Backend        | FastAPI              |
| Task Queue         | Celery + Redis       |
| Video Processing   | OpenCV, ffmpeg       |
| Audio Processing   | MoviePy, Pydub       |
| ML Inference       | TensorFlow           |
| Explainability     | Grad-CAM, SHAP       |
| LLM Explanation    | Langchain            |
| Data Storage       | PostgreSQL           |
| File Storage       | MinIO                |
| Deployment         | Docker               |

## Deepfake Detection Pipeline

1. User uploads a video via FastAPI.
2. Video is stored in MinIO.
3. Video is split into frames and audio chunks.
4. Celery workers process frames and audio independently.
5. Trained ML models evaluate content for deepfake characteristics.
6. Grad-CAM and SHAP explain important features influencing predictions.
7. LLM generates natural-language explanation from results.
8. Results and outputs are returned via API (JSON + visuals).


- **Explanation Generation**:  
  Given model scores, timestamps, and highlighted regions, an LLM (e.g., GPT-4) produces a human-readable explanation like:  
  _“Unnatural lip movement detected between 00:10 and 00:22, strongly suggesting manipulation.”_

- **(Optional) Frame/Metadata Evaluation**:  
  LLMs like LLaVA may evaluate image/text prompts to supplement detection models for fine-grained cues.
