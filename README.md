## Problem Statement
Deepfake content is a growing threat to the authenticity of digital media. This project aims to detect manipulated (deepfake) videos by analyzing both visual and audio components using machine learning models. The backend system is designed to provide accurate and explainable results to help verify media authenticity.

## Objectives

- Build a backend system to detect deepfakes in uploaded videos.
- Extract frames and audio from videos for individual analysis.
- Apply trained ML models to assess the authenticity of content.
- Use Explainable AI (XAI) to generate interpretable results.
- Implement asynchronous and distributed processing using Celery.
- Store and serve detection results through an API.
- (Optional) Use AI agents to manage decision-making and result exp

## System Architecture

**Components:**

- FastAPI (API server)
- Celery (task manager)
- Redis (message broker)
- Video/Audio processing module
- ML models (for visual & audio deepfake detection)
- XAI module (e.g., Grad-CAM, SHAP)
- Database (PostgreSQL or MongoDB)
- Optional AI Agent layer (e.g., crewAI)

## Tech Stack

| Layer              | Technology           |
|--------------------|----------------------|
| API Backend        | FastAPI              |
| Task Queue         | Celery + Redis       |
| Video Processing   | OpenCV, ffmpeg       |
| Audio Processing   | MoviePy, Pydub       |
| ML Inference       | PyTorch / TensorFlow |
| Explainability     | Grad-CAM, SHAP       |
| AI Agents (opt.)   | crewAI, langchain    |
| Data Storage       | PostgreSQL / MongoDB |
| Deployment         | Docker               |

## Deepfake Detection Pipeline

1. User uploads a video via FastAPI.
2. Video is split into frames and audio segments.
3. Celery sends frame/audio tasks to background workers.
4. ML models evaluate both components for manipulation.
5. Results are combined into a final decision.
6. XAI module explains the predictions.
7. API returns detection score, verdict, and explanation.

## Explainable AI (XAI)

- **Purpose**: Help users understand why a video is classified as fake or real.
- **Techniques Used**:
  - Grad-CAM: Heatmaps over suspicious facial regions
  - SHAP/LIME: Feature importance for audio or tabular data
- **Outputs**:
  - Visual overlays on frames
  - Text-based summaries
  - Optional PDF reports

## AI Agents (Optional Module)

- Automatically select which models to run (audio, video, or both)
- Generate human-readable summaries of results
- Manage multi-step inference workflows
- Tools: `crewAI`, `langchain`, or custom logic + LLMs

## Machine Learning Overview

- Visual model: CNN-based model trained on deepfake datasets (e.g., DFDC)
- Audio model: Voice classification using spectrograms or MFCCs
- Evaluation metrics: Accuracy, precision, recall, F1-score
- Combined score: Weighted average of frame and audio scores

## Output Example

- Verdict: **Fake**
- Confidence: **91.3%**
- Fake Frames: **72%**
- Audio Score: **85% fake likelihood**
- XAI Summary: "Fake regions detected in facial area between 00:10–00:25"

## Security & Scalability

- JWT-based authentication (optional)
- Rate limiting to avoid abuse
- Docker-based containerization
- Ready for GPU inference and cloud deployment

## Future Improvements

- Add real-time video streaming support
- Use reinforcement learning agents for adaptive decisions
- Extend to detect image and text-based manipulations
- Deploy to cloud (AWS/GCP) with Kubernetes
