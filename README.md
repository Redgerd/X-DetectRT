[![FastAPI](https://img.shields.io/badge/FastAPI-%2300C7B7.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-37814A.svg?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-%23DC382D.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-%23336791.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23004888.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Langchain](https://img.shields.io/badge/Langchain-%231A202C.svg?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)

## Real-Time Deepfake Detection and Explanation Using Lightweight ML and LLMs
Deepfake content is an increasing threat to digital media authenticity. This project introduces a real-time, explainable deepfake detection pipeline that processes video in memory, minimizing latency while providing interpretable outputs. The system leverages frame reduction techniques, pretrained visual models (e.g., FakeShield), and vision-capable LLMs for generating human-understandable explanations — all without writing any media to disk.

## Objectives

- Perform **real-time deepfake detection** without persistent storage.
- Apply **lightweight motion-based heuristics** (e.g., optical flow, scene detection) to reduce the number of frames analyzed.
- Use pretrained models like **FakeShield** for forgery detection and localization.
- Generate **coherent natural-language explanations** via fine-tuned vision LLMs.
- Serve results instantly through a **FastAPI-based backend**, optionally containerized with Docker.

## System Modules

### 1. Video Input & Streaming

- Supports both **video file upload** and **live frame streaming** via WebSocket.
- Entire pipeline runs **in-memory**, avoiding any disk I/O for performance.

### 2. Frame Selection Pipeline

- **Optical Flow**: Tracks pixel-wise motion (e.g., Farneback or RAFT).
- **Scene Change Detection**: Uses histogram delta or tools like PySceneDetect.
- **Optional**: Background subtraction to eliminate static regions.

> **Goal**: Reduce frame count by 90% while maintaining semantic fidelity.

### 3. Deepfake Detection Engine

**Model**: [FakeShield v1-22b](https://huggingface.co/zhipeixu/fakeshield-v1-22b)  
A multimodal vision-language framework for explainable deepfake detection and localization.

**Input**:
- Frame as a tensor (from OpenCV → NumPy → PyTorch)
- Modules used: `DTE-FDM` and `MFLM`

**Output** *(per frame)*:
- `verdict`: real or fake
- `confidence_score`: e.g., `0.87`
- `forgery_mask`: binary/grayscale image mask (H, W) as `numpy.ndarray` or `torch.Tensor`
- `attention_map` (optional): model attention visualization to highlight decision focus

These outputs are passed to the LLM for final explanation generation.

### 4. Explanation Engine

**Model**: [saakshigupta/deepfake-explainer-new](https://huggingface.co/saakshigupta/deepfake-explainer-new)  
A LLaVA-based adapter fine-tuned to generate deepfake analysis across multiple images.

**Inputs**:
- Original frame
- Forgery mask
- Optional: Attention map or overlay
- Prompt: _"Explain if this frame shows signs of tampering."_

**Output**:
- A **detailed natural language explanation**, highlighting potential manipulation and justifying the verdict.

**Example**:
> “Regions around the mouth and cheek show boundary noise and abnormal motion artifacts, indicating synthetic manipulation.”

### 5. Response API

**Served via FastAPI** as JSON:
```json
{
  "frame_index": 45,
  "verdict": "fake",
  "confidence": 0.87,
  "explanation": "Facial boundary irregularities suggest deepfake generation.",
  "forgery_mask": "<base64-image>",
  "attention_map": "<base64-image>"
}
```

![image](https://github.com/user-attachments/assets/36bff216-3c11-45e6-acd4-bff72eff4020)

## Tech Stack

| Layer              | Stack                         |
|--------------------|-------------------------------|
| API Backend        | FastAPI                       |
| Frame Processing   | OpenCV, FFmpeg                |
| Motion Detection   | Optical Flow, PySceneDetect   |
| Deepfake Detection | FakeShield, PyTorch           |
| LLM Explanation    | Hugging Face, LangChain       |
| Deployment         | Docker (optional)             |


