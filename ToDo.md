deepfake-detector/
│
├── app/
│ ├── main.py # FastAPI entrypoint
│ ├── config.py # Global settings / env vars
│ ├── celery_config.py # Celery broker and backend config
│ ├── tasks/ # Celery tasks (frame/model/LLM)
│ │ ├── **init**.py
│ │ ├── frame_processing.py # Optical flow, scene change
│ │ ├── detection.py # FakeShield inference
│ │ ├── llm_explanation.py # Explanation generation
│ │ └── postprocess.py # Heatmaps, overlays, formatting
│ │
│ ├── api/
│ │ ├── **init**.py
│ │ ├── auth/ # Auth API logic
│ │ │ ├── routes.py
│ │ │ ├── schemas.py
│ │ │ └── security.py
│ │ ├── detection/ # Detection endpoint logic
│ │ │ ├── routes.py
│ │ │ ├── schemas.py
│ │ │ └── utils.py
│ │ └── response/ # Result formatting or downloads
│ │ ├── routes.py
│ │ └── utils.py
│ │
│ ├── core/
│ │ ├── database.py # DB connection & session
│ │ ├── models.py # SQLAlchemy models
│ │ └── schemas.py # Shared Pydantic schemas
│ │
│ ├── services/
│ │ ├── inference.py # Load FakeShield + model utils
│ │ ├── extractor.py # Load video, extract frames
│ │ ├── llm_client.py # Langchain / HuggingFace LLM interface
│ │ └── media_utils.py # Encoding masks, base64 images
│
├── worker.py # Celery worker startup
├── Dockerfile # Backend Docker image
├── docker-compose.yml # Services (FastAPI, Redis, Postgres, etc.)
├── requirements.txt # Python dependencies
├── .env # Environment variables
├── README.md # Project overview
└── frontend/ # (Optional) React / HTML frontend
└── ...
