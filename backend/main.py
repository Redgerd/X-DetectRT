# backend/main.py

# Imports
import asyncio
import logging

from fastapi import FastAPI,  WebSocket
from fastapi.middleware.cors import CORSMiddleware

# import redis.asyncio as redis

#from config import settings

# Internal Imports
# from core.database import test_db_connection
# from core.celery.celery_app import celery_app
# from core.celery.frame_selection import extract_faces_with_optical_flow
import os, uuid
# Model loading is handled by Celery workers, not the main FastAPI app
# confirm_model_loaded();

# FastAPI App Setup
app = FastAPI(
    title="X-DETECT-RT API",
    description="API for deepfake project.",
    version="1.0.0",
    openapi_tags=[{"name": "Auth", "description": "Authentication related endpoints"}],
    openapi_security=[{
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }]
)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and Shutdown Events
# @app.on_event("startup")
# async def startup_db_check():
#     if test_db_connection():
#         logger.info("✅ Database connected successfully.")
#     else:
#         logger.error("❌ Database connection failed on startup.")

#     global redis_client
#     try:
#         redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
#         logger.info("Redis client initialized.")
#     except Exception as e:
#         logger.error(f"❌ Redis connection failed: {e}", exc_info=True)


# @app.on_event("shutdown")
# async def shutdown_event():
#     if redis_client:
#         await redis_client.close()
#         logger.info("Redis client closed.")
#     logger.info("App shutdown complete.")
    
# Routes
from  api.auth.routes import router as auth_router
# from api.video.websocket import router as video_ws_router
# from api.video.routes import router as video_router

# app.include_router(video_ws_router)
# app.include_router(video_router)

# Routes
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Hello World. This is X-DETECT project!"}


# @app.get("/health")
# async def health():
#     logging.info("Health check running...")

#     health_status = {"status": "OK"}

#     # Redis Check
#     try:
#         health_redis_client = redis.from_url(settings.REDIS_URL)
#         redis_ping_status = await health_redis_client.ping()
#         await health_redis_client.close()
#         health_status["redis_check"] = redis_ping_status
#         logger.info(f"Redis ping: {redis_ping_status}")
#     except Exception as e:
#         health_status["redis_check"] = f"Failed: {e}"
#         logger.error(f"Redis health check failed: {e}", exc_info=True)

#     # Database Check
#     db_status = test_db_connection()
#     health_status["sqlalchemy_check"] = db_status
#     logger.info(f"Database check: {db_status}")

#     # Celery Worker Check
#     try:
#         ping_response = celery_app.control.ping(timeout=1)
#         health_status["celery_worker_ping_response"] = ping_response or "No response"
#         logger.info(f"Celery worker ping: {ping_response}")
#     except Exception as e:
#         health_status["celery_worker_ping_response"] = f"Failed: {e}"
#         logger.error(f"Celery ping failed: {e}", exc_info=True)

#     return health_status


