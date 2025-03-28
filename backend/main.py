# base imports
from fastapi import FastAPI
from config import settings
import logging
import redis
import asyncio 

# middlewares
# from middleware.JWTAuth import JWTAuthenticationMiddleware  
# from middleware.ip_middleware import IPMiddleware  
from fastapi.middleware.cors import CORSMiddleware

# database stuff
from core.database import test_db_connection  

# routers (uncomment when needed)
from api.auth.routes import router as auth_router  
from api.cameras.routes import router as cameras_router
from api.users.routes import router as users_router
from api.user_cameras.routes import router as user_cameras_router
from api.alerts.routes import router as alerts_router
from api.intrusion.routes import router as intrusion_router

# websocket for alerts
from api.alerts.websocket import router as alerts_websocket_router, start_redis_listener
from api.cameras.websocket import router as cameras_websocket_router, start_redis_frame_listener

# celery
from core.celery.worker import celery_app

app = FastAPI(
    title="Smart-Campus API",
    description="API for the Smart Campus project.",
    version="1.0.0",
    openapi_tags=[{"name": "Auth", "description": "Authentication related endpoints"}],
    # Adding the security schema directly to Swagger UI
    openapi_security=[{
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }]
)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_db_check():
    """Test database connection on startup."""
    if test_db_connection():
        logger.info("✅ Database connected successfully.")
    else:
        logger.error("❌ Database connection failed on startup. Exiting...")
        exit(1)


# IP middleware
# app.add_middleware(IPMiddleware)
# Authentication middleware
# app.add_middleware(JWTAuthenticationMiddleware)
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend URLs for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Routes
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(cameras_router)
app.include_router(user_cameras_router)
app.include_router(alerts_router)
app.include_router(intrusion_router)

# WebSocket Routes
app.include_router(alerts_websocket_router)
app.include_router(cameras_websocket_router)


# routes for startup and some for testing and health checks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_redis_listener())
    asyncio.create_task(start_redis_frame_listener())


@app.get("/")
async def root():
    return {"message": "Hello World. This is the Smart Campus project!"}


@app.get("/health")
async def health():
    result = celery_app.send_task("core.celery.tasks.add", (4,4))
    red = redis.from_url(settings.REDIS_URL)

    return  {
                "status": "OK",
                "celery_calculation": result.get(),
                "redis_check": red.ping(),
                "sqlalchemy_check": test_db_connection()
            }

import numpy as np
import cv2
import logging
import redis
from core.celery.stream_worker import publish_frame
from core.celery.model_worker import process_frame

redis_client = redis.Redis(host="localhost", port=6379, db=0)


@app.post("/test_publish_feed/{camera_id}")
async def test_publish_feed(camera_id: int):
    """
    Generates a synthetic random frame, processes it, and publishes the result.
    """

    try:
      
        # Create a random frame (720p resolution)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # _, jpeg_bytes = cv2.imencode('.jpg', frame)
        # frame = jpeg_bytes.tobytes()
        # Process the synthetic frame
        # result = publish_frame(camera_id, frame)
        result = process_frame(camera_id, frame.tolist())

        if result:
            return {"status": "Success", "message": "Frame published successfully"}

    except Exception as e:
        logging.error(f"Error in test_publish_feed: {str(e)}", exc_info=True)
        return {"status": "Error", "message": str(e)}
    

from core.celery.model_worker import update_cameras_for_model_workers
from celery import group

@app.post("/update_cameras_for_model_workers")
async def update_cameras():
    """
    Updates the cameras list from the database. Use whenever there is a change in the cameras (add, update, remove).
    Use with priority=0 to ensure that the cameras list is updated before processing any additional frames.
    """
    try:
        # Call the Celery task to update cameras
        task_group = group(update_cameras_for_model_workers.s() for _ in range(settings.MODEL_WORKERS))
        task_group.apply_async(queue='model_tasks')
        return {"status": "Success", "message": "Cameras updated successfully"}
    except Exception as e:
        logging.error(f"Error updating cameras: {str(e)}", exc_info=True)
        return {"status": "Error", "message": str(e)}