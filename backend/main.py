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
from core.database import SessionLocal
from models import Camera

# routers (uncomment when needed)
from api.auth.routes import router as auth_router  
from api.cameras.routes import router as cameras_router
from api.users.routes import router as users_router
from api.user_cameras.routes import router as user_cameras_router
from api.intrusion.routes import router as intrusion_router

# websocket for alerts
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
app.include_router(intrusion_router)

# WebSocket Routes
app.include_router(cameras_websocket_router)


# routes for startup and some for testing and health checks
@app.on_event("startup")
async def startup_event():
    # getting all cam ids:
    db = SessionLocal()
    cameras = db.query(Camera).all()
    camera_ids = [camera.id for camera in cameras]
    for camera_id in camera_ids:
        redis_client.set(f"camera_{camera_id}_websocket_active", "False")
    asyncio.create_task(start_redis_listener())
    asyncio.create_task(start_redis_frame_listener())
    logging.info("Redis listener started.")


@app.get("/")
async def root():
    return {"message": "Hello World. This is the Smart Campus project!"}


@app.get("/health")
async def health():
    logging.info("Health check running...")

    result = celery_app.send_task("core.celery.tasks.add", (4,4))
    from torch.cuda import is_available

    logging.info("Torch CUDA available: %s", is_available())
    red = redis.from_url(settings.REDIS_URL)

    logging.info("Redis ping: %s", red.ping())

    logging.info("SQLAlchemy connection check: %s", test_db_connection())

    return  {
                "status": "OK",
                "celery_calculation": result.get(),
                "redis_check": red.ping(),
                "sqlalchemy_check": test_db_connection(),
                "torch_cuda_check": is_available(),
            }

import numpy as np
import cv2
import logging
import redis


redis_client = redis.from_url(settings.REDIS_URL)


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
    


# @app.post("/update_cameras_for_model_workers")
# async def update_cameras():
#     """
#     Updates the cameras list from the database. Use whenever there is a change in the cameras (add, update, remove).
#     Use with priority=0 to ensure that the cameras list is updated before processing any additional frames.
#     """
#     try:
#         # Call the Celery task to update cameras
#         task_group = group(update_cameras_for_model_workers.s() for _ in range(settings.MODEL_WORKERS))
#         task_group.apply_async(queue='model_tasks')
#         return {"status": "Success", "message": "Cameras updated successfully"}
#     except Exception as e:
#         logging.error(f"Error updating cameras: {str(e)}", exc_info=True)
#         return {"status": "Error", "message": str(e)}
    
from core.celery.full_feed_worker import full_feed_worker_app

@app.post("/gen-intr/{camera_id}")
def generate_intrusion(camera_id: int) :
    """
    Generate sample intr that unsets after 5 seconds
    """
    full_feed_worker_app.send_task('core.celery.full_feed_worker.set_intrusion_flag', args=[camera_id], queue="feed_tasks")
    return {"status" : f"Generated sample intrusion at camera {camera_id}"}
