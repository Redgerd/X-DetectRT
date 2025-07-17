# backend/main.py

from fastapi import FastAPI
from config import settings # <--- Ensure 'backend.' prefix
import logging
import redis.asyncio as redis
import asyncio

# middlewares
# from middleware.JWTAuth import JWTAuthenticationMiddleware
# from middleware.ip_middleware import IPMiddleware
from fastapi.middleware.cors import CORSMiddleware

# database stuff
from core.database import test_db_connection # <--- Ensure 'backend.' prefix
# from backend.core.database import SessionLocal # If you use this, ensure 'backend.' prefix

# routers 
from api.auth.routes import router as auth_router # <--- Ensure 'backend.' prefix

# celery
from core.celery.celery_app import celery_app # <--- Ensure 'backend.' prefix

app = FastAPI(
    title="X-DETEXTRT API",
    description="API for deepfake project.",
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
async def startup_db_check(): # Made async to align with FastAPI startup event pattern
    """Test database connection on startup."""
    if test_db_connection(): # test_db_connection is assumed to be synchronous
        logger.info("✅ Database connected successfully.")
    else:
        logger.error("❌ Database connection failed on startup. Exiting...")
        # In a production app, you might want to raise an exception here to prevent startup
        # raise RuntimeError("Database connection failed")

    # Initialize Redis client for general use, even if not directly for pub/sub here
    global redis_client # Declare as global to be accessible in health check
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("FastAPI app started and Redis client initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Performs shutdown tasks: closes Redis connections."""
    if redis_client:
        await redis_client.close()
        logger.info("Redis client closed.")
    logger.info("FastAPI app shutting down.")


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

# WebSocket Routes (No changes as per request to not implement/modify routes)


@app.get("/")
async def root():
    return {"message": "Hello World. This is X-DETECT project!"}


@app.get("/health")
async def health():
    """
    Provides a health check endpoint for the FastAPI application and its dependencies,
    including a simple Celery worker status check.
    """
    logging.info("Health check running...")

    health_status = {"status": "OK"}

    # 1. Redis Connection Check
    redis_ping_status = False
    try:
        health_redis_client = redis.from_url(settings.REDIS_URL)
        redis_ping_status = await health_redis_client.ping()
        await health_redis_client.close()
        health_status["redis_check"] = redis_ping_status
        logging.info("Redis ping: %s", redis_ping_status)
    except Exception as e:
        health_status["redis_check"] = f"Failed: {e}"
        logging.error(f"Redis health check failed: {e}", exc_info=True)


    # 2. SQLAlchemy Database Connection Check
    db_connection_status = test_db_connection()
    health_status["sqlalchemy_check"] = db_connection_status
    logging.info("SQLAlchemy connection check: %s", db_connection_status)


    # 3. Simple Celery Worker Ping Check
    celery_worker_ping_status = "N/A"
    try:
        # --- FIX: REMOVE 'await' here ---
        ping_response = celery_app.control.ping(timeout=1) # No await here
        if ping_response:
            # If any worker responds, it's considered online
            celery_worker_ping_status = "Workers online"
            health_status["celery_worker_ping_response"] = ping_response # Include actual response for detail
            logging.info(f"Celery workers responded: {ping_response}")
        else:
            celery_worker_ping_status = "No workers responded"
            logging.warning("No Celery workers responded to ping.")
    except Exception as e:
        celery_worker_ping_status = f"Ping failed: {e}"
        logging.error(f"Celery worker ping failed: {e}", exc_info=True)
    health_status["celery_worker_status"] = celery_worker_ping_status


    # 4. Torch CUDA availability (kept as per your original code)
    torch_cuda_status = False
    try:
        from torch.cuda import is_available
        torch_cuda_status = is_available()
        logging.info("Torch CUDA available: %s", torch_cuda_status)
    except ImportError:
        logging.warning("Torch is not installed, cannot check CUDA availability.")
        torch_cuda_status = "Torch not installed"
    except Exception as e:
        logging.error(f"Error checking Torch CUDA: {e}", exc_info=True)
        torch_cuda_status = f"Error: {e}"
    health_status["torch_cuda_check"] = torch_cuda_status


    return health_status