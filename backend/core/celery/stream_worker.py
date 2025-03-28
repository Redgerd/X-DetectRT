import json
import redis
import logging
from celery import Celery
from config import settings

# Redis client for pub/sub
stream_worker_app = Celery('stream_worker', broker=settings.REDIS_URL, backend=settings.REDIS_URL)
stream_worker_app.conf.update(
    task_time_limit=60,  
    broker_transport_options={'visibility_timeout': 3600},
    worker_heartbeat=60,
)

# redis client
redis_client = redis.from_url(settings.REDIS_URL)

@stream_worker_app.task
def publish_frame(camera_id: int, annotated_frame: bytes):
    try:
        logging.info(f"Publishing frame for camera_id: {camera_id}")

        if not isinstance(annotated_frame, bytes):
            logging.error("Frame is not in bytes format.")
            return {"error": "Invalid frame format"}

        # Publish to Redis as raw bytes
        redis_client.publish(f"camera_{camera_id}", annotated_frame)

        logging.info("Frame published successfully.")
        return {"status": "Frame published successfully"}

    except Exception as e:
        logging.exception(f"Error publishing frame: {e}")
        return {"error": "Failed to publish frame"}