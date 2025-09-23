# backend/core/celery/celery_app.py
from celery import Celery, current_app
from celery.signals import worker_process_init
from config import settings
import os
import logging
from kombu import Queue # <--- ADD THIS IMPORT

logger = logging.getLogger(__name__)
# ... (rest of logger setup) ...

celery_app = Celery(
    "deepfake_detector",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_time_limit=300,
    broker_transport_options={"visibility_timeout": 3600},
    worker_heartbeat=60,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,

    # --- FIX: Define Queues as kombu.Queue objects ---
    task_queues = (
        Queue('default', routing_key='default'),
        Queue('frame_selection_queue', routing_key='frame_selection'),
        Queue('deepfake_detection_queue', routing_key='deepfake_detection'),
        Queue('websocket_messages_queue', routing_key='websocket_messages'),
    ),
    task_default_queue = 'default',
    task_default_exchange = 'tasks',
    task_default_routing_key = 'default',

    task_routes = {
        'frame_selection_pipeline.run': {'queue': 'frame_selection_queue'},
        'backend.core.celery.detection_tasks.perform_detection': {'queue': 'deepfake_detection_queue'},
    }
)
# Autodiscover tasks from specified modules
celery_app.autodiscover_tasks([
    "core.celery.frame_selection",
    "core.celery.detection_tasks",
    "core.celery.tasks",
], force=True)