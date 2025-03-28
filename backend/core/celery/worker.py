from celery import Celery
from config import settings
# Create Celery instance

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_time_limit=30,
    broker_transport_options={"visibility_timeout": 3600},
    worker_heartbeat=60,
)

# Auto-discover tasks in alert_tasks and tasks
celery_app.autodiscover_tasks(["core.celery.alert_tasks","core.celery.frame_tasks", "core.celery.tasks"])
