import logging

from celery import Celery

from app.core.config import settings
from app.core.vault_loader import load_config_from_api_v2

# Suppress HTTP request logging from httpx and related libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
load_config_from_api_v2()


celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.jobs.tasks"],  # Explicitly include tasks module
)

# Configure Celery settings for better timeout handling
celery_app.conf.update(
    task_soft_time_limit=300000,
    task_time_limit=600000,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    task_default_retry_delay=60,
    task_max_retries=3,
)
