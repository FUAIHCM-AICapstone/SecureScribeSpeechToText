import json
import time

import redis
import requests
from redis import ConnectionPool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.utils.logging import logger


def _get_redis_auth_kwargs():
    kwargs = {}
    if getattr(settings, "REDIS_USER", None):
        kwargs["username"] = settings.REDIS_USER
    if getattr(settings, "REDIS_PASSWORD", None):
        kwargs["password"] = settings.REDIS_PASSWORD
    return kwargs


redis_pool = ConnectionPool(host=settings.REDIS_HOST, port=int(settings.REDIS_PORT), db=int(settings.REDIS_DB_S2T), decode_responses=True, retry_on_timeout=True, socket_timeout=10, socket_connect_timeout=10, socket_keepalive=True, health_check_interval=30, max_connections=100, **_get_redis_auth_kwargs())

redis_client = redis.Redis(host=settings.REDIS_HOST, port=int(settings.REDIS_PORT), decode_responses=True, db=int(settings.REDIS_DB_S2T), **_get_redis_auth_kwargs())

try:
    import redis.asyncio as aioredis

    redis_async_client = aioredis.Redis(host=settings.REDIS_HOST, port=int(settings.REDIS_PORT), db=int(settings.REDIS_DB_S2T), decode_responses=True, retry_on_timeout=True, socket_timeout=10, socket_connect_timeout=10, socket_keepalive=True, health_check_interval=30, max_connections=50, **_get_redis_auth_kwargs())
except ImportError:
    redis_async_client = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    retry=retry_if_exception_type(redis.exceptions.ConnectionError),
)
def get_redis_client():
    redis_client.ping()
    return redis_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    retry=retry_if_exception_type(redis.exceptions.ConnectionError),
)
async def get_async_redis_client():
    if redis_async_client is None:
        raise RuntimeError("Async Redis client not available. Please install aioredis.")
    await redis_async_client.ping()
    return redis_async_client


async def publish_to_user_channel(user_id: str, message: dict) -> bool:
    try:
        client = await get_async_redis_client()
        channel = f"user:{user_id}:{message.get('type', 'notification')}"
        import json

        data = json.dumps(message)
        await client.publish(channel, data)
        return True
    except Exception:
        return False


async def get_recent_messages_for_user(user_id: str, limit: int = 10) -> list:
    try:
        client = await get_async_redis_client()
        pattern = f"task_progress:*:{user_id}"
        keys = await client.keys(pattern)
        messages = []
        for key in keys[:limit]:
            data = await client.hgetall(key)
            if data:
                parts = key.split(":")
                task_id = parts[1] if len(parts) >= 3 else None
                message = {
                    "type": "task_progress",
                    "data": {**data, "task_id": task_id},
                }
                messages.append(message)
        messages.sort(key=lambda x: x["data"].get("last_update", ""), reverse=True)
        return messages[:limit]
    except Exception:
        return []


def get_connection_pool_info() -> dict:
    sync_info = {
        "pool_size": redis_pool.max_connections,
    }
    async_info = None
    if redis_async_client:
        async_info = {"configured_max": 50, "status": "active"}
    return {"sync_pool": sync_info, "async_pool": async_info, "timestamp": time.time()}


def cleanup_stale_connections():
    if hasattr(redis_pool, "disconnect"):
        redis_pool.disconnect()


# Task Management Functions
def create_task(task_id: str, filename: str, callback_url: str = None) -> bool:
    """Create task metadata in Redis."""
    try:
        # Ensure all values are valid types for Redis (bytes, str, int, float)
        task_data = {"task_id": str(task_id) if task_id is not None else "", "status": "pending", "progress": 0, "filename": str(filename) if filename is not None else "", "callback_url": str(callback_url) if callback_url is not None else "", "created_at": float(time.time()), "completed_at": "", "error": "", "results": ""}
        redis_client.hset(f"task:{task_id}", mapping=task_data)
        redis_client.expire(f"task:{task_id}", 3600)  # Expire after 1 hour
        return True
    except Exception as e:
        logger.error(f"[REDIS] Error creating task {task_id}: {e}")
        return False


def update_task_status(task_id: str, status: str, progress: int = None, results: dict = None, error: str = None) -> bool:
    """Update task status in Redis."""
    try:
        update_data = {"status": status, "completed_at": time.time()}

        if progress is not None:
            update_data["progress"] = progress
        if results is not None:
            update_data["results"] = json.dumps(results)
        if error is not None:
            update_data["error"] = error

        redis_client.hset(f"task:{task_id}", mapping=update_data)
        return True
    except Exception as e:
        logger.error(f"[REDIS] Error updating task {task_id}: {e}")
        return False


def get_task_status(task_id: str) -> dict:
    """Get task status from Redis."""
    try:
        task_data = redis_client.hgetall(f"task:{task_id}")
        if not task_data:
            return None

        # Convert results back from JSON if present
        if task_data.get("results"):
            try:
                task_data["results"] = json.loads(task_data["results"])
            except json.JSONDecodeError:
                pass

        return task_data
    except Exception as e:
        logger.error(f"[REDIS] Error getting task {task_id}: {e}")
        return None


def send_callback(task_id: str, callback_url: str, status: str, results: dict = None, error: str = None) -> bool:
    """Send HTTP callback to specified URL."""
    try:
        callback_data = {"task_id": task_id, "status": status, "timestamp": time.time()}

        if results:
            callback_data["results"] = results
        if error:
            callback_data["error"] = error

        response = requests.post(callback_url, json=callback_data, timeout=10, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            logger.success(f"[CALLBACK] Successfully sent callback for task {task_id}")
            return True
        else:
            logger.error(f"[CALLBACK] Callback failed for task {task_id}: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"[CALLBACK] Error sending callback for task {task_id}: {e}")
        return False


def cleanup_task_data(task_id: str) -> bool:
    """Clean up task data from Redis."""
    try:
        redis_client.delete(f"task:{task_id}")
        return True
    except Exception as e:
        logger.error(f"[REDIS] Error cleaning up task {task_id}: {e}")
        return False
