import logging
import os
from typing import Any, Dict

from app.jobs.celery_worker import celery_app
from app.utils.gemini_transcriber import transcribe_audio_with_gemini
from app.utils.redis import get_redis_client, send_callback, update_task_status

# Setup logging
logger = logging.getLogger(__name__)


# Sync Redis client for Celery tasks
sync_redis_client = get_redis_client()


@celery_app.task
def transcribe_audio_task(task_id: str, audio_path: str, callback_url: str = None) -> Dict[str, Any]:
    """
    Process audio transcription in background using Gemini API.

    This task handles the transcription pipeline and sends callbacks on completion.
    """
    logger.info(f"Starting background transcription for task_id={task_id}, file={audio_path}")

    try:
        # Check if file exists before processing (file should be available via shared volume)
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(f"Task {task_id} failed: {error_msg}")
            update_task_status(task_id, "failed", error=error_msg)
            if callback_url:
                send_callback(task_id, callback_url, "failed", error=error_msg)
            return {"status": "failed", "error": error_msg}

        # Update task status to processing
        update_task_status(task_id, "processing", progress=10)

        # Run transcription with Gemini
        logger.info(f"Running Gemini transcription for task_id={task_id}")
        transcript = transcribe_audio_with_gemini(audio_path)

        if not transcript:
            error_msg = "Empty transcription received from Gemini"
            logger.error(f"Task {task_id} failed: {error_msg}")
            update_task_status(task_id, "failed", error=error_msg)
            if callback_url:
                send_callback(task_id, callback_url, "failed", error=error_msg)
            return {"status": "failed", "error": error_msg}

        # Update task status to completed
        results = {"transcript": transcript}
        update_task_status(task_id, "completed", progress=100, results=results)

        # Send callback if URL provided
        if callback_url:
            send_callback(task_id, callback_url, "completed", results=results)

        logger.info(f"Background transcription completed for task_id={task_id}")

        # Clean up temp file
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info(f"Cleaned up temp file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {audio_path}: {e}")

        return {"status": "completed", "results": results}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Background transcription failed for task_id={task_id}: {error_msg}")

        # Update task status to failed
        update_task_status(task_id, "failed", error=error_msg)

        # Send callback if URL provided
        if callback_url:
            send_callback(task_id, callback_url, "failed", error=error_msg)

        # Clean up temp file on error
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file {audio_path} after error: {cleanup_error}")

        return {"status": "failed", "error": error_msg}
