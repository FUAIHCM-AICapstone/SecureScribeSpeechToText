import os
import shutil
from typing import Set

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.jobs.tasks import transcribe_audio_task
from app.utils.audio_converter import convert_webm_to_wav
from app.utils.logging import logger
from app.utils.models import MODEL_ID
from app.utils.redis import create_task, get_task_status

router = APIRouter(prefix=settings.API_V1_STR, tags=["Audio"])


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), callback_url: str = Form(None)):
    """
    Upload an audio file and get transcription using Google Gemini API.

    Args:
        file: Audio file to transcribe (supports WAV, MP3, M4A, FLAC, etc.)
        callback_url: Optional callback URL for completion notification

    Returns:
        JSON response with task ID and status information
    """
    logger.info(f"[API] Starting background transcription for file: {file.filename}")

    # Validate file type
    allowed_extensions: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
    file_extension = os.path.splitext(file.filename or "")[1].lower()

    if file_extension not in allowed_extensions:
        logger.error(f"[API] Unsupported file format: {file_extension}")
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}")

    logger.success(f"[API] File format validated: {file_extension}")

    # Generate unique task ID
    import uuid

    task_id = str(uuid.uuid4())

    try:
        logger.info(f"[API] Creating temporary file for task_id={task_id}")

        # Create temporary file that will persist until task completes
        # File will be accessible by both API and Celery worker containers via shared volume
        temp_audio_path = f"/tmp/transcribe_{task_id}{file_extension}"

        # Save uploaded file to temp location
        logger.info(f"[API] Saving file to: {temp_audio_path}")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.success("[API] File saved successfully")

        # Handle WebM conversion
        if file_extension == ".webm":
            logger.info("[API] WebM file detected, validating audio stream")

            try:
                # Convert WebM to WAV
                logger.info("[API] Converting WebM to WAV format")
                wav_path = f"/tmp/transcribe_{task_id}.wav"

                if not convert_webm_to_wav(temp_audio_path, wav_path):
                    logger.error("[API] Failed to convert WebM to WAV")
                    # Clean up both WebM and any partial WAV files
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                    raise HTTPException(status_code=500, detail="Failed to convert WebM to WAV")

                # Clean up original WebM file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    logger.info("[API] Cleaned up original WebM file")

                # Update temp_audio_path to point to converted WAV file
                temp_audio_path = wav_path
                logger.success("[API] WebM conversion completed successfully")

            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logger.error(f"[API] WebM processing failed: {str(e)}")
                # Clean up files on error
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                wav_path = f"/tmp/transcribe_{task_id}.wav"
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                raise HTTPException(status_code=500, detail=f"Failed to process WebM file: {str(e)}")

        # Create task metadata in Redis
        logger.info(f"[API] Creating task metadata for task_id={task_id}")
        if not create_task(task_id, file.filename, callback_url):
            logger.error(f"[API] Failed to create task metadata for task_id={task_id}")
            # Clean up temp file on error
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise HTTPException(status_code=500, detail="Failed to create task")

        # Enqueue background task
        logger.info(f"[API] Enqueuing background task for task_id={task_id}")
        transcribe_audio_task.delay(task_id=task_id, audio_path=temp_audio_path, callback_url=callback_url)

        logger.success(f"[API] Background task enqueued successfully for task_id={task_id}")

        # Prepare response
        response_data = {"success": True, "message": "Audio transcription task enqueued successfully", "data": {"task_id": task_id, "filename": file.filename, "status": "pending", "progress": 0, "callback_url": callback_url, "polling_url": f"/api/v1/transcribe/task/{task_id}"}}

        logger.success(f"[API] Returning task information for task_id={task_id}")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"[API] Error enqueuing transcription task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enqueuing transcription task: {str(e)}")


@router.get("/transcribe/task/{task_id}")
async def get_task_status_endpoint(task_id: str):
    """
    Get the status and results of a transcription task.

    Args:
        task_id: Unique task identifier

    Returns:
        JSON response with task status, progress, and results if completed
    """
    logger.info(f"[API] Checking task status for task_id={task_id}")

    # Get task status from Redis
    task_data = get_task_status(task_id)

    if not task_data:
        logger.error(f"[API] Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    logger.success(f"[API] Task status retrieved: {task_data.get('status', 'unknown')}")

    # Prepare response
    response_data = {"success": True, "message": "Task status retrieved successfully", "data": {"task_id": task_data.get("task_id"), "status": task_data.get("status"), "progress": task_data.get("progress", 0), "filename": task_data.get("filename"), "created_at": task_data.get("created_at"), "completed_at": task_data.get("completed_at"), "error": task_data.get("error") if task_data.get("error") else None, "results": task_data.get("results")}}

    logger.success(f"[API] Returning task status for task_id={task_id}")
    return JSONResponse(content=response_data)


@router.get("/transcribe/status")
async def get_transcription_status():
    """
    Get status information about the transcription service.

    Returns:
        JSON response with service status and configuration
    """
    return {"success": True, "message": "Transcription service is available", "data": {"supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"], "transcription_engine": "Google Gemini API", "model": MODEL_ID}}
