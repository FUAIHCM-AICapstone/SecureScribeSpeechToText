# SecureScribe Speech-to-Text Microservice - AI Coding Agent Instructions

## Architecture Overview

A FastAPI-based Vietnamese speech-to-text microservice using Google Gemini API for transcription:

- **Transcription Engine**: Google Gemini API (gemini-2.5-flash-preview-09-2025)
- **Processing**: Celery worker with Redis broker for async audio transcription
- **API Layer**: FastAPI with async endpoints and task status tracking via Redis
- **Audio Handling**: FFmpeg for format conversion, direct Gemini file upload

## Core Architecture

### Gemini Transcription Pipeline
```python
# Main transcription flow in app/utils/gemini_transcriber.py::transcribe_audio_with_gemini()
# 1. Read audio file
# 2. Check file size:
#    - < 18MB: Direct upload using bytes (Part.from_bytes)
#    - >= 18MB: Upload via file API
# 3. Send to Gemini with Vietnamese prompt template
# 4. Clean response text (remove brackets, special chars)
# 5. Return Vietnamese transcript

client = Client(api_key=settings.GOOGLE_API_KEY)
if file_size_mb > 18:
    uploaded_file = client.files.upload(file=audio_path, config={"mime_type": "audio/wav"})
    response = client.models.generate_content(model="gemini-2.5-flash-preview-09-2025", contents=[prompt_template, uploaded_file])
else:
    audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
    response = client.models.generate_content(model="gemini-2.5-flash-preview-09-2025", contents=[prompt_template, audio_part])
```

### Async Task Management with Redis
```python
# Task creation in API (app/api/endpoints/audio.py)
task_id = uuid.uuid4()
temp_audio_path = f"/tmp/transcribe_{task_id}{ext}"  # Shared volume between containers
transcribe_audio_task.delay(task_id, temp_audio_path, callback_url)  # Celery

# Redis task tracking (app/utils/redis.py)
# - create_task(task_id, metadata)
# - update_task_status(task_id, "processing"|"completed"|"failed", progress=0-100)
# - get_task_status(task_id) → {status, progress, results, error}
# - Task TTL: 3600 seconds (auto-cleanup)

# Optional callback on completion
send_callback(task_id, callback_url, status, results)  # HTTP POST to callback_url
```

## Key Technical Details

### Gemini API Configuration
- **Model**: `gemini-2.5-flash-preview-09-2025` (multimodal, supports audio)
- **File Upload Limit**: 18MB threshold
  - < 18MB: Direct upload using bytes (Part.from_bytes)
  - >= 18MB: Use file upload API (client.files.upload)
- **Prompt**: Vietnamese-focused instruction set for transcription
- **Response Cleaning**: Remove brackets, special characters, normalize spacing

### WebM Audio Handling
Special case in `app/api/endpoints/audio.py`:
- WebM files converted to WAV via `app/utils/audio_converter.py::convert_webm_to_wav()`
- Uses FFmpeg subprocess (installed in Docker image)
- Cleanup: Remove original WebM after successful conversion

## Developer Workflows

### Local Development
```bash
# Full stack (API + Celery + Redis)
docker-compose up

# API only (assumes external Redis at redis:6379)
uvicorn app.main:app --reload --host 0.0.0.0 --port 9998

# Celery worker only
celery -A app.jobs.celery_worker worker --loglevel=info
```

### Testing Transcription
```bash
# POST to /api/v1/transcribe with multipart/form-data
# - file: audio file (.wav, .mp3, .webm, .m4a, .flac, .ogg, .aac)
# - callback_url: optional URL for completion notification

# Response: {"task_id": "...", "status": "pending"}
# Poll status: GET /api/v1/transcribe/task/{task_id}
```

### Environment Variables (`.env`)
```bash
REDIS_HOST=redis              # Container name in docker-compose
REDIS_PORT=6379
REDIS_DB_S2T=0
GOOGLE_API_KEY=xxx           # Required for Gemini API
```

## Code Conventions

### Logging with ANSI Colors
Throughout codebase, especially `app/utils/gemini_transcriber.py` and `app/api/endpoints/audio.py`:
```python
print("\033[94m[COMPONENT] Info message\033[0m")    # Blue
print("\033[92m[COMPONENT] Success message\033[0m")  # Green
print("\033[93m[COMPONENT] Warning message\033[0m")  # Yellow
print("\033[91m[COMPONENT] Error message\033[0m")    # Red
```

### Ruff Configuration
From `pyproject.toml`:
- Line length: 9999 (effectively unlimited)
- Minimal rules: E, W, F, I, B, C4, UP, ARG001
- Ignore E501 (line too long), B008, W191, UP045/UP006

## Docker Architecture

### Simplified Build
```dockerfile
FROM python:3.11-slim-bullseye
# Key dependencies:
# - FFmpeg for audio conversion
# - google-genai for Gemini API
```

### Container Startup
`start.sh` runs both services with colored output:
```bash
celery -A app.jobs.celery_worker worker --loglevel=info &  # Green [CELERY]
uvicorn app.main:app --host 0.0.0.0 --port 9998 &          # Blue [UVICORN]
```

### Shared Volumes
- Temp audio files: `/tmp/transcribe_{task_id}{ext}` (accessible by both API and worker containers)

## Integration Points

### Celery Task System
- Broker/Backend: Redis (same instance, DB 0)
- Task definition: `app/jobs/tasks.py::transcribe_audio_task`
- Worker: `app/jobs/celery_worker.py` (Celery app instance)
- Retry: None (manual retry via re-POST)

### External Dependencies
- **Google Gemini API**: For audio transcription
- **FFmpeg**: For audio format conversion

### Callback Mechanism
Optional POST to `callback_url` on task completion:
```json
{
  "task_id": "uuid",
  "status": "completed"|"failed",
  "results": {"transcript": "..."},  // if completed
  "error": "...",    // if failed
  "timestamp": "ISO8601"
}
```

## Common Patterns

### Temporary File Cleanup
```python
# Always cleanup in finally block or after success/failure
try:
    # ... process audio_path ...
finally:
    if os.path.exists(audio_path):
        os.unlink(audio_path)
```

## Key Files Reference

- `app/utils/gemini_transcriber.py` - Gemini API integration + transcription logic
- `app/api/endpoints/audio.py` - `/transcribe` endpoint, file handling
- `app/jobs/tasks.py` - Celery async transcription task
- `app/utils/redis.py` - Redis connection pool, task status management
- `app/utils/audio_converter.py` - FFmpeg audio conversion utilities

## Performance Notes

- Gemini API: Cloud-based transcription (depends on network)
- File size handling: Automatic switching to file API for files > 18MB
- Typical 1-minute audio: ~10-30 seconds total processing (depending on network)

