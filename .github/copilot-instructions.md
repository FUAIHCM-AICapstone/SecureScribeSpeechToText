# SecureScribe Speech-to-Text Microservice - AI Coding Agent Instructions

## Architecture Overview

A FastAPI-based Vietnamese speech-to-text microservice using Conformer-CTC architecture with speaker diarization capabilities:

- **Core Model**: Custom Conformer encoder with CTC decoding + KenLM n-gram language model (6gram)
- **Diarization**: PyAnnote.audio 3.3.2 for speaker identification
- **Processing**: Celery worker with Redis broker for async audio transcription
- **Decoding**: Dual strategy - beam search with KenLM LM integration, fallback to greedy decoding
- **API Layer**: FastAPI with async endpoints and task status tracking via Redis

## Critical Architecture Patterns

### Model Pipeline Architecture
```python
# Dual-decoder approach in app/models/model_ctc.py
# 1. Greedy decoding: Fast, no external dependencies
# 2. Beam search decoding: Uses pyctcdecode + KenLM 6-gram model

# Model initialization happens ONCE on worker startup (not per-request)
# Located in: app/utils/s2t.py::create_model()
model = ModelCTC(encoder_params, tokenizer_params, training_params, decoding_params)
model.load(checkpoint_path)  # Loads checkpoints_56_90h.ckpt

# Beam search requires:
# - ngram_path: "data/6gram_lm_corpus.binary" (100+ character offset mapping)
# - pyctcdecode with KenLM backend (built from source in Dockerfile)
```

### Speaker Diarization + Transcription Flow
```python
# Complete pipeline in app/utils/diarization.py::diarize_and_transcribe_audio()
# 1. PyAnnote diarization → segments with (start, end, speaker_label)
# 2. Extract audio segments per speaker (min 1 second)
# 3. Per-segment transcription with beam search decoder
# 4. Return structured: [{speaker, start_time, end_time, transcription}]

# Audio chunking for long files (>956000 samples @ 16kHz ≈ 59s)
# - Overlap: 1 second (16000 samples) between chunks
# - Prevents word cutting at boundaries
```

### Async Task Management with Redis
```python
# Task creation in API (app/api/endpoints/audio.py)
task_id = uuid.uuid4()
temp_audio_path = f"/tmp/transcribe_{task_id}{ext}"  # Shared volume between containers
transcribe_audio_task.delay(task_id, temp_audio_path, ...)  # Celery

# Redis task tracking (app/utils/redis.py)
# - create_task(task_id, metadata)
# - update_task_status(task_id, "processing"|"completed"|"failed", progress=0-100)
# - get_task_status(task_id) → {status, progress, results, error}
# - Task TTL: 3600 seconds (auto-cleanup)

# Optional callback on completion
send_callback(task_id, callback_url, status, results)  # HTTP POST to callback_url
```

## Key Technical Details

### Model Architecture (EfficientConformerCTCSmall)
- **Encoder**: 15 Conformer blocks with grouped attention (4 heads, kernel_size=15)
- **Striding**: Applied at blocks 4 & 9 (temporal downsampling)
- **Dimensions**: Progressive expansion [120, 168, 240]
- **Tokenizer**: SentencePiece BPE with 1024 vocab size (`datasets/Vietnamese/vi_bpe_1024.model`)
- **Audio**: 80 mel-filterbanks, 16kHz sampling, 25ms windows, 10ms hop

### Beam Search Decoding with KenLM
Located in `app/models/model_ctc.py::beam_search_decoding()`:
```python
# Character mapping: CTC blank (idx 0) + vocab tokens (idx 1-1023)
labels = [""] + [chr(idx + ngram_offset) for idx in range(1, vocab_size)]
# ngram_offset=100 from config (maps token IDs to printable ASCII range)

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path="data/6gram_lm_corpus.binary",  # 6-gram Vietnamese corpus
    alpha=0.4,  # LM weight
    beta=1.0    # Word insertion bonus
)
```

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
# Poll status: GET /api/v1/status/{task_id}
```

### Configuration Files
- **Model configs**: `app/core/configs/*.json` (15+ variants: Conformer/EfficientConformer, Small/Medium/Large)
- **Default**: `EfficientConformerCTCSmall.json` (checkpoint: `checkpoints_56_90h.ckpt`)
- **Tokenizer**: `datasets/Vietnamese/vi_bpe_1024.model` (SentencePiece)
- **LM**: `6gram_lm_corpus.binary` (Vietnamese n-gram, mapped to `/app/data/` in container)

### Environment Variables (`.env`)
```bash
REDIS_HOST=redis              # Container name in docker-compose
REDIS_PORT=6379
REDIS_DB_S2T=0
HF_TOKEN=hf_xxx              # Required for pyannote.audio diarization
GOOGLE_API_KEY=xxx           # Optional for future integrations
```

## Code Conventions

### Logging with ANSI Colors
Throughout codebase, especially `app/utils/diarization.py` and `app/api/endpoints/audio.py`:
```python
print("\033[94m[COMPONENT] Info message\033[0m")    # Blue
print("\033[92m[COMPONENT] Success message\033[0m")  # Green
print("\033[93m[COMPONENT] Warning message\033[0m")  # Yellow
print("\033[91m[COMPONENT] Error message\033[0m")    # Red
```

### Warnings Suppression
All modules importing torchaudio/pyannote suppress deprecation warnings (consistent pattern):
```python
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated", category=UserWarning)
# ... 7 more torchaudio/torchcodec warnings
```

### Ruff Configuration
From `pyproject.toml`:
- Line length: 9999 (effectively unlimited)
- Minimal rules: E, W, F, I, B, C4, UP, ARG001
- Ignore E501 (line too long), B008, W191, UP045/UP006

## Docker Architecture

### Multi-stage Build
```dockerfile
FROM python:3.11-slim-bullseye
# Key dependencies:
# - KenLM built from source (cmake, boost, eigen3)
# - FFmpeg for audio conversion
# - PyTorch 2.8.0 + torchaudio 2.8.0
```

### Container Startup
`start.sh` runs both services with colored output:
```bash
celery -A app.jobs.celery_worker worker --loglevel=info &  # Green [CELERY]
uvicorn app.main:app --host 0.0.0.0 --port 9998 &          # Blue [UVICORN]
```

### Shared Volumes
- Temp audio files: `/tmp/transcribe_{task_id}{ext}` (accessible by both API and worker containers)
- Model files: Baked into image at build time (`COPY` in Dockerfile)

## Testing & Debugging

### Model Loading Verification
```python
# Check model loads correctly with checkpoint
from app.utils.s2t import create_model
import json

with open("app/core/configs/EfficientConformerCTCSmall.json") as f:
    config = json.load(f)
model = create_model(config)
model.load("checkpoints_56_90h.ckpt")  # Should print "Model loaded at step X"
```

### Redis Health Check
```bash
# GET /health/redis
# Returns: connection status, version, memory usage, connected clients
```

### Audio Processing Limits
- Max single chunk: 956000 samples (59.75s @ 16kHz) from `train_audio_max_length`
- Overlap for long audio: 16000 samples (1 second)
- Minimum segment for diarization: 16000 samples (1 second)

## Integration Points

### Celery Task System
- Broker/Backend: Redis (same instance, DB 0)
- Task definition: `app/jobs/tasks.py::transcribe_audio_task`
- Worker: `app/jobs/celery_worker.py` (Celery app instance)
- Retry: None (manual retry via re-POST)

### External Dependencies
- **HuggingFace**: PyAnnote speaker diarization model (`pyannote/speaker-diarization-3.1`)
- **KenLM**: Vietnamese 6-gram corpus (binary format, ~100MB+)
- **SentencePiece**: Vietnamese BPE tokenizer (1024 vocab)

### Callback Mechanism
Optional POST to `callback_url` on task completion:
```json
{
  "task_id": "uuid",
  "status": "completed"|"failed",
  "results": [...],  // if completed
  "error": "...",    // if failed
  "timestamp": "ISO8601"
}
```

## Common Patterns

### Audio Tensor Shape Handling
```python
# Always ensure: (batch=1, samples)
if audio_tensor.dim() == 1:
    audio_tensor = audio_tensor.unsqueeze(0)
elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
    audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # Stereo → mono
```

### Model Inference Pattern
```python
model.eval()  # Always before inference
with torch.no_grad():
    # Use beam_search_decoding() for best quality (requires KenLM)
    # Falls back to gready_search_decoding() if beam search fails
    transcription = model.beam_search_decoding(audio_tensor, x_len, beam_size=15)
```

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

- `app/models/model_ctc.py` - CTC model with dual decoding strategies
- `app/models/encoders.py` - Conformer encoder implementation
- `app/utils/diarization.py` - Speaker diarization + complete pipeline
- `app/utils/s2t.py` - Model creation, transcription logic, chunking
- `app/api/endpoints/audio.py` - `/transcribe` endpoint, file handling
- `app/jobs/tasks.py` - Celery async transcription task
- `app/utils/redis.py` - Redis connection pool, task status management
- `app/core/configs/EfficientConformerCTCSmall.json` - Default model config

## Performance Notes

- Model inference: CPU-only by default (`device="cpu"` in tasks)
- Beam search: ~2-3x slower than greedy, significantly better accuracy
- Diarization overhead: ~10-15% of total processing time
- Typical 1-minute audio: ~30-45 seconds total processing (diarization + transcription)
