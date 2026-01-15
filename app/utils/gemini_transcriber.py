import os
import re
import subprocess

from google.genai import Client, types

from app.core.config import settings
from app.utils.logging import logger
from app.utils.models import FILE_SIZE_THRESHOLD_MB, MODEL_ID, PROMPT_TEMPLATE, TokenUsage, extract_token_usage

SEGMENT_DURATION_SEC = 600  # 10 minutes
MAX_SEGMENT_SIZE_MB = 20


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def create_audio_segment(audio_path: str, start_sec: float, duration_sec: float, output_path: str) -> bool:
    """Create a single audio segment using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ss", str(start_sec), "-t", str(duration_sec), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path],
            capture_output=True,
            check=True,
        )
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"[AUDIO] ERROR creating segment: {e}")
        return False


def split_audio_to_segments(audio_path: str, temp_dir: str = "/tmp") -> list[str]:
    """Split audio into 10-minute segments. Returns list of segment paths."""
    try:
        duration = get_audio_duration(audio_path)
        logger.info(f"[AUDIO] Total duration: {duration / 60:.1f}m, creating segments...")

        segments = []
        job_id = os.urandom(4).hex()
        i = 0

        while True:
            start_sec = i * SEGMENT_DURATION_SEC
            seg_duration = min(SEGMENT_DURATION_SEC, duration - start_sec)

            if seg_duration <= 0:
                break

            seg_path = f"{temp_dir}/seg_{job_id}_{i}.wav"
            if create_audio_segment(audio_path, start_sec, seg_duration, seg_path):
                seg_size_mb = os.path.getsize(seg_path) / (1024 * 1024)
                logger.success(f"[AUDIO] Segment {i + 1}: {seg_duration / 60:.1f}m, {seg_size_mb:.1f}MB")
                segments.append(seg_path)
            else:
                logger.error(f"[AUDIO] Failed to create segment {i + 1}")
                break

            i += 1

        return segments
    except Exception as e:
        logger.error(f"[AUDIO] ERROR splitting audio: {e}")
        return []


def transcribe_audio_with_gemini(audio_path: str) -> tuple[str | None, TokenUsage]:
    """
    Transcribe audio file using Google Gemini API.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (transcribed_text, token_usage)
        - transcribed_text: Transcribed text in Vietnamese, or None if empty
        - token_usage: TokenUsage object with input/output token counts
    """
    try:
        logger.info(f"[GEMINI] Starting transcription for: {audio_path}")

        if not settings.GOOGLE_API_KEY:
            logger.error("[GEMINI] ERROR: GOOGLE_API_KEY not configured")
            raise ValueError("GOOGLE_API_KEY is required for Gemini transcription")

        # Initialize Gemini client
        client = Client(api_key=settings.GOOGLE_API_KEY)
        logger.success("[GEMINI] Gemini client initialized")

        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        file_size_mb = len(audio_data) / (1024 * 1024)
        logger.debug(f"[GEMINI] Audio file size: {file_size_mb:.2f} MB")

        # For files larger than threshold, use file upload API
        if file_size_mb > FILE_SIZE_THRESHOLD_MB:
            logger.info(f"[GEMINI] File size exceeds {FILE_SIZE_THRESHOLD_MB}MB, using file upload API")
            uploaded_file = client.files.upload(file=audio_path, config={"mime_type": "audio/wav"})
            logger.success(f"[GEMINI] File uploaded successfully: {uploaded_file.name}")

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[PROMPT_TEMPLATE, uploaded_file],
            )
        else:
            logger.debug("[GEMINI] File size within limits, using direct upload")
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[PROMPT_TEMPLATE, audio_part],
            )

        # Log raw response object for debugging
        logger.debug(f"[GEMINI] Raw response object type: {type(response)}")
        logger.debug(f"[GEMINI] Raw response attributes: {dir(response)}")
        if hasattr(response, "usage_metadata"):
            logger.debug(f"[GEMINI] Response usage_metadata: {response.usage_metadata}")
            logger.debug(f"[GEMINI] usage_metadata type: {type(response.usage_metadata)}")
            if response.usage_metadata:
                logger.debug(f"[GEMINI] usage_metadata attributes: {dir(response.usage_metadata)}")
                logger.debug(f"[GEMINI] usage_metadata as dict: {response.usage_metadata.__dict__ if hasattr(response.usage_metadata, '__dict__') else 'N/A'}")
        else:
            logger.warning("[GEMINI] Response does not have usage_metadata attribute")

        # Extract token usage
        token_usage = extract_token_usage(response)
        logger.info(f"[GEMINI] Token usage extracted: {token_usage.to_dict()}")

        # Extract and clean response text
        transcript_text = response.text.lower().replace("\n", " ").strip()
        logger.debug(f"[GEMINI] Raw response text: {transcript_text[:200]}...")

        # Remove bracketed text (e.g., [inaudible], [laughter])
        transcript_text = re.sub(r"[\(\[\{].*?[\)\]\}]", "", transcript_text)

        # Remove special characters
        transcript_text = re.sub(r"[!@#$%^&*<>?,./;:'\"\\|`~]", "", transcript_text)

        # Final cleanup
        transcript_text = transcript_text.strip()

        if not transcript_text:
            logger.warning("[GEMINI] WARNING: Empty transcription received")
            return None, token_usage

        logger.success(f"[GEMINI] Transcription completed: {transcript_text[:100]}...")
        return transcript_text, token_usage

    except Exception as e:
        logger.exception(f"[GEMINI] ERROR during transcription: {str(e)}")
        raise RuntimeError(f"Failed to transcribe audio with Gemini: {str(e)}")


def transcribe_audio_with_splitting(audio_path: str) -> tuple[str | None, TokenUsage]:
    """
    Transcribe long audio by splitting into 10-min segments.
    Each segment is transcribed sequentially, results are combined.

    Returns:
        Tuple of (combined_transcript, total_token_usage)
    """
    try:
        duration = get_audio_duration(audio_path)
        logger.info(f"[PIPELINE] Audio duration: {duration / 60:.1f}m")

        # If short, transcribe directly
        if duration < SEGMENT_DURATION_SEC:
            logger.info("[PIPELINE] Audio is short, transcribing directly")
            return transcribe_audio_with_gemini(audio_path)

        # Split into segments
        segments = split_audio_to_segments(audio_path)
        if not segments:
            raise RuntimeError("Failed to create audio segments")

        logger.info(f"[PIPELINE] Processing {len(segments)} segments sequentially")

        # Process segments sequentially (avoid Celery deadlock)
        transcripts = []
        total_token_usage = TokenUsage()

        for idx, seg_path in enumerate(segments):
            try:
                logger.info(f"[PIPELINE] Transcribing segment {idx + 1}/{len(segments)}")
                transcript, token_usage = transcribe_audio_with_gemini(seg_path)

                if transcript:
                    transcripts.append(transcript)

                total_token_usage.input_tokens += token_usage.input_tokens
                total_token_usage.output_tokens += token_usage.output_tokens

                logger.success(f"[PIPELINE] Segment {idx + 1} completed - Tokens: Input={token_usage.input_tokens}, Output={token_usage.output_tokens}")
            except Exception as seg_error:
                logger.warning(f"[PIPELINE] WARNING: Segment {idx + 1} failed: {str(seg_error)}, continuing...")
                continue

        # Cleanup all segment files
        for seg_path in segments:
            try:
                if os.path.exists(seg_path):
                    os.remove(seg_path)
            except:
                pass

        combined = " ".join(transcripts) if transcripts else None
        logger.success(f"[PIPELINE] All segments done: {len(combined) if combined else 0} chars total, Total tokens - Input: {total_token_usage.input_tokens}, Output: {total_token_usage.output_tokens}")

        return combined, total_token_usage

    except Exception as e:
        logger.exception(f"[PIPELINE] ERROR: {str(e)}")
        raise
