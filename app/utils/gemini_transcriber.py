import os
import re
import subprocess

from google.genai import Client, types

from app.core.config import settings

SEGMENT_DURATION_SEC = 600  # 10 minutes
MAX_SEGMENT_SIZE_MB = 20

_PROMPT_TEMPLATE = """RESPONSE IN VIETNAMESE: Listen carefully to the following audio file. PROVIDE DETAIL TRANSCRIPT WITH SPEAKER DIARIZATION IN VIETNAMESE Listen carefully and provide a detailed transcript in Vietnamese. only insert new line if new speaker start speaking. Format: <transcript that you hear>

If you not hear any speak, LEAVE IT BLANK DO NOT RETURN ANYTHING, SKIP the background noise, only focus on the speaker. NO EXTRA INFORMATION NEEDED. do not use number and special character, use only text example 1 -> một, 11 -> mười một (verbose). Do not include any additional information such as [inaudible], [laughter], or other non-speech sounds.""".strip().replace("\n", " ")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def create_audio_segment(audio_path: str, start_sec: float, duration_sec: float, output_path: str) -> bool:
    """Create a single audio segment using ffmpeg."""
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, '-ss', str(start_sec), '-t', str(duration_sec),
             '-c:a', 'libopus', '-ac', '1', '-ar', '16000', '-b:a', '24k', output_path],
            capture_output=True,
            check=True,
        )
        return os.path.exists(output_path)
    except Exception as e:
        print(f"\033[91m[AUDIO] ERROR creating segment: {e}\033[0m")
        return False


def split_audio_to_segments(audio_path: str, temp_dir: str = '/tmp') -> list[str]:
    """Split audio into 10-minute segments. Returns list of segment paths."""
    try:
        duration = get_audio_duration(audio_path)
        print(f"\033[94m[AUDIO] Total duration: {duration/60:.1f}m, creating segments...\033[0m")

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
                print(f"\033[92m[AUDIO] Segment {i+1}: {seg_duration/60:.1f}m, {seg_size_mb:.1f}MB\033[0m")
                segments.append(seg_path)
            else:
                print(f"\033[91m[AUDIO] Failed to create segment {i+1}\033[0m")
                break

            i += 1

        return segments
    except Exception as e:
        print(f"\033[91m[AUDIO] ERROR splitting audio: {e}\033[0m")
        return []




def transcribe_audio_with_gemini(audio_path: str) -> str:
    """
    Transcribe audio file using Google Gemini API.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text in Vietnamese
    """
    try:
        print(f"\033[94m[GEMINI] Starting transcription for: {audio_path}\033[0m")

        if not settings.GOOGLE_API_KEY:
            print("\033[91m[GEMINI] ERROR: GOOGLE_API_KEY not configured\033[0m")
            raise ValueError("GOOGLE_API_KEY is required for Gemini transcription")

        # Initialize Gemini client
        client = Client(api_key=settings.GOOGLE_API_KEY)
        print("\033[92m[GEMINI] Gemini client initialized\033[0m")

        # Read audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        file_size_mb = len(audio_data) / (1024 * 1024)
        print(f"\033[94m[GEMINI] Audio file size: {file_size_mb:.2f} MB\033[0m")

        # For files larger than 18MB, use file upload API
        if file_size_mb > 18:
            print("\033[94m[GEMINI] File size exceeds 18MB, using file upload API\033[0m")
            uploaded_file = client.files.upload(file=audio_path, config={"mime_type": "audio/wav"})
            print(f"\033[92m[GEMINI] File uploaded successfully: {uploaded_file.name}\033[0m")

            response = client.models.generate_content(
                model=settings.EXT_MODEL_ID,
                contents=[_PROMPT_TEMPLATE, uploaded_file],
            )
        else:
            print("\033[94m[GEMINI] File size within limits, using direct upload\033[0m")
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")

            response = client.models.generate_content(
                model=settings.EXT_MODEL_ID,
                contents=[_PROMPT_TEMPLATE, audio_part],
            )

        # Extract and clean response text
        transcript_text = response.text.lower().replace("\n", " ").strip()
        print(f"\033[92m[GEMINI] Raw response received: {transcript_text[:100]}...\033[0m")

        # Remove bracketed text (e.g., [inaudible], [laughter])
        transcript_text = re.sub(r"[\(\[\{].*?[\)\]\}]", "", transcript_text)

        # Remove special characters
        transcript_text = re.sub(r"[!@#$%^&*<>?,./;:'\"\\|`~]", "", transcript_text)

        # Final cleanup
        transcript_text = transcript_text.strip()

        if not transcript_text:
            print("\033[93m[GEMINI] WARNING: Empty transcription received\033[0m")
            return None

        print(f"\033[92m[GEMINI] Transcription completed: {transcript_text[:100]}...\033[0m")
        return transcript_text

    except Exception as e:
        print(f"\033[91m[GEMINI] ERROR during transcription: {str(e)}\033[0m")
        raise RuntimeError(f"Failed to transcribe audio with Gemini: {str(e)}")


def transcribe_audio_with_splitting(audio_path: str) -> str:
    """
    Transcribe long audio by splitting into 10-min segments.
    Each segment is transcribed in parallel via Celery, results are combined.
    """
    from app.jobs.tasks import transcribe_audio_segment_task

    try:
        duration = get_audio_duration(audio_path)
        print(f"\033[94m[PIPELINE] Audio duration: {duration/60:.1f}m\033[0m")

        # If short, transcribe directly
        if duration < SEGMENT_DURATION_SEC:
            print("\033[94m[PIPELINE] Audio is short, transcribing directly\033[0m")
            return transcribe_audio_with_gemini(audio_path)

        # Split into segments
        segments = split_audio_to_segments(audio_path)
        if not segments:
            raise RuntimeError("Failed to create audio segments")

        print(f"\033[94m[PIPELINE] Queueing {len(segments)} segments for parallel transcription\033[0m")

        # Queue all segments in parallel
        task_ids = []
        for i, seg_path in enumerate(segments):
            task = transcribe_audio_segment_task.delay(seg_path, index=i)
            task_ids.append((i, task.id))

        # Collect results in order
        from celery.result import AsyncResult
        transcripts = [""] * len(segments)

        for idx, task_id in task_ids:
            result = AsyncResult(task_id)
            transcript = result.get(timeout=300)  # 5 min timeout per segment
            transcripts[idx] = transcript or ""
            print(f"\033[92m[PIPELINE] Segment {idx+1} completed\033[0m")

        # Cleanup
        for seg_path in segments:
            try:
                os.remove(seg_path)
            except:
                pass

        combined = " ".join(t for t in transcripts if t)
        print(f"\033[92m[PIPELINE] All segments done: {len(combined)} chars total\033[0m")
        return combined

    except Exception as e:
        print(f"\033[91m[PIPELINE] ERROR: {str(e)}\033[0m")
        raise
