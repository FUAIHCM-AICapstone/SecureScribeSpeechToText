import re

from google.genai import Client, types

from app.core.config import settings

_PROMPT_TEMPLATE = """RESPONSE IN VIETNAMESE: Listen carefully to the following audio file. PROVIDE DETAIL TRANSCRIPT WITH SPEAKER DIARIZATION IN VIETNAMESE Listen carefully and provide a detailed transcript in Vietnamese. only insert new line if new speaker start speaking. Format: <transcript that you hear>

If you not hear any speak, LEAVE IT BLANK DO NOT RETURN ANYTHING, SKIP the background noise, only focus on the speaker. NO EXTRA INFORMATION NEEDED. do not use number and special character, use only text example 1 -> một, 11 -> mười một (verbose). Do not include any additional information such as [inaudible], [laughter], or other non-speech sounds.""".strip().replace("\n", " ")


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
