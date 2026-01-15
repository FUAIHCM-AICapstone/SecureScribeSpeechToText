"""
Output parser for transcription results.

Parses Gemini API responses to extract speaker diarization and structure into JSON format.
"""

import re
from typing import Any, Dict, List

from app.utils.logging import logger


def parse_speaker_diarization(transcript: str) -> List[Dict[str, Any]]:
    """
    Parse speaker diarization from transcription response.

    Expected format from Gemini (flexible to handle variations):
    SPEAKER_1: <text>
    SPEAKER_2: <text>

    Also handles:
    - speaker_1: <text> (lowercase)
    - speaker 1: <text> (space instead of underscore)
    - speaker_1 <text> (without colon)

    Args:
        transcript: Raw transcription text from Gemini API

    Returns:
        List of dictionaries with speaker, start_time, end_time, transcription
    """
    if not transcript:
        logger.warning("[PARSER] Empty transcript provided")
        return []

    logger.info(f"[PARSER] Parsing speaker diarization from transcript ({len(transcript)} chars)")
    logger.debug(f"[PARSER] Raw transcript:\n{transcript[:500]}...")

    results = []

    # Split by lines first to find speaker boundaries
    lines = transcript.split("\n")

    # If only one line, try to split by speaker patterns
    if len(lines) == 1 and re.search(r"(?:speaker|SPEAKER|Speaker)\s*[_\s]?\s*\d+\s*:", transcript, re.IGNORECASE):
        logger.debug("[PARSER] Single line with multiple speakers, splitting by speaker patterns")
        # Split by speaker patterns while preserving the speaker labels
        parts = re.split(r"((?:speaker|SPEAKER|Speaker)\s*[_\s]?\s*\d+\s*:)", transcript, flags=re.IGNORECASE)

        # Reconstruct lines with speaker labels
        reconstructed_lines = []
        for i in range(1, len(parts), 2):  # Skip first empty part, take pairs
            if i + 1 < len(parts):
                speaker_label = parts[i].strip()
                speaker_text = parts[i + 1].strip()
                reconstructed_lines.append(f"{speaker_label} {speaker_text}")

        if reconstructed_lines:
            lines = reconstructed_lines
            logger.debug(f"[PARSER] Reconstructed {len(lines)} speaker lines from single line")

    current_time = 0.0
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Pattern 1: SPEAKER_1: text (case-insensitive, with or without underscore)
        # Matches: SPEAKER_1:, speaker_1:, Speaker_1:, SPEAKER 1:, speaker 1:, etc.
        match = re.match(r"(?:speaker|SPEAKER|Speaker)\s*[_\s]?\s*(\d+)\s*:\s*(.+)", line, re.IGNORECASE)

        if match:
            speaker_num = match.group(1)
            speaker_text = match.group(2).strip()

            # Collect text from this speaker across multiple lines until next speaker
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()

                # Check if next line starts with SPEAKER pattern
                if re.match(r"(?:speaker|SPEAKER|Speaker)\s*[_\s]?\s*(\d+)\s*:", next_line, re.IGNORECASE):
                    break

                if next_line:
                    speaker_text += " " + next_line
                i += 1

            # Clean up text
            speaker_text = " ".join(speaker_text.split())  # Normalize whitespace

            if speaker_text and speaker_num != "0":  # Skip "No speech" indicator
                # Estimate duration based on character count
                char_count = len(speaker_text)
                estimated_duration = max(0.3, char_count / 150.0)  # Min 0.3 seconds

                speaker_data = {
                    "speaker": f"SPEAKER_{speaker_num}",
                    "start_time": round(current_time, 2),
                    "end_time": round(current_time + estimated_duration, 2),
                    "transcription": speaker_text,
                }

                results.append(speaker_data)
                logger.debug(f"[PARSER] Parsed SPEAKER_{speaker_num}: {len(speaker_text)} chars, ~{estimated_duration:.1f}s")

                current_time += estimated_duration

        i += 1

    if not results:
        logger.warning("[PARSER] No speakers found with pattern matching, trying fallback")
        # Fallback: treat entire transcript as single speaker
        char_count = len(transcript)
        estimated_duration = max(0.3, char_count / 150.0)
        results = [
            {
                "speaker": "SPEAKER_1",
                "start_time": 0.0,
                "end_time": round(estimated_duration, 5),
                "transcription": transcript,
            }
        ]

    logger.success(f"[PARSER] Parsed {len(results)} speaker segments")
    return results


def format_transcription_results(transcript: str, token_usage: Dict[str, int]) -> Dict[str, Any]:
    """
    Format final transcription results with parsed speakers and token usage.

    Args:
        transcript: Raw transcript from Gemini API
        token_usage: Token usage dictionary from extract_token_usage

    Returns:
        Formatted results dictionary with parsed speakers and metrics
    """
    logger.info("[PARSER] Formatting transcription results")

    speakers = parse_speaker_diarization(transcript)

    results = {
        "raw_transcript": transcript,
        "speakers": speakers,
        "speaker_lines": len(speakers),
        "token_usage": token_usage,
    }

    logger.success(f"[PARSER] Results formatted: {len(speakers)} speakers, {token_usage.get('total_tokens', 0)} tokens")
    return results
