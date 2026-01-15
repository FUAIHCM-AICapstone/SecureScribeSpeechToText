"""
Audio conversion utilities for handling WebM to WAV conversion.

This module provides functions to convert WebM audio files to WAV format
using FFmpeg directly, with proper error handling and validation.
"""

import os
import subprocess

from app.utils.logging import logger


def convert_webm_to_wav(webm_path: str, wav_path: str) -> bool:
    """
    Convert a WebM file to WAV format using FFmpeg.

    This is a wrapper around convert_audio for backward compatibility.

    Args:
        webm_path: Path to the input WebM file
        wav_path: Path where the output WAV file should be saved

    Returns:
        True on successful conversion, False on failure
    """
    return convert_audio(webm_path, wav_path)


def convert_audio(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """
    Convert any audio file to WAV format with specific sample rate using FFmpeg.
    Ensures output is mono (1 channel) and 16-bit PCM.

    Args:
        input_path: Path to the input audio file
        output_path: Path where the output WAV file should be saved
        sample_rate: Target sample rate in Hz (default: 16000)

    Returns:
        True on successful conversion, False on failure
    """
    if not os.path.exists(input_path):
        logger.error(f"[AudioConverter] Input file not found: {input_path}")
        return False

    try:
        logger.info(f"[AudioConverter] Starting audio conversion: {input_path} -> {output_path}")

        command = f'ffmpeg -y -i "{input_path}" -vn -ac 1 -ar {sample_rate} -c:a pcm_s16le "{output_path}"'

        # Suppress output unless error
        result = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result != 0:
            # Try again with stderr visible if it failed
            logger.warning("[AudioConverter] Conversion failed silently, retrying with output...")
            result = subprocess.call(command, shell=True)

        if result != 0:
            logger.error(f"[AudioConverter] FFmpeg conversion failed with code {result}")
            return False

        # Verify the output file was created
        if not os.path.exists(output_path):
            logger.error("[AudioConverter] Output file was not created")
            return False

        # Check file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            logger.error("[AudioConverter] Output file is empty")
            return False

        logger.success(f"[AudioConverter] Conversion completed successfully (output size: {file_size} bytes)")
        return True

    except Exception as e:
        logger.error(f"[AudioConverter] Conversion failed: {str(e)}")

        # Clean up partial file if it exists
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception:
                pass

        return False
