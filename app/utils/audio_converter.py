"""
Audio conversion utilities for handling WebM to WAV conversion.

This module provides functions to convert WebM audio files to WAV format
using FFmpeg directly, with proper error handling and validation.
"""

import os
import subprocess


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
        print(f"\033[91m[AudioConverter] ERROR: Input file not found: {input_path}\033[0m")
        return False

    try:
        print(f"\033[94m[AudioConverter] Starting audio conversion: {input_path} -> {output_path}\033[0m")

        command = f'ffmpeg -y -i "{input_path}" -vn -ac 1 -ar {sample_rate} -c:a pcm_s16le "{output_path}"'

        # Suppress output unless error
        result = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result != 0:
            # Try again with stderr visible if it failed
            print("\033[93m[AudioConverter] Conversion failed silently, retrying with output...\033[0m")
            result = subprocess.call(command, shell=True)

        if result != 0:
            print(f"\033[91m[AudioConverter] ERROR: FFmpeg conversion failed with code {result}\033[0m")
            return False

        # Verify the output file was created
        if not os.path.exists(output_path):
            print("\033[91m[AudioConverter] ERROR: Output file was not created\033[0m")
            return False

        # Check file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            print("\033[91m[AudioConverter] ERROR: Output file is empty\033[0m")
            return False

        print(f"\033[92m[AudioConverter] Conversion completed successfully (output size: {file_size} bytes)\033[0m")
        return True

    except Exception as e:
        print(f"\033[91m[AudioConverter] ERROR: Conversion failed: {str(e)}\033[0m")

        # Clean up partial file if it exists
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception:
                pass

        return False
