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

    Args:
        webm_path: Path to the input WebM file
        wav_path: Path where the output WAV file should be saved

    Returns:
        True on successful conversion, False on failure
    """
    if not os.path.exists(webm_path):
        print(f"\033[91m[AudioConverter] ERROR: Input file not found: {webm_path}\033[0m")
        return False

    try:
        print(f"\033[94m[AudioConverter] Starting WebM to WAV conversion\033[0m")
        print(f"\033[94m[AudioConverter] Input: {webm_path}\033[0m")
        print(f"\033[94m[AudioConverter] Output: {wav_path}\033[0m")
        
        # Use ffmpeg to convert WebM to WAV
        # -vn: no video, -c:a pcm_s16le: PCM 16-bit audio codec, -y: overwrite output
        command = f'ffmpeg -y -i "{webm_path}" -vn -c:a pcm_s16le "{wav_path}"'
        result = subprocess.call(command, shell=True)
        
        if result != 0:
            print(f"\033[91m[AudioConverter] ERROR: FFmpeg conversion failed with code {result}\033[0m")
            return False
        
        # Verify the output file was created
        if not os.path.exists(wav_path):
            print(f"\033[91m[AudioConverter] ERROR: WAV file was not created\033[0m")
            return False
        
        # Check file size
        file_size = os.path.getsize(wav_path)
        if file_size == 0:
            print(f"\033[91m[AudioConverter] ERROR: WAV file is empty\033[0m")
            return False
        
        print(f"\033[92m[AudioConverter] Conversion completed successfully (output size: {file_size} bytes)\033[0m")
        return True
        
    except Exception as e:
        print(f"\033[91m[AudioConverter] ERROR: Conversion failed: {str(e)}\033[0m")
        
        # Clean up partial WAV file if it exists
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                print(f"\033[94m[AudioConverter] Cleaned up partial WAV file\033[0m")
            except Exception as cleanup_error:
                print(f"\033[91m[AudioConverter] ERROR: Failed to clean up partial WAV file: {str(cleanup_error)}\033[0m")
        
        return False
