import json
import os
import warnings

import torchaudio
from pyannote.audio import Pipeline

# Suppress specific deprecation warnings from pyannote.audio/torchaudio
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends has been deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*list_audio_backends.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.utils.info has been deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="torchaudio._backend.common.AudioMetaData has been deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed", category=UserWarning)
warnings.filterwarnings("ignore", message="std\\(\\)\\: degrees of freedom is <= 0", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchaudio\\.info.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AudioMetaData.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)


def perform_speaker_diarization(audio_path, hf_token=None, merge_gap_threshold=5.0):
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file
        hf_token: HuggingFace token for accessing the model
        merge_gap_threshold: Maximum gap (in seconds) between segments of same speaker to merge

    Returns:
        List of tuples: [(start_time, end_time, speaker_label), ...]
    """
    print("\033[94m[DIARIZATION] Starting speaker diarization for: {audio_path}\033[0m")

    # Set environment variable to prevent CUDA usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not hf_token:
        print("\033[91m[DIARIZATION] ERROR: HuggingFace token is required\033[0m")
        raise ValueError("HuggingFace token is required for speaker diarization")

    # Load pretrained pipeline
    print("\033[94m[DIARIZATION] Loading diarization pipeline...\033[0m")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        print("\033[92m[DIARIZATION] Pipeline loaded successfully\033[0m")
    except Exception as e:
        print(f"\033[91m[DIARIZATION] ERROR loading pipeline: {e}\033[0m")
        raise RuntimeError(f"Failed to load diarization pipeline: {e}")

    # Run diarization
    print("\033[94m[DIARIZATION] Running diarization...\033[0m")
    try:
        diarization = pipeline(audio_path)
        print("\033[92m[DIARIZATION] Diarization completed\033[0m")
    except Exception as e:
        print(f"\033[91m[DIARIZATION] ERROR during diarization: {e}\033[0m")
        raise RuntimeError(f"Failed to run diarization: {e}")

    # Collect all segments
    print("\033[94m[DIARIZATION] Collecting segments...\033[0m")
    segments = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        segments.append((segment.start, segment.end, label))

    print(f"\033[92m[DIARIZATION] Collected {len(segments)} raw segments\033[0m")

    # Sort by start time
    segments.sort(key=lambda x: x[0])

    # Merge close segments from same speaker
    print(f"\033[94m[DIARIZATION] Merging close segments (threshold: {merge_gap_threshold}s)...\033[0m")
    if segments:
        merged_segments = []
        current_start, current_end, current_speaker = segments[0]

        for start, end, speaker in segments[1:]:
            # If same speaker and gap is small, merge
            if speaker == current_speaker and start - current_end < merge_gap_threshold:
                current_end = max(current_end, end)
            else:
                # Save current segment and start new one
                merged_segments.append((current_start, current_end, current_speaker))
                current_start, current_end, current_speaker = start, end, speaker

        # Add last segment
        merged_segments.append((current_start, current_end, current_speaker))
        print(f"\033[92m[DIARIZATION] Merged into {len(merged_segments)} segments\033[0m")
        return merged_segments

    print("\033[92m[DIARIZATION] No segments to merge\033[0m")
    return segments


def extract_audio_segments(audio_path, segments, output_dir=None):
    """
    Extract audio segments based on diarization results.

    Args:
        audio_path: Path to original audio file
        segments: List of (start_time, end_time, speaker_label) tuples
        output_dir: Directory to save extracted segments (optional)

    Returns:
        List of tuples: [(speaker_label, start_time, end_time, audio_tensor), ...]
    """
    print(f"\033[94m[DIARIZATION] Extracting audio segments from {len(segments)} segments...\033[0m")

    # Load audio
    print(f"\033[94m[DIARIZATION] Loading audio file: {audio_path}\033[0m")
    audio, sample_rate = torchaudio.load(audio_path)

    # Handle stereo to mono conversion
    if audio.dim() == 2:
        print("\033[93m[DIARIZATION] Converting stereo to mono\033[0m")
        audio = audio.mean(dim=0)

    # Ensure proper shape
    audio = audio.squeeze()

    extracted_segments = []
    skipped_segments = 0

    for start_time, end_time, speaker in segments:
        # Convert time to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract segment
        segment_audio = audio[start_sample:end_sample]

        # Skip if too short
        if len(segment_audio) < sample_rate:  # Less than 1 second
            skipped_segments += 1
            continue

        extracted_segments.append((speaker, start_time, end_time, segment_audio))

        # Optionally save to file
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            segment_path = os.path.join(output_dir, f"{speaker}_{start_time:.2f}_{end_time:.2f}.wav")
            torchaudio.save(segment_path, segment_audio.unsqueeze(0), sample_rate)

    print(f"\033[92m[DIARIZATION] Extracted {len(extracted_segments)} segments, skipped {skipped_segments} short segments\033[0m")
    return extracted_segments


def diarize_and_transcribe_audio(audio_path, config_path, checkpoint_path, hf_token, device="cpu", merge_gap_threshold=5.0):
    """
    Complete pipeline: diarize audio and transcribe each speaker segment.

    Args:
        audio_path: Path to audio file
        config_path: Path to model configuration JSON
        checkpoint_path: Path to model checkpoint
        hf_token: HuggingFace token for diarization
        device: Device for model inference
        merge_gap_threshold: Gap threshold for merging segments

    Returns:
        List of tuples: [(speaker, start_time, end_time, transcription), ...]
    """
    print("\033[94m[PIPELINE] Starting complete diarization and transcription pipeline\033[0m")

    from .s2t import create_model, transcribe_audio_segment

    # Load configuration
    print(f"\033[94m[PIPELINE] Loading model configuration from: {config_path}\033[0m")
    with open(config_path) as f:
        config = json.load(f)
    print(f"\033[92m[PIPELINE] Configuration loaded: {config.get('model_name', 'Unknown Model')}\033[0m")

    # Create model
    print("\033[94m[PIPELINE] Creating model...\033[0m")
    model = create_model(config).to(device)
    model.eval()
    print("\033[92m[PIPELINE] Model created successfully\033[0m")

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print("\033[94m[PIPELINE] Loading checkpoint using model.load()...\033[0m")
        model.load(checkpoint_path)
        print("\033[92m[PIPELINE] Checkpoint loaded successfully\033[0m")

    # Perform diarization
    segments = perform_speaker_diarization(audio_path, hf_token, merge_gap_threshold)

    # Extract and transcribe segments
    print(f"\033[94m[PIPELINE] Starting transcription of {len(segments)} segments...\033[0m")
    results = []
    for i, (speaker, start_time, end_time, audio_segment) in enumerate(extract_audio_segments(audio_path, segments)):
        print(f"\033[94m[PIPELINE] Transcribing segment {i + 1}/{len(segments)} - Speaker: {speaker}, Duration: {end_time - start_time:.2f}s\033[0m")

        # Transcribe segment
        transcription = transcribe_audio_segment(model, audio_segment, device)

        if transcription:  # Only include non-empty transcriptions
            print(f"\033[92m[PIPELINE] Segment {i + 1} transcribed: '{transcription}'\033[0m")
            results.append({"speaker": speaker, "start_time": start_time, "end_time": end_time, "transcription": transcription})
        else:
            print(f"\033[93m[PIPELINE] Segment {i + 1} produced no transcription\033[0m")

    print(f"\033[92m[PIPELINE] Pipeline completed: {len(results)} segments transcribed\033[0m")
    return results
