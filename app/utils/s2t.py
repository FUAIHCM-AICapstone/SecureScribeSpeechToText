import os
import time
import torch

from app.models.model_ctc import ModelCTC


def create_model(config_data):
    model = ModelCTC(encoder_params=config_data["encoder_params"], tokenizer_params=config_data["tokenizer_params"], training_params=config_data["training_params"], decoding_params=config_data["decoding_params"], name=config_data["model_name"])
    return model


def transcribe_audio_segment(model, audio_tensor, device="cpu", sample_rate=16000):
    """
    Transcribe an audio segment using the loaded model.

    Args:
        model: Loaded speech-to-text model
        audio_tensor: Audio tensor (should be 1D or 2D with shape [channels, samples])
        device: Device to run inference on
        sample_rate: Sample rate of audio (default: 16000 Hz)

    Returns:
        Transcribed text
    """
    print(f"\033[94m[S2T] Starting transcription of audio segment with beam search decoding (shape: {audio_tensor.shape})\033[0m")

    # Ensure audio is in the right format
    if audio_tensor.dim() == 1:
        print("\033[94m[S2T] Adding channel dimension to 1D audio\033[0m")
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
    elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
        print(f"\033[93m[S2T] Converting stereo to mono (channels: {audio_tensor.shape[0]})\033[0m")
        # Convert stereo to mono
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    # Ensure proper shape
    audio_tensor = audio_tensor.squeeze()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Check if audio is too long and split into chunks
    max_samples = 956000  # Based on train_audio_max_length from config
    audio_length = audio_tensor.shape[1]

    if audio_length > max_samples:
        print(f"\033[93m[S2T] Audio too long ({audio_length} samples), splitting into chunks\033[0m")
        chunk_size = max_samples
        overlap = 16000  # 1 second overlap at 16kHz
        chunks = []

        start = 0
        while start < audio_length:
            end = min(start + chunk_size, audio_length)
            chunk = audio_tensor[:, start:end]
            chunks.append(chunk)
            start = end - overlap if end < audio_length else end

        print(f"\033[94m[S2T] Split into {len(chunks)} chunks\033[0m")

        # Transcribe each chunk
        transcriptions = []
        for i, chunk in enumerate(chunks):
            print(f"\033[94m[S2T] Transcribing chunk {i + 1}/{len(chunks)} with beam search (shape: {chunk.shape})\033[0m")
            transcription = _transcribe_single_chunk(model, chunk, device)
            if transcription:
                transcriptions.append(transcription)

        # Combine transcriptions
        result = " ".join(transcriptions).strip()
        print(f"\033[92m[S2T] Combined transcription: '{result}'\033[0m")
        return result
    else:
        return _transcribe_single_chunk(model, audio_tensor, device)


def _transcribe_single_chunk(model, audio_tensor, device="cpu"):
    """
    Transcribe a single audio chunk using beam search decoding with fallback to greedy decoding.
    """
    # Create length tensor
    x_len = torch.tensor([audio_tensor.shape[1]], device=device)

    # Run inference
    with torch.no_grad():
        # Start timer
        start_time = time.time()

        # try:
        #     # Check if beam_search_decoding method is available
        #     if not hasattr(model, "beam_search_decoding"):
        #         raise AttributeError("Model does not have beam_search_decoding method")

        #     # Check if ngram_path file exists if configured
        #     if hasattr(model, "ngram_path") and model.ngram_path is not None:
        #         if not os.path.exists(model.ngram_path):
        #             raise FileNotFoundError(f"N-gram model file not found: {model.ngram_path}")

        #     # Log beam search start
        #     beam_size = getattr(model, "beam_size", 1)
        #     print(f"\033[94m[S2T] Using beam search decoding (beam_size={beam_size})\033[0m")

        #     # Call beam search decoding
        #     transcription = model.beam_search_decoding(audio_tensor.to(device), x_len)

        #     # Calculate elapsed time
        #     elapsed = time.time() - start_time

        #     # Handle case where transcription is a list (batch processing)
        #     if isinstance(transcription, list):
        #         if not transcription:
        #             # Handle empty list case
        #             print("\033[91m[S2T] ERROR: Beam search returned empty list\033[0m")
        #             return ""
        #         result = transcription[0]
        #     else:
        #         result = transcription

        #     # Handle None result
        #     if result is None:
        #         print("\033[91m[S2T] ERROR: Beam search returned None\033[0m")
        #         return ""

        #     result = result.lower().strip()
        #     print(f"\033[92m[S2T] Beam search completed in {elapsed:.3f}s: '{result}'\033[0m")
        #     return result

        # except ImportError as e:
        #     # ctcdecode library not installed
        #     elapsed = time.time() - start_time
        #     print(f"\033[93m[S2T] WARNING: ctcdecode library not available ({e}), falling back to greedy decoding\033[0m")

        # except FileNotFoundError as e:
        #     # N-gram model file missing
        #     elapsed = time.time() - start_time
        #     print(f"\033[93m[S2T] WARNING: {e}, falling back to greedy decoding\033[0m")

        # except AttributeError as e:
        #     # beam_search_decoding method not available
        #     elapsed = time.time() - start_time
        #     print(f"\033[93m[S2T] WARNING: {e}, falling back to greedy decoding\033[0m")

        # except Exception as e:
        #     # Any other error during beam search
        #     elapsed = time.time() - start_time
        #     print(f"\033[91m[S2T] ERROR during beam search: {e}\033[0m")
        #     print(f"\033[93m[S2T] Falling back to greedy search decoding\033[0m")

        # Fallback to greedy decoding
        try:
            print("\033[94m[S2T] Running greedy search decoding...\033[0m")
            start_time = time.time()

            # Try greedy_search_decoding first (correct name)
            if hasattr(model, "greedy_search_decoding"):
                transcription = model.greedy_search_decoding(audio_tensor.to(device), x_len)
            # Fallback to gready_search_decoding (typo version)
            elif hasattr(model, "gready_search_decoding"):
                transcription = model.gready_search_decoding(audio_tensor.to(device), x_len)
            else:
                print("\033[91m[S2T] ERROR: No greedy decoding method available\033[0m")
                return ""

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Handle case where transcription is a list (batch processing)
            if isinstance(transcription, list):
                if not transcription:
                    # Handle empty list case
                    print("\033[91m[S2T] ERROR: Greedy decoding returned empty list\033[0m")
                    return ""
                result = transcription[0]
            else:
                result = transcription

            # Handle None result from tokenizer error
            if result is None:
                print("\033[91m[S2T] ERROR: Greedy decoding returned None (tokenizer not available)\033[0m")
                return ""

            result = result.lower().strip()
            print(f"\033[92m[S2T] Greedy decoding completed in {elapsed:.3f}s: '{result}'\033[0m")
            return result

        except Exception as e:
            print(f"\033[91m[S2T] ERROR during greedy decoding: {e}\033[0m")
            return ""
