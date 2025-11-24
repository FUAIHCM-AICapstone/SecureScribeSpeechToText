import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio

from app.core.config import settings
from app.models.model_ctc import ModelCTC

PROMPT_TEMPLATE = """RESPONSE IN VIETNAMESE: Listen carefully to the following audio file. PROVIDE DETAIL TRANSCRIPT WITH SPEAKER DIARIZATION IN VIETNAMESE
    Listen carefully and provide a detailed transcript in Vietnamese.
    only insert new line if new speaker start speaking.
      Format:
      <transcript that you hear>\n\n
      If you not hear any speak, LEAVE IT BLANK DO NOT RETURN ANYTHING, SKIP the background noise, only focus on the speaker. NO EXTRA INFORMATION NEEDED.
      do not use number and special character, use only text example 1 -> một, 11 -> mười một (verbose)
      Do not include any additional information such as [inaudible], [laughter], or other non-speech sounds.
    """


def create_model(config_data):
    model = ModelCTC(encoder_params=config_data["encoder_params"], tokenizer_params=config_data["tokenizer_params"], training_params=config_data["training_params"], decoding_params=config_data["decoding_params"], name=config_data["model_name"])
    return model


_hf_decoder = None


def get_hf_decoder():
    global _hf_decoder
    if _hf_decoder is None and settings.DECODING_STRATEGY == "hf_lm":
        from app.utils.hf_lm_decoder import HFLMBeamDecoder

        print("\033[94m[S2T] Initializing HF LM decoder (one-time setup)\033[0m")
        _hf_decoder = HFLMBeamDecoder(model_id=settings.HF_LM_MODEL_ID, alpha=settings.HF_LM_ALPHA, beta=settings.HF_LM_BETA, beam_width=settings.HF_LM_BEAM_WIDTH, device="cpu", hf_token=settings.HF_TOKEN)
    return _hf_decoder


def transcribe_audio_segment(model, audio_tensor, device="cpu", strategy=None):
    if strategy is None:
        strategy = settings.DECODING_STRATEGY
    print(f"\033[94m[S2T] Starting transcription with {strategy} decoding (shape: {audio_tensor.shape})\033[0m")

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

    # SKIP CHUNKING FOR GEMINI
    if strategy == "gemini":
        print("\033[94m[S2T] Gemini strategy selected, skipping chunking (Gemini handles long context)\033[0m")
        return _transcribe_single_chunk(model, audio_tensor, device, strategy)

    # Check if audio is too long and split into chunks
    max_samples = 1056000  # Based on train_audio_max_length from config
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

        # Transcribe each chunk in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_transcribe_single_chunk, model, chunk, device, strategy): (i, chunk) for i, chunk in enumerate(chunks)}
            transcriptions = []
            for future in futures:
                i, chunk = futures[future]
                print(f"\033[94m[S2T] Transcribing chunk {i + 1}/{len(chunks)} (shape: {chunk.shape})\033[0m")
                transcription = future.result()
                if transcription:
                    transcriptions.append((i, transcription))
            transcriptions.sort(key=lambda x: x[0])
            transcriptions = [t for _, t in transcriptions]

        # Combine transcriptions
        result = " ".join(transcriptions).strip()
        print(f"\033[92m[S2T] Combined transcription: '{result}'\033[0m")
        return result
    else:
        return _transcribe_single_chunk(model, audio_tensor, device, strategy)


def _transcribe_single_chunk(model, audio_tensor, device="cpu", strategy="greedy"):
    x_len = torch.tensor([audio_tensor.shape[1]], device=device)
    with torch.no_grad():
        start_time = time.time()
        try:
            if strategy == "beam":
                return _decode_beam(model, audio_tensor, x_len, device, start_time)
            elif strategy == "hf_lm":
                return _decode_hf_lm(model, audio_tensor, x_len, device, start_time)
            elif strategy == "gemini":
                return _decode_gemini(audio_tensor, start_time)
            else:
                return _decode_greedy(model, audio_tensor, x_len, device, start_time)
        except Exception as e:
            print(f"\033[91m[S2T] ERROR during {strategy} decoding: {e}\033[0m")
            print("\033[93m[S2T] Falling back to greedy decoding\033[0m")
            return _decode_greedy(model, audio_tensor, x_len, device, time.time())


def _decode_greedy(model, audio_tensor, x_len, device, start_time):
    print("\033[94m[S2T] Running greedy decoding\033[0m")
    if hasattr(model, "greedy_search_decoding"):
        transcription = model.greedy_search_decoding(audio_tensor.to(device), x_len)
    elif hasattr(model, "gready_search_decoding"):
        transcription = model.gready_search_decoding(audio_tensor.to(device), x_len)
    else:
        print("\033[91m[S2T] ERROR: No greedy method available\033[0m")
        return ""
    elapsed = time.time() - start_time
    if isinstance(transcription, list):
        result = transcription[0] if transcription else ""
    else:
        result = transcription
    if result is None:
        return ""
    result = result.lower().strip()
    print(f"\033[92m[S2T] Greedy completed in {elapsed:.3f}s: '{result}'\033[0m")
    return result


def _decode_beam(model, audio_tensor, x_len, device, start_time):
    print("\033[94m[S2T] Running beam search decoding\033[0m")
    transcription = model.beam_search_decoding(audio_tensor.to(device), x_len)
    elapsed = time.time() - start_time
    if isinstance(transcription, list):
        result = transcription[0] if transcription else ""
    else:
        result = transcription
    if result is None:
        return ""
    result = result.lower().strip()
    print(f"\033[92m[S2T] Beam completed in {elapsed:.3f}s: '{result}'\033[0m")
    return result


def _decode_hf_lm(model, audio_tensor, x_len, device, start_time):
    print("\033[94m[S2T] Running HF LM beam search decoding\033[0m")
    hf_decoder = get_hf_decoder()
    if hf_decoder is None:
        print("\033[91m[S2T] ERROR: HF decoder not initialized\033[0m")
        return "[HF_LM_ERROR]"
    try:
        transcription = model.hf_lm_beam_search_decoding(audio_tensor.to(device), x_len, hf_decoder)
        elapsed = time.time() - start_time
        if isinstance(transcription, list):
            result = transcription[0] if transcription else ""
        else:
            result = transcription
        if result is None:
            return ""
        result = result.lower().strip()
        print(f"\033[92m[S2T] HF LM completed in {elapsed:.3f}s: '{result}'\033[0m")
        return result
    except Exception as e:
        print(f"\033[91m[S2T] HF_LM_ERROR: {e}\033[0m")
        print("\033[93m[S2T] Marking with error prefix\033[0m")
        return "[HF_LM_ERROR]"


def _decode_gemini(audio_tensor, start_time):
    print("\033[94m[S2T] Running Gemini audio transcription\033[0m")
    if not settings.GOOGLE_API_KEY:
        print("\033[91m[S2T] ERROR: GOOGLE_API_KEY not configured\033[0m")
        return ""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GOOGLE_API_KEY)

        # Convert tensor to WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        # Save audio tensor as WAV
        torchaudio.save(temp_path, audio_tensor.cpu(), 16000, format="wav")

        try:
            # Read audio file in binary mode
            with open(temp_path, "rb") as f:
                audio_bytes = f.read()

            file_size_mb = len(audio_bytes) / (1024 * 1024)
            print(f"\033[94m[S2T] Audio file size: {file_size_mb:.2f} MB\033[0m")

            # Use File API if > 18MB (safe margin for 20MB limit)
            if file_size_mb > 18:
                print("\033[94m[S2T] File > 18MB, using File API upload\033[0m")
                # Upload file
                uploaded_file = client.files.upload(file=temp_path, config={"mime_type": "audio/wav"})
                print(f"\033[94m[S2T] Uploaded file: {uploaded_file.name}\033[0m")

                # Generate content
                response = client.models.generate_content(model=settings.GEMINI_MODEL_ID, contents=[PROMPT_TEMPLATE, uploaded_file])
            else:
                # Create content part from audio bytes
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

                # Generate transcript
                response = client.models.generate_content(model=settings.GEMINI_MODEL_ID, contents=[PROMPT_TEMPLATE, audio_part])

            elapsed = time.time() - start_time
            try:
                result = response.text.lower()
                result = result.replace("\n", " ")
                # remove (<content>) [<content>] patterns
                import re
                result = re.sub(r"[\(\[\{].*?[\)\]\}]", "", result)
                # remove all special symbols and double spaces !@#$%^&*<>?,./;:'"\|`~ " '
                result = re.sub(r"[!@#$%^&*<>?,./;:'\"\\|`~]", "", result)
                if not result:
                    print("\033[93m[S2T] Gemini returned empty transcript\033[0m")
                    return ""
                if result:
                    result = result.strip()
                print(f"\033[92m[S2T] Gemini raw result: '{result}'\033[0m")
            except Exception as e:
                print(f"\033[91m[S2T] ERROR extracting text from Gemini response: {e}\033[0m")
                return ""

            # Don't flatten newlines to preserve diarization format
            # result = re.sub("\n+", " ", result)
            print(f"\033[92m[S2T] Gemini completed in {elapsed:.3f}s (length: {len(result)})\033[0m")
            return result
        except Exception as e:
            print(f"\033[91m[S2T] GEMINI_ERROR during transcription: {e}\033[0m")
            return ""

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"\033[91m[S2T] GEMINI_ERROR: {e}\033[0m")
        return ""
