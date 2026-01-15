"""
Model configuration and constants for Gemini transcription.
"""

from dataclasses import dataclass
from app.utils.logging import logger


# Model Configuration
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
FILE_SIZE_THRESHOLD_MB = 18

# Prompt template for Vietnamese transcription
PROMPT_TEMPLATE = """RESPONSE IN VIETNAMESE ONLY. Listen carefully to the audio file and provide a detailed transcript with clear speaker identification.

CRITICAL OUTPUT FORMAT - STRICTLY FOLLOW THIS:
Each speaker segment MUST be on its own line starting with SPEAKER_<number>:

SPEAKER_1: [what speaker 1 says]
SPEAKER_2: [what speaker 2 says]
SPEAKER_3: [what speaker 3 says]

RULES:
1. Start each speaker segment on a NEW LINE
2. Begin each line with SPEAKER_<number>: (uppercase SPEAKER, underscore, number, colon, space)
3. Write exact words the speaker said after the colon
4. When the same speaker speaks again, use their number again
5. Preserve all Vietnamese words - DO NOT translate or change anything
6. Skip background noise, music, silence - only transcribe speech
7. Do NOT include [descriptions], [sounds], [actions], timestamps
8. Do NOT add explanations or summaries

Example - EXACTLY this format with newlines:
SPEAKER_1: Xin chào mọi người hôm nay chúng ta họp về gì
SPEAKER_2: Dạ để em báo cáo kết quả tháng này
SPEAKER_1: Tốt vậy em cứ báo cáo đi
SPEAKER_3: Em xin góp ý thêm

If NO speech detected, respond EXACTLY:
SPEAKER_0: No speech in audio

Now transcribe the audio file - remember NEWLINE between each SPEAKER segment:"""


@dataclass
class TokenUsage:
    """Token usage information from Gemini API response."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def extract_token_usage(response) -> TokenUsage:
    """
    Extract token usage information from Gemini API response.

    Args:
        response: GenerateContentResponse object from Gemini API

    Returns:
        TokenUsage object with input/output token counts
    """
    token_usage = TokenUsage()

    # Log raw response for debugging
    logger.debug(f"[TOKEN_TRACKING] Response type: {type(response)}")

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage_meta = response.usage_metadata
        logger.debug(f"[TOKEN_TRACKING] usage_metadata found")

        # Gemini API uses: prompt_token_count, candidates_token_count, total_token_count
        if hasattr(usage_meta, "prompt_token_count"):
            token_usage.input_tokens = usage_meta.prompt_token_count or 0
            logger.debug(f"[TOKEN_TRACKING] prompt_token_count: {token_usage.input_tokens}")

        if hasattr(usage_meta, "candidates_token_count"):
            token_usage.output_tokens = usage_meta.candidates_token_count or 0
            logger.debug(f"[TOKEN_TRACKING] candidates_token_count: {token_usage.output_tokens}")

        # Also log prompt_tokens_details for modality breakdown
        if hasattr(usage_meta, "prompt_tokens_details") and usage_meta.prompt_tokens_details:
            logger.debug(f"[TOKEN_TRACKING] Prompt token breakdown by modality:")
            for detail in usage_meta.prompt_tokens_details:
                logger.debug(f"[TOKEN_TRACKING]   - {detail.modality}: {detail.token_count}")
    else:
        logger.warning(f"[TOKEN_TRACKING] Response does not have usage_metadata or it's empty")
        # Fallback: Try to estimate from response text
        if hasattr(response, "text") and response.text:
            # Rough estimate: ~4 characters per token
            estimated_output_tokens = len(response.text) // 4
            token_usage.output_tokens = estimated_output_tokens
            logger.warning(f"[TOKEN_TRACKING] Using estimated output tokens: {estimated_output_tokens}")

    logger.info(f"[TOKEN_TRACKING] Final token usage: input={token_usage.input_tokens}, output={token_usage.output_tokens}, total={token_usage.total_tokens}")
    return token_usage
