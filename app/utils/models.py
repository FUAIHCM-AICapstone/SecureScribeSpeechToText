"""
Model configuration and constants for Gemini transcription.
"""

from dataclasses import dataclass
from app.utils.logging import logger


# Model Configuration
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
FILE_SIZE_THRESHOLD_MB = 18

# Prompt template for Vietnamese transcription
PROMPT_TEMPLATE = """RESPONSE IN VIETNAMESE: Listen carefully to the following audio file. PROVIDE DETAIL TRANSCRIPT WITH SPEAKER DIARIZATION IN VIETNAMESE Listen carefully, focus on speaker diarization, and provide a detailed transcript in Vietnamese. Reduce the line of speech, only insert new line if new speaker start speaking. Focus on matching the voice to a correct speaker. Format: SPEAKER_<number>: <transcript that you hear>

If you not hear any speak, just say there is no speaker in the audio, skip the background noise, only focus on the speaker. NO EXTRA INFORMATION NEEDED.""".strip().replace("\n", " ")


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
