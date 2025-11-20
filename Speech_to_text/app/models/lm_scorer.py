"""
Language Model Scorer for CTC Beam Search Rescoring
Supports multiple LM backends including Google Gemini API
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import google.genai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseLMScorer(ABC):
    """Abstract base class for language model scorers"""

    @abstractmethod
    def score(self, text: str, candidates: List[str]) -> List[float]:
        """
        Score a list of candidate continuations given context text.

        Args:
            text: Context text (partial transcription)
            candidates: List of candidate continuation strings

        Returns:
            List of log probabilities for each candidate
        """
        pass

    @abstractmethod
    def score_complete(self, text: str) -> float:
        """
        Score a complete text sequence.

        Args:
            text: Complete text to score

        Returns:
            Log probability of the text
        """
        pass


class GeminiLMScorer(BaseLMScorer):
    """
    Google Gemini API-based language model scorer.
    Uses Gemini Flash 8B for high-quality, fast LM rescoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash-8b",
        temperature: float = 0.0,
        max_tokens: int = 50,
    ):
        """
        Initialize Gemini LM scorer.

        Args:
            api_key: Google API key (defaults to settings.GOOGLE_API_KEY)
            model_name: Gemini model name
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env or pass to constructor.")

        # Configure Gemini API
        genai.configure(api_key=self.api_key)

        # Initialize model with optimized configuration
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                candidate_count=1,
            ),
        )

        logger.info(f"[LM] Initialized GeminiLMScorer with model={self.model_name}, temperature={self.temperature}")

    def score(self, text: str, candidates: List[str]) -> List[float]:
        """
        Score candidate continuations using Gemini API.
        Uses perplexity-based scoring via prompt engineering.

        Args:
            text: Context text
            candidates: List of candidate continuations

        Returns:
            List of log probabilities (negative perplexity estimates)
        """
        scores = []

        for candidate in candidates:
            try:
                # Construct the complete sequence
                full_text = text + " " + candidate if text else candidate

                # Score using perplexity estimation
                score = self.score_complete(full_text)
                scores.append(score)

            except Exception as e:
                logger.warning(f"[LM] Error scoring candidate '{candidate}': {e}")
                # Return very low score for failed candidates
                scores.append(-100.0)

        return scores

    def score_complete(self, text: str) -> float:
        """
        Score complete text using Gemini API.
        Uses a prompt-based approach to estimate text quality/fluency.

        Args:
            text: Complete text to score

        Returns:
            Log probability estimate (higher = more fluent)
        """
        try:
            # Construct scoring prompt
            prompt = f"""Đánh giá mức độ tự nhiên và đúng ngữ pháp của câu tiếng Việt sau đây.
Cho điểm từ 0 đến 100, trong đó:
- 100: Hoàn toàn tự nhiên, đúng ngữ pháp
- 50: Chấp nhận được nhưng có vấn đề nhỏ
- 0: Hoàn toàn sai hoặc không có nghĩa

Chỉ trả về một số từ 0-100, không giải thích.

Câu: "{text}"

Điểm:"""

            # Generate score using Gemini
            response = self.model.generate_content(prompt)

            # Parse score from response
            score_text = response.text.strip()

            # Extract numeric score
            try:
                score = float(score_text)
                # Clamp to valid range
                score = max(0, min(100, score))
            except ValueError:
                # Try to extract first number from response
                import re

                numbers = re.findall(r"\d+\.?\d*", score_text)
                if numbers:
                    score = float(numbers[0])
                    score = max(0, min(100, score))
                else:
                    logger.warning(f"[LM] Could not parse score from: {score_text}, using default")
                    score = 50.0

            # Convert to log probability scale (0-100 → log scale)
            # Higher Gemini score → higher log prob
            log_prob = math.log(score + 1) - math.log(101)  # Normalize to [-4.6, 0]

            return log_prob

        except Exception as e:
            logger.error(f"[LM] Error scoring text '{text[:50]}...': {e}")
            return -10.0  # Default low score

    def score_batch(self, contexts: List[str], candidates_list: List[List[str]]) -> List[List[float]]:
        """
        Score multiple contexts and their candidates in batch.

        Args:
            contexts: List of context strings
            candidates_list: List of candidate lists (one per context)

        Returns:
            List of score lists (one per context)
        """
        results = []
        for context, candidates in zip(contexts, candidates_list, strict=False):
            scores = self.score(context, candidates)
            results.append(scores)
        return results


class NoOpLMScorer(BaseLMScorer):
    """
    No-op language model scorer that returns uniform scores.
    Used when LM rescoring is disabled.
    """

    def score(self, text: str, candidates: List[str]) -> List[float]:
        """Return uniform scores (0.0) for all candidates"""
        return [0.0] * len(candidates)

    def score_complete(self, text: str) -> float:
        """Return uniform score (0.0) for complete text"""
        return 0.0


def get_lm_scorer(
    lm_type: Optional[str] = None,
    **kwargs,
) -> BaseLMScorer:
    """
    Factory function to create LM scorer based on type.

    Args:
        lm_type: Type of LM scorer ("gemini", "none", or None)
        **kwargs: Additional arguments for specific scorer types

    Returns:
        BaseLMScorer instance
    """
    lm_type = lm_type or settings.LM_TYPE

    if lm_type == "gemini":
        return GeminiLMScorer(
            model_name=kwargs.get("model_name", settings.LM_MODEL_NAME),
            temperature=kwargs.get("temperature", settings.LM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", settings.LM_MAX_TOKENS),
        )
    elif lm_type == "none" or lm_type is None:
        return NoOpLMScorer()
    else:
        raise ValueError(f"Unknown LM type: {lm_type}")
