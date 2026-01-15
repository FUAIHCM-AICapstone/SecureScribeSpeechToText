from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses"""

    page: int
    limit: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class TokenUsageSchema(BaseModel):
    """Token usage information from Gemini API"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class SpeakerSegmentSchema(BaseModel):
    """Speaker segment from diarization"""

    speaker: str
    start_time: float
    end_time: float
    transcription: str


class TranscriptionResultSchema(BaseModel):
    """Transcription results with speakers and token usage"""

    transcript: str
    speakers: List[SpeakerSegmentSchema] = []
    speaker_lines: int = 0
    token_usage: TokenUsageSchema


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper"""

    success: bool = True
    message: Optional[str] = None
    data: Optional[T] = None
    errors: Optional[List[str]] = None


class PaginatedResponse(ApiResponse[List[T]], Generic[T]):
    """Paginated API response with metadata"""

    pagination: Optional[PaginationMeta] = None


def create_pagination_meta(page: int, limit: int, total: int) -> PaginationMeta:
    """Create pagination metadata"""
    total_pages = (total + limit - 1) // limit  # Ceiling division
    return PaginationMeta(
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
