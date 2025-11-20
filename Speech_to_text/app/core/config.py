from typing import Annotated

from pydantic import (
    AnyUrl,
    BeforeValidator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )
    # API Configuration
    API_V1_STR: str = "/api/v1"

    # Server Configuration
    SERVER_NAME: str = "SecureScribeBE"
    SERVER_HOST: str = "http://localhost"
    SERVER_PORT: int = 9999

    # CORS Configuration
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str,
        BeforeValidator(lambda x: x.split(",") if isinstance(x, str) else x),
    ] = []

    # Project Configuration
    PROJECT_NAME: str = "SecureScribeBE"

    # Redis Configuration
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB_S2T: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_USER: str = ""

    # Google AI Configuration
    GOOGLE_API_KEY: str = ""

    # Language Model Configuration
    LM_TYPE: str = "gemini"  # Options: "none", "gemini"
    LM_WEIGHT: float = 0.3  # Weight for LM score in shallow fusion (0.0 to 1.0)
    LM_MODEL_NAME: str = "gemini-1.5-flash-8b"  # Gemini model for LM rescoring
    LM_TEMPERATURE: float = 0.0  # Temperature for LM generation (0.0 = deterministic)
    LM_MAX_TOKENS: int = 50  # Max tokens for LM continuation

    # HuggingFace Configuration
    HF_TOKEN: str = ""

    # Indexing Configuration

    @computed_field  # type: ignore[prop-decorator]
    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_S2T}"

    @computed_field
    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB_S2T}"


settings = Settings()
for attr, value in settings.model_dump().items():
    print(f"[CONFIG] {attr} = {value}")
