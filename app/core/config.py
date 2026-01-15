from typing import Annotated

from pydantic import (
    AnyUrl,
    BeforeValidator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.vault_loader import load_config_from_api_v2

load_config_from_api_v2()


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

    # Redis Configuration
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB_S2T: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_USER: str = ""

    # Google AI Configuration
    GOOGLE_API_KEY: str = ""

    # External Model Configuration

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
