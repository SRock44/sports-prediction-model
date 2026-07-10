from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Minimal settings needed by the training/eval code in this repo.

    The production system has a much larger settings surface (DB, API,
    billing, notifications, etc.) that lives in the private application
    repo and isn't relevant to training models standalone.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "prediction"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
