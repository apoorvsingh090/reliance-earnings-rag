"""Central application configuration (env-driven, no secrets committed)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    # Data layout
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "reliance"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    extracted_dir: Path = PROJECT_ROOT / "data" / "extracted"

    # DB (used from Milestone 2+; Milestone 1 works without a live DB)
    database_url: str = "postgresql+psycopg://reliance:reliance@localhost:5432/reliance_rag"

    # Retrieval (Milestone 3+)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # LLM (Milestone 6+)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

    @property
    def registry_path(self) -> Path:
        return self.processed_dir / "registry.json"

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.json"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / "bm25_index.pkl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # Allow DATA_DIR override to relocate everything (useful for tests).
    data_dir_override = os.environ.get("DATA_DIR")
    if data_dir_override:
        base = Path(data_dir_override)
        settings.data_dir = base
        settings.raw_dir = base / "raw" / "reliance"
        settings.processed_dir = base / "processed"
        settings.extracted_dir = base / "extracted"
    return settings
