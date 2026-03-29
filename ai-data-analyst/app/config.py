import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    host: str = "0.0.0.0"
    port: int = 8000
    max_file_size_mb: int = 50
    # Keep as str so pydantic-settings never tries json.loads() on it.
    # Use get_allowed_extensions() wherever a list is needed.
    allowed_extensions: str = "csv,xlsx,json"
    upload_dir: str = "uploads"

    def get_allowed_extensions(self) -> list[str]:
        return [e.strip() for e in self.allowed_extensions.split(",") if e.strip()]

    def model_post_init(self, __context: object) -> None:
        os.makedirs(self.upload_dir, exist_ok=True)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
