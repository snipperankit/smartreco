from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Mesh API — multi-model routing (free tier, 200 RPD each)
    mesh_api_key: str = "rsk_missing"
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    mesh_chat_model: str = "tencent/hy3"          # reasoning model for creative generation
    mesh_fast_model: str = "minimax/m2-her"        # lightweight model for intent/reflection
    mesh_embed_model: str = "openai/text-embedding-3-small"

    # App
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 1440
    database_url: str = "sqlite+aiosqlite:///./smartreco.db"
    chroma_dir: str = "./chroma_store"

    # Triggering
    rec_cooldown_seconds: int = 45
    rec_category_view_threshold: int = 2
    rec_total_engagement_threshold: int = 4  # mixed-category: fire if total events >= this

    # Scheduler
    enable_scheduler: bool = True
    digest_hour: int = 15
    digest_minute: int = 0
    digest_interval_minutes: int = 0  # >0 = repeat every N min instead of daily cron
    delivery_channels: str = "mailbox"  # comma-separated: mailbox,telegram

    # Telegram Bot API (free — create bot via @BotFather)
    telegram_bot_token: str = ""  # from @BotFather
    telegram_chat_id: str = ""   # your chat ID from @userinfobot

    # Observability
    langsmith_tracing: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "smartreco"

    # Bootstrap admin
    admin_email: str = "admin@smartreco.dev"
    admin_password: str = "admin1234"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
