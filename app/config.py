from functools import lru_cache
from secrets import token_urlsafe

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"

    # Mesh API — multi-model routing (free tier, 200 RPD each)
    mesh_api_key: str = ""
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    mesh_chat_model: str = "tencent/hy3"          # reasoning model for creative generation
    mesh_fast_model: str = "minimax/m2-her"        # lightweight model for intent/reflection
    mesh_embed_model: str = "openai/text-embedding-3-small"

    # App
    secret_key: str = ""
    cookie_secure: bool = False
    enable_api_docs: bool = True
    enable_rate_limiting: bool = True
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
    telegram_recipient_email: str = ""

    # Observability
    langsmith_tracing: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "smartreco"

    # Bootstrap admin
    admin_email: str = ""
    admin_password: str = ""

    # Demo data is opt-in outside local development.
    seed_demo_users: bool = False

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.app_env.lower() != "production":
            if not self.secret_key:
                self.secret_key = token_urlsafe(48)
            return self
        if len(self.secret_key) < 32:
            raise ValueError("SECRET_KEY must contain at least 32 characters")
        if not self.mesh_api_key:
            raise ValueError("MESH_API_KEY is required in production")
        if not self.admin_email or len(self.admin_password) < 12:
            raise ValueError(
                "ADMIN_EMAIL and an ADMIN_PASSWORD of at least 12 characters are required"
            )
        telegram_values = (
            self.telegram_bot_token,
            self.telegram_chat_id,
            self.telegram_recipient_email,
        )
        if any(telegram_values) and not all(telegram_values):
            raise ValueError(
                "TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and "
                "TELEGRAM_RECIPIENT_EMAIL must be configured together"
            )
        self.cookie_secure = True
        self.enable_api_docs = False
        self.enable_rate_limiting = True
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
