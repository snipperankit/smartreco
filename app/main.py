import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import User
from app.routers import admin_analytics, auth, events, pages, products, recommendations
from app.scheduler import shutdown_scheduler, start_scheduler
from app.security import hash_password

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smartreco")


def _configure_observability() -> None:
    """LangSmith tracing (bonus) — opt-in via env, no-op otherwise."""
    if settings.langsmith_tracing and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        log.info("LangSmith tracing enabled -> project=%s", settings.langchain_project)


async def _bootstrap_admin() -> None:
    async with SessionLocal() as db:
        res = await db.execute(select(User).where(User.email == settings.admin_email))
        if res.scalar_one_or_none() is None:
            db.add(
                User(
                    email=settings.admin_email,
                    password_hash=hash_password(settings.admin_password),
                    role="admin",
                )
            )
            await db.commit()
            log.info("Bootstrapped admin: %s", settings.admin_email)


async def _auto_seed() -> None:
    """Seed 50 courses + demo personas if DB is empty (first deploy)."""
    async with SessionLocal() as db:
        from app.models import Product
        res = await db.execute(select(Product).limit(1))
        if res.scalar_one_or_none() is None:
            log.info("Empty catalog detected — running auto-seed…")
            from seed import main as seed_catalog
            from seed_demo import main as seed_personas
            try:
                await seed_catalog()
            except Exception as e:
                log.warning("Auto-seed catalog failed (will retry next restart): %s", e)
                return
            await seed_personas()
            log.info("Auto-seed complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_observability()
    await init_db()
    await _bootstrap_admin()
    await _auto_seed()
    start_scheduler()
    from app.event_buffer import event_buffer
    event_buffer.start()
    yield
    event_buffer.stop()
    shutdown_scheduler()


app = FastAPI(title="SmartReco", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)
app.include_router(products.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(admin_analytics.router)
app.include_router(pages.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
