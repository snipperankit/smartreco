import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.config import settings
from app.database import SessionLocal, init_db
from app.models import User
from app.routers import admin_analytics, auth, events, pages, products, recommendations
from app.scheduler import shutdown_scheduler, start_scheduler
from app.security import hash_password, verify_password

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
    if not settings.admin_email or not settings.admin_password:
        log.warning("Admin bootstrap disabled; set ADMIN_EMAIL and ADMIN_PASSWORD")
        return
    async with SessionLocal() as db:
        res = await db.execute(select(User).where(User.email == settings.admin_email))
        admin = res.scalar_one_or_none()
        if admin is None:
            db.add(
                User(
                    email=settings.admin_email,
                    password_hash=hash_password(settings.admin_password),
                    role="admin",
                )
            )
            await db.commit()
            log.info("Bootstrapped admin: %s", settings.admin_email)
        elif not verify_password(settings.admin_password, admin.password_hash):
            admin.password_hash = hash_password(settings.admin_password)
            admin.role = "admin"
            await db.commit()
            log.info("Rotated bootstrap admin credentials: %s", settings.admin_email)


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
            if settings.seed_demo_users:
                await seed_personas()
            log.info("Auto-seed complete.")


async def _sync_retrieval_index() -> None:
    """Rehydrate BM25 and synchronize persistent vectors from SQL on every boot."""
    from app import vectorstore
    from app.models import Product

    async with SessionLocal() as db:
        products = (await db.execute(select(Product).order_by(Product.id))).scalars().all()
    vectorstore.upsert_many(
        [
            {
                "id": product.id,
                "title": product.title,
                "description": product.description,
                "category": product.category,
                "price": product.price,
                "tags": product.tags or [],
                "level": product.level,
            }
            for product in products
        ]
    )
    log.info("Retrieval index synchronized: %d products", len(products))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_observability()
    await init_db()
    await _bootstrap_admin()
    await _auto_seed()
    await _sync_retrieval_index()
    start_scheduler()
    from app.event_buffer import event_buffer
    event_buffer.start()
    yield
    event_buffer.stop()
    shutdown_scheduler()


app = FastAPI(
    title="SmartReco",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_api_docs else None,
    redoc_url="/redoc" if settings.enable_api_docs else None,
    openapi_url="/openapi.json" if settings.enable_api_docs else None,
)

_request_windows: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_rate_limits = {
    "/api/auth/login": (10, 60),
    "/api/auth/register": (5, 60),
    "/api/events": (120, 60),
    "/api/recommendations/refresh": (10, 60),
    "/api/admin": (20, 60),
}


def _rate_limit_for(path: str) -> tuple[int, int] | None:
    for prefix, limit in _rate_limits.items():
        if path.startswith(prefix):
            return limit
    return None


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    client = request.client.host if request.client else "unknown"
    limit = _rate_limit_for(request.url.path)
    if limit and settings.enable_rate_limiting:
        maximum, window_seconds = limit
        now = time.monotonic()
        window = _request_windows[(client, request.url.path)]
        while window and window[0] <= now - window_seconds:
            window.popleft()
        if len(window) >= maximum:
            return JSONResponse(
                {"detail": "Too many requests"},
                status_code=429,
                headers={"Retry-After": str(window_seconds)},
            )
        window.append(now)

    if (
        request.method not in {"GET", "HEAD", "OPTIONS"}
        and request.cookies.get("access_token")
        and not request.headers.get("Authorization")
    ):
        origin = request.headers.get("Origin")
        if origin:
            origin_url = urlsplit(origin)
            if origin_url.netloc.lower() != request.headers.get("host", "").lower():
                return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if settings.app_env.lower() == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)
app.include_router(products.router)
app.include_router(events.router)
app.include_router(recommendations.router)
app.include_router(admin_analytics.router)
app.include_router(pages.router)


@app.get("/health")
async def health():
    from app import vectorstore
    from app.event_buffer import event_buffer
    from app.rec_cache import rec_cache

    return {
        "status": "ok",
        "vectors": vectorstore.count(),
        "event_buffer": event_buffer.stats,
        "rec_cache": rec_cache.stats,
    }
