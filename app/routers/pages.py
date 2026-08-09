from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user_optional
from app.models import Product, User

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="app/screens")


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    res = await db.execute(select(Product).order_by(Product.category, Product.id))
    products = res.scalars().all()
    return templates.TemplateResponse(
        request, "index.html", {"products": products, "user": user}
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@router.get("/product/{product_id}", response_class=HTMLResponse)
async def product_page(
    product_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    product = await db.get(Product, product_id)
    if not product:
        return RedirectResponse("/")
    related = (
        await db.execute(
            select(Product)
            .where(Product.category == product.category, Product.id != product.id)
            .limit(3)
        )
    ).scalars().all()
    return templates.TemplateResponse(
        request,
        "product.html",
        {"product": product, "user": user, "related": related},
    )


@router.get("/journey", response_class=HTMLResponse)
async def journey_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    """Learner 'My journey' dashboard — interest profile + activity timeline."""
    if not user:
        return RedirectResponse("/login")
    if user.role == "admin":
        return RedirectResponse("/admin")

    # Flush pending events so the timeline is up-to-date
    from app.event_buffer import event_buffer
    await event_buffer._flush()

    from collections import Counter

    from sqlalchemy import desc, func

    from app.models import BehavioralEvent, Product, Recommendation

    events = (
        await db.execute(
            select(BehavioralEvent)
            .where(BehavioralEvent.user_id == user.id)
            .order_by(desc(BehavioralEvent.created_at))
            .limit(30)
        )
    ).scalars().all()

    # Resolve product ids in event payloads to titles for a readable timeline
    pids = {
        (e.payload or {}).get("product_id")
        for e in events
        if (e.payload or {}).get("product_id")
    }
    product_titles: dict[int, str] = {}
    if pids:
        rows = (
            await db.execute(select(Product.id, Product.title).where(Product.id.in_(pids)))
        ).all()
        product_titles = {r.id: r.title for r in rows}

    cats: Counter = Counter()
    for e in events:
        p = e.payload or {}
        if p.get("category"):
            cats[p["category"]] += 1
    total = sum(cats.values()) or 1
    interest_bars = [
        {"category": c, "pct": round(n / total * 100)}
        for c, n in cats.most_common(5)
    ]

    total_events = (
        await db.execute(
            select(func.count(BehavioralEvent.id)).where(
                BehavioralEvent.user_id == user.id
            )
        )
    ).scalar() or 0
    rec_count = (
        await db.execute(
            select(func.count(Recommendation.id)).where(
                Recommendation.user_id == user.id
            )
        )
    ).scalar() or 0

    return templates.TemplateResponse(
        request,
        "journey.html",
        {
            "user": user,
            "events": events,
            "product_titles": product_titles,
            "interest_bars": interest_bars,
            "total_events": total_events,
            "rec_count": rec_count,
        },
    )


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    if not user or user.role != "admin":
        return RedirectResponse("/login")
    res = await db.execute(select(Product).order_by(Product.id))
    products = res.scalars().all()
    from app import vectorstore

    synced_ids = vectorstore.all_ids()
    return templates.TemplateResponse(
        request,
        "admin.html",
        {"products": products, "user": user, "synced_ids": synced_ids},
    )


@router.get("/agent", response_class=HTMLResponse)
async def agent_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User | None = Depends(get_current_user_optional),
):
    learners = []
    if user and user.role == "admin":
        learners = (
            await db.execute(
                select(User).where(User.role != "admin").order_by(User.email)
            )
        ).scalars().all()
    return templates.TemplateResponse(
        request, "agent.html", {"user": user, "learners": learners}
    )
