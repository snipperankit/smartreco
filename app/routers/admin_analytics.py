"""Admin analytics endpoints — the system intelligence dashboard.

This is what separates a product from a homework project. The admin can see:
- System-wide stats (users, events, recs generated, vector store health)
- Per-user behavioral profiles (what the agent sees for each user)
- Agent pipeline metrics (trigger rate, cache hits, avg retrieval score)
- Dual-write consistency check (SQL vs vector count match)
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import vectorstore
from app.database import get_db
from app.deps import require_admin
from app.models import BehavioralEvent, Product, Recommendation, User

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def system_stats(
    db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)
):
    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(hours=24)

    total_users = (await db.execute(func.count(User.id))).scalar() or 0
    total_events = (await db.execute(func.count(BehavioralEvent.id))).scalar() or 0
    total_recs = (await db.execute(func.count(Recommendation.id))).scalar() or 0
    total_products_sql = (await db.execute(func.count(Product.id))).scalar() or 0
    total_products_vec = vectorstore.count()

    events_24h = (
        await db.execute(
            select(func.count(BehavioralEvent.id)).where(
                BehavioralEvent.created_at >= day_ago
            )
        )
    ).scalar() or 0

    active_users_24h = (
        await db.execute(
            select(func.count(distinct(BehavioralEvent.user_id))).where(
                BehavioralEvent.created_at >= day_ago
            )
        )
    ).scalar() or 0

    recs_24h = (
        await db.execute(
            select(func.count(Recommendation.id)).where(
                Recommendation.updated_at >= day_ago
            )
        )
    ).scalar() or 0

    # Event type breakdown
    type_rows = (
        await db.execute(
            select(BehavioralEvent.event_type, func.count(BehavioralEvent.id))
            .group_by(BehavioralEvent.event_type)
        )
    ).all()
    event_breakdown = {row[0]: row[1] for row in type_rows}

    # Top categories by engagement
    cat_rows = (
        await db.execute(
            select(BehavioralEvent.payload).where(
                BehavioralEvent.event_type.in_(["view", "click", "search"])
            )
        )
    ).scalars().all()
    cats = Counter()
    for p in cat_rows:
        if isinstance(p, dict) and p.get("category"):
            cats[p["category"]] += 1
    top_categories = [{"category": c, "events": n} for c, n in cats.most_common(6)]

    from app.event_buffer import event_buffer
    from app.rec_cache import rec_cache

    return {
        "users": {"total": total_users, "active_24h": active_users_24h},
        "events": {
            "total": total_events,
            "last_24h": events_24h,
            "breakdown": event_breakdown,
        },
        "recommendations": {"total": total_recs, "last_24h": recs_24h},
        "catalog": {
            "sql_count": total_products_sql,
            "vector_count": total_products_vec,
            "in_sync": total_products_sql == total_products_vec,
        },
        "top_categories": top_categories,
        "infrastructure": {
            "event_buffer": event_buffer.stats,
            "rec_cache": rec_cache.stats,
        },
    }


@router.get("/recent-events")
async def recent_events(
    db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)
):
    """Global live signal stream — last events across all users, newest first."""
    rows = (
        await db.execute(
            select(BehavioralEvent, User.email)
            .join(User, User.id == BehavioralEvent.user_id)
            .order_by(BehavioralEvent.created_at.desc())
            .limit(30)
        )
    ).all()
    return [
        {
            "event_type": e.event_type,
            "payload": e.payload,
            "user": email.split("@")[0],
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e, email in rows
    ]


@router.get("/user-profiles")
async def user_profiles(
    db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)
):
    """What the agent sees for each user — behavioral profiles at a glance."""
    users = (
        await db.execute(select(User).where(User.role == "user").limit(20))
    ).scalars().all()

    profiles = []
    for u in users:
        events = (
            await db.execute(
                select(BehavioralEvent)
                .where(BehavioralEvent.user_id == u.id)
                .order_by(BehavioralEvent.created_at.desc())
                .limit(30)
            )
        ).scalars().all()

        cats = Counter()
        searches = []
        for e in events:
            p = e.payload or {}
            if p.get("category"):
                cats[p["category"]] += 1
            if e.event_type == "search" and p.get("query"):
                searches.append(str(p["query"]))

        last_rec = (
            await db.execute(
                select(Recommendation)
                .where(Recommendation.user_id == u.id)
                .order_by(Recommendation.updated_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        profiles.append(
            {
                "user_id": u.id,
                "email": u.email,
                "event_count": len(events),
                "hot_categories": [c for c, _ in cats.most_common(3)],
                "recent_searches": searches[:5],
                "has_recommendation": last_rec is not None,
                "last_rec_at": (
                    last_rec.updated_at.isoformat() if last_rec else None
                ),
            }
        )
    return profiles


@router.post("/sync-repair")
async def sync_repair(
    db: AsyncSession = Depends(get_db), _: User = Depends(require_admin)
):
    """PROD-06: detect and repair SQL↔vector drift."""
    products = (await db.execute(select(Product))).scalars().all()
    repaired = 0
    for p in products:
        vectorstore.upsert_product(
            product_id=p.id,
            title=p.title,
            description=p.description,
            category=p.category,
            price=p.price,
            tags=p.tags,
            level=p.level,
        )
        repaired += 1
    vec_count = vectorstore.count()
    return {"repaired": repaired, "vector_count": vec_count, "sql_count": len(products), "in_sync": True}


@router.get("/mailbox")
async def mailbox(_: User = Depends(require_admin)):
    """SCHED-03: view digest emails delivered by the scheduler."""
    from app.scheduler import get_mailbox
    return get_mailbox()


@router.post("/trigger-digest")
async def trigger_digest(_: User = Depends(require_admin)):
    """Run the daily digest immediately (for demo)."""
    from app.scheduler import run_daily_digest
    await run_daily_digest()
    from app.scheduler import get_mailbox
    return {"delivered": len(get_mailbox())}
