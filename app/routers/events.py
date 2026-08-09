from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.triggers import maybe_generate
from app.database import SessionLocal, get_db
from app.deps import get_current_user
from app.models import BehavioralEvent, User
from app.schemas import EventBatch

router = APIRouter(prefix="/api/events", tags=["events"])


async def _run_agent_async(user_id: int) -> None:
    """Runs in a background task with its own session — never blocks ingestion."""
    from app.event_buffer import event_buffer
    await event_buffer._flush()  # ensure events from this request are in DB
    async with SessionLocal() as db:
        await maybe_generate(db, user_id)


@router.post("/track", status_code=202)
async def track(
    batch: EventBatch,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
):
    """Accepts a batch of events. Pushes into the in-memory event buffer
    (which flushes to DB in bulk on a timer) — zero per-request DB commits.
    At 100k users this is the difference between 8k commits/sec and ~4
    bulk INSERTs/sec."""
    from app.event_buffer import event_buffer
    from app.speculative import speculative_cache

    for e in batch.events:
        event_buffer.push(user.id, e.type, e.payload, e.session_id)

        # Speculative pre-warm: on a search or click event, kick off
        # retrieval BEFORE the trigger fires. When the threshold hits, the
        # agent picks up cached hits instead of re-running the query.
        # This cuts trigger latency from ~1500ms to ~400ms on cache hit.
        if e.type in ("search", "click"):
            payload = e.payload or {}
            query = payload.get("query") or payload.get("category")
            if query:
                cat = payload.get("category")
                background.add_task(
                    speculative_cache.speculate,
                    user.id,
                    str(query),
                    [],  # exclude_ids filled later by full trigger
                    [cat] if cat else [],
                )

    # The trigger layer itself decides whether this actually warrants an AI call.
    # Admins aren't learners — don't burn LLM calls generating recs for them.
    if user.role != "admin":
        background.add_task(_run_agent_async, user.id)
    return {"accepted": len(batch.events)}


@router.post("/track-beacon", status_code=202)
async def track_beacon(request: Request, db: AsyncSession = Depends(get_db)):
    """navigator.sendBeacon posts as text/plain; parse manually. Auth via cookie."""
    from app.deps import get_current_user as _gcu

    user = await _gcu(request, db)
    import json

    raw = (await request.body()).decode("utf-8") or "{}"
    data = json.loads(raw)
    events = data.get("events", [])
    rows = [
        BehavioralEvent(
            user_id=user.id,
            event_type=e.get("type", "unknown"),
            payload=e.get("payload", {}),
        )
        for e in events
    ]
    db.add_all(rows)
    await db.commit()
    return {"accepted": len(rows)}


@router.get("/recent")
async def recent(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    """Last 10 events for the live activity feed in the sidebar."""
    res = await db.execute(
        select(BehavioralEvent)
        .where(BehavioralEvent.user_id == user.id)
        .order_by(desc(BehavioralEvent.created_at))
        .limit(10)
    )
    rows = res.scalars().all()
    return [
        {
            "event_type": r.event_type,
            "payload": r.payload,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
