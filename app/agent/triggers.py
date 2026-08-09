"""When (and whether) to run the expensive agent.

Production thinking lives here. We do NOT fire an LLM on every click. A run is
allowed only when a meaningful signal appears AND we're past a per-user cooldown
AND the user's behavior actually changed since the last recommendation
(behavior signature). This kills redundant, wasteful calls.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import run_recommendation
from app.config import settings
from app.models import BehavioralEvent, Product, Recommendation


async def recent_events(db: AsyncSession, user_id: int, limit: int = 20) -> list[dict]:
    res = await db.execute(
        select(BehavioralEvent)
        .where(BehavioralEvent.user_id == user_id)
        .order_by(desc(BehavioralEvent.created_at))
        .limit(limit)
    )
    rows = res.scalars().all()
    return [{"event_type": r.event_type, "payload": r.payload} for r in rows]


def _signature(events: list[dict]) -> str:
    """Stable hash of the recent behavior — same behavior => same signature."""
    parts = []
    for e in events:
        p = e.get("payload", {}) or {}
        parts.append(f"{e.get('event_type')}:{p.get('category')}:{p.get('product_id')}:{p.get('query')}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


async def _last_recommendation(
    db: AsyncSession, user_id: int
) -> Recommendation | None:
    res = await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id == user_id)
        .order_by(desc(Recommendation.updated_at))
        .limit(1)
    )
    return res.scalar_one_or_none()


def _is_meaningful(new_events: list[dict]) -> bool:
    """Decides if the event stream warrants an agent run.

    Triggers on:
      1. Any search (high-intent signal)
      2. Any click with a product_id (deliberate engagement)
      3. Repeated views in one category (≥ threshold)
      4. Total engagement volume across ALL categories (handles mixed/random
         browsing — the user explored enough to have a useful signal)
    """
    from collections import Counter

    if any(e.get("event_type") == "search" for e in new_events):
        return True
    if any(
        e.get("event_type") == "click" and (e.get("payload") or {}).get("product_id")
        for e in new_events
    ):
        return True
    cats = Counter(
        (e.get("payload") or {}).get("category")
        for e in new_events
        if e.get("event_type") in {"view", "click"}
    )
    # Single-category depth
    threshold = settings.rec_category_view_threshold
    if any(c and n >= threshold for c, n in cats.items()):
        return True
    # Cross-category volume: enough total engagement even if spread thin
    total_engaged = sum(n for c, n in cats.items() if c)
    return total_engaged >= settings.rec_total_engagement_threshold


async def maybe_generate(
    db: AsyncSession, user_id: int, *, force: bool = False
) -> Recommendation | None:
    """Central gate. Returns a Recommendation if one was (re)generated, else None.

    `force=True` bypasses the meaningful-signal check (used by the scheduler),
    but still respects the signature dedupe so a daily digest of unchanged
    behavior won't burn a call.
    """
    events = await recent_events(db, user_id, limit=20)
    if not events:
        return None

    if not force and not _is_meaningful(events):
        return None

    last = await _last_recommendation(db, user_id)
    sig = _signature(events)

    if last is not None:
        # Cooldown: don't rerun within the window.
        now = datetime.now(timezone.utc)
        last_ts = last.updated_at
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        cooling = now - last_ts < timedelta(seconds=settings.rec_cooldown_seconds)
        if not force and cooling:
            return None
        # Dedupe: identical behavior => reuse existing recommendation.
        if last.behavior_signature == sig:
            return None

    # The graph (and its LLM calls) is synchronous — run it in a worker
    # thread so a 30-100s generation doesn't freeze the whole event loop.
    result = await asyncio.to_thread(run_recommendation, user_id, events)
    if not result["product_ids"]:
        return None

    rec = Recommendation(
        user_id=user_id,
        narrative_copy=result["narrative"],
        recommended_product_ids=result["product_ids"],
        rationale=result.get("rationale", {}),
        behavior_signature=sig,
        is_sent=False,
    )
    db.add(rec)
    await db.commit()
    await db.refresh(rec)

    # Invalidate cached /latest for this user so the next poll picks up
    # the fresh recommendation immediately.
    from app.rec_cache import rec_cache
    rec_cache.invalidate(user_id)

    return rec


async def hydrate_products(
    db: AsyncSession, product_ids: list[int]
) -> list[Product]:
    if not product_ids:
        return []
    res = await db.execute(select(Product).where(Product.id.in_(product_ids)))
    by_id = {p.id: p for p in res.scalars().all()}
    return [by_id[i] for i in product_ids if i in by_id]  # preserve agent order
