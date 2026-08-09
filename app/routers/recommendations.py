from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.triggers import hydrate_products, maybe_generate
from app.database import get_db
from app.deps import get_current_user
from app.models import Recommendation, User
from app.rec_cache import rec_cache
from app.schemas import ProductOut, RecommendationOut

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


async def _serialize(db: AsyncSession, rec: Recommendation) -> RecommendationOut:
    products = await hydrate_products(db, rec.recommended_product_ids)
    return RecommendationOut(
        id=rec.id,
        narrative_copy=rec.narrative_copy,
        recommended_product_ids=rec.recommended_product_ids,
        rationale=rec.rationale or {},
        products=[ProductOut.model_validate(p) for p in products],
        updated_at=rec.updated_at,
    )


@router.get("/latest", response_model=RecommendationOut | None)
async def latest(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    """Serves from in-memory cache when available. At 100k users polling
    every 10s, this absorbs ~10k reads/sec that would otherwise hit the DB.
    Cache hit rate is exposed on /health for observability."""
    cached = rec_cache.get(user.id)
    if cached is not None:
        return cached

    res = await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id == user.id)
        .order_by(desc(Recommendation.updated_at))
        .limit(1)
    )
    rec = res.scalar_one_or_none()
    if not rec:
        return None
    out = await _serialize(db, rec)
    rec_cache.set(user.id, out)
    return out


@router.post("/refresh", response_model=RecommendationOut | None)
async def refresh(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    """Manual/demo trigger — forces a run past the meaningful-signal gate but
    still respects the behavior-signature dedupe. Invalidates cache on success."""
    rec = await maybe_generate(db, user.id, force=True)
    rec_cache.invalidate(user.id)  # force next /latest to read fresh
    if rec is None:
        return await latest(db, user)
    out = await _serialize(db, rec)
    rec_cache.set(user.id, out)
    return out


@router.get("/stream")
async def stream_recommendations(user: User = Depends(get_current_user)):
    """SSE stream: pushes the full latest recommendation on connect and again
    whenever it changes (agent runs from browsing signals, digest, refresh).
    The homepage listens with EventSource so recs update without reloads."""
    import asyncio

    from fastapi.responses import StreamingResponse

    from app.database import SessionLocal

    async def gen():
        last_stamp = None
        while True:
            try:
                async with SessionLocal() as db:
                    res = await db.execute(
                        select(Recommendation)
                        .where(Recommendation.user_id == user.id)
                        .order_by(desc(Recommendation.updated_at))
                        .limit(1)
                    )
                    rec = res.scalar_one_or_none()
                    if rec is not None and rec.updated_at != last_stamp:
                        last_stamp = rec.updated_at
                        out = await _serialize(db, rec)
                        yield f"data: {out.model_dump_json()}\n\n"
                    else:
                        yield ": keepalive\n\n"
            except Exception:
                yield ": keepalive\n\n"
            await asyncio.sleep(3)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/latest/narrative-stream")
async def stream_narrative(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    """Server-Sent Events stream of the narrative for the user's latest rec.

    Cuts perceived latency: the user sees words appear at token-1 (~200ms)
    instead of waiting for the full response (~1500ms). Same total server
    cost, dramatically better UX. At 100k users this is the single most
    impactful latency optimization we can ship without breaking anything.
    """
    from fastapi.responses import StreamingResponse

    from app.mesh import chat_stream

    res = await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id == user.id)
        .order_by(desc(Recommendation.updated_at))
        .limit(1)
    )
    rec = res.scalar_one_or_none()
    if not rec:
        async def empty():
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    products = await hydrate_products(db, rec.recommended_product_ids)
    catalog = "\n".join(
        f"- {p.title} [{p.category}]" for p in products
    )
    rationale = rec.rationale or {}
    strategy = rationale.get("strategy", "balanced")

    def gen():
        try:
            for chunk in chat_stream(
                [
                    {
                        "role": "system",
                        "content": "Rewrite this recommendation as a short 2-sentence "
                        "streaming pitch. Be specific to the learner's strategy.",
                    },
                    {
                        "role": "user",
                        "content": f"Strategy: {strategy}\nCourses:\n{catalog}",
                    },
                ],
                temperature=0.7,
                # Reasoning models spend tokens on hidden reasoning first;
                # a small cap would end the stream before any visible text.
                max_tokens=2000,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as ex:
            yield f"data: [error: {type(ex).__name__}]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
