"""Seed realistic demo personas with behavioral history.

Run after seed.py to populate the platform with 4 distinct users whose
browsing patterns differ. This makes the admin dashboard meaningful and
proves that the agent produces DIFFERENT recommendations for DIFFERENT
behavioral profiles — the single strongest demo signal for a judge.

Usage:
    python seed_demo.py
"""
import asyncio
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.database import SessionLocal, init_db
from app.models import BehavioralEvent, Product, User
from app.security import hash_password

PERSONAS = [
    {
        "email": "maya@demo.dev",
        "password": "demo1234",
        "story": "Agent builder — deep into agentic-ai, searches for LangGraph and RAG",
        "target_categories": ["agentic-ai"],
        "searches": ["langgraph agents", "agentic ai workflow", "RAG pipeline"],
    },
    {
        "email": "raj@demo.dev",
        "password": "demo1234",
        "story": "ML engineer — focused on machine-learning, explores transformers and MLOps",
        "target_categories": ["machine-learning"],
        "searches": ["deep learning pytorch", "transformers attention", "mlops deploy"],
    },
    {
        "email": "sofia@demo.dev",
        "password": "demo1234",
        "story": "Full-stack dev — browses python and web-dev, practical builder",
        "target_categories": ["python", "web-dev"],
        "searches": ["fastapi production", "react hooks", "async python"],
    },
    {
        "email": "kai@demo.dev",
        "password": "demo1234",
        "story": "Explorer — broad interest across LLM, career, and cloud-devops",
        "target_categories": ["llm", "career", "cloud-devops"],
        "searches": ["prompt engineering", "system design interview", "kubernetes"],
    },
]


async def main() -> None:
    await init_db()
    async with SessionLocal() as db:
        products = (await db.execute(select(Product))).scalars().all()
        if not products:
            print("No products found. Run seed.py first.")
            return

        by_cat: dict[str, list[Product]] = {}
        for p in products:
            by_cat.setdefault(p.category, []).append(p)

        now = datetime.now(timezone.utc)
        created = 0

        for persona in PERSONAS:
            # Skip if already exists
            exists = await db.execute(
                select(User).where(User.email == persona["email"])
            )
            if exists.scalar_one_or_none():
                print(f"  {persona['email']} already exists, skipping")
                continue

            user = User(
                email=persona["email"],
                password_hash=hash_password(persona["password"]),
                role="user",
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

            # Generate 15-25 realistic behavioral events over the last 3 hours
            events = []
            t = now - timedelta(hours=3)

            # Searches
            for q in persona["searches"]:
                t += timedelta(minutes=random.randint(2, 8))
                cat = persona["target_categories"][0]
                events.append(
                    BehavioralEvent(
                        user_id=user.id,
                        event_type="search",
                        payload={"query": q, "category": cat},
                        created_at=t,
                    )
                )

            # Product views in target categories (heavy engagement)
            for cat in persona["target_categories"]:
                cat_products = by_cat.get(cat, [])
                viewed = random.sample(
                    cat_products, min(len(cat_products), random.randint(3, 5))
                )
                for p in viewed:
                    t += timedelta(minutes=random.randint(1, 5))
                    events.append(
                        BehavioralEvent(
                            user_id=user.id,
                            event_type="view",
                            payload={
                                "product_id": p.id,
                                "category": p.category,
                                "time_spent": random.randint(15, 90),
                            },
                            created_at=t,
                        )
                    )
                    # Some clicks on viewed products
                    if random.random() > 0.4:
                        t += timedelta(seconds=random.randint(10, 30))
                        events.append(
                            BehavioralEvent(
                                user_id=user.id,
                                event_type="click",
                                payload={
                                    "product_id": p.id,
                                    "category": p.category,
                                    "label": "enroll",
                                },
                                created_at=t,
                            )
                        )

            # A few off-topic views (realism — users wander)
            other_cats = [c for c in by_cat if c not in persona["target_categories"]]
            if other_cats:
                wander_cat = random.choice(other_cats)
                wander_p = random.choice(by_cat[wander_cat])
                t += timedelta(minutes=random.randint(3, 10))
                events.append(
                    BehavioralEvent(
                        user_id=user.id,
                        event_type="view",
                        payload={
                            "product_id": wander_p.id,
                            "category": wander_p.category,
                            "time_spent": random.randint(5, 15),
                        },
                        created_at=t,
                    )
                )

            db.add_all(events)
            await db.commit()
            created += 1
            print(
                f"  {persona['email']} — {len(events)} events, "
                f"focus: {', '.join(persona['target_categories'])}"
            )

        print(f"\nCreated {created} demo personas. Login with password: demo1234")
        print("Run the app and check /admin to see their behavioral profiles.\n")
        print("Personas:")
        for p in PERSONAS:
            print(f"  {p['email']:20s} — {p['story']}")


if __name__ == "__main__":
    asyncio.run(main())
