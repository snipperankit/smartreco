from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user")  # user | admin
    proactive_delivery_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_digest_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    events: Mapped[list["BehavioralEvent"]] = relationship(back_populates="user")
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="user"
    )


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    tags: Mapped[list] = mapped_column(JSON, default=list)  # list[str]
    level: Mapped[str] = mapped_column(String(32), default="all")  # beginner/advanced
    thumbnail_url: Mapped[str] = mapped_column(String(512), default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class BehavioralEvent(Base):
    __tablename__ = "behavioral_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), index=True)  # view/search/click/hover
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # query, time_spent, product_id, category
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    user: Mapped["User"] = relationship(back_populates="events")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    narrative_copy: Mapped[str] = mapped_column(Text)
    recommended_product_ids: Mapped[list] = mapped_column(JSON, default=list)
    rationale: Mapped[dict] = mapped_column(JSON, default=dict)  # agent reasoning audit trail
    behavior_signature: Mapped[str] = mapped_column(String(64), default="")  # dedupe hash
    is_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, index=True
    )

    user: Mapped["User"] = relationship(back_populates="recommendations")
