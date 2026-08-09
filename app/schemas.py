from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---- Auth ----
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


# ---- Products ----
class ProductIn(BaseModel):
    title: str
    description: str
    category: str
    price: float = 0.0
    tags: list[str] = []
    level: str = "all"
    thumbnail_url: str = ""


class ProductOut(ProductIn):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# ---- Events ----
class EventIn(BaseModel):
    type: str
    payload: dict = {}
    timestamp: int | None = None  # epoch millis from client (optional)
    session_id: str | None = None


class EventBatch(BaseModel):
    events: list[EventIn]


# ---- Recommendations ----
class RecommendationOut(BaseModel):
    id: int
    narrative_copy: str
    recommended_product_ids: list[int]
    rationale: dict = {}  # agent reasoning audit trail
    products: list[ProductOut] = []
    updated_at: datetime

    class Config:
        from_attributes = True
