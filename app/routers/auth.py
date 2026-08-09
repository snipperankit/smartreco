from fastapi import APIRouter, Depends, Form, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas import Token, UserCreate, UserOut
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


async def _authenticate(db: AsyncSession, email: str, password: str) -> User:
    res = await db.execute(select(User).where(User.email == email))
    user = res.scalar_one_or_none()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    return user


@router.post("/register", response_model=Token)
async def register(
    payload: UserCreate, response: Response, db: AsyncSession = Depends(get_db)
):
    exists = await db.execute(select(User).where(User.email == payload.email))
    if exists.scalar_one_or_none():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="user",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    token = create_access_token(user.email, user.role)
    _set_cookie(response, token)
    return Token(access_token=token, role=user.role)


@router.post("/login", response_model=Token)
async def login(
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    user = await _authenticate(db, email, password)
    token = create_access_token(user.email, user.role)
    _set_cookie(response, token)
    return Token(access_token=token, role=user.role)


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"ok": True}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.put("/settings/delivery")
async def update_delivery(
    enabled: bool,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """SCHED-06: opt-in / opt-out of proactive daily digest."""
    user.proactive_delivery_enabled = enabled
    await db.commit()
    return {"proactive_delivery_enabled": enabled}


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24,
    )
