from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import vectorstore
from app.database import get_db
from app.deps import require_admin
from app.models import Product, User
from app.schemas import ProductIn, ProductOut

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("", response_model=list[ProductOut])
async def list_products(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Product).order_by(Product.id))
    return res.scalars().all()


@router.get("/{product_id}", response_model=ProductOut)
async def get_product(product_id: int, db: AsyncSession = Depends(get_db)):
    p = await db.get(Product, product_id)
    if not p:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return p


@router.post("", response_model=ProductOut, status_code=201)
async def create_product(
    payload: ProductIn,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    product = Product(**payload.model_dump())
    db.add(product)
    await db.commit()
    await db.refresh(product)

    # DUAL-WRITE: mirror into the vector store so it's retrievable semantically.
    vectorstore.upsert_product(
        product_id=product.id,
        title=product.title,
        description=product.description,
        category=product.category,
        price=product.price,
        tags=product.tags,
        level=product.level,
    )
    return product


@router.put("/{product_id}", response_model=ProductOut)
async def update_product(
    product_id: int,
    payload: ProductIn,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    for field, value in payload.model_dump().items():
        setattr(product, field, value)
    await db.commit()
    await db.refresh(product)

    # Keep vector store in sync on every edit.
    vectorstore.upsert_product(
        product_id=product.id,
        title=product.title,
        description=product.description,
        category=product.category,
        price=product.price,
        tags=product.tags,
        level=product.level,
    )
    return product


@router.delete("/{product_id}", status_code=204)
async def delete_product(
    product_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    product = await db.get(Product, product_id)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    await db.delete(product)
    await db.commit()
    # Mirror the delete into the vector store.
    vectorstore.delete_product(product_id)
    return None
