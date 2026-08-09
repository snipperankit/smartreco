"""Vector store half of the dual-write.

Uses BM25 keyword search (no external embedding service needed).
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.mesh import embed, embed_many

log = logging.getLogger("smartreco.vectorstore")

_VECTOR_DIMENSION = 1536
_COLLECTION_NAME = "products_mesh_v2"
_client = chromadb.PersistentClient(
    path=settings.chroma_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name=_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

_content_hashes: dict[int, str] = {}
_bm25_docs: dict[int, list[str]] = {}
_bm25_text: dict[int, str] = {}
_bm25_meta: dict[int, dict] = {}
_bm25_df: Counter = Counter()
_bm25_total: int = 0
_bm25_avgdl: float = 0.0


def _doc_text(title: str, description: str, category: str, tags: list[str]) -> str:
    tag_str = ", ".join(tags or [])
    return f"{title}\nCategory: {category}\nTags: {tag_str}\n{description}"


_SPLIT_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _SPLIT_RE.findall(text.lower())


def _fallback_embedding(text: str) -> list[float]:
    """Feature-hashed vector used when paid Mesh embeddings are unavailable."""
    tokens = _tokenize(text)
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    vector = [0.0] * _VECTOR_DIMENSION
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % _VECTOR_DIMENSION
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _embed_one(text: str) -> list[float]:
    try:
        return embed(text)
    except Exception as exc:
        log.warning("Mesh embedding unavailable; using local fallback (%s)", type(exc).__name__)
        return _fallback_embedding(text)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    try:
        return embed_many(texts)
    except Exception as exc:
        log.warning("Mesh embeddings unavailable; using local fallback (%s)", type(exc).__name__)
        return [_fallback_embedding(text) for text in texts]


def _bm25_index_doc(product_id: int, text: str, meta: dict) -> None:
    global _bm25_total, _bm25_avgdl
    old_tokens = _bm25_docs.get(product_id)
    if old_tokens is not None:
        for t in set(old_tokens):
            _bm25_df[t] = max(0, _bm25_df[t] - 1)
        _bm25_total -= 1

    tokens = _tokenize(text)
    _bm25_docs[product_id] = tokens
    _bm25_text[product_id] = text
    _bm25_meta[product_id] = meta
    for t in set(tokens):
        _bm25_df[t] += 1
    _bm25_total += 1
    _bm25_avgdl = sum(len(d) for d in _bm25_docs.values()) / max(_bm25_total, 1)


def _bm25_remove_doc(product_id: int) -> None:
    global _bm25_total, _bm25_avgdl
    tokens = _bm25_docs.pop(product_id, None)
    _bm25_text.pop(product_id, None)
    _bm25_meta.pop(product_id, None)
    if tokens is not None:
        for t in set(tokens):
            _bm25_df[t] = max(0, _bm25_df[t] - 1)
        _bm25_total -= 1
        _bm25_avgdl = sum(len(d) for d in _bm25_docs.values()) / max(_bm25_total, 1)


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        df = _bm25_df.get(qt, 0)
        if df == 0:
            continue
        idf = math.log((_bm25_total - df + 0.5) / (df + 0.5) + 1.0)
        term_freq = tf.get(qt, 0)
        numerator = term_freq * (k1 + 1)
        denominator = term_freq + k1 * (1 - b + b * dl / max(_bm25_avgdl, 1))
        score += idf * numerator / denominator
    return score


def upsert_product(
    product_id: int,
    title: str,
    description: str,
    category: str,
    price: float,
    tags: list[str],
    level: str = "all",
) -> None:
    text = _doc_text(title, description, category, tags)
    meta = {
        "product_id": product_id,
        "title": title,
        "category": category,
        "price": price,
        "level": level,
    }
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    meta["content_hash"] = content_hash
    _collection.upsert(
        ids=[str(product_id)],
        embeddings=[_embed_one(text)],
        documents=[text],
        metadatas=[meta],
    )
    _content_hashes[product_id] = content_hash
    _bm25_index_doc(product_id, text, meta)


def upsert_many(products: list[dict]) -> None:
    """Bulk path for the seed script."""
    if not products:
        return
    existing = _collection.get(
        ids=[str(product["id"]) for product in products],
        include=["metadatas"],
    )
    for product_id, metadata in zip(
        existing.get("ids", []), existing.get("metadatas", [])
    ):
        content_hash = (metadata or {}).get("content_hash")
        if content_hash:
            _content_hashes[int(product_id)] = content_hash
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    pending_meta: list[dict] = []
    for p in products:
        text = _doc_text(p["title"], p["description"], p["category"], p.get("tags", []))
        meta = {
            "product_id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "price": p.get("price", 0.0),
            "level": p.get("level", "all"),
        }
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta["content_hash"] = content_hash
        if _content_hashes.get(p["id"]) != content_hash:
            pending_ids.append(str(p["id"]))
            pending_texts.append(text)
            pending_meta.append(meta)
        _bm25_index_doc(p["id"], text, meta)
    if pending_ids:
        _collection.upsert(
            ids=pending_ids,
            embeddings=_embed_batch(pending_texts),
            documents=pending_texts,
            metadatas=pending_meta,
        )
        for product_id, metadata in zip(pending_ids, pending_meta):
            _content_hashes[int(product_id)] = metadata["content_hash"]


def delete_product(product_id: int) -> None:
    _collection.delete(ids=[str(product_id)])
    _content_hashes.pop(product_id, None)
    _bm25_remove_doc(product_id)


def search(
    query_text: str,
    k: int = 8,
    exclude_ids: list[int] | None = None,
    category_boost: list[str] | None = None,
    sub_topic_boost: list[str] | None = None,
    level_pref: str | None = None,
) -> list[dict]:
    """Fuse Chroma cosine retrieval with BM25, then apply metadata boosts."""
    query_tokens = _tokenize(query_text)
    if not query_tokens or not _bm25_docs:
        return []

    candidate_count = min(max(k * 3, 24), len(_bm25_docs))
    vector_result = _collection.query(
        query_embeddings=[_embed_one(query_text)],
        n_results=candidate_count,
        include=["distances"],
    )
    vector_ids = [int(value) for value in vector_result.get("ids", [[]])[0]]
    distances = vector_result.get("distances", [[]])[0]
    vector_rank = {product_id: rank for rank, product_id in enumerate(vector_ids)}
    vector_similarity = {
        product_id: max(0.0, min(1.0, 1.0 - float(distance)))
        for product_id, distance in zip(vector_ids, distances)
    }

    bm25_scores = {
        product_id: _bm25_score(query_tokens, tokens)
        for product_id, tokens in _bm25_docs.items()
    }
    bm25_ids = sorted(bm25_scores, key=bm25_scores.get, reverse=True)[:candidate_count]
    bm25_rank = {product_id: rank for rank, product_id in enumerate(bm25_ids)}
    excluded = set(exclude_ids or [])
    candidate_ids = (set(vector_rank) | set(bm25_rank)) - excluded

    reciprocal_rank = {
        product_id: (
            (1.0 / (60 + vector_rank[product_id]) if product_id in vector_rank else 0.0)
            + (1.0 / (60 + bm25_rank[product_id]) if product_id in bm25_rank else 0.0)
        )
        for product_id in candidate_ids
    }

    hits: list[dict] = []
    for product_id in sorted(reciprocal_rank, key=reciprocal_rank.get, reverse=True):
        meta = _bm25_meta.get(product_id, {})
        doc_text = _bm25_text.get(product_id, "").lower()

        cat_boost = 0.05 if category_boost and meta.get("category") in category_boost else 0.0

        topic_boost = 0.0
        if sub_topic_boost and doc_text:
            matches = sum(1 for topic in sub_topic_boost if topic.lower() in doc_text)
            topic_boost = min(0.10, 0.03 * matches)

        level_boost = 0.03 if level_pref and meta.get("level") == level_pref else 0.0
        rrf_score = min(1.0, reciprocal_rank[product_id] * 30.0)
        base_score = 0.7 * vector_similarity.get(product_id, 0.0) + 0.3 * rrf_score
        final_score = max(
            0.0, min(1.0, base_score + cat_boost + topic_boost + level_boost)
        )

        hits.append(
            {
                "product_id": product_id,
                "title": meta.get("title"),
                "category": meta.get("category"),
                "price": meta.get("price"),
                "level": meta.get("level"),
                "score": round(final_score, 4),
                "score_breakdown": {
                    "vector": round(vector_similarity.get(product_id, 0.0), 3),
                    "rrf": round(rrf_score, 3),
                    "category": round(cat_boost, 3),
                    "topic": round(topic_boost, 3),
                    "level": round(level_boost, 3),
                },
            }
        )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def count() -> int:
    return _collection.count()


def all_ids() -> set[int]:
    return {int(value) for value in _collection.get(include=[]).get("ids", [])}
