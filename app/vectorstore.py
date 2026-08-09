"""Vector store half of the dual-write.

Uses BM25 keyword search (no external embedding service needed).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.config import settings

_content_hashes: dict[int, str] = {}
_bm25_docs: dict[int, list[str]] = {}
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


def _bm25_index_doc(product_id: int, text: str, meta: dict) -> None:
    global _bm25_total, _bm25_avgdl
    old_tokens = _bm25_docs.get(product_id)
    if old_tokens is not None:
        for t in set(old_tokens):
            _bm25_df[t] = max(0, _bm25_df[t] - 1)
        _bm25_total -= 1

    tokens = _tokenize(text)
    _bm25_docs[product_id] = tokens
    _bm25_meta[product_id] = meta
    for t in set(tokens):
        _bm25_df[t] += 1
    _bm25_total += 1
    _bm25_avgdl = sum(len(d) for d in _bm25_docs.values()) / max(_bm25_total, 1)


def _bm25_remove_doc(product_id: int) -> None:
    global _bm25_total, _bm25_avgdl
    tokens = _bm25_docs.pop(product_id, None)
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
    _bm25_index_doc(product_id, text, meta)


def upsert_many(products: list[dict]) -> None:
    """Bulk path for the seed script."""
    if not products:
        return
    for p in products:
        text = _doc_text(p["title"], p["description"], p["category"], p.get("tags", []))
        meta = {
            "product_id": p["id"],
            "title": p["title"],
            "category": p["category"],
            "price": p.get("price", 0.0),
            "level": p.get("level", "all"),
        }
        _bm25_index_doc(p["id"], text, meta)


def delete_product(product_id: int) -> None:
    _bm25_remove_doc(product_id)


def search(
    query_text: str,
    k: int = 8,
    exclude_ids: list[int] | None = None,
    category_boost: list[str] | None = None,
    sub_topic_boost: list[str] | None = None,
    level_pref: str | None = None,
) -> list[dict]:
    """BM25 keyword search with multi-signal reranking."""
    qt = _tokenize(query_text)
    if not qt:
        return []

    results = []
    for pid, doc_tokens in _bm25_docs.items():
        if exclude_ids and pid in exclude_ids:
            continue
        s = _bm25_score(qt, doc_tokens)
        if s > 0:
            results.append((pid, s))
    results.sort(key=lambda x: x[1], reverse=True)
    candidates = results[: max(k * 3, 24)]

    hits: list[dict] = []
    for pid, base_score in candidates:
        meta = _bm25_meta.get(pid, {})
        doc_text = " ".join(_bm25_docs.get(pid, []))

        cat_boost = 0.05 if category_boost and meta.get("category") in category_boost else 0.0

        topic_boost = 0.0
        if sub_topic_boost and doc_text:
            matches = sum(1 for topic in sub_topic_boost if topic.lower() in doc_text)
            topic_boost = min(0.10, 0.03 * matches)

        level_boost = 0.03 if level_pref and meta.get("level") == level_pref else 0.0
        final_score = base_score + cat_boost + topic_boost + level_boost

        hits.append(
            {
                "product_id": int(meta["product_id"]),
                "title": meta.get("title"),
                "category": meta.get("category"),
                "price": meta.get("price"),
                "level": meta.get("level"),
                "score": round(final_score, 4),
                "score_breakdown": {
                    "base": round(base_score, 3),
                    "category": round(cat_boost, 3),
                    "topic": round(topic_boost, 3),
                    "level": round(level_boost, 3),
                },
            }
        )
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def count() -> int:
    return len(_bm25_docs)


def all_ids() -> set[int]:
    return set(_bm25_docs.keys())
