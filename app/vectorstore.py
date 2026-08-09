"""Vector store half of the dual-write.

Chroma is persisted to disk (no external service needed). Every product write
in the SQL layer is mirrored here via `upsert_product`; deletes mirror via
`delete_product`. Embeddings come from Mesh so retrieval is grounded in the
same model family used for generation.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings

_content_hashes: dict[int, str] = {}
# BM25 index — rebuilt on upsert, maps product_id → tokenized document
_bm25_docs: dict[int, list[str]] = {}
_bm25_df: Counter = Counter()  # document frequency per term
_bm25_total: int = 0
_bm25_avgdl: float = 0.0

_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.mesh_api_key,
    api_base=settings.mesh_base_url,
    model_name=settings.mesh_embed_model,
)
_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"},
    embedding_function=_ef,
)


def _doc_text(title: str, description: str, category: str, tags: list[str]) -> str:
    tag_str = ", ".join(tags or [])
    return f"{title}\nCategory: {category}\nTags: {tag_str}\n{description}"


_SPLIT_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _SPLIT_RE.findall(text.lower())


def _bm25_index_doc(product_id: int, text: str) -> None:
    global _bm25_total, _bm25_avgdl
    old_tokens = _bm25_docs.get(product_id)
    if old_tokens is not None:
        for t in set(old_tokens):
            _bm25_df[t] = max(0, _bm25_df[t] - 1)
        _bm25_total -= 1

    tokens = _tokenize(text)
    _bm25_docs[product_id] = tokens
    for t in set(tokens):
        _bm25_df[t] += 1
    _bm25_total += 1
    _bm25_avgdl = sum(len(d) for d in _bm25_docs.values()) / max(_bm25_total, 1)


def _bm25_remove_doc(product_id: int) -> None:
    global _bm25_total, _bm25_avgdl
    tokens = _bm25_docs.pop(product_id, None)
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


def _bm25_search(query: str, k: int, exclude_ids: list[int] | None = None) -> list[tuple[int, float]]:
    qt = _tokenize(query)
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
    return results[:k]


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
    text_hash = hashlib.sha1(text.encode()).hexdigest()

    need_embed = _content_hashes.get(product_id) != text_hash

    meta = {
        "product_id": product_id,
        "title": title,
        "category": category,
        "price": price,
        "level": level,
    }
    if need_embed:
        _collection.upsert(
            ids=[str(product_id)],
            documents=[text],
            metadatas=[meta],
        )
        _content_hashes[product_id] = text_hash
    else:
        _collection.update(ids=[str(product_id)], metadatas=[meta])
    _bm25_index_doc(product_id, text)


def upsert_many(products: list[dict]) -> None:
    """Bulk path for the seed script."""
    if not products:
        return
    texts = [
        _doc_text(p["title"], p["description"], p["category"], p.get("tags", []))
        for p in products
    ]
    _collection.upsert(
        ids=[str(p["id"]) for p in products],
        documents=texts,
        metadatas=[
            {
                "product_id": p["id"],
                "title": p["title"],
                "category": p["category"],
                "price": p.get("price", 0.0),
                "level": p.get("level", "all"),
            }
            for p in products
        ],
    )
    for p, text in zip(products, texts):
        _bm25_index_doc(p["id"], text)


def delete_product(product_id: int) -> None:
    _collection.delete(ids=[str(product_id)])
    _bm25_remove_doc(product_id)


def search(
    query_text: str,
    k: int = 8,
    exclude_ids: list[int] | None = None,
    category_boost: list[str] | None = None,
    sub_topic_boost: list[str] | None = None,
    level_pref: str | None = None,
) -> list[dict]:
    """Three-stage hybrid search: vector retrieval + BM25 keyword + RRF fusion."""
    where = None
    if exclude_ids:
        where = {"product_id": {"$nin": list(exclude_ids)}}

    candidate_k = max(k * 3, 24)
    res = _collection.query(
        query_texts=[query_text],
        n_results=candidate_k,
        where=where,
    )

    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    docs = res.get("documents", [[]])[0]

    vec_rank: dict[int, int] = {}
    vec_meta: dict[int, dict] = {}
    vec_doc: dict[int, str] = {}
    vec_sim: dict[int, float] = {}
    for rank, (meta, dist, doc) in enumerate(zip(metas, dists, docs)):
        pid = int(meta["product_id"])
        vec_rank[pid] = rank
        vec_meta[pid] = meta
        vec_doc[pid] = doc
        vec_sim[pid] = 1.0 - float(dist)

    bm25_results = _bm25_search(query_text, k=candidate_k, exclude_ids=exclude_ids)
    bm25_rank: dict[int, int] = {pid: rank for rank, (pid, _) in enumerate(bm25_results)}

    rrf_k = 60
    all_pids = set(vec_rank.keys()) | set(bm25_rank.keys())
    rrf_scores: dict[int, float] = {}
    for pid in all_pids:
        score = 0.0
        if pid in vec_rank:
            score += 1.0 / (rrf_k + vec_rank[pid])
        if pid in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[pid])
        rrf_scores[pid] = score

    ranked_pids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:candidate_k]

    hits: list[dict] = []
    for pid in ranked_pids:
        meta = vec_meta.get(pid)
        doc = vec_doc.get(pid, "")
        if meta is None:
            try:
                r = _collection.get(ids=[str(pid)], include=["metadatas", "documents"])
                meta = r["metadatas"][0] if r["metadatas"] else {}
                doc = r["documents"][0] if r["documents"] else ""
            except Exception:
                continue

        base_score = vec_sim.get(pid, 0.3)
        cat_boost = 0.05 if category_boost and meta.get("category") in category_boost else 0.0

        topic_boost = 0.0
        if sub_topic_boost and doc:
            doc_lower = doc.lower()
            matches = sum(1 for topic in sub_topic_boost if topic.lower() in doc_lower)
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
    return _collection.count()


def all_ids() -> set[int]:
    """Product IDs currently present in the vector store (for sync status)."""
    try:
        return {int(i) for i in _collection.get(include=[])["ids"]}
    except Exception:
        return set()
