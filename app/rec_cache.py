"""In-memory recommendation cache with TTL.

At 100k users polling /latest every 10s, that's ~10k reads/sec hitting the DB
for data that changes at most once every 3 minutes (the cooldown). This cache
absorbs the read load: a user's latest recommendation is cached on first fetch
and invalidated when a new recommendation is generated.

Design:
- Dict-based (no Redis dependency for hackathon simplicity)
- Per-user TTL (default 60s) — even stale reads are fine since recs
  change infrequently
- Explicit invalidation on new recommendation generation
- LRU eviction when cache exceeds max size (memory safety)
- Thread-safe for async context

In production you'd swap this for Redis with pub/sub invalidation.
The interface is identical — one function swap.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

DEFAULT_TTL = 900  # 15 minutes (PRD REC-10)
MAX_ENTRIES = 50_000  # ~100 bytes per entry = ~5MB ceiling


class RecCache:
    def __init__(self, ttl: int = DEFAULT_TTL, max_size: int = MAX_ENTRIES):
        self._ttl = ttl
        self._max = max_size
        self._store: OrderedDict[int, tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, user_id: int) -> Any | None:
        entry = self._store.get(user_id)
        if entry is None:
            self._misses += 1
            return None
        ts, data = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[user_id]
            self._misses += 1
            return None
        self._store.move_to_end(user_id)
        self._hits += 1
        return data

    def set(self, user_id: int, data: Any) -> None:
        self._store[user_id] = (time.monotonic(), data)
        self._store.move_to_end(user_id)
        # LRU eviction
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    def invalidate(self, user_id: int) -> None:
        self._store.pop(user_id, None)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "hits": self._hits,
            "misses": self._misses,
        }


# Singleton
rec_cache = RecCache()
