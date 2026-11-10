"""
common/cache.py
---------------
Lightweight, zero-dependency, thread- and async-safe SQLite disk cache for
LLM API completions and token log-probabilities.

Key design properties:
1. Trial-aware deterministic hashing: Includes `(seed, condition, trial_idx)` in
   the SHA-256 key so repeated sessions `1..N` within a run remain independent
   draws, while re-running with the same seed or scaling `N -> N + k` reuses
   sessions `1..N` at zero API cost and <1ms latency.
2. Short-lived connections with SQLite WAL mode: Avoids open file-handle locks
   on Windows during test teardown and concurrent `asyncio` tasks.
"""

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ResponseCache:
    """Persistent SQLite key-value store for serialized OpenRouter API responses."""

    def __init__(self, db_path: str = ".cache/openrouter_responses.sqlite", enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.prompt_tokens_saved = 0
        self.completion_tokens_saved = 0
        self.live_prompt_tokens = 0
        self.live_completion_tokens = 0

        if self.enabled and self.db_path:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_cache (
                        cache_key TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        prompt_tokens INTEGER DEFAULT 0,
                        completion_tokens INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        """Create a canonical SHA-256 hex digest from request parameters."""
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response dictionary if present."""
        if not self.enabled:
            return None
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT response_json, prompt_tokens, completion_tokens FROM api_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
            if row is not None:
                self.hits += 1
                self.prompt_tokens_saved += int(row[1] or 0)
                self.completion_tokens_saved += int(row[2] or 0)
                return json.loads(row[0])
            self.misses += 1
            return None

    def set(
        self,
        key: str,
        model: str,
        response_data: Dict[str, Any],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Persist a serialized API response and record live token usage."""
        with self._lock:
            self.live_prompt_tokens += int(prompt_tokens or 0)
            self.live_completion_tokens += int(completion_tokens or 0)
            if not self.enabled:
                return
            now = datetime.now(timezone.utc).isoformat()
            serialized = json.dumps(response_data, ensure_ascii=False)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_cache
                    (cache_key, model, response_json, prompt_tokens, completion_tokens, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, model, serialized, int(prompt_tokens or 0), int(completion_tokens or 0), now),
                )
                conn.commit()

    def clear(self) -> int:
        """Delete all cached entries and return number of rows removed."""
        if not self.enabled or not os.path.exists(self.db_path):
            return 0
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM api_cache")
                conn.commit()
                return int(cur.rowcount or 0)

    def count_entries(self) -> int:
        """Return total number of stored cache entries."""
        if not self.enabled or not os.path.exists(self.db_path):
            return 0
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()
                return int(row[0]) if row else 0

    def stats(self) -> Dict[str, Any]:
        """Return telemetry summary of cache hits, misses, and saved tokens."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "enabled": self.enabled,
            "db_path": self.db_path,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "prompt_tokens_saved": self.prompt_tokens_saved,
            "completion_tokens_saved": self.completion_tokens_saved,
            "total_tokens_saved": self.prompt_tokens_saved + self.completion_tokens_saved,
            "live_prompt_tokens": self.live_prompt_tokens,
            "live_completion_tokens": self.live_completion_tokens,
            "stored_entries": self.count_entries(),
        }
