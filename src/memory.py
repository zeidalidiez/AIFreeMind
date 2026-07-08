"""
AIFreeMind Memory Module
Persistent semantic memory using ChromaDB with local embeddings.

Dual-retrieval strategy:
  1. Recent memories — the N most recent, regardless of topic (short-term context)
  2. Relevant memories — semantically similar to the current query (long-term recall)
     with optional domain boost/scoping
  Results are merged and deduplicated before injection into the LLM prompt.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings

from .config import Config


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class MemoryStore:
    """
    Persistent vector memory backed by ChromaDB.

    Stores "memories" — distilled facts, preferences, and insights —
    as embedded documents for fast semantic retrieval.

    Uses ChromaDB's default embedding model (all-MiniLM-L6-v2),
    which runs locally with no API calls.
    """

    COLLECTION_NAME = "general_memory"

    def __init__(self, config: Config, db_path: Optional[str] = None):
        path = str(db_path) if db_path is not None else str(config.db_path)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._config = config

    def add_memory(self, text: str, metadata: Optional[dict] = None) -> str:
        """
        Store a single memory with timestamp metadata.
        Returns the generated memory ID.
        """
        memory_id = str(uuid.uuid4())
        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "session",
        }
        if metadata:
            meta.update(metadata)
        # Chroma metadata values must be str/int/float/bool
        clean_meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}

        self._collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[clean_meta],
        )
        return memory_id

    def add_memories(self, memories: list, source: str = "reflection") -> list[str]:
        """
        Batch-add memories from reflection output.
        Accepts either:
          - list of strings (legacy format)
          - list of dicts with 'text' and 'domain' keys (tagged format)
        Returns list of generated IDs.
        """
        if not memories:
            return []

        ids = []
        documents = []
        metadatas = []
        timestamp = datetime.now(timezone.utc).isoformat()

        for m in memories:
            if isinstance(m, dict):
                text = m.get("text", "")
                domain = m.get("domain", "general")
            else:
                text = str(m)
                domain = "general"

            if not text.strip():
                continue

            ids.append(str(uuid.uuid4()))
            documents.append(text)
            metadatas.append({
                "timestamp": timestamp,
                "source": source,
                "domain": str(domain).strip().lower() or "general",
            })

        if not ids:
            return []

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    def remember(self, text: str, domain: str = "general", source: str = "explicit") -> str:
        """Mid-session explicit remember — store one memory immediately."""
        return self.add_memory(
            text.strip(),
            metadata={"domain": domain.strip().lower() or "general", "source": source},
        )

    # ── Dual Retrieval ─────────────────────────────────────

    def query_memory(
        self,
        prompt: str,
        domain: Optional[str] = None,
        preferred_domain: Optional[str] = None,
    ) -> str:
        """
        Dual-retrieval: combines recent memories + semantically relevant memories.

        If `domain` is set, only memories with that domain are considered (hard filter).
        If `preferred_domain` is set (soft), matching domain gets a similarity boost.
        """
        if self._collection.count() == 0:
            return ""

        recent_k = self._config.memory_recent_k
        relevant_k = self._config.memory_relevant_k
        threshold = self._config.memory_relevance_threshold

        recent_section = self._get_recent(recent_k, domain=domain)
        relevant_section = self._get_relevant(
            prompt,
            relevant_k,
            threshold,
            exclude_ids=recent_section["ids"],
            domain=domain,
            preferred_domain=preferred_domain,
        )

        parts = []

        if recent_section["entries"]:
            parts.append(f"=== Recent Context ({len(recent_section['entries'])} memories) ===")
            parts.extend(recent_section["entries"])

        if relevant_section["entries"]:
            parts.append(f"\n=== Relevant to This Query ({len(relevant_section['entries'])} found) ===")
            parts.extend(relevant_section["entries"])

        return "\n".join(parts) if parts else ""

    def _all_items(self) -> list[dict]:
        """Fetch entire collection as list of {id, document, metadata}."""
        count = self._collection.count()
        if count == 0:
            return []

        items: list[dict] = []
        # Fetch full collection then sort by timestamp (Chroma has no ORDER BY)
        results = self._collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return []
        for i in range(len(results["ids"])):
            items.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i] or {},
            })
        # Sort newest first by timestamp
        items.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True,
        )
        return items

    def _get_recent(self, k: int, domain: Optional[str] = None) -> dict:
        """
        Retrieve the K most recent memories by timestamp (true newest, full scan).
        """
        items = self._all_items()
        if domain:
            domain = domain.strip().lower()
            items = [it for it in items if (it["metadata"].get("domain") or "general") == domain]

        top = items[:k]
        ids = {item["id"] for item in top}
        entries = []
        for item in top:
            ts = item["metadata"].get("timestamp", "unknown")
            date_str = ts[:10] if len(ts) >= 10 else ts
            d = item["metadata"].get("domain", "")
            domain_tag = f" [{d}]" if d else ""
            entries.append(f"- [{date_str}]{domain_tag} {item['document']}")
        return {"ids": ids, "entries": entries}

    def _get_relevant(
        self,
        prompt: str,
        k: int,
        threshold: float,
        exclude_ids: set,
        domain: Optional[str] = None,
        preferred_domain: Optional[str] = None,
    ) -> dict:
        """
        Semantically relevant memories above threshold, with optional domain filter/boost.
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": set(), "entries": []}

        fetch_k = min(count, max(k + len(exclude_ids) + 20, k * 3))
        where = None
        if domain:
            where = {"domain": domain.strip().lower()}

        query_kwargs = {
            "query_texts": [prompt],
            "n_results": fetch_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = self._collection.query(**query_kwargs)
        except Exception:
            # If where filter fails (empty/no match), return empty
            return {"ids": set(), "entries": []}

        if not results["documents"] or not results["documents"][0]:
            return {"ids": set(), "entries": []}

        boost = getattr(self._config, "memory_domain_boost", 0.05) or 0.0
        preferred = (preferred_domain or "").strip().lower() or None

        scored: list[tuple[float, str, str, dict]] = []
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            if doc_id in exclude_ids:
                continue
            meta = meta or {}
            similarity = 1.0 - float(dist)
            d = (meta.get("domain") or "general").strip().lower()
            if preferred and d == preferred:
                similarity = min(1.0, similarity + boost)
            if similarity >= threshold:
                scored.append((similarity, doc_id, doc, meta))

        scored.sort(key=lambda x: x[0], reverse=True)

        ids = set()
        entries = []
        for similarity, doc_id, doc, meta in scored[:k]:
            ts = meta.get("timestamp", "unknown")
            date_str = ts[:10] if len(ts) >= 10 else ts
            d = meta.get("domain", "")
            domain_tag = f" [{d}]" if d else ""
            ids.add(doc_id)
            entries.append(f"- [{date_str}]{domain_tag} {doc} (relevance: {similarity:.2f})")

        return {"ids": ids, "entries": entries}

    # ── Browse / search / delete ───────────────────────────

    def get_memory_count(self) -> int:
        return self._collection.count()

    def list_memories(
        self,
        limit: Optional[int] = None,
        domain: Optional[str] = None,
        offset: int = 0,
    ) -> list[dict]:
        """
        Full-store ordered browse (newest first). Not limited to peek(50).
        """
        items = self._all_items()
        if domain:
            domain = domain.strip().lower()
            items = [it for it in items if (it["metadata"].get("domain") or "general") == domain]
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def get_all_memories(self, limit: int = 50) -> list[dict]:
        """Backward-compatible alias — ordered newest first."""
        return self.list_memories(limit=limit)

    def list_domains(self) -> list[tuple[str, int]]:
        """Return (domain, count) pairs sorted by count desc then name."""
        items = self._all_items()
        counts: dict[str, int] = {}
        for it in items:
            d = (it["metadata"].get("domain") or "general").strip().lower()
            counts[d] = counts.get(d, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    def search_memories(
        self,
        query: str,
        *,
        k: int = 10,
        domain: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Semantic search returning structured hits with scores."""
        count = self._collection.count()
        if count == 0 or not query.strip():
            return []

        fetch_k = min(count, max(k * 3, k))
        query_kwargs = {
            "query_texts": [query],
            "n_results": fetch_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if domain:
            query_kwargs["where"] = {"domain": domain.strip().lower()}

        try:
            results = self._collection.query(**query_kwargs)
        except Exception:
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        hits = []
        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            similarity = 1.0 - float(dist)
            if similarity < threshold:
                continue
            hits.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "similarity": similarity,
            })
            if len(hits) >= k:
                break
        return hits

    def delete_memory(self, memory_id: str) -> bool:
        """Delete one memory by ID. Returns True if it existed."""
        existing = self._collection.get(ids=[memory_id], include=[])
        if not existing["ids"]:
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def delete_memories(self, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        existing = self._collection.get(ids=memory_ids, include=[])
        found = existing["ids"] or []
        if not found:
            return 0
        self._collection.delete(ids=found)
        return len(found)

    def consolidate_memories(
        self,
        *,
        similarity_threshold: float = 0.92,
        dry_run: bool = False,
    ) -> dict:
        """
        Merge/remove near-duplicate memories.

        For each memory (newest first), if an older memory is nearly identical
        (normalized text equal OR semantic similarity >= threshold), keep the
        newer one and delete the older duplicate.

        Returns stats: {kept, removed, pairs}.
        """
        items = self._all_items()  # newest first
        if len(items) < 2:
            return {"kept": len(items), "removed": 0, "pairs": []}

        to_remove: set[str] = set()
        pairs: list[dict] = []

        # Exact normalized text groups
        by_text: dict[str, list[dict]] = {}
        for it in items:
            key = _normalize_text(it["document"])
            by_text.setdefault(key, []).append(it)

        for key, group in by_text.items():
            if len(group) < 2 or not key:
                continue
            # keep newest (first); remove rest
            keeper = group[0]
            for dup in group[1:]:
                if dup["id"] in to_remove:
                    continue
                to_remove.add(dup["id"])
                pairs.append({
                    "kept_id": keeper["id"],
                    "removed_id": dup["id"],
                    "reason": "exact_text",
                    "text": keeper["document"][:80],
                })

        remaining = [it for it in items if it["id"] not in to_remove]
        # Semantic near-dup: query each remaining against store
        for it in remaining:
            if it["id"] in to_remove:
                continue
            try:
                results = self._collection.query(
                    query_texts=[it["document"]],
                    n_results=min(10, self._collection.count()),
                    include=["documents", "distances", "metadatas"],
                )
            except Exception:
                continue
            if not results["ids"] or not results["ids"][0]:
                continue
            for other_id, dist in zip(results["ids"][0], results["distances"][0]):
                if other_id == it["id"] or other_id in to_remove:
                    continue
                similarity = 1.0 - float(dist)
                if similarity >= similarity_threshold:
                    # Prefer keeping the newer of the two
                    other_meta = None
                    other_doc = None
                    for r in remaining:
                        if r["id"] == other_id:
                            other_meta = r["metadata"]
                            other_doc = r["document"]
                            break
                    if other_meta is None:
                        continue
                    it_ts = it["metadata"].get("timestamp", "")
                    other_ts = other_meta.get("timestamp", "")
                    if other_ts > it_ts:
                        # other is newer — remove it
                        remove_id, keep_id = it["id"], other_id
                    else:
                        remove_id, keep_id = other_id, it["id"]
                    if remove_id not in to_remove:
                        to_remove.add(remove_id)
                        pairs.append({
                            "kept_id": keep_id,
                            "removed_id": remove_id,
                            "reason": f"semantic:{similarity:.3f}",
                            "text": (it["document"] if keep_id == it["id"] else other_doc or "")[:80],
                        })

        removed = 0
        if to_remove and not dry_run:
            removed = self.delete_memories(list(to_remove))
        elif to_remove:
            removed = len(to_remove)

        kept = self.get_memory_count() if not dry_run else (len(items) - removed)
        return {"kept": kept, "removed": removed, "pairs": pairs}
