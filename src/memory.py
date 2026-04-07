"""
AIFreeMind Memory Module
Persistent semantic memory using ChromaDB with local embeddings.

Dual-retrieval strategy:
  1. Recent memories — the N most recent, regardless of topic (short-term context)
  2. Relevant memories — semantically similar to the current query (long-term recall)
  Results are merged and deduplicated before injection into the LLM prompt.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings

from .config import Config


class MemoryStore:
    """
    Persistent vector memory backed by ChromaDB.
    
    Stores "memories" — distilled facts, preferences, and insights —
    as embedded documents for fast semantic retrieval.
    
    Uses ChromaDB's default embedding model (all-MiniLM-L6-v2),
    which runs locally with no API calls.
    """

    COLLECTION_NAME = "general_memory"

    def __init__(self, config: Config):
        self._client = chromadb.PersistentClient(
            path=str(config.db_path),
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

        self._collection.add(
            ids=[memory_id],
            documents=[text],
            metadatas=[meta],
        )
        return memory_id

    def add_memories(self, memories: list[str], source: str = "reflection") -> list[str]:
        """
        Batch-add memories from reflection output.
        Returns list of generated IDs.
        """
        if not memories:
            return []

        ids = [str(uuid.uuid4()) for _ in memories]
        timestamp = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {"timestamp": timestamp, "source": source}
            for _ in memories
        ]

        self._collection.add(
            ids=ids,
            documents=memories,
            metadatas=metadatas,
        )
        return ids

    # ── Dual Retrieval ─────────────────────────────────────

    def query_memory(self, prompt: str) -> str:
        """
        Dual-retrieval: combines recent memories + semantically relevant memories.
        
        1. Recent: the N most recent memories (short-term working context)
        2. Relevant: semantically similar to the prompt, above threshold (long-term recall)
        
        Results are merged, deduplicated (relevant matches that are already
        in recent don't repeat), and formatted for LLM context injection.
        """
        if self._collection.count() == 0:
            return ""

        recent_k = self._config.memory_recent_k
        relevant_k = self._config.memory_relevant_k
        threshold = self._config.memory_relevance_threshold

        recent_section = self._get_recent(recent_k)
        relevant_section = self._get_relevant(prompt, relevant_k, threshold, exclude_ids=recent_section["ids"])

        parts = []

        if recent_section["entries"]:
            parts.append(f"=== Recent Context ({len(recent_section['entries'])} memories) ===")
            parts.extend(recent_section["entries"])

        if relevant_section["entries"]:
            parts.append(f"\n=== Relevant to This Query ({len(relevant_section['entries'])} found) ===")
            parts.extend(relevant_section["entries"])

        return "\n".join(parts) if parts else ""

    def _get_recent(self, k: int) -> dict:
        """
        Retrieve the K most recent memories by timestamp.
        Returns dict with 'ids' (set) and 'entries' (formatted strings).
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": set(), "entries": []}

        # Fetch all and sort by timestamp (ChromaDB doesn't support ORDER BY)
        fetch_limit = min(count, max(k * 2, 100))  # fetch extra to ensure we get enough
        results = self._collection.get(
            limit=fetch_limit,
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return {"ids": set(), "entries": []}

        # Build list and sort by timestamp descending
        items = []
        for i in range(len(results["ids"])):
            items.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        items.sort(
            key=lambda x: x["metadata"].get("timestamp", ""),
            reverse=True,
        )

        # Take top K
        top = items[:k]

        ids = {item["id"] for item in top}
        entries = []
        for item in top:
            ts = item["metadata"].get("timestamp", "unknown")
            date_str = ts[:10] if len(ts) >= 10 else ts
            entries.append(f"- [{date_str}] {item['document']}")

        return {"ids": ids, "entries": entries}

    def _get_relevant(self, prompt: str, k: int, threshold: float, exclude_ids: set) -> dict:
        """
        Retrieve semantically relevant memories above the threshold,
        excluding any IDs already present in the recent set (dedup).
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": set(), "entries": []}

        # Fetch more than k to account for dedup filtering
        fetch_k = min(count, k + len(exclude_ids) + 5)

        results = self._collection.query(
            query_texts=[prompt],
            n_results=fetch_k,
            include=["documents", "distances", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            return {"ids": set(), "entries": []}

        ids = set()
        entries = []

        for doc_id, doc, dist, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # Skip if already in recent set
            if doc_id in exclude_ids:
                continue

            similarity = 1.0 - dist
            if similarity >= threshold:
                ts = meta.get("timestamp", "unknown")
                date_str = ts[:10] if len(ts) >= 10 else ts
                ids.add(doc_id)
                entries.append(f"- [{date_str}] {doc} (relevance: {similarity:.2f})")

            # Stop once we have enough
            if len(entries) >= k:
                break

        return {"ids": ids, "entries": entries}

    # ── Utilities ──────────────────────────────────────────

    def get_memory_count(self) -> int:
        """How many memories are stored."""
        return self._collection.count()

    def get_all_memories(self, limit: int = 50) -> list[dict]:
        """
        Retrieve stored memories for debugging/inspection.
        Returns list of dicts with 'id', 'document', 'metadata'.
        """
        if self._collection.count() == 0:
            return []

        actual_limit = min(limit, self._collection.count())
        results = self._collection.peek(limit=actual_limit)

        memories = []
        for i in range(len(results["ids"])):
            memories.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return memories
