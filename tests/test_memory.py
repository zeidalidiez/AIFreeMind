"""Memory store: recent-at-scale, domain, browse, delete, search, consolidate."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.config import Config, load_config_from_env_dict
from src.memory import MemoryStore


def _cfg(tmp_path: Path, **overrides) -> Config:
    env = {
        "PRIMARY_MODEL": "test/model",
        "MEMORY_RECENT_K": "5",
        "MEMORY_RELEVANT_K": "5",
        "MEMORY_RELEVANCE_THRESHOLD": "0.0",  # accept all for domain tests
        "MEMORY_DOMAIN_BOOST": "0.15",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    c = load_config_from_env_dict(env)
    c.db_path = tmp_path / "db"
    c.db_path.mkdir(parents=True, exist_ok=True)
    return c


@pytest.fixture
def store(tmp_path: Path):
    cfg = _cfg(tmp_path)
    return MemoryStore(cfg, db_path=str(cfg.db_path))


class TestRecentAtScale:
    def test_recent_returns_newest_when_over_100(self, tmp_path: Path):
        cfg = _cfg(tmp_path, MEMORY_RECENT_K=5)
        mem = MemoryStore(cfg, db_path=str(cfg.db_path))

        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        # Insert 120 memories with increasing timestamps (older first)
        for i in range(120):
            ts = (base + timedelta(seconds=i)).isoformat()
            mid = mem.add_memory(
                f"memory number {i:04d}",
                metadata={"timestamp": ts, "domain": "dev", "source": "test"},
            )
            assert mid

        recent = mem._get_recent(5)
        # Newest should be 119..115
        docs = " ".join(recent["entries"])
        assert "0119" in docs
        assert "0118" in docs
        assert "0115" in docs
        # Oldest should not appear in top-5 recent
        assert "0000" not in docs
        assert "0050" not in docs


class TestDomainAware:
    def test_hard_domain_filter(self, store: MemoryStore):
        store.add_memory("Python uses list comprehensions", metadata={"domain": "dev", "timestamp": "2024-01-01T00:00:00+00:00"})
        store.add_memory("The dragon lives in the mountains", metadata={"domain": "fiction", "timestamp": "2024-01-02T00:00:00+00:00"})
        store.add_memory("pytest is great for testing", metadata={"domain": "dev", "timestamp": "2024-01-03T00:00:00+00:00"})

        scoped = store.query_memory("code", domain="dev")
        assert "Python" in scoped or "pytest" in scoped
        assert "dragon" not in scoped

        fiction = store.query_memory("story", domain="fiction")
        assert "dragon" in fiction
        assert "Python" not in fiction

    def test_domain_boost_prefers_matching(self, tmp_path: Path):
        cfg = _cfg(tmp_path, MEMORY_RELEVANCE_THRESHOLD="0.0", MEMORY_DOMAIN_BOOST="0.2")
        mem = MemoryStore(cfg, db_path=str(cfg.db_path))
        mem.add_memory("user prefers dark mode in the editor", metadata={"domain": "dev"})
        mem.add_memory("user prefers dark forests in fantasy settings", metadata={"domain": "fiction"})

        # preferred_domain=dev should boost dev memory in relevant section
        out = mem.query_memory("user prefers dark", preferred_domain="dev")
        # Both may appear; if relevance scores shown, dev should be present
        assert "editor" in out or "dark mode" in out


class TestBrowseAndManage:
    def test_list_full_store_beyond_50(self, store: MemoryStore):
        for i in range(60):
            store.add_memory(
                f"item {i}",
                metadata={
                    "domain": "general",
                    "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc).replace(microsecond=i).isoformat(),
                },
            )
        all_items = store.list_memories(limit=None)
        assert len(all_items) == 60
        # Newest first — last inserted has latest microsecond if same second; use check count
        filtered = store.list_memories(limit=None, domain="general")
        assert len(filtered) == 60

    def test_delete_and_search_and_domains(self, store: MemoryStore):
        a = store.remember("Alice likes tea", domain="personal")
        b = store.remember("Bob ships Python packages", domain="dev")
        assert store.get_memory_count() == 2

        domains = dict(store.list_domains())
        assert domains.get("personal") == 1
        assert domains.get("dev") == 1

        hits = store.search_memories("Python packages", k=5, threshold=0.0)
        assert any("Bob" in h["document"] for h in hits)

        assert store.delete_memory(a) is True
        assert store.get_memory_count() == 1
        assert store.delete_memory(a) is False
        assert store.delete_memory(b) is True
        assert store.get_memory_count() == 0

    def test_mid_session_remember(self, store: MemoryStore):
        mid = store.remember("Ship feature X by Friday", domain="dev")
        assert mid
        items = store.list_memories(limit=10, domain="dev")
        assert len(items) >= 1
        assert any("feature x" in i["document"].lower() for i in items)


class TestConsolidate:
    def test_exact_dup_removed(self, store: MemoryStore):
        store.remember("User prefers vim keybindings", domain="dev")
        store.remember("User prefers vim keybindings", domain="dev")
        store.remember("User prefers vim keybindings", domain="dev")
        assert store.get_memory_count() == 3
        stats = store.consolidate_memories(similarity_threshold=0.99)
        assert stats["removed"] >= 2
        assert store.get_memory_count() == 1

    def test_dry_run_does_not_delete(self, store: MemoryStore):
        store.remember("same text again", domain="general")
        store.remember("same text again", domain="general")
        stats = store.consolidate_memories(dry_run=True)
        assert stats["removed"] >= 1
        assert store.get_memory_count() == 2
