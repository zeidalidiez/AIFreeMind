"""
AIFreeMind Configuration
Loads all settings from .env and exposes them as a typed Config object.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .policy import ToolPermissionMode, normalize_permission_mode

# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """All AIFreeMind configuration, loaded from .env"""

    # Models
    primary_model: str = ""
    fallback_model: str = ""
    reflect_model: str = ""
    primary_api_base: str = ""
    fallback_api_base: str = ""

    # Memory
    memory_recent_k: int = 25
    memory_relevant_k: int = 15
    memory_relevance_threshold: float = 0.7
    memory_domain_boost: float = 0.05  # similarity boost when domain matches preferred

    # Checkpointing
    checkpoint_interval: int = 10

    # Tools / safety
    tool_permission_mode: ToolPermissionMode = ToolPermissionMode.ASK
    workspace_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    allow_outside_workspace: bool = False

    # Context window
    context_max_messages: int = 40
    context_max_chars: int = 80_000

    # Streaming
    stream_responses: bool = True

    # Paths (derived, not from .env)
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "db")
    checkpoint_path: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    inbox_path: Path = field(default_factory=lambda: PROJECT_ROOT / "inbox_question.txt")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_config(env_file: Optional[Path] = None) -> Config:
    """
    Load configuration from .env file.
    Returns a Config object with all settings.
    Exits with a clear error if required settings are missing.
    """
    path = env_file if env_file is not None else PROJECT_ROOT / ".env"

    if not path.exists():
        example_file = PROJECT_ROOT / ".env.example"
        print(f"\n[ERROR] No .env file found at: {path}")
        if example_file.exists():
            print("  Copy the example and fill in your settings:")
            print("  cp .env.example .env")
        else:
            print("  Create a .env file with at least PRIMARY_MODEL set.")
        sys.exit(1)

    load_dotenv(path, override=False)

    workspace_raw = os.getenv("WORKSPACE_ROOT", "").strip()
    workspace_root = Path(workspace_raw).expanduser().resolve() if workspace_raw else PROJECT_ROOT

    primary_api_base = os.getenv("PRIMARY_API_BASE", "").strip()
    # Backward compat: API_BASE alone can set primary
    if not primary_api_base:
        primary_api_base = os.getenv("API_BASE", "").strip()

    config = Config(
        primary_model=os.getenv("PRIMARY_MODEL", "").strip(),
        fallback_model=os.getenv("FALLBACK_MODEL", "").strip(),
        reflect_model=os.getenv("REFLECT_MODEL", "").strip(),
        primary_api_base=primary_api_base,
        fallback_api_base=os.getenv("FALLBACK_API_BASE", "").strip(),
        memory_recent_k=int(os.getenv("MEMORY_RECENT_K", "25")),
        memory_relevant_k=int(os.getenv("MEMORY_RELEVANT_K", "15")),
        memory_relevance_threshold=float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.7")),
        memory_domain_boost=float(os.getenv("MEMORY_DOMAIN_BOOST", "0.05")),
        checkpoint_interval=int(os.getenv("CHECKPOINT_INTERVAL", "10")),
        tool_permission_mode=normalize_permission_mode(os.getenv("TOOL_PERMISSION_MODE", "ask")),
        workspace_root=workspace_root,
        allow_outside_workspace=_env_bool("TOOL_ALLOW_OUTSIDE_WORKSPACE", False),
        context_max_messages=int(os.getenv("CONTEXT_MAX_MESSAGES", "40")),
        context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "80000")),
        stream_responses=_env_bool("STREAM_RESPONSES", True),
        db_path=PROJECT_ROOT / "db",
        checkpoint_path=PROJECT_ROOT / "checkpoints",
        inbox_path=PROJECT_ROOT / "inbox_question.txt",
    )

    # Reflect model defaults to primary if not set
    if not config.reflect_model:
        config.reflect_model = config.primary_model

    # Validate required fields
    if not config.primary_model:
        print("\n[ERROR] PRIMARY_MODEL is not set in .env")
        print("  Example: PRIMARY_MODEL=openai/<current-model-id>")
        print("  (any current LiteLLM-compatible model string)")
        sys.exit(1)

    # Ensure data directories exist
    config.db_path.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)

    return config


def load_config_from_env_dict(env: dict) -> Config:
    """
    Build Config from an explicit env mapping (for tests). Does not sys.exit.
    """
    workspace_raw = str(env.get("WORKSPACE_ROOT", "") or "").strip()
    workspace_root = Path(workspace_raw).expanduser().resolve() if workspace_raw else PROJECT_ROOT

    primary_api_base = str(env.get("PRIMARY_API_BASE", "") or "").strip()
    if not primary_api_base:
        primary_api_base = str(env.get("API_BASE", "") or "").strip()

    primary = str(env.get("PRIMARY_MODEL", "") or "").strip()
    if not primary:
        raise ValueError("PRIMARY_MODEL is required")

    reflect = str(env.get("REFLECT_MODEL", "") or "").strip() or primary

    def _b(key: str, default: bool = False) -> bool:
        raw = env.get(key)
        if raw is None or str(raw).strip() == "":
            return default
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    return Config(
        primary_model=primary,
        fallback_model=str(env.get("FALLBACK_MODEL", "") or "").strip(),
        reflect_model=reflect,
        primary_api_base=primary_api_base,
        fallback_api_base=str(env.get("FALLBACK_API_BASE", "") or "").strip(),
        memory_recent_k=int(env.get("MEMORY_RECENT_K", 25)),
        memory_relevant_k=int(env.get("MEMORY_RELEVANT_K", 15)),
        memory_relevance_threshold=float(env.get("MEMORY_RELEVANCE_THRESHOLD", 0.7)),
        memory_domain_boost=float(env.get("MEMORY_DOMAIN_BOOST", 0.05)),
        checkpoint_interval=int(env.get("CHECKPOINT_INTERVAL", 10)),
        tool_permission_mode=normalize_permission_mode(str(env.get("TOOL_PERMISSION_MODE", "ask"))),
        workspace_root=workspace_root,
        allow_outside_workspace=_b("TOOL_ALLOW_OUTSIDE_WORKSPACE", False),
        context_max_messages=int(env.get("CONTEXT_MAX_MESSAGES", 40)),
        context_max_chars=int(env.get("CONTEXT_MAX_CHARS", 80_000)),
        stream_responses=_b("STREAM_RESPONSES", True),
    )
