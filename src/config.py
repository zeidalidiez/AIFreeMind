"""
AIFreeMind Configuration
Loads all settings from .env and exposes them as a typed Config object.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """All AIFreeMind configuration, loaded from .env"""

    # Models
    primary_model: str = ""
    fallback_model: str = ""
    reflect_model: str = ""
    fallback_api_base: str = ""

    # Memory
    memory_recent_k: int = 25
    memory_relevant_k: int = 15
    memory_relevance_threshold: float = 0.7

    # Checkpointing
    checkpoint_interval: int = 10

    # Paths (derived, not from .env)
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "db")
    checkpoint_path: Path = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    inbox_path: Path = field(default_factory=lambda: PROJECT_ROOT / "inbox_question.txt")


def load_config() -> Config:
    """
    Load configuration from .env file.
    Returns a Config object with all settings.
    Exits with a clear error if required settings are missing.
    """
    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        example_file = PROJECT_ROOT / ".env.example"
        print(f"\n[ERROR] No .env file found at: {env_file}")
        if example_file.exists():
            print(f"  Copy the example and fill in your settings:")
            print(f"  cp .env.example .env")
        else:
            print(f"  Create a .env file with at least PRIMARY_MODEL set.")
        sys.exit(1)

    load_dotenv(env_file)

    config = Config(
        primary_model=os.getenv("PRIMARY_MODEL", "").strip(),
        fallback_model=os.getenv("FALLBACK_MODEL", "").strip(),
        reflect_model=os.getenv("REFLECT_MODEL", "").strip(),
        fallback_api_base=os.getenv("FALLBACK_API_BASE", "").strip(),
        memory_recent_k=int(os.getenv("MEMORY_RECENT_K", "25")),
        memory_relevant_k=int(os.getenv("MEMORY_RELEVANT_K", "15")),
        memory_relevance_threshold=float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.7")),
        checkpoint_interval=int(os.getenv("CHECKPOINT_INTERVAL", "10")),
    )

    # Reflect model defaults to primary if not set
    if not config.reflect_model:
        config.reflect_model = config.primary_model

    # Validate required fields
    if not config.primary_model:
        print("\n[ERROR] PRIMARY_MODEL is not set in .env")
        print("  Example: PRIMARY_MODEL=gemini/gemini-2.0-flash")
        sys.exit(1)

    # Ensure data directories exist
    config.db_path.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)

    # If fallback model uses a custom API base, set it for LiteLLM
    if config.fallback_api_base:
        os.environ["FALLBACK_API_BASE"] = config.fallback_api_base

    return config
