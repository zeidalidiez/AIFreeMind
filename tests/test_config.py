"""Config loading helpers."""

from pathlib import Path

import pytest

from src.config import load_config_from_env_dict
from src.policy import ToolPermissionMode


def test_required_primary():
    with pytest.raises(ValueError):
        load_config_from_env_dict({})


def test_defaults_and_tool_mode(tmp_path: Path):
    cfg = load_config_from_env_dict({
        "PRIMARY_MODEL": "openai/gpt-4o-mini",
        "TOOL_PERMISSION_MODE": "deny",
        "WORKSPACE_ROOT": str(tmp_path),
        "CONTEXT_MAX_MESSAGES": "12",
        "STREAM_RESPONSES": "false",
    })
    assert cfg.primary_model == "openai/gpt-4o-mini"
    assert cfg.tool_permission_mode == ToolPermissionMode.DENY
    assert cfg.workspace_root == tmp_path.resolve()
    assert cfg.context_max_messages == 12
    assert cfg.stream_responses is False
    assert cfg.reflect_model == "openai/gpt-4o-mini"
