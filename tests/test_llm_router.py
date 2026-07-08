"""Reflection JSON extraction + primary api_base wiring (structural)."""

import json

import pytest

from src.config import load_config_from_env_dict
from src.llm_router import _completion_kwargs, extract_json, primary_api_base_for


class TestExtractJson:
    def test_plain_json(self):
        data = extract_json('{"memories": [], "inbox_question": "hi?"}')
        assert data["inbox_question"] == "hi?"

    def test_fenced_json(self):
        raw = """```json
{"memories": [{"text": "x", "domain": "dev"}], "inbox_question": "q?"}
```"""
        data = extract_json(raw)
        assert data["memories"][0]["text"] == "x"
        assert data["inbox_question"] == "q?"

    def test_json_with_preamble(self):
        raw = 'Here you go:\n{"memories": ["a"], "inbox_question": "z?"}\nThanks'
        data = extract_json(raw)
        assert data["inbox_question"] == "z?"

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("not json at all")


class TestApiBase:
    def test_primary_api_base_on_config(self):
        cfg = load_config_from_env_dict({
            "PRIMARY_MODEL": "ollama/llama3",
            "PRIMARY_API_BASE": "http://localhost:11434",
        })
        assert primary_api_base_for(cfg) == "http://localhost:11434"

    def test_api_base_alias(self):
        cfg = load_config_from_env_dict({
            "PRIMARY_MODEL": "ollama/llama3",
            "API_BASE": "http://127.0.0.1:11434",
        })
        assert primary_api_base_for(cfg) == "http://127.0.0.1:11434"

    def test_completion_kwargs_includes_primary_base(self):
        kwargs = _completion_kwargs(
            "ollama/llama3",
            [{"role": "user", "content": "hi"}],
            None,
            "http://localhost:11434",
            stream=False,
        )
        assert kwargs["model"] == "ollama/llama3"
        assert kwargs["api_base"] == "http://localhost:11434"
        assert "stream" not in kwargs or kwargs.get("stream") is not True

    def test_completion_kwargs_stream(self):
        kwargs = _completion_kwargs(
            "openai/gpt-4o-mini",
            [{"role": "user", "content": "hi"}],
            None,
            "",
            stream=True,
        )
        assert kwargs["stream"] is True
        assert "api_base" not in kwargs

    def test_stream_with_tools_still_streams(self, monkeypatch):
        """CLI always passes tool schemas; streaming must still use stream=True."""
        from src import llm_router as lr

        calls = {"stream": 0, "nostream": 0, "had_tools": False}
        tokens: list[str] = []

        class _Delta:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, delta):
                self.choices = [_Choice(delta)]

        def fake_completion(**kwargs):
            if kwargs.get("stream"):
                calls["stream"] += 1
                if kwargs.get("tools"):
                    calls["had_tools"] = True

                def gen():
                    yield _Chunk(_Delta(content="Hel"))
                    yield _Chunk(_Delta(content="lo"))

                return gen()
            calls["nostream"] += 1

            class _Msg:
                content = "ok"
                tool_calls = None

            class _C:
                message = _Msg()

            class _Resp:
                choices = [_C()]

            return _Resp()

        monkeypatch.setattr(lr.litellm, "completion", fake_completion)
        cfg = load_config_from_env_dict({
            "PRIMARY_MODEL": "openai/gpt-4o-mini",
            "STREAM_RESPONSES": "true",
        })
        tools = [{"type": "function", "function": {"name": "read_file"}}]
        resp = lr.generate_response_stream(
            [{"role": "user", "content": "hi"}],
            tools,
            cfg,
            on_token=tokens.append,
        )
        assert calls["stream"] == 1
        assert calls["nostream"] == 0
        assert calls["had_tools"] is True
        assert "".join(tokens) == "Hello"
        assert resp.choices[0].message.content == "Hello"

    def test_cli_path_run_exchange_streams_with_schemas(self, monkeypatch, tmp_path):
        """run_exchange always passes get_tool_schemas(); must still stream tokens."""
        from src import llm_router as lr
        from src.config import load_config_from_env_dict
        from src.main import run_exchange
        from src.memory import MemoryStore
        from src.tools import get_tool_schemas, make_tool_bindings

        stream_calls = []

        class _Delta:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        class _Choice:
            def __init__(self, delta):
                self.delta = delta

        class _Chunk:
            def __init__(self, delta):
                self.choices = [_Choice(delta)]

        def fake_completion(**kwargs):
            stream_calls.append({
                "stream": bool(kwargs.get("stream")),
                "has_tools": bool(kwargs.get("tools")),
                "n_tools": len(kwargs.get("tools") or []),
            })
            if kwargs.get("stream"):
                def gen():
                    yield _Chunk(_Delta(content="streamed "))
                    yield _Chunk(_Delta(content="reply"))
                return gen()

            class _Msg:
                content = "fallback"
                tool_calls = None

            class _C:
                message = _Msg()

            class _Resp:
                choices = [_C()]

            return _Resp()

        monkeypatch.setattr(lr.litellm, "completion", fake_completion)

        cfg = load_config_from_env_dict({
            "PRIMARY_MODEL": "openai/gpt-4o-mini",
            "STREAM_RESPONSES": "true",
            "MEMORY_RECENT_K": "1",
            "MEMORY_RELEVANT_K": "1",
            "MEMORY_RELEVANCE_THRESHOLD": "0.99",
        })
        cfg.db_path = tmp_path / "db"
        cfg.db_path.mkdir()
        cfg.workspace_root = tmp_path
        mem = MemoryStore(cfg, db_path=str(cfg.db_path))
        bindings = make_tool_bindings(tmp_path)

        # Prove schemas are non-empty (same as CLI)
        schemas = get_tool_schemas()
        assert len(schemas) >= 3

        messages: list = []
        text, streamed = run_exchange(
            "hello there",
            messages,
            cfg,
            mem,
            bindings,
        )
        assert any(c["stream"] is True and c["has_tools"] and c["n_tools"] >= 3 for c in stream_calls), stream_calls
        assert text == "streamed reply"
        assert streamed is True
