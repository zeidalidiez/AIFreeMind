"""Context window trim helpers."""

from src.context import estimate_messages_chars, needs_trim, trim_messages


def _msgs(n_user: int, content_size: int = 10) -> list[dict]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n_user):
        out.append({"role": "user", "content": f"u{i}-" + ("x" * content_size)})
        out.append({"role": "assistant", "content": f"a{i}-" + ("y" * content_size)})
    return out


class TestTrim:
    def test_under_limit_unchanged_length(self):
        m = _msgs(2)
        out = trim_messages(m, max_messages=40, max_chars=80_000)
        assert len(out) == len(m)
        assert out[0]["role"] == "system"

    def test_trims_to_max_messages(self):
        m = _msgs(30)  # 1 + 60 = 61 messages
        assert needs_trim(m, max_messages=20, max_chars=10**9)
        out = trim_messages(m, max_messages=20, max_chars=10**9)
        assert len(out) <= 20
        assert out[0]["role"] == "system"

    def test_trims_to_max_chars(self):
        m = _msgs(5, content_size=5000)
        assert estimate_messages_chars(m) > 1000
        out = trim_messages(m, max_messages=1000, max_chars=3000)
        assert estimate_messages_chars(out) <= 3000 + 500  # soft allowance for system edge
        assert out[0]["role"] == "system"

    def test_drops_tool_cycles(self):
        m = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ]
        out = trim_messages(m, max_messages=4, max_chars=10**9)
        assert len(out) <= 4
        assert out[0]["role"] == "system"
