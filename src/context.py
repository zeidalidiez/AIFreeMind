"""
Conversation context window management — pure helpers (no LLM I/O).

Keeps the message list bounded so long sessions do not grow without limit.
"""

from __future__ import annotations

from typing import Any


def _message_role(msg: dict | Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "") or "")
    return str(getattr(msg, "role", "") or "")


def _approx_chars(msg: dict | Any) -> int:
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if content is None:
            content = ""
        # tool calls / name blobs
        extra = ""
        if "tool_calls" in msg and msg["tool_calls"]:
            extra = str(msg["tool_calls"])
        return len(str(content)) + len(extra)
    content = getattr(msg, "content", "") or ""
    return len(str(content))


def estimate_messages_chars(messages: list) -> int:
    return sum(_approx_chars(m) for m in messages)


def trim_messages(
    messages: list,
    *,
    max_messages: int = 40,
    max_chars: int = 80_000,
    keep_system: bool = True,
) -> list:
    """
    Bound conversation history.

    Strategy:
      1. Always keep the first system message (if present and keep_system).
      2. Drop oldest non-system messages until under max_messages and max_chars.
      3. Prefer dropping complete tool-call cycles from the front of the tail.

    Returns a new list (does not mutate input).
    """
    if not messages:
        return []

    msgs = list(messages)
    system: list = []
    body: list = []

    if keep_system and _message_role(msgs[0]) == "system":
        system = [msgs[0]]
        body = msgs[1:]
    else:
        body = msgs

    def within_limits(candidate_body: list) -> bool:
        total = system + candidate_body
        if len(total) > max_messages:
            return False
        if estimate_messages_chars(total) > max_chars:
            return False
        return True

    # Drop from the front of body until within limits
    while body and not within_limits(body):
        # If next is assistant with tool_calls, drop until past matching tool results
        first = body[0]
        role = _message_role(first)
        if role == "assistant" and (
            (isinstance(first, dict) and first.get("tool_calls"))
            or getattr(first, "tool_calls", None)
        ):
            body = body[1:]
            while body and _message_role(body[0]) == "tool":
                body = body[1:]
        else:
            body = body[1:]

    # If still over (huge system prompt), truncate system content as last resort
    result = system + body
    if system and estimate_messages_chars(result) > max_chars:
        sys_msg = dict(system[0]) if isinstance(system[0], dict) else {"role": "system", "content": str(getattr(system[0], "content", ""))}
        content = str(sys_msg.get("content", "") or "")
        budget = max(2000, max_chars // 4)
        if len(content) > budget:
            sys_msg["content"] = content[:budget] + "\n... [system context truncated]"
        result = [sys_msg] + body

    return result


def needs_trim(
    messages: list,
    *,
    max_messages: int = 40,
    max_chars: int = 80_000,
) -> bool:
    if len(messages) > max_messages:
        return True
    return estimate_messages_chars(messages) > max_chars
