"""
AIFreeMind LLM Router
Model-agnostic LLM communication via LiteLLM.
Handles primary/fallback routing, streaming, and batch reflection.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import litellm

from .config import Config

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True
try:
    litellm.set_verbose = False
except Exception:
    pass


def _completion_kwargs(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    api_base: str,
    *,
    stream: bool = False,
) -> dict:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if api_base:
        kwargs["api_base"] = api_base
    if stream:
        kwargs["stream"] = True
    return kwargs


def primary_api_base_for(config: Config) -> str:
    """Public helper for tests — which base URL primary uses."""
    return (config.primary_api_base or "").strip()


def generate_response(
    messages: list[dict],
    tools: list[dict],
    config: Config,
) -> Any:
    """
    Send a conversation to the LLM and get a response (non-streaming).

    Tries the primary model first (honoring PRIMARY_API_BASE).
    Falls back if configured.
    """
    try:
        kwargs = _completion_kwargs(
            config.primary_model,
            messages,
            tools,
            config.primary_api_base,
            stream=False,
        )
        return litellm.completion(**kwargs)

    except Exception as primary_err:
        if not config.fallback_model:
            raise Exception(
                f"Primary model ({config.primary_model}) failed: {primary_err}"
            ) from primary_err

        try:
            kwargs = _completion_kwargs(
                config.fallback_model,
                messages,
                tools,
                config.fallback_api_base,
                stream=False,
            )
            return litellm.completion(**kwargs)
        except Exception as fallback_err:
            raise Exception(
                f"Both models failed.\n"
                f"  Primary ({config.primary_model}): {primary_err}\n"
                f"  Fallback ({config.fallback_model}): {fallback_err}"
            ) from fallback_err


def generate_response_stream(
    messages: list[dict],
    tools: list[dict],
    config: Config,
    on_token: Optional[Callable[[str], None]] = None,
) -> Any:
    """
    Stream a completion when STREAM_RESPONSES is enabled; fall back to non-streaming.

    Tool schemas may be present (CLI always offers tools). Text deltas go to
    on_token; tool_call fragments are accumulated into a final message so the
    agent loop still works. On stream failure, falls back to generate_response.
    """
    if not config.stream_responses:
        return generate_response(messages, tools, config)

    try:
        kwargs = _completion_kwargs(
            config.primary_model,
            messages,
            tools if tools else None,
            config.primary_api_base,
            stream=True,
        )
        stream = litellm.completion(**kwargs)
        chunks: list[str] = []
        # tool_call index -> {id, name, arguments}
        tool_acc: dict[int, dict[str, str]] = {}

        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
            except Exception:
                continue
            piece = getattr(delta, "content", None) or ""
            if piece:
                chunks.append(piece)
                if on_token:
                    on_token(piece)
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = getattr(tc, "index", 0) or 0
                slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments or ""

        full = "".join(chunks)
        tool_calls = None
        if tool_acc:
            class _Fn:
                def __init__(self, name: str, arguments: str):
                    self.name = name
                    self.arguments = arguments

            class _Tc:
                def __init__(self, id_: str, name: str, arguments: str):
                    self.id = id_ or "call_0"
                    self.type = "function"
                    self.function = _Fn(name, arguments)

                def model_dump(self):
                    return {
                        "id": self.id,
                        "type": "function",
                        "function": {
                            "name": self.function.name,
                            "arguments": self.function.arguments,
                        },
                    }

            tool_calls = [
                _Tc(v["id"], v["name"], v["arguments"])
                for _, v in sorted(tool_acc.items())
            ]

        class _Msg:
            def __init__(self, content: str, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                self.role = "assistant"

            def model_dump(self):
                d: dict[str, Any] = {"role": "assistant", "content": self.content}
                if self.tool_calls:
                    d["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, "model_dump") else tc
                        for tc in self.tool_calls
                    ]
                return d

        class _Choice:
            def __init__(self, msg):
                self.message = msg

        class _Resp:
            def __init__(self, msg):
                self.choices = [_Choice(msg)]

        content_out: Any
        if full:
            content_out = full
        elif tool_calls:
            content_out = None
        else:
            content_out = ""
        return _Resp(_Msg(content_out, tool_calls))

    except Exception:
        return generate_response(messages, tools, config)


REFLECTION_SYSTEM_PROMPT = """You are a memory consolidation system. Analyze the conversation transcript and extract:

1. "memories": An array of 1-5 memory objects. Each object has:
   - "text": A concise 1-3 sentence summary capturing a fact, preference, decision, or insight.
   - "domain": A short category tag for this memory. Use lowercase single words like:
     "dev", "gaming", "fiction", "personal", "music", "design", "science", "general"
     Pick the most specific domain that fits. Use "general" only if nothing else applies.
   
   Each memory must be self-contained. Skip trivial small talk.

2. "inbox_question": A single curious, specific follow-up question for next session that references something concrete from this conversation.

IMPORTANT: Your entire response must be a single valid JSON object. Do not wrap in markdown code fences. Do not include any text before or after the JSON.

{"memories": [{"text": "example memory", "domain": "dev"}, {"text": "another memory", "domain": "gaming"}], "inbox_question": "example question?"}"""


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output, handling common formatting issues:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace or text
    - BOM characters
    """
    cleaned = text.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract JSON from response", cleaned, 0)


# Public alias for tests
extract_json = _extract_json


def batch_reflect(transcript: str, config: Config) -> dict:
    """
    The "mega-prompt" — analyzes a full session transcript and extracts
    structured memories and a curiosity question for next session.
    """
    if not transcript.strip():
        return {"memories": [], "inbox_question": ""}

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the session transcript to analyze:\n\n{transcript}"},
    ]

    content = ""
    try:
        model = config.reflect_model
        api_base = ""
        if model == config.fallback_model and config.fallback_api_base:
            api_base = config.fallback_api_base
        elif model == config.primary_model and config.primary_api_base:
            api_base = config.primary_api_base

        kwargs = _completion_kwargs(model, messages, None, api_base, stream=False)
        response = litellm.completion(**kwargs)
        raw_content = response.choices[0].message.content
        if not raw_content:
            print("  [Warning] Reflection returned empty response from LLM.")
            return {"memories": [], "inbox_question": ""}
        content = raw_content.strip()

        result = _extract_json(content)

        raw_memories = result.get("memories", [])
        if not isinstance(raw_memories, list):
            raw_memories = [raw_memories]

        memories = []
        for m in raw_memories:
            if not m:
                continue
            if isinstance(m, dict):
                text = str(m.get("text", "")).strip()
                domain = str(m.get("domain", "general")).strip().lower()
                if text:
                    memories.append({"text": text, "domain": domain})
            elif isinstance(m, str):
                memories.append({"text": m.strip(), "domain": "general"})

        inbox = result.get("inbox_question", "")
        if not isinstance(inbox, str):
            inbox = str(inbox)

        return {
            "memories": memories,
            "inbox_question": inbox,
        }

    except json.JSONDecodeError as e:
        print(f"  [Warning] Reflection returned invalid JSON: {e}")
        if content:
            print(f"  Raw response: {content[:300]}")
        return {"memories": [], "inbox_question": ""}

    except Exception as e:
        print(f"  [Warning] Reflection failed: {e}")
        return {"memories": [], "inbox_question": ""}
