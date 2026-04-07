"""
AIFreeMind LLM Router
Model-agnostic LLM communication via LiteLLM.
Handles primary/fallback routing and batch reflection.
"""

import json
import os
from typing import Optional

import litellm

from .config import Config

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True
litellm.set_verbose = False


def generate_response(
    messages: list[dict],
    tools: list[dict],
    config: Config,
) -> dict:
    """
    Send a conversation to the LLM and get a response.
    
    Tries the primary model first. If it fails, falls back to the
    fallback model (if configured). Returns the raw LiteLLM response.
    
    Args:
        messages: OpenAI-format message list (role + content)
        tools: OpenAI-format tool schemas
        config: App configuration with model strings
    
    Returns:
        The LiteLLM response object (has .choices[0].message)
    
    Raises:
        Exception: If both primary and fallback models fail
    """
    # Try primary model
    try:
        kwargs = {
            "model": config.primary_model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = litellm.completion(**kwargs)
        return response

    except Exception as primary_err:
        # If no fallback configured, raise the primary error
        if not config.fallback_model:
            raise Exception(
                f"Primary model ({config.primary_model}) failed: {primary_err}"
            ) from primary_err

        # Try fallback
        try:
            kwargs = {
                "model": config.fallback_model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if config.fallback_api_base:
                kwargs["api_base"] = config.fallback_api_base

            response = litellm.completion(**kwargs)
            return response

        except Exception as fallback_err:
            raise Exception(
                f"Both models failed.\n"
                f"  Primary ({config.primary_model}): {primary_err}\n"
                f"  Fallback ({config.fallback_model}): {fallback_err}"
            ) from fallback_err


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
        # Remove opening fence (with optional language tag)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    # Try parsing directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract JSON from response", cleaned, 0)


def batch_reflect(transcript: str, config: Config) -> dict:
    """
    The "mega-prompt" — analyzes a full session transcript and extracts
    structured memories and a curiosity question for next session.
    
    Uses the reflection model (which may be cheaper than the chat model).
    
    Args:
        transcript: The full session transcript as a string
        config: App configuration
    
    Returns:
        Dict with 'memories' (list[str]) and 'inbox_question' (str)
        Returns empty defaults if reflection fails.
    """
    if not transcript.strip():
        return {"memories": [], "inbox_question": ""}

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the session transcript to analyze:\n\n{transcript}"},
    ]

    try:
        # Use reflection model (may differ from chat model)
        kwargs = {
            "model": config.reflect_model,
            "messages": messages,
        }
        # If reflect model is the fallback (ollama), pass its API base
        if config.reflect_model == config.fallback_model and config.fallback_api_base:
            kwargs["api_base"] = config.fallback_api_base

        response = litellm.completion(**kwargs)
        raw_content = response.choices[0].message.content
        if not raw_content:
            print("  [Warning] Reflection returned empty response from LLM.")
            return {"memories": [], "inbox_question": ""}
        content = raw_content.strip()

        # Parse JSON response (handles markdown fences, stray text, etc.)
        result = _extract_json(content)

        # Validate structure
        raw_memories = result.get("memories", [])
        if not isinstance(raw_memories, list):
            raw_memories = [raw_memories]

        # Normalize memories: accept both {"text": ..., "domain": ...} objects
        # and plain strings (backward compatibility)
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
        # LLM didn't return valid JSON even after extraction attempts
        print(f"  [Warning] Reflection returned invalid JSON: {e}")
        try:
            print(f"  Raw response: {content[:300]}")
        except NameError:
            pass
        return {"memories": [], "inbox_question": ""}

    except Exception as e:
        print(f"  [Warning] Reflection failed: {e}")
        return {"memories": [], "inbox_question": ""}
