"""
AIFreeMind Tool Registry
Local execution tools that the AI agent can invoke.
Each tool has an explicit docstring that LiteLLM uses to generate
the OpenAI-compatible tool/function schema for the LLM.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

from .policy import (
    SAFE_COMMAND_PREFIXES,
    is_safe_command,
    resolve_under_root,
)


# Re-export for callers that imported from tools historically
__all__ = [
    "read_file",
    "write_file",
    "run_command",
    "is_safe_command",
    "SAFE_COMMAND_PREFIXES",
    "TOOL_REGISTRY",
    "get_tool_schemas",
    "make_tool_bindings",
]


def read_file(
    filepath: str,
    *,
    workspace_root: Optional[Path] = None,
    allow_outside: bool = False,
) -> str:
    """Read and return the contents of a local file.

    Args:
        filepath: The path to the file to read. Can be absolute or relative.
        workspace_root: Optional jail root (defaults to cwd if omitted).
        allow_outside: If True, skip workspace jail.

    Returns:
        The full text content of the file, or an error message if the file
        cannot be read.
    """
    root = workspace_root if workspace_root is not None else Path.cwd()
    path, err = resolve_under_root(filepath, root, allow_outside=allow_outside)
    if err:
        return err
    assert path is not None
    try:
        if not path.exists():
            return f"Error: File not found: {path}"
        if not path.is_file():
            return f"Error: Not a file: {path}"
        content = path.read_text(encoding="utf-8", errors="replace")
        max_chars = 50_000
        if len(content) > max_chars:
            return content[:max_chars] + f"\n\n... [truncated, file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(
    filepath: str,
    content: str,
    *,
    workspace_root: Optional[Path] = None,
    allow_outside: bool = False,
) -> str:
    """Create or overwrite a file with the given content.

    Args:
        filepath: The path where the file should be written. Parent directories
                  will be created if they don't exist.
        content: The text content to write to the file.
        workspace_root: Optional jail root (defaults to cwd if omitted).
        allow_outside: If True, skip workspace jail.

    Returns:
        A success message with the file path, or an error message.
    """
    root = workspace_root if workspace_root is not None else Path.cwd()
    path, err = resolve_under_root(filepath, root, allow_outside=allow_outside)
    if err:
        return err
    assert path is not None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_command(command: str) -> str:
    """Execute a terminal command and return its output.

    Runs the command in a subprocess and captures both stdout and stderr.
    Has a 30-second timeout to prevent hanging.

    Args:
        command: The shell command to execute (e.g., 'ls -la', 'cat file.txt',
                 'python --version').

    Returns:
        The combined stdout and stderr output, or an error message if the
        command fails or times out.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path.cwd()),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        if not output.strip():
            output = "[Command completed with no output]"
        max_chars = 10_000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [truncated, output is {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def make_tool_bindings(
    workspace_root: Path,
    allow_outside: bool = False,
) -> dict[str, Callable]:
    """
    Build callables bound to workspace jail settings for the agent loop.
    """

    def _read(filepath: str) -> str:
        return read_file(
            filepath,
            workspace_root=workspace_root,
            allow_outside=allow_outside,
        )

    def _write(filepath: str, content: str) -> str:
        return write_file(
            filepath,
            content,
            workspace_root=workspace_root,
            allow_outside=allow_outside,
        )

    return {
        "read_file": _read,
        "write_file": _write,
        "run_command": run_command,
    }


# Default registry (cwd jail) — prefer make_tool_bindings in production
TOOL_REGISTRY: dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}


def get_tool_schemas() -> list[dict]:
    """
    Generate OpenAI-compatible tool definitions for all registered tools.
    LiteLLM passes these to the LLM so it knows what tools are available.
    """
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read and return the contents of a local file within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path to the file to read. Can be absolute or relative to workspace.",
                        }
                    },
                    "required": ["filepath"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Create or overwrite a file with the given content. "
                    "Requires user confirmation before writing. Paths are confined to the workspace."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path where the file should be written.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The text content to write to the file.",
                        },
                    },
                    "required": ["filepath", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": (
                    "Execute a terminal command and return its output. Has a 30-second timeout. "
                    "Unsafe/mutating commands require user confirmation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    return schemas
