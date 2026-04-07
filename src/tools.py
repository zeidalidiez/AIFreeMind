"""
AIFreeMind Tool Registry
Local execution tools that the AI agent can invoke.
Each tool has an explicit docstring that LiteLLM uses to generate
the OpenAI-compatible tool/function schema for the LLM.
"""

import subprocess
import sys
from pathlib import Path
from typing import Callable


# ── Core Tools ──────────────────────────────────────────────


def read_file(filepath: str) -> str:
    """Read and return the contents of a local file.
    
    Args:
        filepath: The path to the file to read. Can be absolute or relative.
    
    Returns:
        The full text content of the file, or an error message if the file
        cannot be read.
    """
    try:
        path = Path(filepath).resolve()
        if not path.exists():
            return f"Error: File not found: {path}"
        if not path.is_file():
            return f"Error: Not a file: {path}"
        content = path.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files to avoid blowing up context
        max_chars = 50_000
        if len(content) > max_chars:
            return content[:max_chars] + f"\n\n... [truncated, file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath: str, content: str) -> str:
    """Create or overwrite a file with the given content.
    
    Args:
        filepath: The path where the file should be written. Parent directories
                  will be created if they don't exist.
        content: The text content to write to the file.
    
    Returns:
        A success message with the file path, or an error message.
    """
    try:
        path = Path(filepath).resolve()
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
        # Truncate very long output
        max_chars = 10_000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [truncated, output is {len(output)} chars total]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


# ── Tool Registry ──────────────────────────────────────────


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
                "description": "Read and return the contents of a local file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "The path to the file to read. Can be absolute or relative.",
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
                "description": "Create or overwrite a file with the given content.",
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
                "description": "Execute a terminal command and return its output. Has a 30-second timeout.",
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
