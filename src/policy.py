"""
AIFreeMind tool policy — pure decision helpers (no I/O / Rich).

Used by tools + main for:
  - command auto-safe classification (chain-resistant)
  - path jail (workspace root)
  - permission mode: auto | ask | deny
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional


class ToolPermissionMode(str, Enum):
    AUTO = "auto"  # safe ops auto; mutating ops ask (default behavior intent)
    ASK = "ask"  # always ask before mutating tools
    DENY = "deny"  # never run mutating tools


# Characters / patterns that make a shell command non-auto-safe (chaining, redirection, etc.)
_UNSAFE_SHELL_PATTERNS = re.compile(
    r"""
    ; | && | \|\| | \| | ` | \$\( | \n | \r |
    > | >> | < |
    \bcmd\s+/c\b | \bpowershell\b | \bpwsh\b |
    \binvoke-expression\b | \biex\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# First-token (or first few tokens) allowlist for genuinely read-only single commands.
# Windows + Unix friendly.
SAFE_COMMAND_PREFIXES: tuple[str, ...] = (
    # File listing / inspection
    "ls",
    "dir",
    "cat",
    "type",
    "head",
    "tail",
    "find",  # further restricted: no -delete/-exec/etc. (see is_safe_command)
    "where",
    "where.exe",
    "which",
    "wc",
    "file",
    "stat",
    "tree",
    "get-childitem",
    "gci",
    "get-content",
    "gc",
    "get-item",
    "gi",
    "get-location",
    "gl",
    "test-path",
    # Environment info
    "echo",
    "pwd",
    "whoami",
    "hostname",
    "uname",
    "env",  # bare only (no VAR=value assignments)
    "printenv",  # bare or single var name only
    "set",  # bare only — `set FOO=bar` is mutation
    "get-command",
    "get-process",
    "gps",
    "get-date",
    # Version checks
    "python --version",
    "python3 --version",
    "py --version",
    "pip --version",
    "pip3 --version",
    "node --version",
    "npm --version",
    "git --version",
    # Git read operations
    "git status",
    "git log",
    "git diff",
    "git branch",
    "git remote",
    "git show",
    "git tag",
    "git ls-files",
    "git rev-parse",
    # Package inspection
    "pip list",
    "pip show",
    "pip freeze",
    "npm list",
    "npm ls",
    "npm info",
    "npm view",
)

# Commands that are only auto-safe with zero arguments (listing only).
_BARE_ONLY_COMMANDS = frozenset({
    "set",
    "env",
    "pwd",
    "whoami",
    "hostname",
    "uname",
    "get-location",
    "gl",
    "get-date",
})

# printenv: bare or a single variable name (no assignments / multi-arg abuse)
_PRINTENV_SAFE = re.compile(r"^printenv(?:\s+[A-Za-z_][A-Za-z0-9_]*)?$")

# Unix find mutating / side-effect flags (must never auto-run)
_FIND_UNSAFE_FLAGS = re.compile(
    r"(?:^|\s)-(?:delete|exec|execdir|ok|okdir|fprint|fprint0|fls|fprintf)\b",
    re.IGNORECASE,
)

# Tools that mutate the environment and always require policy checks
MUTATING_TOOLS = frozenset({"write_file", "run_command"})
# run_command is mutating unless classified auto-safe


def normalize_permission_mode(value: str | None) -> ToolPermissionMode:
    raw = (value or "ask").strip().lower()
    if raw in ("auto", "ask", "deny"):
        return ToolPermissionMode(raw)
    return ToolPermissionMode.ASK


def command_has_shell_metacharacters(command: str) -> bool:
    """True if command uses chaining, pipes, redirects, subshells, or multiline."""
    if not command or not command.strip():
        return True
    return bool(_UNSAFE_SHELL_PATTERNS.search(command))


def is_safe_command(command: str) -> bool:
    """
    True only for single, read-only commands matching known safe prefixes.

    Rejects chaining/redirection so e.g. `ls; rm -rf /` is NOT auto-safe.
    Extra guards:
      - `set FOO=bar` / `env FOO=bar ...` are not safe (bare listing only)
      - `find ... -delete` / `-exec` / similar are not safe
    """
    if not command or not isinstance(command, str):
        return False
    cmd = command.strip()
    if not cmd:
        return False
    if command_has_shell_metacharacters(cmd):
        return False
    lowered = cmd.lower()

    # Bare-only commands: exact match, no args
    first = lowered.split(None, 1)[0]
    if first in _BARE_ONLY_COMMANDS:
        return lowered == first

    # printenv: bare or single env var name
    if first == "printenv":
        return bool(_PRINTENV_SAFE.match(lowered))

    # find: prefix allowlist + reject mutating flags
    if first == "find":
        if _FIND_UNSAFE_FLAGS.search(lowered):
            return False
        # still require it match the general prefix allowlist below

    # Exact prefix match on the full command string (first tokens of known safe forms)
    for prefix in SAFE_COMMAND_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + " "):
            return True
    return False


def resolve_under_root(
    filepath: str,
    workspace_root: Path,
    *,
    allow_outside: bool = False,
) -> tuple[Optional[Path], Optional[str]]:
    """
    Resolve filepath and enforce workspace jail.

    Returns (resolved_path, error_message). On success error_message is None.
    If allow_outside is True, any existing resolvable path is accepted.
    """
    try:
        root = workspace_root.resolve()
        path = Path(filepath)
        # Resolve relative paths against workspace root, not process cwd
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
    except (OSError, RuntimeError, ValueError) as e:
        return None, f"Error: invalid path: {e}"

    if allow_outside:
        return path, None

    try:
        path.relative_to(root)
    except ValueError:
        return None, (
            f"Error: path outside workspace root ({root}): {path}. "
            "Set TOOL_ALLOW_OUTSIDE_WORKSPACE=true to override."
        )
    return path, None


def path_is_inside_workspace(filepath: str, workspace_root: Path) -> bool:
    resolved, err = resolve_under_root(filepath, workspace_root, allow_outside=False)
    return resolved is not None and err is None


class ToolAction(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


def decide_tool_permission(
    tool_name: str,
    args: dict,
    mode: ToolPermissionMode,
) -> ToolAction:
    """
    Decide whether a tool invocation may run without asking, needs confirm, or is denied.

    - deny mode: all mutating tools denied (run_command always mutating here)
    - ask mode: mutating tools ask; read_file always allow
    - auto mode: safe run_command allow; unsafe run_command / write_file ask
    """
    name = (tool_name or "").strip()

    if name == "read_file":
        return ToolAction.ALLOW

    if name == "write_file":
        if mode == ToolPermissionMode.DENY:
            return ToolAction.DENY
        # Always ask for writes (never silent auto-write)
        return ToolAction.ASK if mode in (ToolPermissionMode.ASK, ToolPermissionMode.AUTO) else ToolAction.DENY

    if name == "run_command":
        command = str(args.get("command", "") or "")
        if mode == ToolPermissionMode.DENY:
            return ToolAction.DENY
        if mode == ToolPermissionMode.ASK:
            return ToolAction.ASK
        # AUTO
        if is_safe_command(command):
            return ToolAction.ALLOW
        return ToolAction.ASK

    # Unknown tools: conservative
    if mode == ToolPermissionMode.DENY:
        return ToolAction.DENY
    return ToolAction.ASK
