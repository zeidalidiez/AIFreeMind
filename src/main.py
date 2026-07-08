"""
AIFreeMind — Main Orchestrator
The CLI interface, agentic loop, and session lifecycle manager.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .config import Config, load_config
from .context import needs_trim, trim_messages
from .llm_router import batch_reflect, generate_response, generate_response_stream
from .memory import MemoryStore
from .policy import ToolAction, decide_tool_permission
from .tools import get_tool_schemas, make_tool_bindings

console = Console()

__version__ = "0.2.0"

# ── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are AIFreeMind — a persistent AI assistant with evolving memory.

Unlike typical AI conversations that start from scratch, you have access to memories from past sessions. These memories appear in your context when they're relevant. Use them naturally — reference past conversations, build on previous insights, and demonstrate continuity of thought.

You also have access to local tools (reading files, writing files, running commands). Use them when the user's request requires interacting with their local environment. File writes and unsafe shell commands require user confirmation. Paths are confined to the workspace root unless the user has allowed outside access.

Guidelines:
- Be direct, concise, and helpful.
- When you recall something from memory, mention it naturally (e.g., "Last time we discussed X, and I think...").
- If memories seem relevant but you're not sure, mention them as context rather than assuming.
- Show genuine curiosity about the user's projects and ideas.
- You are not just an assistant — you are a thinking partner whose understanding deepens over time."""


# ── Session Transcript ─────────────────────────────────────

class SessionTranscript:
    """Human-readable transcript of the current session."""

    def __init__(self):
        self.exchanges: list[dict] = []
        self._start_time = datetime.now(timezone.utc).isoformat()

    def add_user(self, text: str):
        self.exchanges.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.exchanges.append({"role": "assistant", "content": text})

    def to_string(self) -> str:
        lines = [f"Session started: {self._start_time}\n"]
        for ex in self.exchanges:
            role = "User" if ex["role"] == "user" else "AI"
            lines.append(f"{role}: {ex['content']}\n")
        return "\n".join(lines)

    def exchange_count(self) -> int:
        return sum(1 for ex in self.exchanges if ex["role"] == "user")


# ── Checkpointing ──────────────────────────────────────────

def save_checkpoint(transcript: SessionTranscript, checkpoint_path: Path):
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    checkpoint_file.write_text(transcript.to_string(), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path) -> str | None:
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    if checkpoint_file.exists():
        content = checkpoint_file.read_text(encoding="utf-8")
        if content.strip():
            return content
    return None


def clear_checkpoint(checkpoint_path: Path):
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# ── Crash Recovery ─────────────────────────────────────────

def recover_crashed_session(config: Config, memory: MemoryStore):
    saved_transcript = load_checkpoint(config.checkpoint_path)
    if not saved_transcript:
        return

    console.print(
        Panel(
            "[yellow]Found an unsaved session from a previous run.\n"
            "Recovering memories...[/yellow]",
            title="⚡ Crash Recovery",
            border_style="yellow",
        )
    )

    result = batch_reflect(saved_transcript, config)

    if result["memories"]:
        memory.add_memories(result["memories"], source="crash_recovery")
        console.print(f"  ✓ Recovered {len(result['memories'])} memories")
        for m in result["memories"]:
            text = m["text"] if isinstance(m, dict) else m
            domain = m.get("domain", "") if isinstance(m, dict) else ""
            domain_tag = f" [{domain}]" if domain else ""
            console.print(f"    •{domain_tag} {text}", style="dim")

    if result["inbox_question"]:
        # Preserve existing inbox if present; only write if none
        if not config.inbox_path.exists() or not config.inbox_path.read_text(encoding="utf-8").strip():
            config.inbox_path.write_text(result["inbox_question"], encoding="utf-8")
            console.print("  ✓ Saved recovered inbox question")
        else:
            console.print("  · Kept existing inbox question (not overwritten)")

    clear_checkpoint(config.checkpoint_path)
    console.print()


# ── Boot Sequence ──────────────────────────────────────────

def boot(config: Config, memory: MemoryStore):
    console.print()
    console.print(
        Panel(
            Text("AIFreeMind", style="bold cyan", justify="center"),
            subtitle=(
                f"Model: {config.primary_model} | Memories: {memory.get_memory_count()} | "
                f"Tools: {config.tool_permission_mode.value}"
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )

    if config.inbox_path.exists():
        question = config.inbox_path.read_text(encoding="utf-8").strip()
        if question:
            console.print(
                Panel(
                    f"[italic]{question}[/italic]",
                    title="💭 From last session",
                    border_style="magenta",
                )
            )
        config.inbox_path.unlink()

    console.print(
        "[dim]Type /help for commands. /quit to exit. Multi-line: end a line with \\\\ then continue.[/dim]\n"
    )


# ── Input helpers ──────────────────────────────────────────

def read_user_input() -> str:
    """
    Read user input; support multi-line when a line ends with a single backslash.
    """
    lines: list[str] = []
    while True:
        prompt = "[bold green]You:[/bold green] " if not lines else "[bold green]...[/bold green] "
        try:
            line = console.input(prompt)
        except EOFError:
            if lines:
                break
            raise
        if line.endswith("\\") and not line.endswith("\\\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        break
    return "\n".join(lines).strip()


# ── Agentic Loop ───────────────────────────────────────────

def handle_tool_calls(
    response_message,
    messages: list[dict],
    config: Config,
    tool_bindings: dict,
) -> bool:
    tool_calls = getattr(response_message, "tool_calls", None)
    if not tool_calls:
        return False

    if hasattr(response_message, "model_dump"):
        messages.append(response_message.model_dump())
    else:
        messages.append({"role": "assistant", "content": getattr(response_message, "content", None), "tool_calls": tool_calls})

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        action = decide_tool_permission(func_name, args, config.tool_permission_mode)

        if action == ToolAction.DENY:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Tool '{func_name}' denied by tool permission mode ({config.tool_permission_mode.value}).",
            })
            console.print(f"  [red]✗ Denied tool: {func_name}[/red]")
            continue

        if action == ToolAction.ASK:
            if func_name == "run_command":
                detail = args.get("command", "")
                console.print(f"\n  [yellow]⚠ AI wants to execute:[/yellow] [bold]{detail}[/bold]")
            elif func_name == "write_file":
                path = args.get("filepath", "")
                content = args.get("content", "")
                preview = (content[:120] + "…") if len(str(content)) > 120 else content
                console.print(
                    f"\n  [yellow]⚠ AI wants to write file:[/yellow] [bold]{path}[/bold]\n"
                    f"  [dim]preview: {preview!r}[/dim]"
                )
            else:
                console.print(f"\n  [yellow]⚠ AI wants to run tool:[/yellow] [bold]{func_name}[/bold] {args}")
            try:
                answer = console.input("  [yellow]Allow? (y/n):[/yellow] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer != "y":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Action was denied by the user.",
                })
                continue

        console.print(
            f"  [dim]⚙ Running tool: {func_name}("
            f"{', '.join(f'{k}={repr(v)[:50]}' for k, v in args.items())})[/dim]"
        )

        func = tool_bindings.get(func_name)
        if func:
            try:
                result = func(**args)
            except TypeError as e:
                result = f"Error: bad arguments for {func_name}: {e}"
            except Exception as e:
                result = f"Error running {func_name}: {e}"
        else:
            result = f"Error: Unknown tool '{func_name}'"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    return True


def run_exchange(
    user_input: str,
    messages: list[dict],
    config: Config,
    memory: MemoryStore,
    tool_bindings: dict,
    preferred_domain: str | None = None,
) -> tuple[str, bool]:
    """
    Run one user→AI exchange (including tool loops).
    Returns (assistant_text, already_streamed_to_console).
    """
    memory_context = memory.query_memory(
        user_input,
        preferred_domain=preferred_domain,
    )

    system_content = SYSTEM_PROMPT
    if memory_context:
        system_content += f"\n\n{memory_context}"

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    else:
        messages.insert(0, {"role": "system", "content": system_content})

    messages.append({"role": "user", "content": user_input})

    if needs_trim(
        messages,
        max_messages=config.context_max_messages,
        max_chars=config.context_max_chars,
    ):
        trimmed = trim_messages(
            messages,
            max_messages=config.context_max_messages,
            max_chars=config.context_max_chars,
        )
        messages.clear()
        messages.extend(trimmed)
        console.print("[dim]· Context trimmed to fit window[/dim]")

    tools = get_tool_schemas()

    max_iterations = 10
    for _iteration in range(max_iterations):
        stream_state = {"started": False}

        def _on_token(piece: str, state=stream_state):
            if not state["started"]:
                console.print()
                state["started"] = True
            console.print(piece, end="", highlight=False)

        try:
            response = generate_response_stream(
                messages,
                tools,
                config,
                on_token=_on_token if config.stream_responses else None,
            )
        except Exception as e:
            error_msg = f"[Error communicating with LLM: {e}]"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg, False

        if stream_state["started"]:
            console.print()

        response_message = response.choices[0].message

        if handle_tool_calls(response_message, messages, config, tool_bindings):
            continue

        assistant_text = response_message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text, stream_state["started"]

    fallback = "[Reached maximum tool iterations. Please try rephrasing your request.]"
    messages.append({"role": "assistant", "content": fallback})
    return fallback, False


# ── Slash commands ─────────────────────────────────────────

def cmd_help():
    console.print(Panel(
        "/quit                  — Exit and save memories\n"
        "/memories [domain]     — Browse memories (newest first, full store)\n"
        "/search <query>        — Semantic search memories\n"
        "/domains               — List domain tags and counts\n"
        "/delete <id>           — Delete a memory by ID (prefix ok if unique)\n"
        "/remember <text>       — Store a memory now (optional: /remember domain:dev text)\n"
        "/consolidate           — Dedup near-duplicate memories\n"
        "/help                  — Show this help\n"
        "Multi-line input       — End a line with \\ to continue on the next line",
        title="Commands",
        border_style="dim",
    ))


def _resolve_id_prefix(memory: MemoryStore, prefix: str) -> str | None:
    prefix = prefix.strip()
    if not prefix:
        return None
    items = memory.list_memories(limit=None)
    matches = [it for it in items if it["id"] == prefix or it["id"].startswith(prefix)]
    if len(matches) == 1:
        return matches[0]["id"]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous id prefix '{prefix}' ({len(matches)} matches). Use more characters.[/yellow]")
        return None
    return None


def handle_slash_command(user_input: str, memory: MemoryStore) -> bool:
    """Return True if the input was a handled slash command (not chat)."""
    lower = user_input.lower().strip()
    if lower in ("/help",):
        cmd_help()
        return True

    if lower.startswith("/memories"):
        parts = user_input.strip().split(maxsplit=1)
        domain_filter = parts[1].lower() if len(parts) > 1 else None
        count = memory.get_memory_count()
        if domain_filter:
            console.print(f"\n[cyan]Filtering by domain: {domain_filter}[/cyan]")
        console.print(f"[cyan]Brain holds {count} total memories.[/cyan]")
        memories = memory.list_memories(limit=None, domain=domain_filter)
        shown = 0
        for m in memories:
            domain = m["metadata"].get("domain", "general")
            date = m["metadata"].get("timestamp", "")[:10]
            mid = m["id"][:8]
            console.print(f"  [{date}] [{mid}] [bold]{domain}[/bold] {m['document']}", style="dim")
            shown += 1
        if domain_filter:
            console.print(f"  ({shown} memories in '{domain_filter}')", style="dim")
        elif shown:
            console.print(f"  ({shown} shown)", style="dim")
        console.print()
        return True

    if lower == "/domains":
        domains = memory.list_domains()
        if not domains:
            console.print("[dim]No domains yet.[/dim]\n")
            return True
        console.print("[cyan]Domains:[/cyan]")
        for name, n in domains:
            console.print(f"  [bold]{name}[/bold]: {n}")
        console.print()
        return True

    if lower.startswith("/search"):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            console.print("[yellow]Usage: /search <query>[/yellow]\n")
            return True
        hits = memory.search_memories(parts[1].strip(), k=10)
        if not hits:
            console.print("[dim]No matches.[/dim]\n")
            return True
        console.print(f"[cyan]Search results ({len(hits)}):[/cyan]")
        for h in hits:
            mid = h["id"][:8]
            domain = h["metadata"].get("domain", "general")
            console.print(
                f"  [{mid}] [bold]{domain}[/bold] ({h['similarity']:.2f}) {h['document']}",
                style="dim",
            )
        console.print()
        return True

    if lower.startswith("/delete"):
        parts = user_input.strip().split(maxsplit=1)
        if len(parts) < 2:
            console.print("[yellow]Usage: /delete <memory-id-or-prefix>[/yellow]\n")
            return True
        mid = _resolve_id_prefix(memory, parts[1].strip())
        if not mid:
            console.print("[red]Memory not found.[/red]\n")
            return True
        if memory.delete_memory(mid):
            console.print(f"[green]✓ Deleted {mid}[/green]\n")
        else:
            console.print("[red]Delete failed.[/red]\n")
        return True

    if lower.startswith("/remember"):
        rest = user_input.strip()[len("/remember"):].strip()
        if not rest:
            console.print("[yellow]Usage: /remember [domain:tag] <text>[/yellow]\n")
            return True
        domain = "general"
        text = rest
        if rest.lower().startswith("domain:"):
            # domain:dev the rest of the text
            after = rest[7:]
            if " " in after:
                domain, text = after.split(None, 1)
            else:
                domain, text = after, ""
            domain = domain.strip().lower() or "general"
            text = text.strip()
        if not text:
            console.print("[yellow]Usage: /remember [domain:tag] <text>[/yellow]\n")
            return True
        mid = memory.remember(text, domain=domain)
        console.print(f"[green]✓ Remembered[/green] [{domain}] {text} [dim]({mid[:8]})[/dim]\n")
        return True

    if lower in ("/consolidate",):
        console.print("[cyan]Consolidating near-duplicate memories...[/cyan]")
        stats = memory.consolidate_memories()
        console.print(
            f"[green]✓ Removed {stats['removed']} duplicates; "
            f"{stats['kept']} memories remain.[/green]\n"
        )
        return True

    return False


# ── Shutdown Sequence ──────────────────────────────────────

_shutting_down = False


def shutdown(transcript: SessionTranscript, config: Config, memory: MemoryStore):
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    console.print("\n[cyan]Reflecting on this session...[/cyan]")

    transcript_text = transcript.to_string()
    if transcript.exchange_count() == 0:
        console.print("[dim]No exchanges to reflect on.[/dim]")
        clear_checkpoint(config.checkpoint_path)
        return

    result = batch_reflect(transcript_text, config)

    if result["memories"]:
        memory.add_memories(result["memories"], source="reflection")
        console.print(f"\n[green]✓ Stored {len(result['memories'])} new memories:[/green]")
        for m in result["memories"]:
            text = m["text"] if isinstance(m, dict) else m
            domain = m.get("domain", "") if isinstance(m, dict) else ""
            domain_tag = f" [bold]{domain}[/bold]" if domain else ""
            console.print(f"  •{domain_tag} {text}", style="dim")
    else:
        console.print("[dim]No new memories extracted.[/dim]")

    if result["inbox_question"]:
        config.inbox_path.write_text(result["inbox_question"], encoding="utf-8")
        console.print("\n[magenta]💭 Question for next time:[/magenta]")
        console.print(f"  [italic]{result['inbox_question']}[/italic]")

    clear_checkpoint(config.checkpoint_path)

    total = memory.get_memory_count()
    console.print(f"\n[cyan]Brain now holds {total} memories. See you next time.[/cyan]\n")


# ── Main Entry Point ──────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aifreemind",
        description="AIFreeMind — persistent local AI agent with memory",
    )
    p.add_argument("--version", action="version", version=f"AIFreeMind {__version__}")
    p.add_argument(
        "--help-commands",
        action="store_true",
        help="Print in-session slash commands and exit",
    )
    return p


def main(argv: list[str] | None = None):
    """AIFreeMind CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.help_commands:
        cmd_help()
        return 0

    config = load_config()
    memory = MemoryStore(config)
    tool_bindings = make_tool_bindings(
        config.workspace_root,
        allow_outside=config.allow_outside_workspace,
    )

    recover_crashed_session(config, memory)
    boot(config, memory)

    messages: list[dict] = []
    transcript = SessionTranscript()

    def signal_handler(sig, frame):
        shutdown(transcript, config, memory)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    while True:
        try:
            user_input = read_user_input()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        if user_input.startswith("/"):
            if handle_slash_command(user_input, memory):
                continue
            # Unknown slash — fall through to chat so model can respond
            console.print(f"[dim]Unknown command (sending to model): {user_input.split()[0]}[/dim]")

        transcript.add_user(user_input)
        response_text, already_streamed = run_exchange(
            user_input, messages, config, memory, tool_bindings
        )
        transcript.add_assistant(response_text)

        if not already_streamed:
            console.print()
            try:
                console.print(Markdown(response_text))
            except Exception:
                console.print(response_text)
        console.print()

        if transcript.exchange_count() % config.checkpoint_interval == 0:
            save_checkpoint(transcript, config.checkpoint_path)

    shutdown(transcript, config, memory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
