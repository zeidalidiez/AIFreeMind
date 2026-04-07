"""
AIFreeMind — Main Orchestrator
The CLI interface, agentic loop, and session lifecycle manager.
"""

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
from .llm_router import batch_reflect, generate_response
from .memory import MemoryStore
from .tools import TOOL_REGISTRY, get_tool_schemas, is_safe_command

console = Console()


# ── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are AIFreeMind — a persistent AI assistant with evolving memory.

Unlike typical AI conversations that start from scratch, you have access to memories from past sessions. These memories appear in your context when they're relevant. Use them naturally — reference past conversations, build on previous insights, and demonstrate continuity of thought.

You also have access to local tools (reading files, writing files, running commands). Use them when the user's request requires interacting with their local environment.

Guidelines:
- Be direct, concise, and helpful.
- When you recall something from memory, mention it naturally (e.g., "Last time we discussed X, and I think...").
- If memories seem relevant but you're not sure, mention them as context rather than assuming.
- Show genuine curiosity about the user's projects and ideas.
- You are not just an assistant — you are a thinking partner whose understanding deepens over time."""


# ── Session Transcript ─────────────────────────────────────

class SessionTranscript:
    """
    Maintains a human-readable transcript of the current session,
    separate from the full message history (which includes system
    prompts, tool calls, etc.).
    """

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
        """Count user-assistant exchange pairs."""
        return sum(1 for ex in self.exchanges if ex["role"] == "user")


# ── Checkpointing ──────────────────────────────────────────

def save_checkpoint(transcript: SessionTranscript, checkpoint_path: Path):
    """Save the current transcript to a checkpoint file."""
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    checkpoint_file.write_text(transcript.to_string(), encoding="utf-8")


def load_checkpoint(checkpoint_path: Path) -> str | None:
    """Check for an unsaved transcript from a crashed session."""
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    if checkpoint_file.exists():
        content = checkpoint_file.read_text(encoding="utf-8")
        if content.strip():
            return content
    return None


def clear_checkpoint(checkpoint_path: Path):
    """Remove checkpoint file after clean shutdown."""
    checkpoint_file = checkpoint_path / "session_checkpoint.txt"
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# ── Crash Recovery ─────────────────────────────────────────

def recover_crashed_session(config: Config, memory: MemoryStore):
    """
    Check for a checkpoint from a previous crashed session.
    If found, run reflection on it to salvage memories.
    """
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
        config.inbox_path.write_text(result["inbox_question"], encoding="utf-8")
        console.print("  ✓ Saved recovered inbox question")

    clear_checkpoint(config.checkpoint_path)
    console.print()


# ── Boot Sequence ──────────────────────────────────────────

def boot(config: Config, memory: MemoryStore):
    """Display welcome message and inbox question."""

    # Header
    console.print()
    console.print(
        Panel(
            Text("AIFreeMind", style="bold cyan", justify="center"),
            subtitle=f"Model: {config.primary_model} | Memories: {memory.get_memory_count()}",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # Inbox question from last session
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

    console.print("[dim]Type /quit to exit. Your memories persist between sessions.[/dim]\n")


# ── Agentic Loop ───────────────────────────────────────────

def handle_tool_calls(response_message, messages: list[dict]) -> bool:
    """
    Process any tool calls in the LLM response.
    Executes tools, appends results to message history.
    Returns True if tool calls were processed.
    """
    tool_calls = getattr(response_message, "tool_calls", None)
    if not tool_calls:
        return False

    # Append the assistant message with tool calls
    messages.append(response_message.model_dump())

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        # ── Command Sandboxing ──
        # Safe commands auto-execute; unsafe commands require confirmation
        if func_name == "run_command":
            command = args.get("command", "")
            if not is_safe_command(command):
                console.print(f"\n  [yellow]⚠ AI wants to execute:[/yellow] [bold]{command}[/bold]")
                try:
                    answer = console.input("  [yellow]Allow? (y/n):[/yellow] ").strip().lower()
                except EOFError:
                    answer = "n"
                if answer != "y":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Command was denied by the user.",
                    })
                    continue

        console.print(f"  [dim]⚙ Running tool: {func_name}({', '.join(f'{k}={repr(v)[:50]}' for k, v in args.items())})[/dim]")

        func = TOOL_REGISTRY.get(func_name)
        if func:
            result = func(**args)
        else:
            result = f"Error: Unknown tool '{func_name}'"

        # Append tool result to message history
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    return True


def run_exchange(user_input: str, messages: list[dict], config: Config, memory: MemoryStore) -> str:
    """
    Run a single user→AI exchange, including any tool call loops.
    Returns the final assistant text response.
    """
    # Memory-first: query the brain before calling the LLM
    memory_context = memory.query_memory(user_input)

    # Build the system message with injected memories
    system_content = SYSTEM_PROMPT
    if memory_context:
        system_content += f"\n\n{memory_context}"

    # Update or set the system message
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system_content
    else:
        messages.insert(0, {"role": "system", "content": system_content})

    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Tool schemas
    tools = get_tool_schemas()

    # Agentic loop: keep going while the LLM wants to call tools
    max_iterations = 10  # safety limit
    for _ in range(max_iterations):
        try:
            response = generate_response(messages, tools, config)
        except Exception as e:
            error_msg = f"[Error communicating with LLM: {e}]"
            console.print(f"[red]{error_msg}[/red]")
            return error_msg

        response_message = response.choices[0].message

        # If there are tool calls, execute them and loop back
        if handle_tool_calls(response_message, messages):
            continue

        # Otherwise, we have a text response — we're done
        assistant_text = response_message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    # Safety: too many tool iterations
    fallback = "[Reached maximum tool iterations. Please try rephrasing your request.]"
    messages.append({"role": "assistant", "content": fallback})
    return fallback


# ── Shutdown Sequence ──────────────────────────────────────

def shutdown(transcript: SessionTranscript, config: Config, memory: MemoryStore):
    """
    Clean shutdown: reflect on the session, store memories,
    save inbox question, clean up checkpoint.
    """
    console.print("\n[cyan]Reflecting on this session...[/cyan]")

    transcript_text = transcript.to_string()
    if transcript.exchange_count() == 0:
        console.print("[dim]No exchanges to reflect on.[/dim]")
        clear_checkpoint(config.checkpoint_path)
        return

    result = batch_reflect(transcript_text, config)

    # Store memories
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

    # Save inbox question
    if result["inbox_question"]:
        config.inbox_path.write_text(result["inbox_question"], encoding="utf-8")
        console.print(f"\n[magenta]💭 Question for next time:[/magenta]")
        console.print(f"  [italic]{result['inbox_question']}[/italic]")

    # Clean up checkpoint
    clear_checkpoint(config.checkpoint_path)

    total = memory.get_memory_count()
    console.print(f"\n[cyan]Brain now holds {total} memories. See you next time.[/cyan]\n")


# ── Main Entry Point ──────────────────────────────────────

def main():
    """AIFreeMind CLI entry point."""

    # Load config
    config = load_config()

    # Initialize memory
    memory = MemoryStore(config)

    # Crash recovery
    recover_crashed_session(config, memory)

    # Boot
    boot(config, memory)

    # Session state
    messages: list[dict] = []
    transcript = SessionTranscript()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        shutdown(transcript, config, memory)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Main input loop
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        # Special commands
        if user_input.lower().startswith("/memories"):
            # Parse optional domain filter: /memories dev
            parts = user_input.strip().split(maxsplit=1)
            domain_filter = parts[1].lower() if len(parts) > 1 else None
            count = memory.get_memory_count()
            if domain_filter:
                console.print(f"\n[cyan]Filtering by domain: {domain_filter}[/cyan]")
            console.print(f"[cyan]Brain holds {count} total memories.[/cyan]")
            if count > 0:
                memories = memory.get_all_memories(limit=50)
                shown = 0
                for m in memories:
                    domain = m["metadata"].get("domain", "general")
                    if domain_filter and domain != domain_filter:
                        continue
                    date = m["metadata"].get("timestamp", "")[:10]
                    console.print(f"  [{date}] [bold]{domain}[/bold] {m['document']}", style="dim")
                    shown += 1
                if domain_filter:
                    console.print(f"  ({shown} memories in '{domain_filter}')", style="dim")
            console.print()
            continue

        if user_input.lower() == "/help":
            console.print(Panel(
                "/quit          — Exit and save memories\n"
                "/memories      — Browse all stored memories\n"
                "/memories dev  — Filter memories by domain\n"
                "/help          — Show this help",
                title="Commands",
                border_style="dim",
            ))
            continue

        # Run the exchange
        transcript.add_user(user_input)
        response_text = run_exchange(user_input, messages, config, memory)
        transcript.add_assistant(response_text)

        # Display response
        console.print()
        try:
            console.print(Markdown(response_text))
        except Exception:
            console.print(response_text)
        console.print()

        # Periodic checkpoint
        if transcript.exchange_count() % config.checkpoint_interval == 0:
            save_checkpoint(transcript, config.checkpoint_path)

    # Clean shutdown
    shutdown(transcript, config, memory)


if __name__ == "__main__":
    main()
