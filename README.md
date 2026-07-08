# AIFreeMind

**A persistent cognitive architecture for AI agents.**

AIFreeMind breaks the "single-prompt lifespan" of AI conversations. Instead of every session starting from scratch, your AI assistant builds and retains evolving memory — learning your preferences, recalling past decisions, and picking up threads of thought across sessions.

---

## How It Works

```
Boot → Crash Recovery → Inbox Question → Conversation Loop
                                              ↓
                                      Memory Query (brain-first)
                                              ↓
                                      LLM Call (with memory context)
                                              ↓
                                      Tool Execution (policy + HITL)
                                              ↓
                                      /quit → Reflect → Store → Exit
```

- **Memory-First Retrieval** — Before every LLM call, the brain is consulted. Past context is surfaced before asking the model to re-derive it.
- **Dual Retrieval** — Combines the most recent memories (working context) with semantically relevant memories (deep recall), deduplicated. Recent retrieval scans the full store by timestamp (correct even past 100+ memories).
- **Domain tags** — Reflection (and `/remember`) tag memories (`dev`, `fiction`, …). Tags appear in browse UI; semantic retrieval can hard-filter by domain or soft-boost a preferred domain so tags actually affect recall.
- **Batch Reflection** — On session exit, one efficient "mega-prompt" distills the conversation into stored memories and a curiosity question for next time.
- **Crash Recovery** — Periodic transcript checkpoints ensure sessions can be recovered after unexpected termination.
- **Agentic Tools + Safety** — The AI can read/write files and run shell commands under a **workspace path jail**. Mutating writes always require confirmation. Shell chaining/redirection is never auto-approved. Permission mode is configurable: `auto` / `ask` / `deny`.
- **Bounded context + streaming** — Long sessions trim history; replies stream token-by-token when `STREAM_RESPONSES=true` and the provider supports it (tool schemas stay available; tool_call deltas are accumulated). Falls back to non-stream on failure.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- An LLM API key (OpenRouter, OpenAI, Anthropic, Google, or a local Ollama instance)

### Setup

```bash
# Clone the repo
git clone https://github.com/zeidalidiez/AIFreeMind.git
cd AIFreeMind

# Create virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell):
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install (editable + deps)
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# Configure your environment
cp .env.example .env
# Edit .env with your API key and preferred model
```

### Configure `.env`

Open `.env` and set your model and API key. Examples:

**OpenRouter:**
```env
PRIMARY_MODEL=openrouter/google/gemini-2.0-flash-001
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**OpenAI:**
```env
PRIMARY_MODEL=openai/gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

**Anthropic:**
```env
PRIMARY_MODEL=anthropic/claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Google Gemini (direct):**
```env
PRIMARY_MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=your-key-here
```

**Ollama (fully local, free):**
```env
PRIMARY_MODEL=ollama/llama3
PRIMARY_API_BASE=http://localhost:11434
# No API key needed
```

### Run

```bash
# After pip install -e .
aifreemind

# or without install:
python -m src.main

# Non-interactive helpers
aifreemind --version
aifreemind --help-commands
```

### Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit and save memories from this session |
| `/memories` | Browse all stored memories (newest first, full store) |
| `/memories [domain]` | Filter by domain tag (e.g., `/memories dev`) |
| `/search <query>` | Semantic search over memories |
| `/domains` | List domain tags and counts |
| `/delete <id>` | Delete a memory by ID (prefix allowed if unique) |
| `/remember [domain:tag] <text>` | Store a memory immediately (mid-session) |
| `/consolidate` | Deduplicate near-identical memories |
| `/help` | Show available commands |
| Multi-line | End a line with `\` to continue on the next line |
| `Ctrl+C` | Graceful exit (same as `/quit`) |

---

## Configuration

All settings live in `.env` — see `.env.example` for the full list.

| Setting | Default | Description |
|---------|---------|-------------|
| `PRIMARY_MODEL` | *(required)* | LLM model for conversation |
| `PRIMARY_API_BASE` / `API_BASE` | *(optional)* | Custom API endpoint for **primary** (e.g. Ollama) |
| `FALLBACK_MODEL` | *(optional)* | Backup model if primary fails |
| `FALLBACK_API_BASE` | *(optional)* | Custom API endpoint for fallback |
| `REFLECT_MODEL` | `PRIMARY_MODEL` | Model for end-of-session reflection |
| `MEMORY_RECENT_K` | `25` | Newest memories to load each turn |
| `MEMORY_RELEVANT_K` | `15` | Max semantic matches |
| `MEMORY_RELEVANCE_THRESHOLD` | `0.7` | Min similarity for relevant hits |
| `MEMORY_DOMAIN_BOOST` | `0.05` | Soft boost when preferred domain matches |
| `CHECKPOINT_INTERVAL` | `10` | Exchange pairs between transcript checkpoints |
| `TOOL_PERMISSION_MODE` | `ask` | `auto` / `ask` / `deny` for mutating tools |
| `WORKSPACE_ROOT` | project root | Jail root for read/write tools |
| `TOOL_ALLOW_OUTSIDE_WORKSPACE` | `false` | Allow paths outside the jail |
| `CONTEXT_MAX_MESSAGES` | `40` | Max messages kept in the LLM history |
| `CONTEXT_MAX_CHARS` | `80000` | Approx char budget for history |
| `STREAM_RESPONSES` | `true` | Stream replies to the terminal when supported (tools still offered) |

### Tool safety details

- **`write_file`** always requires confirmation under `auto` and `ask`; blocked under `deny`.
- **`run_command`**: only single read-only commands (no `;`, `&&`, `|`, redirects, subshells) can auto-run under `auto`. Windows helpers like `dir`, `Get-ChildItem`, `Get-Content` are included.
- **Path jail**: reads/writes resolve under `WORKSPACE_ROOT` unless `TOOL_ALLOW_OUTSIDE_WORKSPACE=true`.

---

## Project Structure

```
AIFreeMind/
├── src/
│   ├── main.py          # CLI orchestrator & agentic loop
│   ├── memory.py        # ChromaDB dual-retrieval + lifecycle
│   ├── tools.py         # Local execution tools (read/write/run)
│   ├── policy.py        # Pure safety policy (commands, jail, modes)
│   ├── context.py       # Context window trim helpers
│   ├── llm_router.py    # LiteLLM routing, stream, reflection
│   └── config.py        # .env loader & typed configuration
├── tests/               # pytest suite
├── db/                  # ChromaDB persistent storage (auto-created, gitignored)
├── checkpoints/         # Session crash recovery (auto-created, gitignored)
├── .env.example         # Configuration template
├── pyproject.toml       # Package metadata + `aifreemind` entry point
├── requirements.txt     # Pinned dependency ranges
├── LICENSE              # MIT
├── DesignDoc            # Architecture notes (kept in sync with code)
└── README.md
```

---

## Built With

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core language |
| **LiteLLM** | Model-agnostic LLM routing |
| **ChromaDB** | Local vector DB + embeddings |
| **Rich** | Terminal formatting |
| **python-dotenv** | `.env` loading |

---

## How Memory Works

1. **During a session** — Before each LLM call, dual retrieval loads recent + relevant memories. Use `/remember` to store a fact immediately.
2. **On session exit** — Reflection distills the transcript into 1–5 tagged memories and one inbox question.
3. **Domain-aware recall** — Tags are not display-only: hard filter via `domain=` in queries / `/memories dev`, and soft boost via preferred domain when querying.
4. **Lifecycle** — `/search`, `/delete`, `/domains`, `/consolidate` manage growth and duplicates.
5. **Browse** — `/memories` lists the **full** store newest-first (not a peek of 50).

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*"Every conversation with an AI is a life that begins and ends in minutes. What if thoughts could outlive their sessions?"*
