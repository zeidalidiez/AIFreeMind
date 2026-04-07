# 🧠 AIFreeMind

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
                                      Tool Execution (if needed)
                                              ↓
                                      /quit → Reflect → Store → Exit
```

- **Memory-First Retrieval** — Before every LLM call, the brain is consulted. Past solutions and context are surfaced before asking the model to re-derive them.
- **Dual Retrieval** — Combines the 25 most recent memories (working context) with up to 15 semantically relevant memories (deep recall), deduplicated.
- **Batch Reflection** — On session exit, one efficient "mega-prompt" distills the conversation into stored memories and a curiosity question for next time.
- **Crash Recovery** — Periodic transcript checkpoints ensure no session is lost, even on unexpected termination.
- **Agentic Tools** — The AI can read/write files and run shell commands on your local machine. Read-only commands (like `ls`, `git status`) auto-execute, while mutating commands (like `rm`, `pip install`) pause for strict **human-in-the-loop** confirmation to prevent destructive actions.

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

# Install dependencies
pip install -r requirements.txt

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
# No API key needed
```

### Run

```bash
python -m src.main
```

### Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit and save memories from this session |
| `/memories` | Browse all stored memories |
| `/memories [domain]`| Filter stored memories by a specific domain tag (e.g., `/memories dev`) |
| `/help` | Show available commands |
| `Ctrl+C` | Graceful exit (same as `/quit`) |

---

## Configuration

All settings live in `.env` — no hardcoded values in source code.

| Setting | Default | Description |
|---------|---------|-------------|
| `PRIMARY_MODEL` | *(required)* | LLM model for conversation ([LiteLLM model list](https://docs.litellm.ai/docs/providers)) |
| `FALLBACK_MODEL` | *(optional)* | Backup model if primary fails |
| `FALLBACK_API_BASE` | *(optional)* | Custom API endpoint for fallback (e.g., Ollama URL) |
| `REFLECT_MODEL` | `PRIMARY_MODEL` | Model for end-of-session reflection (can be cheaper) |
| `MEMORY_RECENT_K` | `25` | Number of most recent memories to load |
| `MEMORY_RELEVANT_K` | `15` | Max semantically relevant memories to load |
| `MEMORY_RELEVANCE_THRESHOLD` | `0.7` | Minimum similarity (0.0–1.0) for relevant memory retrieval |
| `CHECKPOINT_INTERVAL` | `10` | Exchange pairs between transcript checkpoints |

---

## Project Structure

```
AIFreeMind/
├── src/
│   ├── main.py          # CLI orchestrator & agentic loop
│   ├── memory.py         # ChromaDB dual-retrieval memory system
│   ├── tools.py          # Local execution tools (read/write/run)
│   ├── llm_router.py     # LiteLLM model routing & batch reflection
│   └── config.py         # .env loader & typed configuration
├── db/                   # ChromaDB persistent storage (auto-created)
├── checkpoints/          # Session crash recovery (auto-created)
├── .env                  # Your configuration (git-ignored)
├── .env.example          # Configuration template
├── requirements.txt      # Python dependencies
└── DesignDoc             # Original design document
```

---

## Built With

| Technology | Purpose | Link |
|------------|---------|------|
| **Python 3.10+** | Core language | [python.org](https://python.org) |
| **LiteLLM** | Model-agnostic LLM routing — talk to any provider (OpenAI, Anthropic, Google, Ollama, OpenRouter, etc.) through a single interface | [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm) |
| **ChromaDB** | Local vector database for persistent semantic memory with built-in embeddings | [trychroma.com](https://www.trychroma.com/) |
| **all-MiniLM-L6-v2** | Sentence transformer model used by ChromaDB for local embedding generation (no API calls) | [huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Rich** | Beautiful terminal formatting, panels, and markdown rendering | [github.com/Textualize/rich](https://github.com/Textualize/rich) |
| **python-dotenv** | Environment variable management from `.env` files | [pypi.org/project/python-dotenv](https://pypi.org/project/python-dotenv/) |

### Compatible LLM Providers

AIFreeMind works with any provider supported by LiteLLM, including:

- [OpenRouter](https://openrouter.ai) — Multi-provider gateway
- [OpenAI](https://openai.com) — GPT-4o, GPT-4o-mini
- [Anthropic](https://anthropic.com) — Claude Sonnet, Opus, Haiku
- [Google](https://ai.google.dev) — Gemini Flash, Pro
- [Ollama](https://ollama.ai) — Run models locally (Llama, Mistral, etc.)

---

## How Memory Works (Configurable in .env)

1. **During a session** — You converse normally. Before each LLM call, the brain surfaces relevant past context automatically.

2. **On session exit** — A single reflection prompt distills the conversation into 1–5 concise memories and one follow-up question. During this step, the AI automatically assigns a domain tag (e.g., `dev`, `fiction`, `personal`) to each memory.

3. **On next session boot** — The follow-up question is displayed as a greeting. As you converse, past memories are retrieved through two channels:
   - **Recent** (25 most recent memories) — your short-term working context
   - **Relevant** (up to 15 semantically similar) — long-term topical recall

   *Note on Memory Namespacing:* The domain tags generated during reflection help categorize concepts over time, minimizing cross-contamination (e.g. keeping D&D lore separated from Python dev context).

4. **Over time** — The brain accumulates understanding of your projects, preferences, and ongoing work. The AI's context becomes richer with every session.

---

## License

MIT

---

*"Every conversation with an AI is a life that begins and ends in minutes. What if thoughts could outlive their sessions?"*
