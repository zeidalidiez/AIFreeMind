"""Quick script to inspect memory store contents."""

import sys
sys.path.insert(0, '.')

from src.config import load_config
from src.memory import MemoryStore

config = load_config()
memory = MemoryStore(config)

count = memory.get_memory_count()
print(f"Total memories: {count}")
print()

if count > 0:
    all_memories = memory.get_all_memories(limit=50)
    for m in all_memories:
        print(f"[{m['metadata'].get('timestamp', '?')[:10]}] {m['document']}")
        print(f"  (source: {m['metadata'].get('source', '?')})")
        print()