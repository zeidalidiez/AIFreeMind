"""Test memory query with current threshold."""

import sys
sys.path.insert(0, '.')

from src.config import load_config
from src.memory import MemoryStore

config = load_config()
memory = MemoryStore(config)

print(f"Current threshold: {config.memory_relevance_threshold}")
print(f"Current top_k: {config.memory_top_k}")
print()

# Test query - simulating a user asking about past conversations
test_prompts = [
    "what did we work on before?",
    "do you remember our previous conversations?",
    "tell me about the memory system",
]

for prompt in test_prompts:
    print(f"Query: '{prompt}'")
    result = memory.query_memory(prompt)
    if result:
        print(result)
    else:
        print("  (no memories returned)")
    print()