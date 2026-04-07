"""Test memory query with lower threshold."""

import sys
sys.path.insert(0, '.')

from src.config import load_config
from src.memory import MemoryStore

config = load_config()
memory = MemoryStore(config)

# Test with different thresholds
test_prompts = [
    "what did we work on before?",
    "do you remember our previous conversations?",
]

for prompt in test_prompts:
    print(f"Query: '{prompt}'")
    for threshold in [0.4, 0.3, 0.2, 0.1]:
        result = memory.query_memory(prompt, threshold=threshold, k=10)
        if result:
            print(f"  threshold={threshold}: FOUND")
            print(result[:200] + "..." if len(result) > 200 else result)
        else:
            print(f"  threshold={threshold}: nothing")
    print()