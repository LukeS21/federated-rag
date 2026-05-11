"""Cache subsystem — LLM prompt cache + multi-level query cache.

Exports CACHE_VERSION: bump this constant whenever output format, system
prompts, or processing logic changes in a way that would make previously
cached LLM responses or query results misleading. All cache keys include
this version string, so bumping it transparently invalidates stale entries.
"""

CACHE_VERSION = "v3"  # v3: stronger anti-hallucination constraint in Drafter prompt
