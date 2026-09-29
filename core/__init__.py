"""Core pipeline modules for Sentinel.

Each module is independently importable and independently skippable: a missing
LLM key or webhook must degrade a run, never fail it.
"""
