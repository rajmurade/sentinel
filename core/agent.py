"""LLM narration agent.

Takes the output of the detection and drilldown tools and writes a short
incident summary. The agent is a narrator, not an analyst: it may only state
numbers a tool returned, and it falls back to a deterministic template on any
failure or missing API key.

Not implemented yet. See ``CONTEXT.md`` for the contract.
"""
