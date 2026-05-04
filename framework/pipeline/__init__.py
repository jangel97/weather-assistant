"""Agent pipeline — orchestration logic for the multi-layer routing system.

This package splits the agent's processing pipeline into focused modules:

- **parsing**: JSON extraction from LLM output
- **messages**: Message list builders for each LLM layer
- **postprocess**: Response cleaning and link fixing
- **guards**: Hallucination detection and correction
- **rewriter**: Follow-up question rewriting
- **routing**: The main routing loop and tool execution
"""
