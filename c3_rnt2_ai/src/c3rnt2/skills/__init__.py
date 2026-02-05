"""Agent Skills (prompt-only) for the FastAPI OpenAI-compatible server.

This package implements a local skill store + router that injects small, trusted
"skills" (prompt snippets) into /v1/chat/completions as a system message.

Security model:
- Skills are data only (YAML + Markdown). No executable code is allowed.
- Remote installs are staged, scanned, and require explicit approval.
"""

