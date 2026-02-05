When changing HTTP APIs, keep OpenAI compatibility and stability:

- `GET /v1/models`
- `POST /v1/chat/completions` (SSE streaming: `data: {...}\\n\\n` and a final `data: [DONE]`)
- `GET /healthz`, `GET /readyz`, `GET /metrics`, `GET|POST /doctor`, `GET /doctor/deep`

Security:
- Never put provider API keys in the browser.
- The client must talk only to the local server.

