# Vortex API + Vortex UI (local)

Monorepo con:
- `c3_rnt2_ai/`: backend **Vortex** (FastAPI) con endpoints estilo OpenAI (`/v1/*`) y streaming SSE.
- `frontend/`: UI **Vortex** (Vite + React) que **solo** llama al backend local (nunca a proveedores LLM desde el navegador).

## One-command (Windows)

Arranca backend + frontend:

```powershell
.\dev.ps1
```

## Backend (Vortex)

```bash
cd c3_rnt2_ai
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows (PowerShell): .venv\\Scripts\\Activate.ps1
pip install -e .[api]

# opcional (AUTH dev)
# Windows (PowerShell): $env:VORTEX_API_TOKEN=\"devtoken\"
# Linux/macOS: export VORTEX_API_TOKEN=devtoken
# compat:
# Windows (PowerShell): $env:KLIMEAI_API_TOKEN=\"devtoken\"
# Linux/macOS: export KLIMEAI_API_TOKEN=devtoken

vortex serve --host 0.0.0.0 --port 8000
# compat: klimeai serve --host 0.0.0.0 --port 8000
```

Validación rápida:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"core","messages":[{"role":"user","content":"hola"}]}'
```

Si `VORTEX_API_TOKEN` está definido, añade:
`-H "Authorization: Bearer devtoken"` a las llamadas `curl` a `/v1/*`.

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"core","messages":[{"role":"user","content":"hola"}],"stream":true}'
```

Smoke test:

```bash
pytest -q
python scripts/smoke_api.py --base-url http://localhost:8000
klimeai skills validate --all
klimeai doctor --deep --mock
```

Endpoints soportados:
- `GET /healthz`
- `GET /readyz`
- `GET /v1/models`
- `GET /v1/skills`, `POST /v1/skills/stage`, `POST /v1/skills/approve`
- `POST /v1/chat/completions` (stream + no-stream)
- `GET /metrics`
- `GET|POST /doctor`, `GET /doctor/deep`

## Frontend (Vortex UI)

```bash
cd frontend
npm i
cp .env.example .env
npm run dev
```

Abrir `http://localhost:3000`.
