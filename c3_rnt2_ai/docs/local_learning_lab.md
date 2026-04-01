# Local Learning Lab

This repo can act as the control plane for a local-first learning stack:

- Ollama on `127.0.0.1:11434`
- Open WebUI on `127.0.0.1:3000`
- Continue in VS Code using the workspace-scoped `.continue/config.yaml`
- `ai-devbox` as a no-network execution sandbox for coding exercises

Preferred runtime is Ollama inside WSL Ubuntu. If Ubuntu exists but does not complete non-interactive first boot yet, `scripts/bootstrap_ollama_windows.ps1` provides a loopback-only Windows fallback so the stack can still be used immediately.

## CLI

- `python -m c3rnt2.cli local-lab init`
- `python -m c3rnt2.cli local-lab status`
- `python -m c3rnt2.cli local-lab next`
- `python -m c3rnt2.cli local-lab roadmap`
- `python -m c3rnt2.cli local-lab bootstrap-plan`
- `python -m c3rnt2.cli local-lab rag-sources`
- `python -m c3rnt2.cli local-lab lesson python-basics`
- `python -m c3rnt2.cli local-lab check --workspace <path>`

## Generated artifacts

- `D:\Vault\learning\ROADMAP.md`
- `D:\Vault\learning\BOOTSTRAP_PLAN.md`
- `D:\Vault\learning\rag_sources.json`
- `D:\Vault\learning\progress.json`

## Bootstrap order

1. Run `python -m c3rnt2.cli local-lab init --profile local_learning_lab_4080`
2. Run `python -m c3rnt2.cli local-lab bootstrap-plan --profile local_learning_lab_4080`
3. Install Ubuntu/Ollama with `scripts/bootstrap_ollama_wsl.ps1`
4. If WSL Ubuntu is blocked on first-run initialization, start the Windows fallback with `scripts/bootstrap_ollama_windows.ps1 -StartOllama`
5. Pull models with `scripts/pull_local_lab_models.ps1`
6. Start Docker Desktop and then `scripts/start_local_stack.ps1`
7. Create or resume the next lesson with `python -m c3rnt2.cli local-lab next`

## Safety model

- Tutor mode explains and evaluates, but does not write.
- Builder mode operates through the sandbox only.
- Security lab mode is defensive and lab-owned only; public and third-party targets are blocked.
