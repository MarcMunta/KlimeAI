from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .autopilot import run_autopilot_tick
from .config import load_settings
from .instructions import load_instruction_bundle
from .model_init import DEFAULT_MODEL_ID, model_cache_status, resolve_cache_dir


DEFAULT_CONTROL_PORT = 8765
DEFAULT_FRONTEND_PORT = 4173
DEFAULT_API_PORT = 8000
DEFAULT_RUNTIME_PORT = 30000
DEFAULT_API_PROFILE = "rtx4080_16gb_programming_local"
DEFAULT_TRAINING_PROFILE = "rtx4080_16gb_programming_train_docker"


def _utc_ts() -> float:
    return float(time.time())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported_type:{type(value)!r}")


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _tail(path: Path, lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:]
    except Exception:
        return []


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _http_json(url: str, *, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, timeout=float(timeout))
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _parse_structured_output(raw: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        try:
            payload = ast.literal_eval(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return None


def _iter_hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    seen = 0
    for root in paths:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in sorted(root.rglob("*")) if p.is_file()]
        for path in files:
            try:
                digest.update(
                    str(path.relative_to(root.parent if root.parent.exists() else root)).encode(
                        "utf-8",
                        errors="ignore",
                    )
                )
            except Exception:
                digest.update(str(path).encode("utf-8", errors="ignore"))
            try:
                digest.update(path.read_bytes())
                seen += 1
            except Exception:
                continue
    return f"sha256:{digest.hexdigest()}" if seen else "sha256:empty"


class BootstrapRequest(BaseModel):
    force: bool = False


class TrainingStartRequest(BaseModel):
    mode: str = Field(default="quick")


class AllowlistRequest(BaseModel):
    domains: list[str] = Field(default_factory=list)


class AutonomyConfigRequest(BaseModel):
    enabled: bool | None = None
    reflection_enabled: bool | None = None
    training_enabled: bool | None = None
    autoedit_enabled: bool | None = None


class ControlState:
    def __init__(
        self,
        *,
        base_dir: Path,
        compose_file: Path,
        api_profile: str,
        training_profile: str,
        api_url: str,
        runtime_url: str,
        frontend_port: int,
    ) -> None:
        self.base_dir = base_dir
        self.compose_file = compose_file
        self.api_profile = api_profile
        self.training_profile = training_profile
        self.api_url = api_url.rstrip("/")
        self.runtime_url = runtime_url.rstrip("/")
        self.frontend_port = int(frontend_port)

        self.control_dir = self.base_dir / "data" / "control"
        self.bootstrap_state_path = self.control_dir / "bootstrap_state.json"
        self.runs_dir = self.control_dir / "training_runs"
        self.internet_settings_path = self.control_dir / "internet_settings.json"
        self.autonomy_state_path = self.control_dir / "autonomy_state.json"
        self.autonomy_events_path = self.control_dir / "autonomy_events.jsonl"
        self.log_dir = self.base_dir.parent / "logs"

        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._bootstrap_thread: threading.Thread | None = None
        self._training_thread: threading.Thread | None = None
        self._autonomy_thread: threading.Thread | None = None
        self._autonomy_stop = threading.Event()
        self._active_run_id: str | None = None

        if not self.bootstrap_state_path.exists():
            self._set_bootstrap_state(
                {
                    "running": False,
                    "stage": "idle",
                    "message": "control_ready",
                    "updated_at": _utc_ts(),
                }
            )
        if not self.internet_settings_path.exists():
            _write_json(self.internet_settings_path, {"domains": []})
        if not self.autonomy_state_path.exists():
            _write_json(self.autonomy_state_path, self._default_autonomy_state())

        self._ensure_autonomy_worker()

    def _set_bootstrap_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            current = _load_json(self.bootstrap_state_path, {})
            current.update(payload)
            current["updated_at"] = _utc_ts()
            _write_json(self.bootstrap_state_path, current)

    def _compose_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("VORTEX_API_PROFILE", self.api_profile)
        if extra:
            env.update(extra)
        return env

    def _run_compose(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
    ) -> tuple[int, str]:
        cmd = ["docker", "compose", "-f", str(self.compose_file), *args]
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.base_dir),
            env=self._compose_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        lines: list[str] = []
        sink = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = log_path.open("a", encoding="utf-8")
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line.rstrip())
                if sink is not None:
                    sink.write(line)
                    sink.flush()
        finally:
            if sink is not None:
                sink.close()
        return int(proc.wait()), "\n".join(lines)

    def _wait_runtime_ready(self, timeout_s: float = 240.0) -> bool:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            ready = _http_json(f"{self.api_url}/readyz", timeout=2.0)
            if ready and bool(ready.get("ok")):
                return True
            time.sleep(2.0)
        return False

    def _resolve_dataset_hash(self) -> str:
        candidates = [
            self.base_dir / "data" / "registry" / "hf_train",
            self.base_dir / "data" / "episodes",
            self.base_dir / "data" / "corpora" / "programming",
            self.base_dir / "data" / "corpora" / "cybersecurity",
            self.base_dir / "data" / "local_lab" / "lessons",
        ]
        return _iter_hash_files(candidates)

    def _resolve_instruction_meta(self) -> dict[str, Any]:
        try:
            bundle = load_instruction_bundle({}, base_dir=self.base_dir)
        except Exception as exc:
            return {"digest": None, "sources": [], "error": str(exc)}

        if not bundle.get("sources") or bundle.get("sources", [{}])[0].get("kind") == "inline_fallback":
            fallback_sources = []
            for name in ("vortex_system.md", "domain_policy.md", "operator_notes.md"):
                candidate = self.base_dir / "config" / "instructions" / name
                if not candidate.exists():
                    continue
                text = candidate.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                fallback_sources.append(
                    {
                        "kind": name.replace(".md", ""),
                        "path": str(candidate),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                )
            if fallback_sources:
                joined = "\n\n".join(
                    Path(item["path"]).read_text(encoding="utf-8").strip()
                    for item in fallback_sources
                ).strip()
                bundle = {
                    "text": joined,
                    "digest": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
                    "sources": fallback_sources,
                }

        text = str(bundle.get("text") or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
        return {
            "digest": digest,
            "sources": [str(item) for item in (bundle.get("sources") or [])],
        }

    def docker_status(self) -> dict[str, Any]:
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .ServerVersion}}"],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=4.0,
                check=False,
            )
        except FileNotFoundError:
            return {"ready": False, "reason": "docker_not_installed"}
        except Exception as exc:
            return {"ready": False, "reason": f"docker_unavailable:{exc}"}
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            return {"ready": False, "reason": "docker_unavailable", "detail": detail}
        version = (result.stdout or "").strip().strip('"')
        return {"ready": True, "reason": "docker_ready", "server_version": version or None}

    def model_status(self) -> dict[str, Any]:
        cache_dir = resolve_cache_dir(self.base_dir / "data" / "models" / "hf-cache")
        return model_cache_status(DEFAULT_MODEL_ID, cache_dir)

    def runtime_status(self) -> dict[str, Any]:
        ready = _http_json(f"{self.api_url}/readyz", timeout=1.5)
        status = _http_json(f"{self.api_url}/v1/status", timeout=1.5)
        runtime_models = _http_json(f"{self.runtime_url}/v1/models", timeout=1.5)
        return {
            "api_ready": bool(ready and ready.get("ok")),
            "readyz": ready,
            "status": status,
            "runtime_ready": runtime_models is not None,
            "runtime_models": runtime_models,
        }

    def frontend_status(self) -> dict[str, Any]:
        return {
            "ready": _port_open("127.0.0.1", self.frontend_port),
            "port": self.frontend_port,
            "url": f"http://127.0.0.1:{self.frontend_port}",
        }

    def get_allowlist(self) -> list[str]:
        payload = _load_json(self.internet_settings_path, {"domains": []})
        raw = payload.get("domains", []) if isinstance(payload, dict) else []
        items = []
        for item in raw if isinstance(raw, list) else []:
            text = str(item or "").strip().lower()
            if text:
                items.append(text)
        return sorted(set(items))

    def set_allowlist(self, domains: list[str]) -> list[str]:
        cleaned = sorted(
            {
                str(item or "").strip().lower()
                for item in domains
                if str(item or "").strip()
            }
        )
        _write_json(self.internet_settings_path, {"domains": cleaned, "updated_at": _utc_ts()})
        return cleaned

    def _default_autonomy_state(self) -> dict[str, Any]:
        now = _utc_ts()
        return {
            "enabled": True,
            "boot_mode": "always_on",
            "state": "waiting_resources",
            "active_agents": [
                {
                    "id": "analyst",
                    "name": "Analista",
                    "role": "reflection",
                    "status": "waiting",
                    "accent": "ask",
                    "last_event_at": now,
                },
                {
                    "id": "builder",
                    "name": "Constructor",
                    "role": "execution",
                    "status": "waiting",
                    "accent": "agent",
                    "last_event_at": now,
                },
            ],
            "current_cycle": None,
            "last_reflection_at": None,
            "last_train_at": None,
            "last_patch_at": None,
            "autoedit_scope": "repo_versioned",
            "last_rollback": None,
            "config": {
                "reflection_enabled": True,
                "training_enabled": True,
                "autoedit_enabled": True,
                "reflection_interval_s": 300,
                "quick_train_interval_s": 1200,
                "full_train_interval_s": 7200,
                "autoedit_interval_s": 1800,
            },
            "latest_events": [],
            "updated_at": now,
        }

    def _load_autonomy_state(self) -> dict[str, Any]:
        current = _load_json(self.autonomy_state_path, self._default_autonomy_state())
        merged = self._default_autonomy_state()
        if isinstance(current, dict):
            merged.update(current)
            merged["config"] = {
                **self._default_autonomy_state()["config"],
                **(current.get("config") if isinstance(current.get("config"), dict) else {}),
            }
            merged["active_agents"] = current.get("active_agents") or merged["active_agents"]
            merged["latest_events"] = current.get("latest_events") or []
        return merged

    def _write_autonomy_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self._load_autonomy_state()
        state.update(payload)
        if "config" in payload and isinstance(payload["config"], dict):
            state["config"] = {**state.get("config", {}), **payload["config"]}
        if "active_agents" in payload:
            state["active_agents"] = payload["active_agents"]
        if "latest_events" in payload:
            state["latest_events"] = payload["latest_events"][:80]
        state["updated_at"] = _utc_ts()
        _write_json(self.autonomy_state_path, state)
        return state

    def _latest_autonomy_events(self, limit: int = 24) -> list[dict[str, Any]]:
        if not self.autonomy_events_path.exists():
            return []
        try:
            events = [
                json.loads(line)
                for line in self.autonomy_events_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
        except Exception:
            return []
        trimmed = [event for event in events if isinstance(event, dict)][-limit:]
        return sorted(trimmed, key=lambda item: float(item.get("ts") or 0.0), reverse=True)

    def _append_autonomy_event(
        self,
        *,
        agent: str,
        kind: str,
        title: str,
        detail: str,
        cycle_id: str | None = None,
        state_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}",
            "ts": _utc_ts(),
            "agent": agent,
            "kind": kind,
            "title": title,
            "detail": detail,
            "cycle_id": cycle_id,
            "state": state_name,
            "metadata": metadata or {},
        }
        self.autonomy_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.autonomy_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=_json_default) + "\n")
        events = [payload, *self._latest_autonomy_events(limit=23)]
        self._write_autonomy_state({"latest_events": events})
        return payload

    def _agent_state(self, autonomy: dict[str, Any], *, analyst: str, builder: str) -> list[dict[str, Any]]:
        now = _utc_ts()
        return [
            {
                "id": "analyst",
                "name": "Analista",
                "role": "reflection",
                "status": analyst,
                "accent": "ask",
                "last_event_at": autonomy.get("last_reflection_at") or now,
            },
            {
                "id": "builder",
                "name": "Constructor",
                "role": "execution",
                "status": builder,
                "accent": "agent",
                "last_event_at": max(
                    float(autonomy.get("last_train_at") or 0.0),
                    float(autonomy.get("last_patch_at") or 0.0),
                    now,
                ),
            },
        ]

    def _ensure_autonomy_worker(self) -> None:
        with self._lock:
            if self._autonomy_thread and self._autonomy_thread.is_alive():
                return
            self._autonomy_stop.clear()
            self._autonomy_thread = threading.Thread(target=self._autonomy_worker, daemon=True)
            self._autonomy_thread.start()

    def autonomy_status(self) -> dict[str, Any]:
        state = self._load_autonomy_state()
        state["latest_events"] = state.get("latest_events") or self._latest_autonomy_events(limit=20)
        return state

    def start_autonomy(self) -> dict[str, Any]:
        state = self._write_autonomy_state({"enabled": True, "state": "waiting_resources"})
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_start",
            title="Autonomia activada",
            detail="El bucle continuo vuelve a vigilar runtime, aprendizaje, entrenamiento y autoedicion.",
            state_name=state.get("state"),
        )
        self._ensure_autonomy_worker()
        return {"ok": True, "enabled": True, "autonomy": self.autonomy_status()}

    def stop_autonomy(self) -> dict[str, Any]:
        state = self._write_autonomy_state({"enabled": False, "state": "paused"})
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_stop",
            title="Autonomia en pausa",
            detail="Se ha detenido la reflexion continua y no se lanzaran nuevos ciclos autonomos hasta reactivarla.",
            state_name=state.get("state"),
        )
        return {"ok": True, "enabled": False, "autonomy": self.autonomy_status()}

    def configure_autonomy(self, payload: AutonomyConfigRequest) -> dict[str, Any]:
        patch: dict[str, Any] = {"config": {}}
        if payload.enabled is not None:
            patch["enabled"] = bool(payload.enabled)
            patch["state"] = "waiting_resources" if payload.enabled else "paused"
        if payload.reflection_enabled is not None:
            patch["config"]["reflection_enabled"] = bool(payload.reflection_enabled)
        if payload.training_enabled is not None:
            patch["config"]["training_enabled"] = bool(payload.training_enabled)
        if payload.autoedit_enabled is not None:
            patch["config"]["autoedit_enabled"] = bool(payload.autoedit_enabled)
        state = self._write_autonomy_state(patch)
        self._append_autonomy_event(
            agent="system",
            kind="autonomy_config",
            title="Configuracion de autonomia actualizada",
            detail="Se han aplicado nuevos interruptores para reflexion, entrenamiento o autoedicion.",
            state_name=state.get("state"),
            metadata=patch["config"],
        )
        if state.get("enabled"):
            self._ensure_autonomy_worker()
        return {"ok": True, "autonomy": self.autonomy_status()}

    def _git_clean(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0 and not (result.stdout or "").strip()

    def _git_head(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        head = (result.stdout or "").strip()
        return head or None

    def _git_tag_snapshot(self, name: str) -> str | None:
        try:
            subprocess.run(
                ["git", "tag", "-f", name],
                cwd=str(self.base_dir.parent),
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except Exception:
            return None
        return name

    def _autonomy_settings(self) -> dict[str, Any]:
        settings = load_settings(
            profile=self.api_profile,
            settings_path=self.base_dir / "config" / "settings.yaml",
        )
        settings = json.loads(json.dumps(settings))
        settings.setdefault("continuous", {})["ingest_web"] = False
        settings.setdefault("autopilot", {})
        settings["autopilot"].update(
            {
                "enabled": True,
                "reuse_dataset": True,
                "autopatch_enabled": True,
                "autopatch_require_approval": False,
                "autopatch_goal": "continuous self-improvement",
                "autopatch_on_test_fail": True,
                "autopatch_on_doctor_fail": True,
                "autopatch_require_eval": True,
                "train_cooldown_minutes": 20,
                "patch_cooldown_minutes": 30,
                "eval_cooldown_minutes": 20,
                "safe_mode_cooldown_minutes": 30,
            }
        )
        settings.setdefault("self_patch", {})
        allowed_paths = [
            "c3_rnt2_ai/",
            "vortex-chat/",
            "scripts/",
            "docs/",
            "README.md",
            "docker-compose.yml",
        ]
        settings["self_patch"]["enabled"] = True
        settings["self_patch"]["allowed_paths"] = allowed_paths
        forbidden = list(settings["self_patch"].get("forbidden_globs", []))
        forbidden.extend(
            [
                ".git/**",
                "data/control/**",
                "data/models/**",
                "data/registry/**",
                "**/__pycache__/**",
                "logs/**",
            ]
        )
        settings["self_patch"]["forbidden_globs"] = sorted(set(forbidden))
        return settings

    def _run_autonomy_autoedit(self, cycle_id: str) -> dict[str, Any]:
        if not self._git_clean():
            detail = "workspace_dirty"
            self._append_autonomy_event(
                agent="system",
                kind="autoedit_skip",
                title="Autoedicion aplazada",
                detail="El repo tiene cambios sin confirmar. La autoedicion queda en espera para no pisar trabajo manual.",
                cycle_id=cycle_id,
                state_name="waiting_resources",
                metadata={"reason": detail},
            )
            return {"ok": True, "skipped": detail}

        snapshot = self._git_head()
        snapshot_tag = None
        if snapshot:
            snapshot_tag = self._git_tag_snapshot(f"autonomy/snapshot-{time.strftime('%Y%m%d_%H%M%S')}")

        settings = self._autonomy_settings()
        result = run_autopilot_tick(settings, self.base_dir, no_web=True, mock=True, force=True)
        patch_info = result.steps.get("autopatch") if isinstance(result.steps, dict) else {}
        ok_patch = isinstance(patch_info, dict) and bool(patch_info.get("ok", False))
        promoted = isinstance(patch_info, dict) and bool(patch_info.get("promoted", False))
        rollback_status = None

        if ok_patch and promoted:
            self._append_autonomy_event(
                agent="builder",
                kind="autoedit_applied",
                title="Autoedicion promovida",
                detail="El constructor aplico cambios, ejecuto validaciones y dejo el repo en un estado promovido con snapshot previo.",
                cycle_id=cycle_id,
                state_name="autoediting",
                metadata={"branch": patch_info.get("branch"), "snapshot": snapshot_tag or snapshot},
            )
        elif isinstance(patch_info, dict) and patch_info.get("skipped"):
            self._append_autonomy_event(
                agent="builder",
                kind="autoedit_skipped",
                title="Autoedicion sin cambios",
                detail=f"No se aplicaron cambios: {patch_info.get('skipped')}.",
                cycle_id=cycle_id,
                state_name="autoediting",
                metadata={"snapshot": snapshot_tag or snapshot},
            )
        else:
            rollback_status = {
                "ts": _utc_ts(),
                "status": "rollback_ok" if self._git_clean() else "rollback_failed",
                "target": snapshot_tag or snapshot,
                "reason": (patch_info.get("error") if isinstance(patch_info, dict) else result.error) or "autopatch_failed",
            }
            self._append_autonomy_event(
                agent="system",
                kind="rollback",
                title="Rollback autonomo",
                detail="La autoedicion no paso las validaciones y el sistema ha vuelto al snapshot anterior o ha preservado el repo sin aplicar cambios inestables.",
                cycle_id=cycle_id,
                state_name="rollback",
                metadata=rollback_status,
            )
        return {
            "ok": result.ok,
            "steps": result.steps,
            "error": result.error,
            "rollback": rollback_status,
        }

    def _autonomy_worker(self) -> None:
        while not self._autonomy_stop.is_set():
            try:
                autonomy = self._load_autonomy_state()
                config = autonomy.get("config", {}) if isinstance(autonomy.get("config"), dict) else {}
                now = _utc_ts()
                runtime = self.runtime_status()
                training_busy = bool(self._training_thread and self._training_thread.is_alive())

                if not autonomy.get("enabled", True):
                    self._write_autonomy_state(
                        {
                            "state": "paused",
                            "active_agents": self._agent_state(autonomy, analyst="paused", builder="paused"),
                        }
                    )
                    time.sleep(2.0)
                    continue

                if not runtime.get("api_ready") or not runtime.get("runtime_ready"):
                    self._write_autonomy_state(
                        {
                            "state": "waiting_resources",
                            "active_agents": self._agent_state(autonomy, analyst="waiting", builder="waiting"),
                        }
                    )
                    time.sleep(4.0)
                    continue

                if training_busy:
                    self._write_autonomy_state(
                        {
                            "state": "training",
                            "active_agents": self._agent_state(autonomy, analyst="observing", builder="training"),
                        }
                    )
                    time.sleep(3.0)
                    continue

                reflection_due = (
                    bool(config.get("reflection_enabled", True))
                    and (now - float(autonomy.get("last_reflection_at") or 0.0)) >= float(config.get("reflection_interval_s", 300))
                )
                full_train_due = (
                    bool(config.get("training_enabled", True))
                    and (now - float(autonomy.get("last_train_at") or 0.0)) >= float(config.get("full_train_interval_s", 7200))
                )
                quick_train_due = (
                    bool(config.get("training_enabled", True))
                    and (now - float(autonomy.get("last_train_at") or 0.0)) >= float(config.get("quick_train_interval_s", 1200))
                )
                autoedit_due = (
                    bool(config.get("autoedit_enabled", True))
                    and (now - float(autonomy.get("last_patch_at") or 0.0)) >= float(config.get("autoedit_interval_s", 1800))
                )

                cycle_id = autonomy.get("current_cycle") or f"cycle-{time.strftime('%Y%m%d-%H%M%S')}"

                if reflection_due:
                    self._write_autonomy_state(
                        {
                            "state": "learning",
                            "current_cycle": cycle_id,
                            "last_reflection_at": now,
                            "active_agents": self._agent_state(autonomy, analyst="reflecting", builder="planning"),
                        }
                    )
                    self._append_autonomy_event(
                        agent="analyst",
                        kind="reflection",
                        title="Analista revisa sesiones y gaps",
                        detail="Cruza conversaciones, errores recientes, cobertura del repo y runs anteriores para detectar la siguiente mejora con mayor retorno.",
                        cycle_id=cycle_id,
                        state_name="learning",
                    )
                    self._append_autonomy_event(
                        agent="builder",
                        kind="hypothesis",
                        title="Constructor propone siguiente ciclo",
                        detail="Prepara hipotesis para quick learning, entrenamiento completo o autoedicion segun el estado del runtime y el valor esperado.",
                        cycle_id=cycle_id,
                        state_name="learning",
                    )

                if full_train_due:
                    self._append_autonomy_event(
                        agent="builder",
                        kind="train_full",
                        title="Entrenamiento completo en cola",
                        detail="Se ha lanzado un entrenamiento completo para consolidar mejoras acumuladas y medirlas contra bench/eval.",
                        cycle_id=cycle_id,
                        state_name="training",
                    )
                    result = self.start_training("full")
                    if result.get("ok"):
                        self._write_autonomy_state(
                            {
                                "state": "training",
                                "current_cycle": cycle_id,
                                "last_train_at": now,
                                "active_agents": self._agent_state(autonomy, analyst="observing", builder="training"),
                            }
                        )
                elif quick_train_due:
                    self._append_autonomy_event(
                        agent="builder",
                        kind="train_quick",
                        title="Aprendizaje rapido lanzado",
                        detail="Se ha lanzado un ciclo rapido para reusar dataset reciente, validar un adapter incremental y dejarlo listo para revision.",
                        cycle_id=cycle_id,
                        state_name="training",
                    )
                    result = self.start_training("quick")
                    if result.get("ok"):
                        self._write_autonomy_state(
                            {
                                "state": "training",
                                "current_cycle": cycle_id,
                                "last_train_at": now,
                                "active_agents": self._agent_state(autonomy, analyst="observing", builder="training"),
                            }
                        )

                if autoedit_due and not (self._training_thread and self._training_thread.is_alive()):
                    self._write_autonomy_state(
                        {
                            "state": "autoediting",
                            "current_cycle": cycle_id,
                            "last_patch_at": now,
                            "active_agents": self._agent_state(autonomy, analyst="reviewing", builder="patching"),
                        }
                    )
                    self._append_autonomy_event(
                        agent="builder",
                        kind="autoedit_start",
                        title="Autoedicion del repo",
                        detail="El constructor abre un snapshot del repo versionado y ejecuta una ronda de autoedicion con tests, doctor y rollback si algo sale mal.",
                        cycle_id=cycle_id,
                        state_name="autoediting",
                    )
                    autoedit = self._run_autonomy_autoedit(cycle_id)
                    patch_state = "rollback" if autoedit.get("rollback") else "learning"
                    self._write_autonomy_state(
                        {
                            "state": patch_state,
                            "last_rollback": autoedit.get("rollback"),
                            "active_agents": self._agent_state(
                                autonomy,
                                analyst="reviewing" if patch_state != "rollback" else "stabilizing",
                                builder="ready" if patch_state != "rollback" else "rollback",
                            ),
                        }
                    )

                if not reflection_due and not quick_train_due and not full_train_due and not autoedit_due:
                    self._write_autonomy_state(
                        {
                            "state": "learning",
                            "active_agents": self._agent_state(autonomy, analyst="monitoring", builder="ready"),
                        }
                    )
                time.sleep(2.0)
            except Exception as exc:
                self._write_autonomy_state({"state": "waiting_resources"})
                self._append_autonomy_event(
                    agent="system",
                    kind="autonomy_error",
                    title="Autonomia en espera",
                    detail=f"El bucle continuo ha detectado un error recuperable: {exc}",
                    state_name="waiting_resources",
                )
                time.sleep(5.0)

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("*/meta.json"), reverse=True):
            payload = _load_json(path, {})
            if isinstance(payload, dict):
                runs.append(payload)
        runs.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0), reverse=True)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        meta_path = self.runs_dir / run_id / "meta.json"
        if not meta_path.exists():
            return None
        payload = _load_json(meta_path, {})
        return payload if isinstance(payload, dict) else None

    def status(self) -> dict[str, Any]:
        bootstrap = _load_json(self.bootstrap_state_path, {})
        docker = self.docker_status()
        model = self.model_status()
        runtime = self.runtime_status()

        runtime_models = runtime.get("runtime_models") if isinstance(runtime, dict) else None
        runtime_model_ids: list[str] = []
        if isinstance(runtime_models, dict):
            data = runtime_models.get("data")
            if isinstance(data, list):
                runtime_model_ids = [
                    str(item.get("id") or "").strip()
                    for item in data
                    if isinstance(item, dict) and str(item.get("id") or "").strip()
                ]

        if runtime.get("api_ready") and runtime.get("runtime_ready") and bootstrap.get("stage") != "ready":
            bootstrap = {
                **bootstrap,
                "running": False,
                "stage": "ready",
                "message": "stack_ready",
            }
            _write_json(self.bootstrap_state_path, bootstrap)

        if runtime_model_ids and not bool(model.get("cached")):
            model = {
                **model,
                "cached": True,
                "snapshot_count": max(int(model.get("snapshot_count") or 0), len(runtime_model_ids)),
                "last_snapshot": model.get("last_snapshot") or runtime_model_ids[0],
            }

        return {
            "ok": True,
            "bootstrap": bootstrap,
            "docker": docker,
            "model": model,
            "runtime": runtime,
            "frontend": self.frontend_status(),
            "internet": {"allowlist": self.get_allowlist()},
            "instructions": self._resolve_instruction_meta(),
            "autonomy": self.autonomy_status(),
            "active_run_id": self._active_run_id,
            "runs": self.list_runs()[:12],
        }

    def start_bootstrap(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._bootstrap_thread and self._bootstrap_thread.is_alive() and not force:
                return {"ok": True, "started": False, "reason": "bootstrap_already_running"}
            thread = threading.Thread(target=self._bootstrap_worker, daemon=True)
            self._bootstrap_thread = thread
            thread.start()
        return {"ok": True, "started": True}

    def _bootstrap_worker(self) -> None:
        log_path = self.log_dir / "control-bootstrap.log"
        self._set_bootstrap_state({"running": True, "stage": "docker", "message": "checking_docker", "log_path": str(log_path)})

        docker = self.docker_status()
        if not docker.get("ready"):
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": docker.get("reason"), "error": docker})
            return

        runtime = self.runtime_status()
        if runtime.get("api_ready") and runtime.get("runtime_ready"):
            self._set_bootstrap_state({"running": False, "stage": "ready", "message": "stack_ready", "tail": _tail(log_path)})
            return

        code, _ = self._run_compose(["pull", "sglang-runtime"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": "image_pull_failed", "tail": _tail(log_path)})
            return

        self._set_bootstrap_state({"stage": "build", "message": "building_backend_images"})
        code, _ = self._run_compose(["build", "model-init", "vortex-api", "trainer", "eval"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": "build_failed", "tail": _tail(log_path)})
            return

        self._set_bootstrap_state({"stage": "model-init", "message": "ensuring_local_model"})
        code, _ = self._run_compose(["run", "--rm", "model-init"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": "model_init_failed", "tail": _tail(log_path)})
            return

        self._set_bootstrap_state({"stage": "runtime", "message": "starting_runtime"})
        code, _ = self._run_compose(["up", "-d", "sglang-runtime", "vortex-api"], log_path=log_path)
        if code != 0:
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": "runtime_start_failed", "tail": _tail(log_path)})
            return

        self._set_bootstrap_state({"stage": "waiting", "message": "waiting_for_readyz"})
        if not self._wait_runtime_ready():
            self._set_bootstrap_state({"running": False, "stage": "failed", "message": "runtime_not_ready", "tail": _tail(log_path)})
            return

        self._set_bootstrap_state({"running": False, "stage": "ready", "message": "stack_ready", "tail": _tail(log_path)})

    def restart_runtime(self) -> dict[str, Any]:
        log_path = self.log_dir / "control-runtime-restart.log"
        code, _ = self._run_compose(["up", "-d", "--force-recreate", "sglang-runtime", "vortex-api"], log_path=log_path)
        if code != 0:
            raise RuntimeError("runtime_restart_failed")
        ok = self._wait_runtime_ready(timeout_s=180.0)
        return {"ok": bool(ok), "log_path": str(log_path), "tail": _tail(log_path)}

    def _stop_runtime_stack(self, *, log_path: Path) -> None:
        code, _ = self._run_compose(["stop", "vortex-api", "sglang-runtime"], log_path=log_path)
        if code != 0:
            raise RuntimeError("runtime_stop_failed")

    def _resume_runtime_stack(self, *, log_path: Path, force_recreate: bool = False) -> None:
        args = ["up", "-d"]
        if force_recreate:
            args.append("--force-recreate")
        args.extend(["sglang-runtime", "vortex-api"])
        code, _ = self._run_compose(args, log_path=log_path)
        if code != 0:
            raise RuntimeError("runtime_resume_failed")
        if not self._wait_runtime_ready(timeout_s=240.0):
            raise RuntimeError("runtime_not_ready_after_training")

    def _update_run_meta(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        meta_path = self.runs_dir / run_id / "meta.json"
        current = _load_json(meta_path, {})
        current.update(patch)
        current["updated_at"] = _utc_ts()
        _write_json(meta_path, current)
        return current

    def start_training(self, mode: str) -> dict[str, Any]:
        normalized = str(mode or "quick").strip().lower()
        if normalized not in {"quick", "full"}:
            raise HTTPException(status_code=400, detail="training_mode_invalid")

        with self._lock:
            if self._training_thread and self._training_thread.is_alive():
                return {"ok": False, "error": "training_already_running", "run_id": self._active_run_id}

            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            self._active_run_id = run_id
            meta = {
                "ok": True,
                "run_id": run_id,
                "mode": normalized,
                "status": "queued",
                "created_at": _utc_ts(),
                "profile": self.training_profile,
                "base_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                "served_model": DEFAULT_MODEL_ID,
                "dataset_hash": self._resolve_dataset_hash(),
                "instructions": self._resolve_instruction_meta(),
                "promotion": {"manual_only": True, "decision": "pending"},
            }
            self._update_run_meta(run_id, meta)
            thread = threading.Thread(target=self._training_worker, args=(run_id, normalized), daemon=True)
            self._training_thread = thread
            thread.start()
        return {"ok": True, "run_id": run_id, "status": "queued"}

    def _training_worker(self, run_id: str, mode: str) -> None:
        runtime_resumed = False
        try:
            run_dir = self.runs_dir / run_id
            log_path = run_dir / "run.log"
            eval_log_path = run_dir / "eval.log"
            bench_log_path = run_dir / "bench.log"
            runtime_log_path = run_dir / "runtime.log"
            env = {"C3RNT2_TRAIN_MAX_STEPS": str(10 if mode == "quick" else 25)}
            args = [
                "run",
                "--rm",
                "trainer",
                "python",
                "-m",
                "c3rnt2",
                "train-once",
                "--profile",
                self.training_profile,
            ]
            if mode == "quick":
                args.append("--reuse-dataset")

            self._update_run_meta(
                run_id,
                {
                    "status": "running",
                    "stage": "drain_runtime",
                    "max_steps": int(env["C3RNT2_TRAIN_MAX_STEPS"]),
                    "log_path": str(log_path),
                    "runtime_log_path": str(runtime_log_path),
                },
            )
            self._stop_runtime_stack(log_path=runtime_log_path)

            self._update_run_meta(run_id, {"stage": "train"})
            code, output = self._run_compose(args, env=env, log_path=log_path)
            payload = _parse_structured_output(output) or {}
            success = code == 0 and bool(payload.get("ok", False))
            meta_patch: dict[str, Any] = {"train_result": payload, "exit_code": code}
            adapter_dir = payload.get("adapter_dir")
            if adapter_dir:
                meta_patch["adapter_dir"] = str(adapter_dir)
            if not success:
                self._update_run_meta(run_id, {**meta_patch, "status": "failed", "stage": "train_failed", "tail": _tail(log_path)})
                return

            if adapter_dir:
                self._update_run_meta(run_id, {**meta_patch, "stage": "eval", "eval_log_path": str(eval_log_path)})
                eval_args = [
                    "run",
                    "--rm",
                    "trainer",
                    "python",
                    "-m",
                    "c3rnt2",
                    "learn",
                    "eval",
                    "--profile",
                    self.training_profile,
                    "--adapter",
                    str(adapter_dir),
                ]
                eval_code, eval_output = self._run_compose(eval_args, log_path=eval_log_path)
                meta_patch["eval_result"] = _parse_structured_output(eval_output) or {}
                meta_patch["eval_exit_code"] = eval_code

            self._update_run_meta(run_id, {**meta_patch, "stage": "resume_runtime"})
            self._resume_runtime_stack(log_path=runtime_log_path, force_recreate=True)
            runtime_resumed = True

            if mode == "full":
                self._update_run_meta(run_id, {**meta_patch, "stage": "bench", "bench_log_path": str(bench_log_path)})
                bench_args = [
                    "run",
                    "--rm",
                    "eval",
                    "python",
                    "-m",
                    "c3rnt2",
                    "bench",
                    "--profile",
                    self.api_profile,
                    "--scenario",
                    "default",
                ]
                bench_code, bench_output = self._run_compose(bench_args, log_path=bench_log_path)
                meta_patch["bench_result"] = _parse_structured_output(bench_output) or {}
                meta_patch["bench_exit_code"] = bench_code

            bench_ok = bool((meta_patch.get("bench_result") or {}).get("ok", mode != "full"))
            eval_ok = bool((meta_patch.get("eval_result") or {}).get("ok", True))
            self._update_run_meta(
                run_id,
                {
                    **meta_patch,
                    "status": "completed" if eval_ok and bench_ok else "completed_with_warnings",
                    "stage": "done",
                    "promotion": {
                        "manual_only": True,
                        "decision": "manual_review_required",
                        "eval_ok": eval_ok,
                        "bench_ok": bench_ok,
                    },
                    "tail": _tail(log_path),
                },
            )
        except Exception as exc:
            self._update_run_meta(run_id, {"status": "failed", "stage": "exception", "error": str(exc)})
        finally:
            if not runtime_resumed:
                try:
                    self._resume_runtime_stack(
                        log_path=(self.runs_dir / run_id / "runtime.log"),
                        force_recreate=True,
                    )
                except Exception as exc:
                    self._update_run_meta(
                        run_id,
                        {
                            "status": "completed_with_warnings",
                            "stage": "runtime_resume_failed",
                            "runtime_resume_error": str(exc),
                            "runtime_tail": _tail(self.runs_dir / run_id / "runtime.log"),
                        },
                    )
            with self._lock:
                self._active_run_id = None


def create_control_app(state: ControlState) -> FastAPI:
    app = FastAPI(title="Vortex Control", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "vortex-control", "ts": _utc_ts()}

    @app.get("/control/status")
    async def control_status() -> dict[str, Any]:
        return state.status()

    @app.post("/control/bootstrap")
    async def control_bootstrap(payload: BootstrapRequest) -> dict[str, Any]:
        return state.start_bootstrap(force=bool(payload.force))

    @app.post("/control/model/init")
    async def control_model_init() -> dict[str, Any]:
        return state.start_bootstrap(force=False)

    @app.post("/control/runtime/restart")
    async def control_restart() -> dict[str, Any]:
        return state.restart_runtime()

    @app.post("/control/instructions/reload")
    async def control_reload_instructions() -> dict[str, Any]:
        import requests

        resp = requests.post(f"{state.api_url}/v1/instructions/reload", timeout=10.0)
        payload = resp.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="instructions_reload_invalid")
        return payload

    @app.get("/control/internet/allowlist")
    async def control_allowlist_get() -> dict[str, Any]:
        return {"ok": True, "domains": state.get_allowlist()}

    @app.post("/control/internet/allowlist")
    async def control_allowlist_post(payload: AllowlistRequest) -> dict[str, Any]:
        return {"ok": True, "domains": state.set_allowlist(payload.domains)}

    @app.post("/control/training/start")
    async def control_training_start(payload: TrainingStartRequest) -> dict[str, Any]:
        return state.start_training(payload.mode)

    @app.get("/control/training/runs")
    async def control_training_runs() -> dict[str, Any]:
        return {"ok": True, "runs": state.list_runs()}

    @app.get("/control/training/runs/{run_id}")
    async def control_training_run(run_id: str) -> dict[str, Any]:
        payload = state.get_run(run_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="training_run_not_found")
        return {"ok": True, "run": payload}

    @app.get("/control/training/stream")
    async def control_training_stream():
        def _events():
            last = ""
            while True:
                payload = {
                    "ts": _utc_ts(),
                    "active_run_id": state._active_run_id,
                    "runs": state.list_runs()[:6],
                }
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    @app.get("/control/autonomy/status")
    async def control_autonomy_status() -> dict[str, Any]:
        return {"ok": True, "autonomy": state.autonomy_status()}

    @app.post("/control/autonomy/start")
    async def control_autonomy_start() -> dict[str, Any]:
        return state.start_autonomy()

    @app.post("/control/autonomy/stop")
    async def control_autonomy_stop() -> dict[str, Any]:
        return state.stop_autonomy()

    @app.post("/control/autonomy/config")
    async def control_autonomy_config(payload: AutonomyConfigRequest) -> dict[str, Any]:
        return state.configure_autonomy(payload)

    @app.get("/control/autonomy/stream")
    async def control_autonomy_stream():
        def _events():
            last = ""
            while True:
                payload = {
                    "ts": _utc_ts(),
                    "status": state.autonomy_status(),
                    "events": state._latest_autonomy_events(limit=16),
                    "active_run_id": state._active_run_id,
                    "runs": state.list_runs()[:6],
                }
                raw = json.dumps(payload, ensure_ascii=True)
                if raw != last:
                    yield f"data: {raw}\n\n"
                    last = raw
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vortex local control service")
    parser.add_argument("--base-dir", default=".", help="Backend repo root")
    parser.add_argument("--compose-file", default=None, help="Path to docker-compose.yml")
    parser.add_argument("--port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--runtime-port", type=int, default=DEFAULT_RUNTIME_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--api-profile", default=DEFAULT_API_PROFILE)
    parser.add_argument("--training-profile", default=DEFAULT_TRAINING_PROFILE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir).resolve()
    compose_file = Path(args.compose_file).resolve() if args.compose_file else (base_dir / "docker-compose.yml").resolve()
    state = ControlState(
        base_dir=base_dir,
        compose_file=compose_file,
        api_profile=str(args.api_profile),
        training_profile=str(args.training_profile),
        api_url=f"http://127.0.0.1:{int(args.api_port)}",
        runtime_url=f"http://127.0.0.1:{int(args.runtime_port)}",
        frontend_port=int(args.frontend_port),
    )
    app = create_control_app(state)
    uvicorn.run(app, host="127.0.0.1", port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
