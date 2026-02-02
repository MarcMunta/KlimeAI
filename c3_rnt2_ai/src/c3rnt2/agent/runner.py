from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from ..model_loader import load_inference_model
from ..prompting.chat_format import build_chat_prompt
from .tools import AgentTools, ToolResult


@dataclass
class Action:
    type: str
    args: dict


def _parse_action(text: str) -> Action:
    text = text.strip()
    if not text:
        return Action(type="finish", args={"summary": "empty"})
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return Action(type="finish", args={"summary": "invalid_json"})
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return Action(type="finish", args={"summary": "invalid_json"})
    action_type = str(payload.get("type", "finish"))
    args = payload.get("args", {}) or {}
    return Action(type=action_type, args=args)


def _log_episode(base_dir: Path, payload: dict) -> None:
    path = base_dir / "data" / "episodes" / "agent.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_agent(
    task: str,
    settings: dict,
    base_dir: Path,
    *,
    max_iters: int = 5,
    action_provider: Callable[[List[dict]], Action] | None = None,
) -> dict:
    system_prompt = (
        "You are an autonomous coding agent. "
        "You must respond with a single JSON object Action{type,args}. "
        "Valid types: open_docs, search_web, run_tests, propose_patch, sandbox_patch, apply_patch, summarize_diff, finish."
    )
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    agent_cfg = settings.get("agent", {}) or {}
    tools_cfg = settings.get("tools", {}) or {}
    allowlist = tools_cfg.get("web", {}).get("allow_domains", agent_cfg.get("web_allowlist", []))
    sandbox_root = Path(settings.get("selfimprove", {}).get("sandbox_root", "data/workspaces"))
    tools = AgentTools(
        allowlist=list(allowlist or []),
        sandbox_root=sandbox_root,
        web_cfg=tools_cfg,
        agent_cfg=agent_cfg,
        self_patch_cfg=settings.get("self_patch", {}),
    )
    model = None
    if action_provider is None:
        model = load_inference_model(settings)

    tool_calls: List[dict] = []
    patch_id = None
    tests_ok = False
    summary = ""

    for _ in range(max_iters):
        if action_provider is None and model is not None:
            prompt = build_chat_prompt(messages, backend=str(settings.get("core", {}).get("backend", "vortex")), tokenizer=getattr(model, "tokenizer", None), default_system=None)
            output = model.generate(prompt, max_new_tokens=256, temperature=0.0)
            action = _parse_action(output)
        else:
            action = action_provider(messages)
        messages.append({"role": "assistant", "content": json.dumps({"type": action.type, "args": action.args})})

        if action.type == "finish":
            summary = str(action.args.get("summary", "finished"))
            break

        result: ToolResult
        if action.type == "open_docs":
            result = tools.open_docs(str(action.args.get("url", "")))
        elif action.type == "search_web":
            result = tools.search_web(str(action.args.get("query", "")))
        elif action.type == "run_tests":
            result = tools.run_tests(base_dir)
            tests_ok = bool(result.ok)
        elif action.type == "propose_patch":
            goal = str(action.args.get("goal", task))
            result = tools.propose_patch(base_dir, {}, goal=goal, llm_generate_diff=True, llm_context={"task": task, "messages": messages, "tool_calls": tool_calls})
            if result.ok:
                patch_id = result.output
        elif action.type == "sandbox_patch":
            pid = str(action.args.get("patch_id", patch_id or ""))
            result = tools.sandbox_patch(base_dir, pid)
        elif action.type == "apply_patch":
            pid = str(action.args.get("patch_id", patch_id or ""))
            approve_file = base_dir / "data" / "APPROVE_SELF_PATCH"
            result = tools.apply_patch(base_dir, pid, approve=approve_file.exists())
        elif action.type == "summarize_diff":
            result = tools.summarize_diff(base_dir)
        else:
            result = ToolResult(ok=False, output=f"unknown action: {action.type}")

        tool_calls.append({"action": action.type, "args": action.args, "ok": result.ok, "output": result.output[:1000]})
        messages.append({"role": "tool", "content": result.output[:2000]})

    episode = {
        "task": task,
        "messages": messages,
        "tool_calls": tool_calls,
        "patch_id": patch_id,
        "tests_ok": tests_ok,
        "summary": summary,
        "ts": time.time(),
    }
    _log_episode(base_dir, episode)
    return {"ok": True, "patch_id": patch_id, "tests_ok": tests_ok, "summary": summary}
