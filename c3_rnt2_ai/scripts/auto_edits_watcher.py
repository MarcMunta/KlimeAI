from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from c3rnt2.self_edits.store import SelfEditsStore


def _log(event: str, **fields) -> None:
    payload = {"ts": time.time(), "event": event}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=True))


def main() -> int:
    parser = argparse.ArgumentParser(prog="auto_edits_watcher.py")
    parser.add_argument("--profile", default=os.getenv("C3RNT2_PROFILE") or "dev_small")
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--create-demo-on-start", action="store_true")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parents[1]
    store = SelfEditsStore.from_app_dir(app_dir, profile=str(args.profile))

    _log(
        "auto_edits_watcher_start",
        profile=store.profile,
        repo_root=str(store.repo_root),
        proposals_dir=str(store.proposals_dir),
    )

    created_demo = False
    while True:
        try:
            pending = store.list(status="pending")
            _log("auto_edits_pending", pending_total=len(pending))
            want_demo = bool(args.create_demo_on_start) or str(os.getenv("AUTO_EDITS_CREATE_DEMO") or "").strip() == "1"
            if want_demo and not created_demo and not pending:
                created = store.create_demo()
                created_demo = bool(created.get("ok"))
                _log("auto_edits_demo_created", ok=bool(created.get("ok")), id=str(created.get("id") or ""))
        except KeyboardInterrupt:
            _log("auto_edits_watcher_stop", reason="keyboard_interrupt")
            return 0
        except Exception as exc:
            _log("auto_edits_watcher_error", error=str(exc))

        time.sleep(max(1.0, float(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

