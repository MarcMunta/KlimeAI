from __future__ import annotations

import threading
from dataclasses import dataclass, field


def _escape_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class SkillsMetrics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    installed_total: int = 0
    enabled_total: int = 0
    injection_tokens_estimate: int = 0
    selected_total: dict[str, int] = field(default_factory=dict)

    def set_inventory(self, *, installed_total: int, enabled_total: int) -> None:
        with self.lock:
            self.installed_total = max(0, int(installed_total))
            self.enabled_total = max(0, int(enabled_total))

    def observe_injection(self, *, selected: list[str], tokens_estimate: int) -> None:
        with self.lock:
            self.injection_tokens_estimate = max(0, int(tokens_estimate))
            for ref in selected:
                key = str(ref)
                self.selected_total[key] = int(self.selected_total.get(key, 0)) + 1

    def render_prometheus(self) -> str:
        with self.lock:
            lines: list[str] = [
                "# HELP skills_installed_total Installed skills.",
                "# TYPE skills_installed_total gauge",
                f"skills_installed_total {int(self.installed_total)}",
                "# HELP skills_enabled_total Enabled skills.",
                "# TYPE skills_enabled_total gauge",
                f"skills_enabled_total {int(self.enabled_total)}",
                "# HELP skills_injection_tokens_estimate Estimated tokens injected by skills (last request).",
                "# TYPE skills_injection_tokens_estimate gauge",
                f"skills_injection_tokens_estimate {int(self.injection_tokens_estimate)}",
                "# HELP skills_selected_total Total selections by skill.",
                "# TYPE skills_selected_total counter",
            ]
            for skill_id, count in sorted(self.selected_total.items(), key=lambda kv: kv[0]):
                lines.append(f'skills_selected_total{{skill_id="{_escape_label_value(skill_id)}"}} {int(count)}')
            return "\n".join(lines) + "\n"

