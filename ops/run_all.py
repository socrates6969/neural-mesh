"""Scheduled-task entry point: regenerate the dashboard, then run one alert pass (both share one collection)."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import alerts, dashboard, sources  # noqa: E402


def main() -> int:
    out = sources.roots()["out"]
    try:
        st = dashboard.build()
        alerts.run(st=st)
        return 0
    except Exception:  # a scheduled run has no console: leave the reason next to the dashboard
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "ops_error.log", "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())