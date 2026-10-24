"""Swarm sizing with hard caps (used by the mesh-swarm skill). Thousands of agents do not fit on one PC or plan:
the size is the smallest of what RAM, the token budget, the plan's concurrency and the number of shards allow."""
from __future__ import annotations

import ctypes
import math
import os

GB_PER_AGENT = 0.6      # observed resident memory of one concurrent Claude Code subagent process
RESERVE_GB = 2.0        # always leave this much RAM free
PLAN_CONCURRENCY = 8    # practical ceiling for parallel agents on one subscription before rate limits bite


def free_ram_gb() -> float:
    """Free physical RAM in GB (Windows via GlobalMemoryStatusEx, else /proc/meminfo)."""
    if os.name == "nt":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2**30
    with open("/proc/meminfo") as f:
        info = dict(line.split(":", 1) for line in f)
    return int(info["MemAvailable"].split()[0]) / 2**20


def plan_swarm(shards: int, token_budget: int, tokens_per_agent: int, free_gb: float | None = None,
               plan_concurrency: int = PLAN_CONCURRENCY, gb_per_agent: float = GB_PER_AGENT,
               reserve_gb: float = RESERVE_GB) -> dict:
    """Concurrency and total-agent caps. `concurrent` = agents running at once; `total` = agents over all waves.
    The binding constraint is reported so the caller can tell the user why the swarm is not larger."""
    free_gb = free_ram_gb() if free_gb is None else free_gb
    caps = {"ram": max(0, math.floor((free_gb - reserve_gb) / gb_per_agent)),
            "plan": plan_concurrency, "shards": shards}
    concurrent = max(0, min(caps.values()))
    total = min(shards, token_budget // max(1, tokens_per_agent))
    binding = min(caps, key=caps.get) if concurrent < shards else "shards"
    if total < shards:
        binding = "tokens" if total <= concurrent else binding
    concurrent = min(concurrent, total)
    return {"concurrent": concurrent, "total": total, "waves": math.ceil(total / concurrent) if concurrent else 0,
            "caps": {**caps, "tokens": token_budget // max(1, tokens_per_agent)}, "free_gb": round(free_gb, 2),
            "binding": binding, "dropped_shards": shards - total}


if __name__ == "__main__":
    import json
    import sys
    a = [int(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4 else [10, 300000, 20000]
    print(json.dumps(plan_swarm(*a), indent=1))
