"""Orbit Wars CPU benchmark submission: ~93 ms local pure-Python workload."""

_ACC = 123456789
_WORK_ITERS = 800_000


def _burn_cpu(step):
    global _ACC
    x = (_ACC + int(step)) & 0xFFFFFFFF
    for i in range(_WORK_ITERS):
        x = (x * 1664525 + 1013904223 + i) & 0xFFFFFFFF
    _ACC = x


def agent(obs, config=None):
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    _burn_cpu(step)
    return []
