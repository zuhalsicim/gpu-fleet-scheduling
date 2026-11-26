"""What a finished simulation is judged by. Only jobs submitted inside the measured window count."""

from __future__ import annotations

import numpy as np

from fleetsim.simulator import Simulator
from fleetsim.speeds import GPU_TYPES


def jain(x: np.ndarray) -> float:
    """Jain's fairness index: 1 when all values are equal, 1/n when one job gets everything."""
    return float(x.sum() ** 2 / (len(x) * (x**2).sum())) if len(x) else float("nan")


def summarize(sim: Simulator) -> dict:
    t0, t1 = sim.window
    jobs = [j for j in sim.finished if j.measured]
    if not jobs:
        return {"jobs": 0}
    speed = sim.speeds.speed
    submit = np.array([j.submit for j in jobs])
    start = np.array([j.start for j in jobs])
    end = np.array([j.end for j in jobs])
    best = np.array([j.work / max(speed[j.profile, t] for t in sim.allowed(j)) for j in jobs])
    jct, wait = end - submit, start - submit
    slowdown = jct / best
    span = t1 - t0
    samples = [s for s in sim.samples if t0 <= s.time <= t1]
    queued = sum(s.queued for s in samples)
    return {
        "jobs": len(jobs),
        "unfinished": sum(1 for j in sim.jobs if j.measured and j.end is None),
        "mean_jct": float(jct.mean()),
        "p95_jct": float(np.percentile(jct, 95)),
        "median_jct": float(np.median(jct)),
        "mean_wait": float(wait.mean()),
        "p95_wait": float(np.percentile(wait, 95)),
        "makespan": float(end.max() - t0),
        "mean_slowdown": float(slowdown.mean()),
        "jain_slowdown": jain(slowdown),
        "speed_fraction": float((best / (end - start)).mean()),
        "utilisation": {
            GPU_TYPES[t]: sim.used_area[t] / (sim.cluster.capacity[t] * span) for t in range(sim.cluster.n_types)
        },
        "utilisation_all": sum(sim.used_area) / (sum(sim.cluster.capacity) * span),
        "mean_queued": queued / len(samples) if samples else 0.0,
        "mean_blocked_by_fragmentation": sum(s.blocked for s in samples) / len(samples) if samples else 0.0,
        "fragmentation_share": sum(s.blocked for s in samples) / queued if queued else 0.0,
        "migrations": int(sum(j.migrations for j in jobs)),
        "migrated_share": float(np.mean([j.migrations > 0 for j in jobs])),
    }
