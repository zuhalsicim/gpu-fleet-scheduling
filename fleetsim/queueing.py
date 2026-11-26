"""Exact queueing results the simulator must reproduce before any of its other numbers are trusted.

M/M/c: Poisson arrivals at rate lam, exponential service with rate mu, c identical servers, one queue in
arrival order. With offered load a = lam / mu and utilisation rho = a / c < 1, the probability that an
arriving job has to wait is Erlang's C formula, and the mean wait in the queue is

    Wq = C(c, a) / (c * mu - lam).

M/G/1 (Pollaczek-Khinchine): one server, any service-time distribution S,

    Wq = lam * E[S^2] / (2 * (1 - rho)),   rho = lam * E[S].

Little's law holds for any stable system: the time-average number of jobs in the system L equals the
arrival rate times the mean time a job spends there, L = lam * W.
"""

from __future__ import annotations

import numpy as np

from fleetsim.cluster import Cluster
from fleetsim.policies import BlindFifo
from fleetsim.simulator import Job, Simulator
from fleetsim.speeds import single


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in M/M/c, computed through the stable Erlang B recursion."""
    if a >= c:
        return 1.0
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return c * b / (c - a * (1 - b))


def mmc_wait(lam: float, mu: float, c: int) -> float:
    return erlang_c(c, lam / mu) / (c * mu - lam)


def mg1_wait(lam: float, mean_s: float, second_moment_s: float) -> float:
    rho = lam * mean_s
    return lam * second_moment_s / (2 * (1 - rho))


def simulate_queue(
    lam: float,
    service: np.ndarray,
    servers: int,
    percent_per_job: int = 100,
    seed: int = 0,
) -> dict:
    """Run the fleet simulator as a plain queue: one GPU type, one job per arrival, strict FIFO.

    `percent_per_job` < 100 lets several jobs share one GPU (the sharing code path): 4 GPUs with jobs of
    50% behave as 8 servers.
    """
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1 / lam, size=len(service)))
    jobs = [
        Job(i, float(a), float(s), ((1, percent_per_job),))
        for i, (a, s) in enumerate(zip(arrivals, service, strict=True))
    ]
    sim = Simulator(Cluster([servers]), single({}), jobs, BlindFifo(seed), window=(0.0, float(arrivals[-1]))).run()
    warm = len(jobs) // 10  # drop the first tenth: the queue starts empty
    kept = sorted(sim.finished, key=lambda j: j.id)[warm:]
    wait = np.array([j.start - j.submit for j in kept])
    sojourn = np.array([j.end - j.submit for j in kept])
    all_sojourn = np.array([j.end - j.submit for j in sim.finished])
    return {
        "mean_wait": float(wait.mean()),
        "mean_sojourn": float(sojourn.mean()),
        "L": sim.in_system_area / (sim.window[1] - sim.window[0]),
        "lambda_W": len(jobs) / float(arrivals[-1]) * float(all_sojourn.mean()),
    }
