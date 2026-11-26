"""Check the simulator against queueing theory and write results/validation.json.

Each case is run as independent replications; the table reports the theoretical mean wait, the
simulated mean with a 95% confidence interval over replications, and Little's law (L against lam * W).
A case passes when theory lies within three standard errors of the simulated mean.

    python -m fleetsim.validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fleetsim.queueing import mg1_wait, mmc_wait, simulate_queue

MEAN_SERVICE = 100.0

# name, servers (GPUs), percent of a GPU per job, utilisation, service distribution
CASES = [
    ("M/M/1", 1, 100, 0.5, "exp"),
    ("M/M/4", 4, 100, 0.8, "exp"),
    ("M/M/16", 16, 100, 0.9, "exp"),
    ("M/M/8 as 4 GPUs shared by two", 4, 50, 0.8, "exp"),
    ("M/D/1", 1, 100, 0.7, "det"),
]


def theory(servers: int, percent: int, rho: float, dist: str) -> tuple[float, float]:
    c = servers * 100 // percent  # jobs that fit at once
    lam = rho * c / MEAN_SERVICE
    if dist == "det":
        return lam, mg1_wait(lam, MEAN_SERVICE, MEAN_SERVICE**2)
    return lam, mmc_wait(lam, 1 / MEAN_SERVICE, c)


def run_case(servers: int, percent: int, rho: float, dist: str, reps: int, jobs: int, seed: int = 0) -> dict:
    lam, expected = theory(servers, percent, rho, dist)
    waits, ls, lws = [], [], []
    for r in range(reps):
        rng = np.random.default_rng([seed, r])
        service = rng.exponential(MEAN_SERVICE, jobs) if dist == "exp" else np.full(jobs, MEAN_SERVICE)
        out = simulate_queue(lam, service, servers, percent, seed=int(rng.integers(1 << 31)))
        waits.append(out["mean_wait"])
        ls.append(out["L"])
        lws.append(out["lambda_W"])
    w = np.array(waits)
    se = w.std(ddof=1) / np.sqrt(reps)
    return {
        "lambda": lam,
        "theory_wait": expected,
        "sim_wait": float(w.mean()),
        "ci95": float(1.96 * se),
        "z": float((w.mean() - expected) / se),
        "passes": bool(abs(w.mean() - expected) <= 3 * se),
        "L": float(np.mean(ls)),
        "lambda_W": float(np.mean(lws)),
        "reps": reps,
        "jobs_per_rep": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("results/validation.json"))
    args = parser.parse_args()
    cases = []
    for name, servers, percent, rho, dist in CASES:
        result = run_case(servers, percent, rho, dist, args.reps, args.jobs)
        cases.append({"case": name, "servers": servers, "percent": percent, "rho": rho} | result)
        print(
            f"{name:<32} theory {result['theory_wait']:8.2f}  sim {result['sim_wait']:8.2f} +- {result['ci95']:.2f}"
            f"  L {result['L']:.3f}  lambda*W {result['lambda_W']:.3f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"mean_service": MEAN_SERVICE, "cases": cases}, indent=1) + "\n")


if __name__ == "__main__":
    main()
