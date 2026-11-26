"""Run every experiment over several seeds and write results/*.json.

A seed fixes two things: which stretch of the trace is replayed and which job type (speed profile) each
job gets. Seed k replays day 12 + 6k as warm-up (jobs run but are not counted) and days 13 + 6k and
14 + 6k as the measured window, so ten seeds cover ten disjoint stretches spread over the two months.
Every policy in an experiment sees exactly the same jobs for a given seed (paired comparison).

    python -m fleetsim.experiments                  # full run, see the README for timing
    python -m fleetsim.experiments --sample         # CI smoke run on the committed six-hour sample
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from fleetsim import speeds as speed_tables
from fleetsim.cluster import Cluster
from fleetsim.metrics import summarize
from fleetsim.policies import POLICIES, Migrating
from fleetsim.simulator import Job, Simulator
from fleetsim.speeds import GPU_TYPES, TAG_TO_FAMILY, SpeedTable
from fleetsim.stats import paired_change
from fleetsim.trace import PROCESSED, SAMPLE, load_fleet, load_jobs, parse_demand

DAY = 86400.0
SAMPLE_EVERY = 600.0


@dataclass(frozen=True)
class Scenario:
    seed: int
    gamma: float = 1.0  # speed gaps raised to this power (sensitivity)
    uniform: bool = False  # every job gets the same speed row (no job-specific differences)
    sharing: bool = True  # False rounds fractional requests up to a whole GPU
    fleet_scale: float = 1.0  # fraction of the trace's GPUs, per type (a busier fleet below 1)
    sample: bool = False  # use the committed six-hour sample instead of the full trace


MAIN_POLICIES = ("blind-fifo", "blind-backfill", "rank-greedy", "aware-greedy", "aware-wait", "aware-lp")
BUSY_FLEETS = (0.5, 0.3)
COSTS = (30, 300, 1800)
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
NOISES = (0.25, 0.5, 1.0)


def label(policy: str, fleet: float = 1.0, sharing: bool = True, gamma: float = 1.0, uniform: bool = False) -> str:
    parts = [policy]
    if fleet != 1.0:
        parts.append(f"fleet {fleet}")
    if not sharing:
        parts.append("no sharing")
    if gamma != 1.0:
        parts.append(f"gamma {gamma}")
    if uniform:
        parts.append("uniform speeds")
    return " | ".join(parts)


def entry(policy: str, fleet: float = 1.0, sharing: bool = True, gamma: float = 1.0, uniform: bool = False):
    overrides = {"fleet_scale": fleet, "sharing": sharing, "gamma": gamma, "uniform": uniform}
    return label(policy, fleet, sharing, gamma, uniform), overrides, policy


# experiment -> [(label, scenario overrides, policy)]; runs shared between experiments are simulated once
EXPERIMENTS: dict[str, list[tuple[str, dict, str]]] = {
    "main": [entry(p) for p in MAIN_POLICIES],
    "load": [entry(p, fleet=s) for s in BUSY_FLEETS for p in MAIN_POLICIES],
    "migration": [entry("blind-backfill"), entry("aware-greedy")]
    + [entry(f"blind-backfill+migrate@{c}") for c in COSTS]
    + [entry(f"blind-backfill+migrate-always@{c}") for c in COSTS]
    + [entry("aware-greedy", fleet=0.5)]
    + [entry(f"aware-greedy+migrate@{c}", fleet=0.5) for c in COSTS]
    + [entry(f"aware-greedy+migrate-always@{c}", fleet=0.5) for c in COSTS],
    "sharing": [
        entry(p, fleet=s, sharing=sh)
        for s in (1.0, 0.5)
        for p in ("blind-backfill", "aware-greedy")
        for sh in (True, False)
    ],
    "sensitivity": [entry(p, gamma=g) for g in GAMMAS for p in ("blind-backfill", "rank-greedy", "aware-greedy")]
    + [entry(p, uniform=True) for p in ("blind-backfill", "rank-greedy", "aware-greedy")],
    "estimates": [entry(p) for p in ("blind-backfill", "rank-greedy", "aware-greedy")]
    + [entry(f"aware-greedy~noise{s}") for s in NOISES],
}

# experiment -> [(baseline label, candidate label)], compared seed by seed
COMPARISONS: dict[str, list[tuple[str, str]]] = {
    "main": [
        ("blind-backfill", "aware-greedy"),  # the headline, fixed before the first full run
        ("blind-backfill", "rank-greedy"),
        ("rank-greedy", "aware-greedy"),
        ("blind-backfill", "aware-wait"),
        ("blind-backfill", "aware-lp"),
        ("blind-fifo", "blind-backfill"),
    ],
    "load": [
        (label(base, s), label(cand, s))
        for s in BUSY_FLEETS
        for base, cand in [
            ("blind-backfill", "rank-greedy"),
            ("blind-backfill", "aware-greedy"),
            ("blind-backfill", "aware-wait"),
            ("blind-backfill", "aware-lp"),
            ("aware-greedy", "aware-wait"),
            ("aware-greedy", "aware-lp"),
            ("blind-fifo", "blind-backfill"),
        ]
    ],
    "migration": [
        ("blind-backfill", label(p))
        for p in [
            *[f"blind-backfill+migrate@{c}" for c in COSTS],
            *[f"blind-backfill+migrate-always@{c}" for c in COSTS],
            "aware-greedy",
        ]
    ]
    + [
        (label("aware-greedy", 0.5), label(p, 0.5))
        for p in [*[f"aware-greedy+migrate@{c}" for c in COSTS], *[f"aware-greedy+migrate-always@{c}" for c in COSTS]]
    ],
    "sharing": [
        (label(p, s, sharing=False), label(p, s)) for s in (1.0, 0.5) for p in ("blind-backfill", "aware-greedy")
    ],
    "sensitivity": [
        (label("blind-backfill", gamma=g), label(p, gamma=g)) for g in GAMMAS for p in ("rank-greedy", "aware-greedy")
    ]
    + [(label("blind-backfill", uniform=True), label(p, uniform=True)) for p in ("rank-greedy", "aware-greedy")],
    "estimates": [("blind-backfill", f"aware-greedy~noise{s}") for s in NOISES]
    + [("aware-greedy", f"aware-greedy~noise{s}") for s in NOISES]
    + [("rank-greedy", f"aware-greedy~noise{s}") for s in NOISES],
}
COMPARED_METRICS = ("mean_jct", "p95_jct", "mean_wait", "makespan")
TIMELINE_POLICIES = ("blind-backfill", "aware-greedy", "aware-wait", "aware-lp")
TIMELINE_FLEETS = (1.0, 0.3)

_TABLES: dict[bool, pd.DataFrame] = {}


def window_for(scenario: Scenario) -> tuple[float, float, float]:
    """(warm-up start, measured start, measured end) in trace seconds."""
    if scenario.sample:
        start = 40 * DAY
        return start, start + 3600, start + 6 * 3600
    warm = (12 + 6 * scenario.seed) * DAY
    return warm, warm + DAY, warm + 3 * DAY


def speed_table(scenario: Scenario) -> SpeedTable:
    table = speed_tables.load()
    if scenario.uniform:
        table = speed_tables.uniform(table)
    return speed_tables.scaled(table, scenario.gamma)


def build(scenario: Scenario) -> tuple[list[int], SpeedTable, list[Job], tuple[float, float]]:
    if scenario.sample not in _TABLES:
        _TABLES[scenario.sample] = load_jobs(SAMPLE if scenario.sample else PROCESSED)
    table = _TABLES[scenario.sample]
    warm, t0, t1 = window_for(scenario)
    rows = table[(table.submit >= warm) & (table.submit < t1)]
    fleet = [max(1, round(n * scenario.fleet_scale)) for n in load_fleet()]
    speeds = speed_table(scenario)

    # The profile draw depends on the seed only, so every policy and every variant sees the same jobs.
    rng = np.random.default_rng(scenario.seed)
    draws = rng.random(len(rows))
    all_profiles = range(len(speeds.profiles))
    type_index = {t: i for i, t in enumerate(GPU_TYPES)}
    jobs = []
    for i, (row, u) in enumerate(zip(rows.itertuples(index=False), draws, strict=True)):
        family = TAG_TO_FAMILY.get(row.tag)
        choices = speeds.profiles_of(family) if family else all_profiles
        profile = choices[int(u * len(choices))]
        demand = parse_demand(row.demand)
        if not scenario.sharing:
            demand = tuple((n, max(p, 100)) for n, p in demand)
        pinned = type_index[row.pinned] if row.pinned else None
        work = row.duration * speeds.speed[profile, type_index[row.ran_on]]
        job = Job(i, row.submit, work, demand, profile, pinned, measured=row.submit >= t0)
        limit = fleet[pinned] if pinned is not None else max(fleet)
        if job.size <= limit * 100:  # a job larger than its biggest allowed type could never start
            jobs.append(job)
    return fleet, speeds, jobs, (t0, t1)


def make_policy(spec: str, seed: int):
    if "+migrate" in spec:
        base, rest = spec.split("+", 1)
        kind, cost = rest.split("@")
        return Migrating(POLICIES[base](seed), float(cost), cost_aware=(kind == "migrate"))
    return POLICIES[spec](seed)


def run_one(task: tuple[Scenario, str, bool]) -> dict:
    scenario, spec, keep_timeline = task
    started = time.perf_counter()
    fleet, speeds, jobs, window = build(scenario)
    sim = Simulator(Cluster(fleet), speeds, jobs, make_policy(spec, scenario.seed), window, SAMPLE_EVERY).run()
    out = {
        "policy": spec,
        "scenario": asdict(scenario),
        "metrics": summarize(sim),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if keep_timeline:
        samples = sim.samples
        out["timeline"] = {
            "policy": spec,
            "fleet_scale": scenario.fleet_scale,
            "gpus": dict(zip(GPU_TYPES, fleet, strict=True)),
            "queued": [s.queued for s in samples],
            "blocked": [s.blocked for s in samples],
            "used": {
                GPU_TYPES[t]: [round(s.used[t] / sim.cluster.capacity[t], 3) for s in samples]
                for t in range(len(fleet))
            },
            "running": {GPU_TYPES[t]: [s.running[t] for s in samples] for t in range(len(fleet))},
        }
    return out


def compare(runs: list[dict], experiment: str) -> list[dict]:
    by_label: dict[str, dict[int, dict]] = {}
    for r in runs:
        by_label.setdefault(r["label"], {})[r["scenario"]["seed"]] = r["metrics"]
    out = []
    for base, cand in COMPARISONS[experiment]:
        seeds = sorted(set(by_label[base]) & set(by_label[cand]))
        entry = {"baseline": base, "candidate": cand}
        for metric in COMPARED_METRICS:
            a = [by_label[base][s][metric] for s in seeds]
            b = [by_label[cand][s][metric] for s in seeds]
            entry[metric] = paired_change(a, b)
        out.append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--sample", action="store_true", help="smoke run on data/trace/sample_jobs.csv")
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS), help="experiments to run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out or Path("results-sample" if args.sample else "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # one task per distinct (scenario, policy); experiments that share a run reuse it
    wanted: dict[tuple[Scenario, str], list[tuple[str, str]]] = {}
    for name in args.only:
        for seed in range(args.seeds):
            for lab, overrides, spec in EXPERIMENTS[name]:
                scenario = replace(Scenario(seed, sample=args.sample), **overrides)
                wanted.setdefault((scenario, spec), []).append((name, lab))
    tasks = [
        (
            scenario,
            spec,
            scenario.seed == 0
            and scenario.fleet_scale in TIMELINE_FLEETS
            and spec in TIMELINE_POLICIES
            and scenario.sharing
            and scenario.gamma == 1.0
            and not scenario.uniform,
        )
        for scenario, spec in wanted
    ]
    tasks.sort(key=lambda t: (t[0].fleet_scale, t[0].sharing))  # slowest (busiest) first
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one process per core already; per-process BLAS threads only cost memory
    started = time.perf_counter()
    results: dict[str, list[dict]] = {name: [] for name in args.only}
    timelines = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), 1):
            if "timeline" in result:
                timelines.append(result.pop("timeline"))
            scenario = Scenario(**result["scenario"])
            for name, lab in wanted[(scenario, result["policy"])]:
                results[name].append(result | {"label": lab})
            print(
                f"[{i}/{len(tasks)}] seed {scenario.seed} fleet {scenario.fleet_scale} {result['policy']:<34} "
                f"{result['seconds']:>6.1f}s",
                flush=True,
            )

    for name, runs in results.items():
        payload = {
            "experiment": name,
            "seeds": args.seeds,
            "sample": args.sample,
            "gpu_types": list(GPU_TYPES),
            "runs": sorted(runs, key=lambda r: (r["label"], r["scenario"]["seed"])),
            "comparisons": compare(runs, name),
        }
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    if timelines:
        _, t0, _ = window_for(Scenario(0, sample=args.sample))
        timeline = {
            "trace_start": t0,  # trace seconds at the start of the measured window
            "every": SAMPLE_EVERY,
            "runs": sorted(timelines, key=lambda r: (-r["fleet_scale"], r["policy"])),
        }
        (out_dir / "timeline.json").write_text(json.dumps(timeline, separators=(",", ":")) + "\n")
    print(f"done in {time.perf_counter() - started:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
