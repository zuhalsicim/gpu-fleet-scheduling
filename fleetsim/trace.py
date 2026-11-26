"""Turn the Alibaba PAI GPU trace (cluster-trace-gpu-v2020) into jobs the simulator can replay.

What is taken from the trace (measured):
  submit time       pai_job_table.start_time (the job's submission)
  duration          from the first GPU task's start to the last GPU task's end (pai_task_table)
  GPU request       instances x percent of a GPU, per task with a GPU (pai_task_table.inst_num, plan_gpu)
  type it ran on    pai_task_table.gpu_type of the job's largest GPU task
  pinned type       pai_group_tag_table.gpu_type_spec, set when the user demanded a type (1.3% of the kept jobs)
  workload tag      pai_group_tag_table.workload (bert, ctr, resnet, ...), present for 12% of the kept jobs
  fleet             GPUs per type from pai_machine_spec

Kept: jobs that ended as Terminated or Failed (both occupied GPUs), with at least one GPU task with valid
times and a known GPU type. Run `python -m fleetsim.trace` once after scripts/download_trace.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fleetsim.speeds import GPU_TYPES

RAW = Path("data/raw")
PROCESSED = Path("data/processed/jobs.csv.gz")
FLEET = Path("data/trace/fleet.csv")
SAMPLE = Path("data/trace/sample_jobs.csv")

JOB_COLUMNS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
TASK_COLUMNS = [
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
]
TAG_COLUMNS = ["inst_id", "user", "gpu_type_spec", "group", "workload"]
MACHINE_COLUMNS = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]


def fleet_from_machines(raw: Path = RAW) -> pd.DataFrame:
    machines = pd.read_csv(raw / "pai_machine_spec.csv", header=None, names=MACHINE_COLUMNS)
    gpus = machines[machines.cap_gpu > 0].groupby("gpu_type").cap_gpu.sum()
    return pd.DataFrame({"gpu_type": list(GPU_TYPES), "gpus": [int(gpus[t]) for t in GPU_TYPES]})


def build_jobs(raw: Path = RAW) -> pd.DataFrame:
    jobs = pd.read_csv(raw / "pai_job_table.csv", header=None, names=JOB_COLUMNS)
    tasks = pd.read_csv(raw / "pai_task_table.csv", header=None, names=TASK_COLUMNS)
    tags = pd.read_csv(raw / "pai_group_tag_table.csv", header=None, names=TAG_COLUMNS)

    jobs = jobs[jobs.status.isin(["Terminated", "Failed"]) & jobs.start_time.notna()]
    gpu = tasks[
        (tasks.plan_gpu > 0)
        & tasks.gpu_type.isin(GPU_TYPES)
        & tasks.start_time.notna()
        & tasks.end_time.notna()
        & (tasks.end_time > tasks.start_time)
    ].copy()
    gpu["percent"] = gpu.plan_gpu.round().astype(int)
    gpu["instances"] = gpu.inst_num.astype(int)
    gpu["size"] = gpu.instances * gpu.percent
    gpu["part"] = gpu.instances.astype(str) + "x" + gpu.percent.astype(str)
    gpu = gpu.sort_values(["job_name", "size"], ascending=[True, False])

    per_job = gpu.groupby("job_name", sort=False).agg(
        first_start=("start_time", "min"),
        last_end=("end_time", "max"),
        ran_on=("gpu_type", "first"),  # type of the largest GPU task
        demand=("part", ";".join),
    )
    out = (
        jobs.set_index("job_name")[["inst_id", "start_time"]]
        .rename(columns={"start_time": "submit"})
        .join(per_job, how="inner")
        .merge(
            tags.set_index("inst_id")[["gpu_type_spec", "workload"]], left_on="inst_id", right_index=True, how="left"
        )
    )
    out = out[out.first_start >= out.submit]
    out["duration"] = out.last_end - out.first_start
    out["pinned"] = out.gpu_type_spec.where(out.gpu_type_spec.isin(GPU_TYPES), "")
    out["tag"] = out.workload.fillna("")
    out = out.sort_values("submit").reset_index(drop=True)
    return out[["submit", "duration", "ran_on", "demand", "pinned", "tag"]]


def parse_demand(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((int(n), int(p)) for n, p in (part.split("x") for part in text.split(";")))


def load_jobs(path: Path = PROCESSED) -> pd.DataFrame:
    categories = {"ran_on": "category", "pinned": "category", "tag": "category"}
    return pd.read_csv(path, keep_default_na=False, dtype=categories)


def load_fleet(path: Path = FLEET) -> list[int]:
    fleet = pd.read_csv(path).set_index("gpu_type").gpus
    return [int(fleet[t]) for t in GPU_TYPES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/jobs.csv.gz from the raw trace tables.")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--sample", action="store_true", help="also rewrite the small committed sample")
    args = parser.parse_args()
    FLEET.parent.mkdir(parents=True, exist_ok=True)
    fleet_from_machines(args.raw_dir).to_csv(FLEET, index=False)
    jobs = build_jobs(args.raw_dir)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(PROCESSED, index=False, float_format="%.1f")
    print(f"{len(jobs)} jobs -> {PROCESSED}")
    if args.sample:
        # Six hours of one day, every job: small enough to commit (CC BY 4.0), big enough for a smoke run.
        start = 40 * 86400
        sample = jobs[(jobs.submit >= start) & (jobs.submit < start + 6 * 3600)]
        sample.to_csv(SAMPLE, index=False, float_format="%.1f")
        print(f"{len(sample)} jobs -> {SAMPLE}")


if __name__ == "__main__":
    main()
