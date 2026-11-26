"""Rebuild data/speeds/gavel_throughputs.csv from the throughput file published with Gavel.

Gavel (Narayanan et al., "Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads",
OSDI 2020) measured the training throughput of 26 job types (7 model families at several batch sizes) on
K80, P100 and V100 GPUs. This script downloads that file at a fixed commit (MIT licence), checks its
SHA-256 and keeps one number per job type and GPU type: the throughput when the job runs alone on one GPU
(the "null" entry of the single-worker key).

    python scripts/derive_speeds.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "c2bc4b8f218e369e4e17ab86e216753997545d5b"
URL = f"https://raw.githubusercontent.com/stanford-futuredata/gavel/{COMMIT}/scheduler/simulation_throughputs.json"
SHA256 = "17111f7295f03265fd9e5118d7eba31e3dd9d6e50a2e215c13ad3d3f91f74381"
OUT = Path("data/speeds/gavel_throughputs.csv")


def family_of(job_type: str) -> str:
    return job_type.split(" (")[0]


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: {digest}")
    table = json.loads(payload)
    rows = []
    for key in sorted(table["v100"]):
        job_type, workers = ast.literal_eval(key)
        if workers != 1:
            continue
        rows.append(
            [job_type, family_of(job_type), *(f"{table[gpu][key]['null']:.6f}" for gpu in ("k80", "p100", "v100"))]
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        handle.write(f"# Source: Gavel (Narayanan et al., OSDI 2020), {URL}\n")
        handle.write("# Copyright (c) 2020 Stanford Future Data Systems, MIT licence (see data/speeds/README.md).\n")
        handle.write("# Isolated single-GPU training throughput (iterations per second) per job type and GPU type.\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["job_type", "family", "k80", "p100", "v100"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} job types to {OUT}")


if __name__ == "__main__":
    main()
