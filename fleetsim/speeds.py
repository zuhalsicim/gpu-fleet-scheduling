"""How fast each kind of job runs on each GPU type.

The trace says on which GPU type a job ran and for how long, but not how fast it would have run elsewhere.
This module supplies that missing piece as an explicit, sourced assumption:

* K80, P100 and V100: measured training throughput of 26 job types from Gavel (Narayanan et al., OSDI 2020).
* T4: the V100 figure divided by the V100-to-T4 ratio that MLPerf Training v0.5 runs measured per family
  (Dell EMC, 2019); see data/speeds/t4_ratios.csv.
* Trace type to measured type: MISC (older generations such as K40m, K80, M60) is treated as a K80,
  V100M32 (a V100 with 32 GB) as a V100.

Speeds are stored relative to a V100, so a job's work is measured in "V100-seconds" and its runtime on type
t is work / speed[profile, t]. `scaled` widens or narrows every gap for the sensitivity analysis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

GPU_TYPES: tuple[str, ...] = ("MISC", "T4", "P100", "V100", "V100M32")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "speeds"

# Which trace tag (pai_group_tag_table.workload) belongs to which Gavel family. Untagged jobs and tags
# without a counterpart (graphlearn) draw from all 26 job types.
TAG_TO_FAMILY: dict[str, str] = {
    "bert": "Transformer",
    "xlnet": "Transformer",
    "nmt": "Transformer",
    "ctr": "Recommendation",
    "resnet": "ResNet-50",
    "inception": "ResNet-50",
    "vgg": "ResNet-50",
    "rl": "A3C",
}


@dataclass(frozen=True)
class SpeedTable:
    profiles: tuple[str, ...]
    families: tuple[str, ...]
    speed: np.ndarray  # shape (profiles, GPU_TYPES); V100 = 1.0 for every profile

    def order(self, profile: int) -> list[int]:
        """GPU types from fastest to slowest for one profile (ties keep the GPU_TYPES order)."""
        return self._orders[profile]

    @cached_property
    def _orders(self) -> list[list[int]]:
        return [sorted(range(self.speed.shape[1]), key=lambda t: -row[t]) for row in self.speed]

    def profiles_of(self, family: str) -> list[int]:
        return [i for i, f in enumerate(self.families) if f == family]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def load(data_dir: Path = DATA_DIR) -> SpeedTable:
    """The measured table: one row per Gavel job type, one column per trace GPU type."""
    t4_ratio = {r["family"]: float(r["v100_over_t4"]) for r in _read_csv(data_dir / "t4_ratios.csv")}
    rows = _read_csv(data_dir / "gavel_throughputs.csv")
    speed = np.empty((len(rows), len(GPU_TYPES)))
    for i, r in enumerate(rows):
        v100 = float(r["v100"])
        speed[i] = [float(r["k80"]) / v100, 1.0 / t4_ratio[r["family"]], float(r["p100"]) / v100, 1.0, 1.0]
    return SpeedTable(tuple(r["job_type"] for r in rows), tuple(r["family"] for r in rows), speed)


def scaled(table: SpeedTable, gamma: float) -> SpeedTable:
    """Raise every relative speed to the power gamma: 0 makes all GPUs equal, 2 doubles every gap (in log)."""
    return SpeedTable(table.profiles, table.families, table.speed**gamma)


def uniform(table: SpeedTable) -> SpeedTable:
    """Every profile gets the same row (geometric mean): GPU types still differ, jobs no longer do."""
    mean_row = np.exp(np.log(table.speed).mean(axis=0))
    return SpeedTable(table.profiles, table.families, np.tile(mean_row, (len(table.profiles), 1)))


def single(speed_by_type: dict[str, float]) -> SpeedTable:
    """A one-profile table, for synthetic tests."""
    row = np.array([[speed_by_type.get(t, 1.0) for t in GPU_TYPES]])
    return SpeedTable(("synthetic",), ("synthetic",), row)
