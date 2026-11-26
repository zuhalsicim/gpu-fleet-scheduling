"""The discrete-event simulator: an event queue, the cluster state and a policy that decides placements.

Time jumps from event to event. There are three kinds of event: a job arrives, a job finishes, and a
sampling tick that records the cluster's state for the timeline and the fragmentation metric. After all
events at one instant are applied, the policy's `schedule` is called once; it may start queued jobs
(`start`) or move running ones (`migrate`). That is the whole interface between simulator and policy.

A job carries its work in V100-seconds. On GPU type t it progresses at speed[profile, t] V100-seconds per
second, so its finish time is known the moment it starts. A migration releases the old GPUs, takes the new
ones and spends `cost` seconds restoring from a checkpoint before progress resumes at the new speed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Protocol

from fleetsim.cluster import Allocation, Cluster, Demand, total
from fleetsim.speeds import SpeedTable

FINISH, ARRIVAL, SAMPLE = 0, 1, 2  # at equal times, finishes free capacity before arrivals ask for it


@dataclass(eq=False)
class Job:
    id: int
    submit: float
    work: float  # V100-seconds
    demand: Demand
    profile: int = 0
    pinned: int | None = None  # GPU type the user insisted on, if any
    measured: bool = True  # False for warm-up jobs, which run but are not counted
    # filled in by the simulator
    start: float | None = None
    end: float | None = None
    gtype: int | None = None
    alloc: Allocation | None = None
    remaining: float = math.nan  # work left at time `anchor`
    anchor: float = 0.0  # progress at `rate` starts here (after any restore delay)
    rate: float = 0.0
    migrations: int = 0
    version: int = 0  # bumped on migration so the stale finish event is ignored
    tried: dict[int, int] = field(default_factory=dict)  # type -> release_count when it last failed
    size: int = field(init=False)  # capacity the job takes, in percent of a GPU

    def __post_init__(self) -> None:
        self.size = total(self.demand)

    def remaining_at(self, now: float) -> float:
        return self.remaining - max(0.0, now - self.anchor) * self.rate

    def finish_time(self) -> float:
        return self.anchor + self.remaining / self.rate


class Policy(Protocol):
    name: str

    def schedule(self, sim: Simulator) -> None: ...


@dataclass
class Sample:
    time: float
    queued: int
    blocked: int  # queued jobs that fit nowhere although some type has enough free capacity in total
    used: list[int]  # percent of capacity in use, per type
    running: list[int]


class Simulator:
    def __init__(
        self,
        cluster: Cluster,
        speeds: SpeedTable,
        jobs: list[Job],
        policy: Policy,
        window: tuple[float, float] | None = None,
        sample_every: float | None = None,
    ):
        self.cluster, self.speeds, self.jobs, self.policy = cluster, speeds, jobs, policy
        self.window = window or (min(j.submit for j in jobs), max(j.submit for j in jobs))
        self.now = 0.0
        self.queue: list[Job] = []
        self.fresh: list[Job] = []  # arrived since the policy last looked
        self.running: dict[int, Job] = {}
        self.finished: list[Job] = []
        self.samples: list[Sample] = []
        self._events: list[tuple[float, int, int, int, Job | None]] = []
        self._seq = 0
        # time integrals over the window, for utilisation and Little's law
        self._last = self.window[0]
        self.used_area = [0.0] * cluster.n_types
        self.in_system_area = 0.0
        self._in_system = 0
        for job in jobs:
            self._push(job.submit, ARRIVAL, job)
        if sample_every:
            t = self.window[0]
            while t <= self.window[1]:
                self._push(t, SAMPLE, None)
                t += sample_every

    # -- the loop --------------------------------------------------------------------------------------

    def run(self) -> Simulator:
        while self._events:
            time, kind, _, version, job = heapq.heappop(self._events)
            self._advance(time)
            if kind == ARRIVAL:
                self.queue.append(job)
                self.fresh.append(job)
                self._in_system += 1
            elif kind == FINISH and version == job.version:
                self._finish(job)
            elif kind == SAMPLE:
                self._sample()
            if not self._events or self._events[0][0] > time:
                self.policy.schedule(self)
        return self

    # -- the policy's interface ------------------------------------------------------------------------

    def runtime(self, job: Job, t: int) -> float:
        """Seconds the job still needs on type t, if it ran there from now on."""
        left = job.remaining_at(self.now) if job.start is not None else job.work
        return left / self.speeds.speed[job.profile, t]

    def allowed(self, job: Job) -> range | tuple[int]:
        return (job.pinned,) if job.pinned is not None else range(self.cluster.n_types)

    def try_start(self, job: Job, t: int) -> bool:
        """Start a queued job on type t if it fits there now."""
        if job.tried.get(t) == self.cluster.release_count[t]:
            return False  # failed here before and nothing was released on t since
        alloc = self.cluster.place(job.demand, t)
        if alloc is None:
            job.tried[t] = self.cluster.release_count[t]
            return False
        self.queue.remove(job)
        job.start = self.now
        job.remaining = job.work
        self._run_on(job, t, alloc, delay=0.0)
        return True

    def migrate(self, job: Job, t: int, cost: float) -> bool:
        """Checkpoint a running job, move it to type t and restore it there, `cost` seconds without progress."""
        alloc = self.cluster.place(job.demand, t)
        if alloc is None:
            return False
        job.remaining = job.remaining_at(self.now)
        self.cluster.release(job.alloc)
        job.migrations += 1
        job.version += 1
        self._run_on(job, t, alloc, delay=cost)
        return True

    # -- internals -------------------------------------------------------------------------------------

    def _run_on(self, job: Job, t: int, alloc: Allocation, delay: float) -> None:
        job.gtype, job.alloc = t, alloc
        job.anchor = self.now + delay
        job.rate = self.speeds.speed[job.profile, t]
        self.running[job.id] = job
        self._push(job.finish_time(), FINISH, job)

    def _finish(self, job: Job) -> None:
        job.end = self.now
        self.cluster.release(job.alloc)
        del self.running[job.id]
        self._in_system -= 1
        self.finished.append(job)

    def _push(self, time: float, kind: int, job: Job | None) -> None:
        self._seq += 1
        version = job.version if job is not None else 0
        heapq.heappush(self._events, (time, kind, self._seq, version, job))

    def _advance(self, time: float) -> None:
        lo, hi = max(self._last, self.window[0]), min(time, self.window[1])
        if hi > lo:
            span = hi - lo
            for t in range(self.cluster.n_types):
                self.used_area[t] += self.cluster.used(t) * span
            self.in_system_area += self._in_system * span
        self._last = max(self._last, time)
        self.now = time

    def _sample(self) -> None:
        c = self.cluster
        blocked = 0
        for job in self.queue[:2000]:
            types = self.allowed(job)
            if any(job.size <= c.free_total[t] for t in types) and not any(c.fits(job.demand, t) for t in types):
                blocked += 1
        running = [0] * c.n_types
        for job in self.running.values():
            running[job.gtype] += 1
        used = [c.used(t) for t in range(c.n_types)]
        self.samples.append(Sample(self.now, len(self.queue), blocked, used, running))
