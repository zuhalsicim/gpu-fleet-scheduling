"""Scheduling policies. Each one looks at the queue after every batch of events and starts what it wants.

Type-blind (know nothing about speeds):
  blind-fifo       strict arrival order; the first job that does not fit blocks everyone behind it.
  blind-backfill   arrival order, but later jobs that fit may start while earlier ones wait.
  Both pick a GPU type at random, weighted by free capacity, i.e. like taking any free GPU.

Knows only the average ranking of GPU types (newer is faster), not the job:
  rank-greedy      the best-ranked type that has room now.

Type-aware (know each job's relative speed per type):
  aware-greedy     fastest type that has room now; never waits. With ~noiseS it sees speeds with a
                   log-normal estimation error of standard deviation S (in log terms).
  aware-wait       may wait for a busy faster type when, by the running jobs' finish times, waiting is
                   expected to finish the job sooner than starting now on a slower type.
  aware-lp         a linear program over the whole fleet (Gavel-style: maximise total normalised
                   throughput) gives each GPU type a price; a job takes the type where its normalised
                   speed minus the price is highest, and waits when every free type costs more than it is
                   worth to this job.

Any policy can be wrapped in `Migrating`, which moves running jobs to faster free GPUs and charges a
checkpoint/restore cost each time.
"""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import linprog

from fleetsim.simulator import Job, Simulator

BACKFILL_DEPTH = 1000  # how far down the queue a scan looks, like a backfill window in a batch scheduler


class QueuePolicy:
    """The scan all policies share: walk the queue in arrival order and let `choose` place each job.

    Free capacity only grows when something is released. So if nothing was released since the last scan,
    a job that did not fit then cannot fit now, and only newly arrived jobs need a look. Policies whose
    decision also depends on the clock (aware-wait) or on a new plan (aware-lp) force a full scan.
    """

    name = "queue"
    strict = False  # True: stop at the first job that cannot start (FIFO without backfilling)
    rescan_every = math.inf  # seconds; a full scan at least this often

    def __init__(self) -> None:
        self._seen_releases = -1
        self._last_full = -math.inf

    def schedule(self, sim: Simulator) -> None:
        fresh, sim.fresh = sim.fresh, []
        if not any(sim.cluster.free_total):
            return
        releases = sim.cluster.releases
        if self.strict or releases != self._seen_releases or sim.now - self._last_full >= self.rescan_every:
            self._seen_releases, self._last_full = releases, sim.now
            jobs = sim.queue[:BACKFILL_DEPTH]
        else:
            room = BACKFILL_DEPTH - (len(sim.queue) - len(fresh))
            jobs = fresh[: max(0, room)]
        for job in jobs:
            if not self.choose(sim, job) and self.strict:
                return

    def choose(self, sim: Simulator, job: Job) -> bool:
        raise NotImplementedError


class _Blind(QueuePolicy):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = random.Random(seed)

    def choose(self, sim: Simulator, job: Job) -> bool:
        c = sim.cluster
        candidates = [
            t for t in sim.allowed(job) if job.tried.get(t) != c.release_count[t] and c.may_fit(job.demand, t)
        ]
        while candidates:
            t = self.rng.choices(candidates, weights=[c.free_total[t] for t in candidates])[0]
            if sim.try_start(job, t):
                return True
            candidates.remove(t)
        return False


class BlindFifo(_Blind):
    name = "blind-fifo"
    strict = True


class BlindBackfill(_Blind):
    name = "blind-backfill"


class AwareGreedy(QueuePolicy):
    """Fastest type for this job that has room now.

    With noise > 0 the policy sees each job's speeds multiplied by its own log-normal error
    exp(N(0, noise^2)) per GPU type, as with speed estimates from short profiling runs; the job still runs
    at its true speed.
    """

    name = "aware-greedy"

    def __init__(self, noise: float = 0.0, seed: int = 0):
        super().__init__()
        self.noise = noise
        self.rng = np.random.default_rng([seed, 7])
        self._orders: dict[int, list[int]] = {}

    def choose(self, sim: Simulator, job: Job) -> bool:
        if not self.noise:
            return any(sim.try_start(job, t) for t in fastest_first(sim, job))
        if job.id not in self._orders:
            seen = sim.speeds.speed[job.profile] * np.exp(self.rng.normal(0.0, self.noise, sim.cluster.n_types))
            self._orders[job.id] = [int(t) for t in np.argsort(-seen) if job.pinned is None or t == job.pinned]
        return any(sim.try_start(job, t) for t in self._orders[job.id])


class RankGreedy(QueuePolicy):
    """Knows only which GPU type is faster on average (geometric mean over job types), not per job."""

    name = "rank-greedy"

    def schedule(self, sim: Simulator) -> None:
        if not hasattr(self, "ranking"):
            mean = np.exp(np.log(sim.speeds.speed).mean(axis=0))
            self.ranking = sorted(range(sim.cluster.n_types), key=lambda t: -mean[t])
        super().schedule(sim)

    def choose(self, sim: Simulator, job: Job) -> bool:
        return any(sim.try_start(job, t) for t in self.ranking if job.pinned is None or t == job.pinned)


class AwareWait(QueuePolicy):
    """Start on the type with the earliest expected finish; if that type is busy, wait for it.

    Waiting jobs are reconsidered whenever capacity is released and at least once a minute.
    """

    name = "aware-wait"
    rescan_every = 60.0

    def __init__(self) -> None:
        super().__init__()
        self._finishes: dict[int, tuple[list[float], list[int]]] = {}

    def schedule(self, sim: Simulator) -> None:
        self._finishes.clear()
        super().schedule(sim)

    def choose(self, sim: Simulator, job: Job) -> bool:
        best_if_waiting = math.inf  # expected finish (from now) on a faster type that is busy now
        for t in fastest_first(sim, job):
            run = sim.runtime(job, t)
            if run >= best_if_waiting:
                return False
            if sim.try_start(job, t):
                return True
            best_if_waiting = min(best_if_waiting, self.expected_wait(sim, job, t) + run)
        return False

    def expected_wait(self, sim: Simulator, job: Job, t: int) -> float:
        """When enough capacity frees up on t, from the running jobs' finish times (ignores the queue)."""
        if t not in self._finishes:
            ends = sorted((j.finish_time(), j.size) for j in sim.running.values() if j.gtype == t)
            times, cumulative = [e for e, _ in ends], list(np.cumsum([s for _, s in ends]))
            self._finishes[t] = (times, cumulative)
        times, cumulative = self._finishes[t]
        short = job.size - sim.cluster.free_total[t]
        i = bisect.bisect_left(cumulative, short)
        return max(0.0, times[i] - sim.now) if i < len(times) else math.inf


class AwareLP(QueuePolicy):
    """Gavel-style allocation LP, used through its dual prices.

    Variables x[p, t] >= 0: capacity of type t given to jobs of profile p.
    maximise   sum_pt value[p, t] * x[p, t]       value = speed / the profile's best speed
    subject to sum_p x[p, t] <= capacity[t]          (dual price of type t)
               sum_t x[p, t] <= demand[p]            (running + queued jobs of profile p)
    A type's price is what one more unit of it would add to the objective. A job's net value on t is
    value[p, t] - price[t]; it starts on the free type with the highest net value if that is >= 0.
    """

    name = "aware-lp"
    rescan_every = 60.0  # a job's patience runs out with time, so look at waiting jobs once a minute

    def __init__(self, replan_every: float = 360.0, max_wait: float = 1800.0):
        super().__init__()
        self.replan_every, self.max_wait = replan_every, max_wait
        self.prices: np.ndarray | None = None
        self.planned_at = -math.inf

    def schedule(self, sim: Simulator) -> None:
        if sim.now - self.planned_at >= self.replan_every:
            self.prices = self.solve(sim)
            speed = sim.speeds.speed
            self.net = speed / speed.max(axis=1, keepdims=True) - self.prices
            self.order = [sorted(range(len(row)), key=lambda t, row=row: -row[t]) for row in self.net]
            self.planned_at = sim.now
            self._seen_releases = -1  # new prices: every waiting job gets a fresh look
        super().schedule(sim)

    def solve(self, sim: Simulator) -> np.ndarray:
        speed = sim.speeds.speed
        n_profiles, n_types = speed.shape
        value = speed / speed.max(axis=1, keepdims=True)
        capacity = np.array(sim.cluster.capacity, dtype=float)
        demand = np.zeros(n_profiles)
        for job in [*sim.running.values(), *sim.queue]:
            if job.pinned is None:
                demand[job.profile] += job.size
            elif job.start is not None:
                capacity[job.gtype] -= job.size  # pinned jobs are not the LP's to move
        a_ub = np.vstack([np.kron(np.ones(n_profiles), np.eye(n_types)), np.kron(np.eye(n_profiles), np.ones(n_types))])
        b_ub = np.concatenate([np.maximum(capacity, 0.0), demand])
        res = linprog(-value.ravel(), A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
        if res.status != 0:
            return np.zeros(n_types)
        return np.maximum(0.0, -res.ineqlin.marginals[:n_types])

    def choose(self, sim: Simulator, job: Job) -> bool:
        if job.pinned is not None:
            return sim.try_start(job, job.pinned)
        net = self.net[job.profile]
        patient = sim.now - job.submit < self.max_wait
        for t in self.order[job.profile]:
            if patient and net[t] < -1e-9:
                return False
            if sim.try_start(job, t):
                return True
        return False


class Migrating:
    """Wrap a policy; every `every` seconds move running jobs to a faster type that has room.

    cost_aware=True moves a job only if (time left here) > cost + (time left there); False moves whenever
    a faster type has room, to show what ignoring the checkpoint/restore cost does.
    """

    def __init__(self, base, cost: float, cost_aware: bool = True, every: float = 120.0):
        self.base, self.cost, self.cost_aware, self.every = base, cost, cost_aware, every
        self.name = f"{base.name}+migrate" if cost_aware else f"{base.name}+migrate-always"
        self.last = -math.inf

    def schedule(self, sim: Simulator) -> None:
        self.base.schedule(sim)
        if sim.now - self.last < self.every or not any(sim.cluster.free_total):
            return
        self.last = sim.now
        for job, t in self.candidates(sim):
            sim.migrate(job, t, self.cost)

    def candidates(self, sim: Simulator) -> Iterable[tuple[Job, int]]:
        speed = sim.speeds.speed
        moves = []
        for job in sim.running.values():
            if job.pinned is not None or job.anchor > sim.now:
                continue  # pinned, or still restoring from the last move
            here = sim.runtime(job, job.gtype)
            for t in fastest_first(sim, job):
                if speed[job.profile, t] <= speed[job.profile, job.gtype]:
                    break
                if sim.cluster.free_total[t] < job.size:
                    continue
                gain = here - (self.cost + sim.runtime(job, t))
                if gain > 0 or not self.cost_aware:
                    moves.append((gain, job.id, job, t))
                    break
        for _, _, job, t in sorted(moves, key=lambda m: (-m[0], m[1])):
            if sim.cluster.free_total[t] >= job.size:
                yield job, t


def fastest_first(sim: Simulator, job: Job) -> list[int]:
    return [t for t in sim.speeds.order(job.profile) if job.pinned is None or t == job.pinned]


POLICIES: dict[str, Callable[[int], object]] = {
    "blind-fifo": lambda seed: BlindFifo(seed),
    "blind-backfill": lambda seed: BlindBackfill(seed),
    "rank-greedy": lambda seed: RankGreedy(),
    "aware-greedy": lambda seed: AwareGreedy(),
    **{f"aware-greedy~noise{s}": (lambda seed, s=s: AwareGreedy(noise=s, seed=seed)) for s in (0.25, 0.5, 1.0)},
    "aware-wait": lambda seed: AwareWait(),
    "aware-lp": lambda seed: AwareLP(),
}
