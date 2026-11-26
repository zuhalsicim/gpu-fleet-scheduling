"""Cluster state: which GPU has how much room left.

Capacity is counted in percent of one GPU, as in the trace: a GPU has 100, a job asking for a quarter of a
GPU takes 25. A job asks for a list of (instances, percent per instance). An instance of 100 or more takes
whole GPUs (ceil(percent / 100) of them, rounded up); a smaller instance shares one GPU with others.

All instances of a job are placed on GPUs of one type. Machines and network topology are not modelled:
any free GPUs of the right type will do.

Fractional instances go to the fullest GPU they still fit on (best fit), which keeps whole GPUs free for
jobs that need them. Partially used GPUs are kept in buckets by their free amount, so best fit is a scan
over at most 99 buckets, not over all GPUs.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

Demand = tuple[tuple[int, int], ...]  # ((instances, percent per instance), ...)
Allocation = list[tuple[int, int]]  # [(gpu id, percent taken), ...]
FULL = 100


@cache
def total(demand: Demand) -> int:
    """Capacity a demand takes, in percent, counting instances above one GPU as whole GPUs."""
    return sum(n * (FULL * -(-p // FULL) if p >= FULL else p) for n, p in demand)


@cache
def whole_gpus(demand: Demand) -> int:
    return sum(n * -(-p // FULL) for n, p in demand if p >= FULL)


class Cluster:
    def __init__(self, gpus_per_type: Sequence[int]):
        self.n_types = len(gpus_per_type)
        self.capacity = [n * FULL for n in gpus_per_type]
        self.free_total = list(self.capacity)
        self.gpu_type: list[int] = [t for t, n in enumerate(gpus_per_type) for _ in range(n)]
        self.free: list[int] = [FULL] * len(self.gpu_type)
        self.whole: list[set[int]] = [set() for _ in gpus_per_type]
        self.buckets: list[list[set[int]]] = [[set() for _ in range(FULL)] for _ in gpus_per_type]
        for gpu, t in enumerate(self.gpu_type):
            self.whole[t].add(gpu)
        # Bumped whenever capacity is given back. A job that did not fit on type t cannot fit there either
        # until this number changes, which lets policies skip hopeless placement attempts.
        self.release_count = [0] * self.n_types
        self.releases = 0  # sum of release_count

    def may_fit(self, demand: Demand, t: int) -> bool:
        """Cheap necessary condition: enough capacity and enough whole GPUs on type t."""
        return total(demand) <= self.free_total[t] and whole_gpus(demand) <= len(self.whole[t])

    def place(self, demand: Demand, t: int) -> Allocation | None:
        """Take capacity for a demand on type t, or change nothing and return None."""
        if not self.may_fit(demand, t):
            return None
        alloc: Allocation = []
        for n, p in demand:
            if p >= FULL:
                for _ in range(n * -(-p // FULL)):
                    gpu = next(iter(self.whole[t]))
                    self._take(gpu, FULL)
                    alloc.append((gpu, FULL))
        for n, p in demand:
            if p < FULL:
                for _ in range(n):
                    gpu = self._best_fit(t, p)
                    if gpu is None:
                        self._give_back(alloc)
                        return None
                    self._take(gpu, p)
                    alloc.append((gpu, p))
        return alloc

    def fits(self, demand: Demand, t: int) -> bool:
        alloc = self.place(demand, t)
        if alloc is None:
            return False
        self._give_back(alloc)
        return True

    def release(self, alloc: Allocation) -> None:
        self._give_back(alloc)
        for t in {self.gpu_type[gpu] for gpu, _ in alloc}:
            self.release_count[t] += 1
            self.releases += 1

    def used(self, t: int) -> int:
        return self.capacity[t] - self.free_total[t]

    # -- internals -------------------------------------------------------------------------------------

    def _best_fit(self, t: int, p: int) -> int | None:
        buckets = self.buckets[t]
        for room in range(p, FULL):
            if buckets[room]:
                return next(iter(buckets[room]))
        return next(iter(self.whole[t]), None)

    def _take(self, gpu: int, amount: int) -> None:
        self._set_free(gpu, self.free[gpu] - amount)

    def _give_back(self, alloc: Allocation) -> None:
        for gpu, amount in alloc:
            self._set_free(gpu, self.free[gpu] + amount)

    def _set_free(self, gpu: int, new: int) -> None:
        t, old = self.gpu_type[gpu], self.free[gpu]
        if old == FULL:
            self.whole[t].discard(gpu)
        elif old > 0:
            self.buckets[t][old].discard(gpu)
        if new == FULL:
            self.whole[t].add(gpu)
        elif new > 0:
            self.buckets[t][new].add(gpu)
        self.free[gpu] = new
        self.free_total[t] += new - old
