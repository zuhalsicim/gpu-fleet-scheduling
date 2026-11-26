import heapq

import numpy as np
import pytest

from fleetsim.cluster import Cluster
from fleetsim.policies import AwareGreedy, BlindFifo, Migrating
from fleetsim.simulator import Job, Simulator
from fleetsim.speeds import SpeedTable


def two_types(slow: float = 0.25) -> SpeedTable:
    """Profile 0 is 4x faster on type 1; type 0 is the slow one."""
    return SpeedTable(("p",), ("p",), np.array([[slow, 1.0]]))


def test_runtime_is_work_over_speed():
    jobs = [Job(0, 0.0, 100.0, ((1, 100),))]
    Simulator(Cluster([1, 0]), two_types(), jobs, AwareGreedy(), window=(0.0, 1.0)).run()
    assert jobs[0].gtype == 0 and jobs[0].end == pytest.approx(400.0)


def test_fifo_matches_the_exact_recursion_for_a_multi_server_queue():
    """Kiefer-Wolfowitz: in FIFO M/M/c each job starts when the earliest server is free."""
    rng = np.random.default_rng(3)
    arrivals = np.cumsum(rng.exponential(30.0, 3000))
    service = rng.exponential(100.0, 3000)
    free = [0.0] * 4
    expected = []
    for a, s in zip(arrivals, service, strict=True):
        start = max(a, heapq.heappop(free))
        expected.append(start - a)
        heapq.heappush(free, start + s)
    jobs = [Job(i, float(a), float(s), ((1, 100),)) for i, (a, s) in enumerate(zip(arrivals, service, strict=True))]
    one_type = SpeedTable(("p",), ("p",), np.ones((1, 1)))
    Simulator(Cluster([4]), one_type, jobs, BlindFifo(0), window=(0.0, float(arrivals[-1]))).run()
    assert np.allclose([j.start - j.submit for j in jobs], expected, atol=1e-9)


def test_utilisation_integral():
    jobs = [Job(0, 0.0, 50.0, ((1, 50),))]
    sim = Simulator(Cluster([0, 2]), two_types(), jobs, AwareGreedy(), window=(0.0, 100.0)).run()
    # half a GPU of two, busy for 50 of 100 seconds
    assert sim.used_area[1] / (sim.cluster.capacity[1] * 100.0) == pytest.approx(0.125)


def test_migration_pays_the_restore_cost_and_keeps_progress():
    # job A starts on the slow type because the fast GPU is taken by B; B ends at 10, A moves at its next check
    a = Job(0, 0.0, 100.0, ((1, 100),))
    b = Job(1, 0.0, 10.0, ((1, 100),))
    policy = Migrating(AwareGreedy(), cost=5.0, every=0.0)
    Simulator(Cluster([1, 1]), two_types(), [b, a], policy, window=(0.0, 1.0)).run()
    assert a.migrations == 1 and a.gtype == 1
    # 10 s on the slow type at 0.25 -> 2.5 done, 97.5 left; 5 s restore; then 97.5 s at speed 1
    assert a.end == pytest.approx(10.0 + 5.0 + 97.5)


def test_cost_aware_migration_skips_moves_that_do_not_pay():
    a = Job(0, 0.0, 100.0, ((1, 100),))
    b = Job(1, 0.0, 390.0, ((1, 100),))  # frees the fast GPU at 390, when A has 2.5 work (10 s here) left
    policy = Migrating(AwareGreedy(), cost=60.0, every=0.0)
    Simulator(Cluster([1, 1]), two_types(), [b, a], policy, window=(0.0, 1.0)).run()
    assert a.migrations == 0 and a.end == pytest.approx(400.0)
    a2, b2 = Job(0, 0.0, 100.0, ((1, 100),)), Job(1, 0.0, 390.0, ((1, 100),))
    always = Migrating(AwareGreedy(), cost=60.0, cost_aware=False, every=0.0)
    Simulator(Cluster([1, 1]), two_types(), [b2, a2], always, window=(0.0, 1.0)).run()
    assert a2.migrations == 1 and a2.end == pytest.approx(390.0 + 60.0 + 2.5)  # later than staying
