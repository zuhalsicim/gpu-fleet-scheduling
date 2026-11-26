import numpy as np
import pytest

from fleetsim.cluster import Cluster
from fleetsim.policies import AwareGreedy, AwareLP, AwareWait, BlindBackfill, BlindFifo, RankGreedy
from fleetsim.simulator import Job, Simulator
from fleetsim.speeds import SpeedTable


def table(rows):
    rows = np.array(rows, dtype=float)
    names = tuple(f"p{i}" for i in range(len(rows)))
    return SpeedTable(names, names, rows)


def run(fleet, speeds, jobs, policy):
    return Simulator(Cluster(fleet), speeds, jobs, policy, window=(0.0, 1.0)).run()


def test_greedy_takes_the_fastest_type_for_each_job():
    speeds = table([[1.0, 0.5], [0.5, 1.0]])  # profile 0 prefers type 0, profile 1 type 1
    jobs = [Job(0, 0.0, 10.0, ((1, 100),), profile=1), Job(1, 0.0, 10.0, ((1, 100),), profile=0)]
    run([1, 1], speeds, jobs, AwareGreedy())
    assert [j.gtype for j in jobs] == [1, 0]


def test_rank_greedy_uses_the_average_ranking_only():
    speeds = table([[1.0, 0.1], [0.9, 1.0]])  # on average type 0 is faster; profile 1 prefers type 1
    job = Job(0, 0.0, 10.0, ((1, 100),), profile=1)
    run([1, 1], speeds, [job], RankGreedy())
    assert job.gtype == 0


def test_pinned_jobs_stay_on_their_type():
    speeds = table([[1.0, 0.1]])
    job = Job(0, 0.0, 10.0, ((1, 100),), pinned=1)
    run([1, 1], speeds, [job], AwareGreedy())
    assert job.gtype == 1


def test_strict_fifo_blocks_behind_the_head_but_backfill_does_not():
    speeds = table([[1.0]])

    def jobs():
        return [Job(0, 0.0, 100.0, ((1, 100),)), Job(1, 1.0, 10.0, ((2, 100),)), Job(2, 2.0, 10.0, ((1, 100),))]

    fifo = jobs()
    run([2], speeds, fifo, BlindFifo(0))
    assert fifo[2].start == pytest.approx(110.0)  # waits for the two-GPU job ahead of it
    backfill = jobs()
    run([2], speeds, backfill, BlindBackfill(0))
    assert backfill[2].start == pytest.approx(2.0)


def test_wait_policy_waits_for_a_busy_fast_gpu_when_that_finishes_sooner():
    speeds = table([[0.1, 1.0]])  # type 1 is ten times faster
    blocker = Job(0, 0.0, 5.0, ((1, 100),))  # holds the fast GPU until t = 5
    job = Job(1, 1.0, 10.0, ((1, 100),))  # 100 s on the slow GPU, 10 s on the fast one
    run([1, 1], speeds, [blocker, job], AwareWait())
    assert job.gtype == 1 and job.start == pytest.approx(5.0)
    blocker2, job2 = Job(0, 0.0, 500.0, ((1, 100),)), Job(1, 1.0, 10.0, ((1, 100),))
    run([1, 1], speeds, [blocker2, job2], AwareWait())
    assert job2.gtype == 0 and job2.start == pytest.approx(1.0)  # waiting 499 s would not pay


def test_lp_prices_send_the_job_that_gains_least_to_the_slow_type():
    # profile 0 gains 10x from type 1, profile 1 only 1.25x; one fast GPU, one slow GPU
    speeds = table([[0.1, 1.0], [0.8, 1.0]])
    low_gain = Job(0, 0.0, 10.0, ((1, 100),), profile=1)
    high_gain = Job(1, 0.0, 10.0, ((1, 100),), profile=0)
    run([1, 1], speeds, [low_gain, high_gain], AwareLP())
    assert (low_gain.gtype, high_gain.gtype) == (0, 1)
    greedy_low, greedy_high = Job(0, 0.0, 10.0, ((1, 100),), profile=1), Job(1, 0.0, 10.0, ((1, 100),), profile=0)
    run([1, 1], speeds, [greedy_low, greedy_high], AwareGreedy())
    assert (greedy_low.gtype, greedy_high.gtype) == (1, 0)  # first come takes the fast GPU
