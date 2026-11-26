"""The simulator against exact queueing results (smaller versions of fleetsim.validate)."""

import numpy as np
import pytest

from fleetsim.queueing import erlang_c, mg1_wait, mmc_wait, simulate_queue


def test_erlang_c_known_values():
    assert erlang_c(1, 0.6) == pytest.approx(0.6)  # M/M/1: P(wait) = rho
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)
    assert mmc_wait(0.5, 1.0, 1) == pytest.approx(1.0)  # rho / (mu - lam)
    assert mg1_wait(0.5, 1.0, 1.0) == pytest.approx(0.5)  # M/D/1 at rho 0.5


def replicate(servers, percent, rho, service, reps=12, jobs=8000):
    c = servers * 100 // percent
    lam = rho * c / 100.0
    out = [
        simulate_queue(lam, service(np.random.default_rng([7, r]), jobs), servers, percent, seed=r) for r in range(reps)
    ]
    waits = np.array([o["mean_wait"] for o in out])
    return lam, waits.mean(), waits.std(ddof=1) / np.sqrt(reps), out


def exponential(rng, n):
    return rng.exponential(100.0, n)


@pytest.mark.parametrize(("servers", "percent", "rho"), [(1, 100, 0.5), (4, 100, 0.7), (2, 50, 0.7)])
def test_mmc_mean_wait_matches_erlang_c(servers, percent, rho):
    lam, mean, se, _ = replicate(servers, percent, rho, exponential)
    expected = mmc_wait(lam, 1 / 100.0, servers * 100 // percent)
    assert abs(mean - expected) < 3.5 * se


def test_md1_mean_wait_matches_pollaczek_khinchine():
    lam, mean, se, _ = replicate(1, 100, 0.6, lambda rng, n: np.full(n, 100.0))
    assert abs(mean - mg1_wait(lam, 100.0, 100.0**2)) < 3.5 * se


def test_littles_law():
    _, _, _, out = replicate(4, 100, 0.8, exponential, reps=4)
    for o in out:
        assert o["L"] == pytest.approx(o["lambda_W"], rel=0.02)
