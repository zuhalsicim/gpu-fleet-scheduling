from fleetsim.cluster import Cluster, total, whole_gpus


def test_total_counts_large_instances_as_whole_gpus():
    assert total(((2, 50),)) == 100
    assert total(((1, 150),)) == 200
    assert whole_gpus(((3, 200), (4, 25))) == 6


def test_fractional_jobs_share_a_gpu_and_best_fit_keeps_whole_gpus_free():
    c = Cluster([2])
    a = c.place(((1, 50),), 0)
    b = c.place(((1, 25),), 0)
    assert {g for g, _ in a} == {g for g, _ in b}  # the quarter goes onto the half-used GPU
    assert len(c.whole[0]) == 1
    assert c.free_total[0] == 125


def test_failed_placement_changes_nothing():
    c = Cluster([2])
    c.place(((1, 60),), 0)
    c.place(((1, 60),), 0)
    before = (list(c.free), c.free_total[0])
    assert c.place(((1, 50),), 0) is None  # 80% free in total, but only 40% on each GPU
    assert (list(c.free), c.free_total[0]) == before


def test_release_restores_capacity_and_bumps_the_counter():
    c = Cluster([4, 2])
    alloc = c.place(((2, 100),), 1)
    assert c.free_total[1] == 0 and not c.may_fit(((1, 10),), 1)
    c.release(alloc)
    assert c.free_total[1] == 200 and c.release_count == [0, 1]


def test_whole_gpu_instances_need_free_whole_gpus():
    c = Cluster([2])
    c.place(((1, 10),), 0)
    assert c.may_fit(((1, 100),), 0)
    assert not c.may_fit(((2, 100),), 0)
    assert c.fits(((1, 100), (1, 90)), 0)  # one whole GPU plus 90% on the GPU that has 90% left
