import numpy as np
import pytest

from fleetsim import speeds
from fleetsim.experiments import Scenario, build, compare, entry, run_one
from fleetsim.stats import paired_change
from fleetsim.trace import SAMPLE, load_jobs, parse_demand


def test_speed_table_is_relative_to_v100():
    table = speeds.load()
    assert table.speed.shape == (26, 5)
    v100, v100m32 = speeds.GPU_TYPES.index("V100"), speeds.GPU_TYPES.index("V100M32")
    assert np.allclose(table.speed[:, v100], 1.0) and np.allclose(table.speed[:, v100m32], 1.0)
    t4 = speeds.GPU_TYPES.index("T4")
    resnet = table.profiles_of("ResNet-50")[0]
    assert table.speed[resnet, t4] == pytest.approx(1 / 3.6)


def test_gamma_zero_makes_every_gpu_equal_and_uniform_makes_every_job_equal():
    table = speeds.load()
    assert np.allclose(speeds.scaled(table, 0.0).speed, 1.0)
    flat = speeds.uniform(table).speed
    assert np.allclose(flat, flat[0])


def test_parse_demand():
    assert parse_demand("2x50;1x100") == ((2, 50), (1, 100))


def test_sample_builds_and_runs():
    jobs = load_jobs(SAMPLE)
    assert len(jobs) > 1000
    fleet, _, built, window = build(Scenario(0, sample=True))
    assert len(fleet) == 5 and built and window[1] > window[0]
    # the same seed gives the same profiles whatever the variant
    _, _, again, _ = build(Scenario(0, sample=True, gamma=0.5))
    assert [j.profile for j in built] == [j.profile for j in again]
    result = run_one((Scenario(0, sample=True), "aware-greedy", False))
    assert result["metrics"]["jobs"] > 0 and result["metrics"]["unfinished"] == 0


def test_paired_change_and_compare():
    out = paired_change([100.0, 200.0], [50.0, 150.0])
    assert out["mean"] == pytest.approx(-0.375) and out["b_lower_in"] == 2
    assert paired_change([0.0, 1.0], [1.0, 1.0])["mean"] is None
    runs = [
        {
            "label": entry(p)[0],
            "scenario": {"seed": s},
            "metrics": {m: v for m in ("mean_jct", "p95_jct", "mean_wait", "makespan")},
        }
        for s in range(3)
        for p, v in (
            ("blind-backfill", 2.0),
            ("aware-greedy", 1.0),
            ("rank-greedy", 1.5),
            ("aware-wait", 1.0),
            ("aware-lp", 1.0),
            ("blind-fifo", 2.0),
        )
    ]
    first = compare(runs, "main")[0]
    assert first["mean_jct"]["mean"] == pytest.approx(-0.5)


def test_build_jobs_from_tiny_raw_tables(tmp_path):
    from fleetsim.trace import build_jobs, fleet_from_machines

    (tmp_path / "pai_job_table.csv").write_text(
        "j1,i1,u,Terminated,100.0,500.0\n"
        "j2,i2,u,Failed,200.0,300.0\n"
        "j3,i3,u,Running,300.0,\n"  # still running at the end of the trace: dropped
        "j4,i4,u,Terminated,400.0,900.0\n"  # CPU only: dropped
    )
    (tmp_path / "pai_task_table.csv").write_text(
        "j1,worker,2.0,Terminated,110.0,400.0,600.0,29.0,50.0,T4\n"
        "j1,ps,1.0,Terminated,105.0,420.0,600.0,29.0,25.0,V100\n"
        "j2,tensorflow,1.0,Failed,210.0,260.0,600.0,29.0,100.0,MISC\n"
        "j3,tensorflow,1.0,Running,310.0,,600.0,29.0,100.0,P100\n"
        "j4,cpu,1.0,Terminated,410.0,800.0,600.0,29.0,,\n"
    )
    (tmp_path / "pai_group_tag_table.csv").write_text("i1,u,,g,bert\ni2,u,V100,g,\n")
    (tmp_path / "pai_machine_spec.csv").write_text(
        "m0,CPU,96,512,0\nm1,MISC,96,512,8\nm2,T4,96,512,2\nm3,P100,96,512,4\nm4,V100,96,512,8\nm5,V100M32,96,512,8\n"
    )
    jobs = build_jobs(tmp_path)
    assert jobs.to_dict("records") == [
        # submit, first GPU task start to last GPU task end, type of the largest GPU task, all GPU tasks
        {"submit": 100.0, "duration": 315.0, "ran_on": "T4", "demand": "2x50;1x25", "pinned": "", "tag": "bert"},
        {"submit": 200.0, "duration": 50.0, "ran_on": "MISC", "demand": "1x100", "pinned": "V100", "tag": ""},
    ]
    assert fleet_from_machines(tmp_path).gpus.tolist() == [8, 2, 4, 8, 8]
