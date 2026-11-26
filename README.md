# gpu-fleet-scheduling

**Does it pay for a scheduler to know the GPU type?** On a fleet that mixes old and new GPUs, how much faster do jobs finish when the scheduler knows which job runs best on which GPU type, compared with a scheduler that places jobs blind to the type?

This repository answers that with a small discrete-event simulator. It replays a real production GPU cluster: the Alibaba PAI trace from 2020, with 6,742 GPUs of five types and about 720,000 GPU jobs over two months. Before any policy is compared, the simulator is checked against exact results from queueing theory.

## The answer

<!-- table:headline -->
- **On the trace's own fleet and load, knowing the GPU type pays a lot.** A scheduler that sends each job to its fastest free GPU type (`aware-greedy`) cuts the mean job completion time by **53% (95% CI 52 to 54%)** against a type-blind scheduler (`blind-backfill`): from 83.3 to 39.3 minutes, lower in 10 of 10 paired seeds. The p95 completion time falls by 52% (95% CI 50 to 53%).
- **Most of it needs only a ranking of the types.** Always trying the type that is fastest on average first (`rank-greedy`) gives 40% (95% CI 39 to 40%). Knowing each job's own speeds cuts another 22% (95% CI 21 to 23%) off `rank-greedy`.
- **On a busy fleet, placing greedily is not enough.** With 30% of the GPUs, where jobs wait 48.6 minutes on average under `blind-backfill`, `aware-greedy` gains only 15% (95% CI 8 to 22%). Waiting for a busy fast GPU when that finishes the job sooner (`aware-wait`) gains 35% (95% CI 33 to 38%).
<!-- /table:headline -->

Two facts carry these numbers:

1. **At the trace's own load, almost nothing waits.** The mean queueing delay is under one second for every policy. So the gain is not shorter queues. A type-blind scheduler keeps the old GPUs half full (56% of MISC and T4 capacity in use) while the V100s sit two-thirds idle (35% in use). A type-aware one runs every job on its fastest type.
2. **The speeds are assumed.** The trace records where a job ran and for how long, not how fast it would have run elsewhere. The per-type speeds come from published benchmarks (below). With every speed gap halved, the gain falls from 53% to 30.5%; with no gaps it is zero (see [Sensitivity](#sensitivity)).

## Method

```
 Alibaba trace (measured)           published benchmarks (assumed)
 submit time, duration, GPU %,      Gavel: K80/P100/V100 throughput of 26 job types
 type it ran on, pinned type        MLPerf (Dell): V100-to-T4 ratio per model family
            |                                   |
            v                                   v
   jobs: work = duration x speed(type it ran on), in V100-seconds
            |
            v
 +-------------------------- simulator.py --------------------------+
 |  event queue (heap)  ->  arrival | finish | sample tick           |
 |        |                                                          |
 |        v                                                          |
 |  cluster.py: GPUs per type, free % per GPU, best-fit packing      |
 |        ^                                                          |
 |        |  start(job, type) / migrate(job, type, cost)             |
 |  policies.py: blind-fifo, blind-backfill, rank-greedy,            |
 |               aware-greedy, aware-wait, aware-lp, +migration      |
 +-------------------------------------------------------------------+
            |
            v
 metrics.py -> results/*.json -> scripts/make_tables.py -> the tables in this README
                              -> dashboard/ (replay of two days)
```

Each job has a fixed amount of work, measured in V100-seconds. On GPU type *t* it runs at `speed[job type, t]`, so it takes `work / speed` seconds there. The core is three files of about 600 lines, comments included: `fleetsim/cluster.py` (who has room), `fleetsim/simulator.py` (time and events) and `fleetsim/policies.py` (the decisions).

**GPU sharing.** Capacity is counted in percent of a GPU, as in the trace: a job may ask for 25% or 50% of a GPU, and several such jobs share one GPU. Requests of 100% or more take whole GPUs. A fractional request goes to the fullest GPU it still fits on (best fit), so whole GPUs stay free for jobs that need them. All parts of a job go on one GPU type.

**Preemption with a checkpoint cost.** A running job can be paused, checkpointed and restored on a faster GPU. The move costs `c` seconds without progress. With `r` seconds of work left on the slow GPU (speed *s1*), moving to speed *s2* pays exactly when

```
r > c / (1 - s1 / s2)
```

So a job that would run twice as fast must have more than `2c` seconds left; a job that gains 10% needs more than `11c`.

**Policies compared**

| policy | knows speeds? | rule |
|---|---|---|
| `blind-fifo` | no | strict arrival order; the first job that does not fit blocks the rest; any GPU type, picked at random in proportion to its free capacity |
| `blind-backfill` | no | the same, but later jobs that fit may start while earlier ones wait (**the type-blind baseline**) |
| `rank-greedy` | only the average ranking of types | the best-ranked type that has room now, the same order for every job |
| `aware-greedy` | yes, per job | the fastest type for this job that has room now; never waits (**the type-aware policy of the headline**) |
| `aware-wait` | yes, plus running jobs' finish times | waits for a busy faster type when that is expected to finish the job sooner |
| `aware-lp` | yes, per job | a linear program over the whole fleet gives each GPU type a price; a job takes the type with the best value for money and waits (at most 30 minutes) if every free type costs more than it is worth to it |
| `+migrate@c` | | every 2 minutes, move running jobs to a faster type that has room, paying `c` seconds, only when the rule above says it pays |
| `+migrate-always@c` | | the same, but ignores the cost |

The headline pair (`aware-greedy` against `blind-backfill`) was fixed before the full run: both scan the queue the same way, so the only difference is knowing the type. All policies scan the queue in arrival order, up to 1,000 jobs deep, and share the same packing rule. Jobs that pinned a GPU type in the trace (1.3%) stay on that type under every policy.

**The allocation LP** (`aware-lp`, in the style of Gavel). With *x[p, t]* the capacity of type *t* given to jobs of type *p*, *v[p, t]* the speed of *p* on *t* divided by its best speed, *C[t]* the GPUs of type *t*, and *D[p]* the capacity that running and waiting jobs of type *p* ask for:

```
maximise    sum over p, t of  v[p, t] * x[p, t]
subject to  sum over p of x[p, t] <= C[t]     for every GPU type   (dual: price of type t)
            sum over t of x[p, t] <= D[p]     for every job type
            x >= 0
```

It has 130 variables and is solved with HiGHS (`scipy.optimize.linprog`) in a few milliseconds, every 6 minutes. By LP duality, a job type uses type *t* in the optimum only where `v[p, t] - price[t]` is highest. The policy applies exactly that rule to each waiting job.

### Data and assumptions

**Measured, from the trace** (`cluster-trace-gpu-v2020`): each job's submit time; its duration (first GPU task start to last GPU task end); its GPU request (instances times percent of a GPU, which gives the fractional requests); the GPU type it ran on; whether the user pinned a GPU type; the workload tag where present (12% of jobs: bert, ctr, nmt, resnet, ...); and the fleet: MISC 2,240, P100 1,596, V100M32 1,080, T4 994 and V100 832 GPUs. Jobs that ended as Terminated or Failed are kept, since both occupied GPUs.

**Assumed, from published sources:**

1. **How fast each job type runs on each GPU type.** Gavel's measured single-GPU training throughput for 26 job types on K80, P100 and V100 (Narayanan et al., OSDI 2020; `data/speeds/`). The trace's MISC type ("older generations such as K40m, K80, M60") is treated as a K80, and V100M32 as a V100. Gavel's table is used as measured, including its few rows where an older GPU is faster: the recommendation model runs 1.95 to 2.3 times faster on a P100 than on a V100, and up to 1.5 times faster on a K80.
2. **T4.** Gavel has none. A T4's speed is the V100's divided by the V100-PCIe-to-T4 ratio from MLPerf Training v0.5 runs (Dell EMC, 2019): 3.6 for ResNets, 2.7 for Transformer, 3.05 for recurrent models, and 2.81 (the geometric mean of the article's 2.2 to 3.6 range) where no benchmark matches.
3. **Which job type each trace job is.** Tagged jobs draw from the matching Gavel family (bert, xlnet, nmt: Transformer; ctr: Recommendation; resnet, inception, vgg: ResNet-50; rl: A3C). All other jobs draw uniformly from the 26 job types, as Gavel's own simulations do. The draw depends only on the seed, so every policy sees the same jobs.
4. **A job's work** is its trace duration times its speed on the type it actually ran on. Fractional and multi-GPU jobs use the same per-type ratios as single-GPU training.

**Replay window.** Seed *k* replays trace day 12 + 6*k* as warm-up (jobs run but are not counted) and the next two days as the measured window. So ten seeds cover ten separate stretches of the two months, with 18,000 to 30,000 measured jobs each (23,500 on average). The whole fleet and every job in a window are simulated; nothing is subsampled. Each run continues until every job has finished. Changes between two policies are computed per seed and reported with a 95% bootstrap interval over the ten seeds.

## Validation: does the simulator get queueing right?

Before any policy comparison, the simulator runs as a plain queue: one GPU type, Poisson arrivals, one job per GPU (or per share of a GPU), strict first come, first served. Here exact answers exist:

- **M/M/c (Erlang C):** mean wait in the queue `Wq = C(c, a) / (c * mu - lambda)`, where `C(c, a)` is the probability of waiting and `a = lambda / mu` the offered load.
- **M/D/1 (Pollaczek-Khinchine):** `Wq = lambda * E[S^2] / (2 * (1 - rho))`.
- **Little's law:** the time-average number of jobs in the system `L` equals `lambda * W`.

<!-- table:validation -->
| case | utilisation | Wq theory (s) | Wq simulated (s), 95% CI | z | L (time average) | lambda x W |
|---|---:|---:|---:|---:|---:|---:|
| M/M/1 | 0.50 | 100.00 | 100.41 +- 1.33 | +0.60 | 1.002 | 1.002 |
| M/M/4 | 0.80 | 74.55 | 74.41 +- 2.47 | -0.11 | 5.573 | 5.574 |
| M/M/16 | 0.90 | 36.96 | 37.98 +- 3.14 | +0.63 | 19.754 | 19.771 |
| M/M/8 as 4 GPUs shared by two | 0.80 | 28.60 | 28.76 +- 1.13 | +0.27 | 8.226 | 8.229 |
| M/D/1 | 0.70 | 116.67 | 116.63 +- 1.30 | -0.06 | 1.517 | 1.517 |

40 replications of 20,000 jobs per case, first 10% of each discarded; mean service time 100 s.
<!-- /table:validation -->

`z` is the distance between simulation and theory in standard errors. All five cases are within one standard error, and `L` and `lambda * W` agree within 0.1%. The fourth case runs four GPUs with jobs that each ask for half a GPU, so it also checks the GPU-sharing code: it must behave as eight servers, and it does. In addition, `tests/test_simulator.py` checks the FIFO simulator against the exact Kiefer-Wolfowitz recursion, job by job, for 3,000 jobs on four GPUs. Smaller versions of all these checks run in the tests on every push.

## Results

Means over ten seeds. **JCT** is job completion time (finish minus submit). **Queueing delay** is first start minus submit. **Makespan** runs from the start of the measured window to the last measured job's finish; it is dominated by the single longest job. **Speed fraction** is a job's best possible runtime divided by its actual runtime. **Jain's index** is computed over slowdowns (JCT divided by the best possible runtime); 1 means every job was served equally well. **GPU use** is the share of capacity allocated, averaged over the measured window. **Waiting due to fragmentation** is the share of waiting jobs (sampled every 10 minutes) that some type had enough free capacity for in total, but scattered over GPUs so that the job could not be placed. Changes are paired per seed; brackets are 95% bootstrap intervals over seeds.

### The trace's own fleet and load

<!-- table:main -->
| policy | mean JCT (min) | p95 JCT (min) | mean queueing delay (min) | makespan (days) | speed fraction | Jain (slowdown) | GPU use | use per type, % (MISC / T4 / P100 / V100 / V100M32) | waiting due to fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind-fifo | 83.3 | 327.2 | 0.0 | 19.5 | 0.64 | 0.58 | 47% | 56 / 56 / 37 / 35 / 43 | 0.0% |
| blind-backfill | 83.3 | 327.2 | 0.0 | 19.5 | 0.64 | 0.58 | 47% | 56 / 56 / 37 / 35 / 43 | 0.0% |
| rank-greedy | 50.2 | 200.4 | 0.0 | 8.5 | 0.88 | 0.85 | 30% | 0 / 1 / 28 / 93 / 77 | 8.0% |
| aware-greedy | 39.3 | 158.6 | 0.0 | 6.9 | 1.00 | 1.00 | 25% | 0 / 0 / 38 / 82 / 40 | 1.5% |
| aware-wait | 39.3 | 158.6 | 0.0 | 6.9 | 1.00 | 1.00 | 25% | 0 / 0 / 37 / 82 / 41 | 1.3% |
| aware-lp | 39.3 | 158.6 | 0.0 | 6.9 | 1.00 | 1.00 | 25% | 0 / 0 / 38 / 82 / 40 | 1.3% |

| candidate vs baseline | mean JCT | p95 JCT | mean queueing delay (min) | seeds where mean JCT is lower |
|---|---:|---:|---:|---:|
| aware-greedy vs blind-backfill | -52.9% [-53.6, -52.2] | -51.7% [-53.3, -49.9] | +0.0 [+0.0, +0.0] | 10 of 10 |
| rank-greedy vs blind-backfill | -39.8% [-40.4, -39.3] | -38.9% [-40.8, -36.9] | +0.0 [+0.0, +0.0] | 10 of 10 |
| aware-greedy vs rank-greedy | -21.7% [-22.6, -20.7] | -20.9% [-22.4, -19.5] | -0.0 [-0.0, -0.0] | 10 of 10 |
| aware-wait vs blind-backfill | -52.9% [-53.6, -52.2] | -51.7% [-53.3, -49.9] | +0.0 [+0.0, +0.0] | 10 of 10 |
| aware-lp vs blind-backfill | -52.9% [-53.6, -52.2] | -51.7% [-53.3, -49.9] | +0.0 [+0.0, +0.0] | 10 of 10 |
| blind-backfill vs blind-fifo | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 0 of 10 |
<!-- /table:main -->

At this load the three type-aware policies give practically the same result: with free fast GPUs nearly always available, waiting or pricing almost never changes a decision.

### A busier fleet

The trace's cluster is lightly loaded. To see where the policies differ, the same jobs are replayed on a fleet with 50% and 30% of the GPUs (every type shrunk equally).

<!-- table:load -->
**Fleet at 50% of its GPUs**

| policy | mean JCT (min) | p95 JCT (min) | mean queueing delay (min) | makespan (days) | speed fraction | Jain (slowdown) | GPU use | use per type, % (MISC / T4 / P100 / V100 / V100M32) | waiting due to fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind-fifo | 172.9 | 429.4 | 93.6 | 23.3 | 0.66 | 0.07 | 85% | 89 / 88 / 81 / 79 / 83 | 2.3% |
| blind-backfill | 82.5 | 327.3 | 2.6 | 21.9 | 0.66 | 0.23 | 86% | 90 / 88 / 82 / 81 / 84 | 26.4% |
| rank-greedy | 63.0 | 242.0 | 2.5 | 17.4 | 0.84 | 0.26 | 78% | 61 / 76 / 83 / 95 / 94 | 34.7% |
| aware-greedy | 47.3 | 183.1 | 0.7 | 11.4 | 0.97 | 0.59 | 65% | 38 / 58 / 81 / 90 / 82 | 18.2% |
| aware-wait | 45.3 | 182.5 | 5.9 | 7.1 | 0.97 | 0.29 | 47% | 1 / 2 / 84 / 95 / 94 | 1.0% |
| aware-lp | 50.5 | 199.5 | 0.3 | 8.4 | 0.89 | 0.67 | 63% | 44 / 56 / 78 / 84 / 71 | 10.9% |

| candidate vs baseline | mean JCT | p95 JCT | mean queueing delay (min) | seeds where mean JCT is lower |
|---|---:|---:|---:|---:|
| rank-greedy vs blind-backfill | -24.0% [-27.2, -20.0] | -26.5% [-30.4, -21.7] | -0.1 [-0.6, +0.5] | 10 of 10 |
| aware-greedy vs blind-backfill | -43.0% [-46.3, -38.9] | -44.5% [-47.2, -41.3] | -1.9 [-3.8, -0.5] | 10 of 10 |
| aware-wait vs blind-backfill | -45.2% [-47.2, -43.3] | -44.5% [-46.4, -42.5] | +3.3 [+2.0, +4.8] | 10 of 10 |
| aware-lp vs blind-backfill | -39.1% [-42.4, -35.9] | -39.6% [-43.2, -36.0] | -2.4 [-5.1, -0.5] | 10 of 10 |
| aware-wait vs aware-greedy | -3.4% [-7.4, -0.1] | +0.4% [-3.4, +4.0] | +5.2 [+3.2, +7.3] | 8 of 10 |
| aware-lp vs aware-greedy | +7.4% [+1.8, +12.4] | +9.0% [+3.5, +14.6] | -0.5 [-1.3, -0.0] | 1 of 10 |
| blind-backfill vs blind-fifo | -32.3% [-47.9, -18.3] | -14.7% [-27.4, -4.9] | -91.0 [-200.4, -22.9] | 10 of 10 |

**Fleet at 30% of its GPUs**

| policy | mean JCT (min) | p95 JCT (min) | mean queueing delay (min) | makespan (days) | speed fraction | Jain (slowdown) | GPU use | use per type, % (MISC / T4 / P100 / V100 / V100M32) | waiting due to fragmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blind-fifo | 1,729.1 | 2,743.7 | 1,648.9 | 21.6 | 0.66 | 0.15 | 91% | 94 / 91 / 89 / 87 / 90 | 0.7% |
| blind-backfill | 126.0 | 523.1 | 48.6 | 21.6 | 0.68 | 0.03 | 98% | 99 / 98 / 98 / 97 / 98 | 29.7% |
| rank-greedy | 122.7 | 506.1 | 53.6 | 16.6 | 0.78 | 0.03 | 98% | 99 / 98 / 98 / 97 / 98 | 29.6% |
| aware-greedy | 109.6 | 466.6 | 47.8 | 17.0 | 0.84 | 0.03 | 97% | 97 / 96 / 97 / 97 / 97 | 33.1% |
| aware-wait | 82.4 | 305.9 | 42.6 | 8.2 | 0.90 | 0.11 | 57% | 8 / 19 / 98 / 98 / 99 | 1.2% |
| aware-lp | 89.2 | 357.2 | 32.4 | 13.4 | 0.85 | 0.03 | 96% | 96 / 95 / 96 / 95 / 96 | 28.2% |

| candidate vs baseline | mean JCT | p95 JCT | mean queueing delay (min) | seeds where mean JCT is lower |
|---|---:|---:|---:|---:|
| rank-greedy vs blind-backfill | -4.8% [-9.2, +0.4] | -5.2% [-10.9, +1.3] | +5.0 [+0.8, +10.1] | 7 of 10 |
| aware-greedy vs blind-backfill | -15.0% [-21.5, -8.1] | -13.4% [-22.1, -2.8] | -0.9 [-8.8, +6.5] | 9 of 10 |
| aware-wait vs blind-backfill | -35.3% [-37.7, -32.9] | -41.2% [-45.9, -36.8] | -6.0 [-12.8, +0.2] | 10 of 10 |
| aware-lp vs blind-backfill | -29.9% [-34.2, -25.0] | -31.7% [-36.4, -26.4] | -16.3 [-25.0, -8.9] | 10 of 10 |
| aware-wait vs aware-greedy | -22.9% [-28.5, -17.2] | -30.1% [-39.5, -20.7] | -5.1 [-12.9, +1.7] | 10 of 10 |
| aware-lp vs aware-greedy | -16.8% [-22.3, -10.3] | -19.2% [-27.9, -9.5] | -15.4 [-20.3, -9.6] | 8 of 10 |
| blind-backfill vs blind-fifo | -92.1% [-93.2, -91.0] | -80.0% [-83.0, -77.2] | -1600.2 [-2046.9, -1175.6] | 10 of 10 |
<!-- /table:load -->

On the busiest fleet, almost every GPU is in use whatever the policy, so a greedy policy takes whatever is free and its type knowledge helps little. What helps is leaving a slow GPU idle when waiting for a fast one finishes the job sooner: `aware-wait` keeps MISC and T4 mostly idle and still finishes jobs much earlier. Note that `aware-wait` uses the exact remaining work of running jobs (see Limits).

### Preemption with checkpoint/restore

Costs of 30 seconds, 5 minutes and 30 minutes per move. The `migrate` policies apply the rule above with the simulator's exact remaining work; the `migrate-always` policies move whenever a faster type has room.

<!-- table:migration -->
| candidate vs baseline | mean JCT (min) | mean JCT change | p95 JCT change | moves per seed | jobs moved |
|---|---:|---:|---:|---:|---:|
| blind-backfill+migrate@30 vs blind-backfill | 40.3 | -51.8% [-52.4, -51.1] | -51.3% [-52.9, -49.6] | 13,987 | 59.3% |
| blind-backfill+migrate@300 vs blind-backfill | 42.4 | -49.1% [-49.7, -48.6] | -50.3% [-51.8, -48.7] | 9,662 | 40.9% |
| blind-backfill+migrate@1800 vs blind-backfill | 49.7 | -40.3% [-40.9, -39.8] | -44.1% [-45.4, -42.7] | 5,234 | 22.3% |
| blind-backfill+migrate-always@30 vs blind-backfill | 40.3 | -51.8% [-52.4, -51.1] | -51.3% [-53.0, -49.7] | 15,931 | 67.7% |
| blind-backfill+migrate-always@300 vs blind-backfill | 43.3 | -48.0% [-48.5, -47.5] | -50.2% [-51.8, -48.5] | 16,002 | 68.0% |
| blind-backfill+migrate-always@1800 vs blind-backfill | 60.8 | -26.9% [-27.9, -25.8] | -43.4% [-44.8, -42.1] | 15,988 | 68.0% |
| aware-greedy vs blind-backfill | 39.3 | -52.9% [-53.6, -52.2] | -51.7% [-53.3, -49.9] | 0 | 0.0% |
| aware-greedy+migrate@30, fleet 50% vs aware-greedy, fleet 50% | 42.3 | -9.4% [-14.2, -5.4] | -4.9% [-8.2, -2.0] | 1,931 | 7.6% |
| aware-greedy+migrate@300, fleet 50% vs aware-greedy, fleet 50% | 42.8 | -8.4% [-12.6, -4.7] | -3.6% [-6.2, -1.2] | 1,524 | 6.0% |
| aware-greedy+migrate@1800, fleet 50% vs aware-greedy, fleet 50% | 44.4 | -5.0% [-9.5, -1.2] | +1.5% [-1.2, +3.9] | 986 | 3.9% |
| aware-greedy+migrate-always@30, fleet 50% vs aware-greedy, fleet 50% | 42.3 | -9.4% [-13.8, -5.5] | -4.7% [-7.7, -2.0] | 2,049 | 8.1% |
| aware-greedy+migrate-always@300, fleet 50% vs aware-greedy, fleet 50% | 43.0 | -8.0% [-12.5, -4.2] | -3.9% [-7.1, -1.0] | 2,172 | 8.6% |
| aware-greedy+migrate-always@1800, fleet 50% vs aware-greedy, fleet 50% | 46.4 | -0.9% [-4.9, +2.6] | +4.2% [+1.6, +7.0] | 2,374 | 9.5% |
<!-- /table:migration -->

For a blind scheduler on the trace's fleet, moving jobs up when fast GPUs are free recovers almost the whole gain of type-aware placement when moves are cheap (51.8% at 30 seconds, against 52.9% for `aware-greedy`). Ignoring the cost is harmless when moves are cheap, but at 30 minutes per move it gives away a third of the gain. On the half-size fleet, cost-aware moves still add 5 to 9% to `aware-greedy`; ignoring a 30-minute cost gives nothing and makes the p95 worse.

### GPU sharing

The trace's jobs often ask for part of a GPU (25% and 50% are common). With sharing off, every fractional request is rounded up to a whole GPU.

<!-- table:sharing -->
| policy | mean JCT (min), off -> on | mean JCT change | mean queueing delay change (min) | GPU capacity in use, off -> on | waiting due to fragmentation, off -> on |
|---|---:|---:|---:|---:|---:|
| blind-backfill | 79.4 -> 83.3 | +5.3% [+2.1, +8.0] | -1.0 [-2.0, -0.2] | 82% -> 47% | 0.0% -> 0.0% |
| aware-greedy | 43.0 -> 39.3 | -8.1% [-11.3, -5.3] | -0.1 [-0.4, -0.0] | 58% -> 25% | 0.0% -> 1.5% |
| blind-backfill, fleet 50% | 150.9 -> 82.5 | -42.0% [-49.1, -34.8] | -70.5 [-98.0, -49.4] | 100% -> 86% | 0.0% -> 26.4% |
| aware-greedy, fleet 50% | 136.0 -> 47.3 | -63.3% [-66.7, -59.9] | -72.6 [-104.4, -48.6] | 99% -> 65% | 0.0% -> 18.2% |
<!-- /table:sharing -->

On the half-size fleet, sharing is worth far more than type knowledge: without it the fleet is full and jobs wait over an hour longer. On the trace's own fleet, sharing helps `aware-greedy` by 8% but makes the blind scheduler 5% slower. The reason is in the blind pick: without sharing, the old types fill up first (jobs run longer there, so they hold their GPUs longer), and a pick in proportion to free capacity then lands on a fast GPU more often.

### Sensitivity

The speed table is the assumption the answer rests on. Every relative speed is raised to a power gamma: 0 makes all GPU types equal, 0.5 halves every gap on a log scale (a 4x gap becomes 2x), 2 squares it. The last row gives every job the same speed row (the geometric mean over job types), which keeps the ranking of the types but removes job-specific differences.

<!-- table:sensitivity -->
| gamma | blind-backfill (min) | rank-greedy (min) | aware-greedy (min) | rank-greedy vs blind | aware-greedy vs blind |
|---|---:|---:|---:|---:|---:|
| 0.0 | 93.6 | 93.6 | 93.6 | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] |
| 0.5 | 82.5 | 63.4 | 57.4 | -23.1% [-23.7, -22.5] | -30.5% [-31.0, -29.9] |
| 1.0 | 83.3 | 50.2 | 39.3 | -39.8% [-40.4, -39.3] | -52.9% [-53.6, -52.2] |
| 1.5 | 96.8 | 45.6 | 29.8 | -52.9% [-53.8, -51.8] | -69.2% [-69.9, -68.6] |
| 2.0 | 127.9 | 45.7 | 24.5 | -64.0% [-65.4, -62.4] | -80.9% [-81.5, -80.3] |
| same speed row for every job (gamma 1) | 76.4 | 41.9 | 41.9 | -45.3% [-46.0, -44.5] | -45.3% [-46.0, -44.5] |
<!-- /table:sensitivity -->

### Wrong speed estimates

A real scheduler learns a job's speed per GPU type from short profiling runs or from its model family, and the estimate is off. Here `aware-greedy` sees each job's speeds multiplied by its own random error, log-normal with the standard deviation below (0.5 means a typical error of about a factor of 1.65); the job still runs at its true speed.

<!-- table:estimates -->
| error (sd of log speed) | mean JCT (min) | vs blind-backfill | vs rank-greedy | vs exact speeds |
|---|---:|---:|---:|---:|
| 0 (exact) | 39.3 |  |  |  |
| 0.25 | 40.8 | -51.1% [-51.8, -50.5] | -18.8% [-19.7, -17.8] | +3.7% [+3.5, +4.0] |
| 0.5 | 44.4 | -46.8% [-47.6, -46.1] | -11.7% [-12.5, -10.8] | +12.8% [+12.3, +13.4] |
| 1.0 | 52.6 | -37.1% [-37.9, -36.3] | +4.6% [+3.3, +5.9] | +33.7% [+32.3, +35.0] |
<!-- /table:estimates -->

Per-job knowledge beats the plain ranking as long as the estimates are within about a factor of 1.65. With errors of a factor of 2.7 (sd 1.0), `rank-greedy` is better.

## Limits

- **The speed table is the whole game.** The trace never says how fast a job would have run elsewhere. The speeds come from benchmarks of a handful of models on 2019-2020 software, most jobs get a random job type, MISC is assumed to be K80-class, and the T4 is a chained estimate from a second source. The sensitivity table shows how far the answer moves.
- **Load.** At the trace's load, queueing is negligible and every type-aware policy reduces to "fastest free type". The policies that wait, price or move jobs matter only on the busier fleets, which are a what-if, not the trace.
- **No topology.** Machines, NVLink, network, CPU and memory are ignored; a multi-GPU job may take any free GPUs of one type.
- **No interference.** Jobs that share a GPU run at their own speed, as if alone. On a real shared GPU, tenants slow each other down.
- **Oracle knowledge.** `aware-wait` and the migration rule use each job's exact remaining work; a real scheduler has estimates. `aware-greedy` and `aware-lp` use only the speed table.
- **Checkpoint cost** is one fixed number per run, not a function of model size.
- **The blind baseline** picks a type at random in proportion to its free capacity. A blind scheduler that happens to fill new GPUs first would look better; `rank-greedy` is that scheduler.
- **Jobs still running** when the trace ends (about 6% of all jobs) are left out, so the replayed load is slightly lighter than the real one.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate      # Python 3.12
pip install -r requirements-dev.txt
python scripts/download_trace.py                  # about 150 MB, checked against the published SHA-256 sums
python -m fleetsim.trace                          # -> data/processed/jobs.csv.gz
python -m fleetsim.validate                       # -> results/validation.json
python -m fleetsim.experiments                    # -> results/*.json (520 runs, 11 minutes with 12 workers)
python scripts/make_tables.py                     # rewrites the tables in this README
python -m http.server                             # then open http://localhost:8000/dashboard/
pytest -q && ruff check .
```

The dashboard (`dashboard/index.html`) replays two days of seed 0 under two policies side by side: jobs waiting, and the share of each GPU type in use, with a play button, a time slider and hover values. It reads `results/timeline.json`.

CI (`.github/workflows/ci.yml`) runs ruff, the tests, a check that the tables in this README match `results/`, and a small run of every experiment and of the validation on the committed six-hour sample of the trace (`data/trace/sample_jobs.csv`).

## Data, licences and sources

- **Trace.** Alibaba PAI GPU cluster trace, [`cluster-trace-gpu-v2020`](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020), published by Alibaba under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/LICENSE). The full tables are not in this repository; `scripts/download_trace.py` fetches them. Two small derived files are committed with attribution: the GPU count per type (`data/trace/fleet.csv`) and every job submitted in six hours of trace day 40 (`data/trace/sample_jobs.csv`, 4,046 jobs, job and user names removed). The changes made to the original data are listed in `data/trace/README.md`. As the trace's authors ask, please cite their paper:

  > Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, Y. Ding. *MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters.* USENIX NSDI 2022.

- **Speeds.** D. Narayanan, K. Santhanam, F. Kazhamiaka, A. Phanishayee, M. Zaharia. *Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads.* OSDI 2020. Throughput file from [github.com/stanford-futuredata/gavel](https://github.com/stanford-futuredata/gavel) (MIT licence, reproduced in `data/speeds/README.md`).
- **T4.** R. Xu, F. Han, Q. Ta. *Deep Learning Performance on T4 GPUs with MLPerf Benchmarks.* Dell EMC, March 2019.

Code: MIT licence.
