# Trace data

Source: Alibaba PAI GPU cluster trace, `cluster-trace-gpu-v2020`, github.com/alibaba/clusterdata, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Alibaba. Described in:

> Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin, Y. Ding. *MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters.* NSDI 2022.

The full tables are not in this repository (about 150 MB compressed for the four used here). `scripts/download_trace.py` fetches them from the publisher's mirror and checks every archive and CSV against the SHA-256 sums in the trace's README; `python -m fleetsim.trace` then builds `data/processed/jobs.csv.gz`.

Two small derived files are committed, as the licence allows with attribution:

- `fleet.csv`: GPUs per type, summed from `pai_machine_spec`.
- `sample_jobs.csv`: every job submitted in six hours of trace day 40 (4,046 jobs), in the processed form below. Used by the tests and the CI smoke run.

Changes made to the original data: tables joined per job, GPU tasks only, jobs that ended as Terminated or Failed only; columns reduced to submit time, duration (first GPU task start to last GPU task end), the GPU type of the largest GPU task, the GPU request as `instances x percent` per task, the pinned GPU type (`gpu_type_spec`) and the workload tag. Job, user and group names are dropped.
