# Speed data

The trace records where each job ran and for how long, not how fast it would have run on another GPU type. These two files supply that, and they are assumptions, not measurements of the trace's jobs.

## gavel_throughputs.csv

Measured training throughput of 26 job types (7 model families at several batch sizes) on K80, P100 and V100 GPUs, each job alone on one GPU. Taken unchanged (rounded to six decimals) from the `simulation_throughputs.json` file published with Gavel, at commit `c2bc4b8f218e369e4e17ab86e216753997545d5b` of github.com/stanford-futuredata/gavel; `scripts/derive_speeds.py` downloads that file, checks its SHA-256 and rebuilds this CSV.

> D. Narayanan, K. Santhanam, F. Kazhamiaka, A. Phanishayee, M. Zaharia. *Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads.* OSDI 2020.

Licence of the source file:

```
MIT License

Copyright (c) 2020 Stanford Future Data Systems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## t4_ratios.csv

Gavel has no T4. The T4 speed of a job type is its V100 speed divided by how many times faster a V100-PCIe trained than a T4 in MLPerf Training v0.5 runs on one server:

> R. Xu, F. Han, Q. Ta. *Deep Learning Performance on T4 GPUs with MLPerf Benchmarks.* Dell EMC, March 2019 (Dell knowledge base article 000132094).

ResNet-50 3.6x, SSD 3.3 to 3.4x, Mask R-CNN 2.2 to 2.7x, GNMT 2.5 to 3.6x, Transformer 2.6 to 2.8x. The mapping to Gavel's families and the value used for families without a matching benchmark are in the CSV.

## How the trace's GPU types map to these

| trace type | treated as | why |
|---|---|---|
| MISC | K80 | the trace's README describes MISC as older generations such as K40m, K80 and M60 |
| T4 | V100 / ratio above | no T4 in Gavel |
| P100 | P100 | measured |
| V100 | V100 | measured |
| V100M32 | V100 | same chip, 32 GB instead of 16 GB of memory |
