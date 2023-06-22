# GraviDy -  Gravitational Dynamics

Check the original README in the master branch, or the original repo (https://github.com/cmaureir/gravidy).

## This branch, NACT_LO_LIMIT
We set a limit on the `nact` variable, which tells the program how  many particles should be updated this iteration. If it is lower than this, we make the CPU do the computation and if it is higher, the GPU does it. We find that the CPU is categorically slower, even for very small `nact` and so no `NACT_LO_LIMIT` is reasonable. All the work must be done on the GPU.

## About

**GraviDy** is a new GPU, direct-summation N-body integrator written from scratch and based on the
Hermite scheme. The most important features of the code are:

 * Written in C/C++,
 * Using parallel computing techniques and libraries like OpenMP, MPI and CUDA,
 * full double-precision resolution,
 * its high modularity, which allows users to readily introduce new physics into it,
 * the exploitation of all high-performance computing resources available,
 * its maintenance and improvement cycle,
 * the fact that the code is publicly released under a BSD license and will be maintained via planned, public, regular updates.

Publication reference

Maureira-Fredes, C and Amaro-Seoane, P. ["GraviDy, a GPU modular, parallel direct-summation N−body integrator: Dynamics with softening"](https://arxiv.org/abs/1702.00440), MNRAS, vol. 473, pp. 3113-3127, Jan . 2018.

## How to compile?

First, clone the repositotry

```sh
git clone https://github.com/chongchonghe/gravidy
cd gravidy/src
```

Then, depending on the OS you are on,

- On UMD Astronomy machines, the default g++ is version 8, which is outdated. Version 11 is available in PATH as g++11

```sh
make CXX="g++11"
```

The original version of Gravidy use Boost library to do command parsing. I rewrote that part using pure C++ and removed this dependency, so the code does not rely on any library. If you want to use the original version with Boost, pass `use_boost=1` as an argument to make.

- On macOS, use your latest version of g++ isntalled via homebrew. The default g++, or Apple clang, does not support omp. The Boost library failed on macOS, so Makefile automatically set use_boost to 0 on Mac.

```sh
make CXX="g++-13"
```

- On Zaratan

CPU (with OpenMP):

```sh
module load gcc/11.3.0
# module load boost/1.79.0
make
```

To use Boost command parsing, uncomment the line that loads boost, and pass `use_boost=1` as an argument to make.

MPI: Be sure to run `make distclean` before recompiling if you changed your module version or modified Makefile.

```sh
module load gcc/11.3.0
module load openmpi/4.1.4
# module load boost/1.79.0
make mpi
```

CUDA: failed. I couldn't make it work. Let me know if you can.

```sh
module load gcc/8.4.0
module load cuda/10.2.89
# module load boost/1.72.0
make gpu
```

- On Nvidia Cluster (for Hackathon)

```bash
module load gcc/11.2.0
make cpu
module load openmpi/gcc/64/4.1.2
make mpi
module unload openmpi/gcc/64/4.1.2
module load cuda-11.7.1
module load cuda11.7/toolkit/11.7.1
# module load nvhpc-23.3-nompi
make gpu
```

To run the test
```bash
module load conda
conda create -n gravidy python=3.9
conda activate gravidy
conda init bash
exec bash
module load conda
conda activate gravidy
conda install numpy pandas matplotlib
bash test2.bash
```

## Flowchart

You can render UML diagrams using [Mermaid](https://mermaidjs.github.io/). For example, this will produce a sequence diagram:

Here is a flow chart for 32k particles:

TODO:

- There are gaps in CUDA HW which are not included in the following chart. They could related to `cudaMemcpyAsync` but I don’t know exactly what they are about.

```mermaid
graph TB
A[Pure CPU, 80s] --> B[HtoD, 0.5ms] --> C((k_prediction, <0.1ms)) --> D[DtoH, 0.2ms] --> E[HtoD, <0.1ms] --> F((k_update, 0.4ms)) --> G[DtoH, <0.1ms] --> H[HtoD, 0.3ms] -- 11k times --> C
```


## Licence

```
Copyright 2014 Cristián Maureira-Fredes

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
