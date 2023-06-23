#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1         # Max is 1
#SBATCH --ntasks=4        # Max is 16 (1/8 of 2x 64 AMD EPYC CPUs)
#SBATCH --cpus-per-task=1 # Max is 2 (Clustered Multithreading is on)
#SBATCH --gres=gpu:1      # Max is 1 (1 single A100)
#SBATCH --time=00:20:00   # Max is 4 hours

module purge
module load gcc/11.2.0

export OMP_NUM_THREADS=4
export TMPDIR=$(pwd)
mkdir -p out
nsys profile --trace=cuda,nvtx,osrt ../../../../src/gravidy-cpu.orig -i ../../../input/gravidy-default-input/01-nbody-p128_m1.in -o out/output -t 0.1 -z 0.1

