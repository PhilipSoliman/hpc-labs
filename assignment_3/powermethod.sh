#!/bin/sh

#SBATCH --job-name="pwrmthdCUDA"
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_3/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_3/%x.err


module load 2023r1 nvhpc
file=’powermethod’
nvcc –o $file $file.cu
srun ./$file