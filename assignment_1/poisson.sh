#!/bin/sh
#
#SBATCH --job-name="poisson"
#SBATCH --partition=compute
#SBATCH --time=0:00:60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_1/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_1/%x.err


module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/assignment_1/

mpicc poisson.c -o poisson.x
srun poisson.x