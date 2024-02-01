#!/bin/sh
#
#SBATCH --job-name="ppoisson"
#SBATCH --partition=compute
#SBATCH --time=0:00:60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_0/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_0/%x.err


module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/assignment_1/

mpicc ppoisson.c -o ppoisson.x
srun ppoisson.x