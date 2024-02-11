#!/bin/sh
#
#SBATCH --job-name="ppoisson"
#SBATCH --partition=compute
#SBATCH --time=0:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_1/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_1/%x.err


module load 2023r1
module load openmpi/4.1.4

cd ~/HPC/hpc-labs/assignment_1/

mpicc ppoisson.c -o ppoisson.x
srun ppoisson.x 4 1 -omegas 1.90 1.99 0.01 -grids 100 200 100 -output false -latency false -benchmark false