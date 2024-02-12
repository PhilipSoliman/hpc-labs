#!/bin/sh
#
#SBATCH --job-name="ppoisson2"
#SBATCH --partition=compute
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_1/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_1/%x.err


module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/assignment_1/

mpicc ppoisson2.c -o ppoisson2.x
srun ppoisson2.x 2 2 -omegas 1.90 1.99 0.01 -grid 1000 -output false -latency false -benchmark true -errors false