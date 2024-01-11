#!/bin/sh
#
#SBATCH --job-name="helloworld"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
# max number of processes per node = ntask * cpus-per-node
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/%x.err

module load 2023r1
module load openmpi/4.1.4

cd ~/HPC/hpc-labs/intro/

mpicc helloworld.c -o helloworld.x
srun helloworld.x 