#!/bin/sh
#
#SBATCH --job-name="helloworld"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%x.out
#SBATCH --error=HPC/out/%x.err

module load 2023r1
module load openmpi/4.1.4

cd ~/HPC/hpc-labs/intro/

mpicc helloworld.c -o helloworld.x
srun helloworld.x > helloworld.log