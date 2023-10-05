#!/bin/sh
#
#SBATCH --job-name="pingpong"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU

# mkdir -p ~/HPC/out/pingPong.out
# SBATCH --output=pingPong.out

# mkdir -p ~/HPC/out/pingPong.err
# SBATCH --error=pingPong.err

module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/intro/

mpicc pingPong.c ~/HPC/hpc-labs/saveArray.c -o pingPong.x -I "~/HPC/hpc-labs/data_extraction/"
srun pingPong.x