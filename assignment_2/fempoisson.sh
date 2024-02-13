#!/bin/sh
#
#SBATCH --job-name="fempoisson"
#SBATCH --partition=compute
#SBATCH --time=0:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_2/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_2/%x.err


module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/assignment_2/

make
srun GridDist.x 5 5 1000 1000
srun MPI_Fempois.x
