#!/bin/sh
#
#SBATCH --job-name="ppoisson2"
#SBATCH --partition=compute
#SBATCH --time=0:00:60
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
srun ppoisson2.x 1 4 -omega 1.95 -grid 800 -errors true -output false