#!/bin/sh
#
#SBATCH --job-name="MM-product"
#SBATCH --partition=compute
#SBATCH --time=0:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=/home/%u/HPC/hpc-labs/out/assignment_0b/%x.out
#SBATCH --error=/home/%u/HPC/hpc-labs/out/assignment_0b/%x.err

# TODO modify the number of process P=1, 2, 8, 24, 48 and 64

module load 2023r1
module load openmpi

cd ~/HPC/hpc-labs/intro/
# echo > MM-product.txt

mpicc MM-product.c ~/HPC/hpc-labs/mpi_functions/getNodeCount.c ~/HPC/hpc-labs/data_extraction/saveArray.c -o MM-product.x -I ~/HPC/hpc-labs/mpi_functions -I ~/HPC/hpc-labs/data_extraction
srun MM-product.x #>> MM-product.txt