#!/bin/sh
#
#SBATCH --job-name="MM-product"
#SBATCH --partition=compute
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-IN4049TU
#SBATCH --output=HPC/out/%j_%x.out
#SBATCH --error=HPC/out/%j_%x.err

# TODO modify the number of process P=1, 2, 8, 24, 48 and 64

module load 2022r2
module load openmpi

cd HPC/intro/

mpicc MM-product.c -o MM-product.x
srun MM-product.x 