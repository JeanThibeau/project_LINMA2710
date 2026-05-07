#!/bin/bash
#
#SBATCH --job-name=distributed_part3_test
#SBATCH --output=res_distributed.txt
#SBATCH --partition=debug
#
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

# Load MPI (OpenMPI)
module load openmpi

# Navigate to project directory and run
cd "$(dirname "$0")"
srun make test_distributed
srun mpirun -np 4 ./test_distributed
