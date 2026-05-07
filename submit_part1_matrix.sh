#!/bin/bash
#
#SBATCH --job-name=matrix_part1_test
#SBATCH --output=res_matrix.txt
#SBATCH --partition=debug
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

# Load GCC compiler with OpenMP support
module load gcc

# Set number of OpenMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory and compile/run
cd "$(dirname "$0")"
srun make run_matrix
