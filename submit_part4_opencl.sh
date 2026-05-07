#!/bin/bash
#
#SBATCH --job-name=opencl_part4_test
#SBATCH --output=res_opencl.txt
#SBATCH --partition=batch
#
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=10:00

# Load CUDA/GPU module (or OpenCL, depending on your cluster's setup)
module load CUDA

# Navigate to project directory and compile/run
cd "$(dirname "$0")"
srun make test_opencl
srun ./test_opencl
