# Running on the CECI Cluster

This guide explains how to submit and run each part of the matrix project on the CECI HPC cluster using Slurm.

## Prerequisites

- You have SSH access to a CECI login node
- The project code is in your home directory or a shared filesystem
- All necessary modules are available on the cluster

## General Workflow

1. **Connect to login node**: `ssh username@login.ceci.example.com`
2. **Navigate to project**: `cd ~/LINMA2710/project`
3. **Submit job**: `sbatch submit_partX_*.sh`
4. **Monitor job**: `squeue --me`
5. **View results**: `cat res_*.txt` (after job completes)

## Part 1 & 2: Matrix Operations with OpenMP

**Submission script**: `submit_part1_matrix.sh`

### What it does:
- Requests 1 task with 8 CPUs on the **debug** partition
- Sets `OMP_NUM_THREADS=8` to parallelize OpenMP loops
- Compiles and runs matrix tests with OpenMP enabled

### Submit:
```bash
sbatch submit_part1_matrix.sh
```

### View output:
```bash
cat res_matrix.txt
```

### Expected output:
All matrix tests should pass:
```
testConstructorsAndAccessors passed.
testAdditionSubtraction passed.
...
All matrix tests passed.
```

## Part 3: Distributed Matrix with MPI

**Submission script**: `submit_part3_distributed.sh`

### What it does:
- Requests 4 MPI tasks on the **debug** partition
- Loads OpenMPI module
- Compiles and runs distributed matrix tests across 4 processes

### Submit:
```bash
sbatch submit_part3_distributed.sh
```

### View output:
```bash
cat res_distributed.txt
```

### Expected output:
Tests run on all 4 processes, with rank 0 printing final message:
```
Starting DistributedMatrix tests...
testConstructorAndBasics passed.
...
All distributed matrix tests passed.
```

### Troubleshooting:
- If job stays PENDING with reason "Resources": debug partition is busy. Try `--partition=batch` but increase `--time=1:00:00`.
- If MPI module not found: check available modules with `module avail` and adjust `module load` accordingly.

## Part 4: GPU Matrix with OpenCL

**Submission script**: `submit_part4_opencl.sh`

### What it does:
- Requests 1 GPU on the **batch** partition
- Loads CUDA module (provides OpenCL support)
- Compiles and runs OpenCL matrix tests on GPU

### Submit:
```bash
sbatch submit_part4_opencl.sh
```

### View output:
```bash
cat res_opencl.txt
```

### Expected output:
OpenCL kernels compile, then all tests pass:
```
Platform: ...
Device: ...
setupOpenCL passed.
testFill passed.
...
All OpenCL matrix tests passed.
```

### Troubleshooting:
- If GPU not found: verify GPU availability with `sinfo -N -l` and check node features.
- If OpenCL headers missing: try `module load opencl-icd` or similar (check `module avail` for exact name).

## Quick Reference: Check cluster status

Before submitting, check available resources:

```bash
# View partition info
sinfo

# View node details (CPU/memory/GPU)
sinfo -N -l

# View your submitted jobs
squeue --me

# Cancel a job (use JOBID from squeue output)
scancel JOBID
```

## Advanced: Run interactively for debugging

If you want to debug interactively before submitting batch jobs:

```bash
# Start an interactive session on a compute node (8 CPUs, 1 hour)
salloc -N1 --ntasks=1 --cpus-per-task=8 --time=1:00:00 --partition=debug

# Once allocated, you get a shell prompt. Load modules and run commands:
module load gcc
cd ~/LINMA2710/project
make OPENMP_FLAGS="" test_matrix
./test_matrix

# Exit interactive session
exit
```

## Customizing scripts

Adjust the scripts for your needs:

- **More CPUs for Part 1**: Change `--cpus-per-task=8` to `--cpus-per-task=16` (or higher)
- **More MPI tasks for Part 3**: Change `--ntasks=4` to `--ntasks=8` or `--ntasks=16`
- **Longer runtime**: Change `--time=10:00` (10 min) to `--time=1:00:00` (1 hour)
- **Different partition**: Replace `--partition=debug` with `--partition=batch` for longer jobs

## Notes

- Output files (`res_*.txt`) are created in the directory where you ran `sbatch`.
- Jobs on the **debug** partition have a 30-minute limit (see `sinfo`).
- For long-running jobs, use **batch** partition instead (may have longer queue time).
- GPU availability varies by cluster; check with administrators or `sinfo -N -l` for GPU-equipped nodes.
