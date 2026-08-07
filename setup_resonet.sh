#!/bin/bash
# Clean out any stray modules from your login node
module purge

# Load the verified compatible toolchain
module load gcc-12.1.0-gcc-11.2.0
module load cuda-12.8.1-gcc-12.1.0
module load openmpi/5.0.8

# HPC-X libraries needed by OpenMPI 5.0.8 (libhcoll, libocoms)
export LD_LIBRARY_PATH=/packages/apps/hpcx/2.25.1/inbox/hcoll/lib:/packages/apps/hpcx/2.25.1/inbox/ucx/lib:${LD_LIBRARY_PATH}

# Activate your custom conda environment
source /data/bioxfel/user/gihan/Resonet/simforge/etc/profile.d/conda.sh
conda activate simtbx_mpi
