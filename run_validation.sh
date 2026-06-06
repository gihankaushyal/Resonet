#!/bin/bash
#SBATCH --job-name=hitfinder_val
#SBATCH -p general                   # partition
#SBATCH -q grp_cxfel                 #QOS
#SBATCH --gres=gpu:4                 #GPU Resources (4 H100s)
#SBATCH -n 10                        #Number of tasks (10 ranks x 200 shots = 2000 total)
#SBATCH -c 5                         #Number of cpus-per-task
#SBATCH --mem=300G                   #~30GB per rank (comfortable headroom for cache fill)
#SBATCH --time=01:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=hitfinder_val_%j.log

# Load your verified environment stack
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh

# GPU and HPC-X library paths
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Validation run: 2k shots, 10 ranks x 200 shots
# Goal: confirm RSS goes flat after ~200 shots (cache fill) with all fixes active
# If flat -> safe to scale to 90k. If still growing -> Phase 3 investigation needed.
srun --export=ALL resonet-simulate hitfinder_val \
    --nshot 2000 \
    --geom eiger \
    --ngpu=4 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale
