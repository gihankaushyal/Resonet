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
source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh

# GPU and HPC-X library paths
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Validation run: 2k shots, 10 ranks x 200 shots
# Goal: measure per-shot RSS trend in the cache-fill phase. With 117 PDBs selected
# randomly, full cache fill requires ~625-750 shots/rank (coupon collector). At 200
# shots/rank this run does NOT reach steady state — it shows the rising portion of
# the RSS curve. To confirm RSS flattens, run with >=1000 shots/rank.
srun --export=ALL resonet-simulate hitfinder_val \
    --nshot 2000 \
    --geom eiger \
    --ngpu=4 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale
