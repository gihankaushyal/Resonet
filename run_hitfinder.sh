#!/bin/bash
#SBATCH --job-name=hitfinder_100k
#SBATCH -p general                   # partition
#SBATCH -q grp_cxfel                 #QOS
#SBATCH --gres=gpu:8                 #GPU Resources (all 8 H100s: 2.5 ranks/GPU vs 5)
#SBATCH -n 20                        #Number of tasks (20 ranks x 5k shots = 100k total)
#SBATCH -c 5                         #Number of cpus-per-task
#SBATCH --mem=400G                   #20GB per rank (prev 200G/10GB was OOM at ~160 shots)
#SBATCH --time=08:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=hitfinder_100k_%j.log

# Load your verified environment stack
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh

# GPU and HPC-X library paths
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Run using native srun
srun --export=ALL resonet-simulate hitfinder_100k \
    --nshot 100000 \
    --geom eiger \
    --ngpu=8 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles "hitfinder_100k/compressed*.h5" hitfinder_100k_merged.h5
