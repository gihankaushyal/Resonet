#!/bin/bash
#SBATCH --job-name=hitfinder_100k
#SBATCH -p general                   # partition
#SBATCH -q grp_cxfel                 #QOS
#SBATCH --gres=gpu:6                 #GPU Resources (6 H100s: ~2.3 ranks/GPU)
#SBATCH -n 14                        #Number of tasks (14 ranks x 5c = 70 CPUs total)
#SBATCH -c 5                         #Number of cpus-per-task
#SBATCH --mem=600G                   #~43GB per rank
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
    --ngpu=6 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles "hitfinder_100k/compressed*.h5" hitfinder_100k_merged.h5
