#!/bin/bash
#SBATCH --job-name=hitfinder_10k
#SBATCH -p general                   # partition
#SBATCH -q grp_cxfel                 #QOS
#SBATCH --gres=gpu:4                 #GPU Resources (4 H100s)
#SBATCH -n 4                         #Number of tasks (4 ranks x 5c = 20 CPUs total)
#SBATCH -c 5                         #Number of cpus-per-task
#SBATCH --mem=200G                   #~50GB per rank
#SBATCH --time=02:00:00
#SBATCH --nodelist=scg020
#SBATCH --output=hitfinder_10k_%j.log

# Load your verified environment stack
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh

# GPU and HPC-X library paths
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Run using native srun
srun --export=ALL resonet-simulate hitfinder_10k \
    --nshot 10000 \
    --geom eiger \
    --ngpu=4 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles "hitfinder_10k/compressed*.h5" hitfinder_10k_merged.h5
