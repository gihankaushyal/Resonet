#!/bin/bash
#SBATCH --job-name=hitfinder_resume_r10-19
#SBATCH -p general                   # partition
#SBATCH -q grp_cxfel                 # QOS
#SBATCH --gres=gpu:4                 # GPU Resources
#SBATCH -n 10                        # 10 tasks = ranks 10-19 of the original 20-rank job
#SBATCH -c 4                         # Number of cpus-per-task
#SBATCH --mem=0                      # Request all available node memory
#SBATCH --time=08:00:00
#SBATCH --output=hitfinder_resume_%j.log

# Load your verified environment stack
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh

# GPU and HPC-X library paths
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Re-run ranks 10-19 of the original 20-rank job.
# --rankOffset 10  : effective rank = COMM.rank + 10, so files write to compressed10-19.h5
# --totalRanks 20  : shot split uses 20-way partition (5k shots per rank, matching originals)
# Output goes into the same hitfinder_100k/ dir alongside completed compressed0-9.h5
srun --export=ALL resonet-simulate hitfinder_100k \
    --nshot 100000 \
    --geom eiger \
    --ngpu=4 \
    --rankOffset 10 \
    --totalRanks 20 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After this job completes, all 20 files (compressed0-19.h5) will be present.
# Merge into a single master file:
# resonet-mergefiles "hitfinder_100k/compressed*.h5" hitfinder_100k_merged.h5
