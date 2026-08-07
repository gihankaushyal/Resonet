#!/bin/bash
#SBATCH --job-name=cxi_epix10k_5k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 1000 shots = 5k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=50G                    # ~6.4 GB/rank peak; 50 GB total with margin
#SBATCH --time=03:00:00              # ~10 sec/shot x 1000 shots/rank ~ 167 min; 3 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_epix10k_5k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# fluxRange calibrated from 1-shot ePix10k test at default FLUX (4e11):
#   p99.99=10155, max=30634, ~172/2.16M pixels exceed detector limit (11000 ph @ 8 keV)
#   4e11 is the natural upper bound; 5e10 gives ~8x weaker shots (peaks ~100-1000 ph)
srun --export=ALL resonet-simulate cxi_epix10k_5k \
    --nshot 5000 \
    --geom epix10k \
    --ngpu=1 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11

# After job completes, merge per-rank outputs:
# resonet-mergefiles cxi_epix10k_5k cxi_epix10k_5k_merged.cxi --cxi
