#!/bin/bash
#SBATCH --job-name=cxi_1k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 200 shots = 1k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=50G                    # ~6.4 GB/rank peak observed; 50 GB total with margin
#SBATCH --time=01:00:00              # ~10 sec/shot x 200 shots/rank ~ 33 min; 1 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_1k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate cxi_1k \
    --nshot 1000 \
    --geom eiger4m \
    --ngpu=1 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles cxi_1k cxi_1k_merged.cxi --cxi
