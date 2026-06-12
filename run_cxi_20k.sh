#!/bin/bash
#SBATCH --job-name=cxi_20k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 4000 shots = 20k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=400G                   # 80 GB/rank ceiling; peak observed ~6.4 GB/rank (~12x margin)
#SBATCH --time=12:00:00              # ~10 sec/shot x 4000 shots/rank ~ 11 hr; 12 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_20k_%j.log

source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate cxi_20k \
    --nshot 20000 \
    --outfmt cxi \
    --geomfile /data/bioxfel/user/gihan/Resonet/geoms/Eigar.geom \
    --detector-name "EIGER 4M" \
    --ngpu=1 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles cxi_20k cxi_20k_merged.cxi --cxi
