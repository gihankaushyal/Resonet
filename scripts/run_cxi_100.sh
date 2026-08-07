#!/bin/bash
#SBATCH --job-name=cxi_100
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH -n 2                         # 2 ranks x 5c = 10 CPUs total
#SBATCH -c 5
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_100_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate cxi_100 \
    --nshot 100 \
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
# resonet-mergefiles cxi_100 cxi_100_merged.cxi --cxi
