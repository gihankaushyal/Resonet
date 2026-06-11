#!/bin/bash
#SBATCH --job-name=cxi_char
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:4                 # 4 H100s; ranks 0-9 cycle via rank % 4
#SBATCH -n 10                        # 10 ranks x 1000 shots = 10k total
#SBATCH -c 6                         # 6 CPUs/rank = 60 CPUs total
#SBATCH --mem=600G                   # 60 GB/rank headroom
#SBATCH --time=04:00:00              # ~10 sec/shot x 1000 shots ~ 2.8 hr; 4 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_char_%j.log

# Goal: measure post-cache-fill steady-state RSS on the CXI multi-panel path.
# The miller-array cache fills at ~shot 750/rank (117 PDBs x coupon-collector).
# At 1000 shots/rank we see ~250 shots of steady-state — enough to confirm
# whether RSS flattens (leaks fixed) or grows linearly (remaining leak).
# RSS is logged every 10 shots for shots 0-299, then every 50 shots after that.

source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate cxi_char \
    --nshot 10000 \
    --outfmt cxi \
    --geomfile /data/bioxfel/user/gihan/Resonet/geoms/Eigar.geom \
    --detector-name "EIGER 4M" \
    --ngpu=4 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale

# After job completes, merge per-rank outputs:
# resonet-mergefiles cxi_char cxi_char_merged.cxi --cxi
