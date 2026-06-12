#!/bin/bash
#SBATCH --job-name=cxi_char_5k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (matches cxi_100 working config)
#SBATCH -n 5                         # 5 ranks x 1000 shots = 5k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=600G                   # 120 GB/rank headroom
#SBATCH --time=04:00:00              # ~10 sec/shot x 1000 shots/rank ~ 2.8 hr; 4 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=cxi_char_5k_%j.log

# Goal: measure post-cache-fill steady-state RSS on the CXI multi-panel path.
# Observed cache fill: ~950 shots/rank (117 PDBs x coupon-collector; tail extends
# past the ~750-shot theoretical estimate). At 1000 shots/rank only ~50 shots of
# post-fill steady-state are available — marginal but sufficient to confirm whether
# the RSS curve has flattened by shot 1000.
# RSS is logged every 10 shots for shots 0-299, then every 50 shots after that.

source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate cxi_char_5k \
    --nshot 5000 \
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
# resonet-mergefiles cxi_char_5k cxi_char_5k_merged.cxi --cxi
