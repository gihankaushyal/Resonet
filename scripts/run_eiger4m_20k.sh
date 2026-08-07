#!/bin/bash
#SBATCH --job-name=eiger4m_20k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 4000 shots = 20k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=200G                   # ~40 GB/rank ceiling
#SBATCH --time=12:00:00              # ~10 sec/shot x 4000 shots/rank ~ 11 hr; 12 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=eiger4m_20k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate production/eiger4m_20k \
    --nshot 20000 \
    --outfmt cxi \
    --geom eiger4m \
    --ngpu=1 \
    --randDist --randDistChoice 80 400 \
    --randHits \
    --randWave --randWaveRange 9000 12000 \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11

# After job completes, merge per-rank CXI outputs:
# resonet-mergefiles production/eiger4m_20k production/eiger4m_20k_merged.cxi --cxi
