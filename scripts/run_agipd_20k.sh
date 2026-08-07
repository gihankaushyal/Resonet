#!/bin/bash
#SBATCH --job-name=agipd_20k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 4000 shots = 20k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=200G                   # ~40 GB/rank ceiling
#SBATCH --time=12:00:00              # 12 hr wall; AGIPD 3D frames may be slower per shot
#SBATCH --nodelist=scg020
#SBATCH --output=agipd_20k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# AGIPD 1M 3-gain noise model features:
#   Output shape: (N, 16, 512, 128) — 16 modules stacked in CXI
#   --agipd-gain-thresh: HG/MG/LG zone boundaries (65, 2000 photon counts)
#   --agipd-noise-sigma: per-zone readout noise in ADU (HG=7.0, MG=3.0, LG=1.5)
#   --fluxRange:         per-shot flux drawn from [5e10, 4e11] photons/pulse
srun --export=ALL resonet-simulate production/agipd_20k \
    --nshot 20000 \
    --outfmt cxi \
    --geom agipd \
    --ngpu=1 \
    --randDist --randDistChoice 80 400 \
    --randHits \
    --randWave --randWaveRange 9000 12000 \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11 \
    --agipd-gain-thresh 65 2000 \
    --agipd-noise-sigma 7.0 3.0 1.5

# After job completes, merge per-rank CXI outputs:
# resonet-mergefiles production/agipd_20k production/agipd_20k_merged.cxi --cxi
