#!/bin/bash
#SBATCH --job-name=epix10k_20k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 5                         # 5 ranks x 4000 shots = 20k total
#SBATCH -c 12                        # 12 CPUs/rank = 60 CPUs total
#SBATCH --mem=200G                   # ~40 GB/rank ceiling based on 5k run
#SBATCH --time=12:00:00              # ~10 sec/shot x 4000 shots/rank ~ 11 hr; 12 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=epix10k_20k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# ePix10k 3-gain noise model features:
#   --epixGainThresh: HG/MG/LG zone boundaries (80, 270 photon counts)
#   --epixNoiseSigma: per-zone readout noise RMS (HG=0.02, MG=0.023, LG=0.27)
#   --epixSatLG:      LG well-capacity clip (11000 ph → max pixel = 11000)
#   --fluxRange:      per-shot flux drawn from [5e10, 4e11] photons/pulse
srun --export=ALL resonet-simulate production/epix10k_20k \
    --nshot 20000 \
    --outfmt cxi \
    --geom epix10k \
    --ngpu=1 \
    --randDist --randDistChoice 80 400 \
    --randHits \
    --randWave --randWaveRange 9000 12000 \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11 \
    --epixGainThresh 80 270 \
    --epixNoiseSigma 0.02 0.023 0.27 \
    --epixSatLG 11000

# After job completes, merge per-rank CXI outputs:
# resonet-mergefiles production/epix10k_20k production/epix10k_20k_merged.cxi --cxi
