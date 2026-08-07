#!/bin/bash
#SBATCH --job-name=jungfrau_20k
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0 (multi-panel CUDA workaround)
#SBATCH -n 10                         # 10 ranks x 2000 shots = 20k total
#SBATCH -c 8                          # 8 CPUs/rank = 80 CPUs total
#SBATCH --mem=600G                   # ~40 GB/rank ceiling
#SBATCH --time=12:00:00              # ~10 sec/shot x 4000 shots/rank ~ 11 hr; 12 hr wall
#SBATCH --nodelist=scg020
#SBATCH --output=jungfrau_20k_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Jungfrau 3-gain noise model features:
#   --jungfrauGainThresh: G0/G1/G2 zone boundaries (34, 342 photon counts)
#   --jungfrauNoiseSigma: per-zone readout noise RMS (G0=0.2, G1=1.5, G2=15.0)
#   --jungfrauSatG2:      G2 well-capacity clip (3400 photon counts)
#   --fluxRange:          per-shot flux drawn from [5e10, 4e11] photons/pulse
srun --export=ALL resonet-simulate production/jungfrau_20k \
    --nshot 20000 \
    --outfmt cxi \
    --geom jungfrau \
    --ngpu=1 \
    --randDist --randDistChoice 80 400 \
    --randHits \
    --randWave --randWaveRange 9000 12000 \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11 \
    --jungfrauGainThresh 34 342 \
    --jungfrauNoiseSigma 0.2 1.5 15.0 \
    --jungfrauSatG2 3400

# After job completes, merge per-rank CXI outputs:
# resonet-mergefiles production/jungfrau_20k production/jungfrau_20k_merged.cxi --cxi
