#!/bin/bash
#SBATCH --job-name=epix10k_2r_5k_r2
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1                 # 1 H100; all ranks share GPU 0
#SBATCH -n 2                         # 2 ranks x 2500 shots = 5k total
#SBATCH -c 12                        # 12 CPUs/rank = 24 CPUs total
#SBATCH --mem=40G                    # GPU reservation=24G + ~6.4 GB/rank × 2 ranks
#SBATCH --time=02:00:00              # ~10 sec/shot x 2500 shots/rank ~ 83 min
#SBATCH --nodelist=scg020
#SBATCH --output=epix10k_2r_5k_r2%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Test all ePix10k gain-noise branch features:
#   --epixGainThresh: HG/MG/LG zone boundaries (80, 270 photon counts)
#   --epixNoiseSigma: per-zone readout noise RMS
#   --epixSatLG: LG well-capacity clip (11000 ph → max pixel = 11000)
#   --fluxRange: per-shot flux drawn from [5e10, 4e11] photons/pulse
srun --export=ALL resonet-simulate epix10k_2r_5k_r2 \
    --nshot 5000 \
    --geom epix10k \
    --ngpu=1 \
    --randDist --randDistRange 100 300 \
    --randHits \
    --randWave \
    --randCent \
    --varyBgScale \
    --fluxRange 5e10 4e11 \
    --epixGainThresh 80 270 \
    --epixNoiseSigma 0.02 0.023 0.27 \
    --epixSatLG 11000

# After job completes, check pixel stats and memory:
# python3 -c "
# import h5py, numpy as np
# for rank in range(2):
#     with h5py.File(f'epix10k_2r_5k/compressed{rank}.cxi','r') as f:
#         imgs = f['entry_1/data_1/data'][:].astype(np.float32)
#     flat = imgs.ravel()
#     print(f'rank {rank}: shape={imgs.shape} min={flat.min():.0f} max={flat.max():.0f} '
#           f'mean={flat.mean():.2f} p99.9={np.percentile(flat,99.9):.0f} '
#           f'n_saturated={np.sum(flat>=11000)} n_above_sat={np.sum(flat>11000)}')
# "
