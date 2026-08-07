#!/bin/bash
#SBATCH --job-name=epix10k_noise_test
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --nodelist=scg020
#SBATCH --output=epix10k_noise_test_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate /data/bioxfel/user/gihan/Resonet/view_cxi_testing \
    --nshot 10 \
    --geom epix10k \
    --ngpu=1 \
    --randHits \
    --randDist --randDistRange 80 400 \
    --epixGainThresh 80 270 \
    --epixNoiseSigma 0.02 0.023 0.27 \
    --epixSatLG 11000

mv /data/bioxfel/user/gihan/Resonet/view_cxi_testing/compressed0.cxi \
   /data/bioxfel/user/gihan/Resonet/view_cxi_testing/epix10k_test_noise.cxi

echo "Done. Output: view_cxi_testing/epix10k_test_noise.cxi"
