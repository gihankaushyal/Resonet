#!/bin/bash
#SBATCH --job-name=jungfrau_noise_test
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --nodelist=scg020
#SBATCH --output=jungfrau_noise_test_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate /data/bioxfel/user/gihan/Resonet/view_cxi_testing \
    --nshot 10 \
    --geom jungfrau \
    --ngpu=1 \
    --randHits \
    --randDist --randDistRange 80 400 \
    --jungfrauGainThresh 34 342 \
    --jungfrauNoiseSigma 0.2 1.5 15.0 \
    --jungfrauSatG2 3400

mv /data/bioxfel/user/gihan/Resonet/view_cxi_testing/compressed0.cxi \
   /data/bioxfel/user/gihan/Resonet/view_cxi_testing/jungfrau_noise_test.cxi

echo "Done. Output: view_cxi_testing/jungfrau_noise_test.cxi"
