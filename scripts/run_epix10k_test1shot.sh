#!/bin/bash
#SBATCH --job-name=epix10k_test1shot
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --nodelist=scg020
#SBATCH --output=epix10k_test1shot_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

srun --export=ALL resonet-simulate /data/bioxfel/user/gihan/Resonet/view_cxi_testing \
    --nshot 1 \
    --geom epix10k \
    --ngpu=1 \
    --randHits \
    --randDist --randDistRange 100 300

# Rename output to epix10k_test.cxi
mv /data/bioxfel/user/gihan/Resonet/view_cxi_testing/compressed0.cxi \
   /data/bioxfel/user/gihan/Resonet/view_cxi_testing/epix10k_test.cxi

echo "Done. Output: view_cxi_testing/epix10k_test.cxi"
