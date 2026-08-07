#!/bin/bash
#SBATCH --job-name=gpu_permutation
#SBATCH --partition=general
#SBATCH --qos=grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=gpu_permutation_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

python /data/bioxfel/user/gihan/Resonet/test_gpu_permutation.py
