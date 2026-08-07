#!/bin/bash
#SBATCH --job-name=epix_pixcheck
#SBATCH -p general
#SBATCH -q grp_cxfel
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --nodelist=scg020
#SBATCH --output=epix_pixcheck_%j.log

source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

cp resonet/resonet/sims/main.py simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/main.py
cp resonet/resonet/sims/make_sims.py simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/make_sims.py
cp resonet/resonet/sims/simulator.py simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/simulator.py

srun --export=ALL resonet-simulate epix_pixcheck \
    --nshot 100 \
    --geom epix10k \
    --ngpu=1 \
    --randHits \
    --fluxRange 5e10 4e11 \
    --epixGainThresh 80 270 \
    --epixNoiseSigma 0.02 0.023 0.27 \
    --epixSatLG 11000

python3 -c "
import h5py, numpy as np
fname = 'epix_pixcheck/compressed0.cxi'
with h5py.File(fname, 'r') as f:
    imgs = f['entry_1/data_1/data'][:].astype(np.float32)
    flux = f['LCLS/flux'][:] if 'LCLS/flux' in f else None
    hits = f['LCLS/hit'][:] if 'LCLS/hit' in f else None

flat = imgs.ravel()
print(f'Shape: {imgs.shape}')
print(f'min={flat.min():.2f}  max={flat.max():.2f}  mean={flat.mean():.4f}')
print(f'n > 11000 (above sat): {int(np.sum(flat > 11000))}')
print(f'n == 11000 (at sat):   {int(np.sum(flat == 11000))}')
print(f'n < 0 (negative):      {int(np.sum(flat < 0))}')
print(f'p99.9={np.percentile(flat, 99.9):.1f}  p99.99={np.percentile(flat, 99.99):.1f}')
if flux is not None:
    print(f'flux: min={flux.min():.2e}  max={flux.max():.2e}  mean={flux.mean():.2e}')
if hits is not None:
    print(f'hits: {int(hits.sum())}/{len(hits)} shots')
"
