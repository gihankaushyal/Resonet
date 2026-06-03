# Resonet Codebase Mind Map

**Read this file first** before opening any source file. It contains the complete module map,
data flow, model registry, HDF5 schema, and key design invariants for the Resonet project.

---

## 1. Project Purpose

Resonet is an HPC ML pipeline for X-ray crystallography at free-electron lasers (XFELs).

**Goal:** Train deep neural networks on synthetic diffraction images to predict crystal properties
from real experimental images — without needing labelled real data.

**Predicted quantities:** resolution, crystal orientation (rotation matrix / quaternion),
multi-lattice flag, ice-ring content, Bragg spot count.

**Core stack:** simtbx/nanoBragg (physics simulation) → HDF5 → PyTorch ResNets → inference.

---

## 2. Repository Layout

```
/data/bioxfel/user/gihan/Resonet/
├── resonet/                          # build-files repo (pyproject.toml, setup.cfg)
│   └── resonet/                      # GIT SUBMODULE — actual Python source package
│       ├── net.py                    # resonet-train; training loop + full CLI
│       ├── arches.py                 # ALL model classes (RESNetAny, LeNet, OriQuatModel, CounterRn)
│       ├── params.py                 # ARCHES + LOSSES registries (factory pattern)
│       ├── loaders.py                # H5SimDataDset — PyTorch Dataset over HDF5 master files
│       ├── ori_net.py                # standalone orientation training script
│       ├── td_net.py                 # MPI/DDP training entry (calls net.do_training)
│       ├── restart_net.py            # resume training from .chkpt checkpoint
│       ├── laue.py                   # SUNet (U-Net) for Laue/polychromatic diffraction
│       ├── sims/                     # ── SIMULATION PIPELINE ──
│       │   ├── runme.py              # resonet-simulate (MPI via libtbx.mpi4py)
│       │   ├── runme_joblib.py       # resonet-simulate-joblib (CPU, joblib.Parallel)
│       │   ├── main.py               # shared args() parser + run() loop
│       │   ├── simulator.py          # Simulator class — wraps nanoBragg C++ engine
│       │   ├── make_crystal.py       # load_crystal(), load_beam(), get_Nabc()
│       │   ├── make_sims.py          # get_background(), get_theta_map(), choose_mos/res/stol()
│       │   ├── paths_and_const.py    # global constants, PDB paths, scattering profiles
│       │   └── *.cbf                 # detector geometry files (eiger, pilatus, rayonix)
│       ├── scripts/                  # ── CLI UTILITIES ──
│       │   ├── merge_h5s.py          # resonet-mergefiles → HDF5 virtual-dataset master
│       │   ├── compress.py / decompress.py
│       │   ├── view_sims.py          # resonet-viewsims
│       │   ├── get_simdata.py        # resonet-getsimdata (download PDB + scattering data)
│       │   ├── plot_train.py         # resonet-plotloss
│       │   └── mfx.py               # resonet-mfx (MFX experiment utilities)
│       ├── utils/                    # ── INFERENCE & UTILITIES ──
│       │   ├── eval_model.py         # load_model(), raw_img_to_tens_*() preprocessing
│       │   ├── predict.py            # ImagePredict — multi-model inference wrapper
│       │   ├── predict_dxtbx.py      # ImagePredictDxtbx (cbf/mccd via dxtbx)
│       │   ├── predict_fabio.py      # ImagePredictFabio (mar via fabio)
│       │   ├── orientation.py        # gs_mapping(), QuatLoss, Loss, batch_cross()
│       │   ├── ddp.py                # slurm_init(), find_free_port() for torch.distributed
│       │   ├── mpi.py                # get_host_comm(), get_gpu_id_mem()
│       │   ├── ice_mask.py           # IceMasker — ice-ring detection
│       │   ├── maxbin.py             # maximg_downsample() — quad extraction + maxpool DS
│       │   ├── mlp_fit.py            # MLP for B-factor → resolution conversion
│       │   ├── counter_utils.py      # utilities for spot counting models
│       │   └── gpu.py, qmags.py, multi_panel.py
│       └── tests/                    # pytest test suite
├── easyBragg/                        # Python wrappers for nanoBragg (reference / not used directly)
├── simforge/                         # micromamba conda installation — DO NOT EDIT
│   └── envs/simtbx_mpi/             # active conda env with simtbx, torch, h5py, etc.
├── load_resonet.sh                   # source this before ANY resonet command
├── run_hitfinder.sh                  # SLURM job template — full simulation run
├── run_hitfinder_resume.sh           # SLURM job template — partial-rank resumption
├── hitfinder_100k/                   # example 100k-shot dataset directory
└── hitfinder_data*/                  # smaller test datasets
```

---

## 3. CLI Entry Points

| Command | Module path | Purpose |
|---------|-------------|---------|
| `resonet-simulate` | `resonet.sims.runme:main` | GPU+MPI diffraction simulation |
| `resonet-simulate-joblib` | `resonet.sims.runme_joblib:main` | CPU-parallel simulation (joblib) |
| `resonet-train` | `resonet.net:main` | Train ResNet/LeNet model |
| `resonet-mergefiles` | `resonet.scripts.merge_h5s:main` | Merge per-rank HDF5 → virtual master |
| `resonet-viewsims` | `resonet.scripts.view_sims:main` | Visualize simulated images |
| `resonet-getsimdata` | `resonet.scripts.get_simdata:main` | Download PDB/scattering assets |
| `resonet-plotloss` | `resonet.scripts.plot_train:main` | Plot training loss curves |
| `resonet-compress` | `resonet.scripts.compress:main` | Compress HDF5 datasets |
| `resonet-decompress` | `resonet.scripts.decompress:main` | Decompress HDF5 datasets |
| `resonet-mfx` | `resonet.scripts.mfx:main` | MFX experiment utilities |

---

## 4. Complete Data Flow

```
[1] SIMULATION
resonet-simulate (MPI ranks, GPU)
  sims/runme.py → sims/main.run() → sims/simulator.Simulator.simulate()
  Physics: nanoBragg spots + air/water/plastic background + Poisson noise
  Output per rank: compressed{rank}.h5

[2] MERGE
resonet-mergefiles "outdir/compressed*.h5" master.h5
  scripts/merge_h5s.py → HDF5 virtual datasets (zero-copy)
  Output: master.h5 with virtual images/labels/geom datasets

[3] LOAD
H5SimDataDset(master.h5)  [loaders.py]
  Returns: (img_tensor, label_tensor) or (img, label, geom) or (img, label, sgnums)
  → torch DataLoader with DistributedSampler (if MPI)

[4] TRAIN
net.do_training()  [net.py]
  Model: ARCHES[arch](nout, dev, ...)  [params.py → arches.py]
  Loss:  LOSSES[loss]()               [params.py]
  Optimizer: SGD(lr, momentum=0.9)
  → nety_ep{N}.nn  (state_dict)
  → nety_ep{N}.chkpt  (full checkpoint with optimizer state + args)

[5] INFERENCE
ImagePredict  [utils/predict.py]
  load_model(checkpoint, arch)  [utils/eval_model.py]
  raw_img_to_tens_*(raw_img, mask, ds_fact, quad)  [utils/eval_model.py]
  → scalar predictions per image (reso, multi, ice, counts)
```

---

## 5. Model Registry (`params.py` → `ARCHES`)

| `--arch` | Class | Description |
|----------|-------|-------------|
| `le` | `LeNet` | Lightweight 3-conv + BN + MaxPool → FC |
| `res18` | `RESNetAny(netnum=18)` | ResNet-18, optional ImageNet pretrain |
| `res34` | `RESNetAny(netnum=34)` | ResNet-34, optional ImageNet pretrain |
| `res50` | `RESNetAny(netnum=50)` | **Default.** ResNet-50, IMAGENET1K_V2 |
| `res101` | `RESNetAny(netnum=101)` | ResNet-101 |
| `res152` | `RESNetAny(netnum=152)` | ResNet-152 |
| `counter` | `CounterRn` | ResNet for spot-count regression (→ scalar) |

**Not in ARCHES** (used directly in `ori_net.py`):  
`OriQuatModel` — ResNet50 backbone + Transformer encoder → normalized 4D quaternion

### RESNetAny architecture:
```
Input (batch, 1, 512, 512)
  → conv1 [OVERRIDDEN: 1ch in, 64ch out, 7×7]
  → ResNet backbone layers
  → AvgPool → 1000-dim
  → [Dropout(0.5)] → FC1(1000→num_fc) → ReLU → FC2(num_fc→nout)
  Optional: geometry tensor concatenated before final FC
  Optional: gs_mapping() 6D→9D rotation refinement
```

---

## 6. Loss Registry (`params.py` → `LOSSES`)

| `--loss` | PyTorch class | Use case |
|----------|---------------|----------|
| `L1` | `nn.L1Loss` | Regression (**default**) |
| `L2` | `nn.MSELoss` | Regression |
| `BCE` | `nn.BCELoss` | Binary classification (needs sigmoid in model) |
| `BCE2` | `nn.BCEWithLogitsLoss` | Binary classification (logits) |

**Special orientation losses** (`--oriMode`):
- `orientation.Loss` — angular loss with space-group symmetry ops (`--useSGNums`)
- `orientation.loss` — simple rotation matrix angular distance (no symmetry)

---

## 7. HDF5 File Schema

### Per-rank simulation output (`compressed{rank}.h5`)

| Dataset | Shape | Dtype | Notes |
|---------|-------|-------|-------|
| `images` | `(Nshot, 512, 512)` | uint16 | gzip-4, chunks=(1,512,512) |
| `labels` | `(Nshot, 31+)` | float32 | attr `names` lists column names |
| `geom` | `(Nshot, 5)` | float32 | attr `names`=["detdist","wavelen","pixsize","xdim","ydim"] |
| `nominal_mask` | `(H, W)` | bool | Valid-pixel mask from detector CBF |

### `labels` column order (from `main.py`):
```
reso, one_over_reso, radius, one_over_radius,
is_multi, multi_lat_angle_sigma, num_lat, bg_scale,
beamstop_rad, detdist, wavelen,
beam_center_fast, beam_center_slow, cent_fast_train, cent_slow_train,
Na, Nb, Nc, pdb, mos_spread, xtal_scale,
r1, r2, r3, r4, r5, r6, r7, r8, r9,   ← flattened 3×3 rotation matrix
pitch_deg, yaw_deg, bg_only
```

### Master file (`master.h5`)
- Same schema but backed by HDF5 virtual datasets — no data copy, just index mapping.
- Created by `scripts/merge_h5s.py`.

---

## 8. Key Design Invariants

- **Single-channel images**: All models override `conv1` to accept 1-channel (grayscale) input instead of ImageNet's 3-channel.
- **GPU assignment in MPI**: `dev_id = effective_rank % ngpu` — round-robin across available GPUs.
- **MPI library**: Simulation uses `libtbx.mpi4py.MPI` (CCTBX-bundled), NOT standard `mpi4py`. Training uses `torch.distributed` (NCCL).
- **simtbx env-only**: `simtbx`, `nanoBragg`, `libtbx`, `dxtbx` only importable inside `simtbx_mpi` conda env.
- **Checkpoint format**: `torch.save({"epoch", "model_state", "optimizer_state", "loss", "args"})` → `.chkpt` file.
- **DDP prefix**: `DistributedDataParallel` saves with "module." prefix; `restart_net.strip_names_in_state()` removes it on load.
- **Virtual HDF5 master**: `merge_h5s.py` creates a zero-copy index — modifying per-rank files after merge corrupts the master.
- **Lazy dataset open**: `H5SimDataDset` opens the HDF5 file on first `__getitem__` call (not in `__init__`), safe for PyTorch multiprocessing.
- **SGD optimizer**: Currently hardcoded SGD with Nesterov momentum; no Adam.
- **Half-precision**: `--half` flag converts model and data to float16 (experimental).

---

## 9. Simulation Constants (`sims/paths_and_const.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `DOMAINSIZE_MM` | 5e-5 | Crystal mosaic domain size (baseline) |
| `XTALSIZE_MM` | 0.025 | Sample thickness (25 µm) |
| `BEAM_SIZE_MM` | 0.03 | X-ray beam size (30 µm) |
| `FLUX` | 4e11 | Photons per pulse |
| `MOS_MIN / MOS_MAX` | 0.2 / 1.0 | Mosaicity bounds (degrees) |
| `STOL_MIN / STOL_MAX` | 0.15 / 0.35 | Resolution range (sin θ/λ) |
| `CUT_1P2` | True | Use `fmodel_1p2.mtz` (faster pre-computed) |
| `LAUE_MODE` | False | Polychromatic diffraction mode |
| `DIVERGENCE_MRAD` | 0 | Beam divergence (millirads) |
| `CENTER_WINDOW_MM` | 3 | Beam-center jitter window |

**Asset paths** (downloaded via `resonet-getsimdata`):
- `RANDOM_PDBS`: list of PDB folders in `for_tutorial/diffraction_ai_sims_data/pdbs/`
- `RANDOM_STOLS`: plastic scattering profiles in `for_tutorial/diffraction_ai_sims_data/randomstols/`
- `AIR_STOL`, `WATER_STOL`: air and water scattering profiles
- `SGOP_FILE`: `pdb_ops.npy` — space group symmetry operators for orientation loss

---

## 10. Supported Detectors

| Geometry | CBF file | Size | Pixel size | DS to 512×512 |
|----------|----------|------|------------|----------------|
| `eiger` | `eiger_1_00001.cbf` | 3840×3840 | 75 µm | quad 3× or center-crop 4× |
| `pilatus` | `pilatus_1_00001.cbf` | 2463×2527 | 172 µm | quad 2× or center-crop 3× |
| `mar` | `rayonix_1_00001.cbf` | 4096×4096 | 79.1 µm | quad 4× or center-crop 5× |

Downsampling uses `utils/maxbin.maximg_downsample()` (max-pooling, not average).

---

## 11. Orientation Math (`utils/orientation.py`)

**`gs_mapping(6 params) → 3×3 rotation matrix`**  
Gram-Schmidt orthogonalization: given `(a1_xyz, a2_xyz)`, computes orthonormal basis `(b1, b2, b3)`.  
Used in `arches.RESNetBase.forward()` when `ori_mode=True`.

**`QuatLoss` / `Loss`**  
Angular loss = `arccos(0.5 * (trace(R_model @ R_gt^T) - 1))` in radians.  
Symmetry-aware variant: takes min over all space-group symmetry operations `{S_i}`.  
Requires `sgnums` tensor (space group indices per sample) and pre-loaded `sgop_table`.

**Labels `r1–r9`**: Flattened row-major 3×3 rotation matrix stored in `labels` columns 21–29.

---

## 12. Training Loop Summary (`net.py → do_training`)

```python
# Setup
dataset = H5SimDataDset(h5name, labels, images, label_sel, use_geom)
model   = ARCHES[arch](nout=len(label_sel), dev=dev, ...)
loss_fn = LOSSES[loss]()
optim   = SGD(model.parameters(), lr=lr, momentum=0.9)

# Per epoch
for epoch in range(start_ep, max_ep):
    model.train()
    for imgs, labels[, geom_or_sgnums] in train_loader:
        preds = model(imgs[, geom])
        loss  = loss_fn(preds, labels[, sgnums=sgnums])
        loss.backward(); optim.step()
    
    model.eval()
    validate(test_loader, ...)   # accuracy, Pearson R, Spearman R
    
    if (epoch+1) % save_freq == 0:
        torch.save(model.state_dict(), f"nety_ep{epoch+1}.nn")
        save_checkpoint(f"nety_ep{epoch+1}.chkpt", ...)
```

**Validation metrics:**
- Regression (L1/L2): % within threshold, Pearson ρ, Spearman ρ
- Classification (BCE): accuracy, Jaccard index
- Orientation: mean angular error (degrees), % within threshold

---

## 13. SLURM / Environment Quick Reference

```bash
# Always required before any resonet command:
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh

# Additional export for GPU SLURM jobs:
export LD_LIBRARY_PATH=.../simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# Cluster settings:
# Partition: general   QOS: grp_cxfel   GPU: --gres=gpu:N

# Partial-run resumption:
resonet-simulate outdir --rankOffset 10 --totalRanks 20 ...
# Rank 0 → effective rank 10, writes compressed10.h5
```

**SLURM templates:**
- `run_hitfinder.sh` — full 20-rank simulation run
- `run_hitfinder_resume.sh` — resume specific ranks with `--rankOffset`

---

## 14. Inference Quick Reference (`utils/predict.py`)

```python
from resonet.utils.predict import ImagePredict

pred = ImagePredict(
    reso_model="nety_ep102.nn", reso_arch="res50",
    multi_model="multi_ep77.nn", multi_arch="res34",
    ice_model="ice_ep45.nn", ice_arch="res50",
    counts_model="counts_ep30.nn", counts_arch="counter",
    dev="cuda:0"
)

# For a raw 2D detector array + mask:
result = pred.predict(raw_img, mask, detdist_mm, wavelen_A, pixsize_mm)
# Returns dict: {"reso": float, "multi": float, "ice": float, "counts": float}
```

Preprocessing: maxbin downsample → sqrt-compress → extract 512×512 quad → normalize → tensor.

---

## 15. Where to Look for Common Questions

| Question | File(s) to open |
|----------|-----------------|
| How are models defined? | `resonet/resonet/arches.py` |
| How to add a new architecture? | `arches.py` + `params.py` (add to ARCHES dict) |
| How is training orchestrated? | `resonet/resonet/net.py` |
| How to add a new loss? | `params.py` (add to LOSSES dict) |
| How is data loaded from HDF5? | `resonet/resonet/loaders.py` |
| How does simulation work? | `sims/simulator.py`, `sims/main.py` |
| What labels are available? | `sims/main.py` param_names list |
| How are checkpoints saved/loaded? | `net.py:save_checkpoint()`, `restart_net.py` |
| How does inference preprocessing work? | `utils/eval_model.py` |
| How is orientation loss computed? | `utils/orientation.py` |
| How is DDP initialized? | `utils/ddp.py:slurm_init()` |
| How to merge HDF5 files? | `scripts/merge_h5s.py` |
| What are simulation constants? | `sims/paths_and_const.py` |
| How is geometry (detdist, wavelen) fed to models? | `arches.py:RESNetBase.forward()`, `loaders.py:H5SimDataDset` |
