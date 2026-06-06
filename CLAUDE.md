# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project requires a specific HPC environment. Always source the environment before running any resonet commands:

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
```

This loads gcc-12.1.0, cuda-12.8.1, openmpi/5.0.8, and activates the `simtbx_mpi` conda environment located at `/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/`.

When running GPU jobs with srun/sbatch, also export:
```bash
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## Codebase Navigation

Before reading individual source files to answer questions or suggest improvements,
always read `/data/bioxfel/user/gihan/Resonet/MINDMAP.md` first. It contains the
complete module map, data flow, model registry, HDF5 schema, key design invariants,
and a "where to look" table mapping common questions to specific files.

Use MINDMAP.md to identify exactly which file(s) to open rather than scanning broadly.
Only fall back to reading raw source when MINDMAP.md is insufficient for the specific question.

## Package Installation

The main package lives in `resonet/resonet/` (a git submodule inside a build-files repo). Install in editable mode:

```bash
cd resonet && pip install -e .
```

> **Warning:** The active `simtbx_mpi` conda env has a *regular* (non-editable) install at
> `simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/`.
> Changes to source files under `resonet/resonet/sims/` are NOT auto-reflected.
> After editing source files, copy them manually:
> ```bash
> cp resonet/resonet/sims/<file>.py simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
> ```
> Or reinstall: `cd resonet && pip install -e .` (takes ~30s).

## Common Commands

**Simulate diffraction data (MPI, GPU):**
```bash
srun --export=ALL resonet-simulate <outdir> --nshot <N> --geom eiger --randDist --randDistRange 100 300 --ngpu=1 --randHit
```

**Download simulation data assets:**
```bash
resonet-getsimdata
```

**Train a model:**
```bash
resonet-train <epochs> <input.h5> <outdir> --arch res50 --lr 0.000125 --bs 16
```

**Merge HDF5 simulation outputs:**
```bash
resonet-mergefiles <pattern> <output.h5>
```

**Run tests:**
```bash
pip install pytest  # not bundled in simtbx_mpi env
cd resonet && python -m pytest resonet/tests/ -v
```

## Submitting Jobs (SLURM)

Use `run_hitfinder.sh` as a template. Key SLURM parameters for this cluster:
- Partition: `general`, QOS: `grp_cxfel`
- GPU resources: `--gres=gpu:N` (use 2–6 depending on rank count; scg020 has 8x H100)
- Always source `load_resonet.sh` inside the job script

To resume a training run from a checkpoint, use `run_hitfinder_resume.sh` (uses 4 GPUs).
SLURM job logs are written to the project root as `<jobname>_<jobid>.log`.

## Architecture Overview

```
resonet/              # build-files repo (pyproject.toml, setup.cfg)
  resonet/            # git submodule — actual source package
    net.py            # resonet-train entrypoint; training loop + CLI
    arches.py         # ResNet/LeNet/Transformer model definitions (OriQuatModel, RESNetAny, etc.)
    params.py         # ARCHES and LOSSES registries used by net.py
    loaders.py        # H5SimDataDset — PyTorch Dataset over HDF5 master files
    ori_net.py        # orientation-specific network variant
    td_net.py         # time-dependent network variant
    restart_net.py    # checkpoint resume logic
    laue.py           # Laue diffraction mode support
    sims/
      runme.py        # resonet-simulate entrypoint (MPI via libtbx.mpi4py + simtbx)
      runme_joblib.py # resonet-simulate-joblib entrypoint (CPU parallel via joblib)
      main.py         # simulation loop called by both runme variants
      simulator.py    # core diffraction simulator wrapping nanoBragg/simtbx
      paths_and_const.py  # simulation constants (PDB paths, beam params, mosaicity bounds)
    scripts/          # CLI utilities: compress/decompress, merge, view, image feeder/eater
    utils/
      eval_model.py   # image-to-tensor preprocessing (raw_img_to_tens_pil2, to_tens)
      predict.py      # inference helpers
      predict_dxtbx.py / predict_fabio.py  # detector-library-specific prediction wrappers
      orientation.py  # rotation/quaternion utilities
      ddp.py          # DistributedDataParallel helpers
      mpi.py          # MPI wrappers
    tests/            # pytest test suite
    cxidb/            # git submodule for CXI database utilities

easyBragg/            # separate package: Python wrappers for nanoBragg (simtbx_boost submodule)
simforge/             # micromamba-based conda installation (do not edit)
hitfinder_data*/      # HDF5 simulation output directories (test data)
hitfinder_100k/       # larger 100k-shot dataset
hitfinder_10k/        # 10k-shot dataset (current target)
test_shots*/          # additional test shot datasets
docs/
  plans/              # legacy implementation plans
  specs/              # legacy design specs
  superpowers/
    plans/            # feature plans (<feature-name>-<YYYY-MM-DD>.md)
    specs/            # feature specs (<feature-name>-<YYYY-MM-DD>.md)
```

## Data Format

Simulation outputs and training data are stored as HDF5 files (`compressed*.h5`). Each file contains:
- `images` dataset: diffraction images
- `labels` dataset: regression targets with optional `name` attribute for label selection
- Optional `geom` dataset: detector geometry parameters used by orientation models

The `resonet-mergefiles` script combines per-rank HDF5 outputs from MPI runs into a single master file for training.

## Feature Development Workflow

When discussing or implementing a new feature, always follow this sequence:

1. **`/superpowers:brainstorming`** — explore intent, requirements, and design options
2. **`/feature-dev:feature-dev`** — deep codebase analysis and architecture blueprint
3. Write to **`docs/superpowers/`** using the same `<feature-name>-<YYYY-MM-DD>` naming scheme for both:
   - `docs/superpowers/specs/<feature-name>-<YYYY-MM-DD>.md` — design decisions, data flow, invariants
   - `docs/superpowers/plans/<feature-name>-<YYYY-MM-DD>.md` — step-by-step implementation with exact file/line targets
4. **`/superpowers:executing-plans`** — execute the plan with review checkpoints

## Branching Rules

- **Significant features** (new functionality, non-trivial refactors) must be developed on a dedicated branch named after the feature (e.g., `my-feature-name`).
- **Branch naming**: use kebab-case, descriptive, matching the feature name used in `docs/superpowers/`.
- **After merging to main**: do NOT delete the feature branch — it may be revisited in future sessions.
- **Before opening a PR into main**, always run both:
  1. `/code-review` — checks for bugs, correctness, and code quality
  2. `/pr-review-toolkit:review-pr` — ensures compliance with project conventions

## Claude Code Integration

`.claude/settings.json` contains a `UserPromptSubmit` hook that reports remaining
SLURM job time (`ood-virtual-desktop`) at the start of each prompt.

## Key Design Points

- Models accept single-channel (grayscale) diffraction images; `conv1` is overridden from the ImageNet 1-channel default.
- `--arch` choices: `le` (LeNet), `res18/34/50/101/152` (ResNets), `counter` (CounterRn).
- Simulation depends on `simtbx` (CCTBX/nanoBragg) which is only available in the `simtbx_mpi` conda environment — it cannot be imported in a plain Python environment.
- MPI parallelism uses `libtbx.mpi4py` (CCTBX-bundled MPI), not plain `mpi4py`.
- GPU device assignment in MPI runs: `rank % ngpu`.
