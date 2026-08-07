# Resonet Modifications

This document covers all significant feature developments made to Resonet across four dedicated branches. Each section describes what changed, why, and how to use the new capabilities. The branches build on each other: `epix10k-gain-noise` (current) is a superset of all prior work.

---

## Branch Overview

| Branch | Status | PR | Summary |
|---|---|---|---|
| `fix/per-shot-memory-leak` | Merged | PR #2 | RSS logging, `gc.collect()`, `malloc_trim`, 3-phase memory leak fix |
| `eiger4m-unassembled-cxi` | Merged | — | CrystFEL geom parser, CXI writer, multi-panel GPU simulation |
| `multi-detector-presets` | Merged | PR #5 | 4 detector presets, `--geom` extension, `--fluxRange` |
| `epix10k-gain-noise` | Open | pending | ePix10k 3-gain noise model, saturation clip, geometry bug fixes, hardened geom parser |

---

## 1. Memory Leak Fix (`fix/per-shot-memory-leak` → PR #2)

### Why
During long MPI simulation runs, each rank's RSS grew unboundedly, causing out-of-memory crashes at ~160 out of 5000 shots. Three root causes were identified and fixed across 3 phases.

### What Changed

**`resonet/resonet/sims/main.py`**
- `gc.collect()` called after every shot to release Python cycle-collected objects
- `_malloc_trim(0)` via `ctypes.CDLL("libc.so.6")` returns glibc arena fragmentation to the OS after each GC pass
- `/proc/self/status` VmRSS read every 10 shots for current RSS (not the monotonically-increasing `ru_maxrss`); falls back to `ru_maxrss` with a diagnostic print if `/proc` is unavailable
- `del Bfac_img, all_spots` immediately after the noise loop to drop per-shot simulation arrays without waiting for GC

### Outcome
Residual leak ~1.5 MB/shot. Confirmed safe for 20k-shot runs peaking at ~11 GB/rank on scg020 (256 GB RAM).

### How to Observe
RSS is logged automatically — no user action needed. Look for lines like:
```
RANK 1/5: Shot 50 RSS=2134 MB
```
If `/proc/self/status` is unavailable the log will say:
```
RANK 1/5: /proc/self/status unreadable (...); using ru_maxrss.
```

---

## 2. Eiger4M Unassembled CXI Output (`eiger4m-unassembled-cxi`)

### Why
The ML training pipeline needs full-resolution unassembled multi-panel images in CrystFEL/CXIDB HDF5 format (`.cxi`). The previous output was assembled 512×512 HDF5 only, which discards panel geometry and can't be read by CrystFEL tools.

### What Changed

#### New: `resonet/resonet/sims/geom_parser.py`
Parses CrystFEL `.geom` files into dxtbx `Detector` objects.

Returns `(detector, panel_map, globals_dict)` where:
- `detector` — dxtbx multi-panel `Detector` ready for `simtbx`
- `panel_map` — list of dicts with `min_fs/max_fs/min_ss/max_ss/n_fast/n_slow` for unassembled image placement
- `globals_dict` — `{'clen': float, 'res': float, 'photon_energy': float}`

Handles LCLS-style dynamic `clen` HDF5 references (rejects with a clear error), floating-point `res`, and comment lines (`;`).

#### New: `resonet/resonet/sims/cxi_writer.py`
Streams per-shot unassembled images to HDF5 in CXI layout.

```
/entry_1/data_1/data              — uint16 images (n_shots, n_ss, n_fs)
/entry_1/instrument_1/detector_1/ — distance, x/y pixel size, photon energy
/entry_1/instrument_1/source_1/   — wavelength, energy
/LCLS/                            — detector name, shot labels
```

Accepts arbitrary per-shot label dict (`hit`, `detector_distance`, `wavelength`, `flux`). Pre-allocates extendable datasets; safe `close()` in `finally` block.

#### `resonet/resonet/sims/main.py` (modified)
- `--outfmt cxi` — selects CXI output (default: `hdf5`)
- `--geomfile PATH` — path to CrystFEL `.geom` file (required when `--outfmt cxi`)
- `--detector-name STRING` — written to CXI metadata (e.g. `"EIGER 4M"`)
- CXI simulation path: calls `simulate(multi_panel=True)`, assembles panel flat arrays into the unassembled frame, writes via `CXIWriter`

#### `resonet/resonet/sims/simulator.py` (modified)
- `simulate(multi_panel=True)` — returns a flat 1D array of all panel pixels concatenated in panel order instead of a 2D image
- `sim_background_multipanel()` — single GPU kernel call covers all panels via `exascale_api.add_background(gpu_det)`, eliminating per-panel loops

#### `resonet/resonet/scripts/merge_h5s.py` (modified)
- `resonet-mergefiles` gained `--cxi` flag for merging per-rank `.cxi` outputs into a single master file

#### New: `resonet/resonet/sims/geoms/Eiger4m.geom`
64-panel EIGER 4M geometry (848 lines, CrystFEL format).
- clen = 300 mm, photon_energy = 8750 eV, res = 10000.075 px/m

### How to Use

```bash
# Manual CXI output (explicit geomfile)
srun --export=ALL resonet-simulate <outdir> --nshot 5000 \
    --outfmt cxi \
    --geomfile resonet/resonet/sims/geoms/Eiger4m.geom \
    --detector-name "EIGER 4M" \
    --ngpu=1 --randHits --randDist --randDistRange 100 300

# Merge per-rank CXI files
resonet-mergefiles "<outdir>/compressed*.cxi" merged.cxi --cxi
```

---

## 3. Multi-Detector Presets (`multi-detector-presets` → PR #5)

### Why
Passing `--geomfile` and `--detector-name` manually every run was error-prone. A preset system lets `--geom <name>` auto-configure everything. Per-shot flux variation was also added so simulations can span a realistic range of beam intensities without rerunning at fixed flux.

### What Changed

#### `resonet/resonet/sims/paths_and_const.py` (modified)
```python
MULTI_PANEL_PRESETS = {
    "agipd":    (os.path.join(_GEOMS_DIR, "AGIPD.geom"),    "AGIPD 1M"),
    "jungfrau": (os.path.join(_GEOMS_DIR, "Jungfrau.geom"), "Jungfrau 4M"),
    "epix10k":  (os.path.join(_GEOMS_DIR, "Epix10k.geom"),  "ePix10k 2.2M"),
    "eiger4m":  (os.path.join(_GEOMS_DIR, "Eiger4m.geom"),  "EIGER 4M"),
}
CALIB_NOISE_PCT = 3   # % pixel-to-pixel gain variation (used by ePix10k noise model)
```

#### `resonet/resonet/sims/main.py` (modified)
- `--geom` now accepts `agipd | jungfrau | epix10k | eiger4m` (previously only `eiger | pilatus | mar`)
- Multi-panel preset auto-configures: forces CXI output, resolves `--geomfile` and `--detector-name` from `MULTI_PANEL_PRESETS` — no manual flags needed
- `--fluxRange MIN MAX` — per-shot flux drawn uniformly from [MIN, MAX] photons/pulse; stored in CXI labels as `flux`
- Rank 0 prints `INFO:` messages confirming preset resolution

#### New geometry files (`resonet/resonet/sims/geoms/`)

| File | Detector | Panels | clen | Energy | Notes |
|---|---|---|---|---|---|
| `AGIPD.geom` | AGIPD 1M | 1024 | 151.65 mm | 9385 eV | EuXFEL; **known panel-ordering bug — not for production** |
| `Jungfrau.geom` | Jungfrau 4M | 8 | 103 mm | 11560 eV | adu_per_photon = 478.6 |
| `Epix10k.geom` | ePix10k 2.2M | 16 | 96 mm | 9500 eV | 4-quadrant assembly |
| `Eiger4m.geom` | EIGER 4M | 64 | 300 mm | 8750 eV | Standard LCLS detector |

#### `resonet/setup.cfg` (modified)
Added `[options.package_data]` so `.geom` and `.cbf` files are distributed with `pip install`:
```ini
[options.package_data]
resonet.sims.geoms = *.geom
resonet.sims = *.cbf
```

### How to Use

```bash
# Preset — no --geomfile or --detector-name needed
srun --export=ALL resonet-simulate <outdir> --nshot 5000 \
    --geom eiger4m --ngpu=1 --randHits --randDist --randDistRange 100 300

# Per-shot flux variation
srun --export=ALL resonet-simulate <outdir> --nshot 5000 \
    --geom jungfrau --ngpu=1 --randHits --fluxRange 1e11 4e11
```

**New SLURM template:** `run_cxi_1k.sh` — Eiger4M, 5 ranks × 200 shots = 1k total.

---

## 4. ePix10k Gain-Switching Noise Model (`epix10k-gain-noise`)

### Why
The ePix10k 2.2M is a 3-gain auto-ranging detector: each pixel independently operates in High-Gain (HG), Medium-Gain (MG), or Low-Gain (LG) mode depending on its photon count. Each mode has different readout noise characteristics, and the LG mode has a physical saturation limit (11,000 photons at 8–9 keV). nanoBragg's built-in noise model assumes a single gain mode and has no saturation concept, so a dedicated per-pixel noise pipeline was needed.

### What Changed

#### `resonet/resonet/sims/make_sims.py` (modified)

New function `apply_epix_noise(img, t1, t2, sigma_hg, sigma_mg, sigma_lg, sat_lg, rng)`:

Physical noise pipeline in order:
1. **Poisson shot noise** — photon counting statistics (`rng.poisson(max(img, 0))`)
2. **Gain-zone classification** — `[0, t1]` → HG; `(t1, t2]` → MG; `> t2` → LG
3. **Gaussian readout noise** — zone-dependent RMS added per pixel
4. **Non-negative floor** — `max(out, 0)`
5. **LG saturation clip** — `min(out, sat_lg)` applied to full array (not just the LG zone mask, which is stale after step 3)

Returns `float32` array clipped to `[0, sat_lg]`. Accepts `rng` for seeded reproducibility.

`set_noise()` default `calib_noise_percent` now reads `paths_and_const.CALIB_NOISE_PCT` instead of a hardcoded `3`.

#### `resonet/resonet/sims/simulator.py` (modified)

New `Simulator` attributes:

| Attribute | Default | Description |
|---|---|---|
| `self.flux` | `paths_and_const.FLUX` | Per-shot flux; set per-shot externally via `HS.flux` |
| `self.epix_mode` | `False` | Enable ePix10k noise pipeline; skips nanoBragg `add_noise()` |
| `self.epix_gain_thresh` | `(80, 270)` | HG→MG and MG→LG thresholds in photon counts |
| `self.epix_noise_sigma` | `(0.02, 0.023, 0.27)` | Readout noise RMS per zone (photon-equivalent) |
| `self.epix_sat_lg` | `11000` | LG well-capacity saturation limit (photon counts) |
| `self._epix_rng` | unseeded | Persistent RNG; seeded from `seeds[jid]` by `main.py` |

ePix10k noise pipeline inside `simulate()`:
1. Linear flux rescale: `img *= self.flux / paths_and_const.FLUX`  
   *(correct because both Bragg spots and background scale identically with photon flux)*
2. Calibration jitter: per-pixel `×Normal(1.0, CALIB_NOISE_PCT/100)` — models pixel-to-pixel gain variation
3. `apply_epix_noise(img × calib, ...)` — Poisson + readout + saturation clip

**Bug fixes also included:**
- `shift_center(det, dx, dy)` — rigid-body translation: now uses panel-0 axes for all panels. Previously applied per-panel axes, which sheared multi-panel assemblies and triggered dxtbx panel-normal assertion failures.
- `shift_distance(det, dz)` — sign corrected: now subtracts along normal (`O − N×dz`). Previously added, which moved ePix10k panels toward the sample and pushed them into positive z, crashing dxtbx.

#### `resonet/resonet/sims/geom_parser.py` (modified)

New helper functions for geometrically robust axis handling:
- `_normalize(v)` — unit vector; raises `ValueError` for zero-length input
- `_orthogonalize(fast, slow)` — Gram-Schmidt orthogonalization; raises `ValueError` if axes are parallel

Both are applied to every panel's fast/slow axes during `parse_geom()`. This was required because ePix10k panel axes in the `.geom` file have a near-zero but non-zero dot product (~3×10⁻⁶) that nanoBragg would otherwise reject.

`_parse_axis(s)` — now raises `ValueError` if the parsed result is the zero vector (no x/y/z tokens found), with a clear error message.

`parse_geom()` wraps all axis parsing with try-except providing panel name and axis type in the error message. Global field parsing handles LCLS-style HDF5 `clen` references by rejecting them explicitly.

#### `resonet/resonet/sims/main.py` (modified)

New CLI arguments (all active only with `--geom epix10k`):

| Flag | Default | Description |
|---|---|---|
| `--epixGainThresh T1 T2` | `80 270` | HG→MG / MG→LG boundaries in photon counts |
| `--epixNoiseSigma HG MG LG` | `0.02 0.023 0.27` | Readout noise RMS per zone (photon-equivalent) |
| `--epixSatLG SAT` | `11000` | LG well-capacity saturation limit (photon counts) |

Other changes:
- All three ePix10k arguments validated at startup before any I/O opens
- `--fluxRange` validation also moved before CXI writer opens
- `assert d1 < d2` / `assert en1 < en2` replaced with `raise ValueError` (safe under Python `-O` flag)
- `_epix_rng` seeded from `seeds[jid]` immediately after `HS.epix_mode = True` for deterministic, rank-reproducible noise
- `params['flux']` written to CXI labels for every shot
- RSS logging fallback now prints a diagnostic message instead of silently switching from VmRSS to `ru_maxrss`

#### Tests added

**`resonet/resonet/tests/test_simulator_funcs.py`**
- `test_apply_epix_noise_output_shape_and_dtype` — output shape and dtype match input
- `test_apply_epix_noise_nonnegative` — no negative pixel values after noise
- `test_apply_epix_noise_saturation_clip` — high-flux pixels clipped exactly to `sat_lg`
- `test_apply_epix_noise_deterministic_with_rng` — same seed → identical output
- `test_shift_center_applies_to_all_panels` — updated to verify all panels receive identical lab-frame displacement (rigid-body correctness)

**`resonet/resonet/tests/test_geom_parser.py`**
- `test_normalize_zero_vector_raises` — `_normalize((0,0,0))` raises `ValueError`
- `test_orthogonalize_parallel_axes_raises` — parallel fast/slow raises `ValueError`
- `test_parse_axis_no_xyz_token_raises` — `_parse_axis("0.5")` raises `ValueError`

**`resonet/resonet/tests/conftest.py`**
- `GEOM_PATH` updated from `Eigar.geom` → `Eiger4m.geom` (typo rename)

#### New SLURM templates

| Script | Ranks | Shots/rank | Total | Wall time |
|---|---|---|---|---|
| `run_cxi_epix10k_1k.sh` | 5 | 200 | 1,000 | 1 h |
| `run_cxi_epix10k_5k.sh` | 5 | 1,000 | 5,000 | 3 h |
| `run_cxi_epix10k_2rank_5k.sh` | 2 | 2,500 | 5,000 | 2 h |

### How to Use

```bash
# ePix10k with defaults — preset handles geomfile, detector-name, CXI format
srun --export=ALL resonet-simulate <outdir> \
    --nshot 5000 --geom epix10k --ngpu=1 \
    --randHits --randDist --randDistRange 100 300 \
    --randWave --randCent --varyBgScale \
    --fluxRange 5e10 4e11

# Override gain thresholds and readout noise
srun --export=ALL resonet-simulate <outdir> \
    --nshot 1000 --geom epix10k --ngpu=1 \
    --epixGainThresh 60 240 \
    --epixNoiseSigma 0.015 0.020 0.25 \
    --epixSatLG 11000 \
    --fluxRange 5e10 4e11 --randHits

# Post-run: verify saturation clip and pixel stats
python3 -c "
import h5py, numpy as np
with h5py.File('<outdir>/compressed0.cxi', 'r') as f:
    imgs = f['entry_1/data_1/data'][:].astype(np.float32)
flat = imgs.ravel()
print(f'min={flat.min():.0f}  max={flat.max():.0f}  mean={flat.mean():.2f}')
print(f'n_saturated={np.sum(flat >= 11000)}  n_above_sat={np.sum(flat > 11000)}')
"

# Merge per-rank CXI files after job completes
resonet-mergefiles "epix10k_5k/compressed*.cxi" epix10k_5k_merged.cxi --cxi
```

---

## Physics Notes

**FLUX = 4×10¹¹** is the total photons per pulse across the full 30 µm beam — not per-pixel. It is the reference level at which background and spot intensities are precomputed.

**`--fluxRange 5e10 4e11`** spans a realistic dose range for ePix10k experiments: the low end (5×10¹⁰) is ~8× weaker than nominal; the high end (4×10¹¹) is the standard LCLS dose.

**Flux rescaling** of the combined (spots + background) image is correct: both Bragg diffraction and background scatter scale identically with photon flux, so multiplying the combined image by `flux / FLUX` gives the right result for any per-shot flux value.

**Saturation physics:** `sat_lg = 11000` photons corresponds to the LG well capacity of the ePix10k detector at 8–9 keV. Pixel values above this cannot be measured — the ADC clips them. The full-array clip in `apply_epix_noise` is intentional: after readout noise is added, some pixels that were in the MG zone may cross the LG threshold, and the stale LG mask would miss them.

**Noise model order:** Gain-zone classification is done on Poisson-drawn counts (not on the noiseless image). This is physically correct — the detector's analog circuitry switches gain based on the actual number of photons collected, not the expected value.

---

## Running Tests

```bash
source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/ -v
```

Tests that require dxtbx (`test_simulator_funcs.py`, `test_geom_parser.py`) run in the `simtbx_mpi` environment. Tests for `apply_epix_noise` are pure numpy and pass in any environment with numpy installed.
