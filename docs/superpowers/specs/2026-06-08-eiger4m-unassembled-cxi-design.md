# Spec: Full-Size Unassembled Detector Images (Eiger4M → CXI)

**Date:** 2026-06-08
**Branch:** `eiger4m-unassembled-cxi`

---

## Context

Resonet currently simulates diffraction images and aggressively downsamples them to
512×512 before storing in HDF5. The Hitfinder codebase has its own geometry-assembly
and downsampling pipeline (via `reborn`) that operates on unassembled CXI files keyed
by detector metadata. The goal is to make Resonet output full-size **unassembled**
detector images in CXI HDF5 format so that Hitfinder's pipeline can treat synthetic
data as a drop-in replacement for real experimental data, gaining access to full spatial
resolution and the real geometry-assembly path.

Start with Eiger4M, then extend to AGIPD, Epix10k, Jungfrau using the `.geom` files
already present in `/data/bioxfel/user/gihan/Resonet/geoms/`.

---

## Goals

1. Simulate full-size unassembled Eiger4M images (5632 × 384 pixels per frame)
2. Write output in CXI HDF5 format compatible with Hitfinder's `reborn` pipeline
3. Preserve all 43 existing CLI flags; add 3 new flags for the CXI path
4. Keep the existing 512×512 HDF5 path completely unchanged
5. Design extensible to AGIPD, Epix10k, Jungfrau via the same `.geom`-driven flow

---

## Approach

**Plan A (primary):** Parse CrystFEL `.geom` file → build dxtbx multi-panel Detector →
simulate with nanoBragg/simtbx → extract per-panel pixel blocks → write unassembled CXI
HDF5. Self-consistent: Hitfinder assembles exactly what was simulated.

**Plan B (fallback):** Keep using `eiger_1_00001.cbf` for simulation geometry; use
`.geom` purely for pixel extraction/remapping into the unassembled layout. Lower risk
but introduces a small geometry mismatch between simulation and Hitfinder assembly.
Acceptable for hitfinding; not suitable for orientation recovery.

**Fallback trigger:** if `geom_parser.py` detects any panel with
`dot(fast_axis, slow_axis) > 0.01`, or if dxtbx rejects the constructed geometry,
log a warning and fall back to Plan B automatically.

---

## New Components

```
resonet/resonet/sims/
  geom_parser.py     (NEW) CrystFEL .geom → dxtbx Detector + panel_map
  cxi_writer.py      (NEW) writes unassembled (N, ss, fs) CXI HDF5
  main.py            (MODIFIED) +3 new flags, new outfmt=cxi branch
  simulator.py       (POSSIBLY MODIFIED) if per-panel loop needed
```

---

## CLI

Three new flags added to the existing argparser in `main.py`. All 43 existing flags
are inherited unchanged by the CXI path.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--geomfile` | str | None | Path to CrystFEL `.geom` file |
| `--outfmt` | str | `hdf5` | Output format: `hdf5` (unchanged) or `cxi` |
| `--detector-name` | str | None | Detector string for CXI metadata (e.g. `"EIGER 4M"`) |

When `--outfmt hdf5` (default), behaviour is identical to today. The `--geomfile` and
`--detector-name` flags are ignored.

Example invocation for the new path:
```bash
srun --export=ALL resonet-simulate <outdir> \
  --nshot 10000 --outfmt cxi \
  --geomfile /data/bioxfel/user/gihan/Resonet/geoms/Eigar.geom \
  --detector-name "EIGER 4M" \
  --randDist --randDistRange 100 300 --ngpu=1 --randHit
```

---

## Data Flow

```
.geom file
    │
    ▼
geom_parser.py
    • reads panel blocks: corner_x/y, fs/ss axes, min/max_fs/ss, res, clen
    • converts CrystFEL lab-frame coords → dxtbx Panel (fast_axis, slow_axis, origin, pixel_size)
    • returns:
        dxtbx Detector  — multi-panel object for nanoBragg/simtbx
        panel_map       — list of {name, min_ss, max_ss, min_fs, max_fs}
    │
    ▼
simulator.py  (unchanged if simtbx handles multi-panel natively via dxtbx Detector)
    • receives dxtbx Detector with N panels
    • simulates diffraction → assembled or per-panel pixel array
    │
    ▼
main.py  (per-shot loop, outfmt=cxi branch)
    • uses panel_map to extract each panel's pixel block from simulated image
    • places block into unassembled array at [min_ss:max_ss+1, min_fs:max_fs+1]
    • sqrt compression + uint16 clipping applied per-panel (same as current path)
    │
    ▼
cxi_writer.py
    • accumulates frames → (N_rank, max_ss+1, max_fs+1) buffer
    • writes compressed_{rank}.cxi per MPI rank (see CXI format below)
```

---

## Coordinate Conversion (`geom_parser.py`)

CrystFEL `.geom` and dxtbx use different coordinate conventions:

| Property | CrystFEL `.geom` | dxtbx |
|---|---|---|
| Distance units | pixels (corner), metres (clen) | mm |
| Origin reference | beam center = (0, 0), z = 0 at sample | mm from detector reference point |
| Axis vectors | unit vectors in lab x/y plane | `fast_axis`, `slow_axis` unit vectors |
| y direction | +y upward | +y downward |

Per-panel conversion:
```python
pixel_size_mm = 1000.0 / res              # res is pixels/m in .geom
fast_axis = (fs_x, fs_y, 0.0)            # already unit vectors in .geom
slow_axis = (ss_x, ss_y, 0.0)
origin_mm = (
    corner_x * pixel_size_mm,            # x: same sign in both conventions
    -corner_y * pixel_size_mm,           # y: flip sign (geom +y up, dxtbx +y down)
    -clen * 1000.0                        # z: downstream is -z in dxtbx
)
panel_size = (max_fs - min_fs + 1, max_ss - min_ss + 1)   # (n_fast, n_slow)
```

---

## CXI Output Format

One `.cxi` file per MPI rank (mirrors current `.h5` per rank):

```
compressed_{rank}.cxi
  /entry_1/
    data_1/
      data                  (N_rank, ss_total, fs_total)  uint16  gzip-4  shuffle
    instrument_1/
      detector_1/
        description         str  e.g. "EIGER 4M"     ← --detector-name
        distance            f32  e.g. 0.300           (metres, from .geom clen)
        x_pixel_size        f32  e.g. 7.5e-5          (metres, from .geom res)
        y_pixel_size        f32  e.g. 7.5e-5
      source_1/
        energy              f32  e.g. 8750.0          (eV, from .geom photon_energy)
        wavelength          f32  e.g. 1.417e-10        (metres, derived: hc/energy)
    labels/                                            ← same label datasets as current HDF5
      (miller_indices, hit_flag, etc. — unchanged)
```

`resonet-mergefiles` extended to handle `.cxi` files (same merge logic, updated HDF5
path `/entry_1/data_1/data` and file extension).

---

## Unassembled Dimensions Per Detector

| Detector | Geom file | Unassembled shape (ss, fs) | ASICs | Pixel size |
|---|---|---|---|---|
| Eiger 4M | `Eigar.geom` | (5632, 384) | 64 | 75 µm |
| AGIPD 1M | `AGIPD.geom` | TBD (parse at impl) | TBD | 200 µm |
| ePix10k | `Epix10k.geom` | TBD | TBD | 100 µm |
| Jungfrau | `Junfrau.geom` | TBD | TBD | 75 µm |

AGIPD/Epix/Jungfrau shapes are derived at implementation time by running `geom_parser`
against their respective `.geom` files.

---

## Testing & Verification

**Unit tests** (`resonet/tests/`):

- `test_geom_parser.py`
  - parse `Eigar.geom` → assert 64 panels
  - assert pixel size = 0.075 mm
  - spot-check a few `corner_x/y` → `origin_mm` conversions against manual calculation
  - assert `dot(fast, slow) < 0.01` for every panel

- `test_cxi_writer.py`
  - write synthetic `(2, 5632, 384)` uint16 array
  - read back and assert shape, dtype, HDF5 paths, metadata values

**Integration test** (single-rank, CPU, 4 shots):
```bash
source setup_resonet.sh
resonet-simulate /tmp/cxi_test --nshot 4 --outfmt cxi \
  --geomfile geoms/Eigar.geom --detector-name "EIGER 4M" \
  --ngpu 0 --cpuMode
python -c "
import h5py
f = h5py.File('/tmp/cxi_test/compressed_0.cxi')
print(f['/entry_1/data_1/data'].shape)          # expect (4, 5632, 384)
print(f['/entry_1/instrument_1/detector_1/description'][()])
"
```

**Hitfinder round-trip** (manual, post-implementation):
- Feed a `.cxi` file into Hitfinder's `reborn` pipeline
- Confirm assembled image shows recognisable diffraction rings/spots
- Confirm hitfinding scores are non-trivial on synthetic hits vs. blanks

---

## Non-Goals

- No changes to the existing `--outfmt hdf5` path
- No changes to Hitfinder (Resonet only writes; Hitfinder reads as-is)
- No geometry optimisation or refinement (`.geom` files used verbatim)
- Orientation/quaternion label output not changed for this feature
