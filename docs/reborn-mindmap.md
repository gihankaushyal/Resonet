# reborn Package Mindmap

Source: `/data/bioxfel/user/gihan/reborn/`
Purpose: X-ray diffraction data analysis for pixel-array detectors (PADs) at XFELs and synchrotrons.

## Where to Look

| Question | File |
|----------|------|
| Detector geometry model | `reborn/detector.py` (PADGeometry line 97, PADGeometryList line 1092) |
| Data container | `reborn/dataframe.py` (DataFrame class) |
| CXI/HDF5 file reading | `reborn/external/crystfel.py` (CXIFrameGetter line 529) |
| Generic frame I/O | `reborn/fileio/getters.py` (FrameGetter ABC) |
| Image display | `reborn/viewers/qtviews/padviews.py` (PADView) |
| Geometry file loading | `reborn/external/crystfel.py` (geometry_file_to_pad_geometry_list) |
| LCLS / psana integration | `reborn/external/lcls.py` |
| EuXFEL / extra_data | `reborn/external/euxfel.py` |
| Built-in detector presets | `reborn/detector.py` (lines 2144+), `reborn/data/geom/*.json` |
| CrystFEL .geom examples | `reborn/data/geom/cspad_crystfel.geom`, `pnccd_crystfel.geom`, etc. |

---

## Core Data Model

### PADGeometry (single detector panel)
Five required fields, all in SI units (meters):

| Field | Type | Meaning |
|-------|------|---------|
| `n_fs` | int | Pixels in fast-scan direction (rightmost numpy index) |
| `n_ss` | int | Pixels in slow-scan direction (leftmost numpy index) |
| `t_vec` | (3,) float64 | Lab position of corner pixel (0,0) in meters from sample |
| `fs_vec` | (3,) float64 | Displacement to next pixel along fast-scan; magnitude = pixel size |
| `ss_vec` | (3,) float64 | Displacement to next pixel along slow-scan; magnitude = pixel size |

Pixel position in lab frame:
```
pos(i_ss, i_fs) = t_vec + i_ss * ss_vec + i_fs * fs_vec   (meters, 3D)
```

Optional slicing metadata (for panels embedded in larger arrays):

| Field | Meaning |
|-------|---------|
| `parent_data_shape` | Shape of the source array (e.g., [5632, 384] for ePix10k) |
| `parent_data_slice` | numpy slice selecting this panel's region from the parent array |

### PADGeometryList
Python `list` subclass; one `PADGeometry` per panel. Multi-panel detectors are
a `PADGeometryList` of N panels.

### DataFrame (single exposure event)
Central data carrier:
- `raw_data`: ndarray shaped (n_pads, n_ss, n_fs) or list of 2D arrays
- `pad_geometry`: PADGeometry or PADGeometryList
- `beam`: Beam object (wavelength, direction, polarization)
- `mask`: optional, same shape as raw_data
- `processed_data`: optional, post-correction data
- `parameters`: dict of arbitrary metadata

---

## Coordinate System

- **Origin**: sample/interaction point
- **Units**: meters (SI) for all position vectors
- **Beam**: typically along −z (user-configurable)
- **fs axis**: rightmost numpy index; ss axis: leftmost numpy index
- **Panel normal**: `normalize(fs_vec × ss_vec)` points away from sample

Derived quantities (all require a `Beam` object):
- `q_vecs(beam)` — scattering vectors (m⁻¹)
- `s_vecs()` — unit vectors from sample to each pixel
- `solid_angles()` — per-pixel solid angle
- `polarization_factors(beam)` — correction factors

---

## Geometry File Formats

### JSON (reborn native)
Stored in `reborn/data/geom/*.json`. Example single-panel entry:
```json
{
  "n_fs": 2070, "n_ss": 2167,
  "fs_vec": [7.5e-05, 0.0, 0.0],
  "ss_vec": [0.0, 7.5e-05, 0.0],
  "t_vec": [-0.07755, -0.081075, 0.0]
}
```
Multi-panel: JSON array of such dicts. `parent_data_shape` and `parent_data_slice`
fields define how each panel maps into the raw data array.

### CrystFEL .geom
Parsed by `reborn/external/crystfel.py` → `geometry_file_to_pad_geometry_list()`.

Key CrystFEL → reborn conversions:

| CrystFEL field | Reborn equivalent | Notes |
|---------------|-------------------|-------|
| `res` (px/m) | `\|fs_vec\|` = `\|ss_vec\|` = 1/res | Pixel size in meters |
| `corner_x, corner_y` (px) | `t_vec[:2]` | Multiplied by pixel size to get meters |
| `clen + coffset` (m) | `t_vec[2]` | Negated: t_vec[2] = −(clen+coffset) |
| `fs = ax + by` | `fs_vec` = (a/res, b/res, 0) | Direction scaled by pixel size |
| `ss = ax + by` | `ss_vec` = (a/res, b/res, 0) | Same |
| `min_ss, max_ss, min_fs, max_fs` | `parent_data_slice` | Slice into the assembled 2D array |

---

## Image Display Pipeline

```
CXI/HDF5 file
   ↓
CXIFrameGetter / SimpleH5FrameGetter   (reborn/fileio/getters.py)
   ↓ reads /entry_1/data_1/data[frame, :, :]
DataFrame (raw_data + pad_geometry + beam + mask)
   ↓
PADView.__init__()                      (padviews.py)
   ↓ creates one ImageItemPlus per panel + ViewBox
_apply_pad_transform(panel)
   ↓ builds QTransform from (fs_vec, ss_vec, t_vec) → screen coords
   ↓ setTransform() on each ImageItemPlus
update_pads()
   ↓ data[i].shape = (n_ss, n_fs) per panel
   ↓ ImageItemPlus[i].setImage(data[i])
pyqtgraph renders all panels at correct lab-frame positions projected to 2D
```

**Key design decisions:**
- Panels rendered **independently** — no global assembly step in the GUI
- `PADAssembler` (detector.py ~line 1846) only used for static export/PNG, not interactive display
- QTransform encodes the affine lab-frame → screen projection for each panel

---

## Built-in Detector Presets

Functions in `reborn/detector.py` (lines 2144+) return a `PADGeometryList`.
All accept optional `detector_distance` (m) and `binning` parameters.
Panel counts and shapes verified directly from `reborn/data/geom/*.json`.

| Preset function | Detector | Panels | Panel shape (n_ss × n_fs) | Pixel size |
|----------------|----------|--------|--------------------------|-----------|
| `cspad_pad_geometry_list()` | CSPAD | 64 | 185 × 388 | 109.9 µm |
| `cspad_2x2_pad_geometry_list()` | CSPAD 2×2 | 4 | 1472 × 1480 | — |
| `jungfrau4m_pad_geometry_list()` | Jungfrau 4M | 64 | 256 × 256 | 75 µm |
| `eiger4M_pad_geometry_list()` | Eiger 4M | 1 (assembled) | 2167 × 2070 | 75 µm |
| `epix100_pad_geometry_list()` | ePix100 | 4 | 176 × 192 | 100 µm |
| `epix10k_pad_geometry_list()` | ePix10k | 64 | 176 × 192 | 100 µm |
| `agipd_pad_geometry_list()` | AGIPD 1M | 128 | 64 × 128 | 200 µm |
| `mpccd_pad_geometry_list()` | MPCCD (SACLA) | — | — | — |
| `pnccd_pad_geometry_list()` | pnCCD | 4 | 512 × 512 | — |
| `pilatus100k_pad_geometry_list()` | Pilatus 100K | 1 | 487 × 407 | 172 µm |
| `rayonix_mx340_xfel_pad_geometry_list()` | Rayonix MX340 | 1 (assembled) | 3840 × 3840 | ~79 µm |

---

## CXI File Reading

**Standard HDF5 path:** `/entry_1/data_1/data`
- Shape: `(n_frames, n_ss, n_fs)` — one 2D frame per exposure
- `CXIFrameGetter` reads one frame at a time: `h5_file["/entry_1/data_1/data"][frame_idx, :, :]`
- Data cast to `np.double` automatically

Other paths read by reborn:
- `/entry_1/instrument_1/source_1/energy` — photon energy (eV)
- `/entry_1/instrument_1/detector_1/distance` — detector distance (m)
- `/entry_1/instrument_1/detector_1/x_pixel_size` / `y_pixel_size` — pixel size (m)

---

## Supported Facilities

| Facility | Module | Data access | Key detectors |
|----------|--------|-------------|---------------|
| LCLS | `external/lcls.py` | psana | CSPAD, ePix, Rayonix, Jungfrau |
| EuXFEL | `external/euxfel.py` | extra_data | AGIPD |
| Generic CXI | `external/crystfel.py` | h5py | Any with .geom file |
| Custom | `fileio/getters.py` | FrameGetter ABC | Any |

---

## Comparison: Resonet CXI Output vs reborn Expectations

### What Resonet writes (cxi_writer.py)

```
compressed<rank>.cxi
├── entry_1/data_1/data               shape: (n_frames, n_ss, n_fs), uint16
├── entry_1/labels/hit                shape: (n_frames,), float  [0=non-hit, 1=hit]
├── entry_1/labels/wavelength         shape: (n_frames,), float
├── entry_1/labels/detector_distance  shape: (n_frames,), float
├── entry_1/instrument_1/detector_1/description  str (e.g., "ePix10k 2.2M")
├── entry_1/instrument_1/detector_1/distance     float (m)
├── entry_1/instrument_1/detector_1/x_pixel_size float (m)
├── entry_1/instrument_1/detector_1/y_pixel_size float (m)
├── entry_1/instrument_1/source_1/energy         float (eV)
└── entry_1/instrument_1/source_1/wavelength     float (m)
```

### Compatibility assessment per detector

All panel counts and assembled shapes verified against reborn JSON files and Resonet
smoke-test output.

| Detector | reborn panels | reborn parent_data_shape | Resonet panels | Resonet CXI shape | Compatible? |
|----------|--------------|--------------------------|----------------|-------------------|-------------|
| **ePix10k** | 64 (176×192 each) | [5632, 384] | 64 | (n, 5632, 384) | ✓ **Full match** — same panel count, same assembled shape; use `epix10k_pad_geometry_list()` or CrystFEL loader with Epix10k.geom |
| **Jungfrau 4M** | 64 (256×256 each) | [4096, 1024] | 8 (module-level) | (n, 2164, 2068) | ⚠ **Granularity mismatch** — Jungfrau.geom uses 8 module-level panels; reborn JSON uses 64 ASICs (256×256); assembled shapes differ. Must use the same .geom file in both tools for consistent slicing. |
| **AGIPD 1M** | 128 (64×128 each) | [16, 512, 128] (3D) | 128 | (n, 512, 128) | ⚠ **Layout incompatibility** — reborn uses 3D parent [16 modules × 512 × 128]; Resonet's geom_parser ignores `dim0 = %` (module dim) and maps all 128 ASICs into a flat 2D (512×128), causing overlapping writes. Resonet AGIPD output is **currently incorrect**. |
| **Eiger 4M** | 1 assembled (2167×2070) | None | 64 | (n, 5632, 384) | ⚠ **Built-in is assembled** — reborn's `eiger4M_pad_geometry_list()` returns 1 assembled panel. For Resonet's unassembled CXI, load via `geometry_file_to_pad_geometry_list("Eigar.geom")` instead. |
| **ePix100** | 4 (176×192 each) | None | ✗ no geom file | N/A | ✗ **Not in Resonet** — no .geom file or CBF template; add `Epix100.geom` to simulate. |
| **Rayonix MX340** | 1 assembled (3840×3840) | None | `--geom mar` (HDF5 only) | (n, 512, 512) downsampled | ⚠ **HDF5 only in Resonet** — `--geom mar` outputs assembled HDF5; reborn preset expects 3840×3840 but Resonet downsamples to 512×512 for training. No CXI path. |
| **Pilatus 6M** | 1 assembled (487×407 preset) | None | `--geom pilatus` (HDF5 only) | (n, 512, 512) downsampled | ⚠ **HDF5 only** — assembled HDF5 output only; Pilatus CBF is 2463×2527 pixels, downsampled by Resonet. No CXI path. |
| **CSPAD** | 64 (185×388 each) | [32, 185, 388] | ✗ not supported | N/A | ✗ **Not in Resonet** — no geom file or CBF. |

### How to load Resonet CXI files in reborn (compatible detectors)

```python
from reborn.external.crystfel import geometry_file_to_pad_geometry_list
import h5py, numpy as np

# Step 1 — load geometry from the SAME .geom file used to simulate
pad_geom = geometry_file_to_pad_geometry_list(
    "/data/bioxfel/user/gihan/Resonet/geoms/Epix10k.geom"
)

# Step 2 — load a CXI frame
with h5py.File("compressed0.cxi") as f:
    frame = np.array(f["/entry_1/data_1/data"][0])  # shape (n_ss, n_fs)
    hit   = f["/entry_1/labels/hit"][0]             # 1=hit, 0=non-hit

# Step 3 — slice each panel's data using parent_data_slice
for pad in pad_geom:
    panel_data = frame[pad.parent_data_slice]  # shape (n_ss_panel, n_fs_panel)
```

Resonet's `_n_ss` and `_n_fs` (computed from `max_ss+1` / `max_fs+1` across all panels
in `main.py`) match reborn's `parent_data_shape` exactly when both tools use the same
.geom file — guaranteed consistent.

### Known issue: AGIPD geom_parser fix required

AGIPD's CrystFEL .geom uses `dim0 = %` (module index), making it a 3D layout
`[16 modules, 512 ss, 128 fs]`. Resonet's `geom_parser.py` ignores this extra dimension
and incorrectly maps all 128 ASICs into a flat 2D (512×128) space, causing panels to
overwrite each other. Fix: detect `dim0 = %` in the parser and output a proper
`(16×8×64, 128)` or geometrically-correct assembled layout. **AGIPD simulation output
is not usable until this is addressed.**

---

## Design Invariants (reborn)

1. **PAD = single rectangular pixel grid** — no curved or hexagonal panels
2. **fs_vec ⊥ ss_vec** — orthogonality strictly required (dxtbx enforces at C++ level)
3. **SI units everywhere** — meters, radians; no pixel-unit or mm geometry
4. **Lazy evaluation + caching** — position vectors computed on first access via `@cached`
5. **Beam required for q/scattering** — geometry alone cannot compute q-vectors
6. **Unassembled internally** — GUI displays panels separately via per-panel QTransform; assembly is export-only
