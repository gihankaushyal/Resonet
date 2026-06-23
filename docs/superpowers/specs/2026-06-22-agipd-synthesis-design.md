# AGIPD 1M Image Synthesis — Design Spec (Revised)

**Date:** 2026-06-22
**Supersedes:** `docs/superpowers/specs/2026-06-19-agipd-synthesis-design.md`
**Branch:** `feature/agipd-synthesis`
**Status:** Approved for implementation

---

## Context

Resonet can simulate diffraction images for Eiger4M, ePix10k, and Jungfrau detectors via
`--geom <preset>`. AGIPD 1M is next. The AGIPD preset already exists in
`paths_and_const.py` and `AGIPD.geom` is present in `resonet/resonet/sims/geoms/`, but
synthesis is broken: `main.py` computes output image size as `(max_ss+1, max_fs+1) =
(512, 128)` — one module's size — causing all 16 modules to overwrite each other.

AGIPD.geom uses the CrystFEL 3D convention: all 128 ASICs share the same ss/fs coordinate
space and `dim0 = %` selects which module's data slice to read. The fix produces
`(16, 512, 128)` 3D frames that match real AGIPD data layout.

Scope is **simulation only**. No changes to model training, validation, inference, or any
file under `resonet/resonet/utils/`.

---

## AGIPD 1M Layout

- **128 panels** total: 16 modules (`p0`–`p15`) × 8 ASICs (`a0`–`a7`) each
- **Panel naming:** `pXaY` — module X, ASIC Y
- **Per-ASIC size:** 128 fs × 64 ss pixels
- **Per-module size:** 128 fs × 512 ss (8 ASICs stacked in slow-scan)
- **Unassembled frame shape:** `(16, 512, 128)` — module-major 3D array
- **Pixel size:** 200 µm, **Detector distance:** ~151.65 mm

---

## Design

### 1. Module detection and 3D image assembly (`main.py`)

After `parse_geom()` returns `_panel_map` (128 dicts), a new helper
`_group_by_module(panel_map)` detects AGIPD-style 3D layout by scanning panel names for the
`pXaY` pattern:

```python
_MODPANEL_RE = re.compile(r'^p(\d+)a\d+$')

def _group_by_module(panel_map):
    """Returns dict {module_idx: [panel_dicts]} if pXaY naming found, else None."""
    modules = {}
    for pm in panel_map:
        m = _MODPANEL_RE.match(pm['name'])
        if not m:
            return None
        modules.setdefault(int(m.group(1)), []).append(pm)
    return modules if len(modules) > 1 else None
```

If `_group_by_module` returns `None` → existing 2D path unchanged.
If it returns a dict → 3D path:

```python
n_modules = len(modules)                                    # 16
ss_per_mod = max(pm['max_ss'] for pm in modules[0]) + 1   # 512
fs_per_mod = max(pm['max_fs'] for pm in modules[0]) + 1   # 128
frame_shape = (n_modules, ss_per_mod, fs_per_mod)          # (16, 512, 128)
```

Panel placement fills `img[mod_idx, min_ss:max_ss+1, min_fs:max_fs+1]` instead of the
current 2D slice. `CXIWriter` receives `frame_shape=(16, 512, 128)` — no changes to
`cxi_writer.py` are needed since it already accepts arbitrary-shaped tuples.

### 2. AGIPD noise model (`make_sims.py`)

New function `apply_agipd_noise(img, rng)` following the `apply_jungfrau_noise` pattern.
Input: raw photon-count array of any shape. Output: `uint16` array of same shape.

**3-gain zones (threshold-based, per pixel):**

| Zone | Trigger (photons) | ADU/photon | Readout noise σ | Source |
|------|-------------------|-----------|-----------------|--------|
| HG (high gain)  | photons ≤ 65   | 64 | 7 ADU  | 350 e⁻ r.m.s. @ ~9.4 keV beam |
| MG (medium gain)| 65 < photons ≤ 2000 | 8 | 3 ADU | literature approximation |
| LG (low gain)   | photons > 2000  |  1 | 1.5 ADU | literature approximation |

Threshold notes:
- HG→MG transition: 50–80 photons at ~12.4 keV; 65 used as midpoint.
- MG→LG transition: ~2000 photons.
- Bad-pixel outlier thresholds (>3.5 keV noise in MG, >18 keV in LG) are calibration
  quantities, not simulation noise values.

Pipeline (5 steps):
1. Poisson shot noise on raw photon counts
2. Assign gain zone per pixel based on thresholds above
3. Multiply by zone ADU/photon factor
4. Add Gaussian readout noise (σ per zone, drawn independently per pixel)
5. Clip negatives to 0, cast to `uint16`

The function is shape-agnostic — it operates elementwise on the ndarray, so it works on
`(16, 512, 128)` without special 3D handling.

### 3. `simulator.py` — AGIPD dispatch

Add `agipd_mode` boolean attribute. Update `set_noise()` guard to include `agipd_mode`.
`_apply_noise()` dispatches to `apply_agipd_noise()` when `agipd_mode` is set. Follows the
identical pattern as `jungfrau_mode` and `epix10k_mode`.

### 4. `main.py` — CLI flag and auto-enable

Add `--agipd-noise` flag with `sys.argv`-based mutual-exclusion guard (same pattern as
`--jungfrau-noise` and `--epix10k-noise` — raises an error if two noise flags are passed
together).

Auto-enable `--agipd-noise` when `--geom agipd` is detected, with a warning printed if a
conflicting noise flag was also passed explicitly.

### 5. `paths_and_const.py`

Verify the AGIPD preset's `geomfile` points to `resonet/resonet/sims/geoms/AGIPD.geom`
(the 3D-indexed `dim0 = %` file), not the flat `AGIPD-2.geom` in the top-level `geoms/`
directory.

---

## What is NOT changed

- `cxi_writer.py` — already accepts arbitrary `frame_shape` tuples; no changes needed.
- `geom_parser.py` — no changes; module detection works off panel names in `main.py`.
- `resonet/resonet/utils/` — out of scope (inference/training code).
- `loaders.py`, `net.py`, `arches.py`, `eval_model.py` — out of scope.

---

## Data Flow

```
resonet-simulate outdir --geom agipd --nshot 1000 --ngpu 1 --randHits
        │
        ▼
parse_geom(AGIPD.geom)  →  panel_map (128 dicts, all sharing ss/fs 0–511 × 0–127)
        │
        ▼
_group_by_module(panel_map)  →  {0: [8 ASICs], 1: [...], ..., 15: [...]}
        │
        ▼
img = np.zeros((16, 512, 128))
Fill img[mod_idx, min_ss:max_ss+1, min_fs:max_fs+1] per panel
        │
        ▼
apply_agipd_noise(img, rng)  →  (16, 512, 128) uint16
        │
        ▼
CXIWriter.add_frame(img)  →  /entry_1/data_1/data  shape (N, 16, 512, 128)
```

---

## Output CXI Schema

```
/entry_1/data_1/data                                  shape: (n_shots, 16, 512, 128), dtype: uint16
/entry_1/instrument_1/detector_1/description          = "AGIPD 1M"
/entry_1/instrument_1/detector_1/distance             = 0.15165 (m)
/entry_1/instrument_1/detector_1/x_pixel_size         = 0.0002 (m)
/entry_1/instrument_1/detector_1/y_pixel_size         = 0.0002 (m)
/entry_1/instrument_1/source_1/energy                 = 9385 (eV)
/entry_1/labels/is_hit                                shape: (n_shots,), dtype: float32
```

---

## Files to Modify

| File | Change |
|------|--------|
| `resonet/resonet/sims/main.py` | `_group_by_module()`, 3D image init, 3D panel placement, `--agipd-noise` flag + auto-enable |
| `resonet/resonet/sims/make_sims.py` | `apply_agipd_noise()` |
| `resonet/resonet/sims/simulator.py` | `agipd_mode` flag, noise dispatch |
| `resonet/resonet/sims/paths_and_const.py` | Verify AGIPD preset geomfile path |
| `resonet/resonet/tests/` | 3 smoke tests (see below) |

---

## Tests

All tests are simulation-only; no GPU required (`--ngpu 0` for shot tests).

1. **`test_agipd_noise`** — unit test for `apply_agipd_noise()`:
   - All-zero input → near-zero output (only readout noise, clipped to 0)
   - Low-count pixel (≤65 photons) activates HG zone (output ≈ photons × 64)
   - High-count pixel (>2000 photons) activates LG zone (output ≈ photons × 1)
   - Output shape matches input shape; output dtype is `uint16`

2. **`test_agipd_shape`** — 1-shot CXI smoke test (no noise):
   - Runs simulation with `--geom agipd --nshot 1 --ngpu 0`
   - Asserts `/entry_1/data_1/data` shape is `(1, 16, 512, 128)`

3. **`test_agipd_noise_smoke`** — 1-shot with `--agipd-noise`:
   - Asserts pixel max > 0; asserts shape still `(1, 16, 512, 128)`

---

## Verification (manual, post-implementation)

1. Run 1-shot simulation, confirm CXI shape `(1, 16, 512, 128)`.
2. Visual check: `python resonet/resonet/scripts/view_cxi.py <out.cxi> --geom geoms/AGIPD.geom` — confirm assembled image shows a diffraction pattern (hit) or background (non-hit).
3. Noise sanity: pixel max in HG zone ≈ 65 × 64 = 4160 ADU; LG-zone pixels reach higher raw counts but lower ADU due to ×1 factor.
4. Production run: 20 ranks × 1000 shots = 20k-shot dataset; verify peak RSS within bounds.
