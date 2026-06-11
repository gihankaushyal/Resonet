# Eiger4M Unassembled CXI Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--outfmt cxi` path to `resonet-simulate` that writes full-size unassembled Eiger4M images `(N, 5632, 384)` to CXI HDF5 files consumable by Hitfinder's `reborn` geometry-assembly pipeline.

**Architecture:** Parse CrystFEL `.geom` file → build dxtbx multi-panel Detector → simulate with existing nanoBragg engine (multi-panel output) → extract per-panel pixels using panel_map → write to CXI HDF5 per MPI rank. The existing `--outfmt hdf5` (512×512) path is untouched. `merge_h5s.py` is extended to handle `.cxi` files.

**Tech Stack:** dxtbx (`DetectorFactory.from_dict`), h5py, numpy, existing `resonet-simulate` MPI infrastructure, CrystFEL `.geom` format.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `resonet/resonet/sims/geom_parser.py` | CrystFEL .geom → dxtbx Detector + panel_map |
| Create | `resonet/resonet/sims/cxi_writer.py` | Accumulate frames → CXI HDF5 |
| Create | `resonet/tests/test_geom_parser.py` | Unit tests for geom_parser |
| Create | `resonet/tests/test_cxi_writer.py` | Unit tests for cxi_writer |
| Modify | `resonet/resonet/sims/simulator.py` | Add `multi_panel=False` flag to `simulate()` |
| Modify | `resonet/resonet/sims/main.py` | +3 CLI flags, CXI output branch |
| Modify | `resonet/resonet/scripts/merge_h5s.py` | Extend to merge `.cxi` files |

---

## Task 1: `geom_parser.py` — CrystFEL .geom → dxtbx Detector

**Files:**
- Create: `resonet/resonet/sims/geom_parser.py`
- Create: `resonet/tests/test_geom_parser.py`

- [ ] **Step 1.1: Write failing tests**

Create `resonet/tests/test_geom_parser.py`:

```python
import math
import os
import pytest

GEOM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "geoms", "Eigar.geom"
)

def test_panel_count():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    assert len(panel_map) == 64

def test_pixel_size():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, globals_ = parse_geom(GEOM_PATH)
    panel = detector[0]
    px, py = panel.get_pixel_size()
    assert abs(px - 0.075) < 1e-4   # 75 µm = 0.075 mm
    assert abs(py - 0.075) < 1e-4

def test_panel_map_fields():
    from resonet.sims.geom_parser import parse_geom
    _, panel_map, _ = parse_geom(GEOM_PATH)
    required = {'name', 'min_fs', 'max_fs', 'min_ss', 'max_ss',
                'panel_idx', 'n_fast', 'n_slow'}
    for pm in panel_map:
        assert required.issubset(pm.keys())

def test_unassembled_shape():
    from resonet.sims.geom_parser import parse_geom
    _, panel_map, _ = parse_geom(GEOM_PATH)
    n_ss = max(pm['max_ss'] for pm in panel_map) + 1
    n_fs = max(pm['max_fs'] for pm in panel_map) + 1
    assert n_ss == 5632
    assert n_fs == 384

def test_panel_orthogonality():
    from resonet.sims.geom_parser import parse_geom
    detector, panel_map, _ = parse_geom(GEOM_PATH)
    for i, pm in enumerate(panel_map):
        panel = detector[i]
        fast = panel.get_fast_axis()
        slow = panel.get_slow_axis()
        dot = sum(f * s for f, s in zip(fast, slow))
        assert abs(dot) < 0.01, f"Panel {pm['name']} not orthogonal: dot={dot:.4f}"

def test_global_params():
    from resonet.sims.geom_parser import parse_geom
    _, _, globals_ = parse_geom(GEOM_PATH)
    assert 'clen' in globals_
    assert 'res' in globals_
    assert 'photon_energy' in globals_
    assert abs(globals_['clen'] - 0.300) < 1e-6
    assert abs(globals_['res'] - 10000.075) < 0.01
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_geom_parser.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError` or `ImportError` — `geom_parser` does not exist yet.

- [ ] **Step 1.3: Implement `geom_parser.py`**

Create `resonet/resonet/sims/geom_parser.py`:

```python
"""Parse CrystFEL .geom files and convert to dxtbx Detector objects."""
import re
from typing import Any
from dxtbx.model.detector import DetectorFactory


def _parse_axis(s: str) -> tuple:
    """Parse CrystFEL axis string e.g. '-0.999991x +0.004221y' to (x, y, z)."""
    x = y = z = 0.0
    for m in re.finditer(r'([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([xyz])', s):
        v, a = float(m.group(1)), m.group(2)
        if a == 'x':
            x = v
        elif a == 'y':
            y = v
        else:
            z = v
    return x, y, z


def _panel_sort_key(p: dict) -> tuple:
    """Numeric sort key so p10a0 sorts after p9a3, not before p1a0."""
    return tuple(int(n) for n in re.findall(r'\d+', p['name']))


def parse_geom(path: str) -> tuple:
    """Parse a CrystFEL .geom file.

    Returns:
        detector  : dxtbx Detector with one Panel per ASIC block
        panel_map : list of dicts with keys:
                    name, min_ss, max_ss, min_fs, max_fs,
                    panel_idx, n_fast, n_slow
        globals_  : dict with keys clen (m), res (px/m), photon_energy (eV)
    """
    globals_: dict[str, Any] = {}
    panels: dict[str, dict[str, Any]] = {}

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.split(';')[0].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if '/' in key:
                panel_name, _, field = key.partition('/')
                panel_name = panel_name.strip()
                field = field.strip()
                panels.setdefault(panel_name, {})['name'] = panel_name
                if field == 'fs':
                    panels[panel_name]['fs'] = _parse_axis(value)
                elif field == 'ss':
                    panels[panel_name]['ss'] = _parse_axis(value)
                elif field in ('corner_x', 'corner_y',
                               'min_fs', 'max_fs', 'min_ss', 'max_ss'):
                    try:
                        panels[panel_name][field] = float(value)
                    except ValueError:
                        pass
            else:
                if key in ('clen', 'photon_energy', 'res'):
                    try:
                        globals_[key] = float(value)
                    except ValueError:
                        pass  # dynamic field like /LCLS/photon_energy_eV

    required_panel_fields = {'fs', 'ss', 'corner_x', 'corner_y',
                             'min_fs', 'max_fs', 'min_ss', 'max_ss'}
    valid_panels = [
        p for p in panels.values()
        if required_panel_fields.issubset(p.keys())
    ]
    valid_panels.sort(key=_panel_sort_key)

    clen = globals_['clen']
    res = globals_['res']
    pixel_size_mm = 1000.0 / res

    panel_dicts = []
    panel_map = []

    for idx, p in enumerate(valid_panels):
        fast_axis = p['fs']
        slow_axis = p['ss']

        dot = sum(f * s for f, s in zip(fast_axis, slow_axis))
        if abs(dot) > 0.01:
            raise ValueError(
                f"Panel {p['name']}: fast/slow axes not orthogonal (dot={dot:.4f}). "
                "Use Plan B (CBF-based) fallback."
            )

        # CrystFEL origin is in pixels from beam center in the detector plane.
        # dxtbx origin is in mm from the lab origin.
        # CrystFEL convention: +y up; dxtbx (CBF/imgCIF): same +y up — no y flip.
        # z: detector is downstream at +clen along +z beam direction.
        # NOTE: verify sign of z against eiger_1_00001.cbf if patterns look wrong.
        origin_mm = (
            p['corner_x'] * pixel_size_mm,
            p['corner_y'] * pixel_size_mm,
            clen * 1000.0,
        )
        n_fast = int(p['max_fs'] - p['min_fs'] + 1)
        n_slow = int(p['max_ss'] - p['min_ss'] + 1)

        panel_dicts.append({
            'name': p['name'],
            'type': '',
            'fast_axis': fast_axis,
            'slow_axis': slow_axis,
            'origin': origin_mm,
            'pixel_size': (pixel_size_mm, pixel_size_mm),
            'image_size': (n_fast, n_slow),
            'trusted_range': (0.0, 65536.0),
            'thickness': 0.0,
            'material': 'Si',
            'mu': 0.0,
            'gain': 1.0,
            'pedestal': 0.0,
            'identifier': '',
            'mask': [],
            'raw_image_offset': (0, 0),
            'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
        })
        panel_map.append({
            'name': p['name'],
            'min_fs': int(p['min_fs']),
            'max_fs': int(p['max_fs']),
            'min_ss': int(p['min_ss']),
            'max_ss': int(p['max_ss']),
            'panel_idx': idx,
            'n_fast': n_fast,
            'n_slow': n_slow,
        })

    detector = DetectorFactory.from_dict({'panels': panel_dicts})
    return detector, panel_map, globals_
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_geom_parser.py -v
```
Expected: 6 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add resonet/resonet/sims/geom_parser.py resonet/tests/test_geom_parser.py
git commit -m "feat: add CrystFEL .geom parser producing dxtbx Detector + panel_map

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `cxi_writer.py` — Accumulate Frames → CXI HDF5

**Files:**
- Create: `resonet/resonet/sims/cxi_writer.py`
- Create: `resonet/tests/test_cxi_writer.py`

- [ ] **Step 2.1: Write failing tests**

Create `resonet/tests/test_cxi_writer.py`:

```python
import os
import tempfile
import numpy as np
import h5py
import pytest


METADATA = {
    'detector_name': 'EIGER 4M',
    'distance_m': 0.300,
    'pixel_size_m': 7.5e-5,
    'photon_energy_eV': 8750.0,
    'wavelength_m': 1.417e-10,
}


def test_cxi_data_shape():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        for _ in range(3):
            w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            assert f['/entry_1/data_1/data'].shape == (3, 5632, 384)
            assert f['/entry_1/data_1/data'].dtype == np.uint16


def test_cxi_metadata():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            det = f['/entry_1/instrument_1/detector_1']
            assert det['description'][()].decode() == 'EIGER 4M'
            assert abs(float(det['distance'][()]) - 0.300) < 1e-6
            assert abs(float(det['x_pixel_size'][()]) - 7.5e-5) < 1e-9
            src = f['/entry_1/instrument_1/source_1']
            assert abs(float(src['energy'][()]) - 8750.0) < 0.01
            assert abs(float(src['wavelength'][()]) - 1.417e-10) < 1e-14


def test_cxi_labels():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16),
                    labels={'hit': 1.0, 'resolution': 2.5})
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16),
                    labels={'hit': 0.0, 'resolution': 0.0})
        w.close()
        with h5py.File(path, 'r') as f:
            assert f['/entry_1/labels/hit'].shape == (2,)
            assert f['/entry_1/labels/resolution'].shape == (2,)
            assert float(f['/entry_1/labels/hit'][0]) == 1.0


def test_cxi_compression():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16))
        w.close()
        with h5py.File(path, 'r') as f:
            ds = f['/entry_1/data_1/data']
            assert ds.compression == 'gzip'


def test_add_frame_wrong_shape_raises():
    from resonet.sims.cxi_writer import CXIWriter
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'test.cxi')
        w = CXIWriter(path, (5632, 384), METADATA)
        with pytest.raises(AssertionError):
            w.add_frame(np.zeros((512, 512), dtype=np.uint16))
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_cxi_writer.py -v 2>&1 | head -20
```
Expected: `ImportError` — `cxi_writer` not yet created.

- [ ] **Step 2.3: Implement `cxi_writer.py`**

Create `resonet/resonet/sims/cxi_writer.py`:

```python
"""Write unassembled detector images to CXI HDF5 format."""
import numpy as np
import h5py


class CXIWriter:
    """Accumulates per-shot unassembled images and writes to a CXI HDF5 file."""

    def __init__(self, filepath: str, frame_shape: tuple, metadata: dict):
        """
        Args:
            filepath:     output .cxi path
            frame_shape:  (n_ss, n_fs) unassembled image shape
            metadata:     dict with keys:
                          detector_name (str), distance_m (float),
                          pixel_size_m (float), photon_energy_eV (float),
                          wavelength_m (float)
        """
        self._filepath = filepath
        self._frame_shape = frame_shape
        self._metadata = metadata
        self._frames: list = []
        self._labels: list = []

    def add_frame(self, image: np.ndarray, labels: dict = None):
        """Append one unassembled frame. image must have shape == frame_shape."""
        assert image.shape == self._frame_shape, (
            f"Expected shape {self._frame_shape}, got {image.shape}"
        )
        self._frames.append(image.astype(np.uint16))
        self._labels.append(labels or {})

    def close(self):
        """Write all accumulated frames to disk. No-op if no frames added."""
        if not self._frames:
            return
        data = np.stack(self._frames, axis=0)
        meta = self._metadata

        with h5py.File(self._filepath, 'w') as f:
            det = f.require_group('entry_1/instrument_1/detector_1')
            det.create_dataset('description',
                               data=np.bytes_(meta['detector_name']))
            det.create_dataset('distance', data=float(meta['distance_m']))
            det.create_dataset('x_pixel_size',
                               data=float(meta['pixel_size_m']))
            det.create_dataset('y_pixel_size',
                               data=float(meta['pixel_size_m']))

            src = f.require_group('entry_1/instrument_1/source_1')
            src.create_dataset('energy',
                               data=float(meta['photon_energy_eV']))
            src.create_dataset('wavelength',
                               data=float(meta['wavelength_m']))

            f.create_dataset(
                'entry_1/data_1/data',
                data=data,
                compression='gzip',
                compression_opts=4,
                shuffle=True,
            )

            if any(self._labels):
                all_keys = set()
                for d in self._labels:
                    all_keys.update(d.keys())
                lbl = f.require_group('entry_1/labels')
                for key in sorted(all_keys):
                    vals = [d.get(key, float('nan')) for d in self._labels]
                    lbl.create_dataset(key, data=np.array(vals,
                                                           dtype=np.float32))
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_cxi_writer.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add resonet/resonet/sims/cxi_writer.py resonet/tests/test_cxi_writer.py
git commit -m "feat: add CXI HDF5 writer for unassembled detector images

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `simulator.py` — Add `multi_panel` Support

The current `simulate()` reshapes `raw_pixels` using `shot_det[0].get_image_size()` (single-panel assumption). For multi-panel output we need the flat 1D pixel array in panel order.

**Files:**
- Modify: `resonet/resonet/sims/simulator.py` (lines ~197-200, ~293, ~320)

- [ ] **Step 3.1: Add `multi_panel=False` parameter and conditional reshape**

In `simulator.py`, find the `simulate()` signature at line 54 and add `multi_panel=False`:

```python
# BEFORE (line 54):
def simulate(self, rot_mat=None, multi_lattice_chance=0, max_lat=2, mos_min_max=None,
             pdb_name=None, plastic_stol=None, dev=0, mos_dom_override=None, vary_background_scale=False,
             randomize_dist=None, randomize_wavelen=None, randomize_center=False,
             randomize_scale=False, low_bg_chance=0, uniform_reso=False, roi=None,
             old_multi_spread=True, cbf_name=None):

# AFTER:
def simulate(self, rot_mat=None, multi_lattice_chance=0, max_lat=2, mos_min_max=None,
             pdb_name=None, plastic_stol=None, dev=0, mos_dom_override=None, vary_background_scale=False,
             randomize_dist=None, randomize_wavelen=None, randomize_center=False,
             randomize_scale=False, low_bg_chance=0, uniform_reso=False, roi=None,
             old_multi_spread=True, cbf_name=None, multi_panel=False):
```

Then find lines 197-200 and modify:

```python
# BEFORE (lines 197-200):
spots = S.D.raw_pixels.as_numpy_array()
xdim, ydim = shot_det[0].get_image_size()
img_sh = ydim, xdim
spots = spots.reshape(img_sh)

# AFTER:
spots = S.D.raw_pixels.as_numpy_array()
xdim, ydim = shot_det[0].get_image_size()
if multi_panel and len(shot_det) > 1:
    img_sh = (spots.size,)   # keep flat for multi-panel extraction
else:
    img_sh = ydim, xdim
spots = spots.reshape(img_sh)
```

Find line 293 (noise_img reshape) and modify:

```python
# BEFORE (line 293):
noise_img = S.D.raw_pixels.as_numpy_array().reshape(img_sh)

# AFTER:
noise_img = S.D.raw_pixels.as_numpy_array().reshape(img_sh)
# img_sh is (total_pixels,) in multi_panel mode — flat 1D, caller extracts panels
```

Find line 320 (CBF writing resize):

```python
# BEFORE (line 320):
raw_pix.resize(flex.grid((ydim, xdim)))

# AFTER (skip CBF writing for multi-panel; single-panel unchanged):
if not (multi_panel and len(shot_det) > 1):
    raw_pix.resize(flex.grid((ydim, xdim)))
    S.D.raw_pixels = raw_pix
    S.D.to_cbf(cbf_name, toggle_conventions=True)
```

- [ ] **Step 3.2: Run existing tests to confirm no regressions**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/ -v -k "not test_geom_parser and not test_cxi_writer"
```
Expected: all pre-existing tests PASS.

- [ ] **Step 3.3: Commit**

```bash
git add resonet/resonet/sims/simulator.py
git commit -m "feat: add multi_panel flag to Simulator.simulate() for flat pixel output

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: `main.py` — CLI Flags + CXI Output Branch

**Files:**
- Modify: `resonet/resonet/sims/main.py`

- [ ] **Step 4.1: Add three new CLI flags**

Find the argparse block in `main.py`. After the last existing `add_argument` call, add:

```python
parser.add_argument(
    "--outfmt", type=str, default="hdf5", choices=["hdf5", "cxi"],
    help="Output format: hdf5 (default, 512x512) or cxi (full unassembled)")
parser.add_argument(
    "--geomfile", type=str, default=None,
    help="Path to CrystFEL .geom file (required when --outfmt cxi)")
parser.add_argument(
    "--detector-name", dest="detector_name", type=str, default=None,
    help="Detector description string written to CXI metadata "
         "(e.g. 'EIGER 4M'); required when --outfmt cxi")
```

- [ ] **Step 4.2: Validate CXI-required flags early**

Find the section in `main.py` where `args` is first used after `parse_args()`. Add:

```python
if args.outfmt == "cxi":
    if args.geomfile is None:
        raise ValueError("--geomfile is required when --outfmt cxi")
    if args.detector_name is None:
        raise ValueError("--detector-name is required when --outfmt cxi")
    if not os.path.exists(args.geomfile):
        raise FileNotFoundError(f"Geom file not found: {args.geomfile}")
```

- [ ] **Step 4.3: Load CXI geometry when outfmt=cxi**

Find the geometry-loading section in `main.py` (around lines 128-180 where `--geom eiger` is handled). Add a new branch at the **top** of that section so it takes priority:

```python
if args.outfmt == "cxi":
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.cxi_writer import CXIWriter
    _geom_det, _panel_map, _geom_globals = parse_geom(args.geomfile)
    DET = _geom_det
    # Derive unassembled image dimensions from panel_map
    _n_ss = max(pm['max_ss'] for pm in _panel_map) + 1
    _n_fs = max(pm['max_fs'] for pm in _panel_map) + 1
    # Pre-compute per-panel pixel offsets (panels ordered by panel_idx)
    _pixel_offsets = []
    _offset = 0
    for pm in _panel_map:
        _pixel_offsets.append(_offset)
        _offset += pm['n_fast'] * pm['n_slow']
    _total_pixels = _offset
    # Wavelength from photon energy: λ(m) = hc / E = 1239.84193e-9 / E_eV
    _wavelength_m = 1239.84193e-9 / _geom_globals['photon_energy']
    _cxi_meta = {
        'detector_name': args.detector_name,
        'distance_m': _geom_globals['clen'],
        'pixel_size_m': 1.0 / _geom_globals['res'],
        'photon_energy_eV': _geom_globals['photon_energy'],
        'wavelength_m': _wavelength_m,
    }
else:
    # ... existing geometry loading block unchanged ...
```

- [ ] **Step 4.4: Initialize CXI writer (replace HDF5 output setup)**

Find the output-file initialization section (where HDF5 datasets are created for the `hdf5` path, around lines 221-240). Add a parallel CXI branch:

```python
if args.outfmt == "cxi":
    _cxi_outpath = os.path.join(args.outdir,
                                f"compressed_{rank}.cxi")
    _cxi_writer = CXIWriter(_cxi_outpath, (_n_ss, _n_fs), _cxi_meta)
else:
    # ... existing HDF5 setup unchanged ...
```

- [ ] **Step 4.5: Per-shot CXI output**

Find the per-shot processing loop in `main.py` (around lines 449-524 where images are downsampled and written to HDF5). Add a parallel CXI branch that fires instead of the downsample path when `outfmt == "cxi"`:

```python
if args.outfmt == "cxi":
    # simulate() returns (param_dict, all_spots_scaled, noise_imgs, shot_det, shot_beam)
    param_dict, all_spots_scaled, noise_imgs, shot_det, shot_beam = SIM.simulate(
        ...,                 # all existing simulate() kwargs unchanged
        multi_panel=True,    # add this keyword
    )
    # noise_imgs[0] is the flat 1D array of all panel pixels
    flat_img = noise_imgs[0]
    # Build unassembled image
    unassembled = np.zeros((_n_ss, _n_fs), dtype=np.float32)
    for pm, pix_off in zip(_panel_map, _pixel_offsets):
        n_px = pm['n_fast'] * pm['n_slow']
        panel_data = flat_img[pix_off:pix_off + n_px].reshape(
            pm['n_slow'], pm['n_fast']
        )
        unassembled[
            pm['min_ss']:pm['max_ss'] + 1,
            pm['min_fs']:pm['max_fs'] + 1
        ] = panel_data
    # sqrt compression + uint16 clip (same as hdf5 path)
    unassembled = np.sqrt(np.maximum(unassembled, 0.0))
    unassembled = np.clip(unassembled, 0, np.sqrt(65535)).astype(np.uint16)
    # Collect labels (same keys as current HDF5 labels)
    shot_labels = {
        'hit': float(param_dict.get('is_hit', 0)),
        'detector_distance': float(param_dict['detector_distance']),
        'wavelength': float(param_dict['wavelength']),
    }
    _cxi_writer.add_frame(unassembled, labels=shot_labels)
else:
    # ... existing HDF5 downsampling/writing path unchanged ...
```

> **Note on labels:** Inspect the existing `params` dict keys and HDF5 label dataset names used in the `hdf5` path to add the matching keys to `shot_labels`. The exact label keys vary by simulation flags.

- [ ] **Step 4.6: Close CXI writer at end of rank**

Find where the HDF5 file is closed/flushed at the end of the simulation loop. Add:

```python
if args.outfmt == "cxi":
    _cxi_writer.close()
    if rank == 0:
        print(f"CXI output written: {_cxi_outpath}", flush=True)
else:
    # ... existing HDF5 close logic ...
```

- [ ] **Step 4.7: Smoke-test CLI flags with --help**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
resonet-simulate --help | grep -E "outfmt|geomfile|detector.name"
```
Expected: all three new flags appear in the help output.

- [ ] **Step 4.8: Commit**

```bash
git add resonet/resonet/sims/main.py
git commit -m "feat: add --outfmt cxi path to resonet-simulate

Adds --geomfile, --outfmt, --detector-name flags. CXI branch uses
geom_parser + CXIWriter to write unassembled (N, 5632, 384) frames.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Extend `merge_h5s.py` for CXI Files

The existing merge script uses HDF5 VirtualDataset to combine per-rank `.h5` files. For `.cxi` files the image path is `/entry_1/data_1/data` instead of `images`.

**Files:**
- Modify: `resonet/resonet/scripts/merge_h5s.py`

- [ ] **Step 5.1: Add `--cxi` flag and CXI merge logic**

Replace the contents of `merge_h5s.py` with:

```python
import h5py
import glob
import os
from argparse import ArgumentParser


def _get_cxi_shapes(fnames):
    dummie = h5py.File(fnames[0], 'r')
    shapes = {}
    shapes['data'] = dummie['/entry_1/data_1/data'].shape[1:]
    for key in ['labels']:
        grp_path = f'entry_1/{key}'
        if grp_path in dummie:
            for k in dummie[grp_path]:
                shapes[f'labels/{k}'] = dummie[f'{grp_path}/{k}'].shape[1:]
    return shapes, dummie


def _merge_cxi(fnames, outname, prefix):
    shapes, dummie_h = _get_cxi_shapes(fnames)
    imgs_per_fname = [
        h5py.File(f, 'r')['/entry_1/data_1/data'].shape[0] for f in fnames
    ]
    total_imgs = sum(imgs_per_fname)

    Layouts = {}
    for key, shape in shapes.items():
        hdf_key = f'entry_1/data_1/data' if key == 'data' else f'entry_1/{key}'
        dtype = dummie_h[hdf_key].dtype
        Layouts[key] = h5py.VirtualLayout(
            shape=(total_imgs,) + shape, dtype=dtype
        )

    start = 0
    for i_f, f in enumerate(fnames):
        print(f"virtualizing file {i_f+1} / {len(fnames)}")
        nimg = imgs_per_fname[i_f]
        for key in Layouts:
            hdf_key = ('entry_1/data_1/data' if key == 'data'
                       else f'entry_1/{key}')
            vsource = h5py.VirtualSource(
                f, hdf_key, shape=(nimg,) + shapes[key]
            )
            Layouts[key][start:start + nimg] = vsource
        start += nimg

    print(f"Saving to {outname}, total shots={total_imgs}")
    with h5py.File(outname, 'w') as H:
        # Copy metadata from first file
        with h5py.File(fnames[0], 'r') as src:
            for grp in ('entry_1/instrument_1',):
                if grp in src:
                    src.copy(grp, H.require_group('entry_1'), name='instrument_1')
        for key, layout in Layouts.items():
            hdf_key = ('entry_1/data_1/data' if key == 'data'
                       else f'entry_1/{key}')
            H.create_virtual_dataset(hdf_key, layout)
    print("Done!")


def _merge_h5(fnames, outname, prefix, more_keys):
    """Original HDF5 merge logic (unchanged)."""
    dummie_h = h5py.File(fnames[0], 'r')
    shapes = {}
    for key in ['images_mean', 'images', 'labels', 'full_maximg', 'geom'] + more_keys:
        try:
            shapes[key] = dummie_h[key].shape[1:]
        except KeyError:
            pass

    imgs_per_fname = [h5py.File(f, 'r')['labels'].shape[0] for f in fnames]
    total_imgs = sum(imgs_per_fname)

    Layouts = {}
    for key, shape in shapes.items():
        Layouts[key] = h5py.VirtualLayout(
            shape=(total_imgs,) + shape, dtype=dummie_h[key].dtype
        )

    start = 0
    for i_f, f in enumerate(fnames):
        print(f"virtualizing file {i_f+1} / {len(fnames)}")
        nimg = imgs_per_fname[i_f]
        for key in Layouts:
            vsource = h5py.VirtualSource(f, key, shape=(nimg,) + shapes[key])
            Layouts[key][start:start + nimg] = vsource
        start += nimg

    print(f"Saving to {outname}! Total shots={total_imgs}")
    with h5py.File(outname, 'w') as H:
        for key in Layouts:
            vd = H.create_virtual_dataset(key, Layouts[key])
            for attr in ['names', 'pdbmap']:
                if attr in dummie_h[key].attrs:
                    vd.attrs[attr] = dummie_h[key].attrs[attr]
    print("Done!")


def main():
    parser = ArgumentParser()
    parser.add_argument('dirnames', nargs='+',
                        help='output folders from runme.py')
    parser.add_argument('outname', help='name of the master file')
    parser.add_argument('--moreKeys', nargs='+', default=[],
                        help='additional datasets to virtualize (h5 mode)')
    parser.add_argument('--prefix', default='compressed',
                        help='merge files starting with this prefix')
    parser.add_argument('--cxi', action='store_true',
                        help='merge .cxi files instead of .h5 files')
    args = parser.parse_args()

    ext = 'cxi' if args.cxi else 'h5'
    fnames = []
    for dirname in args.dirnames:
        fnames += glob.glob(os.path.join(dirname, f'{args.prefix}*.{ext}'))
    fnames = [os.path.abspath(f) for f in sorted(fnames)]
    print(f"Combining {len(fnames)} files")

    if args.cxi:
        _merge_cxi(fnames, args.outname, args.prefix)
    else:
        _merge_h5(fnames, args.outname, args.prefix, args.moreKeys)


if __name__ == '__main__':
    main()
```

- [ ] **Step 5.2: Test merge script on two synthetic CXI files**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
python - <<'EOF'
import numpy as np, h5py, os, tempfile

# Create two synthetic CXI files
d = tempfile.mkdtemp()
meta = {'detector_name': 'EIGER 4M', 'distance_m': 0.3,
        'pixel_size_m': 7.5e-5, 'photon_energy_eV': 8750., 'wavelength_m': 1.417e-10}
from resonet.sims.cxi_writer import CXIWriter
for i in range(2):
    w = CXIWriter(f'{d}/compressed_{i}.cxi', (5632, 384), meta)
    for _ in range(3):
        w.add_frame(np.zeros((5632, 384), dtype=np.uint16),
                    labels={'hit': float(i)})
    w.close()

# Merge
import subprocess
result = subprocess.run(
    ['python', 'resonet/scripts/merge_h5s.py', d, f'{d}/merged.cxi', '--cxi'],
    cwd='/data/bioxfel/user/gihan/Resonet/resonet',
    capture_output=True, text=True
)
print(result.stdout); print(result.stderr)

with h5py.File(f'{d}/merged.cxi', 'r') as f:
    print('Merged shape:', f['/entry_1/data_1/data'].shape)  # expect (6, 5632, 384)
EOF
```
Expected output: `Merged shape: (6, 5632, 384)`.

- [ ] **Step 5.3: Commit**

```bash
git add resonet/resonet/scripts/merge_h5s.py
git commit -m "feat: extend merge_h5s.py to support --cxi flag for CXI file merging

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: End-to-End Integration Test

This task runs a real (CPU-mode, 4-shot) simulation end-to-end to confirm the full pipeline.

**Files:** none (test only)

- [ ] **Step 6.1: Run 4-shot CPU simulation with --outfmt cxi**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
export LD_LIBRARY_PATH=/data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
resonet-simulate /tmp/cxi_e2e_test \
  --nshot 4 \
  --outfmt cxi \
  --geomfile /data/bioxfel/user/gihan/Resonet/geoms/Eigar.geom \
  --detector-name "EIGER 4M" \
  --ngpu 0 --cpuMode --seed 42 --randHit
```
Expected: completes without error, writes `/tmp/cxi_e2e_test/compressed_0.cxi`.

- [ ] **Step 6.2: Verify CXI output**

```bash
python - <<'EOF'
import h5py
f = h5py.File('/tmp/cxi_e2e_test/compressed_0.cxi', 'r')
print('data shape:', f['/entry_1/data_1/data'].shape)
# Expect: (4, 5632, 384)
print('dtype:     ', f['/entry_1/data_1/data'].dtype)
# Expect: uint16
print('detector:  ', f['/entry_1/instrument_1/detector_1/description'][()])
# Expect: b'EIGER 4M'
print('labels:    ', list(f['entry_1/labels'].keys()))
# Expect: at least ['hit', 'detector_distance', 'wavelength']
EOF
```

- [ ] **Step 6.3: Verify unassembled image is non-trivial**

```bash
python - <<'EOF'
import h5py, numpy as np
f = h5py.File('/tmp/cxi_e2e_test/compressed_0.cxi', 'r')
imgs = f['/entry_1/data_1/data'][:]
print('max pixel value:', imgs.max())         # expect > 0
print('non-zero pixels:', np.count_nonzero(imgs))  # expect > 0
# Check that background ring is visible (mean > 0)
print('mean pixel value:', imgs.mean())       # expect > 0
EOF
```

- [ ] **Step 6.4: Verify merge works on real CXI output**

```bash
source /data/bioxfel/user/gihan/Resonet/load_resonet.sh
cd /data/bioxfel/user/gihan/Resonet/resonet
python resonet/scripts/merge_h5s.py /tmp/cxi_e2e_test \
  /tmp/cxi_e2e_test/merged.cxi --cxi
python -c "
import h5py
f = h5py.File('/tmp/cxi_e2e_test/merged.cxi', 'r')
print('merged shape:', f['/entry_1/data_1/data'].shape)
# Expect: (4, 5632, 384) for single rank
"
```

- [ ] **Step 6.5: Final commit**

```bash
git add -p   # stage any remaining uncommitted changes from integration testing
git commit -m "test: end-to-end integration of CXI output pipeline

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Notes for Implementer

**Coordinate system verification (Task 1):** If simulated diffraction patterns look geometrically wrong (rings off-center, reflections in wrong positions), try negating the `z` component of `origin_mm` in `geom_parser.py`. Compare against `eiger_1_00001.cbf` by loading it with dxtbx and printing `detector[0].get_origin()` for the first panel.

**Label keys (Task 4, Step 4.5):** The exact keys available in `param_dict` depend on simulation flags. Run a quick smoke test with the existing `hdf5` path and inspect `param_dict` keys printed with `print(param_dict.keys())` to get the full list before writing `shot_labels`.

**Multi-panel GPU path (future optimization):** The current plan simulates with `multi_panel=True` using the CPU path. If GPU mode is used with the CXI path, `sim_background()` in `simulator.py` also needs to be updated to handle multi-panel output (it similarly reshapes using `shot_det[0].get_image_size()`). This is a separate task when GPU performance is needed for CXI output.

**Extending to other detectors (AGIPD, Epix10k, Jungfrau):** Once Task 6 passes for Eiger, the same pipeline works for any detector with a `.geom` file. Change `--geomfile` to point to the appropriate file and update `--detector-name` to match what Hitfinder's `reborn` expects for that detector.
