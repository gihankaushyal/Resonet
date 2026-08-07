# AGIPD 1M Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `resonet-simulate` to produce correct `(N, 16, 512, 128)` 3D CXI frames for AGIPD 1M with a 3-gain-zone noise model.

**Architecture:** `_group_by_module()` in `main.py` detects AGIPD-style `pXaY` panel naming and branches into a 3D assembly path that fills `img[mod_idx, ss, fs]`. `CXIWriter` already accepts arbitrary `frame_shape` tuples — no changes needed. AGIPD noise lives in `make_sims.py` and is dispatched via `agipd_mode` in `simulator.py`, auto-enabled when `--geom agipd` is passed.

**Tech stack:** NumPy, h5py, CrystFEL .geom parser (`geom_parser.py`), `CXIWriter`, pytest

**Spec:** `docs/superpowers/specs/2026-06-22-agipd-synthesis-design.md`

---

## File Map

| File | Change |
|------|--------|
| `resonet/resonet/sims/make_sims.py` | Add `apply_agipd_noise()` |
| `resonet/resonet/sims/simulator.py` | Add `agipd_mode` attrs + noise dispatch branch |
| `resonet/resonet/sims/main.py` | Add `_group_by_module()`, 3D `_frame_shape`, 3D assembly loop, `--agipd-gain-thresh`/`--agipd-noise-sigma` CLI flags + auto-enable |
| `resonet/resonet/tests/test_simulator_funcs.py` | Add `apply_agipd_noise` unit tests |
| `resonet/resonet/tests/test_agipd_shape.py` | New file: shape smoke test + noise smoke test |

**No changes to:** `cxi_writer.py`, `geom_parser.py`, `paths_and_const.py` (AGIPD preset already correct), anything under `resonet/resonet/utils/`.

---

## Environment Setup

All commands must be run with the resonet environment loaded:

```bash
source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
```

Run tests with:

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_simulator_funcs.py -v
```

**Important:** After editing any source file, copy it to site-packages:

```bash
cp resonet/resonet/sims/make_sims.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
cp resonet/resonet/sims/simulator.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
cp resonet/resonet/sims/main.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
```

---

## Task 1: `apply_agipd_noise()` — unit tests + implementation

**Files:**
- Modify: `resonet/resonet/sims/make_sims.py` (after `apply_jungfrau_noise`, ~line 321)
- Modify: `resonet/resonet/tests/test_simulator_funcs.py` (append at end)

### Background

`apply_agipd_noise` converts raw photon counts to ADU using three gain zones:

| Zone | Trigger (photons) | ADU/photon | Readout σ (ADU) |
|------|-------------------|-----------|-----------------|
| HG   | photons ≤ 65      | 64        | 7.0             |
| MG   | 65 < photons ≤ 2000 | 8       | 3.0             |
| LG   | photons > 2000    | 1         | 1.5             |

Unlike `apply_epix_noise`/`apply_jungfrau_noise` (which stay in photon-equivalent units),
AGIPD outputs **ADU values** — the gain multiplication happens inside the function. Returns
`float32`. The `np.clip(..., 0, 65535).astype(np.uint16)` conversion happens in `main.py`.

- [ ] **Step 1: Write the failing tests**

Append to `resonet/resonet/tests/test_simulator_funcs.py`:

```python
# ---------------------------------------------------------------------------
# apply_agipd_noise
# ---------------------------------------------------------------------------

def test_apply_agipd_noise_output_shape_and_dtype():
    """Output shape matches input; dtype is float32."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.zeros((16, 512, 128), dtype=np.float32)
    out = apply_agipd_noise(img, rng=np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_apply_agipd_noise_nonnegative():
    """All output pixels are >= 0."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.zeros((200, 200), dtype=np.float32)
    out = apply_agipd_noise(img, rng=np.random.default_rng(1))
    assert np.all(out >= 0)


def test_apply_agipd_noise_hg_zone_adu_scale():
    """HG pixels (≤ t1) are multiplied by 64 ADU/photon."""
    from resonet.sims.make_sims import apply_agipd_noise
    # 30 photons, all HG; with zero sigma, output ≈ 30*64 = 1920 ADU (Poisson noise)
    img = np.full((5000,), 30.0, dtype=np.float32)
    out = apply_agipd_noise(img, t1=65, t2=2000,
                            adu_hg=64, adu_mg=8, adu_lg=1,
                            sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                            rng=np.random.default_rng(0))
    np.testing.assert_allclose(np.mean(out), 30.0 * 64, rtol=0.05)


def test_apply_agipd_noise_lg_zone_adu_scale():
    """LG pixels (> t2) are multiplied by 1 ADU/photon."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.full((5000,), 3000.0, dtype=np.float32)
    out = apply_agipd_noise(img, t1=65, t2=2000,
                            adu_hg=64, adu_mg=8, adu_lg=1,
                            sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                            rng=np.random.default_rng(0))
    np.testing.assert_allclose(np.mean(out), 3000.0 * 1, rtol=0.05)


def test_apply_agipd_noise_hg_receives_sigma_hg():
    """HG pixels receive sigma_hg readout noise; larger sigma → more spread."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.full((2000,), 10.0, dtype=np.float32)  # all HG (≤ 65)
    out_large = apply_agipd_noise(img, t1=65, t2=2000,
                                  sigma_hg=100.0, sigma_mg=0.0, sigma_lg=0.0,
                                  rng=np.random.default_rng(0))
    out_zero = apply_agipd_noise(img, t1=65, t2=2000,
                                 sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                 rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 10.0


def test_apply_agipd_noise_mg_receives_sigma_mg():
    """MG pixels receive sigma_mg readout noise."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.full((2000,), 500.0, dtype=np.float32)  # all MG (65 < 500 ≤ 2000)
    out_large = apply_agipd_noise(img, t1=65, t2=2000,
                                  sigma_hg=0.0, sigma_mg=100.0, sigma_lg=0.0,
                                  rng=np.random.default_rng(0))
    out_zero = apply_agipd_noise(img, t1=65, t2=2000,
                                 sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                 rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 10.0


def test_apply_agipd_noise_lg_receives_sigma_lg():
    """LG pixels receive sigma_lg readout noise."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.full((2000,), 5000.0, dtype=np.float32)  # all LG (> 2000)
    out_large = apply_agipd_noise(img, t1=65, t2=2000,
                                  sigma_hg=0.0, sigma_mg=0.0, sigma_lg=100.0,
                                  rng=np.random.default_rng(0))
    out_zero = apply_agipd_noise(img, t1=65, t2=2000,
                                 sigma_hg=0.0, sigma_mg=0.0, sigma_lg=0.0,
                                 rng=np.random.default_rng(0))
    assert np.std(out_large) > np.std(out_zero) + 5.0


def test_apply_agipd_noise_default_rng():
    """rng=None constructs an internal RNG; runs without error."""
    from resonet.sims.make_sims import apply_agipd_noise
    out = apply_agipd_noise(np.zeros((20,), dtype=np.float32))
    assert out.shape == (20,)
    assert np.all(out >= 0)


def test_apply_agipd_noise_t1_ge_t2_raises():
    """t1 >= t2 raises ValueError."""
    from resonet.sims.make_sims import apply_agipd_noise
    with pytest.raises(ValueError, match="t1"):
        apply_agipd_noise(np.zeros((10,), dtype=np.float32), t1=2000, t2=65)


def test_apply_agipd_noise_negative_sigma_raises():
    """Negative sigma raises ValueError."""
    from resonet.sims.make_sims import apply_agipd_noise
    with pytest.raises(ValueError, match="sigma"):
        apply_agipd_noise(np.zeros((10,), dtype=np.float32), sigma_hg=-1.0)


def test_apply_agipd_noise_deterministic_with_rng():
    """Same RNG seed produces identical output."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = (np.random.default_rng(7).random((100, 100)) * 100).astype(np.float32)
    out1 = apply_agipd_noise(img, rng=np.random.default_rng(42))
    out2 = apply_agipd_noise(img, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out1, out2)
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_simulator_funcs.py::test_apply_agipd_noise_output_shape_and_dtype -v
```

Expected: `FAILED` with `ImportError: cannot import name 'apply_agipd_noise'`

- [ ] **Step 3: Implement `apply_agipd_noise()` in `make_sims.py`**

Insert after `apply_jungfrau_noise` (after the `main()` function definition at ~line 323):

```python
def apply_agipd_noise(img, t1=65, t2=2000,
                      adu_hg=64, adu_mg=8, adu_lg=1,
                      sigma_hg=7.0, sigma_mg=3.0, sigma_lg=1.5,
                      rng=None):
    """Per-pixel auto-ranging noise model for AGIPD 1M detector.

    Physical order: Poisson shot noise → gain-zone classification →
    ADU conversion (multiply by zone gain) → Gaussian readout noise →
    floor clip to 0.

    Unlike apply_epix_noise/apply_jungfrau_noise, output is in ADU (not
    photon-equivalent) because AGIPD has large gain differences between zones.

    Gain zones (threshold-based, per pixel):
      HG (high gain):   photons ≤ t1   → adu_hg ADU/photon, sigma_hg ADU readout
      MG (medium gain): t1 < photons ≤ t2 → adu_mg ADU/photon, sigma_mg ADU readout
      LG (low gain):    photons > t2   → adu_lg ADU/photon,  sigma_lg ADU readout

    Defaults from AGIPD 1M at ~9.4 keV:
      HG→MG threshold: 65 photons (midpoint of 50–80 ph range at 12.4 keV)
      MG→LG threshold: 2000 photons
      HG gain: 64 ADU/photon, σ_read ≈ 7 ADU (from 350 e⁻ r.m.s. noise floor)
      MG gain: 8 ADU/photon,  σ_read ≈ 3 ADU (literature approximation)
      LG gain: 1 ADU/photon,  σ_read ≈ 1.5 ADU (literature approximation)

    :param img: noiseless photon-count array (numpy float32, any shape)
    :param t1: HG→MG switch threshold in photon counts (default 65)
    :param t2: MG→LG switch threshold in photon counts (default 2000)
    :param adu_hg: ADU per photon for HG zone (default 64)
    :param adu_mg: ADU per photon for MG zone (default 8)
    :param adu_lg: ADU per photon for LG zone (default 1)
    :param sigma_hg: readout noise RMS in ADU for HG zone (default 7.0)
    :param sigma_mg: readout noise RMS in ADU for MG zone (default 3.0)
    :param sigma_lg: readout noise RMS in ADU for LG zone (default 1.5)
    :param rng: numpy.random.Generator instance (created internally if None)
    :return: noised image as float32 array in ADU, same shape as img, clipped to [0, ∞)
    """
    if t1 >= t2:
        raise ValueError(f"apply_agipd_noise: t1 ({t1}) must be < t2 ({t2}).")
    if any(s < 0 for s in (sigma_hg, sigma_mg, sigma_lg)):
        raise ValueError(
            f"apply_agipd_noise: all sigma values must be non-negative; "
            f"got hg={sigma_hg}, mg={sigma_mg}, lg={sigma_lg}."
        )
    if rng is None:
        rng = np.random.default_rng()
    out = rng.poisson(np.maximum(img, 0)).astype(np.float32)
    hg = out <= t1
    mg = (out > t1) & (out <= t2)
    lg = out > t2
    # Convert to ADU per zone
    out[hg] *= adu_hg
    out[mg] *= adu_mg
    out[lg] *= adu_lg
    # Add Gaussian readout noise in ADU
    for mask, sigma in [(hg, sigma_hg), (mg, sigma_mg), (lg, sigma_lg)]:
        n = int(np.sum(mask))
        if n:
            out[mask] += rng.normal(0, sigma, size=n).astype(np.float32)
    out = np.maximum(out, 0)
    return out
```

- [ ] **Step 4: Run all AGIPD noise tests — expect PASS**

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_simulator_funcs.py -k agipd -v
```

Expected: all 11 `test_apply_agipd_noise_*` tests PASS.

- [ ] **Step 5: Copy to site-packages**

```bash
cp /data/bioxfel/user/gihan/Resonet/resonet/resonet/sims/make_sims.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
```

- [ ] **Step 6: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/sims/make_sims.py \
        resonet/resonet/tests/test_simulator_funcs.py
git commit -m "feat: apply_agipd_noise() — 3-gain-zone ADU noise model for AGIPD 1M"
```

---

## Task 2: `_group_by_module()` — module detection helper

**Files:**
- Modify: `resonet/resonet/sims/main.py` (add near top of file, after imports)

`_group_by_module` is a pure function — no side effects, testable in isolation.

- [ ] **Step 1: Add import and function to `main.py`**

At the top of `main.py`, add `import re` if not already present (check first with `grep -n "^import re" resonet/resonet/sims/main.py`).

Then add `_group_by_module` as a module-level function, before `def args(...)`. Place it right after the imports block:

```python
import re

_MODPANEL_RE = re.compile(r'^p(\d+)a\d+$')


def _group_by_module(panel_map):
    """Detect AGIPD-style module grouping from pXaY panel names.

    Returns dict {module_idx: [panel_dicts]} if all panels match pXaY naming
    and there are at least 2 distinct modules. Returns None otherwise (2D path).
    """
    modules = {}
    for pm in panel_map:
        m = _MODPANEL_RE.match(pm['name'])
        if not m:
            return None
        modules.setdefault(int(m.group(1)), []).append(pm)
    return modules if len(modules) > 1 else None
```

- [ ] **Step 2: Verify manually**

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -c "
from resonet.sims.main import _group_by_module
# Simulate panel_map with pXaY names (AGIPD-style)
fake_map = [{'name': f'p{m}a{a}', 'min_ss': a*64, 'max_ss': a*64+63,
             'min_fs': 0, 'max_fs': 127, 'n_fast': 128, 'n_slow': 64}
            for m in range(16) for a in range(8)]
groups = _group_by_module(fake_map)
print('n_modules:', len(groups))
print('panels per module:', len(groups[0]))
# Non-AGIPD panel names should return None
flat_map = [{'name': 'p0'}, {'name': 'p1'}]
print('non-pXaY returns None:', _group_by_module(flat_map) is None)
"
```

Expected output:
```
n_modules: 16
panels per module: 8
non-pXaY returns None: True
```

- [ ] **Step 3: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/sims/main.py
git commit -m "feat: _group_by_module() — pXaY panel-name detection for 3D AGIPD layout"
```

---

## Task 3: 3D frame shape + panel assembly in the CXI path

**Files:**
- Modify: `resonet/resonet/sims/main.py`
  - Geom-parsing block: ~lines 193–214 (where `_n_ss`, `_n_fs`, `xdim`, `ydim`, `mask` are set)
  - Shot loop: ~lines 449–460 (where `unassembled` is built and `_cxi_writer.add_frame` is called)
  - CXIWriter init: line 367 (where `CXIWriter(outname, (_n_ss, _n_fs), _cxi_meta)` is called)

### Change 1: geom-parsing block (lines 193–197)

Current code:
```python
_geom_det, _panel_map, _geom_globals = parse_geom(args.geomfile)
DET = _geom_det
_n_ss = max(pm['max_ss'] for pm in _panel_map) + 1
_n_fs = max(pm['max_fs'] for pm in _panel_map) + 1
xdim, ydim = _n_fs, _n_ss
mask = np.ones((_n_ss, _n_fs), bool)
```

- [ ] **Step 1: Replace with 3D-aware geom block**

```python
_geom_det, _panel_map, _geom_globals = parse_geom(args.geomfile)
DET = _geom_det
_module_groups = _group_by_module(_panel_map)
if _module_groups is not None:
    _n_modules = len(_module_groups)
    _ss_per_mod = max(pm['max_ss'] for pm in _module_groups[0]) + 1
    _fs_per_mod = max(pm['max_fs'] for pm in _module_groups[0]) + 1
    _frame_shape = (_n_modules, _ss_per_mod, _fs_per_mod)
    xdim, ydim = _fs_per_mod, _ss_per_mod
    mask = np.ones((_ss_per_mod, _fs_per_mod), bool)
    # Build name→pixel_offset map for fast lookup during assembly
    _panel_pix_offset = {}
    _offset = 0
    for pm in _panel_map:
        _panel_pix_offset[pm['name']] = _offset
        _offset += pm['n_fast'] * pm['n_slow']
else:
    _n_ss = max(pm['max_ss'] for pm in _panel_map) + 1
    _n_fs = max(pm['max_fs'] for pm in _panel_map) + 1
    _frame_shape = (_n_ss, _n_fs)
    xdim, ydim = _n_fs, _n_ss
    mask = np.ones((_n_ss, _n_fs), bool)
```

Note: the existing `_pixel_offsets` list computation (lines 200–204) is still needed for the 2D path. Ensure it stays for the `else` branch (it's referenced at line 450). The 3D path uses `_panel_pix_offset` dict instead.

### Change 2: CXIWriter instantiation (line 367)

Current code:
```python
_cxi_writer = CXIWriter(outname, (_n_ss, _n_fs), _cxi_meta)
```

- [ ] **Step 2: Replace with `_frame_shape`**

```python
_cxi_writer = CXIWriter(outname, _frame_shape, _cxi_meta)
```

### Change 3: Shot assembly loop (lines 449–460)

Current code:
```python
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
unassembled = np.clip(unassembled, 0, 65535).astype(np.uint16)
_cxi_writer.add_frame(unassembled, labels=shot_labels)
```

- [ ] **Step 3: Replace with 3D-aware assembly**

```python
unassembled = np.zeros(_frame_shape, dtype=np.float32)
if _module_groups is not None:
    for mod_idx, panels in sorted(_module_groups.items()):
        for pm in panels:
            pix_off = _panel_pix_offset[pm['name']]
            n_px = pm['n_fast'] * pm['n_slow']
            panel_data = flat_img[pix_off:pix_off + n_px].reshape(
                pm['n_slow'], pm['n_fast']
            )
            unassembled[
                mod_idx,
                pm['min_ss']:pm['max_ss'] + 1,
                pm['min_fs']:pm['max_fs'] + 1,
            ] = panel_data
else:
    for pm, pix_off in zip(_panel_map, _pixel_offsets):
        n_px = pm['n_fast'] * pm['n_slow']
        panel_data = flat_img[pix_off:pix_off + n_px].reshape(
            pm['n_slow'], pm['n_fast']
        )
        unassembled[
            pm['min_ss']:pm['max_ss'] + 1,
            pm['min_fs']:pm['max_fs'] + 1,
        ] = panel_data
unassembled = np.clip(unassembled, 0, 65535).astype(np.uint16)
_cxi_writer.add_frame(unassembled, labels=shot_labels)
```

- [ ] **Step 4: Copy to site-packages**

```bash
cp /data/bioxfel/user/gihan/Resonet/resonet/resonet/sims/main.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
```

- [ ] **Step 5: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/sims/main.py
git commit -m "feat: 3D frame assembly for AGIPD — _frame_shape + pXaY module grouping in CXI path"
```

---

## Task 4: `agipd_mode` in `simulator.py`

**Files:**
- Modify: `resonet/resonet/sims/simulator.py`
  - `__init__`: add `agipd_mode` attributes (~line 76, after `jungfrau_mode` block)
  - `set_noise` guard (~line 415)
  - noise dispatch block (~line 435–465, after `jungfrau_mode` branch)

- [ ] **Step 1: Add `agipd_mode` attributes in `__init__`**

After the `self._jungfrau_rng` line (~line 80), add:

```python
        self.agipd_mode = False             # enable AGIPD per-pixel gain-switching noise
        self.agipd_gain_thresh = (65, 2000) # HG→MG and MG→LG thresholds (photon counts)
        self.agipd_noise_sigma = (7.0, 3.0, 1.5)  # readout noise RMS per zone (ADU)
        self._agipd_rng = np.random.default_rng()  # persistent RNG; seed via HS._agipd_rng = np.random.default_rng(seed)
```

- [ ] **Step 2: Update `set_noise` guard (~line 415)**

Current:
```python
        if not self.epix_mode and not self.jungfrau_mode:
            make_sims.set_noise(S.D)
```

Replace with:
```python
        if not self.epix_mode and not self.jungfrau_mode and not self.agipd_mode:
            make_sims.set_noise(S.D)
```

- [ ] **Step 3: Add `agipd_mode` dispatch branch (~line 462, after `jungfrau_mode` branch)**

Current structure:
```python
            if self.epix_mode:
                ...
            elif self.jungfrau_mode:
                ...
            else:
                S.D.raw_pixels = flex.double(img.ravel())
                S.D.add_noise()
                noise_img = S.D.raw_pixels.as_numpy_array().reshape(img_sh)
```

Replace with:
```python
            if self.epix_mode:
                # ePix10k noise pipeline: CALIB_NOISE_PCT% calibration jitter → apply_epix_noise
                calib = epix_rng.normal(1.0, paths_and_const.CALIB_NOISE_PCT / 100, size=img.shape).clip(0).astype(np.float32)
                noise_img = make_sims.apply_epix_noise(
                    img * calib,
                    t1=self.epix_gain_thresh[0], t2=self.epix_gain_thresh[1],
                    sigma_hg=self.epix_noise_sigma[0],
                    sigma_mg=self.epix_noise_sigma[1],
                    sigma_lg=self.epix_noise_sigma[2],
                    sat_lg=self.epix_sat_lg,
                    rng=epix_rng,
                )
            elif self.jungfrau_mode:
                # Jungfrau noise pipeline: CALIB_NOISE_PCT% calibration jitter → apply_jungfrau_noise
                jungfrau_rng = self._jungfrau_rng
                calib = jungfrau_rng.normal(1.0, paths_and_const.CALIB_NOISE_PCT / 100, size=img.shape).clip(0).astype(np.float32)
                noise_img = make_sims.apply_jungfrau_noise(
                    img * calib,
                    t1=self.jungfrau_gain_thresh[0], t2=self.jungfrau_gain_thresh[1],
                    sigma_g0=self.jungfrau_noise_sigma[0],
                    sigma_g1=self.jungfrau_noise_sigma[1],
                    sigma_g2=self.jungfrau_noise_sigma[2],
                    sat_g2=self.jungfrau_sat_g2,
                    rng=jungfrau_rng,
                )
            elif self.agipd_mode:
                # AGIPD noise pipeline: CALIB_NOISE_PCT% calibration jitter → apply_agipd_noise
                # Output is in ADU (not photon-equivalent); main.py clips to uint16.
                agipd_rng = self._agipd_rng
                calib = agipd_rng.normal(1.0, paths_and_const.CALIB_NOISE_PCT / 100, size=img.shape).clip(0).astype(np.float32)
                noise_img = make_sims.apply_agipd_noise(
                    img * calib,
                    t1=self.agipd_gain_thresh[0], t2=self.agipd_gain_thresh[1],
                    sigma_hg=self.agipd_noise_sigma[0],
                    sigma_mg=self.agipd_noise_sigma[1],
                    sigma_lg=self.agipd_noise_sigma[2],
                    rng=agipd_rng,
                )
            else:
                S.D.raw_pixels = flex.double(img.ravel())
                S.D.add_noise()
                noise_img = S.D.raw_pixels.as_numpy_array().reshape(img_sh)
```

- [ ] **Step 4: Copy to site-packages**

```bash
cp /data/bioxfel/user/gihan/Resonet/resonet/resonet/sims/simulator.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
```

- [ ] **Step 5: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/sims/simulator.py
git commit -m "feat: agipd_mode in Simulator — noise dispatch for AGIPD 3-gain-zone model"
```

---

## Task 5: CLI flags + auto-enable in `main.py`

**Files:**
- Modify: `resonet/resonet/sims/main.py`
  - `args()` function: add `--agipd-gain-thresh` and `--agipd-noise-sigma` arguments
  - `run()` function: add AGIPD validation + `HS.agipd_mode` wiring (after the jungfrau block, ~line 333)

- [ ] **Step 1: Add CLI arguments in `args()`**

After the `--jungfrauSatG2` argument (~line 56), add:

```python
    parser.add_argument("--agipd-gain-thresh", dest="agipdGainThresh", nargs=2, type=float,
                        default=[65, 2000], metavar=("T1", "T2"),
                        help="AGIPD HG→MG and MG→LG thresholds in photon counts. "
                             "Only active with --geom agipd. Default: 65 2000")
    parser.add_argument("--agipd-noise-sigma", dest="agipdNoiseSigma", nargs=3, type=float,
                        default=[7.0, 3.0, 1.5], metavar=("HG", "MG", "LG"),
                        help="AGIPD readout noise RMS in ADU per gain zone (HG, MG, LG). "
                             "Only active with --geom agipd. Default: 7.0 3.0 1.5")
```

- [ ] **Step 2: Add AGIPD wiring in `run()` after jungfrau block (~line 333)**

After the block ending with `HS._jungfrau_rng = np.random.default_rng(seeds[jid])`, add:

```python
    _agipd_argv_flags = any(
        f'--{flag}' in sys.argv
        for flag in ('agipd-gain-thresh', 'agipd-noise-sigma')
    )
    if _agipd_argv_flags and getattr(args, 'geom', None) != 'agipd':
        raise ValueError(
            "--agipd-gain-thresh/--agipd-noise-sigma were specified but --geom is not 'agipd'. "
            "The AGIPD noise model requires --geom agipd. "
            "Either remove the agipd flags or add '--geom agipd'."
        )
    if getattr(args, 'geom', None) == 'agipd':
        _t1, _t2 = args.agipdGainThresh
        if _t1 >= _t2:
            raise ValueError(f"--agipd-gain-thresh T1 ({_t1}) must be < T2 ({_t2}).")
        if any(s < 0 for s in args.agipdNoiseSigma):
            raise ValueError(
                f"--agipd-noise-sigma values must be non-negative; got {args.agipdNoiseSigma}."
            )
        HS.agipd_mode = True
        HS.agipd_gain_thresh = tuple(args.agipdGainThresh)
        HS.agipd_noise_sigma = tuple(args.agipdNoiseSigma)
        HS._agipd_rng = np.random.default_rng(seeds[jid])
```

- [ ] **Step 3: Copy to site-packages**

```bash
cp /data/bioxfel/user/gihan/Resonet/resonet/resonet/sims/main.py \
   /data/bioxfel/user/gihan/Resonet/simforge/envs/simtbx_mpi/lib/python3.9/site-packages/resonet/sims/
```

- [ ] **Step 4: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/sims/main.py
git commit -m "feat: --agipd-gain-thresh/--agipd-noise-sigma CLI flags; auto-enable with --geom agipd"
```

---

## Task 6: Smoke tests — shape and noise

**Files:**
- Create: `resonet/resonet/tests/test_agipd_shape.py`

These tests use `geom_parser.py` + `_group_by_module()` + `CXIWriter` directly
(no simtbx, no GPU). They simulate the exact code path in `main.py` using synthetic
flat pixel arrays rather than running the full `resonet-simulate` command.

- [ ] **Step 1: Write failing tests**

Create `resonet/resonet/tests/test_agipd_shape.py`:

```python
"""Smoke tests for AGIPD 3D frame assembly and noise application.

These tests exercise the exact code path in main.py using synthetic data
rather than full simtbx simulation (no GPU required).
"""
import os
import tempfile
import numpy as np
import pytest
import h5py


AGIPD_GEOM = os.path.join(
    os.path.dirname(__file__), "..", "sims", "geoms", "AGIPD.geom"
)


def _build_fake_flat_img(panel_map):
    """Build a synthetic flat pixel array with sequential values per panel."""
    total = sum(pm['n_fast'] * pm['n_slow'] for pm in panel_map)
    return np.arange(total, dtype=np.float32)


def test_group_by_module_agipd_geom():
    """_group_by_module returns 16 modules with 8 panels each for AGIPD.geom."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)
    assert groups is not None, "_group_by_module returned None for AGIPD.geom"
    assert len(groups) == 16, f"Expected 16 modules, got {len(groups)}"
    assert all(len(v) == 8 for v in groups.values()), "Each module must have 8 ASICs"


def test_3d_frame_shape_from_agipd_geom():
    """3D assembly of a synthetic flat array produces (16, 512, 128) frame."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)
    assert groups is not None

    n_modules = len(groups)
    ss_per_mod = max(pm['max_ss'] for pm in groups[0]) + 1
    fs_per_mod = max(pm['max_fs'] for pm in groups[0]) + 1
    frame_shape = (n_modules, ss_per_mod, fs_per_mod)

    assert frame_shape == (16, 512, 128), f"Expected (16,512,128), got {frame_shape}"


def test_3d_frame_no_overwrite():
    """Each ASIC writes to a unique (module, ss, fs) location — no panel overwrites another."""
    from resonet.sims.geom_parser import parse_geom
    from resonet.sims.main import _group_by_module
    _, panel_map, _ = parse_geom(AGIPD_GEOM)
    groups = _group_by_module(panel_map)

    frame_shape = (16, 512, 128)
    written = np.zeros(frame_shape, dtype=np.int32)

    _panel_pix_offset = {}
    _offset = 0
    for pm in panel_map:
        _panel_pix_offset[pm['name']] = _offset
        _offset += pm['n_fast'] * pm['n_slow']

    flat_img = _build_fake_flat_img(panel_map)

    for mod_idx, panels in sorted(groups.items()):
        for pm in panels:
            pix_off = _panel_pix_offset[pm['name']]
            n_px = pm['n_fast'] * pm['n_slow']
            panel_data = flat_img[pix_off:pix_off + n_px].reshape(pm['n_slow'], pm['n_fast'])
            written[
                mod_idx,
                pm['min_ss']:pm['max_ss'] + 1,
                pm['min_fs']:pm['max_fs'] + 1,
            ] += 1  # count writes; >1 means overwrite

    assert np.all(written == 1), (
        f"{np.sum(written != 1)} pixels were written != 1 times "
        f"(overwrite detected if > 1, missed if == 0)"
    )


def test_cxi_writer_accepts_3d_frame_shape():
    """CXIWriter with frame_shape=(16,512,128) creates (N,16,512,128) dataset."""
    from resonet.sims.cxi_writer import CXIWriter
    frame_shape = (16, 512, 128)
    meta = {
        'detector_name': 'AGIPD 1M',
        'distance_m': 0.15165,
        'pixel_size_m': 0.0002,
        'photon_energy_eV': 9385.0,
        'wavelength_m': 1239.84193e-9 / 9385.0,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.cxi')
        writer = CXIWriter(path, frame_shape, meta)
        img = np.zeros(frame_shape, dtype=np.uint16)
        writer.add_frame(img, labels={'hit': 0.0})
        writer.add_frame(img, labels={'hit': 1.0})
        writer.close()

        with h5py.File(path, 'r') as f:
            data = f['entry_1/data_1/data']
            assert data.shape == (2, 16, 512, 128), (
                f"Expected (2,16,512,128), got {data.shape}"
            )
            assert data.dtype == np.uint16


def test_agipd_noise_on_3d_array():
    """apply_agipd_noise works on (16,512,128) input without shape change."""
    from resonet.sims.make_sims import apply_agipd_noise
    img = np.random.default_rng(0).random((16, 512, 128)).astype(np.float32) * 100
    out = apply_agipd_noise(img, rng=np.random.default_rng(1))
    assert out.shape == (16, 512, 128)
    assert out.dtype == np.float32
    assert np.all(out >= 0)
    assert np.max(out) > 0
```

- [ ] **Step 2: Run tests — expect FAIL on `test_group_by_module_agipd_geom` and `test_3d_frame_no_overwrite` (since `_group_by_module` is not yet importable from `main`)**

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/test_agipd_shape.py -v
```

If Tasks 2–3 are complete, all 5 tests should PASS. If any fail, diagnose before proceeding.

- [ ] **Step 3: Run all tests to check for regressions**

```bash
cd /data/bioxfel/user/gihan/Resonet/resonet
python -m pytest resonet/tests/ -v --ignore=resonet/tests/test_models.py \
    --ignore=resonet/tests/test_net.py \
    --ignore=resonet/tests/test_predict.py
```

Expected: all previously-passing tests still pass; new AGIPD tests pass.

- [ ] **Step 4: Commit**

```bash
cd /data/bioxfel/user/gihan/Resonet
git add resonet/resonet/tests/test_agipd_shape.py
git commit -m "test: AGIPD shape smoke tests — module grouping, 3D assembly, CXIWriter, noise"
```

---

## Task 7: Branch + PR

- [ ] **Step 1: Create feature branch**

All work above should be done on `feature/agipd-synthesis`. If commits were made on `main`, move them:

```bash
git checkout -b feature/agipd-synthesis
# If commits were already made on main, they are now on both branches.
# Push the feature branch; do NOT force-push main.
git push -u origin feature/agipd-synthesis
```

- [ ] **Step 2: Run `/code-review` then `/pr-review-toolkit:review-pr`**

Per CLAUDE.md: both reviews must pass before opening the PR.

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "feat: AGIPD 1M synthesis — 3D (16,512,128) CXI frames + 3-gain noise" \
  --body "$(cat <<'EOF'
## Summary
- `apply_agipd_noise()` in `make_sims.py`: 3-gain-zone noise model (HG/MG/LG) with ADU output
- `_group_by_module()` in `main.py`: pXaY panel-name detection → 3D frame shape
- CXI path produces `(N, 16, 512, 128)` frames matching real AGIPD data layout
- `agipd_mode` in `Simulator`, auto-enabled by `--geom agipd`
- `CXIWriter` required no changes (already handles arbitrary frame_shape)

## Test plan
- [ ] `test_apply_agipd_noise_*` (11 unit tests in `test_simulator_funcs.py`)
- [ ] `test_agipd_shape.py` (5 integration smoke tests, no GPU)
- [ ] Full test suite passes with no regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Manual Verification (after PR merges)

```bash
source /data/bioxfel/user/gihan/Resonet/setup_resonet.sh
mkdir -p /tmp/agipd_test

# 1-shot test — no GPU needed to verify shape
srun --export=ALL resonet-simulate /tmp/agipd_test \
    --nshot 1 --geom agipd --ngpu 1 --randHits

# Check shape
python -c "
import h5py
with h5py.File('/tmp/agipd_test/compressed0.cxi', 'r') as f:
    print('shape:', f['entry_1/data_1/data'].shape)
    print('dtype:', f['entry_1/data_1/data'].dtype)
"
# Expected: shape: (1, 16, 512, 128)  dtype: uint16

# Visual check
python /data/bioxfel/user/gihan/Resonet/resonet/resonet/scripts/view_cxi.py \
    /tmp/agipd_test/compressed0.cxi \
    --geom /data/bioxfel/user/gihan/Resonet/geoms/AGIPD.geom
```
